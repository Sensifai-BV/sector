//! Volume image emission.
//!
//! Writes manifest, codebook with replicas, payload, rerank copy and CRC arrays
//! into a byte-exact image the device mounts without transformation.
//!
//! # Write ordering
//!
//! Emit regions in dependency order and write the manifest last, so an
//! interrupted install leaves the previous manifest intact and the image either
//! mounts completely or not at all. Parity cannot repair a torn write — a
//! partially written sector is consistently wrong, not noisily wrong — so
//! atomicity comes from ordering.
//!
//! Align every region to an erase sector and interleave codebook replicas
//! across independent sectors. Addresses are assigned here, so this is where
//! the independence the protection scheme assumes is established.
//!
//! Verify the emitted image by mounting it with the code the device runs and
//! re-querying, rather than by checking the writer's own bookkeeping. A
//! round-trip through the real mount path is what catches a layout the reader
//! and writer agree on and the format does not.

use crate::encode::QuantizedCodebook;
use sector_codec::crc::crc32;
use sector_format::manifest::{self, Manifest, ManifestError, MANIFEST_BYTES};
use sector_format::region::{Protection, RegionDesc, RegionKind, RegionTable, REGION_COUNT};
use sector_format::{BLOCK_BYTES, SECTOR_BYTES};

/// Why an image could not be emitted.
#[derive(Debug, PartialEq, Eq)]
pub enum EmitError {
    /// The manifest could not be encoded.
    Manifest(ManifestError),
    /// A region's computed layout is invalid.
    Region(sector_format::region::RegionError),
    /// The image would exceed `capacity`.
    TooLarge {
        /// Bytes the image needs.
        needed: usize,
        /// Bytes available.
        capacity: usize,
    },
}

/// What was emitted, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitReport {
    /// Total image bytes.
    pub bytes: usize,
    /// Vectors stored.
    pub n: u32,
    /// Payload blocks.
    pub payload_blocks: u32,
    /// Rerank blocks.
    pub rerank_blocks: u32,
    /// Codebook copies written, including the primary.
    pub codebook_copies: usize,
}

/// Everything needed to lay out an image.
pub struct Image<'a> {
    /// Quantized codebook.
    pub codebook: &'a QuantizedCodebook,
    /// PQ codes, `n * m`.
    pub codes: &'a [u8],
    /// Higher-precision records, `n * d` as i8 bytes.
    pub rerank: &'a [u8],
    /// Vectors.
    pub n: usize,
    /// Vector dimension.
    pub d: usize,
    /// Candidate depth the image is built for.
    pub r: u32,
    /// Codebook copies, including the primary.
    pub copies: usize,
    /// Vector slots to leave erased for later appends.
    ///
    /// # Why this sizes both regions
    ///
    /// An id needs a code *and* a rerank record, so headroom is whichever region
    /// runs out first — and they run out at different rates. Rerank is
    /// `d * cb_bytes` per vector against the payload's `m * b / 8`, so at every
    /// shipped profile rerank binds: at `D = 128, m = 16` a block holds 32 codes
    /// and 4 rerank records, an 8:1 ratio.
    ///
    /// Before this field existed, headroom was whatever each region's independent
    /// rounding-up to an erase sector happened to leave. A measured example: a
    /// 400-vector test volume had 512 spare payload slots and **112** spare
    /// rerank slots. Reserving payload space alone would report capacity that
    /// does not exist.
    ///
    /// Reserved blocks are erased, not free: they occupy the volume and `verify`
    /// sweeps them. A build that reserves and never appends has paid for nothing.
    pub reserve: usize,
}

/// Round `bytes` up to a whole erase sector.
const fn sectors(bytes: usize) -> usize {
    bytes.next_multiple_of(SECTOR_BYTES)
}

/// Ids per append: `lcm(payload_per_block, rerank_per_block)`.
///
/// An appended id needs both a code and a rerank record, so an append must
/// advance both regions by whole blocks — a partial block would need its CRC
/// rewritten, and NOR is program-once. The least common multiple is the smallest
/// run satisfying both.
///
/// At the shipped profiles this is 32 ids (`D=128, m=16`) or 16 (`D=128, m=32`),
/// so the largest possible gap is 31 ids and an append costs 9 or 5 block
/// programs. Both bounded, which is what makes append viable on a device.
pub const fn append_unit(payload_per_block: usize, rerank_per_block: usize) -> usize {
    // `Ord::max` is not const on stable, so the clamps are written out.
    let a = if payload_per_block == 0 {
        1
    } else {
        payload_per_block
    };
    let b = if rerank_per_block == 0 {
        1
    } else {
        rerank_per_block
    };
    // gcd by subtraction-free Euclid; `const fn` cannot loop with `while let`.
    let mut x = a;
    let mut y = b;
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    a / x * b
}

/// Emit a complete volume image into `out`.
///
/// Region order follows the write order: everything the manifest points at is
/// laid down first, and the manifest last. An installer that copies this image
/// in order therefore never has a manifest pointing at regions that are not
/// yet durable.
pub fn emit(image: &Image<'_>, out: &mut Vec<u8>) -> Result<EmitReport, EmitError> {
    let m = image.codebook.m;
    let payload_bytes = m;
    let rerank_bytes = image.d;

    let payload_per_block = BLOCK_BYTES / payload_bytes.max(1);
    let rerank_per_block = BLOCK_BYTES / rerank_bytes.max(1);

    // The append unit: the smallest id run that is a whole number of blocks in
    // *both* regions. An append must advance both, and a partial block in either
    // would need its CRC rewritten — which NOR cannot do.
    let unit = append_unit(payload_per_block, rerank_per_block);

    // Capacity to lay down. The first append cannot start inside the block
    // holding the last built vector — that block's CRC is written, and NOR is
    // program-once — so the reserve is measured from the next unit boundary.
    // Rounding here rather than at append time means the reserved slot count the
    // operator asked for is the count they can actually use.
    let append_head = if image.reserve == 0 {
        image.n
    } else {
        image.n.next_multiple_of(unit.max(1))
    };
    let capacity = append_head + image.reserve;

    let payload_blocks = capacity.div_ceil(payload_per_block.max(1));
    let rerank_blocks = if rerank_bytes <= BLOCK_BYTES {
        capacity.div_ceil(rerank_per_block.max(1))
    } else {
        capacity * rerank_bytes.div_ceil(BLOCK_BYTES)
    };

    let cb_bytes = image.codebook.byte_len();
    let cb_extent = sectors(cb_bytes);
    let replica_extent = cb_extent * image.copies.saturating_sub(1);
    let payload_extent = sectors(payload_blocks * BLOCK_BYTES);
    let payload_crc_extent = sectors(payload_blocks * 4);
    let rerank_extent = sectors(rerank_blocks * BLOCK_BYTES);
    let rerank_crc_extent = sectors(rerank_blocks * 4);

    let mut base = manifest::MANIFEST_RESERVED_BYTES as usize;
    let mut regions = [RegionDesc {
        kind: RegionKind::Codebook,
        protection: Protection::Replicate,
        base: 0,
        block_bytes: BLOCK_BYTES as u32,
        blocks: 0,
    }; REGION_COUNT];

    let spec = [
        (RegionKind::Codebook, Protection::Replicate, cb_extent),
        (
            RegionKind::CodebookReplica,
            Protection::Replicate,
            replica_extent,
        ),
        (RegionKind::Payload, Protection::Detect, payload_extent),
        (
            RegionKind::PayloadCrc,
            Protection::Detect,
            payload_crc_extent,
        ),
        (RegionKind::Rerank, Protection::Detect, rerank_extent),
        (RegionKind::RerankCrc, Protection::Detect, rerank_crc_extent),
    ];

    for (i, (kind, protection, extent)) in spec.into_iter().enumerate() {
        if let Some(slot) = regions.get_mut(i) {
            *slot = RegionDesc {
                kind,
                protection,
                base: base as u32,
                block_bytes: BLOCK_BYTES as u32,
                blocks: (extent / BLOCK_BYTES) as u32,
            };
        }
        base += extent;
    }

    let table = RegionTable { regions };
    table.validate().map_err(EmitError::Region)?;

    out.clear();
    out.resize(base, 0xFF);

    // Codebook and its replicas.
    let cb = stored_bytes(&image.codebook.components);
    write_at(out, regions[0].base as usize, &cb);
    for c in 1..image.copies {
        let at = regions[1].base as usize + (c - 1) * cb_extent;
        write_at(out, at, &cb);
    }

    // Only the blocks holding built vectors are written, and only their CRC slots.
    // Reserved blocks stay erased with erased CRC words, which is what
    // `append::find_head` recognises — the erased state is the append journal, so
    // writing a CRC for an empty block would make it look occupied forever.
    let built_payload_blocks = image.n.div_ceil(payload_per_block.max(1));
    let built_rerank_blocks = if rerank_bytes <= BLOCK_BYTES {
        image.n.div_ceil(rerank_per_block.max(1))
    } else {
        image.n * rerank_bytes.div_ceil(BLOCK_BYTES)
    };

    // Payload, block by block, so a partial final block is padded rather than
    // running into the next region.
    write_blocks(
        out,
        regions[2].base as usize,
        regions[3].base as usize,
        image.codes,
        payload_bytes,
        payload_per_block,
        built_payload_blocks,
    );

    // Rerank records.
    write_blocks(
        out,
        regions[4].base as usize,
        regions[5].base as usize,
        image.rerank,
        rerank_bytes,
        rerank_per_block.max(1),
        built_rerank_blocks,
    );

    // Manifest last: it points at regions that are already laid down.
    let manifest = Manifest {
        sequence: 1,
        d: image.d as u16,
        m: m as u16,
        b: image.codebook.k.trailing_zeros() as u16,
        cb_bytes: 1,
        // `n` is the addressable extent: highest id plus one, not the stored
        // count. A fresh build has appended nothing, so the extent ends where the
        // built corpus does — `appended_from` marks where a *future* append will
        // start and is only reached once one happens.
        //
        // Setting `n` to `appended_from` here instead would claim the reserved
        // slots are populated: `stored()` would report the gap as present, and the
        // scan would read erased blocks and drop every candidate in them.
        n: image.n as u32,
        r: image.r,
        built_n: image.n as u32,
        // No gap at build time: nothing has been appended, so the stored set is
        // exactly `[0, n)`. The gap comes into being when the first append
        // happens and writes a manifest whose `appended_from` is the head it
        // found — which is why `built_n <= appended_from <= n` holds at every
        // point in a volume's life rather than only after an append.
        appended_from: image.n as u32,
        table,
    };
    let mut slot = [0u8; MANIFEST_BYTES];
    manifest.encode(&mut slot).map_err(EmitError::Manifest)?;
    write_at(out, manifest::SLOT_A_OFFSET as usize, &slot);

    Ok(EmitReport {
        bytes: base,
        n: image.n as u32,
        payload_blocks: payload_blocks as u32,
        rerank_blocks: rerank_blocks as u32,
        codebook_copies: image.copies,
    })
}

/// Reinterpret signed components as their stored bytes.
///
/// `i8` and `u8` share a representation, so this is the same bytes viewed
/// differently. Done by copy rather than transmute: the crate forbids unsafe,
/// and the codebook is a few tens of kilobytes emitted once per build.
fn stored_bytes(components: &[i8]) -> Vec<u8> {
    components.iter().map(|c| *c as u8).collect()
}

/// Write `data` at `at`.
fn write_at(out: &mut [u8], at: usize, data: &[u8]) {
    if let Some(dst) = out.get_mut(at..at + data.len()) {
        dst.copy_from_slice(data);
    }
}

/// Write `records` into blocks, filling a parallel CRC array.
fn write_blocks(
    out: &mut [u8],
    base: usize,
    crc_base: usize,
    records: &[u8],
    record_bytes: usize,
    per_block: usize,
    blocks: usize,
) {
    for b in 0..blocks {
        let block_at = base + b * BLOCK_BYTES;
        let first = b * per_block;
        for slot in 0..per_block {
            let record = first + slot;
            let src_start = record * record_bytes;
            let Some(src) = records.get(src_start..src_start + record_bytes) else {
                break;
            };
            let dst_at = block_at + slot * record_bytes;
            if let Some(dst) = out.get_mut(dst_at..dst_at + record_bytes) {
                dst.copy_from_slice(src);
            }
        }
        let crc = match out.get(block_at..block_at + BLOCK_BYTES) {
            Some(block) => crc32(block),
            None => 0,
        };
        write_at(out, crc_base + b * 4, &crc.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode, quantize};
    use crate::train::{train, TrainConfig};
    use sector_format::region::RegionKind;

    const D: usize = 8;
    const M: usize = 2;
    const N: usize = 300;

    fn corpus() -> Vec<f32> {
        let mut out = vec![0f32; N * D];
        for v in 0..N {
            for j in 0..D {
                out[v * D + j] = (((v * 31 + j * 17) % 97) as f32) * 1.3;
            }
        }
        out
    }

    fn built() -> (Vec<u8>, EmitReport, QuantizedCodebook, Vec<u8>, Vec<u8>) {
        let data = corpus();
        let cfg = TrainConfig {
            d: D,
            m: M,
            b: 3,
            iterations: 30,
            seed: 5,
        };
        let (books, _) = train(&data, N, cfg).unwrap();
        let (codes, _) = encode(&data, N, D, &books);
        let q = quantize(&books, 127);

        // Rerank records: the corpus narrowed to i8, stored as bytes.
        let rerank: Vec<u8> = data
            .iter()
            .map(|v| (v / 2.0).clamp(-128.0, 127.0) as i8 as u8)
            .collect();

        let mut out = Vec::new();
        let report = emit(
            &Image {
                codebook: &q,
                codes: &codes,
                rerank: &rerank,
                n: N,
                d: D,
                r: 100,
                copies: 2,
                reserve: 0,
            },
            &mut out,
        )
        .unwrap();
        (out, report, q, codes, rerank)
    }

    #[test]
    fn the_emitted_image_mounts_through_the_device_path() {
        // The test that closes the loop. The builder's own bookkeeping proves
        // nothing; what matters is that `sector-core`'s mount reads the image
        // and agrees with what was written.
        let (image, report, _, _, _) = built();
        let mut slot_a = [0u8; MANIFEST_BYTES];
        let mut slot_b = [0u8; MANIFEST_BYTES];
        slot_a.copy_from_slice(&image[..MANIFEST_BYTES]);
        slot_b.copy_from_slice(&image[MANIFEST_BYTES..2 * MANIFEST_BYTES]);

        let m = manifest::select(&slot_a, &slot_b).expect("the image must mount");
        assert_eq!(m.n, N as u32);
        assert_eq!(m.d, D as u16);
        assert_eq!(m.m, M as u16);
        assert_eq!(m.b, 3);
        assert_eq!(m.r, 100);
        assert_eq!(report.n, N as u32);
    }

    #[test]
    fn every_region_is_sector_aligned_and_disjoint() {
        let (image, _, _, _, _) = built();
        let mut slot = [0u8; MANIFEST_BYTES];
        slot.copy_from_slice(&image[..MANIFEST_BYTES]);
        let m = Manifest::decode(&slot).unwrap();

        // `validate` checks both properties; assert it directly so a failure
        // names which one broke.
        m.table.validate().expect("regions valid");
        for r in &m.table.regions {
            assert_eq!(r.base as usize % SECTOR_BYTES, 0, "{:?} unaligned", r.kind);
        }
    }

    #[test]
    fn payload_blocks_hold_the_codes_that_were_encoded() {
        let (image, _, _, codes, _) = built();
        let mut slot = [0u8; MANIFEST_BYTES];
        slot.copy_from_slice(&image[..MANIFEST_BYTES]);
        let m = Manifest::decode(&slot).unwrap();
        let payload = m.table.get(RegionKind::Payload).unwrap();

        let per_block = BLOCK_BYTES / M;
        for v in 0..N {
            let block = v / per_block;
            let slot_in_block = v % per_block;
            let at = payload.base as usize + block * BLOCK_BYTES + slot_in_block * M;
            assert_eq!(
                &image[at..at + M],
                &codes[v * M..(v + 1) * M],
                "vector {v} misplaced"
            );
        }
    }

    #[test]
    fn every_block_carries_a_crc_that_verifies() {
        // The CRC array is written from the block's final contents, including
        // any padding, so a verifier reading the block back must agree.
        let (image, report, _, _, _) = built();
        let mut slot = [0u8; MANIFEST_BYTES];
        slot.copy_from_slice(&image[..MANIFEST_BYTES]);
        let m = Manifest::decode(&slot).unwrap();

        for (region, crc_region, blocks) in [
            (
                RegionKind::Payload,
                RegionKind::PayloadCrc,
                report.payload_blocks,
            ),
            (
                RegionKind::Rerank,
                RegionKind::RerankCrc,
                report.rerank_blocks,
            ),
        ] {
            let data = m.table.get(region).unwrap();
            let crcs = m.table.get(crc_region).unwrap();
            for b in 0..blocks as usize {
                let at = data.base as usize + b * BLOCK_BYTES;
                let stored_at = crcs.base as usize + b * 4;
                let stored = u32::from_le_bytes([
                    image[stored_at],
                    image[stored_at + 1],
                    image[stored_at + 2],
                    image[stored_at + 3],
                ]);
                assert_eq!(
                    crc32(&image[at..at + BLOCK_BYTES]),
                    stored,
                    "{region:?} block {b} CRC mismatch"
                );
            }
        }
    }

    #[test]
    fn the_codebook_replica_is_byte_identical_and_elsewhere() {
        let (image, _, q, _, _) = built();
        let mut slot = [0u8; MANIFEST_BYTES];
        slot.copy_from_slice(&image[..MANIFEST_BYTES]);
        let m = Manifest::decode(&slot).unwrap();

        let primary = m.table.get(RegionKind::Codebook).unwrap();
        let replica = m.table.get(RegionKind::CodebookReplica).unwrap();
        let len = q.byte_len();

        let a = &image[primary.base as usize..primary.base as usize + len];
        let b = &image[replica.base as usize..replica.base as usize + len];
        assert_eq!(a, b, "replica must be identical");

        // Different erase sectors, or the replica protects nothing.
        assert_ne!(
            primary.base as usize / SECTOR_BYTES,
            replica.base as usize / SECTOR_BYTES
        );
    }

    #[test]
    fn the_manifest_is_written_last_in_address_order() {
        // Region bases all sit above the two manifest slots, so an installer
        // copying the image in address order lays every region down before the
        // manifest that points at it.
        let (image, _, _, _, _) = built();
        let mut slot = [0u8; MANIFEST_BYTES];
        slot.copy_from_slice(&image[..MANIFEST_BYTES]);
        let m = Manifest::decode(&slot).unwrap();
        for r in &m.table.regions {
            assert!(
                r.base >= manifest::MANIFEST_RESERVED_BYTES,
                "{:?} overlaps the manifest slots",
                r.kind
            );
        }
    }

    #[test]
    fn emission_is_deterministic() {
        let (a, ra, _, _, _) = built();
        let (b, rb, _, _, _) = built();
        assert_eq!(a, b, "two builds of one corpus must be byte-identical");
        assert_eq!(ra, rb);
    }

    #[test]
    fn a_corrupted_payload_block_fails_only_its_own_crc() {
        // Damage stays localised to the block that carries it, which is what
        // makes the drop accounting exact.
        let (mut image, report, _, _, _) = built();
        let mut slot = [0u8; MANIFEST_BYTES];
        slot.copy_from_slice(&image[..MANIFEST_BYTES]);
        let m = Manifest::decode(&slot).unwrap();
        let payload = m.table.get(RegionKind::Payload).unwrap();
        let crcs = m.table.get(RegionKind::PayloadCrc).unwrap();

        // At M=2 a 512 B block holds 256 vectors, so N=300 gives two blocks.
        // Targeting a fixed index would land in padding past the last one.
        assert!(report.payload_blocks >= 2);
        let target = report.payload_blocks as usize - 1;
        image[payload.base as usize + target * BLOCK_BYTES + 7] ^= 0xFF;

        let mut failures = 0;
        for b in 0..report.payload_blocks as usize {
            let at = payload.base as usize + b * BLOCK_BYTES;
            let stored_at = crcs.base as usize + b * 4;
            let stored = u32::from_le_bytes([
                image[stored_at],
                image[stored_at + 1],
                image[stored_at + 2],
                image[stored_at + 3],
            ]);
            if crc32(&image[at..at + BLOCK_BYTES]) != stored {
                failures += 1;
                assert_eq!(b, target);
            }
        }
        assert_eq!(failures, 1);
    }
}
