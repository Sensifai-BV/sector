//! Appending vectors to a built volume.
//!
//! Wires [`sector_core::append`] to a host file. That path has existed and been
//! tested since before this module; what was missing was anything that called it,
//! because the read backends refuse writes.
//!
//! # Append is insert-only, and bounded
//!
//! Whole blocks only, in both regions at once. An appended id needs a code *and*
//! a rerank record, so an append advances by
//! `lcm(payload_per_block, rerank_per_block)` ids — 32 at `D=128, m=16`, 16 at
//! `D=128, m=32`. A partial block would need its CRC rewritten, and NOR is
//! program-once.
//!
//! Order within an append is payload, then rerank, then the CRC of each, then the
//! manifest. A power loss at any point leaves either an erased CRC slot (the block
//! reads as absent) or a manifest still naming the old extent (the blocks are
//! invisible). Neither state returns wrong bytes.
//!
//! # There is no delete and no update
//!
//! An id, once written, is permanent. A validity bitmap would add a per-candidate
//! lookup to the scan loop, which is the loop the project's cost argument is
//! about, and insert-only growth does not need it. `sector build` is how a vector
//! is removed.
//!
//! # Appended vectors carry drift
//!
//! They are encoded against a codebook trained without them. [`AppendReport`]
//! carries the [`Drift`] the engine accumulates, and any recall figure measured on
//! an appended volume must state it: retraining is corpus-global and has no
//! bounded-RAM formulation, so the error is reported rather than corrected.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::Path;

use sector_core::append::{append_block, find_head, Drift};
use sector_format::manifest::{self, Manifest, MANIFEST_BYTES};
use sector_format::region::RegionKind;
use sector_format::BLOCK_BYTES;
use sector_hal::{NorFlash, ERASED_BYTE};

use crate::volume::{Geometry, HostVolume};
use crate::{Error, FileFlash};

/// Why an append was refused.
#[derive(Debug)]
pub enum AppendError {
    /// The volume could not be read or mounted.
    Volume(String),
    /// The file could not be opened for writing.
    Io(std::io::Error),
    /// No erased block remains in one of the regions.
    ///
    /// Names both so the operator can see which bound first — rerank normally
    /// does, at 8x the payload's bytes per vector.
    Full {
        /// Appendable payload slots remaining.
        payload_slots: usize,
        /// Appendable rerank slots remaining.
        rerank_slots: usize,
    },
    /// The batch is not a whole number of append units.
    NotAligned {
        /// Vectors offered.
        count: usize,
        /// Ids per append at this geometry.
        unit: usize,
    },
    /// A vector's dimension does not match the volume.
    Dimension {
        /// Dimension offered.
        found: usize,
        /// Dimension the volume was built for.
        expected: usize,
    },
    /// A component was not finite.
    ///
    /// Refused rather than quantized: `NaN as i8` is 0 in Rust, which would drop
    /// the component from every later score without any signal.
    NonFinite {
        /// Index of the offending component.
        at: usize,
    },
    /// Another process is appending to this volume.
    ///
    /// Refused rather than waited on: an append is short, and a caller blocked on
    /// a lock it did not ask for cannot tell a slow append from a crashed one that
    /// left the lock held.
    Locked,
    /// The engine refused the write.
    Engine(String),
}

impl std::fmt::Display for AppendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Volume(e) | Self::Engine(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Full {
                payload_slots,
                rerank_slots,
            } => write!(
                f,
                "no room to append: {payload_slots} payload and {rerank_slots} rerank \
                 slots remain; rebuild with a larger --reserve"
            ),
            Self::NotAligned { count, unit } => write!(
                f,
                "{count} vectors is not a multiple of the {unit}-vector append unit \
                 at this geometry"
            ),
            Self::Dimension { found, expected } => {
                write!(
                    f,
                    "vector has {found} components, volume expects {expected}"
                )
            }
            Self::NonFinite { at } => write!(f, "component {at} is not finite"),
            Self::Locked => write!(
                f,
                "another process is appending to this volume; retry when it finishes"
            ),
        }
    }
}

impl std::error::Error for AppendError {}

impl From<std::io::Error> for AppendError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// What an append did.
#[derive(Clone, Copy, Debug)]
pub struct AppendReport {
    /// First id written.
    pub first_id: u32,
    /// Vectors written.
    pub count: u32,
    /// Ids absent from storage after this append.
    pub gap: (u32, u32),
    /// New addressable extent.
    pub n: u32,
    /// Payload blocks programmed.
    pub payload_blocks: u32,
    /// Rerank blocks programmed.
    pub rerank_blocks: u32,
    /// Accumulated drift against the build codebook.
    pub drift: Drift,
    /// Appendable slots left, payload and rerank.
    pub remaining: (usize, usize),
}

/// Ids per append at this geometry.
///
/// `lcm(payload_per_block, rerank_per_block)`. See the module documentation.
pub fn append_unit(g: &Geometry) -> usize {
    let per_p = (BLOCK_BYTES / g.payload_bytes.max(1)).max(1);
    let per_r = (BLOCK_BYTES / g.rerank_bytes.max(1)).max(1);
    let mut x = per_p;
    let mut y = per_r;
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    per_p / x * per_r
}

/// How many more vectors a volume can take.
///
/// Counts fully-erased blocks in each region and returns the slots they hold.
/// Rerank normally binds first.
pub fn capacity(path: &Path) -> Result<(usize, usize), AppendError> {
    let mut flash = FileFlash::open(path).map_err(|e| AppendError::Volume(e.to_string()))?;
    let volume =
        HostVolume::mount(&mut flash, None).map_err(|e| AppendError::Volume(e.to_string()))?;
    let g = volume.geometry;
    let payload = erased_slots(
        &mut flash,
        &volume,
        RegionKind::Payload,
        g.payload_bytes,
        g.payload.blocks(),
    )?;
    let rerank = erased_slots(
        &mut flash,
        &volume,
        RegionKind::Rerank,
        g.rerank_bytes,
        g.rerank.blocks(),
    )?;
    Ok((payload, rerank))
}

/// Slots in fully-erased blocks of `kind`, past the built extent.
fn erased_slots(
    flash: &mut FileFlash,
    volume: &HostVolume,
    kind: RegionKind,
    record_bytes: usize,
    built_blocks: usize,
) -> Result<usize, AppendError> {
    let Some(desc) = volume.manifest.table.get(kind) else {
        return Ok(0);
    };
    let per_block = (BLOCK_BYTES / record_bytes.max(1)).max(1);
    let mut buf = [0u8; BLOCK_BYTES];
    let mut slots = 0usize;
    // From the first block past the built data: the built tail block is sealed
    // even when partially filled, so it is not appendable.
    for b in built_blocks..desc.blocks as usize {
        let addr = desc.base + (b * BLOCK_BYTES) as u32;
        flash
            .read(addr, &mut buf)
            .map_err(|e| AppendError::Volume(e.to_string()))?;
        if buf.iter().all(|x| *x == ERASED_BYTE) {
            slots += per_block;
        }
    }
    Ok(slots)
}

/// Append `vectors` to the volume at `path`.
///
/// Each entry is `d` f32 components. The count must be a whole number of append
/// units; [`append_unit`] reports the size and the CLI pads a short batch rather
/// than making the caller compute it.
pub fn append(path: &Path, vectors: &[Vec<f32>]) -> Result<AppendReport, AppendError> {
    let (geometry, manifest_before) = {
        let mut flash = FileFlash::open(path).map_err(|e| AppendError::Volume(e.to_string()))?;
        let v =
            HostVolume::mount(&mut flash, None).map_err(|e| AppendError::Volume(e.to_string()))?;
        (v.geometry, v.manifest)
    };

    let unit = append_unit(&geometry);
    if vectors.is_empty() || !vectors.len().is_multiple_of(unit) {
        return Err(AppendError::NotAligned {
            count: vectors.len(),
            unit,
        });
    }
    for v in vectors {
        if v.len() != geometry.d {
            return Err(AppendError::Dimension {
                found: v.len(),
                expected: geometry.d,
            });
        }
        if let Some(at) = v.iter().position(|x| !x.is_finite()) {
            return Err(AppendError::NonFinite { at });
        }
    }

    // The lock is taken **before** anything is read that the write depends on.
    // Capacity and the append head are read-then-act decisions: acquiring the lock
    // after checking them would let two appends both observe room and both target
    // the same head, which is the race the lock exists to prevent.
    let mut writer = WritableVolume::open(path)?;

    let (payload_slots, rerank_slots) = capacity(path)?;
    if payload_slots < vectors.len() || rerank_slots < vectors.len() {
        return Err(AppendError::Full {
            payload_slots,
            rerank_slots,
        });
    }

    // Encode against the existing codebook. This is where the drift comes from,
    // and it is why retraining would invalidate every stored code: the codebook is
    // shared by the whole corpus.
    let codebook = read_codebook(path, &geometry)?;
    let mut codes = Vec::with_capacity(vectors.len() * geometry.m);
    let mut rerank = Vec::with_capacity(vectors.len() * geometry.rerank_bytes);
    let mut error_sum = 0u64;
    for v in vectors {
        let (c, r, err) = encode_one(v, &geometry, &codebook);
        codes.extend_from_slice(&c);
        rerank.extend_from_slice(&r);
        error_sum += err;
    }

    let payload_desc = geometry_region(&manifest_before, RegionKind::Payload)?;
    let payload_crc = geometry_region(&manifest_before, RegionKind::PayloadCrc)?;
    let rerank_desc = geometry_region(&manifest_before, RegionKind::Rerank)?;
    let rerank_crc = geometry_region(&manifest_before, RegionKind::RerankCrc)?;

    let payload_per_block = (BLOCK_BYTES / geometry.payload_bytes.max(1)).max(1);
    let rerank_per_block = (BLOCK_BYTES / geometry.rerank_bytes.max(1)).max(1);

    // The **id** is what both regions must agree on, not the block number.
    //
    // `find_head` returns the first erased block of a region, and the two regions
    // do not reach the same id at the same block: each is independently rounded up
    // to a whole erase sector, so the built data in one may end mid-sector while
    // the other ends exactly. Taking each region's own head would write a
    // vector's code at the id implied by the payload region and its rerank record
    // at a *different* id — the record would then be read for the wrong vector,
    // and stage two would rescore against another vector's bytes.
    //
    // So the payload head fixes the id, and the rerank block follows from it.
    let mut probe = [0u8; BLOCK_BYTES];
    let head_payload = find_head(
        &mut writer,
        payload_desc.0,
        payload_desc.1,
        BLOCK_BYTES,
        &mut probe,
    )
    .map_err(|e| AppendError::Engine(format!("{e:?}")))?
    .ok_or(AppendError::Full {
        payload_slots,
        rerank_slots,
    })?;
    let first_id = (head_payload as usize * payload_per_block) as u32;
    let head_rerank = (first_id as usize / rerank_per_block) as u32;

    // The derived rerank block must be erased, or the id ranges have diverged and
    // an append would overwrite live records. Checked rather than assumed: this is
    // the invariant the bug above violated.
    if head_rerank >= rerank_desc.1 {
        return Err(AppendError::Full {
            payload_slots,
            rerank_slots,
        });
    }
    let mut check = [0u8; BLOCK_BYTES];
    writer
        .read(rerank_desc.0 + head_rerank * BLOCK_BYTES as u32, &mut check)
        .map_err(|e| AppendError::Engine(format!("{e:?}")))?;
    if !check.iter().all(|b| *b == ERASED_BYTE) {
        return Err(AppendError::Engine(format!(
            "rerank block {head_rerank} for id {first_id} is not erased: the payload              and rerank extents have diverged"
        )));
    }

    // Payload blocks, then rerank blocks. Each `append_block` writes the data and
    // then its CRC, so a torn write leaves an erased CRC slot rather than a block
    // that passes with wrong contents.
    let payload_blocks = vectors.len() / payload_per_block;
    for i in 0..payload_blocks {
        let from = i * BLOCK_BYTES;
        append_block(
            &mut writer,
            payload_desc.0,
            payload_crc.0,
            head_payload + i as u32,
            BLOCK_BYTES,
            &codes[from..from + BLOCK_BYTES],
        )
        .map_err(|e| AppendError::Engine(format!("{e:?}")))?;
    }
    let rerank_blocks = vectors.len() / rerank_per_block;
    for i in 0..rerank_blocks {
        let from = i * BLOCK_BYTES;
        append_block(
            &mut writer,
            rerank_desc.0,
            rerank_crc.0,
            head_rerank + i as u32,
            BLOCK_BYTES,
            &rerank[from..from + BLOCK_BYTES],
        )
        .map_err(|e| AppendError::Engine(format!("{e:?}")))?;
    }

    // Manifest last, into the slot that is not live. Until this lands the appended
    // blocks are invisible: `n` still names the old extent, so nothing scans them.
    // `appended` and `built` are recoverable from the manifest, so they are true
    // totals. `error_sum` is **this append's** error only: the manifest has no field
    // for it, so a cumulative figure cannot survive a restart and claiming one would
    // be a number that silently resets. `mean_error_x1024` is therefore the mean over
    // this batch, which is the honest reading of it.
    let drift = Drift {
        appended: manifest_before.appended() + vectors.len() as u32,
        error_sum,
        built: manifest_before.built_n,
    };
    let updated = Manifest {
        sequence: manifest_before.sequence + 1,
        n: first_id + vectors.len() as u32,
        // The gap is created here, not at build: the built tail block was sealed,
        // so ids between it and `first_id` are absent.
        appended_from: if manifest_before.appended() == 0 {
            first_id
        } else {
            manifest_before.appended_from
        },
        ..manifest_before
    };
    let mut slot = [0u8; MANIFEST_BYTES];
    updated
        .encode(&mut slot)
        .map_err(|e| AppendError::Engine(format!("{e:?}")))?;
    // Erase the spare slot, then program it. The live slot is untouched until this
    // succeeds, so a power loss here leaves the previous manifest selected and the
    // appended blocks simply invisible.
    let target = manifest::next_slot_offset(manifest_before.sequence);
    writer
        .erase(target)
        .map_err(|e| AppendError::Engine(format!("{e:?}")))?;
    writer
        .program(target, &slot)
        .map_err(|e| AppendError::Engine(format!("{e:?}")))?;
    writer.sync()?;

    let (rem_p, rem_r) = capacity(path)?;
    Ok(AppendReport {
        first_id,
        count: vectors.len() as u32,
        gap: updated.gap(),
        n: updated.n,
        payload_blocks: payload_blocks as u32,
        rerank_blocks: rerank_blocks as u32,
        drift,
        remaining: (rem_p, rem_r),
    })
}

/// A region's base and block count.
fn geometry_region(m: &Manifest, kind: RegionKind) -> Result<(u32, u32), AppendError> {
    m.table
        .get(kind)
        .map(|d| (d.base, d.blocks))
        .ok_or_else(|| AppendError::Volume(format!("volume has no {kind:?} region")))
}

/// A file opened for the append path only.
///
/// Separate from [`FileFlash`], which refuses writes on purpose: a served volume
/// must not be mutable through the same handle that answers queries. This type
/// exists solely so `sector append` can exist, and it emulates NOR faithfully —
/// `program` refuses to clear a bit that is already 0, which is the constraint
/// real flash imposes and the one an append must respect.
struct WritableVolume {
    file: std::fs::File,
    len: u64,
}

impl WritableVolume {
    /// Open for writing, holding an exclusive lock for the append's duration.
    ///
    /// # Why the lock is not optional
    ///
    /// An append is read-head-then-install, and the install is not idempotent.
    /// Two concurrent appends both read sequence `S`, both compute the same spare
    /// manifest slot, and both erase it before programming `S + 1`. The second
    /// erase destroys the first's manifest, and `manifest::select` breaks an
    /// equal-sequence tie by slot position rather than recency — so one append's
    /// vectors end up written, durable, CRC-valid and described by no live
    /// manifest. Orphaned, while its caller was told it succeeded.
    ///
    /// The data blocks themselves are safe without a lock, by accident of the
    /// medium: NOR programming cannot set a cleared bit, so a second write to the
    /// same block is refused rather than blended. It is the bookkeeping that
    /// needs serialising.
    ///
    /// The lock is released when this value drops, including on error and on
    /// panic, and the kernel releases it if the process dies — so a crashed
    /// append does not leave a volume permanently unappendable.
    fn open(path: &Path) -> Result<Self, AppendError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        // `try_lock` rather than `lock`: see `AppendError::Locked` for why waiting
        // is the worse behaviour here.
        file.try_lock().map_err(|_| AppendError::Locked)?;
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }

    fn sync(&mut self) -> Result<(), AppendError> {
        self.file.flush()?;
        // The append's durability boundary. Without this the manifest write may
        // reach the page cache and not the device, and a power loss would leave a
        // volume whose blocks are written and whose extent is not.
        self.file.sync_all()?;
        Ok(())
    }
}

impl NorFlash for WritableVolume {
    type Error = Error;

    fn page_size(&self) -> usize {
        BLOCK_BYTES
    }

    fn sector_size(&self) -> usize {
        sector_format::SECTOR_BYTES
    }

    fn capacity(&self) -> u32 {
        // Truncated deliberately: the format addresses with a u32, so a volume
        // beyond 4 GiB is refused at open by `FileFlash` before reaching here.
        self.len.min(u32::MAX as u64) as u32
    }

    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), Self::Error> {
        let end = addr as u64 + buf.len() as u64;
        if end > self.len {
            return Err(Error::OutOfBounds {
                addr,
                len: buf.len(),
            });
        }
        self.file.read_exact_at(buf, addr as u64)?;
        Ok(())
    }

    fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), Self::Error> {
        let end = addr as u64 + buf.len() as u64;
        if end > self.len {
            return Err(Error::OutOfBounds {
                addr,
                len: buf.len(),
            });
        }
        // NOR programming clears bits; it cannot set them. A write that would
        // require a 0 to become 1 is only possible after an erase, and permitting
        // it here would let the host produce an image no device could have
        // produced — which is exactly the class of bug a simulation must not hide.
        let mut existing = vec![0u8; buf.len()];
        self.file.read_exact_at(&mut existing, addr as u64)?;
        for (i, (new, old)) in buf.iter().zip(existing.iter()).enumerate() {
            if new & !old != 0 {
                return Err(Error::OutOfBounds {
                    addr: addr + i as u32,
                    len: 1,
                });
            }
        }
        self.file.write_all_at(buf, addr as u64)?;
        Ok(())
    }

    fn erase(&mut self, sector_addr: u32) -> Result<(), Self::Error> {
        // Only the two manifest slots may be erased, and only because installing a
        // manifest *requires* it: a slot that already holds one cannot be
        // reprogrammed, since NOR cannot set a cleared bit. This is why the format
        // has two slots — the spare is erased and rewritten while the live one
        // stays intact, so an interrupted install falls back rather than losing
        // the volume.
        //
        // Erasing a data region would mean rewriting live vectors, which is
        // `sector build`'s job and not an append's. Refused, so a bug in the append
        // path cannot destroy a corpus.
        let sector = sector_format::SECTOR_BYTES as u32;
        if sector_addr != manifest::SLOT_A_OFFSET && sector_addr != manifest::SLOT_B_OFFSET {
            return Err(Error::ReadOnly);
        }
        let mut erased = vec![ERASED_BYTE; sector as usize];
        self.file.write_all_at(&erased, sector_addr as u64)?;
        erased.clear();
        Ok(())
    }
}

/// Read the codebook as i8 components.
fn read_codebook(path: &Path, g: &Geometry) -> Result<Vec<i8>, AppendError> {
    let mut flash = FileFlash::open(path).map_err(|e| AppendError::Volume(e.to_string()))?;
    let volume =
        HostVolume::mount(&mut flash, None).map_err(|e| AppendError::Volume(e.to_string()))?;
    let desc = volume
        .manifest
        .table
        .get(RegionKind::Codebook)
        .ok_or_else(|| AppendError::Volume("volume has no codebook region".into()))?;
    let bytes = g.centroids * g.d;
    let mut raw = vec![0u8; bytes];
    flash
        .read(desc.base, &mut raw)
        .map_err(|e| AppendError::Volume(e.to_string()))?;
    Ok(raw.into_iter().map(|b| b as i8).collect())
}

/// Encode one vector: nearest centroid per subspace, plus its rerank record.
///
/// Returns the codes, the record, and the squared quantization error that feeds
/// [`Drift`].
fn encode_one(v: &[f32], g: &Geometry, codebook: &[i8]) -> (Vec<u8>, Vec<u8>, u64) {
    let ds = g.ds.max(1);
    let mut codes = vec![0u8; g.m];
    let mut error = 0u64;

    // Quantize to i8 the same way the builder does, so an appended record is
    // comparable with a built one under `exact_score`.
    let peak = v.iter().fold(0f32, |acc, x| acc.max(x.abs())).max(1e-9);
    let scale = 127.0 / peak;
    let record: Vec<u8> = v
        .iter()
        .map(|x| ((x * scale).round().clamp(-127.0, 127.0) as i8) as u8)
        .collect();

    for j in 0..g.m {
        let sub = &record[j * ds..(j + 1) * ds];
        let mut best = 0usize;
        let mut best_dist = i64::MAX;
        for c in 0..g.centroids {
            let centroid = &codebook[c * g.d + j * ds..c * g.d + (j + 1) * ds];
            let mut dist = 0i64;
            for (a, b) in sub.iter().zip(centroid.iter()) {
                let d = (*a as i8 as i64) - (*b as i64);
                dist += d * d;
            }
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        codes[j] = best as u8;
        error += best_dist as u64;
    }
    (codes, record, error)
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use crate::volume::test_support::{build_reserved_image, TempDir};

    const D: usize = 32;
    const M: usize = 4;
    const N: usize = 384;

    /// A reserved volume on disk. N is a multiple of the 128-id append unit here,
    /// so the no-gap case is the default and the gap is tested separately.
    fn reserved(tag: &str, n: usize, reserve: usize) -> (TempDir, std::path::PathBuf, Vec<f32>) {
        let dir = TempDir::new(tag);
        let (image, corpus) = build_reserved_image(D, M, n, reserve);
        let path = dir.path().join("volume.sector");
        std::fs::write(&path, &image).expect("write");
        (dir, path, corpus)
    }

    #[test]
    fn the_append_unit_is_the_lcm_of_both_regions() {
        // The constraint that makes an append whole-block in payload and rerank
        // at once. At D=32 m=4: 128 codes and 16 records per block.
        let (_d, path, _) = reserved("unit", N, 256);
        let mut f = FileFlash::open(&path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        assert_eq!(append_unit(&v.geometry), 128);
    }

    #[test]
    fn a_fresh_reserved_volume_reports_capacity_and_no_gap() {
        let (_d, path, _) = reserved("cap", N, 256);
        let mut f = FileFlash::open(&path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        // No gap until something is appended.
        assert_eq!(v.manifest.gap(), (N as u32, N as u32));
        assert_eq!(v.manifest.stored(), N as u32);
        assert_eq!(v.manifest.appended(), 0);

        let (p, r) = capacity(&path).expect("capacity");
        assert!(p >= 256, "payload slots {p}");
        assert!(r >= 256, "rerank slots {r}");
    }

    #[test]
    fn appending_extends_the_volume_and_the_new_ids_are_queryable() {
        let (_d, path, corpus) = reserved("append", N, 256);
        // Reuse corpus vectors as the appended batch: their nearest centroids are
        // populated, so this tests the mechanism rather than codebook coverage.
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();

        let report = append(&path, &batch).expect("append");
        assert_eq!(report.count, 128);
        assert_eq!(report.first_id, N as u32, "N is unit-aligned, so no gap");
        assert_eq!(report.gap, (N as u32, N as u32));
        assert_eq!(report.n, N as u32 + 128);
        assert_eq!(report.payload_blocks, 1);
        assert_eq!(report.rerank_blocks, 8);

        // The volume still mounts, and reports the new extent.
        let mut f = FileFlash::open(&path).expect("reopen");
        let v = HostVolume::mount(&mut f, None).expect("remount");
        assert_eq!(v.manifest.n, N as u32 + 128);
        assert_eq!(v.manifest.stored(), N as u32 + 128);
        assert_eq!(v.manifest.sequence, 2, "the manifest rotated");
    }

    #[test]
    fn an_appended_vector_is_retrievable_by_query() {
        // The property that matters: an appended id can be returned by a search,
        // with a CRC that verifies. If the CRC arrays were out of step this drops.
        let (_d, path, corpus) = reserved("queryable", N, 256);
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        let report = append(&path, &batch).expect("append");

        let mut s: crate::search::Searcher<FileFlash> =
            crate::search::Searcher::open(&path, None).expect("open");
        let answer = s.search(&batch[0], 10).expect("search");
        assert_eq!(
            answer.stats.scan.scanned, report.n,
            "the scan must cover the extended corpus"
        );
        assert_eq!(
            answer.stats.rerank.dropped, 0,
            "no candidate may drop on a cleanly appended volume"
        );
    }

    #[test]
    fn a_misaligned_batch_is_refused_with_the_unit() {
        // Refused rather than padded here: padding is a CLI convenience, and the
        // library must not invent vectors.
        let (_d, path, corpus) = reserved("misaligned", N, 256);
        let batch: Vec<Vec<f32>> = (0..5)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        match append(&path, &batch) {
            Err(AppendError::NotAligned { count, unit }) => {
                assert_eq!(count, 5);
                assert_eq!(unit, 128);
            }
            other => panic!("expected NotAligned, got {other:?}"),
        }
    }

    #[test]
    fn a_wrong_dimension_is_refused() {
        let (_d, path, _) = reserved("dim", N, 256);
        let batch: Vec<Vec<f32>> = (0..128).map(|_| vec![0.0f32; D + 1]).collect();
        assert!(matches!(
            append(&path, &batch),
            Err(AppendError::Dimension { .. })
        ));
    }

    #[test]
    fn a_non_finite_component_is_refused_rather_than_quantized() {
        // `NaN as i8` is 0 in Rust, so quantizing would drop the component from
        // every later score with no signal at all.
        let (_d, path, _) = reserved("nan", N, 256);
        let mut v = vec![1.0f32; D];
        v[7] = f32::NAN;
        let batch: Vec<Vec<f32>> = (0..128).map(|_| v.clone()).collect();
        match append(&path, &batch) {
            Err(AppendError::NonFinite { at }) => assert_eq!(at, 7),
            other => panic!("expected NonFinite, got {other:?}"),
        }
    }

    #[test]
    fn appending_past_the_reserve_is_refused_and_names_which_region_bound() {
        // Rerank binds first at every shipped profile: 8x the payload's bytes per
        // vector at D=128, and 8x here too.
        let (_d, path, corpus) = reserved("full", N, 128);
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        append(&path, &batch).expect("first append fits the reserve");

        match append(&path, &batch) {
            Err(AppendError::Full {
                payload_slots,
                rerank_slots,
            }) => {
                assert!(
                    rerank_slots < 128,
                    "rerank should bind first: payload {payload_slots}, rerank {rerank_slots}"
                );
            }
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn a_volume_with_no_reserve_cannot_be_appended_to() {
        let (_d, path, corpus) = reserved("noreserve", N, 0);
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        assert!(matches!(
            append(&path, &batch),
            Err(AppendError::Full { .. })
        ));
    }

    #[test]
    fn a_misaligned_built_corpus_produces_exactly_one_gap() {
        // The case the manifest fields exist for. N=400 is not a multiple of 128,
        // so the tail block is sealed with 112 slots unreachable.
        let (_d, path, corpus) = reserved("gap", 400, 256);
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        let report = append(&path, &batch).expect("append");

        assert_eq!(
            report.first_id, 512,
            "the append starts at a block boundary"
        );
        assert_eq!(report.gap, (400, 512), "112 phantom ids");

        let mut f = FileFlash::open(&path).expect("reopen");
        let v = HostVolume::mount(&mut f, None).expect("remount");
        assert_eq!(v.manifest.stored(), 400 + 128);
        assert_eq!(v.manifest.n, 640);
        // Ids in the gap are not held; ids either side are.
        assert!(v.manifest.holds(399));
        assert!(!v.manifest.holds(400));
        assert!(!v.manifest.holds(511));
        assert!(v.manifest.holds(512));
        assert!(v.manifest.holds(639));
        assert!(!v.manifest.holds(640));
    }

    #[test]
    fn a_second_append_does_not_create_a_second_gap() {
        // Appended runs are block-aligned at both ends, so only the boundary with
        // the built corpus can be misaligned. One pair of fields is sufficient.
        let (_d, path, corpus) = reserved("twogaps", 400, 512);
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        let first = append(&path, &batch).expect("first");
        let second = append(&path, &batch).expect("second");

        assert_eq!(first.gap, second.gap, "the gap must not move");
        assert_eq!(second.first_id, first.first_id + 128, "runs are contiguous");
        let mut f = FileFlash::open(&path).expect("reopen");
        let v = HostVolume::mount(&mut f, None).expect("remount");
        assert_eq!(v.manifest.stored(), 400 + 256);
    }

    #[test]
    fn drift_accumulates_and_reports_the_appended_share() {
        let (_d, path, corpus) = reserved("drift", N, 512);
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        let first = append(&path, &batch).expect("first");
        assert_eq!(first.drift.appended, 128);
        assert_eq!(first.drift.built, N as u32);

        let second = append(&path, &batch).expect("second");
        assert_eq!(
            second.drift.appended, 256,
            "drift must accumulate across appends, not reset"
        );
        // The appended share is what `warrants_rebuild` thresholds on.
        assert!(second.drift.appended_ppm() > first.drift.appended_ppm());
    }

    #[test]
    fn the_writable_backend_refuses_to_set_a_cleared_bit() {
        // NOR programming clears bits and cannot set them. Permitting it would let
        // the host build an image no device could have produced, which is exactly
        // what a simulation must not hide.
        let (_d, path, _) = reserved("norsemantics", N, 256);
        let mut w = WritableVolume::open(&path).expect("open");
        let mut first = [0u8; BLOCK_BYTES];
        w.read(0, &mut first).expect("read");
        // Slot A holds a written manifest, so some bits are already 0.
        assert!(first.iter().any(|b| *b != ERASED_BYTE));
        assert!(
            w.program(0, &[0xFF; BLOCK_BYTES]).is_err(),
            "setting cleared bits must be refused"
        );

        // A manifest slot may be erased: installing a manifest requires it, since a
        // slot holding one cannot be reprogrammed.
        assert!(w.erase(sector_format::manifest::SLOT_B_OFFSET).is_ok());

        // A data region may not. An erase there would destroy live vectors, which
        // is `sector build`'s job and never an append's.
        let (_d2, path2, _) = reserved("norsemantics_data", N, 256);
        let mut w2 = WritableVolume::open(&path2).expect("open");
        let mut f2 = FileFlash::open(&path2).expect("open");
        let v2 = HostVolume::mount(&mut f2, None).expect("mount");
        let payload_base = v2
            .manifest
            .table
            .get(RegionKind::Payload)
            .expect("payload region")
            .base;
        assert!(
            matches!(w2.erase(payload_base), Err(Error::ReadOnly)),
            "erasing a data region must be refused"
        );
    }

    #[test]
    fn the_scan_skips_gap_ids_entirely() {
        // The recall property, and the reason the reader takes the gap.
        //
        // A gap id's payload code lives inside the built corpus's last block —
        // which has a valid CRC, because the builder padded and sealed it. Nothing
        // downstream can tell padding from a vector: stage one would score it, it
        // would occupy a candidate slot, and stage two would drop it when the
        // erased rerank block failed its CRC. Correct answers, silently worse
        // recall.
        let (_d, path, corpus) = reserved("gapscan", 400, 256);
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        append(&path, &batch).expect("append");

        let mut s: crate::search::Searcher<FileFlash> =
            crate::search::Searcher::open(&path, None).expect("open");
        let answer = s.search(&batch[0], 10).expect("search");

        // 400 built + 128 appended, not the 640-id addressable extent.
        assert_eq!(
            answer.stats.scan.scanned, 528,
            "the scan must cover stored vectors, not addressable ids"
        );
        // No gap id may appear in a result.
        for id in &answer.ids {
            assert!(
                *id < 400 || *id >= 512,
                "id {id} is in the gap and must never be returned"
            );
        }
        // And nothing is dropped: a drop here would mean a gap id reached stage two.
        assert_eq!(
            answer.stats.rerank.dropped, 0,
            "a gap id reached stage two and was dropped, wasting a candidate slot"
        );
    }

    #[test]
    fn a_second_concurrent_append_is_refused_rather_than_orphaning_vectors() {
        // The race this lock exists for, and it is a bookkeeping race rather than a
        // data one. Two appends both read sequence S, both compute the same spare
        // manifest slot, and both erase it before programming S+1. The second erase
        // destroys the first's manifest, and `select` breaks an equal-sequence tie
        // by slot position rather than recency — so one append's vectors would be
        // written, durable, CRC-valid, and described by no live manifest.
        let (_d, path, corpus) = reserved("locked", N, 512);

        // Hold the lock the way an in-progress append does.
        let _held = WritableVolume::open(&path).expect("first writer");

        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        match append(&path, &batch) {
            Err(AppendError::Locked) => {}
            other => panic!("expected Locked, got {other:?}"),
        }

        // Released on drop, so a finished — or crashed — append does not leave the
        // volume permanently unappendable.
        drop(_held);
        let report = append(&path, &batch).expect("append after the lock is released");
        assert_eq!(report.count, 128);
    }

    #[test]
    fn an_append_does_not_disturb_a_reader_that_mounted_before_it() {
        // The claim in `sector-serve`'s api docs: an append programs only erased
        // blocks and installs into the SPARE manifest slot, so a reader holding an
        // older mount stays correct — stale, not wrong. Worth pinning, because it is
        // what makes `sector append` safe to run while something else reads.
        let (_d, path, corpus) = reserved("stalereader", N, 512);
        let mut before: crate::search::Searcher<FileFlash> =
            crate::search::Searcher::open(&path, None).expect("open");
        let query = corpus[..D].to_vec();
        let first = before.search(&query, 10).expect("search before");

        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        append(&path, &batch).expect("append");

        // Same handle, after the append: still answers, still verifies, still sees
        // exactly the corpus it mounted.
        let after = before.search(&query, 10).expect("search after");
        assert_eq!(after.ids, first.ids, "a mounted reader must not shift");
        assert_eq!(
            after.stats.scan.scanned, N as u32,
            "the stale reader must see its own extent, not the new one"
        );
        assert_eq!(
            after.stats.rerank.dropped, 0,
            "nothing the reader had already mounted may fail its CRC"
        );

        // A fresh mount sees the appended vectors.
        let mut fresh: crate::search::Searcher<FileFlash> =
            crate::search::Searcher::open(&path, None).expect("remount");
        let seen = fresh.search(&query, 10).expect("search fresh");
        assert_eq!(seen.stats.scan.scanned, N as u32 + 128);
    }

    #[test]
    fn a_clean_appended_volume_sweeps_clean() {
        // The invariant that keeps `dropped` meaningful: reserved and gap blocks
        // must not be reported as damage, or every appended volume looks corrupt.
        let (_d, path, corpus) = reserved("sweep", 400, 256);
        let batch: Vec<Vec<f32>> = (0..128)
            .map(|i| corpus[i * D..(i + 1) * D].to_vec())
            .collect();
        append(&path, &batch).expect("append");

        let mut f = FileFlash::open(&path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        let report = crate::verify::verify(&mut f, &v).expect("verify");
        assert!(
            report.is_clean(),
            "a cleanly appended volume reported damage: {:?}",
            report.regions
        );
    }
}
