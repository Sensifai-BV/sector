//! Volume manifest: magic, version, profile, region table, digest.
//!
//! The first object read at mount and the only one at a fixed address. It
//! declares the tier profile the image was built for, so a device refuses an
//! image whose parameters it cannot host rather than mis-reading it.
//!
//! # Write ordering
//!
//! Write the manifest last, after every region it points at is durable, with
//! its digest covering the region table. A manifest that verifies then implies
//! the regions beneath it are complete, and a torn install leaves the previous
//! manifest intact.
//!
//! Keep two manifest slots and alternate between them so an interrupted update
//! can fall back. Parity cannot repair a torn write — a partially written
//! sector is consistently wrong, not noisily wrong — so the remedy is the
//! atomic version switch, and it lives here.
//!
//! Refuse unknown `FORMAT_VERSION` values outright. Best-effort parsing of an
//! unrecognised layout turns a corrupted read into a wrong answer.

use crate::region::{RegionError, RegionTable};
use crate::{FORMAT_VERSION, MAGIC_VOLUME, SECTOR_BYTES};
use sector_codec::crc::crc32;

/// Encoded manifest size. One erase sector, so a slot is independently
/// erasable and a write to one cannot disturb the other.
pub const MANIFEST_BYTES: usize = SECTOR_BYTES;

/// Byte offset of manifest slot A.
pub const SLOT_A_OFFSET: u32 = 0;

/// Byte offset of manifest slot B.
pub const SLOT_B_OFFSET: u32 = SECTOR_BYTES as u32;

/// Bytes the two manifest slots occupy at the head of a volume.
pub const MANIFEST_RESERVED_BYTES: u32 = 2 * SECTOR_BYTES as u32;

/// Bytes of the encoded manifest the digest covers: everything before the
/// digest field itself.
const DIGEST_SCOPE: usize = 4 + 2 + 2 + 8 + 2 + 2 + 2 + 2 + 4 + 4 + RegionTable::ENCODED_BYTES;

/// Offset of the digest word within the encoded manifest.
const DIGEST_OFFSET: usize = DIGEST_SCOPE;

/// Why a manifest was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestError {
    /// Leading bytes are not [`MAGIC_VOLUME`].
    BadMagic,
    /// Layout version the reader does not implement.
    ///
    /// Refused outright: best-effort parsing of an unrecognised layout turns a
    /// corrupted read into a wrong answer.
    UnsupportedVersion {
        /// The version the image declares.
        found: u16,
    },
    /// The digest does not match the bytes it covers.
    DigestMismatch {
        /// Digest computed over the stored bytes.
        computed: u32,
        /// Digest the image carries.
        stored: u32,
    },
    /// Buffer is shorter than an encoded manifest.
    Truncated,
    /// A region descriptor is invalid, or the table is not disjoint.
    Region(RegionError),
    /// Neither slot verified.
    NoValidSlot,
}

/// Volume manifest: magic, version, profile parameters, region table, digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Manifest {
    /// Increments on every install. The higher valid slot is current.
    pub sequence: u64,
    /// Vector dimension the image was built for.
    pub d: u16,
    /// PQ subspaces.
    pub m: u16,
    /// Bits per code.
    pub b: u16,
    /// Bytes per codebook component.
    pub cb_bytes: u16,
    /// Vectors stored.
    pub n: u32,
    /// Rerank candidate depth the image was built for.
    pub r: u32,
    /// Region table.
    pub table: RegionTable,
}

impl Manifest {
    /// Encode into `out`, computing the digest over everything preceding it.
    ///
    /// The digest covers the region table, so a manifest that verifies pins the
    /// addresses of every region beneath it.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, ManifestError> {
        let buf = out
            .get_mut(..MANIFEST_BYTES)
            .ok_or(ManifestError::Truncated)?;
        buf.fill(0xFF);

        let mut w = Writer { buf, at: 0 };
        w.bytes(&MAGIC_VOLUME);
        w.u16(FORMAT_VERSION);
        w.u16(0); // reserved, keeps the profile block 4-byte aligned
        w.u64(self.sequence);
        w.u16(self.d);
        w.u16(self.m);
        w.u16(self.b);
        w.u16(self.cb_bytes);
        w.u32(self.n);
        w.u32(self.r);
        let at = w.at;

        let written = self
            .table
            .encode(buf.get_mut(at..).ok_or(ManifestError::Truncated)?)
            .ok_or(ManifestError::Truncated)?;
        let scope = at + written;
        debug_assert!(scope == DIGEST_SCOPE);

        let digest = crc32(buf.get(..scope).ok_or(ManifestError::Truncated)?);
        buf.get_mut(scope..scope + 4)
            .ok_or(ManifestError::Truncated)?
            .copy_from_slice(&digest.to_le_bytes());
        Ok(MANIFEST_BYTES)
    }

    /// Decode and verify, in that order: magic, version, digest, then regions.
    ///
    /// No field is trusted before the digest that covers it verifies.
    pub fn decode(raw: &[u8]) -> Result<Self, ManifestError> {
        let buf = raw
            .get(..DIGEST_OFFSET + 4)
            .ok_or(ManifestError::Truncated)?;

        if buf.get(..4) != Some(&MAGIC_VOLUME[..]) {
            return Err(ManifestError::BadMagic);
        }
        let version = u16::from_le_bytes([buf[4], buf[5]]);
        if version != FORMAT_VERSION {
            return Err(ManifestError::UnsupportedVersion { found: version });
        }

        let stored = u32::from_le_bytes([
            buf[DIGEST_OFFSET],
            buf[DIGEST_OFFSET + 1],
            buf[DIGEST_OFFSET + 2],
            buf[DIGEST_OFFSET + 3],
        ]);
        let computed = crc32(buf.get(..DIGEST_SCOPE).ok_or(ManifestError::Truncated)?);
        if computed != stored {
            return Err(ManifestError::DigestMismatch { computed, stored });
        }

        let mut r = Reader { buf, at: 8 };
        let sequence = r.u64();
        let d = r.u16();
        let m = r.u16();
        let b = r.u16();
        let cb_bytes = r.u16();
        let n = r.u32();
        let rerank_depth = r.u32();

        let table = RegionTable::decode(buf.get(r.at..).ok_or(ManifestError::Truncated)?)
            .map_err(ManifestError::Region)?;
        table.validate().map_err(ManifestError::Region)?;

        Ok(Self {
            sequence,
            d,
            m,
            b,
            cb_bytes,
            n,
            r: rerank_depth,
            table,
        })
    }
}

/// Choose the live manifest from the two slots.
///
/// The higher sequence among verifying slots wins. A torn install leaves its
/// half-written slot failing its digest, so the previous manifest is still
/// selected and the volume mounts at its old contents.
pub fn select(slot_a: &[u8], slot_b: &[u8]) -> Result<Manifest, ManifestError> {
    match (Manifest::decode(slot_a), Manifest::decode(slot_b)) {
        (Ok(a), Ok(b)) => Ok(if b.sequence > a.sequence { b } else { a }),
        (Ok(a), Err(_)) => Ok(a),
        (Err(_), Ok(b)) => Ok(b),
        (Err(_), Err(_)) => Err(ManifestError::NoValidSlot),
    }
}

/// Offset of the slot the next install writes: the one not currently live.
#[allow(clippy::manual_is_multiple_of)] // `is_multiple_of` is not const on stable
pub const fn next_slot_offset(live_sequence: u64) -> u32 {
    if live_sequence % 2 == 0 {
        SLOT_B_OFFSET
    } else {
        SLOT_A_OFFSET
    }
}

struct Writer<'a> {
    buf: &'a mut [u8],
    at: usize,
}

impl Writer<'_> {
    fn bytes(&mut self, v: &[u8]) {
        if let Some(dst) = self.buf.get_mut(self.at..self.at + v.len()) {
            dst.copy_from_slice(v);
        }
        self.at += v.len();
    }
    fn u16(&mut self, v: u16) {
        self.bytes(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        if let Some(src) = self.buf.get(self.at..self.at + N) {
            out.copy_from_slice(src);
        }
        self.at += N;
        out
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take::<2>())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take::<4>())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take::<8>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::{Protection, RegionDesc, RegionKind, REGION_COUNT};
    use crate::BLOCK_BYTES;

    fn table() -> RegionTable {
        let sector = SECTOR_BYTES as u32;
        let kinds = [
            RegionKind::Codebook,
            RegionKind::CodebookReplica,
            RegionKind::Payload,
            RegionKind::PayloadCrc,
            RegionKind::Rerank,
            RegionKind::RerankCrc,
        ];
        let mut regions = [RegionDesc {
            kind: RegionKind::Codebook,
            protection: Protection::Detect,
            base: 0,
            block_bytes: BLOCK_BYTES as u32,
            blocks: 8,
        }; REGION_COUNT];
        for (i, slot) in regions.iter_mut().enumerate() {
            slot.kind = kinds[i];
            slot.base = MANIFEST_RESERVED_BYTES + i as u32 * sector;
        }
        RegionTable { regions }
    }

    fn manifest(sequence: u64) -> Manifest {
        Manifest {
            sequence,
            d: 128,
            m: 16,
            b: 8,
            cb_bytes: 1,
            n: 8_966,
            r: 500,
            table: table(),
        }
    }

    #[test]
    fn round_trips() {
        let m = manifest(1);
        let mut buf = [0u8; MANIFEST_BYTES];
        assert_eq!(m.encode(&mut buf), Ok(MANIFEST_BYTES));
        assert_eq!(Manifest::decode(&buf), Ok(m));
    }

    #[test]
    fn digest_covers_the_region_table() {
        let mut buf = [0u8; MANIFEST_BYTES];
        manifest(1).encode(&mut buf).unwrap();
        // Move the payload region by one sector: a field the digest must cover.
        let table_at = DIGEST_SCOPE - RegionTable::ENCODED_BYTES;
        buf[table_at + 2 * 16 + 4] ^= 0x10;
        assert!(matches!(
            Manifest::decode(&buf),
            Err(ManifestError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn unknown_version_is_refused() {
        let mut buf = [0u8; MANIFEST_BYTES];
        manifest(1).encode(&mut buf).unwrap();
        buf[4] = 0xFF;
        buf[5] = 0x00;
        assert_eq!(
            Manifest::decode(&buf),
            Err(ManifestError::UnsupportedVersion { found: 255 })
        );
    }

    #[test]
    fn bad_magic_is_refused() {
        let mut buf = [0u8; MANIFEST_BYTES];
        manifest(1).encode(&mut buf).unwrap();
        buf[0] = b'X';
        assert_eq!(Manifest::decode(&buf), Err(ManifestError::BadMagic));
    }

    #[test]
    fn truncated_image_fails_rather_than_mounting_partially() {
        let mut buf = [0u8; MANIFEST_BYTES];
        manifest(1).encode(&mut buf).unwrap();
        // Every prefix short of the digest must refuse, never half-mount.
        for cut in 0..DIGEST_OFFSET + 4 {
            assert!(
                Manifest::decode(&buf[..cut]).is_err(),
                "prefix of {cut} bytes decoded"
            );
        }
    }

    #[test]
    fn overlapping_regions_are_refused_at_decode() {
        let mut m = manifest(1);
        m.table.regions[0].blocks = 64; // 32 KiB, over its neighbour
        let mut buf = [0u8; MANIFEST_BYTES];
        m.encode(&mut buf).unwrap();
        assert!(matches!(
            Manifest::decode(&buf),
            Err(ManifestError::Region(RegionError::Overlap { .. }))
        ));
    }

    #[test]
    fn torn_install_falls_back_to_the_previous_slot() {
        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        manifest(7).encode(&mut a).unwrap();
        manifest(8).encode(&mut b).unwrap();
        assert_eq!(select(&a, &b).map(|m| m.sequence), Ok(8));

        // Slot B interrupted mid-write: valid prefix, erased tail.
        let torn_at = 64;
        b[torn_at..].fill(0xFF);
        assert_eq!(select(&a, &b).map(|m| m.sequence), Ok(7));
    }

    #[test]
    fn both_slots_bad_is_an_error_not_a_guess() {
        let a = [0xFFu8; MANIFEST_BYTES];
        let b = [0xFFu8; MANIFEST_BYTES];
        assert_eq!(select(&a, &b), Err(ManifestError::NoValidSlot));
    }

    #[test]
    fn install_alternates_slots() {
        assert_eq!(next_slot_offset(0), SLOT_B_OFFSET);
        assert_eq!(next_slot_offset(1), SLOT_A_OFFSET);
        assert_eq!(next_slot_offset(8), SLOT_B_OFFSET);
    }
}
