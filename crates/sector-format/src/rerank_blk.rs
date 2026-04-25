//! Rerank region: the higher-precision copy, CRC-detected and unprotected.
//!
//! The largest object in the system: 128 B/vector at `D = 128` int8 against a
//! 16 B payload, and the dominant term in flash capacity. It carries no parity.
//!
//! The argument is quantitative. Only `R` of `N` vectors are read per query, so
//! rerank exposure carries a factor `R/N` — 5e-3 at N=20,000, 1e-4 at N=10^6,
//! orders below codebook exposure. A corrupted entry mis-scores one candidate
//! rather than relocating a class of vectors, and the region is reconstructible
//! from the source corpus where one is retained.
//!
//! # Capacity note
//!
//! CRC detection with drop-on-mismatch, no parity. Parity here would consume
//! the budget protecting the least critical bytes.
//!
//! This region binds capacity: a 4 MB part holds roughly 32,000 copies at T0.
//! Lowering its precision or holding only a subset trades recall for capacity
//! more efficiently than any change to the payload, and its access cost is
//! bounded by `R` rather than `N`.

use crate::{BLOCK_BYTES, SECTOR_BYTES};
use sector_codec::CRC_BYTES;

/// Placement of `N` higher-precision copies into fixed-size blocks.
///
/// One vector per record at `D * rerank_bytes`, and — unlike the payload —
/// a record may exceed a block. At `D = 128` int8 a record is 128 B, so four
/// fit in a 512 B block; at `D = 768` it spans two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RerankLayout {
    /// Bytes per stored copy, `D * rerank_bytes`.
    pub record_bytes: usize,
    /// Bytes per block.
    pub block_bytes: usize,
    /// Vectors stored.
    pub n: usize,
}

impl RerankLayout {
    /// Layout for `n` copies of `record_bytes` at [`BLOCK_BYTES`].
    pub const fn new(record_bytes: usize, n: usize) -> Self {
        Self {
            record_bytes,
            block_bytes: BLOCK_BYTES,
            n,
        }
    }

    /// Blocks one record spans.
    pub const fn blocks_per_record(&self) -> usize {
        self.record_bytes.div_ceil(self.block_bytes)
    }

    /// Records per block, zero when a record spans more than one.
    pub const fn records_per_block(&self) -> usize {
        self.block_bytes / self.record_bytes
    }

    /// Blocks needed for `n` records.
    pub const fn blocks(&self) -> usize {
        if self.record_bytes <= self.block_bytes {
            self.n.div_ceil(self.records_per_block())
        } else {
            self.n * self.blocks_per_record()
        }
    }

    /// Byte offset of record `id` from the start of the region.
    pub const fn offset_of(&self, id: usize) -> Option<usize> {
        if id >= self.n {
            return None;
        }
        if self.record_bytes <= self.block_bytes {
            let per = self.records_per_block();
            Some((id / per) * self.block_bytes + (id % per) * self.record_bytes)
        } else {
            Some(id * self.blocks_per_record() * self.block_bytes)
        }
    }

    /// Blocks a candidate's rescore must verify, as a half-open range.
    ///
    /// Stage two verifies exactly these before rescoring, and drops the
    /// candidate if any fails.
    pub const fn blocks_of(&self, id: usize) -> Option<(usize, usize)> {
        let off = match self.offset_of(id) {
            Some(o) => o,
            None => return None,
        };
        let first = off / self.block_bytes;
        let last = (off + self.record_bytes - 1) / self.block_bytes;
        Some((first, last + 1))
    }

    /// Region extent, rounded up to a whole erase sector.
    pub const fn region_bytes(&self) -> usize {
        (self.blocks() * self.block_bytes).next_multiple_of(SECTOR_BYTES)
    }

    /// CRC array extent, rounded up to a whole erase sector.
    pub const fn crc_region_bytes(&self) -> usize {
        (self.blocks() * CRC_BYTES).next_multiple_of(SECTOR_BYTES)
    }

    /// Records that fit in `flash_bytes` of region.
    ///
    /// This region binds capacity, so the figure is what decides how large `N`
    /// can grow on a given part.
    pub const fn capacity(&self, flash_bytes: usize) -> usize {
        if self.record_bytes <= self.block_bytes {
            (flash_bytes / self.block_bytes) * self.records_per_block()
        } else {
            flash_bytes / (self.blocks_per_record() * self.block_bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T0: D=128 int8.
    const T0_RECORD: usize = 128;

    #[test]
    fn t0_record_is_eight_times_the_payload() {
        assert_eq!(T0_RECORD / 16, 8);
        let l = RerankLayout::new(T0_RECORD, 8_966);
        assert_eq!(l.records_per_block(), 4);
        assert_eq!(l.blocks_per_record(), 1);
    }

    #[test]
    fn offsets_never_straddle_a_block_when_the_record_fits() {
        let l = RerankLayout::new(T0_RECORD, 1_000);
        for id in 0..l.n {
            let off = l.offset_of(id).unwrap();
            let within = off % l.block_bytes;
            assert!(within + T0_RECORD <= l.block_bytes, "record {id} straddles");
            assert_eq!(l.blocks_of(id).map(|(a, b)| b - a), Some(1));
        }
    }

    #[test]
    fn a_wide_record_spans_and_reports_both_blocks() {
        // D=768 int8: 768 B over 512 B blocks.
        let l = RerankLayout::new(768, 10);
        assert_eq!(l.blocks_per_record(), 2);
        assert_eq!(l.blocks(), 20);
        assert_eq!(l.offset_of(1), Some(1_024));
        assert_eq!(l.blocks_of(1), Some((2, 4)));
    }

    #[test]
    fn capacity_in_four_megabytes_matches_the_budget_figure() {
        let l = RerankLayout::new(T0_RECORD, 8_966);
        // 4 MB NOR less two 32 KiB codebook copies.
        let usable = 4 * 1024 * 1024 - 2 * 32 * 1024;
        assert!(l.capacity(usable) > 30_000);
    }

    #[test]
    fn regions_are_sector_aligned() {
        let l = RerankLayout::new(T0_RECORD, 8_966);
        assert_eq!(l.region_bytes() % SECTOR_BYTES, 0);
        assert_eq!(l.crc_region_bytes() % SECTOR_BYTES, 0);
        assert_eq!(l.blocks(), 2_242);
        assert_eq!(l.offset_of(8_966), None);
    }
}
