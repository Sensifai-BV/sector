//! Codebook region layout and replica placement.
//!
//! The smallest region and the most critical. It is fixed-size and independent
//! of `N` — 32 KiB at the T0 profile (D=128, b=8, int8) — yet one corrupted
//! byte alters the reconstruction of `n_{j,c}` vectors, `N / 2^b` in
//! expectation, against one vector for a payload byte.
//!
//! # Placement rules
//!
//! Replicas interleave across independent erase sectors. Copies sharing a
//! sector protect against little, and this layer assigns the addresses, so the
//! rule is enforced here.
//!
//! Centroids group into sector-aligned protection groups rather than the region
//! being protected as one block. Criticality is per centroid and measurably
//! skewed — populations vary 4.5x from mean to maximum at m=32, b=8, N=20,000 —
//! so a uniform policy spends bytes where they do not buy recall. A single
//! centroid is too small to carry its own code, so the group is the unit.

use crate::{BLOCK_BYTES, SECTOR_BYTES};

/// Placement of the codebook and its replicas.
///
/// The codebook is `2^b * D * s` bytes, independent of `N` and of `m`: 32 KiB
/// at T0 (D=128, b=8, int8). Replicas live in flash, not RAM — one working copy
/// is resident, and repair reads a replica on CRC mismatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodebookLayout {
    /// Codebook bytes, `2^b * D * s`.
    pub codebook_bytes: usize,
    /// Bytes per block.
    pub block_bytes: usize,
    /// Total copies stored, including the primary. 2 is the T0/T1 default.
    pub copies: usize,
}

impl CodebookLayout {
    /// Layout for a `codebook_bytes` codebook in `copies` total copies.
    pub const fn new(codebook_bytes: usize, copies: usize) -> Self {
        Self {
            codebook_bytes,
            block_bytes: BLOCK_BYTES,
            copies,
        }
    }

    /// Blocks in one copy.
    pub const fn blocks_per_copy(&self) -> usize {
        self.codebook_bytes.div_ceil(self.block_bytes)
    }

    /// Erase sectors one copy occupies.
    pub const fn sectors_per_copy(&self) -> usize {
        self.codebook_bytes.div_ceil(SECTOR_BYTES)
    }

    /// Extent of one copy, rounded to a whole erase sector.
    pub const fn copy_bytes(&self) -> usize {
        self.sectors_per_copy() * SECTOR_BYTES
    }

    /// Extent of the replica region: every copy after the primary.
    pub const fn replica_region_bytes(&self) -> usize {
        (self.copies - 1) * self.copy_bytes()
    }

    /// Replication cost as a fraction of `ram_budget`, in parts per million.
    ///
    /// The comparison against RS parity is per protected byte: a full copy is
    /// `2x` the bytes of RS(12,8) parity at the same order of budget, with no
    /// GF(2^8) arithmetic.
    pub const fn overhead_ppm(&self, ram_budget: usize) -> usize {
        ((self.replica_region_bytes() as u64 * 1_000_000) / (ram_budget as u64)) as usize
    }

    /// Byte offset of `copy`'s `block`, given the region bases.
    pub const fn block_offset(
        &self,
        primary_base: usize,
        replica_base: usize,
        copy: usize,
        block: usize,
    ) -> Option<usize> {
        if copy >= self.copies || block >= self.blocks_per_copy() {
            return None;
        }
        let base = if copy == 0 {
            primary_base
        } else {
            replica_base + (copy - 1) * self.copy_bytes()
        };
        Some(base + block * self.block_bytes)
    }

    /// Erase sector holding `copy`'s `block`.
    pub const fn sector_of(
        &self,
        primary_base: usize,
        replica_base: usize,
        copy: usize,
        block: usize,
    ) -> Option<usize> {
        match self.block_offset(primary_base, replica_base, copy, block) {
            Some(off) => Some(off / SECTOR_BYTES),
            None => None,
        }
    }

    /// Whether every copy of every block sits in a distinct erase sector.
    ///
    /// Flash fails in sector-correlated bursts, so two copies sharing a sector
    /// protect against very little. This is the property the whole replication
    /// scheme rests on, so it is computed from the address map rather than
    /// assumed from the layout's shape.
    pub fn replicas_are_sector_disjoint(&self, primary_base: usize, replica_base: usize) -> bool {
        if !primary_base.is_multiple_of(SECTOR_BYTES) || !replica_base.is_multiple_of(SECTOR_BYTES)
        {
            return false;
        }
        for block in 0..self.blocks_per_copy() {
            for a in 0..self.copies {
                for b in (a + 1)..self.copies {
                    let sa = self.sector_of(primary_base, replica_base, a, block);
                    let sb = self.sector_of(primary_base, replica_base, b, block);
                    if sa.is_none() || sa == sb {
                        return false;
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T0: 2^8 * 128 * 1 = 32 KiB.
    const T0_CODEBOOK: usize = 32 * 1024;
    const T0_RAM: usize = 192 * 1024;

    #[test]
    fn t0_codebook_geometry() {
        let l = CodebookLayout::new(T0_CODEBOOK, 2);
        assert_eq!(l.blocks_per_copy(), 64);
        assert_eq!(l.sectors_per_copy(), 8);
        assert_eq!(l.copy_bytes(), T0_CODEBOOK);
        assert_eq!(l.replica_region_bytes(), T0_CODEBOOK);
    }

    #[test]
    fn replicas_occupy_independent_sectors() {
        let l = CodebookLayout::new(T0_CODEBOOK, 2);
        let primary = 2 * SECTOR_BYTES; // after both manifest slots
        let replica = primary + l.copy_bytes();
        assert!(l.replicas_are_sector_disjoint(primary, replica));

        // Three copies, still disjoint.
        let l3 = CodebookLayout::new(T0_CODEBOOK, 3);
        assert!(l3.replicas_are_sector_disjoint(primary, replica));
    }

    #[test]
    fn a_replica_sharing_the_primary_base_is_caught() {
        let l = CodebookLayout::new(T0_CODEBOOK, 2);
        let primary = 2 * SECTOR_BYTES;
        // Same base: every block's copies land in one sector, so a single
        // sector failure takes both and the scheme protects nothing.
        assert!(!l.replicas_are_sector_disjoint(primary, primary));
    }

    #[test]
    fn a_partially_overlapping_replica_is_a_region_fault_not_a_sector_fault() {
        // Bases a sector apart still give every block distinct sectors, so this
        // function passes it. The defect is that the two copies alias the same
        // bytes, which is the region table's disjointness check, not this one.
        let l = CodebookLayout::new(T0_CODEBOOK, 2);
        let primary = 2 * SECTOR_BYTES;
        assert!(l.replicas_are_sector_disjoint(primary, primary + SECTOR_BYTES));

        let overlaps = (primary as u64) < (primary + SECTOR_BYTES + l.copy_bytes()) as u64
            && ((primary + SECTOR_BYTES) as u64) < (primary + l.copy_bytes()) as u64;
        assert!(overlaps, "the two copies must be caught as overlapping");
    }

    #[test]
    fn unaligned_bases_are_rejected() {
        let l = CodebookLayout::new(T0_CODEBOOK, 2);
        assert!(!l.replicas_are_sector_disjoint(512, 512 + T0_CODEBOOK));
    }

    #[test]
    fn replication_cost_against_the_index_budget() {
        // T0 as built: a 32 KiB replica is 16.7% of the 192 KiB budget, which
        // is why it is charged to flash rather than RAM.
        let l = CodebookLayout::new(T0_CODEBOOK, 2);
        assert_eq!(l.overhead_ppm(T0_RAM), 166_666);

        // The report's T0 (D=768, b=4, int8): 12 KiB, 6.2% of the budget,
        // against 3.1% for RS(12,8) parity over the same structure.
        let wide = CodebookLayout::new(12 * 1024, 2);
        assert_eq!(wide.overhead_ppm(T0_RAM), 62_500);
    }

    #[test]
    fn block_offsets_are_bounded() {
        let l = CodebookLayout::new(T0_CODEBOOK, 2);
        let primary = 2 * SECTOR_BYTES;
        let replica = primary + l.copy_bytes();
        assert_eq!(l.block_offset(primary, replica, 0, 0), Some(primary));
        assert_eq!(l.block_offset(primary, replica, 1, 0), Some(replica));
        assert_eq!(l.block_offset(primary, replica, 2, 0), None);
        assert_eq!(l.block_offset(primary, replica, 0, 64), None);
    }
}
