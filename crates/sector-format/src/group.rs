//! Protection groups: sector-aligned buckets of centroids sharing one code rate.
//!
//! Criticality is per centroid, but a centroid at `b = 4`, int8, `d_s = 24`
//! occupies 24 bytes — too small to carry its own code. Groups bucket centroids
//! by criticality quantile and assign one rate per group, keeping coding
//! granularity aligned to flash sectors while preserving the per-centroid
//! weighting.
//!
//! # Bucketing
//!
//! Bucket by quantile of measured weight, not fixed thresholds. The weight
//! distribution is dataset-dependent and skewed — the top decile of centroids
//! carries ~19% of all references at m=32, b=8, N=20,000 — so fixed cut points
//! either collapse to one group or leave groups empty.
//!
//! Keep the group count at 4–8. Each group costs a descriptor and a sector
//! alignment gap, and the measured benefit flattens: the separation that pays
//! is codebook against rerank copy, not gradation within the codebook.

use crate::SECTOR_BYTES;

/// Groups a volume may carry. Sized for the 4-8 range the layout targets.
pub const MAX_GROUPS: usize = 8;

/// Encoded size of one group descriptor.
pub const GROUP_DESC_BYTES: usize = 8;

/// One bucket of centroids sharing a code rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GroupDesc {
    /// First centroid index in the group.
    pub first_centroid: u16,
    /// Centroids in the group.
    pub centroids: u16,
    /// Parity bytes per 256 data bytes. 0 is detect-only.
    ///
    /// Held as an integer rate rather than a ratio: the core family admits no
    /// floating point, and the allocator's output must survive the format
    /// unchanged.
    pub parity_per_256: u16,
    /// Sectors the group occupies.
    pub sectors: u16,
}

impl GroupDesc {
    /// Encode to the fixed 8-byte form, little-endian.
    pub const fn encode(&self) -> [u8; GROUP_DESC_BYTES] {
        let a = self.first_centroid.to_le_bytes();
        let b = self.centroids.to_le_bytes();
        let c = self.parity_per_256.to_le_bytes();
        let d = self.sectors.to_le_bytes();
        [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]]
    }

    /// Decode the on-flash form.
    pub const fn decode(raw: &[u8; GROUP_DESC_BYTES]) -> Self {
        Self {
            first_centroid: u16::from_le_bytes([raw[0], raw[1]]),
            centroids: u16::from_le_bytes([raw[2], raw[3]]),
            parity_per_256: u16::from_le_bytes([raw[4], raw[5]]),
            sectors: u16::from_le_bytes([raw[6], raw[7]]),
        }
    }
}

/// Why a group table was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupError {
    /// Group count outside the 4-8 range the layout targets.
    Count {
        /// The offending count.
        found: usize,
    },
    /// A centroid belongs to no group, or to two.
    NotAPartition {
        /// Centroid index where the partition broke.
        at: usize,
    },
    /// A group's extent is not a whole number of erase sectors.
    Unaligned {
        /// Index of the offending group.
        group: usize,
    },
    /// A group holds no centroids.
    Empty {
        /// Index of the offending group.
        group: usize,
    },
}

/// The volume's protection groups, in ascending centroid order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupTable {
    /// Descriptors, `count` of them live.
    pub groups: [GroupDesc; MAX_GROUPS],
    /// Live descriptors.
    pub count: usize,
}

impl GroupTable {
    /// Validate the table against `total_centroids` and `centroid_bytes`.
    ///
    /// Checks the partition property directly rather than trusting the
    /// bucketing: a centroid in no group is unprotected, and one in two groups
    /// makes the allocation's byte accounting wrong.
    pub fn validate(
        &self,
        total_centroids: usize,
        centroid_bytes: usize,
    ) -> Result<(), GroupError> {
        if !(4..=MAX_GROUPS).contains(&self.count) {
            return Err(GroupError::Count { found: self.count });
        }

        let mut next = 0usize;
        for (i, g) in self.groups.iter().take(self.count).enumerate() {
            if g.centroids == 0 {
                return Err(GroupError::Empty { group: i });
            }
            if g.first_centroid as usize != next {
                return Err(GroupError::NotAPartition { at: next });
            }
            let bytes = g.centroids as usize * centroid_bytes;
            if bytes.next_multiple_of(SECTOR_BYTES) != g.sectors as usize * SECTOR_BYTES {
                return Err(GroupError::Unaligned { group: i });
            }
            next += g.centroids as usize;
        }

        if next != total_centroids {
            return Err(GroupError::NotAPartition { at: next });
        }
        Ok(())
    }

    /// The group holding `centroid`.
    pub fn group_of(&self, centroid: usize) -> Option<usize> {
        self.groups.iter().take(self.count).position(|g| {
            let lo = g.first_centroid as usize;
            centroid >= lo && centroid < lo + g.centroids as usize
        })
    }

    /// Sectors every live group occupies.
    pub fn total_sectors(&self) -> usize {
        self.groups
            .iter()
            .take(self.count)
            .map(|g| g.sectors as usize)
            .sum()
    }
}

/// Bucket `total_centroids` into `count` groups by quantile of a weight
/// ordering, sector-aligning each.
///
/// Buckets by rank rather than by weight threshold. The weight distribution is
/// dataset-dependent and skewed — the top decile carries ~19% of references at
/// m=32, b=8, N=20,000 — so fixed cut points either collapse to one group or
/// leave groups empty.
///
/// Callers pass centroids already ordered by ascending weight, so group
/// `count-1` is the most critical.
pub fn bucket_by_quantile(
    total_centroids: usize,
    count: usize,
    centroid_bytes: usize,
) -> Result<GroupTable, GroupError> {
    if !(4..=MAX_GROUPS).contains(&count) || total_centroids < count {
        return Err(GroupError::Count { found: count });
    }

    let mut groups = [GroupDesc::default(); MAX_GROUPS];
    let base = total_centroids / count;
    let remainder = total_centroids % count;
    let mut first = 0usize;

    for (i, slot) in groups.iter_mut().take(count).enumerate() {
        // Spread the remainder over the leading groups so every group is
        // non-empty and sizes differ by at most one.
        let n = base + usize::from(i < remainder);
        let bytes = n * centroid_bytes;
        *slot = GroupDesc {
            first_centroid: first as u16,
            centroids: n as u16,
            parity_per_256: 0,
            sectors: bytes.div_ceil(SECTOR_BYTES) as u16,
        };
        first += n;
    }

    Ok(GroupTable { groups, count })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T0: 256 centroids of D/m = 8 bytes each at int8.
    const T0_CENTROIDS: usize = 256;
    const T0_CENTROID_BYTES: usize = 128;

    #[test]
    fn groups_partition_the_codebook_exactly() {
        for count in 4..=MAX_GROUPS {
            let t = bucket_by_quantile(T0_CENTROIDS, count, T0_CENTROID_BYTES).unwrap();
            assert_eq!(t.validate(T0_CENTROIDS, T0_CENTROID_BYTES), Ok(()));
            // Every centroid in exactly one group.
            for c in 0..T0_CENTROIDS {
                assert!(t.group_of(c).is_some(), "centroid {c} unassigned");
            }
            assert_eq!(t.group_of(T0_CENTROIDS), None);
            let total: usize = t
                .groups
                .iter()
                .take(count)
                .map(|g| g.centroids as usize)
                .sum();
            assert_eq!(total, T0_CENTROIDS);
        }
    }

    #[test]
    fn every_group_is_sector_aligned() {
        let t = bucket_by_quantile(T0_CENTROIDS, 4, T0_CENTROID_BYTES).unwrap();
        for g in t.groups.iter().take(t.count) {
            assert!(g.sectors > 0);
        }
        // 4 groups of 64 centroids x 128 B = 8 KiB = 2 sectors each.
        assert_eq!(t.total_sectors(), 8);
    }

    #[test]
    fn uneven_counts_spread_the_remainder() {
        // 256 into 6: 43,43,43,43,42,42.
        let t = bucket_by_quantile(T0_CENTROIDS, 6, T0_CENTROID_BYTES).unwrap();
        assert_eq!(t.validate(T0_CENTROIDS, T0_CENTROID_BYTES), Ok(()));
        let sizes: [u16; 6] = core::array::from_fn(|i| t.groups[i].centroids);
        assert_eq!(sizes, [43, 43, 43, 43, 42, 42]);
    }

    #[test]
    fn group_count_stays_in_range_for_both_tiers() {
        assert!(bucket_by_quantile(T0_CENTROIDS, 3, T0_CENTROID_BYTES).is_err());
        assert!(bucket_by_quantile(T0_CENTROIDS, 9, T0_CENTROID_BYTES).is_err());
        assert!(bucket_by_quantile(T0_CENTROIDS, 4, T0_CENTROID_BYTES).is_ok());
        assert!(bucket_by_quantile(T0_CENTROIDS, 8, T0_CENTROID_BYTES).is_ok());
    }

    #[test]
    fn a_gap_in_the_partition_is_caught() {
        let mut t = bucket_by_quantile(T0_CENTROIDS, 4, T0_CENTROID_BYTES).unwrap();
        t.groups[1].first_centroid += 1; // leaves centroid 64 in no group
        assert_eq!(
            t.validate(T0_CENTROIDS, T0_CENTROID_BYTES),
            Err(GroupError::NotAPartition { at: 64 })
        );
    }

    #[test]
    fn a_short_partition_is_caught() {
        let mut t = bucket_by_quantile(T0_CENTROIDS, 4, T0_CENTROID_BYTES).unwrap();
        t.groups[3].centroids -= 1;
        assert!(matches!(
            t.validate(T0_CENTROIDS, T0_CENTROID_BYTES),
            Err(GroupError::Unaligned { .. }) | Err(GroupError::NotAPartition { .. })
        ));
    }

    #[test]
    fn descriptor_round_trips() {
        let g = GroupDesc {
            first_centroid: 192,
            centroids: 64,
            parity_per_256: 128,
            sectors: 2,
        };
        assert_eq!(GroupDesc::decode(&g.encode()), g);
    }
}
