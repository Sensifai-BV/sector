//! The `sector` flash partition.
//!
//! A dedicated partition, not a file in a filesystem. Erase-sector alignment is
//! a format requirement, and a filesystem may relocate or fragment a file —
//! which would break the guarantee that codebook replicas occupy independent
//! erase sectors. The XIP window also maps an address range, and a file has no
//! stable address.

/// Byte offset of the volume partition, matching `partitions.csv`.
pub const VOLUME_OFFSET: u32 = 0x0020_0000;

/// Volume partition size: 2 MB of a 4 MB part.
pub const VOLUME_SIZE: u32 = 0x0020_0000;

/// Erase sector size on the ESP32 flash controller.
pub const SECTOR_BYTES: u32 = 4096;

/// Program page size.
pub const PAGE_BYTES: u32 = 256;

/// Base of the memory-mapped flash window on the ESP32-C3.
///
/// The window is probed at mount rather than assumed: whether a range is mapped
/// depends on the partition layout, and a wrong assumption gives either a
/// silent fallback to buffered reads or a borrow of unmapped memory.
pub const XIP_BASE: u32 = 0x3C00_0000;

/// Whether `offset` and `len` lie inside the volume partition.
pub const fn in_volume(offset: u32, len: u32) -> bool {
    match offset.checked_add(len) {
        Some(end) => end <= VOLUME_SIZE,
        None => false,
    }
}

/// Vectors the partition holds at a given per-vector stored size.
///
/// Stored size is payload + rerank + the CRC share of both, so this is the
/// capacity figure the tier profile must agree with.
pub const fn capacity(stored_bytes_per_vector: u32) -> u32 {
    if stored_bytes_per_vector == 0 {
        return 0;
    }
    VOLUME_SIZE / stored_bytes_per_vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_partition_is_sector_aligned() {
        // A partition straddling an erase boundary shares a failure domain
        // with its neighbour and cannot bound its own failure probability.
        assert!(VOLUME_OFFSET.is_multiple_of(SECTOR_BYTES));
        assert!(VOLUME_SIZE.is_multiple_of(SECTOR_BYTES));
    }

    #[test]
    fn bounds_checks_refuse_overflow_rather_than_wrapping() {
        assert!(in_volume(0, VOLUME_SIZE));
        assert!(!in_volume(0, VOLUME_SIZE + 1));
        assert!(!in_volume(VOLUME_SIZE, 1));
        // A wrapping add would report this as inside the partition.
        assert!(!in_volume(u32::MAX, 16));
    }

    #[test]
    fn capacity_matches_the_t0_arithmetic() {
        // T0: 16 B payload + 128 B rerank + 8 B CRC share = 152 B per vector.
        assert_eq!(capacity(152), 13_797);
        assert_eq!(capacity(0), 0);
    }
}
