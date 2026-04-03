//! N-copy replication with CRC-directed copy selection.
//!
//! The codebook is fixed-size and independent of `N` (`2^b * D * s`), so a full
//! second copy costs a fixed number of bytes. At the report's T0 configuration
//! (D=768, b=4, int8) that is 12 KiB, 6.2% of the 192 KiB index budget against
//! 3.1% for RS(12,8) parity. At the shipped T0 profile (D=128, b=8, int8) the
//! codebook is 32 KiB and its replica is charged to flash rather than RAM.
//!
//! Both mechanisms sit at the same order of budget, and replication needs no
//! GF(2^8) arithmetic, no syndrome computation and no Forney algorithm: repair
//! is a CRC compare and a copy. Reed–Solomon applies where the protected
//! structure is large enough that halving the parity bytes outweighs the decode
//! cost. At MCU codebook sizes it is not.
//!
//! # Placement rules
//!
//! Replicas interleave across independent erase sectors. Flash fails in
//! sector-correlated bursts, so copies sharing a sector protect against little.
//! Addresses are assigned in `sector-format`, which is where the rule is
//! enforced.
//!
//! Copy selection is per block, not per region. With two copies and a per-block
//! CRC, any pattern leaving one good copy of each block is recoverable — which
//! covers patterns that leave no single replica wholly intact.

use crate::crc::verify;

/// Copies a replicated structure may carry.
pub const MAX_COPIES: usize = 4;

/// Outcome of selecting a good copy of one block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// The primary verified; no repair needed.
    Primary,
    /// Copy `index` verified while an earlier one did not.
    Replica {
        /// Index of the copy that verified.
        index: usize,
    },
    /// Every copy failed its CRC.
    Unrecoverable,
}

/// Choose a verifying copy of one block.
///
/// Selection is per block rather than per region: with a per-block CRC, any
/// corruption pattern leaving one good copy of each block is recoverable, which
/// covers patterns where no single replica is wholly intact.
pub fn select_copy(copies: &[&[u8]], expected_crc: u32) -> Selection {
    for (i, copy) in copies.iter().enumerate() {
        if verify(copy, expected_crc) {
            return if i == 0 {
                Selection::Primary
            } else {
                Selection::Replica { index: i }
            };
        }
    }
    Selection::Unrecoverable
}

/// Repair `dst` from the first verifying copy.
///
/// Returns the selection made. `dst` is written only when a replica is used, so
/// an intact primary costs no erase cycle — a rewrite is charged against a
/// ~10^5-cycle endurance budget.
pub fn repair_block(dst: &mut [u8], copies: &[&[u8]], expected_crc: u32) -> Selection {
    match select_copy(copies, expected_crc) {
        Selection::Primary => Selection::Primary,
        Selection::Replica { index } => match copies.get(index) {
            Some(src) if src.len() == dst.len() => {
                dst.copy_from_slice(src);
                Selection::Replica { index }
            }
            _ => Selection::Unrecoverable,
        },
        Selection::Unrecoverable => Selection::Unrecoverable,
    }
}

/// Whether every block has at least one verifying copy.
///
/// `copies[c][b]` is block `b` of copy `c`; `crcs[b]` is block `b`'s expected
/// digest.
pub fn all_blocks_recoverable(copies: &[&[&[u8]]], crcs: &[u32]) -> bool {
    crcs.iter().enumerate().all(|(b, &crc)| {
        copies.iter().any(|copy| match copy.get(b) {
            Some(block) => verify(block, crc),
            None => false,
        })
    })
}

/// Replica bytes for `data_bytes` at `copies` total copies.
///
/// `copies = 2` is the T0/T1 default: one full extra copy, repaired by a CRC
/// compare and a copy, with no GF(2^8) arithmetic at all.
pub const fn replica_bytes(data_bytes: usize, copies: usize) -> usize {
    data_bytes * (copies - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crc::crc32;

    const BLOCK: usize = 512;

    fn block(fill: u8) -> [u8; BLOCK] {
        [fill; BLOCK]
    }

    #[test]
    fn an_intact_primary_is_used_without_a_rewrite() {
        let good = block(0xA5);
        let crc = crc32(&good);
        let replica = block(0xA5);
        let mut dst = good;
        assert_eq!(
            repair_block(&mut dst, &[&good[..], &replica[..]], crc),
            Selection::Primary
        );
        assert_eq!(dst, good);
    }

    #[test]
    fn a_damaged_primary_repairs_from_the_replica() {
        let good = block(0xA5);
        let crc = crc32(&good);
        let mut damaged = good;
        damaged[100] ^= 0x01;

        let mut dst = damaged;
        assert_eq!(
            repair_block(&mut dst, &[&damaged[..], &good[..]], crc),
            Selection::Replica { index: 1 }
        );
        assert_eq!(dst, good);
    }

    #[test]
    fn every_copy_damaged_is_unrecoverable_not_a_guess() {
        let good = block(0xA5);
        let crc = crc32(&good);
        let mut a = good;
        let mut b = good;
        a[0] ^= 0x01;
        b[1] ^= 0x02;
        assert_eq!(
            select_copy(&[&a[..], &b[..]], crc),
            Selection::Unrecoverable
        );
    }

    #[test]
    fn recovery_needs_one_good_copy_per_block_not_one_good_replica() {
        // Four blocks. Copy 0 loses blocks 1 and 3; copy 1 loses 0 and 2.
        // Neither replica is wholly intact, yet every block survives — the
        // property per-block selection buys over per-region selection.
        let clean: [[u8; BLOCK]; 4] = [block(1), block(2), block(3), block(4)];
        let crcs: [u32; 4] = core::array::from_fn(|i| crc32(&clean[i]));

        let mut c0 = clean;
        let mut c1 = clean;
        c0[1][0] ^= 0xFF;
        c0[3][0] ^= 0xFF;
        c1[0][0] ^= 0xFF;
        c1[2][0] ^= 0xFF;

        let c0r: [&[u8]; 4] = core::array::from_fn(|i| &c0[i][..]);
        let c1r: [&[u8]; 4] = core::array::from_fn(|i| &c1[i][..]);
        assert!(all_blocks_recoverable(&[&c0r[..], &c1r[..]], &crcs));

        // Per-region selection would have to reject both replicas.
        assert!(!c0.iter().zip(&crcs).all(|(b, &c)| verify(b, c)));
        assert!(!c1.iter().zip(&crcs).all(|(b, &c)| verify(b, c)));
    }

    #[test]
    fn losing_the_same_block_in_every_copy_is_unrecoverable() {
        let clean: [[u8; BLOCK]; 4] = [block(1), block(2), block(3), block(4)];
        let crcs: [u32; 4] = core::array::from_fn(|i| crc32(&clean[i]));

        let mut c0 = clean;
        let mut c1 = clean;
        c0[2][0] ^= 0xFF;
        c1[2][7] ^= 0xFF;

        let c0r: [&[u8]; 4] = core::array::from_fn(|i| &c0[i][..]);
        let c1r: [&[u8]; 4] = core::array::from_fn(|i| &c1[i][..]);
        assert!(!all_blocks_recoverable(&[&c0r[..], &c1r[..]], &crcs));
    }

    #[test]
    fn three_copies_survive_two_failures_of_one_block() {
        let good = block(0x5A);
        let crc = crc32(&good);
        let mut a = good;
        let mut b = good;
        a[0] ^= 1;
        b[1] ^= 1;
        assert_eq!(
            select_copy(&[&a[..], &b[..], &good[..]], crc),
            Selection::Replica { index: 2 }
        );
    }

    #[test]
    fn replica_cost_is_one_full_copy_at_the_default() {
        assert_eq!(replica_bytes(32 * 1024, 2), 32 * 1024);
        assert_eq!(replica_bytes(32 * 1024, 3), 64 * 1024);
    }
}
