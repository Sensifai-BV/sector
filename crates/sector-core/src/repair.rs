//! Located-erasure repair.
//!
//! Repair runs only on damage the CRC has localised. Mechanism differs by
//! region and by scale: codebook blocks repair from a replica at T0/T1, or from
//! RS parity at larger scale; payload and rerank blocks are dropped rather than
//! repaired.
//!
//! # Policy
//!
//! Repair in place when the target sector can be erased and rewritten. Prefer
//! not repairing to repairing speculatively: a rewrite costs an erase cycle
//! against a ~10^5-cycle endurance budget.
//!
//! An unrepairable codebook region is a mount-time refusal. An unrepairable
//! payload or rerank region degrades gracefully. The asymmetry follows the
//! fan-out: a lost codebook block perturbs `n_{j,c}` vectors, a lost 512 B
//! payload block loses the 32 it contained.

use crate::error::Error;
use sector_codec::crc::verify;
use sector_codec::replicate::Selection;
use sector_format::region::{Protection, RegionKind};
use sector_hal::NorFlash;

/// What a repair attempt did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The block verified; nothing was written.
    ///
    /// Distinct from a successful repair because a rewrite costs an erase
    /// cycle against a ~10^5-cycle endurance budget, and the count of erases
    /// spent is what the lifetime model is checked against.
    Clean,
    /// Repaired from copy `index` and rewritten.
    Repaired {
        /// Which copy supplied the bytes.
        index: usize,
    },
    /// Damage was localised but no source could supply the bytes.
    Unrepairable,
    /// The region's policy is detect-only; the block is dropped.
    Dropped,
}

/// What a repair pass spent, in blocks touched and erase cycles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RepairStats {
    /// Blocks examined.
    pub examined: u32,
    /// Blocks that verified with no action.
    pub clean: u32,
    /// Blocks rewritten from a replica.
    pub repaired: u32,
    /// Blocks dropped, either by policy or because no source was good.
    pub dropped: u32,
    /// Erase cycles spent. The quantity the lifetime model predicts.
    pub erases: u32,
}

/// Where a block lives and how it is protected.
///
/// The primary address, its replicas and the policy are one fact about one
/// block; splitting them across a parameter list invites a call site that
/// repairs the right address under the wrong policy.
#[derive(Clone, Copy, Debug)]
pub struct Target<'a> {
    /// What the region holds.
    pub kind: RegionKind,
    /// How it is repaired.
    pub protection: Protection,
    /// Byte address of the block in the primary copy.
    pub primary_addr: u32,
    /// Byte addresses of the same block in each replica.
    pub replica_addrs: &'a [u32],
    /// CRC the block must match.
    pub expected_crc: u32,
}

/// Repair one block of a replicated region.
///
/// Repairs only damage the CRC has localised, and prefers not repairing to
/// repairing speculatively.
pub fn repair_replicated<F: NorFlash>(
    flash: &mut F,
    target: Target<'_>,
    scratch: &mut [u8],
    stats: &mut RepairStats,
) -> Result<Outcome, Error> {
    stats.examined += 1;
    let primary_addr = target.primary_addr;

    flash
        .read(primary_addr, scratch)
        .map_err(|_| Error::Read { addr: primary_addr })?;
    if verify(scratch, target.expected_crc) {
        stats.clean += 1;
        return Ok(Outcome::Clean);
    }

    // Detect-only regions degrade rather than repair. Parity here would spend
    // the budget protecting the least critical bytes.
    if matches!(target.protection, Protection::Detect) {
        stats.dropped += 1;
        return Ok(Outcome::Dropped);
    }

    for (i, &addr) in target.replica_addrs.iter().enumerate() {
        flash
            .read(addr, scratch)
            .map_err(|_| Error::Read { addr })?;
        if !verify(scratch, target.expected_crc) {
            continue;
        }
        // A good copy exists: erase the primary's sector and rewrite.
        let sector = flash.sector_size() as u32;
        let sector_base = (primary_addr / sector) * sector;
        flash
            .erase(sector_base)
            .map_err(|_| Error::Erase { addr: sector_base })?;
        stats.erases += 1;
        flash
            .program(primary_addr, scratch)
            .map_err(|_| Error::Program { addr: primary_addr })?;
        stats.repaired += 1;
        return Ok(Outcome::Repaired { index: i + 1 });
    }

    stats.dropped += 1;
    Ok(Outcome::Unrepairable)
}

/// Whether an unrepairable region prevents the volume from serving queries.
///
/// The asymmetry follows the fan-out, not a policy preference: a lost codebook
/// block perturbs the reconstruction of every vector referencing its centroids,
/// while a lost payload block loses only the vectors it contained.
pub const fn is_fatal(kind: RegionKind) -> bool {
    matches!(kind, RegionKind::Codebook | RegionKind::CodebookReplica)
}

/// Convert a copy selection into an outcome, for callers that select
/// separately from repairing.
pub const fn outcome_of(selection: Selection) -> Outcome {
    match selection {
        Selection::Primary => Outcome::Clean,
        Selection::Replica { index } => Outcome::Repaired { index },
        Selection::Unrecoverable => Outcome::Unrepairable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_codec::crc::crc32;
    use sector_hal::ERASED_BYTE;

    const BLOCK: usize = 512;
    const SECTOR: usize = 4096;
    const IMAGE: usize = 32 * 1024;

    struct TestFlash {
        bytes: [u8; IMAGE],
        erases: usize,
    }

    impl TestFlash {
        fn new() -> Self {
            Self {
                bytes: [0u8; IMAGE],
                erases: 0,
            }
        }
        fn write(&mut self, addr: u32, data: &[u8]) {
            let start = addr as usize;
            self.bytes[start..start + data.len()].copy_from_slice(data);
        }
    }

    impl NorFlash for TestFlash {
        type Error = ();
        fn page_size(&self) -> usize {
            256
        }
        fn sector_size(&self) -> usize {
            SECTOR
        }
        fn capacity(&self) -> u32 {
            IMAGE as u32
        }
        fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), ()> {
            let start = addr as usize;
            buf.copy_from_slice(self.bytes.get(start..start + buf.len()).ok_or(())?);
            Ok(())
        }
        fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), ()> {
            let start = addr as usize;
            let dst = self.bytes.get_mut(start..start + buf.len()).ok_or(())?;
            dst.copy_from_slice(buf);
            Ok(())
        }
        fn erase(&mut self, sector_addr: u32) -> Result<(), ()> {
            self.erases += 1;
            let start = sector_addr as usize;
            self.bytes
                .get_mut(start..start + SECTOR)
                .ok_or(())?
                .fill(ERASED_BYTE);
            Ok(())
        }
    }

    fn block(fill: u8) -> [u8; BLOCK] {
        [fill; BLOCK]
    }

    #[test]
    fn a_clean_block_costs_no_erase_cycle() {
        // Preferring not to repair is the policy, and the erase count is how it
        // is checked: endurance is ~10^5 cycles.
        let mut f = TestFlash::new();
        let good = block(0xA5);
        let crc = crc32(&good);
        f.write(0, &good);
        f.write(SECTOR as u32, &good);

        let mut scratch = [0u8; BLOCK];
        let mut stats = RepairStats::default();
        let out = repair_replicated(
            &mut f,
            Target {
                kind: RegionKind::Codebook,
                protection: Protection::Replicate,
                primary_addr: 0,
                replica_addrs: &[SECTOR as u32],
                expected_crc: crc,
            },
            &mut scratch,
            &mut stats,
        )
        .unwrap();

        assert_eq!(out, Outcome::Clean);
        assert_eq!(stats.erases, 0);
        assert_eq!(f.erases, 0);
        assert_eq!(stats.clean, 1);
    }

    #[test]
    fn a_damaged_primary_is_rewritten_from_the_replica() {
        let mut f = TestFlash::new();
        let good = block(0xA5);
        let crc = crc32(&good);
        let mut damaged = good;
        damaged[10] ^= 0xFF;
        f.write(0, &damaged);
        f.write(SECTOR as u32, &good);

        let mut scratch = [0u8; BLOCK];
        let mut stats = RepairStats::default();
        let out = repair_replicated(
            &mut f,
            Target {
                kind: RegionKind::Codebook,
                protection: Protection::Replicate,
                primary_addr: 0,
                replica_addrs: &[SECTOR as u32],
                expected_crc: crc,
            },
            &mut scratch,
            &mut stats,
        )
        .unwrap();

        assert_eq!(out, Outcome::Repaired { index: 1 });
        assert_eq!(stats.erases, 1);
        let mut check = [0u8; BLOCK];
        f.read(0, &mut check).unwrap();
        assert_eq!(check, good, "the primary now holds the good bytes");
    }

    #[test]
    fn a_detect_only_region_drops_rather_than_repairing() {
        // The rerank region carries no parity: it is the largest object in the
        // system and the least critical per byte.
        let mut f = TestFlash::new();
        let good = block(0x5A);
        let crc = crc32(&good);
        let mut damaged = good;
        damaged[0] ^= 1;
        f.write(0, &damaged);
        f.write(SECTOR as u32, &good);

        let mut scratch = [0u8; BLOCK];
        let mut stats = RepairStats::default();
        let out = repair_replicated(
            &mut f,
            Target {
                kind: RegionKind::Rerank,
                protection: Protection::Detect,
                primary_addr: 0,
                replica_addrs: &[SECTOR as u32],
                expected_crc: crc,
            },
            &mut scratch,
            &mut stats,
        )
        .unwrap();

        assert_eq!(out, Outcome::Dropped);
        assert_eq!(stats.erases, 0, "a drop must not spend an erase cycle");
    }

    #[test]
    fn damage_in_every_copy_is_unrepairable_not_a_guess() {
        let mut f = TestFlash::new();
        let good = block(0xA5);
        let crc = crc32(&good);
        let mut a = good;
        let mut b = good;
        a[0] ^= 1;
        b[1] ^= 2;
        f.write(0, &a);
        f.write(SECTOR as u32, &b);

        let mut scratch = [0u8; BLOCK];
        let mut stats = RepairStats::default();
        let out = repair_replicated(
            &mut f,
            Target {
                kind: RegionKind::Codebook,
                protection: Protection::Replicate,
                primary_addr: 0,
                replica_addrs: &[SECTOR as u32],
                expected_crc: crc,
            },
            &mut scratch,
            &mut stats,
        )
        .unwrap();

        assert_eq!(out, Outcome::Unrepairable);
        assert_eq!(stats.erases, 0);
        assert!(is_fatal(RegionKind::Codebook));
    }

    #[test]
    fn a_second_replica_is_consulted_when_the_first_is_bad() {
        let mut f = TestFlash::new();
        let good = block(0xC3);
        let crc = crc32(&good);
        let mut damaged = good;
        damaged[5] ^= 0xFF;
        f.write(0, &damaged);
        f.write(SECTOR as u32, &damaged);
        f.write(2 * SECTOR as u32, &good);

        let mut scratch = [0u8; BLOCK];
        let mut stats = RepairStats::default();
        let out = repair_replicated(
            &mut f,
            Target {
                kind: RegionKind::Codebook,
                protection: Protection::Replicate,
                primary_addr: 0,
                replica_addrs: &[SECTOR as u32, 2 * SECTOR as u32],
                expected_crc: crc,
            },
            &mut scratch,
            &mut stats,
        )
        .unwrap();
        assert_eq!(out, Outcome::Repaired { index: 2 });
    }

    #[test]
    fn severity_follows_the_fan_out() {
        assert!(is_fatal(RegionKind::Codebook));
        assert!(is_fatal(RegionKind::CodebookReplica));
        assert!(!is_fatal(RegionKind::Payload));
        assert!(!is_fatal(RegionKind::Rerank));
    }

    #[test]
    fn selection_maps_onto_outcome() {
        assert_eq!(outcome_of(Selection::Primary), Outcome::Clean);
        assert_eq!(
            outcome_of(Selection::Replica { index: 2 }),
            Outcome::Repaired { index: 2 }
        );
        assert_eq!(outcome_of(Selection::Unrecoverable), Outcome::Unrepairable);
    }
}
