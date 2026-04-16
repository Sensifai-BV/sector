//! Background scrub: walk blocks, verify CRCs, repair what is repairable.
//!
//! Bit rot accumulates whether or not anything reads the affected block. Scrub
//! converts latent corruption into detected corruption while a replica or
//! parity is still intact to repair from. The failure it prevents is two copies
//! rotting before either is noticed.
//!
//! # Scheduling
//!
//! Prioritise by criticality, not address order. The codebook is 32 KiB of a
//! multi-megabyte volume (D=128, b=8, int8) and carries the whole
//! shared-failure exposure, so scrubbing it more often than the rerank copy
//! costs little and is where the benefit sits.
//!
//! Scrub is interruptible and incremental with its cursor in the workspace. It
//! must not extend query latency in a way the power budget cannot predict, and
//! there is no scheduler to preempt it.
//!
//! # Scope
//!
//! Scrub addresses the flash-wear channel. At MCU scale, with ~10^5-cycle
//! endurance and small volumes, software defects are plausibly a likelier
//! corruption source; scrub is not evidence the store is correct.

use crate::error::Error;
use sector_codec::crc::verify;
use sector_format::region::RegionKind;
use sector_hal::NorFlash;

/// How much of a region a single scrub call may examine.
///
/// Scrub is interruptible and incremental because it must not extend query
/// latency in a way the power budget cannot predict, and there is no scheduler
/// to preempt it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Blocks this call may verify before yielding.
    pub blocks: u32,
}

/// Where a scrub pass left off.
///
/// Held in the workspace so the cursor survives between calls without
/// allocation, and so an interrupted scrub resumes rather than restarting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    /// Region being walked.
    pub region: u8,
    /// Next block within it.
    pub block: u32,
}

/// What a scrub pass found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScrubStats {
    /// Blocks verified.
    pub verified: u32,
    /// Blocks that failed their CRC.
    pub failed: u32,
    /// Whether the region was walked to its end.
    pub completed: bool,
}

/// Scrub priority for a region.
///
/// The codebook is scrubbed most often. It is 32 KiB of a multi-megabyte volume
/// at the T0 profile (D=128, b=8, int8) and carries the whole shared-failure
/// exposure, so visiting it more often than the rerank copy costs almost
/// nothing and is where the benefit sits.
///
/// Lower is more urgent.
pub const fn priority(kind: RegionKind) -> u8 {
    match kind {
        RegionKind::Codebook | RegionKind::CodebookReplica => 0,
        RegionKind::Payload | RegionKind::PayloadCrc => 1,
        RegionKind::Rerank | RegionKind::RerankCrc => 2,
    }
}

/// Order regions by scrub priority, most urgent first.
///
/// Ties keep their input order, so a scrub schedule is reproducible.
pub fn order_by_priority(kinds: &mut [RegionKind]) {
    for i in 1..kinds.len() {
        let mut j = i;
        while j > 0 && priority(kinds[j]) < priority(kinds[j - 1]) {
            kinds.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Geometry of the region being walked.
///
/// Base, extent, block size and the CRC array describe one region; passing
/// them separately invites a call that walks one region's blocks against
/// another's CRCs.
#[derive(Clone, Copy, Debug)]
pub struct RegionView<'a> {
    /// Byte offset of the region.
    pub base: u32,
    /// Length in blocks.
    pub blocks: u32,
    /// Bytes per block.
    pub block_bytes: usize,
    /// CRC per block, in block order.
    pub crcs: &'a [u32],
}

/// Verify up to `budget.blocks` blocks starting at `cursor`.
///
/// Returns the blocks that failed, up to `failed_out.len()`. Advances `cursor`
/// so the next call resumes where this one stopped.
pub fn scrub_region<F: NorFlash>(
    flash: &mut F,
    region: RegionView<'_>,
    cursor: &mut Cursor,
    budget: Budget,
    scratch: &mut [u8],
    failed_out: &mut [u32],
) -> Result<ScrubStats, Error> {
    let mut stats = ScrubStats::default();
    let supplied = scratch.len();
    let scratch = match scratch.get_mut(..region.block_bytes) {
        Some(s) => s,
        None => {
            return Err(Error::OutputTooSmall {
                found: supplied,
                expected: region.block_bytes,
            })
        }
    };

    let mut written = 0usize;
    let mut done = 0u32;
    while done < budget.blocks && cursor.block < region.blocks {
        let b = cursor.block;
        let addr = region.base + b * region.block_bytes as u32;
        flash
            .read(addr, scratch)
            .map_err(|_| Error::Read { addr })?;

        let expected = *region.crcs.get(b as usize).ok_or(Error::OutOfRange {
            kind: RegionKind::Payload,
            offset: b,
        })?;
        stats.verified += 1;
        if !verify(scratch, expected) {
            stats.failed += 1;
            if let Some(slot) = failed_out.get_mut(written) {
                *slot = b;
                written += 1;
            }
        }
        cursor.block += 1;
        done += 1;
    }

    if cursor.block >= region.blocks {
        stats.completed = true;
        cursor.block = 0;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_codec::crc::crc32;
    use sector_hal::ERASED_BYTE;

    const BLOCK: usize = 512;
    const BLOCKS: u32 = 8;
    const IMAGE: usize = 8 * 1024;

    struct TestFlash {
        bytes: [u8; IMAGE],
        reads: usize,
    }

    impl NorFlash for TestFlash {
        type Error = ();
        fn page_size(&self) -> usize {
            256
        }
        fn sector_size(&self) -> usize {
            4096
        }
        fn capacity(&self) -> u32 {
            IMAGE as u32
        }
        fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), ()> {
            self.reads += 1;
            let start = addr as usize;
            buf.copy_from_slice(self.bytes.get(start..start + buf.len()).ok_or(())?);
            Ok(())
        }
        fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), ()> {
            let start = addr as usize;
            self.bytes
                .get_mut(start..start + buf.len())
                .ok_or(())?
                .copy_from_slice(buf);
            Ok(())
        }
        fn erase(&mut self, sector_addr: u32) -> Result<(), ()> {
            let start = sector_addr as usize;
            self.bytes
                .get_mut(start..start + 4096)
                .ok_or(())?
                .fill(ERASED_BYTE);
            Ok(())
        }
    }

    fn populated() -> (TestFlash, [u32; BLOCKS as usize]) {
        let mut f = TestFlash {
            bytes: [0u8; IMAGE],
            reads: 0,
        };
        let mut crcs = [0u32; BLOCKS as usize];
        for b in 0..BLOCKS {
            let data: [u8; BLOCK] = core::array::from_fn(|i| ((b as usize * 31 + i) % 251) as u8);
            f.program(b * BLOCK as u32, &data).unwrap();
            crcs[b as usize] = crc32(&data);
        }
        (f, crcs)
    }

    #[test]
    fn a_clean_region_reports_no_failures() {
        let (mut f, crcs) = populated();
        let mut cursor = Cursor::default();
        let mut scratch = [0u8; BLOCK];
        let mut failed = [0u32; 8];
        let stats = scrub_region(
            &mut f,
            RegionView {
                base: 0,
                blocks: BLOCKS,
                block_bytes: BLOCK,
                crcs: &crcs,
            },
            &mut cursor,
            Budget { blocks: BLOCKS },
            &mut scratch,
            &mut failed,
        )
        .unwrap();

        assert_eq!(stats.verified, BLOCKS);
        assert_eq!(stats.failed, 0);
        assert!(stats.completed);
    }

    #[test]
    fn latent_corruption_is_found_before_anything_reads_it() {
        // The failure scrub prevents: two copies rotting before either is
        // noticed. Detection is what makes repair possible while a replica is
        // still intact.
        let (mut f, crcs) = populated();
        f.bytes[3 * BLOCK + 17] ^= 0xFF;

        let mut cursor = Cursor::default();
        let mut scratch = [0u8; BLOCK];
        let mut failed = [0u32; 8];
        let stats = scrub_region(
            &mut f,
            RegionView {
                base: 0,
                blocks: BLOCKS,
                block_bytes: BLOCK,
                crcs: &crcs,
            },
            &mut cursor,
            Budget { blocks: BLOCKS },
            &mut scratch,
            &mut failed,
        )
        .unwrap();

        assert_eq!(stats.failed, 1);
        assert_eq!(failed[0], 3);
    }

    #[test]
    fn scrub_yields_at_its_budget_and_resumes_where_it_stopped() {
        // Scrub must not extend query latency unpredictably, so it is bounded
        // per call and its cursor survives between calls.
        let (mut f, crcs) = populated();
        let mut cursor = Cursor::default();
        let mut scratch = [0u8; BLOCK];
        let mut failed = [0u32; 8];

        let first = scrub_region(
            &mut f,
            RegionView {
                base: 0,
                blocks: BLOCKS,
                block_bytes: BLOCK,
                crcs: &crcs,
            },
            &mut cursor,
            Budget { blocks: 3 },
            &mut scratch,
            &mut failed,
        )
        .unwrap();
        assert_eq!(first.verified, 3);
        assert!(!first.completed);
        assert_eq!(cursor.block, 3);

        let second = scrub_region(
            &mut f,
            RegionView {
                base: 0,
                blocks: BLOCKS,
                block_bytes: BLOCK,
                crcs: &crcs,
            },
            &mut cursor,
            Budget { blocks: 3 },
            &mut scratch,
            &mut failed,
        )
        .unwrap();
        assert_eq!(second.verified, 3);
        assert_eq!(cursor.block, 6);

        let third = scrub_region(
            &mut f,
            RegionView {
                base: 0,
                blocks: BLOCKS,
                block_bytes: BLOCK,
                crcs: &crcs,
            },
            &mut cursor,
            Budget { blocks: 3 },
            &mut scratch,
            &mut failed,
        )
        .unwrap();
        // Only two blocks remained; the walk completed and the cursor wrapped.
        assert_eq!(third.verified, 2);
        assert!(third.completed);
        assert_eq!(cursor.block, 0);
    }

    #[test]
    fn a_bounded_pass_reads_exactly_its_budget() {
        // Bounded work per call is the property; an unbounded read count would
        // make the latency unpredictable regardless of what it verified.
        let (mut f, crcs) = populated();
        let mut cursor = Cursor::default();
        let mut scratch = [0u8; BLOCK];
        let mut failed = [0u32; 8];
        scrub_region(
            &mut f,
            RegionView {
                base: 0,
                blocks: BLOCKS,
                block_bytes: BLOCK,
                crcs: &crcs,
            },
            &mut cursor,
            Budget { blocks: 2 },
            &mut scratch,
            &mut failed,
        )
        .unwrap();
        assert_eq!(f.reads, 2);
    }

    #[test]
    fn the_codebook_is_scrubbed_before_the_rerank_copy() {
        let mut kinds = [
            RegionKind::Rerank,
            RegionKind::Payload,
            RegionKind::Codebook,
            RegionKind::RerankCrc,
            RegionKind::CodebookReplica,
            RegionKind::PayloadCrc,
        ];
        order_by_priority(&mut kinds);
        assert_eq!(kinds[0], RegionKind::Codebook);
        assert_eq!(kinds[1], RegionKind::CodebookReplica);
        assert_eq!(kinds[5], RegionKind::RerankCrc);
        // Priorities are ascending after ordering.
        for w in kinds.windows(2) {
            assert!(priority(w[0]) <= priority(w[1]));
        }
    }

    #[test]
    fn every_failing_block_is_reported_when_room_allows() {
        let (mut f, crcs) = populated();
        for b in [1usize, 4, 6] {
            f.bytes[b * BLOCK] ^= 0xFF;
        }
        let mut cursor = Cursor::default();
        let mut scratch = [0u8; BLOCK];
        let mut failed = [0u32; 8];
        let stats = scrub_region(
            &mut f,
            RegionView {
                base: 0,
                blocks: BLOCKS,
                block_bytes: BLOCK,
                crcs: &crcs,
            },
            &mut cursor,
            Budget { blocks: BLOCKS },
            &mut scratch,
            &mut failed,
        )
        .unwrap();
        assert_eq!(stats.failed, 3);
        assert_eq!(&failed[..3], &[1, 4, 6]);
    }
}
