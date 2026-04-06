//! Append path for post-build insertion into erase-aligned segments.
//!
//! The volume is built offline. Appending adds vectors to erase-aligned
//! segments without rewriting the codebook, which is what keeps it tractable:
//! the codebook is shared by the whole corpus, and rewriting it invalidates
//! every stored code.
//!
//! # Constraints
//!
//! New vectors are encoded against the existing codebook, accepting the
//! quantization drift. Retraining is corpus-global and has no bounded-RAM
//! formulation; the drift is measured and reported instead.
//!
//! Append whole blocks only. The CRC array stays in step, and a torn append
//! leaves a block that fails its CRC rather than one that passes with wrong
//! contents.
//!
//! Track accumulated drift and report when it warrants a host-side rebuild.
//! On-device append is a bounded extension of a host-built index, not a
//! replacement for rebuilding it.

use crate::error::Error;
use sector_hal::{NorFlash, ERASED_BYTE};

/// Accumulated quantization drift since the last host build.
///
/// Appended vectors are encoded against the existing codebook. Retraining is
/// corpus-global and has no bounded-RAM formulation, so the drift is measured
/// and reported rather than corrected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Drift {
    /// Vectors appended since the build.
    pub appended: u32,
    /// Sum of per-vector quantization error, in scaled integer units.
    pub error_sum: u64,
    /// Vectors present at build time.
    pub built: u32,
}

impl Drift {
    /// Mean quantization error over appended vectors, scaled by 1024.
    pub const fn mean_error_x1024(&self) -> u64 {
        if self.appended == 0 {
            return 0;
        }
        (self.error_sum * 1024) / self.appended as u64
    }

    /// Appended share of the corpus, in parts per million.
    pub const fn appended_ppm(&self) -> u64 {
        let total = self.built as u64 + self.appended as u64;
        if total == 0 {
            return 0;
        }
        (self.appended as u64 * 1_000_000) / total
    }

    /// Whether a host-side rebuild is warranted.
    ///
    /// The threshold is a reporting heuristic, not a correctness boundary: an
    /// index past it still answers queries, with recall the build never
    /// measured.
    pub const fn warrants_rebuild(&self, threshold_ppm: u64) -> bool {
        self.appended_ppm() > threshold_ppm
    }
}

/// Find the append head: the first block in `region` that is fully erased.
///
/// The erased state is recognisable without a journal, which is why the append
/// path needs no separate write-ahead structure.
pub fn find_head<F: NorFlash>(
    flash: &mut F,
    base: u32,
    blocks: u32,
    block_bytes: usize,
    probe: &mut [u8],
) -> Result<Option<u32>, Error> {
    let supplied = probe.len();
    let probe = match probe.get_mut(..block_bytes) {
        Some(p) => p,
        None => {
            return Err(Error::OutputTooSmall {
                found: supplied,
                expected: block_bytes,
            })
        }
    };
    for b in 0..blocks {
        let addr = base + b * block_bytes as u32;
        flash.read(addr, probe).map_err(|_| Error::Read { addr })?;
        if probe.iter().all(|byte| *byte == ERASED_BYTE) {
            return Ok(Some(b));
        }
    }
    Ok(None)
}

/// Append one whole block of codes at `block`, with its CRC.
///
/// Whole blocks only. The CRC array stays in step, and a torn append leaves a
/// block that fails its CRC rather than one that passes with wrong contents.
pub fn append_block<F: NorFlash>(
    flash: &mut F,
    payload_base: u32,
    crc_base: u32,
    block: u32,
    block_bytes: usize,
    data: &[u8],
) -> Result<(), Error> {
    if data.len() != block_bytes {
        return Err(Error::OutputTooSmall {
            found: data.len(),
            expected: block_bytes,
        });
    }
    let addr = payload_base + block * block_bytes as u32;

    // Payload first, then the CRC that certifies it. A power loss between the
    // two leaves a block whose CRC slot is still erased, which reads as
    // absent rather than as valid.
    flash
        .program(addr, data)
        .map_err(|_| Error::Program { addr })?;

    let crc = sector_codec::crc::crc32(data);
    let crc_addr = crc_base + block * 4;
    flash
        .program(crc_addr, &crc.to_le_bytes())
        .map_err(|_| Error::Program { addr: crc_addr })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_codec::crc::verify;

    const BLOCK: usize = 512;
    const IMAGE: usize = 8 * 1024;

    struct TestFlash {
        bytes: [u8; IMAGE],
        /// Program calls to accept before failing, simulating a power loss.
        fail_after: Option<usize>,
        programs: usize,
    }

    impl TestFlash {
        fn new() -> Self {
            Self {
                bytes: [ERASED_BYTE; IMAGE],
                fail_after: None,
                programs: 0,
            }
        }
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
            let start = addr as usize;
            buf.copy_from_slice(self.bytes.get(start..start + buf.len()).ok_or(())?);
            Ok(())
        }
        fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), ()> {
            if let Some(limit) = self.fail_after {
                if self.programs >= limit {
                    return Err(());
                }
            }
            self.programs += 1;
            let start = addr as usize;
            let dst = self.bytes.get_mut(start..start + buf.len()).ok_or(())?;
            // Program-once semantics: bits only clear.
            for (d, s) in dst.iter_mut().zip(buf.iter()) {
                *d &= *s;
            }
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

    #[test]
    fn the_head_is_the_first_erased_block() {
        let mut f = TestFlash::new();
        let mut probe = [0u8; BLOCK];
        // Empty region: head is block 0.
        assert_eq!(find_head(&mut f, 0, 4, BLOCK, &mut probe).unwrap(), Some(0));

        // Write blocks 0 and 1.
        f.program(0, &[0xAA; BLOCK]).unwrap();
        f.program(BLOCK as u32, &[0xBB; BLOCK]).unwrap();
        assert_eq!(find_head(&mut f, 0, 4, BLOCK, &mut probe).unwrap(), Some(2));
    }

    #[test]
    fn a_full_region_reports_no_head() {
        let mut f = TestFlash::new();
        for b in 0..4u32 {
            f.program(b * BLOCK as u32, &[0xAA; BLOCK]).unwrap();
        }
        let mut probe = [0u8; BLOCK];
        assert_eq!(find_head(&mut f, 0, 4, BLOCK, &mut probe).unwrap(), None);
    }

    #[test]
    fn an_appended_block_verifies_against_its_crc() {
        let mut f = TestFlash::new();
        let crc_base = 4096u32;
        let data: [u8; BLOCK] = core::array::from_fn(|i| (i % 251) as u8);
        append_block(&mut f, 0, crc_base, 0, BLOCK, &data).unwrap();

        let mut read_back = [0u8; BLOCK];
        f.read(0, &mut read_back).unwrap();
        assert_eq!(read_back, data);

        let mut crc_bytes = [0u8; 4];
        f.read(crc_base, &mut crc_bytes).unwrap();
        assert!(verify(&read_back, u32::from_le_bytes(crc_bytes)));
    }

    #[test]
    fn a_torn_append_fails_its_crc_rather_than_passing_wrongly() {
        // The property that makes the append path safe without a journal: an
        // interrupted write leaves a block that is detectably bad, never one
        // that is silently wrong.
        let mut f = TestFlash::new();
        let crc_base = 4096u32;
        let data: [u8; BLOCK] = core::array::from_fn(|i| (i % 251) as u8);

        // Power loss after the payload, before the CRC.
        f.fail_after = Some(1);
        let err = append_block(&mut f, 0, crc_base, 0, BLOCK, &data);
        assert!(matches!(err, Err(Error::Program { .. })));

        // The CRC slot is still erased, so the block reads as absent.
        let mut crc_bytes = [0u8; 4];
        f.read(crc_base, &mut crc_bytes).unwrap();
        assert_eq!(crc_bytes, [ERASED_BYTE; 4]);

        let mut read_back = [0u8; BLOCK];
        f.read(0, &mut read_back).unwrap();
        assert!(
            !verify(&read_back, u32::from_le_bytes(crc_bytes)),
            "an erased CRC must not validate any block"
        );
    }

    #[test]
    fn a_partial_block_is_refused() {
        let mut f = TestFlash::new();
        let short = [0u8; 100];
        assert!(matches!(
            append_block(&mut f, 0, 4096, 0, BLOCK, &short),
            Err(Error::OutputTooSmall { .. })
        ));
    }

    #[test]
    fn drift_reports_the_appended_share() {
        let d = Drift {
            appended: 500,
            error_sum: 12_000,
            built: 9_500,
        };
        assert_eq!(d.appended_ppm(), 50_000); // 5%
        assert_eq!(d.mean_error_x1024(), 24_576);
        assert!(!d.warrants_rebuild(100_000));
        assert!(d.warrants_rebuild(10_000));
    }

    #[test]
    fn an_untouched_index_reports_no_drift() {
        let d = Drift {
            appended: 0,
            error_sum: 0,
            built: 9_000,
        };
        assert_eq!(d.appended_ppm(), 0);
        assert_eq!(d.mean_error_x1024(), 0);
        assert!(!d.warrants_rebuild(1));
    }
}
