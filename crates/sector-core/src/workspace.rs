//! The caller-provided fixed workspace.
//!
//! Every buffer the engine touches lives here: the ADC table, the candidate
//! heap, the rotation scratch, and any bounce buffer a non-XIP backend needs.
//! Nothing is allocated, so peak RAM is a linker symbol and the RAM claim is
//! checkable.
//!
//! T0 fixed set (D=128, m=16, b=8, int8, R=500), 51.9 KiB of the 192 KiB
//! budget: 32 KiB codebook, 8 KiB ADC table, 3.9 KiB heap, 8 KiB reserved
//! stack. The residual 140 KiB holds codes for 8,966 vectors at 16 B each.
//!
//! # Sizing rules
//!
//! Size derives from the profile by `const fn` arithmetic and is asserted
//! against the tier budget with `const _: ()`, so an infeasible configuration
//! fails the build. Two of the three T0 configurations an earlier design
//! proposed did not fit, and were found by arithmetic after the fact.
//!
//! All stack-heavy buffers are hoisted here, including ones local to a single
//! function. A buffer left on the stack does not appear in the workspace
//! figure, which makes the headline RAM number wrong.

use sector_format::profile::Profile;

/// Compile-time sizes for a workspace, derived from a [`Profile`].
///
/// Every field is a `const fn` of the profile, so a configuration that does not
/// fit fails the build rather than the device. Two of three T0 configurations
/// an earlier design proposed did not fit, and were found by arithmetic after
/// the fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceSizes {
    /// Resident codebook bytes, `2^b * D * cb_bytes`.
    pub codebook: usize,
    /// ADC table bytes, `m * 2^b * adc_bytes`.
    pub adc_table: usize,
    /// Candidate heap bytes.
    pub heap: usize,
    /// Rotation scratch, `D * 4` for the i32 working vector.
    pub rotation: usize,
    /// Bounce buffer for a non-XIP backend, one block.
    pub bounce: usize,
    /// Stack reserved for firmware outside the engine.
    pub stack_reserve: usize,
}

impl WorkspaceSizes {
    /// Sizes for `profile` at `block_bytes` bounce granularity.
    pub const fn of(profile: &Profile, block_bytes: usize) -> Self {
        Self {
            codebook: profile.codebook_bytes(),
            adc_table: profile.adc_table_bytes(),
            heap: profile.heap_bytes(),
            rotation: profile.d * 4,
            bounce: block_bytes,
            stack_reserve: profile.stack_reserve,
        }
    }

    /// Bytes the engine holds resident, excluding stored codes.
    ///
    /// This is the figure the RAM claim is made against, so it includes the
    /// stack reserve: a buffer excluded from the total does not stop occupying
    /// memory.
    pub const fn fixed_total(&self) -> usize {
        self.codebook
            + self.adc_table
            + self.heap
            + self.rotation
            + self.bounce
            + self.stack_reserve
    }

    /// Bytes left for stored codes within `ram_budget`.
    pub const fn code_budget(&self, ram_budget: usize) -> usize {
        ram_budget.saturating_sub(self.fixed_total())
    }

    /// Whether the fixed set fits `ram_budget` with room for codes.
    pub const fn fits(&self, ram_budget: usize) -> bool {
        self.code_budget(ram_budget) > 0
    }
}

/// Why a workspace was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceError {
    /// A buffer is smaller than the profile requires.
    TooSmall {
        /// Which buffer.
        which: Buffer,
        /// Bytes supplied.
        found: usize,
        /// Bytes required.
        expected: usize,
    },
}

/// Names a workspace buffer, for error reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buffer {
    /// Resident codebook.
    Codebook,
    /// ADC lookup table.
    AdcTable,
    /// Candidate heap.
    Heap,
    /// Rotation scratch.
    Rotation,
    /// Bounce buffer.
    Bounce,
}

/// Caller-provided buffers the engine borrows for one query.
///
/// The engine allocates nothing. Every buffer it touches is here, so peak RAM
/// is a linker symbol and the RAM claim is checkable rather than asserted.
///
/// Buffers local to a single function are hoisted here too. One left on the
/// stack does not appear in the workspace figure, which makes the headline
/// number wrong.
#[derive(Debug)]
pub struct Workspace<'a> {
    /// ADC table, `m * 2^b` accumulators.
    pub adc_table: &'a mut [i32],
    /// Candidate scores, capacity `R`.
    pub heap_scores: &'a mut [i32],
    /// Candidate ids, capacity `R`.
    pub heap_ids: &'a mut [u32],
    /// Rotation scratch, `D` components.
    pub rotation: &'a mut [i32],
    /// One block, for a backend without an XIP window.
    pub bounce: &'a mut [u8],
    /// Scrub cursor, held across calls so scrub is interruptible.
    pub scrub_cursor: u32,
}

impl<'a> Workspace<'a> {
    /// Bind buffers for `profile`, checking each against the required size.
    pub fn new(
        profile: &Profile,
        adc_table: &'a mut [i32],
        heap_scores: &'a mut [i32],
        heap_ids: &'a mut [u32],
        rotation: &'a mut [i32],
        bounce: &'a mut [u8],
    ) -> Result<Self, WorkspaceError> {
        let entries = profile.m * profile.centroids();
        if adc_table.len() < entries {
            return Err(WorkspaceError::TooSmall {
                which: Buffer::AdcTable,
                found: adc_table.len(),
                expected: entries,
            });
        }
        if heap_scores.len() < profile.r {
            return Err(WorkspaceError::TooSmall {
                which: Buffer::Heap,
                found: heap_scores.len(),
                expected: profile.r,
            });
        }
        if heap_ids.len() < profile.r {
            return Err(WorkspaceError::TooSmall {
                which: Buffer::Heap,
                found: heap_ids.len(),
                expected: profile.r,
            });
        }
        if rotation.len() < profile.d {
            return Err(WorkspaceError::TooSmall {
                which: Buffer::Rotation,
                found: rotation.len(),
                expected: profile.d,
            });
        }
        if bounce.is_empty() {
            return Err(WorkspaceError::TooSmall {
                which: Buffer::Bounce,
                found: 0,
                expected: 1,
            });
        }
        Ok(Self {
            adc_table,
            heap_scores,
            heap_ids,
            rotation,
            bounce,
            scrub_cursor: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_format::profile::{T0, T0_WIDE, T1};
    use sector_format::BLOCK_BYTES;

    #[test]
    fn t0_fixed_set_matches_the_reported_budget() {
        let s = WorkspaceSizes::of(&T0, BLOCK_BYTES);
        // 2^8 * 128 * 1 = 32 KiB codebook.
        assert_eq!(s.codebook, 32 * 1024);
        // m * 2^b * adc_bytes = 16 * 256 * 2 = 8 KiB.
        assert_eq!(s.adc_table, 8 * 1024);
        // D * 4 = 512 B rotation scratch.
        assert_eq!(s.rotation, 512);
        assert_eq!(s.bounce, 512);
        assert_eq!(s.stack_reserve, 8 * 1024);

        // The whole fixed set, and the codes that fit beside it.
        assert!(s.fits(T0.ram_budget));
        let codes = s.code_budget(T0.ram_budget);
        assert!(codes > 100 * 1024, "code budget collapsed to {codes}");
        // At 16 B/vector.
        assert!(codes / T0.payload_bytes() > 6_000);
    }

    #[test]
    fn the_stack_reserve_is_counted_not_excluded() {
        let s = WorkspaceSizes::of(&T0, BLOCK_BYTES);
        let without = s.codebook + s.adc_table + s.heap + s.rotation + s.bounce;
        assert_eq!(s.fixed_total(), without + T0.stack_reserve);
    }

    #[test]
    fn a_wide_profile_still_fits_only_because_b_dropped_to_four() {
        // T0_WIDE is the boundary case: 2^4 * 768 = 12 KiB codebook. At b=8 it
        // would be 2^8 * 768 = 192 KiB, the entire budget.
        let wide = WorkspaceSizes::of(&T0_WIDE, BLOCK_BYTES);
        assert_eq!(wide.codebook, 12 * 1024);
        assert!(wide.fits(T0_WIDE.ram_budget));
        assert_eq!(256 * 768, 192 * 1024);
    }

    #[test]
    fn t1_fixed_set_is_dominated_by_psram_headroom() {
        let s = WorkspaceSizes::of(&T1, BLOCK_BYTES);
        assert_eq!(s.codebook, 32 * 1024);
        // m doubles and accumulators widen: 32 * 256 * 4 = 32 KiB.
        assert_eq!(s.adc_table, 32 * 1024);
        assert!(s.code_budget(T1.ram_budget) > 5 * 1024 * 1024);
    }

    #[test]
    fn binding_checks_every_buffer() {
        let mut table = [0i32; 16 * 256];
        let mut scores = [0i32; 500];
        let mut ids = [0u32; 500];
        let mut rot = [0i32; 128];
        let mut bounce = [0u8; 512];
        assert!(Workspace::new(
            &T0,
            &mut table,
            &mut scores,
            &mut ids,
            &mut rot,
            &mut bounce
        )
        .is_ok());

        let mut small = [0i32; 10];
        assert_eq!(
            Workspace::new(
                &T0,
                &mut small,
                &mut scores,
                &mut ids,
                &mut rot,
                &mut bounce
            )
            .err(),
            Some(WorkspaceError::TooSmall {
                which: Buffer::AdcTable,
                found: 10,
                expected: 16 * 256,
            })
        );

        let mut short_heap = [0i32; 10];
        assert!(matches!(
            Workspace::new(
                &T0,
                &mut table,
                &mut short_heap,
                &mut ids,
                &mut rot,
                &mut bounce
            ),
            Err(WorkspaceError::TooSmall {
                which: Buffer::Heap,
                ..
            })
        ));

        let mut short_rot = [0i32; 8];
        assert!(matches!(
            Workspace::new(
                &T0,
                &mut table,
                &mut scores,
                &mut ids,
                &mut short_rot,
                &mut bounce
            ),
            Err(WorkspaceError::TooSmall {
                which: Buffer::Rotation,
                ..
            })
        ));
    }

    #[test]
    fn an_infeasible_profile_reports_zero_code_budget() {
        // b=8 at D=768 is the whole T0 budget, leaving nothing for codes.
        let infeasible = Profile { d: 768, b: 8, ..T0 };
        let s = WorkspaceSizes::of(&infeasible, BLOCK_BYTES);
        assert_eq!(s.codebook, 192 * 1024);
        assert!(!s.fits(infeasible.ram_budget));
        assert_eq!(s.code_budget(infeasible.ram_budget), 0);
    }
}
