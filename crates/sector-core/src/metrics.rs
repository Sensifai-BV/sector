//! Per-phase counters, behind the `metrics` feature.
//!
//! Off by default, so the bare-metal build pays nothing. When enabled: cycles
//! and bytes moved per phase, candidate drops, intruder counts, heap
//! insertions — the quantities the energy model is stated in.
//!
//! # Counter rules
//!
//! Count bytes moved, not only time elapsed. Energy on this class of device
//! tracks flash traffic and wake time more closely than instruction count, so a
//! cycle counter alone cannot validate a joules-per-query claim.
//!
//! Counters are `u32` and saturating. A wrapping counter that restarts
//! mid-campaign produces a plausible wrong measurement; a saturated one is
//! visibly wrong.
//!
//! Counters are emitted at the same phase boundaries the instrument marks, so
//! host-side timing and on-device counters cross-check against each other.

use sector_hal::Phase;

/// Counters for one phase.
///
/// `u32` and saturating. A wrapping counter that restarts mid-campaign produces
/// a plausible wrong measurement; a saturated one is visibly wrong.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseCounters {
    /// Cycles spent in the phase.
    pub cycles: u32,
    /// Bytes moved.
    ///
    /// Counted separately from time because energy on this class of device
    /// tracks flash traffic and wake time more closely than instruction count,
    /// so a cycle counter alone cannot validate a joules-per-query claim.
    pub bytes: u32,
    /// Times the phase was entered.
    pub entries: u32,
}

impl PhaseCounters {
    /// Add `cycles`, saturating.
    pub fn add_cycles(&mut self, cycles: u32) {
        self.cycles = self.cycles.saturating_add(cycles);
    }
    /// Add `bytes`, saturating.
    pub fn add_bytes(&mut self, bytes: u32) {
        self.bytes = self.bytes.saturating_add(bytes);
    }
    /// Record a phase entry, saturating.
    pub fn enter(&mut self) {
        self.entries = self.entries.saturating_add(1);
    }
    /// Whether any counter has saturated and the record is no longer usable.
    pub const fn saturated(&self) -> bool {
        self.cycles == u32::MAX || self.bytes == u32::MAX || self.entries == u32::MAX
    }
}

/// Per-phase counters plus the query-level quantities the energy model uses.
///
/// Counters are emitted at the same boundaries the instrument marks, so
/// host-side timing and on-device counters cross-check against each other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    /// Rotate, Table, Scan, Rerank, Finalize, indexed by [`phase_index`].
    pub phases: [PhaseCounters; 5],
    /// Vectors examined in stage one.
    pub scanned: u32,
    /// Heap insertions attempted.
    pub heap_insertions: u32,
    /// Incumbents displaced.
    pub evictions: u32,
    /// Candidates dropped on a CRC mismatch.
    pub drops: u32,
}

/// Index of `phase` in [`Metrics::phases`].
pub const fn phase_index(phase: Phase) -> usize {
    match phase {
        Phase::Rotate => 0,
        Phase::Table => 1,
        Phase::Scan => 2,
        Phase::Rerank => 3,
        Phase::Finalize => 4,
    }
}

impl Metrics {
    /// Counters for `phase`.
    pub fn phase(&self, phase: Phase) -> &PhaseCounters {
        const ZERO: PhaseCounters = PhaseCounters {
            cycles: 0,
            bytes: 0,
            entries: 0,
        };
        self.phases.get(phase_index(phase)).unwrap_or(&ZERO)
    }

    /// Mutable counters for `phase`.
    pub fn phase_mut(&mut self, phase: Phase) -> Option<&mut PhaseCounters> {
        self.phases.get_mut(phase_index(phase))
    }

    /// Total cycles across all phases, saturating.
    pub fn total_cycles(&self) -> u32 {
        self.phases
            .iter()
            .fold(0u32, |acc, p| acc.saturating_add(p.cycles))
    }

    /// Total bytes moved, saturating.
    pub fn total_bytes(&self) -> u32 {
        self.phases
            .iter()
            .fold(0u32, |acc, p| acc.saturating_add(p.bytes))
    }

    /// Whether any counter has saturated.
    pub fn saturated(&self) -> bool {
        self.phases.iter().any(|p| p.saturated())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_phase_has_its_own_slot() {
        let mut m = Metrics::default();
        for (i, phase) in [
            Phase::Rotate,
            Phase::Table,
            Phase::Scan,
            Phase::Rerank,
            Phase::Finalize,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(phase_index(phase), i);
            if let Some(c) = m.phase_mut(phase) {
                c.add_cycles(100 * (i as u32 + 1));
                c.enter();
            }
        }
        // Table and Rerank are the terms that decide the tier configuration, so
        // they must be separable from the total.
        assert_eq!(m.phase(Phase::Table).cycles, 200);
        assert_eq!(m.phase(Phase::Rerank).cycles, 400);
        assert_eq!(m.total_cycles(), 100 + 200 + 300 + 400 + 500);
    }

    #[test]
    fn counters_saturate_rather_than_wrap() {
        let mut c = PhaseCounters::default();
        c.add_cycles(u32::MAX - 1);
        assert!(!c.saturated());
        c.add_cycles(10);
        assert_eq!(c.cycles, u32::MAX);
        assert!(c.saturated(), "a saturated counter must be visibly wrong");
    }

    #[test]
    fn bytes_are_counted_separately_from_cycles() {
        // Energy tracks flash traffic; a cycle count alone cannot validate a
        // joules-per-query claim.
        let mut m = Metrics::default();
        if let Some(c) = m.phase_mut(Phase::Scan) {
            c.add_cycles(1_000);
            c.add_bytes(143_456);
        }
        assert_eq!(m.total_cycles(), 1_000);
        assert_eq!(m.total_bytes(), 143_456);
    }

    #[test]
    fn a_saturated_phase_marks_the_whole_record() {
        let mut m = Metrics::default();
        if let Some(c) = m.phase_mut(Phase::Rerank) {
            c.add_bytes(u32::MAX);
        }
        assert!(m.saturated());
    }
}
