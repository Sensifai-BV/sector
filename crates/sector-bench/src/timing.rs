//! Wall-clock and per-phase instrumentation for the host.
//!
//! # Why the counter's properties are measured, not assumed
//!
//! A phase shorter than the counter's resolution is unmeasurable, and a
//! measurement whose overhead is comparable to what it measures reports mostly
//! itself. Both are quantified by [`calibrate`] and reported alongside every
//! result, so a phase that turns out to be unresolvable is visible rather than
//! silently reported as a small number.

use sector_hal::{Edge, Instrument, Phase};
use std::time::Instant;

/// Number of phases in the query path.
pub const PHASES: usize = 5;

/// Index of a phase in the per-phase arrays.
pub const fn index(phase: Phase) -> usize {
    match phase {
        Phase::Rotate => 0,
        Phase::Table => 1,
        Phase::Scan => 2,
        Phase::Rerank => 3,
        Phase::Finalize => 4,
    }
}

/// Name of a phase, for reporting.
pub const fn name(i: usize) -> &'static str {
    match i {
        0 => "rotate",
        1 => "table",
        2 => "scan",
        3 => "rerank",
        4 => "finalize",
        _ => "unknown",
    }
}

/// What the timer can and cannot resolve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    /// Smallest non-zero interval the clock reports, in nanoseconds.
    pub resolution_ns: u64,
    /// Cost of one `mark` call, in nanoseconds.
    ///
    /// Subtracted from phase totals rather than assumed negligible: with ten
    /// marks per query, an overhead of even 100 ns is 1 microsecond charged to
    /// the phases it bounds.
    pub mark_overhead_ns: u64,
}

impl Calibration {
    /// Whether a phase of `ns` is resolvable.
    ///
    /// The threshold is ten ticks: below that the quantisation error is over
    /// 10% and the number is not worth reporting.
    pub const fn resolves(&self, ns: u64) -> bool {
        ns >= self.resolution_ns.saturating_mul(10)
    }
}

/// Measure clock resolution and marking overhead.
pub fn calibrate() -> Calibration {
    // Smallest observed non-zero delta between consecutive clock reads.
    let mut resolution_ns = u64::MAX;
    for _ in 0..1000 {
        let a = Instant::now();
        let mut b = Instant::now();
        while b == a {
            b = Instant::now();
        }
        let d = b.duration_since(a).as_nanos() as u64;
        if d > 0 && d < resolution_ns {
            resolution_ns = d;
        }
    }
    if resolution_ns == u64::MAX {
        resolution_ns = 1;
    }

    // Cost of a mark, amortised over enough calls to exceed the resolution.
    let mut t = HostTimer::new();
    let reps = 10_000u64;
    let start = Instant::now();
    for _ in 0..reps {
        t.mark(Phase::Scan, Edge::Enter);
    }
    let elapsed = start.elapsed().as_nanos() as u64;

    Calibration {
        resolution_ns,
        mark_overhead_ns: elapsed / reps,
    }
}

/// Per-phase timing and byte counters.
///
/// Bytes as well as time: energy on the target class tracks flash traffic and
/// wake time more closely than instruction count, so a time measurement alone
/// cannot validate a joules-per-query claim.
#[derive(Clone, Debug)]
pub struct HostTimer {
    /// Nanoseconds accumulated per phase.
    pub ns: [u64; PHASES],
    /// Bytes read per phase.
    pub bytes: [u64; PHASES],
    /// Marks emitted, for overhead subtraction.
    pub marks: u64,
    origin: Instant,
    open: [Option<Instant>; PHASES],
}

impl HostTimer {
    /// A fresh timer.
    pub fn new() -> Self {
        Self {
            ns: [0; PHASES],
            bytes: [0; PHASES],
            marks: 0,
            origin: Instant::now(),
            open: [None; PHASES],
        }
    }

    /// Clear the accumulators, keeping the origin.
    pub fn reset(&mut self) {
        self.ns = [0; PHASES];
        self.bytes = [0; PHASES];
        self.marks = 0;
        self.open = [None; PHASES];
    }

    /// Charge `n` bytes to `phase`.
    pub fn add_bytes(&mut self, phase: Phase, n: u64) {
        self.bytes[index(phase)] += n;
    }

    /// Total nanoseconds across all phases.
    pub fn total_ns(&self) -> u64 {
        self.ns.iter().sum()
    }

    /// Phase totals with marking overhead removed.
    ///
    /// Each phase is bounded by two marks, so two overheads are charged to it.
    /// Saturating, because an overhead exceeding a phase's duration means the
    /// phase was unresolvable — reporting zero is honest, a negative number is
    /// not representable, and a wrapped one would be absurd.
    pub fn corrected_ns(&self, cal: &Calibration) -> [u64; PHASES] {
        let mut out = [0u64; PHASES];
        for (i, raw) in self.ns.iter().enumerate() {
            out[i] = raw.saturating_sub(cal.mark_overhead_ns * 2);
        }
        out
    }
}

impl Default for HostTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl Instrument for HostTimer {
    fn cycles(&self) -> u64 {
        self.origin.elapsed().as_nanos() as u64
    }

    fn mark(&mut self, phase: Phase, edge: Edge) {
        self.marks += 1;
        let i = index(phase);
        match edge {
            Edge::Enter => self.open[i] = Some(Instant::now()),
            Edge::Leave => {
                if let Some(start) = self.open[i].take() {
                    self.ns[i] += start.elapsed().as_nanos() as u64;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_phase_maps_to_a_distinct_slot() {
        // Two phases sharing a slot would silently merge their costs.
        let phases = [
            Phase::Rotate,
            Phase::Table,
            Phase::Scan,
            Phase::Rerank,
            Phase::Finalize,
        ];
        let mut seen = [false; PHASES];
        for p in phases {
            let i = index(p);
            assert!(!seen[i], "{p:?} collides at slot {i}");
            seen[i] = true;
        }
        assert!(seen.iter().all(|x| *x));
    }

    #[test]
    fn a_leave_without_an_enter_is_ignored_rather_than_charged() {
        // An unbalanced mark would otherwise attribute the time since the
        // epoch to one phase.
        let mut t = HostTimer::new();
        t.mark(Phase::Scan, Edge::Leave);
        assert_eq!(t.ns[index(Phase::Scan)], 0);
    }

    #[test]
    fn phases_accumulate_independently() {
        let mut t = HostTimer::new();
        t.mark(Phase::Scan, Edge::Enter);
        std::thread::sleep(std::time::Duration::from_micros(200));
        t.mark(Phase::Scan, Edge::Leave);
        t.mark(Phase::Rerank, Edge::Enter);
        t.mark(Phase::Rerank, Edge::Leave);

        assert!(t.ns[index(Phase::Scan)] >= 150_000, "{:?}", t.ns);
        assert!(t.ns[index(Phase::Scan)] > t.ns[index(Phase::Rerank)]);
        assert_eq!(t.ns[index(Phase::Rotate)], 0);
        assert_eq!(t.marks, 4);
    }

    #[test]
    fn calibration_reports_a_usable_resolution_and_overhead() {
        // If either came back zero the corrected numbers would be meaningless,
        // so the calibration itself is checked.
        let cal = calibrate();
        assert!(cal.resolution_ns > 0, "clock reported zero resolution");
        assert!(
            cal.resolution_ns < 1_000_000,
            "resolution {} ns is too coarse to separate query phases",
            cal.resolution_ns
        );
        // A 1 ms phase must be resolvable, or nothing here is measurable.
        assert!(cal.resolves(1_000_000));
        // A phase at the resolution itself is not.
        assert!(!cal.resolves(cal.resolution_ns));
    }

    #[test]
    fn overhead_subtraction_saturates_at_zero() {
        // An unresolvable phase reports zero, not a wrapped enormous number.
        let mut t = HostTimer::new();
        t.ns[0] = 10;
        let cal = Calibration {
            resolution_ns: 1,
            mark_overhead_ns: 1_000,
        };
        assert_eq!(t.corrected_ns(&cal)[0], 0);
        t.ns[1] = 5_000;
        assert_eq!(t.corrected_ns(&cal)[1], 3_000);
    }

    #[test]
    fn bytes_are_counted_separately_from_time() {
        let mut t = HostTimer::new();
        t.add_bytes(Phase::Scan, 16_000);
        t.add_bytes(Phase::Rerank, 12_800);
        assert_eq!(t.bytes[index(Phase::Scan)], 16_000);
        assert_eq!(t.bytes[index(Phase::Rerank)], 12_800);
        assert_eq!(t.total_ns(), 0, "bytes must not imply time");
    }
}
