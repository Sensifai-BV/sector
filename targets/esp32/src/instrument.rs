//! GPIO-toggling `Instrument` for the per-phase energy trace.
//!
//! # Why per-phase and not a total
//!
//! The cost model attributes query energy to Rotate, Table, Scan, Rerank and
//! Finalize separately, and `Table` and `Rerank` are the terms that decide the
//! tier configuration. A measurement reporting only a total cannot falsify the
//! model, so the five edges must be individually distinguishable on the trace.
//!
//! # Encoding
//!
//! One pin per phase would cost five GPIOs and five scope channels. Instead a
//! single pin carries a pulse train: entering phase `p` emits `p + 1` short
//! pulses, and leaving it emits one long pulse. The decoder counts pulses, so a
//! two-channel scope (shunt + marker) is enough.

use sector_hal::{Edge, Instrument, Phase};

/// Pulse widths, in the timer ticks the marker pin is driven with.
///
/// The short pulse must be resolvable at the scope's sample rate and short
/// against the phase it marks, or the marker itself perturbs the measurement it
/// is taking.
pub const SHORT_TICKS: u32 = 4;

/// Long pulse, marking a phase exit.
pub const LONG_TICKS: u32 = 20;

/// Pulses that mark entry to `phase`.
pub const fn entry_pulses(phase: Phase) -> u32 {
    match phase {
        Phase::Rotate => 1,
        Phase::Table => 2,
        Phase::Scan => 3,
        Phase::Rerank => 4,
        Phase::Finalize => 5,
    }
}

/// A marker event as the decoder sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mark {
    /// Pulse count.
    pub pulses: u32,
    /// Width of each pulse in ticks.
    pub width: u32,
}

/// Encode a phase boundary as a pulse train.
pub const fn encode(phase: Phase, edge: Edge) -> Mark {
    match edge {
        Edge::Enter => Mark {
            pulses: entry_pulses(phase),
            width: SHORT_TICKS,
        },
        Edge::Leave => Mark {
            pulses: 1,
            width: LONG_TICKS,
        },
    }
}

/// Decode a pulse train back to its phase, for the trace analyser.
pub const fn decode_entry(pulses: u32) -> Option<Phase> {
    match pulses {
        1 => Some(Phase::Rotate),
        2 => Some(Phase::Table),
        3 => Some(Phase::Scan),
        4 => Some(Phase::Rerank),
        5 => Some(Phase::Finalize),
        _ => None,
    }
}

/// Instrument driving a marker GPIO.
///
/// # Status
///
/// The encoding is here; the pin is not. Bring-up replaces `emit` with a real
/// GPIO toggle. The encoding is testable on the host meanwhile, which is where
/// an off-by-one in the pulse count would otherwise cost an oscilloscope
/// session to find.
pub struct GpioInstrument {
    /// Marks emitted, for a host-side test of the encoding.
    pub emitted: [Mark; 16],
    /// Marks recorded.
    pub len: usize,
    /// Cycle counter, filled by the device timer at bring-up.
    pub cycles: u64,
}

impl GpioInstrument {
    /// A fresh instrument.
    pub const fn new() -> Self {
        Self {
            emitted: [Mark {
                pulses: 0,
                width: 0,
            }; 16],
            len: 0,
            cycles: 0,
        }
    }

    /// Record a mark. At bring-up this also toggles the pin.
    fn emit(&mut self, mark: Mark) {
        if self.len < self.emitted.len() {
            self.emitted[self.len] = mark;
            self.len += 1;
        }
    }
}

impl Default for GpioInstrument {
    fn default() -> Self {
        Self::new()
    }
}

impl Instrument for GpioInstrument {
    fn cycles(&self) -> u64 {
        self.cycles
    }

    fn mark(&mut self, phase: Phase, edge: Edge) {
        self.emit(encode(phase, edge));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_phase_gets_a_distinct_entry_code() {
        // Two phases sharing a code would make the trace unattributable, which
        // is the one thing this instrument exists to prevent.
        let phases = [
            Phase::Rotate,
            Phase::Table,
            Phase::Scan,
            Phase::Rerank,
            Phase::Finalize,
        ];
        for (i, a) in phases.iter().enumerate() {
            for b in phases.iter().skip(i + 1) {
                assert_ne!(
                    entry_pulses(*a),
                    entry_pulses(*b),
                    "{a:?} and {b:?} collide"
                );
            }
        }
    }

    #[test]
    fn entry_codes_round_trip_through_the_decoder() {
        // The decoder is what reads the scope trace; a mismatch here would be
        // found on hardware, at the cost of a session.
        for phase in [
            Phase::Rotate,
            Phase::Table,
            Phase::Scan,
            Phase::Rerank,
            Phase::Finalize,
        ] {
            let mark = encode(phase, Edge::Enter);
            assert_eq!(decode_entry(mark.pulses), Some(phase));
            assert_eq!(mark.width, SHORT_TICKS);
        }
        assert_eq!(decode_entry(0), None);
        assert_eq!(decode_entry(6), None);
    }

    #[test]
    fn entry_and_exit_are_distinguishable_by_width() {
        // Same pin, so width is the only thing separating an exit from a
        // single-pulse entry.
        let enter = encode(Phase::Rotate, Edge::Enter);
        let leave = encode(Phase::Rotate, Edge::Leave);
        assert_eq!(enter.pulses, leave.pulses);
        assert!(leave.width > enter.width * 2, "widths too close to resolve");
    }

    #[test]
    fn a_full_query_emits_ten_marks_in_order() {
        // Five phases, entry and exit each. A missing edge makes the phase it
        // bounds unmeasurable.
        let mut inst = GpioInstrument::new();
        for phase in [
            Phase::Rotate,
            Phase::Table,
            Phase::Scan,
            Phase::Rerank,
            Phase::Finalize,
        ] {
            inst.mark(phase, Edge::Enter);
            inst.mark(phase, Edge::Leave);
        }
        assert_eq!(inst.len, 10);
        assert_eq!(inst.emitted[0], encode(Phase::Rotate, Edge::Enter));
        assert_eq!(inst.emitted[6], encode(Phase::Rerank, Edge::Enter));
        assert_eq!(inst.emitted[9], encode(Phase::Finalize, Edge::Leave));
    }
}
