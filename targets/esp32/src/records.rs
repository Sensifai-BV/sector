//! Record formats for the P1 measurement campaign.
//!
//! The formats live here rather than in the measurement binaries because the
//! host-side analyser consumes them and must be testable before hardware
//! exists. The binaries are `no_main` and cannot host-link; these types can.

/// One query's measurement, as written to `measurements/`.
///
/// Cycles *and* bytes. Energy on this class of device tracks flash traffic and
/// wake time more closely than instruction count, so a cycle count alone cannot
/// validate a joules-per-query claim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryRecord {
    /// Query index.
    pub query: u32,
    /// Cycles per phase, in `Phase` order: Rotate, Table, Scan, Rerank,
    /// Finalize.
    pub cycles: [u32; 5],
    /// Bytes read per phase.
    pub bytes: [u32; 5],
    /// Candidates dropped on a CRC mismatch.
    ///
    /// A drop is indistinguishable from an eviction in the result, so without
    /// this counter a recall regression is untraceable.
    pub drops: u32,
}

/// Index of each phase in the per-phase arrays.
pub const ROTATE: usize = 0;
/// Table construction.
pub const TABLE: usize = 1;
/// Payload scan.
pub const SCAN: usize = 2;
/// Rerank.
pub const RERANK: usize = 3;
/// Drain.
pub const FINALIZE: usize = 4;

impl QueryRecord {
    /// Total cycles across all phases.
    pub const fn total_cycles(&self) -> u32 {
        let mut sum = 0u32;
        let mut i = 0usize;
        while i < 5 {
            sum = sum.saturating_add(self.cycles[i]);
            i += 1;
        }
        sum
    }

    /// Microseconds a cycle count represents at `hz`.
    pub const fn micros(cycles: u32, hz: u32) -> u32 {
        if hz == 0 {
            return 0;
        }
        ((cycles as u64 * 1_000_000) / hz as u64) as u32
    }

    /// Rerank latency in microseconds at `hz`.
    ///
    /// The figure the campaign exists to test: the T0 estimate is 1.92 ms at
    /// `R = 100`, stated as refutable.
    pub const fn rerank_micros(&self, hz: u32) -> u32 {
        Self::micros(self.cycles[RERANK], hz)
    }

    /// Whether one phase dominates, in per-mille of total cycles.
    ///
    /// The cost model's tier choice rests on `Table` and `Rerank` being the
    /// large terms. If the scan dominates instead, the model is wrong about
    /// where the energy goes.
    pub const fn share_permille(&self, phase: usize) -> u32 {
        let total = self.total_cycles();
        if total == 0 || phase >= 5 {
            return 0;
        }
        ((self.cycles[phase] as u64 * 1000) / total as u64) as u32
    }
}

/// Timings for one flash operation class, in microseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlashTiming {
    /// Fastest observed.
    pub min_us: u32,
    /// Median.
    pub median_us: u32,
    /// Slowest observed.
    ///
    /// A power budget must cover the worst case, and NOR erase time varies with
    /// wear, so the maximum is reported rather than the median alone.
    pub max_us: u32,
    /// Operations timed.
    pub samples: u32,
}

impl FlashTiming {
    /// Whether the spread is wide enough that a median alone misleads.
    pub const fn spread_is_material(&self) -> bool {
        self.median_us > 0 && self.max_us >= self.median_us * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0_record() -> QueryRecord {
        // Cycle counts in the shape the cost model predicts at T0: table
        // construction and rerank dominate, scan is comparatively cheap.
        QueryRecord {
            query: 0,
            cycles: [1_200, 98_000, 41_000, 307_000, 800],
            bytes: [0, 0, 143_456, 12_800, 0],
            drops: 2,
        }
    }

    #[test]
    fn rerank_latency_converts_against_the_t0_estimate() {
        // 307,000 cycles at 160 MHz is 1.92 ms — the estimate under test. The
        // arithmetic is checked here so a disagreement on hardware is a
        // disagreement about the device, not about the conversion.
        let r = t0_record();
        assert_eq!(r.rerank_micros(160_000_000), 1_918);
        assert_eq!(QueryRecord::micros(0, 160_000_000), 0);
        assert_eq!(QueryRecord::micros(1_000, 0), 0);
    }

    #[test]
    fn phase_shares_show_where_the_energy_goes() {
        // The tier choice rests on Table and Rerank being the large terms.
        let r = t0_record();
        assert_eq!(r.total_cycles(), 448_000);
        assert_eq!(r.share_permille(RERANK), 685);
        assert_eq!(r.share_permille(TABLE), 218);
        assert_eq!(r.share_permille(SCAN), 91);
        assert!(
            r.share_permille(RERANK) + r.share_permille(TABLE) > 500,
            "the model expects these two to dominate"
        );
        assert_eq!(r.share_permille(9), 0);
    }

    #[test]
    fn an_empty_record_reports_zero_rather_than_dividing_by_zero() {
        let r = QueryRecord::default();
        assert_eq!(r.total_cycles(), 0);
        assert_eq!(r.share_permille(SCAN), 0);
        assert_eq!(r.rerank_micros(160_000_000), 0);
    }

    #[test]
    fn total_cycles_saturates_rather_than_wrapping() {
        // A wrapping total restarting mid-campaign gives a plausible wrong
        // measurement; a saturated one is visibly wrong.
        let r = QueryRecord {
            cycles: [u32::MAX, u32::MAX, 0, 0, 0],
            ..QueryRecord::default()
        };
        assert_eq!(r.total_cycles(), u32::MAX);
    }

    #[test]
    fn a_wide_timing_spread_is_flagged() {
        // A budget built on the median under-provisions the worst case when
        // erase time varies with wear.
        let tight = FlashTiming {
            min_us: 40,
            median_us: 45,
            max_us: 60,
            samples: 100,
        };
        let wide = FlashTiming {
            min_us: 20_000,
            median_us: 45_000,
            max_us: 120_000,
            samples: 100,
        };
        assert!(!tight.spread_is_material());
        assert!(wide.spread_is_material());
        assert!(!FlashTiming::default().spread_is_material());
    }
}
