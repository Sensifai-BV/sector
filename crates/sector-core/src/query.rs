//! The two-stage query entry point and its phase markers.
//!
//! Stage one scans the compressed payload to `R` candidates; stage two rescores
//! them against the higher-precision copy. Stage two is mandatory: single-stage
//! PQ recall is unusable at every configuration measured.
//!
//! Phases, each marked for the instrument: rotate in place; build the ADC table
//! (`2^b · D` MACs, independent of `N`); scan into a bounded heap; CRC-verify
//! and rescore the `R` survivors; drain the top `k`.
//!
//! # Instrumentation
//!
//! Every phase boundary is marked through the `Instrument` trait even in builds
//! that do not measure; `NoInstrument` compiles to nothing. The cost model
//! attributes query energy to these phases separately, and `Table` and `Rerank`
//! are the terms that decide the tier configuration, so a total-only
//! measurement cannot falsify it.
//!
//! `Table` scales with `2^b · D` and `Scan` with `N`, so they are benchmarked
//! independently. At T0 the table build can dominate.

use crate::heap::{Candidate, Heap};
use crate::metrics::Metrics;
use crate::rerank::{self, RerankSource, RerankStats};
use crate::scan::{self, ScanStats};
use crate::workspace::Workspace;
use sector_hal::{Edge, Instrument, Phase};
use sector_quant::adc;
use sector_quant::codebook::Codebook;
use sector_quant::rotate;

/// What one query measured, beyond the results themselves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryStats {
    /// Stage one.
    pub scan: ScanStats,
    /// Stage two.
    pub rerank: RerankStats,
    /// Multiply-accumulates spent building the ADC table.
    pub table_macs: u32,
    /// Results returned. Fewer than `k` when candidates were dropped.
    pub returned: u32,
}

/// Source of payload codes for stage one.
///
/// One call per contiguous run, so an execute-in-place backend can hand back
/// the whole region and a buffered one can hand back a block at a time. The
/// engine does not know which it has.
pub trait PayloadSource {
    /// Backend error type.
    type Error;

    /// Borrow the next run of codes, with the vector id its first record holds.
    ///
    /// Returns `None` when the corpus is exhausted.
    fn next_run(&mut self) -> Result<Option<(&[u8], u32)>, Self::Error>;

    /// Reset to the start of the corpus.
    fn rewind(&mut self);
}

/// Run one query end to end.
///
/// The five phases, each marked for the instrument even when instrumentation is
/// a no-op: rotate in place, build the ADC table, scan into a bounded heap,
/// CRC-verify and rescore the survivors, drain the top `k`.
///
/// Every buffer comes from `workspace`. Nothing is allocated.
#[allow(clippy::too_many_arguments)]
pub fn query<P, S, I>(
    query_vec: &[i8],
    codebooks: &[Codebook<'_>],
    signs: &[bool],
    rounds: usize,
    payload: &mut P,
    rerank_src: &mut S,
    workspace: &mut Workspace<'_>,
    instrument: &mut I,
    payload_bytes: usize,
    k: usize,
    out: &mut [Candidate],
    metrics: &mut Metrics,
) -> Result<QueryStats, QueryError<P::Error, S::Error>>
where
    P: PayloadSource,
    S: RerankSource,
    I: Instrument,
{
    let mut stats = QueryStats::default();
    let centroids = match codebooks.first() {
        Some(cb) => cb.centroids(),
        None => return Err(QueryError::NoCodebook),
    };

    // --- Rotate ---------------------------------------------------------
    instrument.mark(Phase::Rotate, Edge::Enter);
    let d = query_vec.len();
    let scratch = workspace
        .rotation
        .get_mut(..d)
        .ok_or(QueryError::WorkspaceTooSmall)?;
    for (dst, src) in scratch.iter_mut().zip(query_vec.iter()) {
        *dst = *src as i32;
    }
    if rounds > 0 {
        rotate::rotate(scratch, signs, rounds).map_err(QueryError::Rotate)?;
    }
    // Narrow back to i8 for the table build. The rotation's scale factor is
    // common to every component, so the ordering the table induces is
    // unchanged by the shift.
    let shift = rotation_shift(d, rounds);
    let mut rotated = [0i8; MAX_D];
    let rotated = rotated.get_mut(..d).ok_or(QueryError::DimensionTooLarge)?;
    for (dst, src) in rotated.iter_mut().zip(scratch.iter()) {
        *dst = (*src >> shift).clamp(i8::MIN as i32, i8::MAX as i32) as i8;
    }
    instrument.mark(Phase::Rotate, Edge::Leave);

    // --- Table ----------------------------------------------------------
    instrument.mark(Phase::Table, Edge::Enter);
    let macs =
        adc::build_table(rotated, codebooks, workspace.adc_table).map_err(QueryError::Adc)?;
    stats.table_macs = macs as u32;
    instrument.mark(Phase::Table, Edge::Leave);

    // --- Scan -----------------------------------------------------------
    instrument.mark(Phase::Scan, Edge::Enter);
    let r = workspace.heap_scores.len().min(workspace.heap_ids.len());
    let mut heap = Heap::new(workspace.heap_scores, workspace.heap_ids, r)
        .ok_or(QueryError::WorkspaceTooSmall)?;
    payload.rewind();
    while let Some((run, first_id)) = payload.next_run().map_err(QueryError::Payload)? {
        let s = scan::scan_b8_auto(
            run,
            first_id,
            payload_bytes,
            workspace.adc_table,
            centroids,
            &mut heap,
        );
        stats.scan.scanned += s.scanned;
        stats.scan.offered += s.offered;
        stats.scan.evicted += s.evicted;
    }
    metrics.scanned = stats.scan.scanned;
    metrics.heap_insertions = stats.scan.offered;
    metrics.evictions = stats.scan.evicted;
    instrument.mark(Phase::Scan, Edge::Leave);

    // --- Rerank ---------------------------------------------------------
    instrument.mark(Phase::Rerank, Edge::Enter);
    let mut candidates = [Candidate { score: 0, id: 0 }; MAX_R];
    let held = heap.len();
    let slots = candidates
        .get_mut(..held)
        .ok_or(QueryError::CandidateDepthTooLarge)?;
    let drained = heap.drain_sorted(slots);

    let mut survivors = [Candidate { score: 0, id: 0 }; MAX_R];
    let dst = survivors
        .get_mut(..drained)
        .ok_or(QueryError::CandidateDepthTooLarge)?;
    let n = rerank::rerank(
        &slots[..drained],
        rerank_src,
        query_vec,
        dst,
        &mut stats.rerank,
    )
    .map_err(QueryError::Rerank)?;
    metrics.drops = stats.rerank.dropped;
    instrument.mark(Phase::Rerank, Edge::Leave);

    // --- Finalize -------------------------------------------------------
    instrument.mark(Phase::Finalize, Edge::Enter);
    let survivors = survivors
        .get_mut(..n)
        .ok_or(QueryError::WorkspaceTooSmall)?;
    rerank::sort_desc(survivors);
    let take = k.min(n).min(out.len());
    let src = survivors.get(..take).ok_or(QueryError::WorkspaceTooSmall)?;
    let dst = out.get_mut(..take).ok_or(QueryError::WorkspaceTooSmall)?;
    dst.copy_from_slice(src);
    stats.returned = take as u32;
    instrument.mark(Phase::Finalize, Edge::Leave);

    Ok(stats)
}

/// Largest dimension the fixed rotation buffer supports.
pub const MAX_D: usize = 1024;

/// Largest candidate depth the fixed candidate buffers support.
pub const MAX_R: usize = 512;

/// Bits to shift a rotated component back into `i8` range.
///
/// One round scales the squared norm by `len * 2`, so the amplitude scales by
/// `sqrt(len * 2)`. The shift is per-component and uniform, so it cannot change
/// the ranking the table induces.
const fn rotation_shift(d: usize, rounds: usize) -> u32 {
    if rounds == 0 {
        return 0;
    }
    // log2(sqrt(2d)) per round, rounded down.
    let bits = (usize::BITS - (2 * d).leading_zeros()) / 2;
    bits * rounds as u32
}

/// Why a query failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryError<P, S> {
    /// No codebooks were supplied.
    NoCodebook,
    /// A workspace buffer is too small.
    WorkspaceTooSmall,
    /// Dimension exceeds [`MAX_D`].
    DimensionTooLarge,
    /// Candidate depth exceeds [`MAX_R`].
    CandidateDepthTooLarge,
    /// The rotation refused.
    Rotate(rotate::RotateError),
    /// Table construction refused.
    Adc(adc::AdcError),
    /// The payload source failed.
    Payload(P),
    /// The rerank source failed.
    Rerank(S),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rerank::exact_score;
    use sector_codec::crc::crc32;
    use sector_hal::NoInstrument;
    use sector_quant::codebook::Scale;

    const D: usize = 32;
    const M: usize = 8;
    const DS: usize = D / M;
    const CENTROIDS: usize = 256;
    const N: usize = 64;
    const BLOCK: usize = 16;

    /// Records the phases an instrument was told about, in order.
    struct RecordingInstrument {
        marks: [(Phase, Edge); 16],
        len: usize,
    }

    impl Default for RecordingInstrument {
        fn default() -> Self {
            Self {
                marks: [(Phase::Rotate, Edge::Enter); 16],
                len: 0,
            }
        }
    }

    impl Instrument for RecordingInstrument {
        fn cycles(&self) -> u64 {
            0
        }
        fn mark(&mut self, phase: Phase, edge: Edge) {
            if let Some(slot) = self.marks.get_mut(self.len) {
                *slot = (phase, edge);
                self.len += 1;
            }
        }
    }

    /// Payload source that counts how many times it handed out bytes, and
    /// whether it ever copied.
    struct BorrowingPayload {
        codes: [u8; N * M],
        served: bool,
        runs: usize,
    }

    impl PayloadSource for BorrowingPayload {
        type Error = ();
        fn next_run(&mut self) -> Result<Option<(&[u8], u32)>, ()> {
            if self.served {
                return Ok(None);
            }
            self.served = true;
            self.runs += 1;
            Ok(Some((&self.codes[..], 0)))
        }
        fn rewind(&mut self) {
            self.served = false;
        }
    }

    struct Records {
        data: [[u8; D]; N],
        crcs: [[u32; D / BLOCK]; N],
        scratch: [u32; D / BLOCK],
    }

    impl RerankSource for Records {
        type Error = ();
        fn record(&mut self, id: u32) -> Result<Option<crate::rerank::Guarded<'_>>, ()> {
            let i = id as usize;
            if i >= N {
                return Ok(None);
            }
            self.scratch = self.crcs[i];
            // A record that is a whole number of blocks: offset zero, length the
            // full span. The sub-block case is covered in `rerank`'s own tests.
            Ok(Some(crate::rerank::Guarded {
                blocks: &self.data[i][..],
                offset: 0,
                len: D,
                crcs: &self.scratch[..],
            }))
        }
        fn block_bytes(&self) -> usize {
            BLOCK
        }
    }

    fn records() -> Records {
        let data: [[u8; D]; N] =
            core::array::from_fn(|v| core::array::from_fn(|i| ((v * 17 + i * 5) % 251) as u8));
        let crcs: [[u32; D / BLOCK]; N] = core::array::from_fn(|v| {
            core::array::from_fn(|b| crc32(&data[v][b * BLOCK..(b + 1) * BLOCK]))
        });
        Records {
            data,
            crcs,
            scratch: [0; D / BLOCK],
        }
    }

    struct Fixture {
        comps: [i8; CENTROIDS * DS],
        signs: [bool; D],
    }

    fn fixture() -> Fixture {
        Fixture {
            comps: core::array::from_fn(|i| ((i * 37) % 101) as i8 - 50),
            signs: core::array::from_fn(|i| i % 3 == 0),
        }
    }

    #[allow(clippy::type_complexity)]
    fn run(
        rounds: usize,
        instrument: &mut RecordingInstrument,
        out: &mut [Candidate],
    ) -> (QueryStats, Metrics) {
        let f = fixture();
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&f.comps, CENTROIDS, DS, scale).unwrap());

        let mut payload = BorrowingPayload {
            codes: core::array::from_fn(|i| ((i * 29) % 256) as u8),
            served: false,
            runs: 0,
        };
        let mut recs = records();

        let mut table = [0i32; M * CENTROIDS];
        let mut scores = [0i32; 32];
        let mut ids = [0u32; 32];
        let mut rot = [0i32; D];
        let mut bounce = [0u8; 512];
        let mut ws = Workspace {
            adc_table: &mut table,
            heap_scores: &mut scores,
            heap_ids: &mut ids,
            rotation: &mut rot,
            bounce: &mut bounce,
            scrub_cursor: 0,
        };

        let q: [i8; D] = core::array::from_fn(|i| ((i * 11) % 61) as i8 - 30);
        let mut metrics = Metrics::default();
        let stats = query(
            &q,
            &cbs,
            &f.signs,
            rounds,
            &mut payload,
            &mut recs,
            &mut ws,
            instrument,
            M,
            10,
            out,
            &mut metrics,
        )
        .expect("query");
        (stats, metrics)
    }

    #[test]
    fn all_five_phases_are_marked_in_order() {
        // The cost model attributes energy to these phases separately, so a
        // missing marker makes a measurement unattributable.
        let mut inst = RecordingInstrument::default();
        let mut out = [Candidate { score: 0, id: 0 }; 10];
        run(1, &mut inst, &mut out);

        let expected = [
            (Phase::Rotate, Edge::Enter),
            (Phase::Rotate, Edge::Leave),
            (Phase::Table, Edge::Enter),
            (Phase::Table, Edge::Leave),
            (Phase::Scan, Edge::Enter),
            (Phase::Scan, Edge::Leave),
            (Phase::Rerank, Edge::Enter),
            (Phase::Rerank, Edge::Leave),
            (Phase::Finalize, Edge::Enter),
            (Phase::Finalize, Edge::Leave),
        ];
        assert_eq!(inst.len, expected.len());
        assert_eq!(&inst.marks[..inst.len], &expected[..]);
    }

    #[test]
    fn the_query_returns_k_results_from_a_clean_corpus() {
        let mut inst = RecordingInstrument::default();
        let mut out = [Candidate { score: 0, id: 0 }; 10];
        let (stats, metrics) = run(1, &mut inst, &mut out);

        assert_eq!(stats.scan.scanned, N as u32);
        assert_eq!(stats.returned, 10);
        assert_eq!(stats.rerank.dropped, 0);
        assert_eq!(metrics.drops, 0);
        // Descending by exact score.
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score, "not sorted: {out:?}");
        }
    }

    #[test]
    fn results_carry_exact_scores_not_stage_one_estimates() {
        let mut inst = RecordingInstrument::default();
        let mut out = [Candidate { score: 0, id: 0 }; 10];
        run(1, &mut inst, &mut out);
        let recs = records();
        let q: [i8; D] = core::array::from_fn(|i| ((i * 11) % 61) as i8 - 30);
        for c in out.iter() {
            assert_eq!(c.score, exact_score(&q, &recs.data[c.id as usize]));
        }
    }

    #[test]
    fn table_macs_are_reported_and_match_the_configuration() {
        let mut inst = RecordingInstrument::default();
        let mut out = [Candidate { score: 0, id: 0 }; 10];
        let (stats, _) = run(1, &mut inst, &mut out);
        // 2^b * D = 256 * 32.
        assert_eq!(stats.table_macs, 8_192);
    }

    #[test]
    fn a_zero_round_query_skips_rotation_but_keeps_the_marker() {
        // The phase boundary exists whether or not the phase does work, so a
        // trace is comparable across configurations.
        let mut inst = RecordingInstrument::default();
        let mut out = [Candidate { score: 0, id: 0 }; 10];
        let (stats, _) = run(0, &mut inst, &mut out);
        assert_eq!(inst.marks[0], (Phase::Rotate, Edge::Enter));
        assert_eq!(inst.marks[1], (Phase::Rotate, Edge::Leave));
        assert_eq!(stats.returned, 10);
    }

    #[test]
    fn drops_reduce_the_result_and_are_counted() {
        let f = fixture();
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&f.comps, CENTROIDS, DS, scale).unwrap());
        let mut payload = BorrowingPayload {
            codes: core::array::from_fn(|i| ((i * 29) % 256) as u8),
            served: false,
            runs: 0,
        };
        let mut recs = records();
        // Corrupt every record: nothing can survive stage two.
        for v in 0..N {
            recs.data[v][0] ^= 0xFF;
        }

        let mut table = [0i32; M * CENTROIDS];
        let mut scores = [0i32; 32];
        let mut ids = [0u32; 32];
        let mut rot = [0i32; D];
        let mut bounce = [0u8; 512];
        let mut ws = Workspace {
            adc_table: &mut table,
            heap_scores: &mut scores,
            heap_ids: &mut ids,
            rotation: &mut rot,
            bounce: &mut bounce,
            scrub_cursor: 0,
        };
        let q: [i8; D] = core::array::from_fn(|i| ((i * 11) % 61) as i8 - 30);
        let mut out = [Candidate { score: 0, id: 0 }; 10];
        let mut metrics = Metrics::default();
        let mut inst = NoInstrument;

        let stats = query(
            &q,
            &cbs,
            &f.signs,
            1,
            &mut payload,
            &mut recs,
            &mut ws,
            &mut inst,
            M,
            10,
            &mut out,
            &mut metrics,
        )
        .expect("query survives total rerank corruption");

        assert_eq!(stats.returned, 0, "no candidate may survive");
        assert_eq!(stats.rerank.dropped, 32, "every candidate dropped");
        assert_eq!(metrics.drops, 32);
    }

    #[test]
    fn the_payload_is_borrowed_not_copied() {
        // The XIP path: one run served for the whole corpus, no bounce buffer
        // touched. A backend that had to copy would serve many runs.
        let f = fixture();
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&f.comps, CENTROIDS, DS, scale).unwrap());
        let mut payload = BorrowingPayload {
            codes: core::array::from_fn(|i| ((i * 29) % 256) as u8),
            served: false,
            runs: 0,
        };
        let mut recs = records();
        let mut table = [0i32; M * CENTROIDS];
        let mut scores = [0i32; 32];
        let mut ids = [0u32; 32];
        let mut rot = [0i32; D];
        let mut bounce = [0u8; 512];
        let bounce_before = bounce;
        let mut ws = Workspace {
            adc_table: &mut table,
            heap_scores: &mut scores,
            heap_ids: &mut ids,
            rotation: &mut rot,
            bounce: &mut bounce,
            scrub_cursor: 0,
        };
        let q: [i8; D] = core::array::from_fn(|i| ((i * 11) % 61) as i8 - 30);
        let mut out = [Candidate { score: 0, id: 0 }; 10];
        let mut metrics = Metrics::default();
        let mut inst = NoInstrument;
        query(
            &q,
            &cbs,
            &f.signs,
            1,
            &mut payload,
            &mut recs,
            &mut ws,
            &mut inst,
            M,
            10,
            &mut out,
            &mut metrics,
        )
        .unwrap();

        assert_eq!(payload.runs, 1, "the corpus was served in one borrow");
        assert_eq!(bounce, bounce_before, "the bounce buffer was never used");
    }

    #[test]
    fn metrics_mirror_the_returned_statistics() {
        // Host-side stats and on-device counters must agree, since they
        // cross-check each other during the measurement campaign.
        let mut inst = RecordingInstrument::default();
        let mut out = [Candidate { score: 0, id: 0 }; 10];
        let (stats, metrics) = run(1, &mut inst, &mut out);
        assert_eq!(metrics.scanned, stats.scan.scanned);
        assert_eq!(metrics.heap_insertions, stats.scan.offered);
        assert_eq!(metrics.evictions, stats.scan.evicted);
        assert_eq!(metrics.drops, stats.rerank.dropped);
    }
}
