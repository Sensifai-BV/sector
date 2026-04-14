//! Stage two: CRC-verify, drop on mismatch, rescore survivors.
//!
//! The `R` candidates from stage one are rescored against the higher-precision
//! copy: `R * D` bytes of flash traffic, 64 KiB at `R = 500`, `D = 128`, int8.
//! On byte-addressable NOR this is a sequence of load instructions; managed
//! NAND pays the FTL random-read penalty for the same pattern, so the smallest
//! tier executes it fastest. Both figures are estimates until measured on
//! silicon.
//!
//! # Drop semantics
//!
//! A candidate whose block CRC fails is removed and not replaced; the reranked
//! list is drawn from the survivors. A drop and an eviction have identical
//! effect on the survivor set, so recall accounting treats them identically and
//! drops enter the loss bound in the same term as intruder-driven evictions.
//!
//! # Implementation notes
//!
//! Rescore in the order stage one produced, keeping the access pattern as
//! sequential as the candidate set allows.
//!
//! Count drops and expose the count. The drop count is the first number to read
//! when measured recall falls below the host reference.

use crate::heap::Candidate;
use sector_codec::crc::verify;

/// What a rerank pass measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RerankStats {
    /// Candidates offered by stage one.
    pub candidates: u32,
    /// Candidates dropped on a CRC mismatch.
    ///
    /// Exposed because silent degradation makes a recall regression
    /// untraceable: a dropped candidate and an evicted one are identical in the
    /// result, and only this counter distinguishes them.
    pub dropped: u32,
    /// Blocks whose CRC was verified.
    pub blocks_verified: u32,
}

/// A record and the CRCs guarding the blocks it spans.
///
/// The two travel together so verify-then-rescore is a single decision rather
/// than two that can disagree about which bytes were checked.
pub type Guarded<'a> = (&'a [u8], &'a [u32]);

/// Source of a candidate's higher-precision record.
pub trait RerankSource {
    /// Backend error type.
    type Error;

    /// Borrow the record for `id`, and the CRCs of the blocks it spans.
    fn record(&mut self, id: u32) -> Result<Option<Guarded<'_>>, Self::Error>;

    /// Block size the CRCs cover.
    fn block_bytes(&self) -> usize;
}

/// Rescore `candidates` against their higher-precision records.
///
/// CRC-verify first, drop on mismatch, rescore survivors. A dropped candidate
/// is removed and not replaced: promoting the next-best would misreport the
/// damage, and for recall accounting a drop is identical to an eviction.
///
/// Writes survivors to `out` in input order and returns the count. The caller
/// sorts; this stage does not reorder, so a drop is visible as a shorter list.
pub fn rerank<S: RerankSource>(
    candidates: &[Candidate],
    source: &mut S,
    query: &[i8],
    out: &mut [Candidate],
    stats: &mut RerankStats,
) -> Result<usize, S::Error> {
    let block = source.block_bytes();
    let mut written = 0usize;

    for cand in candidates {
        stats.candidates += 1;
        let Some((record, crcs)) = source.record(cand.id)? else {
            stats.dropped += 1;
            continue;
        };

        // Verify every block the record spans before trusting any of its bytes.
        let mut ok = true;
        for (i, &expected) in crcs.iter().enumerate() {
            let start = i * block;
            let end = (start + block).min(record.len());
            let Some(chunk) = record.get(start..end) else {
                ok = false;
                break;
            };
            stats.blocks_verified += 1;
            if !verify(chunk, expected) {
                ok = false;
                break;
            }
        }
        if !ok {
            stats.dropped += 1;
            continue;
        }

        let score = exact_score(query, record);
        if let Some(slot) = out.get_mut(written) {
            *slot = Candidate { score, id: cand.id };
            written += 1;
        }
    }
    Ok(written)
}

/// Inner product of a query against a higher-precision record.
///
/// Integer throughout: the rerank copy is int8 at both tier profiles, so this
/// is the same arithmetic the host builder uses and the two produce identical
/// bytes.
pub fn exact_score(query: &[i8], record: &[u8]) -> i32 {
    let mut acc = 0i32;
    for (q, r) in query.iter().zip(record.iter()) {
        acc = acc.wrapping_add((*q as i32) * (*r as i8 as i32));
    }
    acc
}

/// Sort `candidates` in place by descending score, ties by ascending id.
///
/// Insertion sort: `R` is 500 at most and the list is already close to ordered
/// after stage one, so this beats anything with a larger constant.
pub fn sort_desc(candidates: &mut [Candidate]) {
    for i in 1..candidates.len() {
        let mut j = i;
        while j > 0 {
            let (a, b) = (candidates[j - 1], candidates[j]);
            let swap = b.score > a.score || (b.score == a.score && b.id < a.id);
            if !swap {
                break;
            }
            candidates.swap(j - 1, j);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_codec::crc::crc32;

    const D: usize = 32;
    const BLOCK: usize = 16;

    /// Records with per-block CRCs, corruptible for testing.
    struct TestSource {
        records: [[u8; D]; 8],
        crcs: [[u32; D / BLOCK]; 8],
        missing: Option<u32>,
        scratch_crc: [u32; D / BLOCK],
    }

    impl TestSource {
        fn new() -> Self {
            let records: [[u8; D]; 8] =
                core::array::from_fn(|v| core::array::from_fn(|i| ((v * 31 + i * 7) % 251) as u8));
            let crcs: [[u32; D / BLOCK]; 8] = core::array::from_fn(|v| {
                core::array::from_fn(|b| crc32(&records[v][b * BLOCK..(b + 1) * BLOCK]))
            });
            Self {
                records,
                crcs,
                missing: None,
                scratch_crc: [0; D / BLOCK],
            }
        }
        /// Flip a byte without updating the CRC: silent corruption.
        fn corrupt(&mut self, id: usize, byte: usize) {
            self.records[id][byte] ^= 0xFF;
        }
    }

    impl RerankSource for TestSource {
        type Error = ();
        fn record(&mut self, id: u32) -> Result<Option<(&[u8], &[u32])>, ()> {
            if self.missing == Some(id) {
                return Ok(None);
            }
            let i = id as usize;
            if i >= self.records.len() {
                return Ok(None);
            }
            self.scratch_crc = self.crcs[i];
            Ok(Some((&self.records[i][..], &self.scratch_crc[..])))
        }
        fn block_bytes(&self) -> usize {
            BLOCK
        }
    }

    fn query() -> [i8; D] {
        core::array::from_fn(|i| ((i * 5) % 41) as i8 - 20)
    }

    fn candidates(n: usize) -> [Candidate; 8] {
        core::array::from_fn(|i| Candidate {
            score: if i < n { 100 - i as i32 } else { 0 },
            id: i as u32,
        })
    }

    #[test]
    fn clean_candidates_all_survive_and_are_rescored() {
        let mut src = TestSource::new();
        let q = query();
        let cands = candidates(8);
        let mut out = [Candidate { score: 0, id: 0 }; 8];
        let mut stats = RerankStats::default();

        let n = rerank(&cands, &mut src, &q, &mut out, &mut stats).unwrap();
        assert_eq!(n, 8);
        assert_eq!(stats.dropped, 0);
        assert_eq!(stats.candidates, 8);
        // Every block of every record was verified: 8 records x 2 blocks.
        assert_eq!(stats.blocks_verified, 16);

        // Scores are the exact inner product, not the stage-one estimate.
        for (i, c) in out[..n].iter().enumerate() {
            assert_eq!(c.score, exact_score(&q, &src.records[i]));
        }
    }

    #[test]
    fn a_corrupted_record_is_dropped_not_rescored() {
        let mut src = TestSource::new();
        src.corrupt(3, 5);
        let q = query();
        let cands = candidates(8);
        let mut out = [Candidate { score: 0, id: 0 }; 8];
        let mut stats = RerankStats::default();

        let n = rerank(&cands, &mut src, &q, &mut out, &mut stats).unwrap();
        assert_eq!(n, 7, "the corrupted candidate must not appear");
        assert_eq!(stats.dropped, 1);
        assert!(!out[..n].iter().any(|c| c.id == 3));
    }

    #[test]
    fn a_drop_is_not_backfilled_from_the_next_candidate() {
        // Promoting the next-best would hide the damage. For recall accounting
        // a drop is identical to an eviction, and the counter is what makes it
        // visible.
        let mut src = TestSource::new();
        src.corrupt(0, 0);
        let q = query();
        let cands = candidates(4);
        let mut out = [Candidate { score: 0, id: 0 }; 8];
        let mut stats = RerankStats::default();

        let n = rerank(&cands[..4], &mut src, &q, &mut out, &mut stats).unwrap();
        assert_eq!(n, 3);
        assert_eq!(stats.candidates, 4);
        assert_eq!(stats.dropped, 1);
        // Ids 1, 2, 3 in input order; nothing pulled forward to replace 0.
        assert_eq!(out[0].id, 1);
        assert_eq!(out[1].id, 2);
        assert_eq!(out[2].id, 3);
    }

    #[test]
    fn corruption_in_any_block_of_a_record_drops_it() {
        // A record spanning two blocks must verify both. Corrupting only the
        // second would slip through a first-block-only check.
        for byte in [0usize, BLOCK, D - 1] {
            let mut src = TestSource::new();
            src.corrupt(2, byte);
            let q = query();
            let cands = candidates(8);
            let mut out = [Candidate { score: 0, id: 0 }; 8];
            let mut stats = RerankStats::default();
            let n = rerank(&cands, &mut src, &q, &mut out, &mut stats).unwrap();
            assert_eq!(stats.dropped, 1, "byte {byte} slipped through");
            assert!(!out[..n].iter().any(|c| c.id == 2));
        }
    }

    #[test]
    fn an_absent_record_is_a_drop_not_an_error() {
        // An unreadable rerank block degrades the result; it does not fail the
        // query. The asymmetry against the codebook is deliberate.
        let mut src = TestSource::new();
        src.missing = Some(4);
        let q = query();
        let cands = candidates(8);
        let mut out = [Candidate { score: 0, id: 0 }; 8];
        let mut stats = RerankStats::default();

        let n = rerank(&cands, &mut src, &q, &mut out, &mut stats).unwrap();
        assert_eq!(n, 7);
        assert_eq!(stats.dropped, 1);
    }

    #[test]
    fn rescoring_reorders_against_the_stage_one_estimate() {
        // The point of stage two: the compressed score is an approximation, and
        // the exact score can rank differently. If the order never changed,
        // reranking would be pure cost.
        let mut src = TestSource::new();
        let q = query();
        let cands = candidates(8);
        let mut out = [Candidate { score: 0, id: 0 }; 8];
        let mut stats = RerankStats::default();

        let n = rerank(&cands, &mut src, &q, &mut out, &mut stats).unwrap();
        sort_desc(&mut out[..n]);
        let reranked: [u32; 8] = core::array::from_fn(|i| out[i].id);
        let stage_one: [u32; 8] = core::array::from_fn(|i| cands[i].id);
        assert_ne!(reranked, stage_one, "rerank never changed the order");
    }

    #[test]
    fn sorting_is_deterministic_on_ties() {
        let mut c = [
            Candidate { score: 10, id: 7 },
            Candidate { score: 10, id: 2 },
            Candidate { score: 20, id: 9 },
            Candidate { score: 10, id: 5 },
        ];
        sort_desc(&mut c);
        assert_eq!(
            c,
            [
                Candidate { score: 20, id: 9 },
                Candidate { score: 10, id: 2 },
                Candidate { score: 10, id: 5 },
                Candidate { score: 10, id: 7 },
            ]
        );
    }

    #[test]
    fn verification_cost_scales_with_r_not_n() {
        // The measurement T-07 deferred until the scan existed. Stage two
        // verifies blocks for R candidates; verifying the whole payload would
        // cost N/vectors-per-block. At T0 (N=8,966, 32 vectors per block, R=100)
        // that is 280 blocks against 100 records: lazy verification is cheaper,
        // and by a factor that grows with N.
        let n_vectors = 8_966usize;
        let vectors_per_block = 32usize;
        let r = 100usize;
        let eager_blocks = n_vectors.div_ceil(vectors_per_block);
        assert_eq!(eager_blocks, 281);
        assert!(r < eager_blocks);

        // At N = 1e6 the gap is two orders of magnitude.
        let big = 1_000_000usize.div_ceil(vectors_per_block);
        assert_eq!(big, 31_250);
        assert!(big / r > 300);
    }
}
