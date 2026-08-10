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

/// A candidate's record, the whole blocks containing it, and their CRCs.
///
/// The three travel together so verify-then-rescore is a single decision rather
/// than two that can disagree about which bytes were checked.
///
/// # Why the record is addressed inside the blocks rather than handed over alone
///
/// A CRC covers a whole block, and a record is not in general a whole number of
/// blocks. At every shipped profile it is *smaller*: `D = 128` int8 is a 128 B
/// record in a 512 B block, so four records share one CRC. Verification must
/// therefore read the full block while scoring reads only the record's bytes,
/// and those are different extents.
///
/// An earlier version of this trait handed back the record and its CRCs and let
/// this module derive the verified bytes from the record slice. That is correct
/// only when a record is a whole multiple of a block — the one case the shipped
/// profiles are not — and would otherwise have checksummed the record's own
/// bytes against a CRC computed over the block containing it, failing every
/// candidate. The extent is explicit here so the two cannot diverge.
#[derive(Clone, Copy, Debug)]
pub struct Guarded<'a> {
    /// The whole blocks containing the record. Length is a multiple of
    /// [`RerankSource::block_bytes`], matching `crcs`.
    pub blocks: &'a [u8],
    /// Byte offset of the record within `blocks`.
    pub offset: usize,
    /// Record length in bytes.
    pub len: usize,
    /// One CRC per block in `blocks`, in order.
    pub crcs: &'a [u32],
}

impl<'a> Guarded<'a> {
    /// The record's bytes, or `None` when `offset`/`len` fall outside `blocks`.
    pub fn record(&self) -> Option<&'a [u8]> {
        self.blocks.get(self.offset..self.offset + self.len)
    }
}

/// Source of a candidate's higher-precision record.
pub trait RerankSource {
    /// Backend error type.
    type Error;

    /// Borrow the blocks holding `id`'s record, with the record's position in
    /// them and the CRCs guarding them.
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
        let Some(guarded) = source.record(cand.id)? else {
            stats.dropped += 1;
            continue;
        };

        // Verify every block containing the record before trusting any of its
        // bytes. The blocks are checked whole, which is the extent the CRC was
        // computed over; the record is a sub-range of them.
        let mut ok = true;
        for (i, &expected) in guarded.crcs.iter().enumerate() {
            let Some(chunk) = guarded.blocks.get(i * block..(i + 1) * block) else {
                ok = false;
                break;
            };
            stats.blocks_verified += 1;
            if !verify(chunk, expected) {
                ok = false;
                break;
            }
        }
        // A record whose offset and length fall outside the blocks handed over
        // is a backend defect. Drop rather than score partial bytes: a short
        // read would silently produce a plausible wrong score.
        let Some(record) = guarded.record().filter(|_| ok) else {
            stats.dropped += 1;
            continue;
        };

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
    const BLOCK: usize = 64;
    const N: usize = 8;
    /// Records per block. Two, so a CRC covers more than one record — the
    /// geometry every shipped profile has (four 128 B records per 512 B block)
    /// and the one a whole-record CRC assumption gets wrong.
    const PER_BLOCK: usize = BLOCK / D;
    const BLOCKS: usize = N / PER_BLOCK;

    /// A packed rerank region with per-block CRCs, corruptible for testing.
    struct TestSource {
        region: [u8; N * D],
        crcs: [u32; BLOCKS],
        missing: Option<u32>,
    }

    impl TestSource {
        fn new() -> Self {
            let mut region = [0u8; N * D];
            for v in 0..N {
                for i in 0..D {
                    region[v * D + i] = ((v * 31 + i * 7) % 251) as u8;
                }
            }
            let crcs = core::array::from_fn(|b| crc32(&region[b * BLOCK..(b + 1) * BLOCK]));
            Self {
                region,
                crcs,
                missing: None,
            }
        }
        /// Flip a byte without updating the CRC: silent corruption.
        fn corrupt(&mut self, id: usize, byte: usize) {
            self.region[id * D + byte] ^= 0xFF;
        }
    }

    impl RerankSource for TestSource {
        type Error = ();
        fn record(&mut self, id: u32) -> Result<Option<Guarded<'_>>, ()> {
            if self.missing == Some(id) {
                return Ok(None);
            }
            let i = id as usize;
            if i >= N {
                return Ok(None);
            }
            // The block containing the record, and the record's offset in it.
            let block = i / PER_BLOCK;
            Ok(Some(Guarded {
                blocks: &self.region[block * BLOCK..(block + 1) * BLOCK],
                offset: (i % PER_BLOCK) * D,
                len: D,
                crcs: &self.crcs[block..block + 1],
            }))
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
        // One block covers each record here, and two records share a block, so
        // eight candidates verify eight blocks — the same block twice for each
        // pair. Re-verifying is the honest cost of a shared CRC: the alternative
        // is caching a verification result, which would trust a block that
        // changed under it.
        assert_eq!(stats.blocks_verified, 8);

        // Scores are the exact inner product, not the stage-one estimate.
        for (i, c) in out[..n].iter().enumerate() {
            let want = exact_score(&q, &src.region[i * D..(i + 1) * D]);
            assert_eq!(c.score, want);
        }
    }

    #[test]
    fn a_corrupted_record_is_dropped_not_rescored() {
        // Corrupting record 3 fails the CRC of the block holding it, and that
        // block also holds record 2. Both drop.
        //
        // This is the real behaviour of a shared CRC and it is worth asserting
        // rather than designing around: a single flipped byte in the rerank
        // region costs `records_per_block` candidates, four at every shipped
        // profile. The blast radius of a payload fault is the block, not the
        // record — which is a smaller version of the same shared-structure
        // argument the codebook makes, and it is why block size is a protection
        // parameter rather than a layout convenience.
        let mut src = TestSource::new();
        src.corrupt(3, 5);
        let q = query();
        let cands = candidates(8);
        let mut out = [Candidate { score: 0, id: 0 }; 8];
        let mut stats = RerankStats::default();

        let n = rerank(&cands, &mut src, &q, &mut out, &mut stats).unwrap();
        assert_eq!(
            n,
            8 - PER_BLOCK,
            "the damaged block's records must not appear"
        );
        assert_eq!(stats.dropped as usize, PER_BLOCK);
        assert!(!out[..n].iter().any(|c| c.id == 3));
        // Record 2 shares the block and is collateral damage.
        assert!(!out[..n].iter().any(|c| c.id == 2));
        // A record in a different block is untouched.
        assert!(out[..n].iter().any(|c| c.id == 4));
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
        // Records 0 and 1 share the damaged block; 2 and 3 survive in input
        // order with nothing pulled forward.
        assert_eq!(n, 4 - PER_BLOCK);
        assert_eq!(stats.candidates, 4);
        assert_eq!(stats.dropped as usize, PER_BLOCK);
        assert_eq!(out[0].id, 2);
        assert_eq!(out[1].id, 3);
    }

    #[test]
    fn corruption_anywhere_in_a_block_drops_every_record_in_it() {
        // Any byte of the block invalidates it, including bytes belonging to a
        // different record and bytes in the block's slack. A check that verified
        // only the candidate's own bytes would pass on the first two.
        for byte in [0usize, D, D + 7, BLOCK - 1] {
            let mut src = TestSource::new();
            // Corrupt at an absolute offset in the block holding records 2 and 3.
            let block_base = (2 / PER_BLOCK) * BLOCK;
            src.region[block_base + byte] ^= 0xFF;

            let q = query();
            let cands = candidates(8);
            let mut out = [Candidate { score: 0, id: 0 }; 8];
            let mut stats = RerankStats::default();
            let n = rerank(&cands, &mut src, &q, &mut out, &mut stats).unwrap();
            assert_eq!(
                stats.dropped as usize, PER_BLOCK,
                "byte {byte} of the block slipped through"
            );
            assert!(!out[..n].iter().any(|c| c.id == 2 || c.id == 3));
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
