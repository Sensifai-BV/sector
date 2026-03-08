//! The index pipeline, as the benchmarks drive it.
//!
//! One place builds an index and answers queries, so every axis measures the
//! same code path. A benchmark that reimplemented stage one would be measuring
//! the benchmark.

use crate::Config;
use sector_build::encode::{encode, quantize, Populations, QuantizedCodebook};
use sector_build::label_opt::{optimise, permute_centroids, permute_codes};
use sector_build::train::{train, TrainConfig, TrainReport};
use sector_core::heap::{Candidate, Heap};
use sector_core::scan as core_scan;
use sector_hal::{Edge, Instrument, Phase};
use sector_quant::adc;
use sector_quant::codebook::{Codebook, Scale};

/// A built index, held in the shape the query path consumes.
pub struct Pipeline {
    /// Quantized codebook — what ships and what the device holds.
    pub codebook: QuantizedCodebook,
    /// Codes, `n * m`.
    pub codes: Vec<u8>,
    /// Per-centroid populations, the criticality input.
    pub populations: Populations,
    /// Training statistics.
    pub report: TrainReport,
    /// Vectors.
    pub n: usize,
    /// Dimension.
    pub d: usize,
    /// Subspaces.
    pub m: usize,
    /// Centroids per subspace.
    pub centroids: usize,
    /// Subspace dimension.
    pub ds: usize,
    /// Displacement reduction achieved by label optimisation, per subspace.
    pub relabel_reduction: Vec<f64>,
}

impl Pipeline {
    /// Codebook bytes — `2^b * D * s`, independent of `N`.
    pub fn codebook_bytes(&self) -> usize {
        self.codebook.byte_len()
    }

    /// Payload bytes per vector, `m * b / 8`.
    ///
    /// Matches `Profile::payload_bytes` rather than assuming one byte per code.
    /// The earlier `self.m` was correct only at b=8 and silently reported
    /// **double** the stored size at b=4, which is exactly the configuration a
    /// GIST-class dimension is forced into — so the figure was wrong precisely
    /// where it mattered most.
    ///
    /// This is the accounting figure. The in-memory `codes` array is still one
    /// byte per code, because unpacking nibbles per lookup costs more host time
    /// than the memory saves; [`Self::codes_bytes`] reports what is actually
    /// resident, so the two are never confused.
    pub const fn payload_bytes(&self) -> usize {
        // b from centroids = 2^b, rather than storing b as a second copy of the
        // same fact: two stored copies of one fact eventually disagree.
        self.m * self.centroids.trailing_zeros() as usize / 8
    }

    /// Bytes the in-memory `codes` array actually occupies.
    ///
    /// Distinct from [`Self::payload_bytes`]: that is what a device stores in
    /// flash, this is what the host benchmark holds in RAM. Reporting one as
    /// the other would misstate either the storage claim or the memory
    /// measurement.
    pub const fn codes_bytes(&self, n: usize) -> usize {
        n * self.m
    }

    /// Stage one candidate list: the top `r` by ADC score.
    ///
    /// Runs `sector_core`'s scan and bounded heap — the same code the device
    /// executes. An earlier version reimplemented stage one here, scoring every
    /// vector into a `Vec` and sorting all `N`; that made the benchmark measure
    /// the benchmark, and left two implementations free to diverge silently.
    pub fn stage_one(&self, q: &[f32], r: usize) -> Vec<u32> {
        self.stage_one_instrumented(q, r, &mut NoTimer).0
    }

    /// Quantize a query to `i8` per subspace, using each subspace's own scale.
    ///
    /// The device receives an integer query. Quantizing here rather than
    /// scoring in floats is what lets the host and device paths agree, and it
    /// is why the scan can be integer-only.
    fn quantize_query(&self, q: &[f32]) -> Vec<i8> {
        let mut out = vec![0i8; self.m * self.ds];
        for j in 0..self.m {
            let scale = self
                .codebook
                .scales
                .get(j)
                .copied()
                .unwrap_or(Scale { num: 1, den: 1 });
            for i in 0..self.ds {
                // Invert the codebook's own scale so query and centroid share
                // one integer domain.
                let v = q[j * self.ds + i];
                let scaled = if scale.num != 0 {
                    v * (scale.den as f32) / (scale.num as f32)
                } else {
                    v
                };
                out[j * self.ds + i] = scaled.round().clamp(-127.0, 127.0) as i8;
            }
        }
        out
    }

    /// Borrow the codebook as per-subspace [`Codebook`] views.
    fn books(&self) -> Vec<Codebook<'_>> {
        (0..self.m)
            .map(|j| {
                let start = j * self.centroids * self.ds;
                let end = start + self.centroids * self.ds;
                Codebook::new(
                    &self.codebook.components[start..end],
                    self.centroids,
                    self.ds,
                    self.codebook.scales[j],
                )
                .expect("codebook shape fixed at build")
            })
            .collect()
    }

    /// Stage one, with each phase marked. Returns the candidates and the bytes
    /// the scan read.
    ///
    /// Bytes are returned rather than pushed into the instrument: how many
    /// bytes a scan touches is a property of the corpus and the profile, not of
    /// whatever is doing the timing.
    pub fn stage_one_instrumented<I: Instrument>(
        &self,
        q: &[f32],
        r: usize,
        inst: &mut I,
    ) -> (Vec<u32>, u64) {
        // Rotation is applied at build time in this harness, so the per-query
        // rotate phase holds only the query quantization. Marked explicitly:
        // an unmarked phase and an empty one are different claims.
        inst.mark(Phase::Rotate, Edge::Enter);
        let qi = self.quantize_query(q);
        inst.mark(Phase::Rotate, Edge::Leave);

        // Table build: 2^b * D multiply-accumulates, independent of N. The L2
        // form folds -||c||^2 into each entry, so the scan's maximum is the
        // nearest neighbour by distance rather than by inner product — the two
        // orderings disagree badly on vectors of unequal norm.
        inst.mark(Phase::Table, Edge::Enter);
        let mut table = vec![0i32; self.m * self.centroids];
        let books = self.books();
        adc::build_table_l2(&qi, &books, &mut table).expect("table shape fixed at build");
        inst.mark(Phase::Table, Edge::Leave);

        // Scan: sector_core's own loop and bounded heap — m lookups, m adds, a
        // threshold test against the heap minimum, and no sort of N.
        inst.mark(Phase::Scan, Edge::Enter);
        let cap = r.min(self.n).max(1);
        let mut scores = vec![0i32; cap];
        let mut ids = vec![0u32; cap];
        let mut heap = Heap::new(&mut scores, &mut ids, cap).expect("buffers sized above");
        core_scan::scan_b8(&self.codes, 0, self.m, &table, self.centroids, &mut heap);
        inst.mark(Phase::Scan, Edge::Leave);

        inst.mark(Phase::Finalize, Edge::Enter);
        let mut out = vec![Candidate { score: 0, id: 0 }; cap];
        let n = heap.drain_sorted(&mut out);
        let picked: Vec<u32> = out[..n].iter().map(|c| c.id).collect();
        inst.mark(Phase::Finalize, Edge::Leave);

        (picked, (self.n * self.m) as u64)
    }

    /// Stage two: rescore candidates against their full-precision records.
    pub fn rerank(
        &self,
        base: &[f32],
        d: usize,
        q: &[f32],
        candidates: &[u32],
        k: usize,
    ) -> Vec<u32> {
        self.rerank_instrumented(base, d, q, candidates, k, &mut NoTimer)
            .0
    }

    /// Stage two, with the phase marked. Returns the top `k` and bytes read.
    ///
    /// Instrumented separately because the cost model attributes energy to
    /// Rerank as its own term. An unmarked phase reports zero, which is
    /// indistinguishable from a phase too fast to resolve — worse than no
    /// measurement, because it looks like one.
    pub fn rerank_instrumented<I: Instrument>(
        &self,
        base: &[f32],
        d: usize,
        q: &[f32],
        candidates: &[u32],
        k: usize,
        inst: &mut I,
    ) -> (Vec<u32>, u64) {
        inst.mark(Phase::Rerank, Edge::Enter);
        let out = self.rerank_inner(base, d, q, candidates, k);
        inst.mark(Phase::Rerank, Edge::Leave);
        // Each candidate's full-precision record: D f32 components.
        (out, (candidates.len() * d * 4) as u64)
    }

    fn rerank_inner(
        &self,
        base: &[f32],
        d: usize,
        q: &[f32],
        candidates: &[u32],
        k: usize,
    ) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = candidates
            .iter()
            .map(|id| {
                let row = &base[*id as usize * d..(*id as usize + 1) * d];
                let mut acc = 0f32;
                for (a, b) in row.iter().zip(q.iter()) {
                    let diff = a - b;
                    acc += diff * diff;
                }
                (acc, *id)
            })
            .collect();
        scored.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        scored.truncate(k);
        scored.into_iter().map(|(_, id)| id).collect()
    }
}

struct NoTimer;

impl Instrument for NoTimer {
    fn cycles(&self) -> u64 {
        0
    }
    fn mark(&mut self, _phase: Phase, _edge: Edge) {}
}

/// Build an index through the real offline pipeline.
///
/// Training uses at most `cfg.train_n` vectors. PQ codebooks are trained on a
/// sample rather than the whole corpus — the codebook is `2^b · D` bytes
/// regardless of `N`, and k-means over 10^6 vectors costs ten times what it
/// costs over the 100,000-vector learn set the standard datasets ship for
/// exactly this purpose. Encoding still covers every vector.
pub fn build_index(base: &[f32], n: usize, d: usize, cfg: &Config) -> Result<Pipeline, String> {
    let train_n = if cfg.train_n == 0 {
        n
    } else {
        cfg.train_n.min(n)
    };
    let tcfg = TrainConfig {
        d,
        m: cfg.m,
        b: cfg.b,
        iterations: 25,
        seed: cfg.seed,
    };
    let (books, report) =
        train(&base[..train_n * d], train_n, tcfg).map_err(|e| format!("{e:?}"))?;
    let (mut codes, populations) = encode(base, n, d, &books);

    // Label optimisation is lossless, so it is applied unconditionally. The
    // permutation must reach the codebook and the codes together — permuting
    // only one produces an image that is structurally valid and reconstructs
    // every vector wrongly.
    let mut relabelled = Vec::with_capacity(books.len());
    let mut reductions = Vec::with_capacity(books.len());
    for (j, book) in books.iter().enumerate() {
        let perm = optimise(book, cfg.b as u32, 8);
        permute_codes(&mut codes, cfg.m, j, &perm.map);
        relabelled.push(permute_centroids(book, &perm.map));
        reductions.push(perm.reduction());
    }

    // Quantization is a separate stage from training, and recall is measured
    // after it: the quantized codebook is what ships.
    let codebook = quantize(&relabelled, 127);

    Ok(Pipeline {
        codebook,
        codes,
        populations,
        report,
        n,
        d,
        m: cfg.m,
        centroids: 1 << cfg.b,
        ds: d / cfg.m,
        relabel_reduction: reductions,
    })
}

/// Capture the exact top-`k` this pipeline returns for a seeded query set.
///
/// Recall averages can stay flat while individual answers change, so the lock
/// is on the identifiers themselves. An optimisation that returns a different
/// set is a behaviour change whether or not the average moved.
pub fn capture_topk(
    pipeline: &Pipeline,
    base: &[f32],
    d: usize,
    queries: &[f32],
    nq: usize,
    r: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    (0..nq)
        .map(|qi| {
            let q = &queries[qi * d..(qi + 1) * d];
            let candidates = pipeline.stage_one(q, r);
            pipeline.rerank(base, d, q, &candidates, k)
        })
        .collect()
}

/// Compare a capture against a locked baseline.
///
/// Returns the query indices that differ, so a failure names the cases rather
/// than only reporting that something moved.
pub fn diff_topk(baseline: &[Vec<u32>], now: &[Vec<u32>]) -> Vec<usize> {
    baseline
        .iter()
        .zip(now.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(n: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0f32; n * d];
        for v in 0..n {
            let cluster = (v % 4) as f32;
            for j in 0..d {
                out[v * d + j] = cluster * 20.0 + (((v * 13 + j * 7) % 37) as f32) - 18.0;
            }
        }
        out
    }

    fn cfg() -> Config {
        Config {
            n: 0,
            m: 2,
            b: 4,
            r: 50,
            k: 10,
            seed: 7,
            // Small fixtures train on everything they have.
            train_n: 0,
        }
    }

    #[test]
    fn the_pipeline_builds_and_answers_queries() {
        let d = 8usize;
        let n = 400usize;
        let base = corpus(n, d);
        let p = build_index(&base, n, d, &cfg()).unwrap();

        assert_eq!(p.codes.len(), n * 2);
        assert_eq!(p.centroids, 16);
        // Codebook is 2^b * D * s, independent of N.
        assert_eq!(p.codebook_bytes(), 16 * d);

        let q = &base[0..d];
        let candidates = p.stage_one(q, 50);
        assert_eq!(candidates.len(), 50);
        let top = p.rerank(&base, d, q, &candidates, 10);
        assert_eq!(top.len(), 10);
        // The query is a corpus vector, so it must rank itself first.
        assert_eq!(top[0], 0);
    }

    #[test]
    fn the_codebook_size_does_not_depend_on_corpus_size() {
        // The claim the whole protection argument rests on.
        let d = 8usize;
        let small = corpus(200, d);
        let large = corpus(800, d);
        let a = build_index(&small, 200, d, &cfg()).unwrap();
        let b = build_index(&large, 800, d, &cfg()).unwrap();
        assert_eq!(a.codebook_bytes(), b.codebook_bytes());
        assert_ne!(a.codes.len(), b.codes.len());
    }

    #[test]
    fn stage_one_is_deterministic() {
        // Two runs must agree, or no comparison between configurations means
        // anything.
        let d = 8usize;
        let n = 300usize;
        let base = corpus(n, d);
        let p = build_index(&base, n, d, &cfg()).unwrap();
        let q = &base[40..40 + d];
        assert_eq!(p.stage_one(q, 30), p.stage_one(q, 30));
    }

    #[test]
    fn rerank_narrows_the_candidate_list_without_inventing_ids() {
        let d = 8usize;
        let n = 300usize;
        let base = corpus(n, d);
        let p = build_index(&base, n, d, &cfg()).unwrap();
        let q = &base[7 * d..8 * d];
        let candidates = p.stage_one(q, 40);
        let top = p.rerank(&base, d, q, &candidates, 10);
        for id in &top {
            assert!(candidates.contains(id), "rerank invented id {id}");
        }
    }
}
