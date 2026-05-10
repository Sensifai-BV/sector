//! Ground-truth recall@k measurement.
//!
//! Recall is the figure of merit: any index can be made arbitrarily fast by
//! returning wrong answers, so a latency claim without recall carries no
//! information.
//!
//! # Reporting rules
//!
//! When measuring corruption damage, report recall relative to clean two-stage
//! recall rather than a perfect oracle. At `R = 100` roughly 36% of true
//! top-`k` items are already outside the clean candidate list — clean
//! two-stage recall is 0.637 at m=32, b=8, D=256, N=20,000 — and charging that
//! gap to corruption measures the baseline's limits.
//!
//! Report the baseline alongside every damaged figure, so the two cannot be
//! separated in a table quoted from elsewhere.
//!
//! Break score ties deterministically. Ties at the candidate boundary are real
//! at these scales, and measurement resolution is `1/(N_q · k)`: a discrepancy
//! of exactly that size is a tie rather than a defect.

/// The corpus and its encoding, as the sweeps see them.
#[derive(Clone, Copy, Debug)]
pub struct Encoded<'a> {
    /// Row-major corpus, `n * d`.
    pub corpus: &'a [f32],
    /// Vectors.
    pub n: usize,
    /// Dimension.
    pub d: usize,
    /// PQ codes, `n * m`.
    pub codes: &'a [u8],
    /// Subspaces.
    pub m: usize,
    /// Centroid components, `m * k * ds`, row-major by (subspace, centroid).
    ///
    /// Carried so stage-one scores are reconstructions rather than exact inner
    /// products. Without it the candidate list would be the ground truth.
    pub centroids: &'a [f32],
    /// Centroids per subspace.
    pub k: usize,
}

impl<'a> Encoded<'a> {
    /// Components of centroid `c` in subspace `j`.
    pub fn centroid(&self, j: usize, c: usize) -> Option<&'a [f32]> {
        let ds = self.d / self.m.max(1);
        let start = (j * self.k + c) * ds;
        self.centroids.get(start..start + ds)
    }
}

/// Ids of the top `r` by **exact** inner product with `q`.
///
/// This is the ground-truth ranking, not the candidate list. Using it as both
/// makes clean recall identically 1.0 and charges every measured loss to a
/// baseline no real query path achieves.
///
/// Ties break by id: ties at the candidate boundary are real at these scales,
/// and non-deterministic ordering makes a recall measurement irreproducible.
pub fn exact_top_ids(data: Encoded<'_>, q: &[f32], r: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..data.n).map(|v| (score(data, v, q), v as u32)).collect();
    sort_desc(&mut scored);
    scored.into_iter().take(r).map(|(_, id)| id).collect()
}

/// Ids of the top `r` by **reconstructed** inner product — the stage-one
/// candidate list a device actually produces.
///
/// Scores come from the PQ reconstruction, so this list is not the exact
/// ranking. Clean two-stage recall is therefore below 1, and every corruption
/// loss is reported against it rather than against a perfect oracle: at
/// `R = 100` a substantial share of true top-`k` items are already outside the
/// clean candidate list, and charging that gap to corruption would measure the
/// baseline's limits.
pub fn top_ids(data: Encoded<'_>, q: &[f32], r: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..data.n)
        .map(|v| (reconstructed_score(data, v, q), v as u32))
        .collect();
    sort_desc(&mut scored);
    scored.into_iter().take(r).map(|(_, id)| id).collect()
}

/// Ids of the top `r` when every vector using centroid `c` of subspace `j`
/// has its stage-one score shifted by `shift`.
pub fn top_ids_shifted(
    data: Encoded<'_>,
    q: &[f32],
    r: usize,
    j: usize,
    c: usize,
    shift: f32,
) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..data.n)
        .map(|v| {
            let mut s = reconstructed_score(data, v, q);
            // The shift is identical for every vector using the centroid,
            // which is what makes the fan-out matter.
            if data.codes.get(v * data.m + j).copied() == Some(c as u8) {
                s += shift;
            }
            (s, v as u32)
        })
        .collect();
    sort_desc(&mut scored);
    scored.into_iter().take(r).map(|(_, id)| id).collect()
}

/// Stage-one score: the query against the vector's PQ reconstruction.
fn reconstructed_score(data: Encoded<'_>, v: usize, q: &[f32]) -> f32 {
    let ds = data.d / data.m.max(1);
    let mut s = 0f32;
    for j in 0..data.m {
        let Some(&code) = data.codes.get(v * data.m + j) else {
            continue;
        };
        let Some(centroid) = data.centroid(j, code as usize) else {
            continue;
        };
        let Some(qsub) = q.get(j * ds..(j + 1) * ds) else {
            continue;
        };
        for (a, b) in centroid.iter().zip(qsub.iter()) {
            s += a * b;
        }
    }
    s
}

fn score(data: Encoded<'_>, v: usize, q: &[f32]) -> f32 {
    let mut s = 0f32;
    if let Some(row) = data.corpus.get(v * data.d..(v + 1) * data.d) {
        for (a, b) in row.iter().zip(q.iter()) {
            s += a * b;
        }
    }
    s
}

fn sort_desc(scored: &mut [(f32, u32)]) {
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
}

/// Fraction of the first `k` of `truth` present in `candidates`.
pub fn recall_at_k(candidates: &[u32], truth: &[u32], k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let hits = truth
        .iter()
        .take(k)
        .filter(|t| candidates.contains(t))
        .count();
    hits as f32 / k as f32
}

/// Measurement resolution: the smallest recall difference a run can express.
///
/// A discrepancy of exactly this size is a tie, not a defect.
pub const fn resolution(queries: usize, k: usize) -> f32 {
    if queries == 0 || k == 0 {
        return 0.0;
    }
    1.0 / (queries * k) as f32
}

/// Mean recall over a query set.
pub fn mean_recall(per_query: &[f32]) -> f32 {
    if per_query.is_empty() {
        return 0.0;
    }
    per_query.iter().sum::<f32>() / per_query.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial codebook where centroid `c` is the constant vector `c`.
    fn flat(n: usize, d: usize, m: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
        let ds = d / m;
        let centroids: Vec<f32> = (0..m * k * ds).map(|i| ((i / ds) % k) as f32).collect();
        let corpus = vec![0f32; n * d];
        (corpus, centroids)
    }

    #[test]
    fn recall_counts_only_the_first_k_of_truth() {
        let candidates = vec![1u32, 2, 3, 9];
        let truth = vec![1u32, 5, 3, 7, 2];
        // Of truth[..3] = [1,5,3], two are present.
        assert!((recall_at_k(&candidates, &truth, 3) - 2.0 / 3.0).abs() < 1e-6);
        // Of all five, three are present.
        assert!((recall_at_k(&candidates, &truth, 5) - 3.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn ties_break_deterministically_by_id() {
        // Irreproducible ordering makes a recall measurement meaningless.
        let (corpus, centroids) = flat(4, 2, 1, 1);
        let codes = vec![0u8; 4];
        let data = Encoded {
            corpus: &corpus,
            n: 4,
            d: 2,
            codes: &codes,
            m: 1,
            centroids: &centroids,
            k: 1,
        };
        let q = vec![1.0f32, 1.0];
        assert_eq!(top_ids(data, &q, 4), vec![0, 1, 2, 3]);
        assert_eq!(top_ids(data, &q, 2), vec![0, 1]);
    }

    #[test]
    fn the_candidate_list_is_not_the_ground_truth() {
        // The defect this separation exists to prevent: if stage one scored
        // exactly, clean recall would be 1.0 and every measured loss would be
        // charged against an oracle no device achieves.
        //
        // Six vectors on two centroids. Quantization collapses each pair onto
        // a shared reconstruction, so the approximate top-2 admits a vector the
        // exact ranking places fourth.
        let corpus: Vec<f32> = vec![
            10.0, 0.0, // 0: best exactly
            1.0, 0.0, // 1: shares centroid 0 with vector 0
            9.0, 0.0, // 2: second exactly
            8.0, 0.0, // 3
            2.0, 0.0, // 4
            0.0, 0.0, // 5
        ];
        // Centroid 0 sits high, centroid 1 low.
        let centroids = vec![9.0f32, 0.0, 1.0, 0.0];
        // Vectors 0 and 1 both quantize to centroid 0 despite differing by 9.
        let codes = vec![0u8, 0, 1, 1, 1, 1];
        let data = Encoded {
            corpus: &corpus,
            n: 6,
            d: 2,
            codes: &codes,
            m: 1,
            centroids: &centroids,
            k: 2,
        };
        let q = vec![1.0f32, 0.0];

        let exact = exact_top_ids(data, &q, 6);
        let approx = top_ids(data, &q, 2);
        assert_eq!(exact[..2], [0, 2], "exact ranking");
        assert_eq!(approx, vec![0, 1], "stage one promotes a poor vector");

        // Clean two-stage recall is therefore below 1: vector 2 is a true
        // neighbour the candidate list never offers.
        let clean_recall = recall_at_k(&approx, &exact, 2);
        assert!(
            clean_recall < 1.0,
            "clean recall was {clean_recall}; an oracle baseline hides real loss"
        );
    }

    #[test]
    fn a_shift_moves_only_vectors_using_the_centroid() {
        let corpus: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let centroids = vec![0.0f32, 0.0, 6.0, 6.0];
        let codes = vec![0u8, 1, 0, 1];
        let data = Encoded {
            corpus: &corpus,
            n: 4,
            d: 2,
            codes: &codes,
            m: 1,
            centroids: &centroids,
            k: 2,
        };
        let q = vec![1.0f32, 0.0];
        let clean = top_ids(data, &q, 4);
        let shifted = top_ids_shifted(data, &q, 4, 0, 1, -1000.0);
        // Vectors 1 and 3 use centroid 1, so a large negative shift sends both
        // to the bottom while 0 and 2 keep their relative order.
        assert_eq!(&shifted[2..], &[1, 3]);
        assert_ne!(clean, shifted);
    }

    #[test]
    fn resolution_is_one_over_the_measurement_count() {
        assert!((resolution(100, 10) - 0.001).abs() < 1e-9);
        assert_eq!(resolution(0, 10), 0.0);
    }

    #[test]
    fn mean_recall_handles_an_empty_set() {
        assert_eq!(mean_recall(&[]), 0.0);
        assert!((mean_recall(&[0.5, 1.0]) - 0.75).abs() < 1e-6);
    }
}
