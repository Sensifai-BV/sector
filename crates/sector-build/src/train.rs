//! PQ codebook training.
//!
//! Runs on the host, in floating point, with a heap. Splits `D` into `m`
//! subspaces and trains `2^b` centroids per subspace.
//!
//! # Implementation notes
//!
//! Train in f32 and quantize to bounded fixed point as a separate later stage,
//! then re-measure recall. The quantized codebook is what ships, so a figure
//! taken on the f32 codebook is not the figure the device produces.
//!
//! k-means++ initialisation with a fixed seed. The corruption experiments
//! compare recall across builds, and an unseeded build makes those comparisons
//! meaningless.
//!
//! Report per-subspace quantization error. A subspace with visibly worse error
//! indicates the rotation is not spreading energy as intended — a rotation
//! problem rather than a training one, and diagnosable only if the per-subspace
//! numbers exist.

use crate::dataset::DatasetError;

/// A trained codebook for one subspace, in f32.
///
/// Training runs in floating point on the host. Quantization to bounded fixed
/// point is a separate later stage, and recall is re-measured after it: the
/// quantized codebook is what ships, so a figure taken here is not the figure
/// the device produces.
#[derive(Clone, Debug)]
pub struct SubspaceCodebook {
    /// Centroids, row-major, `centroids * ds`.
    pub centroids: Vec<f32>,
    /// Centroid count, `2^b`.
    pub k: usize,
    /// Subspace dimension, `D / m`.
    pub ds: usize,
}

impl SubspaceCodebook {
    /// Centroid `c`.
    pub fn centroid(&self, c: usize) -> Option<&[f32]> {
        self.centroids.get(c * self.ds..(c + 1) * self.ds)
    }

    /// Index of the nearest centroid to `v`, and its squared distance.
    pub fn nearest(&self, v: &[f32]) -> (usize, f32) {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..self.k {
            let Some(row) = self.centroid(c) else {
                continue;
            };
            let mut d = 0f32;
            for (a, b) in v.iter().zip(row.iter()) {
                let diff = a - b;
                d += diff * diff;
            }
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        (best, best_d)
    }
}

/// Training configuration.
#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    /// Vector dimension.
    pub d: usize,
    /// Subspaces.
    pub m: usize,
    /// Bits per code; `2^b` centroids.
    pub b: usize,
    /// Lloyd iterations.
    pub iterations: usize,
    /// Seed for k-means++ initialisation.
    ///
    /// Fixed, because the corruption experiments compare recall across builds
    /// and an unseeded build makes those comparisons meaningless.
    pub seed: u64,
}

impl TrainConfig {
    /// Subspace dimension.
    pub const fn ds(&self) -> usize {
        self.d / self.m
    }
    /// Centroids per subspace.
    pub const fn centroids(&self) -> usize {
        1 << self.b
    }
}

/// Per-subspace quantization error after training.
///
/// Reported per subspace rather than as a total: a subspace with visibly worse
/// error indicates the rotation is not spreading energy as intended, which is a
/// rotation problem rather than a training one, and diagnosable only if the
/// per-subspace numbers exist.
#[derive(Clone, Debug, Default)]
pub struct TrainReport {
    /// Mean squared quantization error per subspace.
    pub subspace_mse: Vec<f32>,
    /// Lloyd iterations actually run before convergence.
    pub iterations_run: Vec<usize>,
}

impl TrainReport {
    /// Mean over subspaces.
    pub fn mean_mse(&self) -> f32 {
        if self.subspace_mse.is_empty() {
            return 0.0;
        }
        self.subspace_mse.iter().sum::<f32>() / self.subspace_mse.len() as f32
    }

    /// Ratio of worst to mean subspace error.
    ///
    /// A large value points at the rotation, not the training.
    pub fn imbalance(&self) -> f32 {
        let mean = self.mean_mse();
        if mean <= 0.0 {
            return 1.0;
        }
        let worst = self
            .subspace_mse
            .iter()
            .copied()
            .fold(0f32, |a, b| if b > a { b } else { a });
        worst / mean
    }
}

/// Deterministic PRNG for k-means++ seeding.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Train `m` subspace codebooks over `corpus`, row-major `n * d`.
pub fn train(
    corpus: &[f32],
    n: usize,
    config: TrainConfig,
) -> Result<(Vec<SubspaceCodebook>, TrainReport), DatasetError> {
    let ds = config.ds();
    let k = config.centroids();
    let mut books = Vec::with_capacity(config.m);
    let mut report = TrainReport::default();

    for j in 0..config.m {
        // Gather this subspace's slice of every vector.
        let mut sub = vec![0f32; n * ds];
        for v in 0..n {
            let src = corpus
                .get(v * config.d + j * ds..v * config.d + (j + 1) * ds)
                .unwrap_or(&[]);
            if let Some(dst) = sub.get_mut(v * ds..(v + 1) * ds) {
                dst[..src.len()].copy_from_slice(src);
            }
        }

        let mut rng = Rng::new(config.seed.wrapping_add(j as u64));
        let centroids = kmeans_pp_init(&sub, n, ds, k, &mut rng);
        let (centroids, iters, mse) = lloyd(&sub, n, ds, k, centroids, config.iterations);

        report.subspace_mse.push(mse);
        report.iterations_run.push(iters);
        books.push(SubspaceCodebook { centroids, k, ds });
    }

    Ok((books, report))
}

/// k-means++ initialisation: seed centroids proportional to squared distance.
fn kmeans_pp_init(sub: &[f32], n: usize, ds: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut centroids = vec![0f32; k * ds];
    if n == 0 {
        return centroids;
    }

    let first = (rng.next_u64() % n as u64) as usize;
    if let (Some(dst), Some(src)) = (
        centroids.get_mut(0..ds),
        sub.get(first * ds..(first + 1) * ds),
    ) {
        dst.copy_from_slice(src);
    }

    let mut best = vec![f32::INFINITY; n];
    for c in 1..k {
        let mut total = 0f32;
        for v in 0..n {
            let Some(point) = sub.get(v * ds..(v + 1) * ds) else {
                continue;
            };
            let Some(prev) = centroids.get((c - 1) * ds..c * ds) else {
                continue;
            };
            let mut d = 0f32;
            for (a, b) in point.iter().zip(prev.iter()) {
                let diff = a - b;
                d += diff * diff;
            }
            if let Some(slot) = best.get_mut(v) {
                if d < *slot {
                    *slot = d;
                }
                total += *slot;
            }
        }

        // Sample proportional to squared distance; fall back to uniform when
        // every point coincides with a centroid.
        let target = rng.unit() * total;
        let mut acc = 0f32;
        let mut chosen = (rng.next_u64() % n as u64) as usize;
        if total > 0.0 {
            for (v, d) in best.iter().enumerate() {
                acc += d;
                if acc >= target {
                    chosen = v;
                    break;
                }
            }
        }
        if let (Some(dst), Some(src)) = (
            centroids.get_mut(c * ds..(c + 1) * ds),
            sub.get(chosen * ds..(chosen + 1) * ds),
        ) {
            dst.copy_from_slice(src);
        }
    }
    centroids
}

/// Lloyd iterations. Returns the centroids, iterations run, and final MSE.
fn lloyd(
    sub: &[f32],
    n: usize,
    ds: usize,
    k: usize,
    mut centroids: Vec<f32>,
    max_iters: usize,
) -> (Vec<f32>, usize, f32) {
    let mut assign = vec![0usize; n];
    let mut sums = vec![0f32; k * ds];
    let mut counts = vec![0u32; k];
    let mut mse = 0f32;
    let mut run = 0usize;

    for iter in 0..max_iters {
        run = iter + 1;
        let mut moved = 0usize;
        sums.iter_mut().for_each(|s| *s = 0.0);
        counts.iter_mut().for_each(|c| *c = 0);
        mse = 0.0;

        for v in 0..n {
            let Some(point) = sub.get(v * ds..(v + 1) * ds) else {
                continue;
            };
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let Some(row) = centroids.get(c * ds..(c + 1) * ds) else {
                    continue;
                };
                let mut d = 0f32;
                for (a, b) in point.iter().zip(row.iter()) {
                    let diff = a - b;
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if assign.get(v).copied() != Some(best) {
                moved += 1;
            }
            if let Some(a) = assign.get_mut(v) {
                *a = best;
            }
            mse += best_d;
            if let Some(c) = counts.get_mut(best) {
                *c += 1;
            }
            if let Some(dst) = sums.get_mut(best * ds..(best + 1) * ds) {
                for (s, p) in dst.iter_mut().zip(point.iter()) {
                    *s += *p;
                }
            }
        }
        if n > 0 {
            mse /= n as f32;
        }

        // Move each centroid to its members' mean. An empty cluster keeps its
        // position rather than being reseeded: a reseed changes the codebook
        // between otherwise-identical builds and breaks build-to-build
        // comparison.
        for c in 0..k {
            let count = counts.get(c).copied().unwrap_or(0);
            if count == 0 {
                continue;
            }
            if let (Some(dst), Some(src)) = (
                centroids.get_mut(c * ds..(c + 1) * ds),
                sums.get(c * ds..(c + 1) * ds),
            ) {
                for (d, s) in dst.iter_mut().zip(src.iter()) {
                    *d = s / count as f32;
                }
            }
        }

        if moved == 0 {
            break;
        }
    }
    (centroids, run, mse)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three well-separated clusters in 2-D, repeated across `m` subspaces.
    fn clustered(n: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0f32; n * d];
        for v in 0..n {
            let cluster = v % 3;
            for j in 0..d {
                let base = match cluster {
                    0 => 0.0,
                    1 => 50.0,
                    _ => 100.0,
                };
                out[v * d + j] = base + ((v * 7 + j * 13) % 5) as f32;
            }
        }
        out
    }

    fn config() -> TrainConfig {
        TrainConfig {
            d: 8,
            m: 2,
            b: 2,
            iterations: 25,
            seed: 42,
        }
    }

    #[test]
    fn training_is_deterministic_under_a_fixed_seed() {
        // Corruption experiments compare recall across builds; an unseeded
        // build makes those comparisons meaningless.
        let corpus = clustered(120, 8);
        let (a, _) = train(&corpus, 120, config()).unwrap();
        let (b, _) = train(&corpus, 120, config()).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.centroids, y.centroids);
        }

        let other = TrainConfig {
            seed: 43,
            ..config()
        };
        let (c, _) = train(&corpus, 120, other).unwrap();
        assert_ne!(a[0].centroids, c[0].centroids, "seed must matter");
    }

    #[test]
    fn training_separates_well_separated_clusters() {
        let corpus = clustered(150, 8);
        let (books, report) = train(&corpus, 150, config()).unwrap();
        assert_eq!(books.len(), 2);
        assert_eq!(books[0].k, 4);
        assert_eq!(books[0].ds, 4);

        // Three clusters 50 apart; residual error must be far below that.
        assert!(
            report.mean_mse() < 100.0,
            "mse {} too high for separated clusters",
            report.mean_mse()
        );
    }

    #[test]
    fn every_vector_maps_to_a_centroid_within_range() {
        let corpus = clustered(60, 8);
        let (books, _) = train(&corpus, 60, config()).unwrap();
        for v in 0..60 {
            for (j, book) in books.iter().enumerate() {
                let slice = &corpus[v * 8 + j * 4..v * 8 + (j + 1) * 4];
                let (c, d) = book.nearest(slice);
                assert!(c < book.k);
                assert!(d.is_finite());
            }
        }
    }

    #[test]
    fn per_subspace_error_is_reported_not_only_a_total() {
        // A subspace with visibly worse error points at the rotation rather
        // than the training, and is diagnosable only if the numbers exist.
        let corpus = clustered(90, 8);
        let (_, report) = train(&corpus, 90, config()).unwrap();
        assert_eq!(report.subspace_mse.len(), 2);
        assert_eq!(report.iterations_run.len(), 2);
        assert!(report.imbalance() >= 1.0);
    }

    #[test]
    fn lloyd_converges_before_its_iteration_cap() {
        // Well-separated clusters should converge quickly; hitting the cap
        // means the codebook shipped is not the one the algorithm intended.
        let corpus = clustered(120, 8);
        let cfg = TrainConfig {
            iterations: 50,
            ..config()
        };
        let (_, report) = train(&corpus, 120, cfg).unwrap();
        for (j, iters) in report.iterations_run.iter().enumerate() {
            assert!(*iters < 50, "subspace {j} did not converge: {iters}");
        }
    }

    #[test]
    fn more_centroids_reduce_quantization_error() {
        // The relationship the whole configuration argument depends on: recall
        // buys with b, and b costs codebook bytes.
        let corpus = clustered(200, 8);
        let (_, coarse) = train(&corpus, 200, TrainConfig { b: 1, ..config() }).unwrap();
        let (_, fine) = train(&corpus, 200, TrainConfig { b: 3, ..config() }).unwrap();
        assert!(
            fine.mean_mse() <= coarse.mean_mse(),
            "b=3 mse {} exceeded b=1 mse {}",
            fine.mean_mse(),
            coarse.mean_mse()
        );
    }

    #[test]
    fn an_empty_cluster_keeps_its_position_rather_than_reseeding() {
        // Reseeding changes the codebook between otherwise-identical builds.
        // With more centroids than distinct points, some must stay empty.
        let corpus = clustered(6, 8);
        let cfg = TrainConfig { b: 3, ..config() };
        let (a, _) = train(&corpus, 6, cfg).unwrap();
        let (b, _) = train(&corpus, 6, cfg).unwrap();
        assert_eq!(a[0].centroids, b[0].centroids);
    }
}
