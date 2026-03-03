//! Dataset loading shared by the subcommands.
//!
//! Loads stream through `sector-build`'s readers. GIST1M's base file is 5.4 GB
//! on disk, so a subcommand that needs only a subset must not pay for the whole
//! file.

use sector_build::dataset::VecsReader;
use std::path::Path;

/// A loaded slice of a dataset.
pub struct Loaded {
    /// Row-major components, `count * dim`.
    pub data: Vec<f32>,
    /// Vectors loaded.
    pub count: usize,
    /// Components per vector.
    pub dim: usize,
    /// Vectors the file holds in total.
    ///
    /// Kept because a subset invalidates the shipped ground truth, which
    /// indexes the full corpus — reporting recall against it anyway would give
    /// a number that looks plausible and means nothing.
    pub file_count: usize,
}

/// Load up to `n` vectors from `path`. `n = 0` loads the file.
pub fn load(path: &Path, n: usize) -> Result<Loaded, String> {
    let mut reader = VecsReader::open(path).map_err(|e| format!("{}: {e:?}", path.display()))?;
    let dim = reader.layout().dim as usize;
    let file_count = reader.len();
    let count = if n == 0 {
        file_count
    } else {
        n.min(file_count)
    };

    let mut data = vec![0f32; count * dim];
    let mut row = vec![0f32; dim];
    for v in 0..count {
        match reader.next_f32(&mut row).map_err(|e| format!("{e:?}"))? {
            Some(_) => data[v * dim..(v + 1) * dim].copy_from_slice(&row),
            None => {
                return Err(format!(
                    "{} ended at vector {v}, expected {count}",
                    path.display()
                ))
            }
        }
    }
    Ok(Loaded {
        data,
        count,
        dim,
        file_count,
    })
}

/// Exact top-`r` by **L2 distance**.
///
/// SIFT and GIST ground truth is L2. Scoring by inner product against an L2
/// ground truth reports a recall that is wrong for a reason no amount of tuning
/// would reveal, so the metric is fixed here rather than chosen per call.
pub fn exact_top_l2(base: &[f32], n: usize, d: usize, q: &[f32], r: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..n)
        .map(|v| {
            let row = &base[v * d..(v + 1) * d];
            let mut acc = 0f32;
            for (a, b) in row.iter().zip(q.iter()) {
                let diff = a - b;
                acc += diff * diff;
            }
            (acc, v as u32)
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.into_iter().take(r).map(|(_, id)| id).collect()
}

/// Fraction of the first `k` of `truth` present in `candidates`.
pub fn recall_at(candidates: &[u32], truth: &[u32], k: usize) -> f64 {
    if k == 0 || truth.is_empty() {
        return 0.0;
    }
    let take = k.min(truth.len());
    let hits = truth
        .iter()
        .take(take)
        .filter(|t| candidates.contains(t))
        .count();
    hits as f64 / take as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_counts_only_the_first_k_of_truth() {
        let candidates = vec![1u32, 2, 3, 9];
        let truth = vec![1u32, 5, 3, 7, 2];
        assert!((recall_at(&candidates, &truth, 3) - 2.0 / 3.0).abs() < 1e-9);
        assert!((recall_at(&candidates, &truth, 5) - 3.0 / 5.0).abs() < 1e-9);
        assert_eq!(recall_at(&candidates, &truth, 0), 0.0);
        assert_eq!(recall_at(&candidates, &[], 10), 0.0);
    }

    #[test]
    fn exact_top_uses_l2_not_inner_product() {
        // With inner product the far vector [5,5] would win; by L2 it loses.
        let base = vec![0.0, 0.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0];
        let q = vec![1.0f32, 0.0];
        let top = exact_top_l2(&base, 4, 2, &q, 4);
        assert_eq!(top[0], 2, "nearest by L2 is the identical vector");
        assert_eq!(top[3], 1, "farthest by L2 would be first by inner product");
    }

    #[test]
    fn exact_top_breaks_ties_by_id() {
        // Vectors 0 and 3 are identical; irreproducible ordering would make a
        // recall measurement non-repeatable.
        let base = vec![0.0, 0.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0];
        let q = vec![1.0f32, 0.0];
        let top = exact_top_l2(&base, 4, 2, &q, 4);
        assert_eq!(&top[1..3], &[0, 3]);
    }
}
