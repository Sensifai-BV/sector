//! Corpus encoding to PQ codes.
//!
//! Assigns each vector's subvectors to their nearest centroids, producing the
//! `pi`-byte payload the device scans.
//!
//! # Ordering
//!
//! Record per-centroid populations `n_{j,c}` during encoding rather than in a
//! second pass; they are the criticality input and they are free here. The
//! distribution is materially non-uniform — measured maximum 4.5x the mean,
//! 99th percentile 2.3x, top decile carrying ~19% of references at m=32, b=8,
//! N=20,000 — and assuming uniformity discards the skew the allocation
//! exploits.
//!
//! Encode after rotation and after label optimisation, in that order. Encoding
//! against pre-permutation labels and then permuting only the codebook breaks
//! the losslessness property, producing an image that is structurally valid and
//! reconstructs every vector wrongly.

use crate::train::SubspaceCodebook;
use sector_quant::codebook::Scale;

/// Per-centroid reference counts, `m * k`, row-major by subspace.
///
/// Recorded during encoding rather than in a second pass: they are the
/// criticality input and they are free here.
#[derive(Clone, Debug)]
pub struct Populations {
    /// Counts, `m * k`.
    pub counts: Vec<u32>,
    /// Subspaces.
    pub m: usize,
    /// Centroids per subspace.
    pub k: usize,
}

impl Populations {
    /// Count for centroid `c` of subspace `j`.
    pub fn get(&self, j: usize, c: usize) -> u32 {
        self.counts.get(j * self.k + c).copied().unwrap_or(0)
    }

    /// Mean references per centroid, scaled by 1024.
    ///
    /// Integer-scaled so the figure is comparable across builds without a
    /// float in the record.
    pub fn mean_x1024(&self) -> u64 {
        if self.counts.is_empty() {
            return 0;
        }
        let total: u64 = self.counts.iter().map(|c| *c as u64).sum();
        (total * 1024) / self.counts.len() as u64
    }

    /// Largest count.
    pub fn max(&self) -> u32 {
        self.counts.iter().copied().max().unwrap_or(0)
    }

    /// Ratio of maximum to mean, scaled by 1024.
    ///
    /// The skew the allocation exploits. Assuming uniformity discards it.
    pub fn skew_x1024(&self) -> u64 {
        let mean = self.mean_x1024();
        if mean == 0 {
            return 0;
        }
        (self.max() as u64 * 1024 * 1024) / mean
    }

    /// Share of all references held by the top `decile_count` centroids of
    /// subspace `j`, in parts per million.
    pub fn top_share_ppm(&self, j: usize, decile_count: usize) -> u64 {
        let Some(row) = self.counts.get(j * self.k..(j + 1) * self.k) else {
            return 0;
        };
        let total: u64 = row.iter().map(|c| *c as u64).sum();
        if total == 0 {
            return 0;
        }
        let mut sorted: Vec<u32> = row.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let top: u64 = sorted.iter().take(decile_count).map(|c| *c as u64).sum();
        (top * 1_000_000) / total
    }
}

/// A quantized codebook ready to ship, with the scale that reconstructs it.
#[derive(Clone, Debug)]
pub struct QuantizedCodebook {
    /// Components as `i8`, `m * k * ds`.
    pub components: Vec<i8>,
    /// Per-subspace scale.
    pub scales: Vec<Scale>,
    /// Subspaces.
    pub m: usize,
    /// Centroids per subspace.
    pub k: usize,
    /// Subspace dimension.
    pub ds: usize,
}

impl QuantizedCodebook {
    /// Components of centroid `c` in subspace `j`.
    pub fn centroid(&self, j: usize, c: usize) -> Option<&[i8]> {
        let start = (j * self.k + c) * self.ds;
        self.components.get(start..start + self.ds)
    }

    /// Reconstruct a component as an f32, for error measurement on the host.
    pub fn dequantize(&self, j: usize, value: i8) -> f32 {
        match self.scales.get(j) {
            Some(s) => (value as f32) * (s.num as f32) / (s.den as f32),
            None => 0.0,
        }
    }

    /// Stored bytes: `m * k * ds`, independent of `N`.
    pub const fn byte_len(&self) -> usize {
        self.m * self.k * self.ds
    }
}

/// Quantize f32 codebooks to bounded fixed point.
///
/// The scale per subspace maps its largest magnitude onto `i8::MAX`, so the
/// full range is used and the displacement bound is as tight as the format
/// allows. `den` is fixed at 127 and `num` carries the subspace's extent, both
/// integers: the device reconstructs with no float, so host and device produce
/// identical bytes.
pub fn quantize(books: &[SubspaceCodebook], den: i32) -> QuantizedCodebook {
    let m = books.len();
    let k = books.first().map(|b| b.k).unwrap_or(0);
    let ds = books.first().map(|b| b.ds).unwrap_or(0);
    let mut components = vec![0i8; m * k * ds];
    let mut scales = Vec::with_capacity(m);

    for (j, book) in books.iter().enumerate() {
        let extent = book
            .centroids
            .iter()
            .fold(0f32, |acc, v| if v.abs() > acc { v.abs() } else { acc });
        // A degenerate subspace (all zeros) still needs a valid scale.
        let num = if extent > 0.0 {
            extent.ceil().max(1.0) as i32
        } else {
            // A degenerate all-zero subspace still needs a usable scale.
            1
        };
        scales.push(Scale::new(num, den).unwrap_or(Scale { num: 1, den }));

        for c in 0..k {
            let Some(row) = book.centroid(c) else {
                continue;
            };
            for (i, v) in row.iter().enumerate() {
                let scaled = (v / num as f32) * den as f32;
                let clamped = scaled.round().clamp(i8::MIN as f32, i8::MAX as f32);
                if let Some(slot) = components.get_mut((j * k + c) * ds + i) {
                    *slot = clamped as i8;
                }
            }
        }
    }

    QuantizedCodebook {
        components,
        scales,
        m,
        k,
        ds,
    }
}

/// Encode `corpus` to PQ codes, recording per-centroid populations.
///
/// Encoding happens after rotation and after label optimisation, in that order.
/// Encoding against pre-permutation labels and then permuting only the codebook
/// breaks the losslessness property, producing an image that is structurally
/// valid and reconstructs every vector wrongly.
pub fn encode(
    corpus: &[f32],
    n: usize,
    d: usize,
    books: &[SubspaceCodebook],
) -> (Vec<u8>, Populations) {
    let m = books.len();
    let k = books.first().map(|b| b.k).unwrap_or(0);
    let ds = d.checked_div(m).unwrap_or(0);
    let mut codes = vec![0u8; n * m];
    let mut counts = vec![0u32; m * k];

    for v in 0..n {
        for (j, book) in books.iter().enumerate() {
            let Some(slice) = corpus.get(v * d + j * ds..v * d + (j + 1) * ds) else {
                continue;
            };
            let (c, _) = book.nearest(slice);
            if let Some(slot) = codes.get_mut(v * m + j) {
                *slot = c as u8;
            }
            if let Some(count) = counts.get_mut(j * k + c) {
                *count += 1;
            }
        }
    }

    (codes, Populations { counts, m, k })
}

/// Mean squared reconstruction error of `codes` against `corpus`.
///
/// Measured on the *quantized* codebook, because that is what ships. A figure
/// taken on the f32 codebook is not the figure the device produces.
pub fn reconstruction_mse(
    corpus: &[f32],
    n: usize,
    d: usize,
    codes: &[u8],
    quantized: &QuantizedCodebook,
) -> f32 {
    if n == 0 || d == 0 {
        return 0.0;
    }
    let m = quantized.m;
    let ds = quantized.ds;
    let mut total = 0f64;

    for v in 0..n {
        for j in 0..m {
            let Some(&code) = codes.get(v * m + j) else {
                continue;
            };
            let Some(row) = quantized.centroid(j, code as usize) else {
                continue;
            };
            for (i, stored) in row.iter().enumerate() {
                let original = corpus.get(v * d + j * ds + i).copied().unwrap_or(0.0);
                let reconstructed = quantized.dequantize(j, *stored);
                let diff = (original - reconstructed) as f64;
                total += diff * diff;
            }
        }
    }
    (total / (n * d) as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{train, TrainConfig};

    fn skewed(n: usize, d: usize) -> Vec<f32> {
        // Deliberately unequal cluster sizes, so populations are non-uniform.
        let mut out = vec![0f32; n * d];
        for v in 0..n {
            let cluster = if v % 10 < 6 {
                0
            } else if v % 10 < 9 {
                1
            } else {
                2
            };
            for j in 0..d {
                out[v * d + j] = (cluster as f32) * 40.0 + ((v + j) % 3) as f32;
            }
        }
        out
    }

    fn config() -> TrainConfig {
        TrainConfig {
            d: 8,
            m: 2,
            b: 2,
            iterations: 30,
            seed: 11,
        }
    }

    #[test]
    fn populations_are_recorded_during_encoding() {
        let corpus = skewed(200, 8);
        let (books, _) = train(&corpus, 200, config()).unwrap();
        let (codes, pops) = encode(&corpus, 200, 8, &books);

        assert_eq!(codes.len(), 200 * 2);
        // Every vector contributes exactly one reference per subspace.
        for j in 0..2 {
            let total: u32 = (0..4).map(|c| pops.get(j, c)).sum();
            assert_eq!(total, 200, "subspace {j} population total");
        }
    }

    #[test]
    fn the_population_distribution_is_measurably_skewed() {
        // Assuming uniformity discards the skew the allocation exploits.
        let corpus = skewed(500, 8);
        let (books, _) = train(&corpus, 500, config()).unwrap();
        let (_, pops) = encode(&corpus, 500, 8, &books);

        assert!(
            pops.skew_x1024() > 1024,
            "max/mean ratio was {} (x1024), expected skew",
            pops.skew_x1024()
        );
        // The top centroid of subspace 0 holds a disproportionate share.
        assert!(pops.top_share_ppm(0, 1) > 250_000);
    }

    #[test]
    fn encoding_assigns_every_vector_to_its_nearest_centroid() {
        let corpus = skewed(100, 8);
        let (books, _) = train(&corpus, 100, config()).unwrap();
        let (codes, _) = encode(&corpus, 100, 8, &books);
        for v in 0..100 {
            for (j, book) in books.iter().enumerate() {
                let slice = &corpus[v * 8 + j * 4..v * 8 + (j + 1) * 4];
                let (expected, _) = book.nearest(slice);
                assert_eq!(codes[v * 2 + j] as usize, expected);
            }
        }
    }

    #[test]
    fn quantization_uses_the_full_i8_range() {
        // A scale leaving the range unused would widen the displacement bound
        // for nothing.
        let corpus = skewed(200, 8);
        let (books, _) = train(&corpus, 200, config()).unwrap();
        let q = quantize(&books, 127);

        let extreme = q
            .components
            .iter()
            .map(|c| (*c as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(
            extreme > 100,
            "largest component was {extreme}, range unused"
        );
        assert_eq!(q.byte_len(), 2 * 4 * 4);
    }

    #[test]
    fn quantization_error_is_measured_on_the_shipped_codebook() {
        // The quantized codebook is what the device holds, so recall must be
        // re-measured after this stage rather than inherited from training.
        //
        // The two errors are normalised differently and are not comparable as
        // reported: `TrainReport::mean_mse` is squared distance per vector per
        // subspace (summed over `ds` components), while `reconstruction_mse` is
        // per component. Dividing the first by `ds` puts them on one basis.
        let corpus = skewed(300, 8);
        let (books, report) = train(&corpus, 300, config()).unwrap();
        let (codes, _) = encode(&corpus, 300, 8, &books);
        let q = quantize(&books, 127);

        let quantized = reconstruction_mse(&corpus, 300, 8, &codes, &q);
        let trained_per_component = report.mean_mse() / q.ds as f32;
        assert!(quantized.is_finite());
        assert!(
            quantized >= trained_per_component,
            "quantization reduced error: {quantized} < {trained_per_component}"
        );
        // And it does not inflate it wildly: rounding to i8 at full range costs
        // a fraction of the residual, not a multiple of it.
        assert!(
            quantized < trained_per_component * 2.0,
            "quantization cost too much: {quantized} vs {trained_per_component}"
        );
    }

    #[test]
    fn quantization_is_deterministic() {
        let corpus = skewed(150, 8);
        let (books, _) = train(&corpus, 150, config()).unwrap();
        let a = quantize(&books, 127);
        let b = quantize(&books, 127);
        assert_eq!(a.components, b.components);
        for (x, y) in a.scales.iter().zip(b.scales.iter()) {
            assert_eq!(x.num, y.num);
            assert_eq!(x.den, y.den);
        }
    }

    #[test]
    fn a_degenerate_subspace_still_gets_a_valid_scale() {
        // An all-zero subspace would give a zero scale and a division by zero
        // at reconstruction.
        let books = vec![SubspaceCodebook {
            centroids: vec![0f32; 16],
            k: 4,
            ds: 4,
        }];
        let q = quantize(&books, 127);
        assert_eq!(q.scales[0].den, 127);
        assert!(q.scales[0].num != 0);
        assert_eq!(q.dequantize(0, 0), 0.0);
    }

    #[test]
    fn codebook_size_is_independent_of_corpus_size() {
        // The property the whole protection argument rests on: the codebook is
        // 2^b * D * s regardless of N, so a replica costs a fixed number of
        // bytes.
        let small = skewed(50, 8);
        let large = skewed(500, 8);
        let (b1, _) = train(&small, 50, config()).unwrap();
        let (b2, _) = train(&large, 500, config()).unwrap();
        assert_eq!(quantize(&b1, 127).byte_len(), quantize(&b2, 127).byte_len());
    }
}
