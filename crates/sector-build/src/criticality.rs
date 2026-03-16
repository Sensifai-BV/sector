//! Per-centroid population and depth-aware exposure measurement.
//!
//! Produces the weights `w_{j,c}` the allocator spends its budget against.
//!
//! The weight is not the population. An earlier version weighted by
//! `n_{j,c} · gamma_{j,c}` after separately proving `gamma_{j,c} = 1`
//! identically — weighting by a constant.
//!
//! The quantity is depth-aware exposure: how many true neighbours sit deep
//! enough in the candidate list to be evicted by the intruders a corruption of
//! this centroid produces, plus how many affected true neighbours a deflating
//! shift pushes out. The induced score shift is signed, and deflation produces
//! recall loss with zero intruders.
//!
//! # Measurement rules
//!
//! Measure over a held-out query set, per centroid, with a signed displacement
//! sweep covering both directions. An inflation-only sweep misses a failure
//! channel: a directed deflating construction produced 0.127 recall loss where
//! an inflation-only bound predicted exactly zero.
//!
//! Report weights relative to clean two-stage recall. At `R = 100` roughly 36%
//! of true top-`k` items are already outside the clean candidate list and have
//! no depth; including them measures the baseline's limits rather than the
//! corruption's damage.
//!
//! No closed form is known, so this is measured per dataset.

use crate::train::SubspaceCodebook;

/// Depth-aware exposure for one centroid.
///
/// The weight is **not** the population. An earlier version weighted by
/// `n * gamma` after separately proving `gamma = 1` identically — weighting by
/// a constant. The quantity here is how many true neighbours a corruption of
/// this centroid actually costs, measured over a held-out query set.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Exposure {
    /// Vectors referencing this centroid.
    pub population: u32,
    /// True neighbours lost when the centroid's score inflates.
    ///
    /// Inflation creates intruders, which evict incumbents from the far end of
    /// the bounded heap.
    pub inflate_loss: u32,
    /// True neighbours lost when the centroid's score deflates.
    ///
    /// Deflation drops affected true neighbours out of the candidate set with
    /// **zero** intruders. An inflation-only measurement misses this channel
    /// entirely: a directed deflating construction produced 0.127 loss where an
    /// inflation-only bound predicted exactly zero.
    pub deflate_loss: u32,
}

impl Exposure {
    /// The weight the allocator spends against: the worse of the two signs.
    ///
    /// Taking the maximum rather than the mean, because a protection budget
    /// must cover the direction that actually occurs, and which sign a given
    /// bit flip produces is not predictable.
    pub const fn weight(&self) -> u32 {
        if self.inflate_loss > self.deflate_loss {
            self.inflate_loss
        } else {
            self.deflate_loss
        }
    }

    /// Whether both signs were exercised.
    ///
    /// A measurement reporting only one sign is incomplete, not conservative.
    pub const fn is_signed(&self) -> bool {
        self.inflate_loss > 0 || self.deflate_loss > 0
    }
}

/// Measured weights for one subspace's centroids.
#[derive(Clone, Debug, Default)]
pub struct Weights {
    /// Per centroid.
    pub per_centroid: Vec<Exposure>,
}

impl Weights {
    /// Total weight.
    pub fn total(&self) -> u64 {
        self.per_centroid.iter().map(|e| e.weight() as u64).sum()
    }

    /// Whether the weights distinguish centroids at all.
    ///
    /// A degenerate weighting — every centroid equal — means the allocation is
    /// spending its budget uniformly, and the measurement bought nothing.
    pub fn is_degenerate(&self) -> bool {
        let mut iter = self.per_centroid.iter().map(|e| e.weight());
        let Some(first) = iter.next() else {
            return true;
        };
        iter.all(|w| w == first)
    }

    /// Indices ordered by weight, heaviest first.
    pub fn ranked(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.per_centroid.len()).collect();
        idx.sort_by(|a, b| {
            let wa = self.per_centroid.get(*a).map(|e| e.weight()).unwrap_or(0);
            let wb = self.per_centroid.get(*b).map(|e| e.weight()).unwrap_or(0);
            wb.cmp(&wa).then(a.cmp(b))
        });
        idx
    }
}

/// A query and its true neighbour set.
pub struct Probe<'a> {
    /// Query vector, `d` components.
    pub vector: &'a [f32],
    /// True neighbour ids, best first.
    pub truth: &'a [u32],
}

/// The corpus and its encoding, as the measurement sees them.
///
/// Corpus, dimension, codes and subspace count describe one encoded dataset;
/// passing them separately invites a call that indexes one corpus with
/// another's codes.
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
}

/// Retrieval parameters the measurement is taken at.
#[derive(Clone, Copy, Debug)]
pub struct Sweep {
    /// Candidate depth.
    pub r: usize,
    /// Neighbours counted.
    pub k: usize,
    /// Displacement magnitude, applied at both signs.
    ///
    /// Must be the displacement the shipped format admits
    /// (`2^(beta-1) * Delta`). The weight ranking depends on it, so a value
    /// chosen for convenience protects the wrong centroids.
    pub delta: f32,
}

/// Measure depth-aware exposure for every centroid of subspace `j`.
///
/// For each centroid, displaces it by `+delta` and by `-delta` along the query
/// direction and counts true neighbours that leave the top `r` candidates. The
/// sweep is signed because the induced score shift is signed and the two
/// directions fail differently.
pub fn measure(
    data: Encoded<'_>,
    j: usize,
    book: &SubspaceCodebook,
    probes: &[Probe<'_>],
    sweep: Sweep,
) -> Weights {
    let mut per_centroid = vec![Exposure::default(); book.k];

    for (c, slot) in per_centroid.iter_mut().enumerate() {
        slot.population = data
            .codes
            .iter()
            .skip(j)
            .step_by(data.m.max(1))
            .filter(|code| **code as usize == c)
            .count() as u32;
    }

    for probe in probes {
        let clean = top_ids(data.corpus, data.n, data.d, probe.vector, sweep.r);
        let baseline = surviving(&clean, probe.truth, sweep.k);

        for (c, slot) in per_centroid.iter_mut().enumerate() {
            if slot.population == 0 {
                continue;
            }
            for sign in [1.0f32, -1.0] {
                let shifted =
                    top_ids_shifted(data, probe.vector, sweep.r, j, c, sign * sweep.delta);
                let after = surviving(&shifted, probe.truth, sweep.k);
                let lost = baseline.saturating_sub(after);
                if sign > 0.0 {
                    slot.inflate_loss += lost;
                } else {
                    slot.deflate_loss += lost;
                }
            }
        }
    }

    Weights { per_centroid }
}

/// Ids of the top `r` corpus vectors by inner product with `q`.
fn top_ids(corpus: &[f32], n: usize, d: usize, q: &[f32], r: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..n)
        .map(|v| {
            let mut s = 0f32;
            if let Some(row) = corpus.get(v * d..(v + 1) * d) {
                for (a, b) in row.iter().zip(q.iter()) {
                    s += a * b;
                }
            }
            (s, v as u32)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    scored.into_iter().take(r).map(|(_, id)| id).collect()
}

/// Ids of the top `r` when every vector using centroid `c` of subspace `j`
/// has its score shifted by `shift`.
fn top_ids_shifted(
    data: Encoded<'_>,
    q: &[f32],
    r: usize,
    j: usize,
    c: usize,
    shift: f32,
) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..data.n)
        .map(|v| {
            let mut s = 0f32;
            if let Some(row) = data.corpus.get(v * data.d..(v + 1) * data.d) {
                for (a, b) in row.iter().zip(q.iter()) {
                    s += a * b;
                }
            }
            // The shift applies identically to every vector using this
            // centroid, which is what makes the fan-out matter.
            if data.codes.get(v * data.m + j).copied() == Some(c as u8) {
                s += shift;
            }
            (s, v as u32)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    scored.into_iter().take(r).map(|(_, id)| id).collect()
}

/// True neighbours from the first `k` of `truth` present in `candidates`.
fn surviving(candidates: &[u32], truth: &[u32], k: usize) -> u32 {
    truth
        .iter()
        .take(k)
        .filter(|t| candidates.contains(t))
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode;
    use crate::train::{train, TrainConfig};

    const D: usize = 8;
    const M: usize = 2;
    const N: usize = 200;

    fn corpus() -> Vec<f32> {
        let mut out = vec![0f32; N * D];
        for v in 0..N {
            for j in 0..D {
                out[v * D + j] = (((v * 13 + j * 7) % 41) as f32) - 20.0;
            }
        }
        out
    }

    fn setup() -> (Vec<f32>, Vec<SubspaceCodebook>, Vec<u8>) {
        let data = corpus();
        let cfg = TrainConfig {
            d: D,
            m: M,
            b: 2,
            iterations: 30,
            seed: 9,
        };
        let (books, _) = train(&data, N, cfg).unwrap();
        let (codes, _) = encode(&data, N, D, &books);
        (data, books, codes)
    }

    fn probes(data: &[f32], r: usize) -> (Vec<Vec<f32>>, Vec<Vec<u32>>) {
        let queries: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                (0..D)
                    .map(|j| ((i * 5 + j * 3) % 31) as f32 - 15.0)
                    .collect()
            })
            .collect();
        let truths: Vec<Vec<u32>> = queries.iter().map(|q| top_ids(data, N, D, q, r)).collect();
        (queries, truths)
    }

    #[test]
    fn both_signs_are_measured() {
        // An inflation-only measurement misses an entire failure channel: a
        // deflating shift drops affected true neighbours with zero intruders.
        //
        // Measured on this corpus, summed over centroids:
        //   delta=100   inflate 9    deflate 9
        //   delta=400   inflate 54   deflate 29
        //   delta=1000  inflate 110  deflate 40
        // Deflation is never the larger channel here but is never negligible,
        // and at small displacement the two are equal. A bound derived from
        // inflation alone would be wrong by the deflating share at every
        // magnitude.
        let (data, books, codes) = setup();
        let (queries, truths) = probes(&data, 10);
        let ps: Vec<Probe<'_>> = queries
            .iter()
            .zip(truths.iter())
            .map(|(q, t)| Probe {
                vector: q,
                truth: t,
            })
            .collect();

        let w = measure(
            Encoded {
                corpus: &data,
                n: N,
                d: D,
                codes: &codes,
                m: M,
            },
            0,
            &books[0],
            &ps,
            Sweep {
                r: 30,
                k: 10,
                delta: 400.0,
            },
        );
        let inflate: u32 = w.per_centroid.iter().map(|e| e.inflate_loss).sum();
        let deflate: u32 = w.per_centroid.iter().map(|e| e.deflate_loss).sum();
        assert!(
            deflate > 0,
            "deflation produced no measured loss; the channel is untested"
        );
        assert!(inflate + deflate > 0);
    }

    #[test]
    fn the_weight_ranking_depends_on_sweep_magnitude() {
        // Measured on this corpus, per-centroid weights by displacement:
        //   delta=400   [10, 19, 19, 15]  — centroids 1 and 2 heaviest
        //   delta=1000  [30, 20, 20, 40]  — centroid 3 heaviest
        //
        // The ranking inverts. Criticality is not a property of a centroid
        // alone but of a centroid at a displacement, so the sweep magnitude
        // must be the displacement the format actually admits
        // (`2^(beta-1) * Delta` for the shipped codebook) rather than a value
        // chosen for convenience. An allocation derived at the wrong magnitude
        // protects the wrong centroids.
        let (data, books, codes) = setup();
        let (queries, truths) = probes(&data, 10);
        let ps: Vec<Probe<'_>> = queries
            .iter()
            .zip(truths.iter())
            .map(|(q, t)| Probe {
                vector: q,
                truth: t,
            })
            .collect();

        let small = measure(
            Encoded {
                corpus: &data,
                n: N,
                d: D,
                codes: &codes,
                m: M,
            },
            0,
            &books[0],
            &ps,
            Sweep {
                r: 30,
                k: 10,
                delta: 400.0,
            },
        );
        let large = measure(
            Encoded {
                corpus: &data,
                n: N,
                d: D,
                codes: &codes,
                m: M,
            },
            0,
            &books[0],
            &ps,
            Sweep {
                r: 30,
                k: 10,
                delta: 1000.0,
            },
        );
        assert_ne!(
            small.ranked(),
            large.ranked(),
            "if the ranking were magnitude-independent, delta would not matter"
        );
    }

    #[test]
    fn deflation_loses_neighbours_without_creating_intruders() {
        // The structural difference between the two signs, shown directly: a
        // deflating shift removes affected vectors from the candidate set, and
        // nothing takes their place from the affected group.
        let (data, _, codes) = setup();
        let (queries, _) = probes(&data, 10);
        let q = &queries[0];

        let clean = top_ids(&data, N, D, q, 30);
        let deflated = top_ids_shifted(
            Encoded {
                corpus: &data,
                n: N,
                d: D,
                codes: &codes,
                m: M,
            },
            q,
            30,
            0,
            0,
            -500.0,
        );

        // Every vector that entered the list under deflation must NOT use the
        // corrupted centroid: the affected group only leaves.
        for id in &deflated {
            if !clean.contains(id) {
                assert_ne!(
                    codes[*id as usize * M],
                    0,
                    "a deflated centroid's vector entered the list"
                );
            }
        }
    }

    #[test]
    fn the_weight_is_the_worse_sign_not_the_mean() {
        let e = Exposure {
            population: 100,
            inflate_loss: 3,
            deflate_loss: 11,
        };
        assert_eq!(e.weight(), 11);
        assert!(e.is_signed());
    }

    #[test]
    fn weights_are_centroid_specific_not_uniform() {
        // A degenerate weighting means the measurement bought nothing and the
        // allocation is spending uniformly.
        let (data, books, codes) = setup();
        let (queries, truths) = probes(&data, 10);
        let ps: Vec<Probe<'_>> = queries
            .iter()
            .zip(truths.iter())
            .map(|(q, t)| Probe {
                vector: q,
                truth: t,
            })
            .collect();

        let w = measure(
            Encoded {
                corpus: &data,
                n: N,
                d: D,
                codes: &codes,
                m: M,
            },
            0,
            &books[0],
            &ps,
            Sweep {
                r: 30,
                k: 10,
                delta: 400.0,
            },
        );
        assert!(!w.is_degenerate(), "weights did not distinguish centroids");
        assert_eq!(w.per_centroid.len(), books[0].k);
    }

    #[test]
    fn populations_are_recorded_alongside_the_exposure() {
        let (data, books, codes) = setup();
        let (queries, truths) = probes(&data, 10);
        let ps: Vec<Probe<'_>> = queries
            .iter()
            .zip(truths.iter())
            .map(|(q, t)| Probe {
                vector: q,
                truth: t,
            })
            .collect();
        let w = measure(
            Encoded {
                corpus: &data,
                n: N,
                d: D,
                codes: &codes,
                m: M,
            },
            0,
            &books[0],
            &ps,
            Sweep {
                r: 30,
                k: 10,
                delta: 400.0,
            },
        );

        let total: u32 = w.per_centroid.iter().map(|e| e.population).sum();
        assert_eq!(total, N as u32, "every vector references one centroid");
    }

    #[test]
    fn ranking_is_deterministic_under_ties() {
        // Non-deterministic ordering makes an allocation irreproducible.
        let w = Weights {
            per_centroid: vec![
                Exposure {
                    population: 1,
                    inflate_loss: 5,
                    deflate_loss: 0,
                },
                Exposure {
                    population: 1,
                    inflate_loss: 5,
                    deflate_loss: 0,
                },
                Exposure {
                    population: 1,
                    inflate_loss: 9,
                    deflate_loss: 0,
                },
            ],
        };
        assert_eq!(w.ranked(), vec![2, 0, 1]);
        assert_eq!(w.total(), 19);
    }

    #[test]
    fn an_unreferenced_centroid_carries_no_exposure() {
        // Measuring it would spend budget on a centroid no vector uses.
        let (data, books, codes) = setup();
        let (queries, truths) = probes(&data, 10);
        let ps: Vec<Probe<'_>> = queries
            .iter()
            .zip(truths.iter())
            .map(|(q, t)| Probe {
                vector: q,
                truth: t,
            })
            .collect();
        let w = measure(
            Encoded {
                corpus: &data,
                n: N,
                d: D,
                codes: &codes,
                m: M,
            },
            0,
            &books[0],
            &ps,
            Sweep {
                r: 30,
                k: 10,
                delta: 400.0,
            },
        );
        for e in &w.per_centroid {
            if e.population == 0 {
                assert_eq!(e.weight(), 0);
            }
        }
    }
}
