//! Signed, adversarially directed corruption constructions.
//!
//! # Why constructions are directed
//!
//! A first attempt to reproduce a known defect failed: random displacements
//! almost never strike a query's own top-`k` neighbours, since a single
//! centroid holds around 1.8% of the corpus. A directed construction —
//! corrupting the centroid used by a query's own true neighbour, displaced
//! anti-parallel to the query subvector — produced a 0.127 recall loss the
//! bound predicted to be exactly zero.
//!
//! Failing to reproduce a defect is not evidence of its absence.
//!
//! # Sweep design
//!
//! Sweep both signs. The induced score shift is signed and identical for every
//! vector using the corrupted centroid: inflation creates intruders that evict
//! clean neighbours, deflation drops affected neighbours out of the candidate
//! set with no intruder. On a 40-case signed sweep an inflation-only bound was
//! violated 15 times, 12 of them deflating.
//!
//! Target the highest-population centroid and the query's own neighbours.
//! Random centroid sampling measures the typical case, which is not what a
//! protection scheme must cover.

use crate::recall::{recall_at_k, top_ids, top_ids_shifted, Encoded};

/// Which centroid a construction targets, and why.
///
/// # Aim must match sign
///
/// The worst case for the two signs is not the same centroid, which a single
/// aim policy cannot express.
///
/// Measured on the synthetic instance (`D=16, m=2, b=4, N=1500, R=100, k=10`),
/// aiming at the query's own neighbour's centroid and *inflating*:
///
/// | magnitude | recall change |
/// |---:|---:|
/// | 200 | **+0.088** |
/// | 600 | **+0.213** |
/// | 2000 | **+0.213** |
///
/// Inflation there *improves* recall, because the promoted vectors are the
/// query's true neighbours — quantization had pushed them below the candidate
/// boundary and the corruption pushes them back. Reporting that as damage
/// would be wrong by sign.
///
/// The inflating worst case aims at a centroid the query's neighbours do **not**
/// use, whose vectors become intruders. The deflating worst case aims at the
/// centroid they do use. [`worst_case_for`] applies this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aim {
    /// The centroid with the most referencing vectors.
    ///
    /// Random centroid sampling measures the typical case, which is not what a
    /// protection scheme must cover.
    HighestPopulation,
    /// The centroid used by this query's own nearest neighbour.
    ///
    /// A single centroid holds roughly `1/2^b` of the corpus, so random
    /// displacement almost never strikes a query's own top-`k`. The first
    /// attempt to reproduce a known defect failed for exactly this reason.
    QueryNeighbour {
        /// Query index.
        query: usize,
    },
    /// The most populated centroid this query's top-`k` neighbours avoid.
    ///
    /// The inflating worst case: its vectors can only enter the candidate list
    /// as intruders, never as recovered true neighbours.
    PopulatedNonNeighbour {
        /// Query index.
        query: usize,
    },
    /// A centroid chosen by index, for a controlled comparison.
    Fixed {
        /// Centroid index.
        centroid: usize,
    },
}

/// The aim that maximises damage for `sign`.
pub const fn worst_case_for(sign: Sign, query: usize) -> Aim {
    match sign {
        // Inflation damages via intruders, so target vectors that are not
        // already true neighbours.
        Sign::Inflate => Aim::PopulatedNonNeighbour { query },
        // Deflation damages by removing true neighbours, so target the
        // centroid they use.
        Sign::Deflate => Aim::QueryNeighbour { query },
    }
}

/// Direction of the induced score shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    /// Scores rise: affected vectors become intruders and evict incumbents
    /// from the far end of the bounded heap.
    Inflate,
    /// Scores fall: affected true neighbours drop out of the candidate set
    /// with **zero** intruders. Invisible to an inflation-only bound.
    Deflate,
}

impl Sign {
    /// Multiplier for a displacement magnitude.
    pub const fn multiplier(self) -> f32 {
        match self {
            Sign::Inflate => 1.0,
            Sign::Deflate => -1.0,
        }
    }
}

/// One directed construction.
#[derive(Clone, Copy, Debug)]
pub struct Construction {
    /// Which centroid.
    pub aim: Aim,
    /// Which direction.
    pub sign: Sign,
    /// Displacement magnitude.
    pub magnitude: f32,
    /// Subspace whose centroid is corrupted.
    pub subspace: usize,
}

/// What a construction cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Damage {
    /// Centroid actually targeted.
    pub centroid: usize,
    /// Vectors referencing it.
    pub population: u32,
    /// Clean two-stage recall.
    ///
    /// Every loss figure is relative to this, never to a perfect oracle: at
    /// `R = 100` roughly a third of true top-`k` items are already outside the
    /// clean candidate list, and charging that gap to corruption measures the
    /// baseline's limits.
    pub clean_recall: f32,
    /// Recall after corruption.
    pub corrupted_recall: f32,
    /// Candidates that entered the list and use the corrupted centroid.
    pub intruders: u32,
}

impl Damage {
    /// Recall lost, relative to the clean baseline.
    pub fn loss(&self) -> f32 {
        self.clean_recall - self.corrupted_recall
    }
}

/// Resolve an aim to a centroid index.
pub fn resolve(
    aim: Aim,
    data: Encoded<'_>,
    j: usize,
    centroids: usize,
    truths: &[Vec<u32>],
) -> usize {
    match aim {
        Aim::Fixed { centroid } => centroid,
        Aim::HighestPopulation => {
            let mut best = 0usize;
            let mut best_n = 0usize;
            for c in 0..centroids {
                let n = data
                    .codes
                    .iter()
                    .skip(j)
                    .step_by(data.m.max(1))
                    .filter(|code| **code as usize == c)
                    .count();
                if n > best_n {
                    best_n = n;
                    best = c;
                }
            }
            best
        }
        Aim::QueryNeighbour { query } => truths
            .get(query)
            .and_then(|t| t.first())
            .and_then(|id| data.codes.get(*id as usize * data.m + j))
            .map(|c| *c as usize)
            .unwrap_or(0),
        Aim::PopulatedNonNeighbour { query } => {
            let used: Vec<usize> = truths
                .get(query)
                .map(|t| {
                    t.iter()
                        .filter_map(|id| data.codes.get(*id as usize * data.m + j))
                        .map(|c| *c as usize)
                        .collect()
                })
                .unwrap_or_default();
            let mut best = 0usize;
            let mut best_n = 0usize;
            for c in 0..centroids {
                if used.contains(&c) {
                    continue;
                }
                let n = data
                    .codes
                    .iter()
                    .skip(j)
                    .step_by(data.m.max(1))
                    .filter(|code| **code as usize == c)
                    .count();
                if n > best_n {
                    best_n = n;
                    best = c;
                }
            }
            best
        }
    }
}

/// Run one construction against one query.
pub fn apply(
    data: Encoded<'_>,
    query: &[f32],
    truth: &[u32],
    construction: Construction,
    centroid: usize,
    r: usize,
    k: usize,
) -> Damage {
    let clean = top_ids(data, query, r);
    let shift = construction.sign.multiplier() * construction.magnitude;
    let corrupted = top_ids_shifted(data, query, r, construction.subspace, centroid, shift);

    let population = data
        .codes
        .iter()
        .skip(construction.subspace)
        .step_by(data.m.max(1))
        .filter(|code| **code as usize == centroid)
        .count() as u32;

    // An intruder is a candidate that was not in the clean list and uses the
    // corrupted centroid. Deflation produces none by construction, which is
    // the whole point of measuring it separately.
    let intruders = corrupted
        .iter()
        .filter(|id| !clean.contains(id))
        .filter(|id| {
            data.codes
                .get(**id as usize * data.m + construction.subspace)
                .map(|c| *c as usize == centroid)
                .unwrap_or(false)
        })
        .count() as u32;

    Damage {
        centroid,
        population,
        clean_recall: recall_at_k(&clean, truth, k),
        corrupted_recall: recall_at_k(&corrupted, truth, k),
        intruders,
    }
}

/// The two-sided depth-aware bound on recall loss.
///
/// Both channels are bounded by how deep the true neighbours sit in the clean
/// candidate list, which is why the depths are an argument and not a constant.
///
/// **Inflation.** A bounded heap evicts from its far end, so `n` intruders
/// displace exactly the incumbents at ranks `r-n+1 ..= r`. A true neighbour at
/// depth `d` survives whenever `n <= r - d`. The bound counts neighbours with
/// `d > r - intruders`.
///
/// **Deflation.** An affected true neighbour is pushed out regardless of its
/// depth, so every true neighbour using the corrupted centroid is at risk.
/// This channel has no depth threshold, and it is why an inflation-only bound
/// predicts zero where measured loss is not.
///
/// `depths` gives the clean-list rank of each true neighbour, and `affected`
/// marks which of them use the corrupted centroid; both are indexed alike.
pub fn bound(depths: &[usize], affected: &[bool], intruders: u32, r: usize, k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let threshold = (r as u32).saturating_sub(intruders) as usize;

    let mut at_risk = 0usize;
    for (i, &d) in depths.iter().take(k).enumerate() {
        // Deflation: an affected neighbour leaves whatever its depth.
        if affected.get(i).copied().unwrap_or(false) {
            at_risk += 1;
            continue;
        }
        // Inflation: only neighbours deeper than the eviction threshold.
        if intruders > 0 && d > threshold {
            at_risk += 1;
        }
    }
    (at_risk.min(k) as f32) / k as f32
}

/// Clean-list depths of the first `k` true neighbours.
///
/// A neighbour absent from the clean list is already lost to the baseline and
/// is reported at depth `r`, since corruption cannot lose it twice.
pub fn depths_of(clean: &[u32], truth: &[u32], k: usize, r: usize) -> Vec<usize> {
    truth
        .iter()
        .take(k)
        .map(|t| clean.iter().position(|c| c == t).unwrap_or(r))
        .collect()
}

/// Which of the first `k` true neighbours use `centroid` in subspace `j`.
pub fn affected_of(
    data: Encoded<'_>,
    truth: &[u32],
    k: usize,
    j: usize,
    centroid: usize,
) -> Vec<bool> {
    truth
        .iter()
        .take(k)
        .map(|t| {
            data.codes
                .get(*t as usize * data.m + j)
                .map(|c| *c as usize == centroid)
                .unwrap_or(false)
        })
        .collect()
}

/// How loose a bound was, as `bound / measured`.
///
/// A bound holding by 58x is valid and carries no engineering information.
/// Tracking the ratio is what shows when a bound needs replacing.
pub fn looseness(bound: f32, measured: f32) -> Option<f32> {
    if measured <= 0.0 {
        return None;
    }
    Some(bound / measured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recall::exact_top_ids;
    use sector_build::encode::encode;
    use sector_build::train::{train, TrainConfig};

    const D: usize = 16;
    const M: usize = 2;
    const N: usize = 1500;
    const B: usize = 4;

    struct Fixture {
        corpus: Vec<f32>,
        codes: Vec<u8>,
        centroids: Vec<f32>,
        queries: Vec<Vec<f32>>,
        truths: Vec<Vec<u32>>,
    }

    fn fixture() -> Fixture {
        let mut corpus = vec![0f32; N * D];
        for v in 0..N {
            let c = if v % 10 < 6 {
                0
            } else if v % 10 < 9 {
                1
            } else {
                2
            };
            for j in 0..D {
                corpus[v * D + j] = (c as f32) * 30.0 + (((v * 13 + j * 7) % 41) as f32) - 20.0;
            }
        }
        let cfg = TrainConfig {
            d: D,
            m: M,
            b: B,
            iterations: 40,
            seed: 9,
        };
        let (books, _) = train(&corpus, N, cfg).unwrap();
        let (codes, _) = encode(&corpus, N, D, &books);
        // Flatten the trained codebooks so stage one can reconstruct.
        let mut centroids = vec![0f32; M * (1 << B) * (D / M)];
        for (j, book) in books.iter().enumerate() {
            let at = j * (1 << B) * (D / M);
            centroids[at..at + book.centroids.len()].copy_from_slice(&book.centroids);
        }

        let queries: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                (0..D)
                    .map(|j| ((i * 5 + j * 3) % 31) as f32 - 15.0)
                    .collect()
            })
            .collect();
        let data = Encoded {
            corpus: &corpus,
            n: N,
            d: D,
            codes: &codes,
            m: M,
            centroids: &centroids,
            k: 1 << B,
        };
        // Truth is the exact ranking; the candidate list is approximate, so
        // clean two-stage recall is below 1 and losses are measured against it.
        let truths: Vec<Vec<u32>> = queries.iter().map(|q| exact_top_ids(data, q, 10)).collect();
        Fixture {
            corpus,
            codes,
            centroids,
            queries,
            truths,
        }
    }

    fn view(f: &Fixture) -> Encoded<'_> {
        Encoded {
            corpus: &f.corpus,
            n: N,
            d: D,
            codes: &f.codes,
            m: M,
            centroids: &f.centroids,
            k: 1 << B,
        }
    }

    #[test]
    fn a_directed_construction_reproduces_loss_that_random_sampling_misses() {
        // The methodological point. A random centroid rarely holds a query's
        // own neighbours, so random sweeps report no damage; aiming at the
        // neighbour's own centroid produces it.
        let f = fixture();
        let data = view(&f);
        let mut directed_loss = 0f32;
        let mut random_loss = 0f32;

        for (qi, (q, t)) in f.queries.iter().zip(f.truths.iter()).enumerate() {
            let aimed = Aim::QueryNeighbour { query: qi };
            let c = resolve(aimed, data, 0, 1 << B, &f.truths);
            let d = apply(
                data,
                q,
                t,
                Construction {
                    aim: aimed,
                    sign: Sign::Deflate,
                    magnitude: 800.0,
                    subspace: 0,
                },
                c,
                100,
                10,
            );
            directed_loss += d.loss();

            // A centroid the query's neighbours do not use.
            let other = (c + 1) % (1 << B);
            let d2 = apply(
                data,
                q,
                t,
                Construction {
                    aim: Aim::Fixed { centroid: other },
                    sign: Sign::Deflate,
                    magnitude: 800.0,
                    subspace: 0,
                },
                other,
                100,
                10,
            );
            random_loss += d2.loss();
        }

        assert!(
            directed_loss > random_loss,
            "directed {directed_loss} did not exceed undirected {random_loss}"
        );
        assert!(
            directed_loss > 0.0,
            "the directed construction found nothing"
        );
    }

    #[test]
    fn deflation_causes_loss_with_zero_intruders() {
        // The channel an inflation-only bound predicts as exactly zero.
        let f = fixture();
        let data = view(&f);
        let mut total_loss = 0f32;
        let mut total_intruders = 0u32;

        for (qi, (q, t)) in f.queries.iter().zip(f.truths.iter()).enumerate() {
            let aim = Aim::QueryNeighbour { query: qi };
            let c = resolve(aim, data, 0, 1 << B, &f.truths);
            let d = apply(
                data,
                q,
                t,
                Construction {
                    aim,
                    sign: Sign::Deflate,
                    magnitude: 900.0,
                    subspace: 0,
                },
                c,
                100,
                10,
            );
            total_loss += d.loss();
            total_intruders += d.intruders;
        }

        assert_eq!(
            total_intruders, 0,
            "deflation must not admit vectors using the corrupted centroid"
        );
        assert!(
            total_loss > 0.0,
            "an inflation-only bound would predict exactly zero here"
        );
    }

    #[test]
    fn inflation_admits_intruders() {
        // The complementary channel, so the two are shown to differ in
        // mechanism and not only in magnitude.
        let f = fixture();
        let data = view(&f);
        let mut intruders = 0u32;
        for (qi, (q, t)) in f.queries.iter().zip(f.truths.iter()).enumerate() {
            let aim = Aim::QueryNeighbour { query: qi };
            let c = resolve(aim, data, 0, 1 << B, &f.truths);
            let d = apply(
                data,
                q,
                t,
                Construction {
                    aim,
                    sign: Sign::Inflate,
                    magnitude: 900.0,
                    subspace: 0,
                },
                c,
                100,
                10,
            );
            intruders += d.intruders;
        }
        assert!(intruders > 0, "inflation produced no intruders");
    }

    #[test]
    fn the_worst_case_centroid_is_chosen_by_population_not_index() {
        let f = fixture();
        let data = view(&f);
        let worst = resolve(Aim::HighestPopulation, data, 0, 1 << B, &f.truths);

        let count_of = |c: usize| {
            f.codes
                .iter()
                .step_by(M)
                .filter(|code| **code as usize == c)
                .count()
        };
        for c in 0..(1 << B) {
            assert!(
                count_of(worst) >= count_of(c),
                "centroid {worst} is not the most populated"
            );
        }
    }

    #[test]
    fn loss_is_reported_against_clean_two_stage_recall() {
        // Charging the baseline's own gap to corruption would measure the
        // baseline's limits, not the damage.
        let f = fixture();
        let data = view(&f);
        let q = &f.queries[0];
        let t = &f.truths[0];
        let aim = Aim::QueryNeighbour { query: 0 };
        let c = resolve(aim, data, 0, 1 << B, &f.truths);
        let d = apply(
            data,
            q,
            t,
            Construction {
                aim,
                sign: Sign::Deflate,
                magnitude: 700.0,
                subspace: 0,
            },
            c,
            100,
            10,
        );
        assert!(d.clean_recall > 0.0);
        assert!(d.corrupted_recall <= d.clean_recall);
        assert!((d.loss() - (d.clean_recall - d.corrupted_recall)).abs() < 1e-6);
    }

    #[test]
    fn looseness_is_reported_and_undefined_at_zero_loss() {
        // A bound that holds by a wide margin is valid and uninformative; the
        // ratio is what shows when it needs replacing.
        assert_eq!(looseness(0.5, 0.1), Some(5.0));
        assert_eq!(looseness(0.5, 0.0), None);
    }

    #[test]
    fn the_bound_covers_both_channels() {
        // Ten true neighbours at increasing depth, none affected: with 3
        // intruders only those deeper than r-3 = 97 are at risk.
        let depths: Vec<usize> = vec![1, 5, 12, 27, 40, 60, 80, 95, 98, 99];
        let none = vec![false; 10];
        let inflation_only = bound(&depths, &none, 3, 100, 10);
        assert!((inflation_only - 0.2).abs() < 1e-6, "got {inflation_only}");

        // The same neighbours with zero intruders: an inflation-only bound is
        // exactly zero.
        assert_eq!(bound(&depths, &none, 0, 100, 10), 0.0);

        // Deflation: two shallow neighbours affected, still zero intruders.
        // The two-sided bound is non-zero where the inflation-only one is not.
        let mut affected = vec![false; 10];
        affected[0] = true;
        affected[1] = true;
        let two_sided = bound(&depths, &affected, 0, 100, 10);
        assert!((two_sided - 0.2).abs() < 1e-6, "got {two_sided}");
    }

    #[test]
    fn a_shallow_neighbour_survives_more_intruders_than_a_deep_one() {
        // The depth-margin property, which is what makes the bound
        // depth-aware: an incumbent at depth d falls only once n > r - d.
        let shallow = vec![10usize];
        let deep = vec![90usize];
        let none = vec![false; 1];
        // 20 intruders: threshold 80. The deep one is at risk, the shallow is
        // not.
        assert_eq!(bound(&shallow, &none, 20, 100, 1), 0.0);
        assert_eq!(bound(&deep, &none, 20, 100, 1), 1.0);
    }

    #[test]
    fn the_bound_holds_against_measured_loss() {
        // The falsification check: if measured loss exceeds the bound, the
        // bound is refuted. Looseness is reported whatever the outcome.
        let f = fixture();
        let data = view(&f);
        let mut worst_ratio: Option<f32> = None;
        let mut violations = 0usize;

        for (qi, (q, t)) in f.queries.iter().zip(f.truths.iter()).enumerate() {
            for sign in [Sign::Inflate, Sign::Deflate] {
                let aim = Aim::QueryNeighbour { query: qi };
                let c = resolve(aim, data, 0, 1 << B, &f.truths);
                let d = apply(
                    data,
                    q,
                    t,
                    Construction {
                        aim,
                        sign,
                        magnitude: 800.0,
                        subspace: 0,
                    },
                    c,
                    100,
                    10,
                );
                let clean = top_ids(data, q, 100);
                let depths = depths_of(&clean, t, 10, 100);
                let affected = match sign {
                    Sign::Deflate => affected_of(data, t, 10, 0, c),
                    Sign::Inflate => vec![false; 10],
                };
                let b = bound(&depths, &affected, d.intruders, 100, 10);
                if d.loss() > b + 1e-6 {
                    violations += 1;
                }
                if let Some(l) = looseness(b, d.loss()) {
                    worst_ratio = Some(worst_ratio.map_or(l, |w: f32| w.max(l)));
                }
            }
        }
        assert_eq!(violations, 0, "the two-sided bound was violated");
        // Looseness is reported, not asserted tight: a bound holding by a wide
        // margin is valid and uninformative, and the ratio is the signal.
        assert!(
            worst_ratio.is_some(),
            "no construction produced measurable loss"
        );
    }
}
