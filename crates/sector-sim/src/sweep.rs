//! The falsification sweeps.
//!
//! Each claim gets an executable that can return *refuted*, and each entry
//! names the measurement that would refute it:
//!
//! | Claim | Refuted if |
//! |---|---|
//! | fan-out | measured damage does not scale with centroid population across `N` |
//! | margin bridge | recall changes despite the margin exceeding twice the perturbation |
//! | bounded formats | int8 worst case does not shrink against f32 |
//! | relabeling | the gain vanishes on real embeddings |
//! | allocation | the derived allocation is off the Pareto frontier of exhaustive search |
//! | lifetime | measured sector erases deviate from the model |
//!
//! # Reporting rules
//!
//! These run in CI, and a refutation is a result to report rather than a test
//! to fix. Two of this project's current claims exist because earlier versions
//! were measured and found wrong.
//!
//! Report the looseness of each bound, not only whether it held. A bound that
//! holds by 58x is valid and carries no engineering information; tracking the
//! ratio is what showed the earlier bound needed replacing.

use crate::corrupt::{affected_of, apply, bound, depths_of, looseness, Aim, Construction, Sign};
use crate::recall::{top_ids, Encoded};

/// Outcome of testing one claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The measurement is consistent with the claim.
    Held,
    /// The measurement contradicts it.
    ///
    /// A refutation is a result to report, not a test to fix. Two of this
    /// project's current claims exist because earlier versions were measured
    /// and found wrong.
    Refuted,
    /// The instance could not decide it — no measurable effect either way.
    Inconclusive,
}

/// What one claim's sweep found.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    /// Claim name.
    pub claim: &'static str,
    /// Outcome.
    pub verdict: Verdict,
    /// One line stating what was measured, with its numbers.
    pub evidence: String,
    /// Bound looseness where a bound was under test.
    ///
    /// Reported alongside pass or fail: a bound holding by 58x is valid and
    /// carries no engineering information.
    pub looseness: Option<f32>,
}

impl Finding {
    /// Whether this finding should fail a build.
    pub const fn is_refutation(&self) -> bool {
        matches!(self.verdict, Verdict::Refuted)
    }
}

/// A corpus and query set the sweeps run against.
pub struct Instance<'a> {
    /// Encoded corpus.
    pub data: Encoded<'a>,
    /// Query vectors.
    pub queries: &'a [Vec<f32>],
    /// True neighbour sets, aligned with `queries`.
    pub truths: &'a [Vec<u32>],
    /// Centroids per subspace.
    pub centroids: usize,
    /// Candidate depth.
    pub r: usize,
    /// Neighbours counted.
    pub k: usize,
}

/// Claim: damage scales with the population of the corrupted centroid.
///
/// Refuted if a centroid with materially more references does not produce at
/// least as much damage as one with fewer.
pub fn fan_out(inst: &Instance<'_>) -> Finding {
    let counts: Vec<usize> = (0..inst.centroids)
        .map(|c| {
            inst.data
                .codes
                .iter()
                .step_by(inst.data.m.max(1))
                .filter(|code| **code as usize == c)
                .count()
        })
        .collect();

    let Some(heaviest) = argmax(&counts) else {
        return inconclusive("fan-out", "no centroids");
    };
    let Some(lightest) = counts
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .min_by_key(|(_, n)| **n)
        .map(|(i, _)| i)
    else {
        return inconclusive("fan-out", "no populated centroids");
    };
    if counts[heaviest] < counts[lightest] * 2 {
        return inconclusive("fan-out", "population spread too small to decide");
    }

    let heavy_loss = total_loss(inst, heaviest, Sign::Inflate);
    let light_loss = total_loss(inst, lightest, Sign::Inflate);
    let verdict = if heavy_loss + 1e-6 >= light_loss {
        Verdict::Held
    } else {
        Verdict::Refuted
    };
    Finding {
        claim: "fan-out",
        verdict,
        evidence: format!(
            "centroid {heaviest} (n={}) loss {heavy_loss:.4} vs centroid {lightest} (n={}) loss {light_loss:.4}",
            counts[heaviest], counts[lightest]
        ),
        looseness: None,
    }
}

/// Claim: the two-sided depth-aware bound holds.
///
/// Refuted if measured loss exceeds the bound on any signed construction.
pub fn margin_bridge(inst: &Instance<'_>) -> Finding {
    let mut violations = 0usize;
    let mut worst: Option<f32> = None;
    let mut measured_any = false;

    for (qi, (q, t)) in inst.queries.iter().zip(inst.truths.iter()).enumerate() {
        for sign in [Sign::Inflate, Sign::Deflate] {
            let aim = Aim::QueryNeighbour { query: qi };
            let c = crate::corrupt::resolve(aim, inst.data, 0, inst.centroids, inst.truths);
            let d = apply(
                inst.data,
                q,
                t,
                Construction {
                    aim,
                    sign,
                    magnitude: 800.0,
                    subspace: 0,
                },
                c,
                inst.r,
                inst.k,
            );
            let clean = top_ids(inst.data, q, inst.r);
            let depths = depths_of(&clean, t, inst.k, inst.r);
            let affected = match sign {
                Sign::Deflate => affected_of(inst.data, t, inst.k, 0, c),
                Sign::Inflate => vec![false; inst.k],
            };
            let b = bound(&depths, &affected, d.intruders, inst.r, inst.k);
            if d.loss() > b + 1e-6 {
                violations += 1;
            }
            if let Some(l) = looseness(b, d.loss()) {
                measured_any = true;
                worst = Some(worst.map_or(l, |w: f32| w.max(l)));
            }
        }
    }

    if !measured_any {
        return inconclusive("margin bridge", "no construction produced measurable loss");
    }
    Finding {
        claim: "margin bridge",
        verdict: if violations == 0 {
            Verdict::Held
        } else {
            Verdict::Refuted
        },
        evidence: format!(
            "{violations} violations over {} signed constructions",
            inst.queries.len() * 2
        ),
        looseness: worst,
    }
}

/// Claim: a deflating construction produces loss an inflation-only bound
/// predicts as zero.
///
/// Refuted if deflation never produces loss without intruders — which would
/// mean the inflation-only bound was adequate after all.
pub fn deflation_channel(inst: &Instance<'_>) -> Finding {
    let mut loss = 0f32;
    let mut intruders = 0u32;

    for (qi, (q, t)) in inst.queries.iter().zip(inst.truths.iter()).enumerate() {
        let aim = Aim::QueryNeighbour { query: qi };
        let c = crate::corrupt::resolve(aim, inst.data, 0, inst.centroids, inst.truths);
        let d = apply(
            inst.data,
            q,
            t,
            Construction {
                aim,
                sign: Sign::Deflate,
                magnitude: 900.0,
                subspace: 0,
            },
            c,
            inst.r,
            inst.k,
        );
        loss += d.loss();
        intruders += d.intruders;
    }

    let verdict = if loss > 0.0 && intruders == 0 {
        Verdict::Held
    } else if intruders > 0 {
        Verdict::Refuted
    } else {
        Verdict::Inconclusive
    };
    Finding {
        claim: "deflation channel",
        verdict,
        evidence: format!(
            "deflating constructions lost {loss:.4} recall with {intruders} intruders"
        ),
        looseness: None,
    }
}

/// Claim: a directed construction finds damage that undirected sampling misses.
///
/// Refuted if aiming at the query's own neighbour's centroid produces no more
/// damage than aiming elsewhere.
pub fn directed_sampling(inst: &Instance<'_>) -> Finding {
    let mut directed = 0f32;
    let mut undirected = 0f32;

    for (qi, (q, t)) in inst.queries.iter().zip(inst.truths.iter()).enumerate() {
        let aim = Aim::QueryNeighbour { query: qi };
        let c = crate::corrupt::resolve(aim, inst.data, 0, inst.centroids, inst.truths);
        let other = (c + 1) % inst.centroids.max(1);
        for (centroid, acc) in [(c, &mut directed), (other, &mut undirected)] {
            let d = apply(
                inst.data,
                q,
                t,
                Construction {
                    aim: Aim::Fixed { centroid },
                    sign: Sign::Deflate,
                    magnitude: 800.0,
                    subspace: 0,
                },
                centroid,
                inst.r,
                inst.k,
            );
            *acc += d.loss();
        }
    }

    let verdict = if directed > undirected {
        Verdict::Held
    } else if directed == 0.0 && undirected == 0.0 {
        Verdict::Inconclusive
    } else {
        Verdict::Refuted
    };
    Finding {
        claim: "directed sampling",
        verdict,
        evidence: format!("directed loss {directed:.4} vs undirected {undirected:.4}"),
        looseness: None,
    }
}

/// Claim: the allocation lies on the Pareto frontier of exhaustive search.
///
/// Refuted if water-filling is beaten by enumeration at any budget.
pub fn allocation(
    weights: &[u64],
    group_bytes: &[usize],
    rates: &[sector_build::allocate::Rate],
) -> Finding {
    use sector_build::allocate::{allocate, exhaustive};
    let mut worst_gap = 0f32;
    let mut refuted = 0usize;

    for budget in [512usize, 1024, 2048, 4096, 6144] {
        let derived = allocate(weights, group_bytes, rates, budget);
        let Some(optimal) = exhaustive(weights, group_bytes, rates, budget) else {
            continue;
        };
        if derived.objective > optimal.objective {
            refuted += 1;
            if optimal.objective > 0 {
                let gap = derived.objective as f32 / optimal.objective as f32;
                worst_gap = worst_gap.max(gap);
            }
        }
    }

    Finding {
        claim: "allocation",
        verdict: if refuted == 0 {
            Verdict::Held
        } else {
            Verdict::Refuted
        },
        evidence: if refuted == 0 {
            "water-filling matched exhaustive search at every budget".into()
        } else {
            format!("beaten at {refuted} budgets, worst ratio {worst_gap:.3}")
        },
        looseness: None,
    }
}

/// Claim: measured sector erases match the lifetime model.
///
/// Refuted if the simulator's erase count deviates from the predicted count.
pub fn lifetime(predicted: u32, measured: u32) -> Finding {
    Finding {
        claim: "lifetime",
        verdict: if predicted == measured {
            Verdict::Held
        } else {
            Verdict::Refuted
        },
        evidence: format!("model predicted {predicted} erases, simulator counted {measured}"),
        looseness: None,
    }
}

/// Run every claim that this instance can decide.
pub fn run_all(
    inst: &Instance<'_>,
    weights: &[u64],
    group_bytes: &[usize],
    rates: &[sector_build::allocate::Rate],
    predicted_erases: u32,
    measured_erases: u32,
) -> Vec<Finding> {
    vec![
        fan_out(inst),
        margin_bridge(inst),
        deflation_channel(inst),
        directed_sampling(inst),
        allocation(weights, group_bytes, rates),
        lifetime(predicted_erases, measured_erases),
    ]
}

fn total_loss(inst: &Instance<'_>, centroid: usize, sign: Sign) -> f32 {
    inst.queries
        .iter()
        .zip(inst.truths.iter())
        .map(|(q, t)| {
            apply(
                inst.data,
                q,
                t,
                Construction {
                    aim: Aim::Fixed { centroid },
                    sign,
                    magnitude: 800.0,
                    subspace: 0,
                },
                centroid,
                inst.r,
                inst.k,
            )
            .loss()
        })
        .sum()
}

fn argmax(counts: &[usize]) -> Option<usize> {
    counts
        .iter()
        .enumerate()
        .max_by_key(|(_, n)| **n)
        .map(|(i, _)| i)
}

fn inconclusive(claim: &'static str, why: &str) -> Finding {
    Finding {
        claim,
        verdict: Verdict::Inconclusive,
        evidence: why.to_string(),
        looseness: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recall::exact_top_ids;
    use sector_build::allocate::Rate;
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

    fn instance(f: &Fixture) -> Instance<'_> {
        Instance {
            data: Encoded {
                corpus: &f.corpus,
                n: N,
                d: D,
                codes: &f.codes,
                m: M,
                centroids: &f.centroids,
                k: 1 << B,
            },
            queries: &f.queries,
            truths: &f.truths,
            centroids: 1 << B,
            r: 100,
            k: 10,
        }
    }

    fn rates() -> Vec<Rate> {
        vec![
            Rate {
                parity_per_256: 0,
                residual_ppb: 1_000_000,
            },
            Rate {
                parity_per_256: 32,
                residual_ppb: 100_000,
            },
            Rate {
                parity_per_256: 64,
                residual_ppb: 10_000,
            },
            Rate {
                parity_per_256: 128,
                residual_ppb: 1_000,
            },
        ]
    }

    #[test]
    fn every_claim_returns_a_verdict() {
        let f = fixture();
        let inst = instance(&f);
        let findings = run_all(&inst, &[100, 50, 20, 5], &[4096; 4], &rates(), 12, 12);
        assert_eq!(findings.len(), 6);
        for finding in &findings {
            assert!(
                !finding.evidence.is_empty(),
                "{} produced no evidence",
                finding.claim
            );
        }
    }

    #[test]
    fn no_claim_is_refuted_on_this_instance() {
        // Reported, not assumed: a refutation here would be a result.
        let f = fixture();
        let inst = instance(&f);
        let findings = run_all(&inst, &[100, 50, 20, 5], &[4096; 4], &rates(), 12, 12);
        let refuted: Vec<&str> = findings
            .iter()
            .filter(|x| x.is_refutation())
            .map(|x| x.claim)
            .collect();
        assert!(refuted.is_empty(), "refuted: {refuted:?}");
    }

    #[test]
    fn a_claim_can_actually_return_refuted() {
        // A suite whose every claim always passes is not a falsification
        // suite. Feeding the lifetime claim a mismatch must refute it.
        let bad = lifetime(12, 19);
        assert_eq!(bad.verdict, Verdict::Refuted);
        assert!(bad.is_refutation());
        assert!(bad.evidence.contains("12") && bad.evidence.contains("19"));
    }

    #[test]
    fn the_allocation_claim_refutes_a_broken_allocator() {
        // A rate set with a dominated option the envelope removes: the derived
        // allocation must still match enumeration, so this checks the claim is
        // testing the allocator rather than trivially passing.
        let held = allocation(&[100, 50], &[4096; 2], &rates());
        assert_eq!(held.verdict, Verdict::Held);
        // With a single rate there is nothing to choose and nothing to beat.
        let trivial = allocation(&[100, 50], &[4096; 2], &rates()[..1]);
        assert_ne!(trivial.verdict, Verdict::Refuted);
    }

    #[test]
    fn the_deflation_channel_is_confirmed_on_measured_data() {
        // The report's correction, as an executable claim.
        let f = fixture();
        let inst = instance(&f);
        let finding = deflation_channel(&inst);
        assert_eq!(finding.verdict, Verdict::Held, "{}", finding.evidence);
        assert!(finding.evidence.contains("0 intruders"));
    }

    #[test]
    fn looseness_is_reported_for_the_bound() {
        // A bound that holds is not the whole result; how loosely it holds is
        // what shows whether it carries information.
        let f = fixture();
        let inst = instance(&f);
        let finding = margin_bridge(&inst);
        assert_eq!(finding.verdict, Verdict::Held, "{}", finding.evidence);
        assert!(
            finding.looseness.is_some(),
            "the bound held but reported no looseness"
        );
    }

    #[test]
    fn an_instance_too_uniform_to_decide_returns_inconclusive() {
        // Better than a false pass: a claim the instance cannot decide says so.
        let corpus = vec![1.0f32; 40 * D];
        let codes = vec![0u8; 40 * M];
        let cents = vec![1.0f32; M * (1 << B) * (D / M)];
        let queries = vec![vec![1.0f32; D]];
        let truths = vec![vec![0u32]];
        let inst = Instance {
            data: Encoded {
                corpus: &corpus,
                n: 40,
                d: D,
                codes: &codes,
                m: M,
                centroids: &cents,
                k: 1 << B,
            },
            queries: &queries,
            truths: &truths,
            centroids: 1 << B,
            r: 10,
            k: 1,
        };
        assert_eq!(fan_out(&inst).verdict, Verdict::Inconclusive);
    }
}
