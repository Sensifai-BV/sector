//! Protection allocation over the convex hull of achievable rates.
//!
//! Given per-group criticality weights and a parity budget, choose each group's
//! code rate to minimise expected recall loss. The solution is water-filling:
//! equalise the marginal `-w_c f_c'(beta_c)` across groups receiving parity.
//!
//! # Hypotheses
//!
//! Achievable code rates are discrete, so the exact problem is a
//! multiple-choice knapsack. Optimising over the lower convex envelope and
//! realising fractional rates by time-sharing across stripes makes the
//! relaxation exact on the envelope, with a rounding gap bounded by adjacent
//! rate spacing — `1/255` over GF(256).
//!
//! Expected recall loss is not linear in corrupted units; the measurements show
//! threshold behaviour. The linear objective is a first-order rare-failure
//! surrogate, valid when total residual failure probability over influential
//! units is far below 1. Outside that regime it optimises a different
//! objective, not merely an inaccurate one.
//!
//! # Scope of the claim
//!
//! First-order optimality within the surrogate class, over the convex hull of
//! achievable rates. Not optimality for the exact discrete recall-loss problem.
//! Water-filling is standard; what is derived here is the criticality model
//! feeding it, taken from index structure rather than tuned.
//!
//! Validate against exhaustive search on a small instance and check the derived
//! allocation lies on the Pareto frontier. That is a falsification criterion,
//! and it is cheap at small group counts.

use crate::criticality::Weights;

/// A protection option a group may be assigned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rate {
    /// Parity bytes per 256 data bytes. 0 is detect-only.
    pub parity_per_256: u16,
    /// Residual failure probability, in parts per billion.
    ///
    /// Integer so the allocation is reproducible across platforms: a float
    /// comparison that ties differently on two machines gives two different
    /// images from one input.
    pub residual_ppb: u32,
}

impl Rate {
    /// Storage cost of protecting `bytes` at this rate.
    pub const fn cost(&self, bytes: usize) -> usize {
        (bytes * self.parity_per_256 as usize) / 256
    }
}

/// The allocation for one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assignment {
    /// Group index.
    pub group: usize,
    /// Rate chosen.
    pub rate: Rate,
    /// Bytes spent.
    pub cost: usize,
}

/// What an allocation achieved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocation {
    /// Per group, in group order.
    pub assignments: Vec<Assignment>,
    /// Total parity bytes.
    pub spent: usize,
    /// Budget it was solved against.
    pub budget: usize,
    /// Expected loss surrogate: sum of `weight * residual_ppb`.
    ///
    /// The objective, not a measured recall figure. It is a first-order
    /// rare-failure surrogate, valid when the summed residual failure
    /// probability over influential units is far below 1.
    pub objective: u64,
}

impl Allocation {
    /// Budget left unspent.
    ///
    /// Non-zero because rates are discrete: the gap against the achievable
    /// rate spacing is reported rather than hidden, since a large gap means the
    /// rate set is too coarse for the budget.
    pub const fn slack(&self) -> usize {
        self.budget.saturating_sub(self.spent)
    }
}

/// Reduce `rates` to its lower convex envelope in (cost, residual) space.
///
/// A rate that is dominated — costs more and protects less than a mixture of
/// two others — can never be optimal in the continuous relaxation, and keeping
/// it only widens the search. The envelope is what water-filling walks.
pub fn envelope(rates: &[Rate], bytes: usize) -> Vec<Rate> {
    let mut sorted: Vec<Rate> = rates.to_vec();
    sorted.sort_by_key(|r| (r.cost(bytes), core::cmp::Reverse(r.residual_ppb)));
    sorted.dedup_by_key(|r| r.cost(bytes));

    let mut hull: Vec<Rate> = Vec::new();
    for candidate in sorted {
        // Strictly decreasing residual: a costlier rate that protects no better
        // is dominated outright and cannot appear on the envelope.
        if hull
            .last()
            .is_some_and(|last| candidate.residual_ppb >= last.residual_ppb)
        {
            continue;
        }
        // Drop points the new one makes non-convex: if the marginal gain from
        // the previous step is worse than from this one, the previous point is
        // interior.
        while hull.len() >= 2 {
            let Some(&b) = hull.last() else { break };
            let Some(&a) = hull.get(hull.len() - 2) else {
                break;
            };
            let d_ab = (a.residual_ppb - b.residual_ppb) as u64;
            let c_ab = (b.cost(bytes) - a.cost(bytes)).max(1) as u64;
            let d_bc = (b.residual_ppb - candidate.residual_ppb) as u64;
            let c_bc = (candidate.cost(bytes) - b.cost(bytes)).max(1) as u64;
            // b is interior when the slope into it is shallower than out of it.
            if d_ab * c_bc <= d_bc * c_ab {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(candidate);
    }
    hull
}

/// Allocate `budget` parity bytes across groups by water-filling.
///
/// Greedy on marginal objective reduction per byte, walking the convex
/// envelope. On the continuous relaxation this is exact; the discreteness of
/// the rate set makes the integer problem a knapsack, and the result claims
/// first-order optimality within the surrogate class and nothing more.
pub fn allocate(
    group_weights: &[u64],
    group_bytes: &[usize],
    rates: &[Rate],
    budget: usize,
) -> Allocation {
    let g = group_weights.len();
    let hulls: Vec<Vec<Rate>> = group_bytes.iter().map(|b| envelope(rates, *b)).collect();

    // Start every group at the cheapest rate on its envelope.
    let mut level = vec![0usize; g];
    let mut spent = 0usize;
    for (i, hull) in hulls.iter().enumerate() {
        if let (Some(r), Some(bytes)) = (hull.first(), group_bytes.get(i)) {
            spent += r.cost(*bytes);
        }
    }

    loop {
        // Best marginal objective reduction per byte across all groups.
        let mut best: Option<(usize, u64, u64)> = None;
        for i in 0..g {
            let Some(hull) = hulls.get(i) else { continue };
            let Some(&at) = level.get(i) else { continue };
            let (Some(current), Some(next)) = (hull.get(at), hull.get(at + 1)) else {
                continue;
            };
            let Some(&bytes) = group_bytes.get(i) else {
                continue;
            };
            let Some(&w) = group_weights.get(i) else {
                continue;
            };
            let extra = next.cost(bytes).saturating_sub(current.cost(bytes));
            if extra == 0 || spent + extra > budget {
                continue;
            }
            let gain = w * (current.residual_ppb - next.residual_ppb) as u64;
            let ratio = gain / extra as u64;
            let take = match best {
                None => true,
                Some((_, best_ratio, best_gain)) => {
                    ratio > best_ratio || (ratio == best_ratio && gain > best_gain)
                }
            };
            if take {
                best = Some((i, ratio, gain));
            }
        }

        let Some((i, _, _)) = best else { break };
        let Some(hull) = hulls.get(i) else { break };
        let Some(&at) = level.get(i) else { break };
        let (Some(current), Some(next), Some(&bytes)) =
            (hull.get(at), hull.get(at + 1), group_bytes.get(i))
        else {
            break;
        };
        spent += next.cost(bytes) - current.cost(bytes);
        if let Some(slot) = level.get_mut(i) {
            *slot = at + 1;
        }
    }

    let mut assignments = Vec::with_capacity(g);
    let mut objective = 0u64;
    for i in 0..g {
        let rate = hulls
            .get(i)
            .and_then(|h| h.get(level.get(i).copied().unwrap_or(0)))
            .copied()
            .unwrap_or(Rate {
                parity_per_256: 0,
                residual_ppb: 0,
            });
        let bytes = group_bytes.get(i).copied().unwrap_or(0);
        objective += group_weights.get(i).copied().unwrap_or(0) * rate.residual_ppb as u64;
        assignments.push(Assignment {
            group: i,
            rate,
            cost: rate.cost(bytes),
        });
    }

    Allocation {
        assignments,
        spent,
        budget,
        objective,
    }
}

/// Group weights from per-centroid exposures, by contiguous bucket.
pub fn group_weights(weights: &Weights, groups: usize) -> Vec<u64> {
    let k = weights.per_centroid.len();
    if groups == 0 || k == 0 {
        return Vec::new();
    }
    let base = k / groups;
    let extra = k % groups;
    let mut out = Vec::with_capacity(groups);
    let mut at = 0usize;
    for g in 0..groups {
        let size = base + usize::from(g < extra);
        let sum: u64 = weights
            .per_centroid
            .get(at..at + size)
            .map(|s| s.iter().map(|e| e.weight() as u64).sum())
            .unwrap_or(0);
        out.push(sum);
        at += size;
    }
    out
}

/// Exhaustive search over every rate assignment, for falsification.
///
/// Feasible only at small group counts, which is the point: the derived
/// allocation is checked against ground truth on an instance small enough to
/// enumerate, rather than trusted.
pub fn exhaustive(
    group_weights: &[u64],
    group_bytes: &[usize],
    rates: &[Rate],
    budget: usize,
) -> Option<Allocation> {
    let g = group_weights.len();
    if g == 0 || rates.is_empty() {
        return None;
    }
    let combinations = rates.len().checked_pow(g as u32)?;
    let mut best: Option<Allocation> = None;

    for combo in 0..combinations {
        let mut idx = combo;
        let mut spent = 0usize;
        let mut objective = 0u64;
        let mut assignments = Vec::with_capacity(g);
        for i in 0..g {
            let choice = idx % rates.len();
            idx /= rates.len();
            let Some(&rate) = rates.get(choice) else {
                continue;
            };
            let bytes = group_bytes.get(i).copied().unwrap_or(0);
            let cost = rate.cost(bytes);
            spent += cost;
            objective += group_weights.get(i).copied().unwrap_or(0) * rate.residual_ppb as u64;
            assignments.push(Assignment {
                group: i,
                rate,
                cost,
            });
        }
        if spent > budget {
            continue;
        }
        let candidate = Allocation {
            assignments,
            spent,
            budget,
            objective,
        };
        let better = match &best {
            None => true,
            Some(b) => {
                candidate.objective < b.objective
                    || (candidate.objective == b.objective && candidate.spent < b.spent)
            }
        };
        if better {
            best = Some(candidate);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criticality::Exposure;

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
            Rate {
                parity_per_256: 256,
                residual_ppb: 100,
            },
        ]
    }

    #[test]
    fn the_envelope_drops_dominated_rates() {
        // A rate costing more and protecting no better can never be optimal.
        let mut set = rates();
        set.push(Rate {
            parity_per_256: 96,
            residual_ppb: 100_000,
        });
        let hull = envelope(&set, 4096);
        assert!(
            !hull.iter().any(|r| r.parity_per_256 == 96),
            "dominated rate survived: {hull:?}"
        );
        // Costs strictly increase and residuals strictly decrease along it.
        for w in hull.windows(2) {
            assert!(w[0].cost(4096) < w[1].cost(4096));
            assert!(w[0].residual_ppb > w[1].residual_ppb);
        }
    }

    #[test]
    fn the_budget_is_respected() {
        let weights = vec![100u64, 50, 10];
        let bytes = vec![4096usize, 4096, 4096];
        let budget = 2048usize;
        let a = allocate(&weights, &bytes, &rates(), budget);
        assert!(a.spent <= budget, "spent {} over {budget}", a.spent);
        assert_eq!(a.assignments.len(), 3);
    }

    #[test]
    fn heavier_groups_receive_at_least_as_much_parity() {
        // The whole point of measuring criticality: a uniform policy spends
        // bytes where they do not buy recall.
        let weights = vec![1000u64, 100, 1];
        let bytes = vec![4096usize; 3];
        let a = allocate(&weights, &bytes, &rates(), 3000);
        let p: Vec<u16> = a
            .assignments
            .iter()
            .map(|x| x.rate.parity_per_256)
            .collect();
        assert!(p[0] >= p[1], "weights 1000 vs 100 gave {p:?}");
        assert!(p[1] >= p[2], "weights 100 vs 1 gave {p:?}");
    }

    #[test]
    fn the_derived_allocation_is_on_the_pareto_frontier() {
        // The falsification criterion, not a regression test: if water-filling
        // is off the frontier of exhaustive search, the claim of first-order
        // optimality is refuted.
        //
        // Measured: **exactly** optimal at every budget tried, not merely
        // close. Because the envelope is convex, each greedy step takes the
        // steepest remaining slope, and no later step can be steeper — so the
        // greedy order is the optimal order. The gap the report anticipates
        // from discreteness does not appear on this rate set, where every rate
        // cost is a multiple of the smallest.
        //
        // Asserting equality rather than a tolerance is deliberate: a
        // tolerance would let a real regression pass as rounding.
        let weights = vec![500u64, 200, 80, 10];
        let bytes = vec![4096usize; 4];
        for budget in [512usize, 1024, 2048, 4096, 6144, 8192] {
            let derived = allocate(&weights, &bytes, &rates(), budget);
            let optimal = exhaustive(&weights, &bytes, &rates(), budget).expect("a solution");
            assert_eq!(
                derived.objective, optimal.objective,
                "budget {budget}: derived {} vs optimal {}",
                derived.objective, optimal.objective
            );
        }
    }

    #[test]
    fn an_irregular_rate_set_can_leave_a_gap() {
        // The discreteness the report anticipates, shown where it appears:
        // with rate costs that are not multiples of one another, the greedy
        // walk can be forced to stop short of the budget. The allocation is
        // still on the frontier for what it spends, but the slack is real and
        // is reported rather than absorbed.
        let irregular = vec![
            Rate {
                parity_per_256: 0,
                residual_ppb: 1_000_000,
            },
            Rate {
                parity_per_256: 48,
                residual_ppb: 90_000,
            },
            Rate {
                parity_per_256: 176,
                residual_ppb: 900,
            },
        ];
        let weights = vec![400u64, 90];
        let bytes = vec![4096usize; 2];
        let budget = 3000usize;

        let derived = allocate(&weights, &bytes, &irregular, budget);
        let optimal = exhaustive(&weights, &bytes, &irregular, budget).expect("a solution");
        assert_eq!(derived.objective, optimal.objective);
        assert!(
            derived.slack() > 0,
            "expected unspendable budget, spent {}/{budget}",
            derived.spent
        );
    }

    #[test]
    fn the_rounding_gap_is_reported_not_hidden() {
        // Rates are discrete, so a budget is rarely spent exactly. A large gap
        // means the rate set is too coarse for the budget.
        let weights = vec![100u64, 100];
        let bytes = vec![4096usize; 2];
        let a = allocate(&weights, &bytes, &rates(), 1500);
        assert_eq!(a.slack(), a.budget - a.spent);
        assert!(a.slack() < 1500);
    }

    #[test]
    fn a_zero_budget_leaves_every_group_at_the_cheapest_rate() {
        let weights = vec![100u64, 50];
        let bytes = vec![4096usize; 2];
        let a = allocate(&weights, &bytes, &rates(), 0);
        assert_eq!(a.spent, 0);
        for x in &a.assignments {
            assert_eq!(x.rate.parity_per_256, 0);
        }
    }

    #[test]
    fn group_weights_partition_the_centroids() {
        // A centroid in no group is unprotected; one in two makes the byte
        // accounting wrong.
        let w = Weights {
            per_centroid: (0..10)
                .map(|i| Exposure {
                    population: 1,
                    inflate_loss: i as u32,
                    deflate_loss: 0,
                })
                .collect(),
        };
        let g = group_weights(&w, 4);
        assert_eq!(g.len(), 4);
        assert_eq!(g.iter().sum::<u64>(), w.total());
    }

    #[test]
    fn allocation_is_deterministic() {
        // Two builds of one input must give one image.
        let weights = vec![300u64, 300, 100];
        let bytes = vec![4096usize; 3];
        let a = allocate(&weights, &bytes, &rates(), 2000);
        let b = allocate(&weights, &bytes, &rates(), 2000);
        assert_eq!(a, b);
    }
}
