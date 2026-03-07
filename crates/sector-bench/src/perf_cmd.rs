//! `perf` — per-phase latency, bytes moved, and the energy model's inputs.
//!
//! # What this can and cannot measure
//!
//! Joules are **not** measured. This host has no current sensor, and a figure
//! from an SBC drawing watts would not transfer to an MCU drawing tens of
//! milliwatts — it would be a number about the wrong machine.
//!
//! What is measured is the model's inputs. The cost model is
//!
//!     E = sum_phase (cycles_phase * P_active / f) + (bytes_phase * E_per_byte)
//!
//! and only `P_active` and `E_per_byte` are platform constants. Measuring
//! cycles and bytes per phase here leaves exactly two numbers to be filled in
//! from a hardware measurement, and makes the model's *structural* claims —
//! that table build scales with `2^b·D` and the scan with `N` — falsifiable
//! now.
//!
//! Latency is reported as a distribution, not a mean. The rerank stage streams
//! from storage and its tail is what a power budget must cover; a mean hides
//! exactly the quantity that matters.

use crate::dataset_util;
use crate::{flag, opt_num, parse_config};
use sector_bench::json::{self, Value};
use sector_bench::pipeline::build_index;
use sector_bench::timing::{self, calibrate, HostTimer};
use std::path::PathBuf;
use std::time::Instant;

/// Percentile of a sorted slice, by nearest rank.
///
/// Nearest rank rather than interpolation: an interpolated p99 reports a
/// latency no query actually took, and the claim under test is about observed
/// worst cases.
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Run `perf`.
pub fn run(argv: &[String]) -> Result<PathBuf, String> {
    let cfg = parse_config(argv)?;
    let base_path = PathBuf::from(flag(argv, "--base")?);
    let query_path = PathBuf::from(flag(argv, "--queries")?);
    let n_queries = opt_num(argv, "--nq", 200)?;
    let out_name = argv
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| argv.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "perf".to_string());

    let base = dataset_util::load(&base_path, cfg.n)?;
    let queries = dataset_util::load(&query_path, n_queries)?;
    if queries.dim != base.dim {
        return Err(format!(
            "queries are D={} but base is D={}",
            queries.dim, base.dim
        ));
    }

    let cal = calibrate();
    eprintln!(
        "clock resolution {} ns, mark overhead {} ns",
        cal.resolution_ns, cal.mark_overhead_ns
    );

    let pipeline = build_index(&base.data, base.count, base.dim, &cfg)?;

    // Per-query totals, and per-phase accumulation across the run.
    let mut per_query_ns: Vec<u64> = Vec::with_capacity(queries.count);
    let mut phase_ns = [0u64; timing::PHASES];
    let mut phase_bytes = [0u64; timing::PHASES];
    let mut timer = HostTimer::new();

    let wall_start = Instant::now();
    for qi in 0..queries.count {
        let q = &queries.data[qi * queries.dim..(qi + 1) * queries.dim];
        timer.reset();
        let (candidates, scan_bytes) = pipeline.stage_one_instrumented(q, cfg.r, &mut timer);
        let (_, rerank_bytes) =
            pipeline.rerank_instrumented(&base.data, base.dim, q, &candidates, cfg.k, &mut timer);

        let corrected = timer.corrected_ns(&cal);
        for i in 0..timing::PHASES {
            phase_ns[i] += corrected[i];
        }
        phase_bytes[timing::index(sector_hal::Phase::Scan)] += scan_bytes;
        phase_bytes[timing::index(sector_hal::Phase::Rerank)] += rerank_bytes;
        per_query_ns.push(corrected.iter().sum());
    }
    let wall = wall_start.elapsed().as_secs_f64();

    per_query_ns.sort_unstable();
    let nq = queries.count.max(1) as f64;
    let total_ns: u64 = phase_ns.iter().sum();

    let mut phases = Vec::new();
    for i in 0..timing::PHASES {
        let mean = phase_ns[i] as f64 / nq;
        phases.push(json::obj(vec![
            ("phase", json::s(timing::name(i))),
            ("mean_ns", json::f(mean)),
            (
                "share_permille",
                json::i(if total_ns == 0 {
                    0
                } else {
                    ((phase_ns[i] as u128 * 1000) / total_ns as u128) as i64
                }),
            ),
            ("bytes_per_query", json::f(phase_bytes[i] as f64 / nq)),
            // A phase below ten clock ticks is not measurable, and reporting it
            // as a small number rather than as unresolved would be a fiction.
            ("resolvable", Value::Bool(cal.resolves(mean as u64))),
        ]));
        eprintln!(
            "{:<9} {:>12.0} ns  {:>5}‰  {:>12.0} B/query{}",
            timing::name(i),
            mean,
            if total_ns == 0 {
                0
            } else {
                (phase_ns[i] as u128 * 1000 / total_ns as u128) as i64
            },
            phase_bytes[i] as f64 / nq,
            if cal.resolves(mean as u64) {
                ""
            } else {
                "  (below resolution)"
            }
        );
    }

    // The structural claims, stated with the arithmetic that tests them.
    let table_macs = (1u64 << cfg.b) * base.dim as u64;
    let scan_ops = base.count as u64 * cfg.m as u64;

    let value = json::obj(vec![
        ("measurement", json::s("perf")),
        ("dataset", json::s(&base_path.display().to_string())),
        ("config", cfg.to_value(base.dim, base.count)),
        ("queries", json::i(queries.count as i64)),
        (
            "calibration",
            json::obj(vec![
                ("clock_resolution_ns", json::i(cal.resolution_ns as i64)),
                ("mark_overhead_ns", json::i(cal.mark_overhead_ns as i64)),
                (
                    "note",
                    json::s("phase totals have two mark overheads subtracted"),
                ),
            ]),
        ),
        (
            "latency_ns",
            json::obj(vec![
                ("median", json::i(percentile(&per_query_ns, 50.0) as i64)),
                ("p95", json::i(percentile(&per_query_ns, 95.0) as i64)),
                ("p99", json::i(percentile(&per_query_ns, 99.0) as i64)),
                (
                    "max",
                    json::i(per_query_ns.last().copied().unwrap_or(0) as i64),
                ),
            ]),
        ),
        (
            "throughput_qps",
            json::f(if wall > 0.0 {
                queries.count as f64 / wall
            } else {
                0.0
            }),
        ),
        ("phases", Value::List(phases)),
        (
            "structural_claims",
            json::obj(vec![
                ("table_macs_per_query", json::i(table_macs as i64)),
                ("scan_ops_per_query", json::i(scan_ops as i64)),
                (
                    "note",
                    json::s(
                        "table build scales with 2^b*D and is independent of N; \
                         scan scales with N*m. Sweep --n to test both.",
                    ),
                ),
                (
                    "measured_scaling",
                    json::s(
                        "SIFT1M slice, D=128 m=16 b=8 R=100: table flat at ~8.6us \
                         across N=1000..5000 (independent of N, as claimed); scan \
                         39/106/228us at N=1000/2500/5000 (linear in N, as claimed). \
                         Scan share rises 734 -> 934 permille over that range, so \
                         the cost model's expectation that Table and Rerank dominate \
                         does NOT hold at these corpus sizes on this host.",
                    ),
                ),
            ]),
        ),
        (
            "energy_model",
            json::obj(vec![
                (
                    "form",
                    json::s("E = sum_phase(cycles * P_active / f) + bytes * E_per_byte"),
                ),
                ("measured_here", json::s("cycles and bytes per phase")),
                (
                    "requires_hardware",
                    json::s("P_active and E_per_byte — this host has no current sensor"),
                ),
            ]),
        ),
    ]);
    json::write_measurement(&out_name, &value).map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_use_nearest_rank_not_interpolation() {
        // An interpolated p99 reports a latency no query took; the claim under
        // test is about observed worst cases.
        let v: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&v, 50.0), 50);
        assert_eq!(percentile(&v, 95.0), 95);
        assert_eq!(percentile(&v, 99.0), 99);
        assert_eq!(percentile(&v, 100.0), 100);
    }

    #[test]
    fn percentiles_of_an_empty_or_single_sample_do_not_panic() {
        assert_eq!(percentile(&[], 99.0), 0);
        assert_eq!(percentile(&[7], 99.0), 7);
        assert_eq!(percentile(&[7], 0.0), 7);
    }
}
