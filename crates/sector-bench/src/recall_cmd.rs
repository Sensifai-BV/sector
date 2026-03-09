//! `recall` — the P2 measurement.
//!
//! Recall against a dataset's **shipped** ground truth. Recomputing it would
//! admit a metric mismatch that biases every number in the same direction,
//! which is the kind of error that looks like a result.
//!
//! Two recalls are reported, not one. Stage-one recall is how many true
//! neighbours the scan offered at depth `R`; two-stage recall is how many
//! survived rerank into the top `k`. Their difference says which stage loses,
//! and a single figure hides that.

use crate::dataset_util::{self, recall_at};
use crate::{flag, opt_num, parse_config};
use sector_bench::json::{self, Value};
use sector_bench::pipeline::build_index;
use sector_build::dataset::GroundTruth;
use std::path::PathBuf;

/// Candidate depths swept.
const DEPTHS: [usize; 5] = [10, 50, 100, 500, 1000];

/// Neighbour counts reported.
const KS: [usize; 3] = [1, 10, 100];

/// Run `recall`.
pub fn run(argv: &[String]) -> Result<PathBuf, String> {
    let cfg = parse_config(argv)?;
    let base_path = PathBuf::from(flag(argv, "--base")?);
    let query_path = PathBuf::from(flag(argv, "--queries")?);
    let truth_path = PathBuf::from(flag(argv, "--truth")?);
    let n_queries = opt_num(argv, "--nq", 0)?;
    let out_name = argv
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| argv.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "recall".to_string());

    let base = dataset_util::load(&base_path, cfg.n)?;
    if !base.dim.is_multiple_of(cfg.m) {
        return Err(format!("D={} is not divisible by m={}", base.dim, cfg.m));
    }
    let queries = dataset_util::load(&query_path, n_queries)?;
    if queries.dim != base.dim {
        return Err(format!(
            "queries are D={} but base is D={}",
            queries.dim, base.dim
        ));
    }
    let truth = GroundTruth::load(&truth_path).map_err(|e| format!("{e:?}"))?;

    // A subset changes which vectors exist, so the shipped ground truth — which
    // indexes the full corpus — no longer applies to it. Recomputing for the
    // subset is correct; silently reporting against the shipped rows would give
    // a plausible number that means nothing.
    let is_subset = base.count < base.file_count;
    eprintln!(
        "base {} x {} ({}), {} queries, truth k={}",
        base.count,
        base.dim,
        if is_subset { "subset" } else { "full file" },
        queries.count,
        truth.k
    );

    let started = std::time::Instant::now();
    let pipeline = build_index(&base.data, base.count, base.dim, &cfg)?;
    let build_secs = started.elapsed().as_secs_f64();
    eprintln!(
        "index built in {build_secs:.1}s: codebook {} B, {} B/vector payload",
        pipeline.codebook_bytes(),
        pipeline.payload_bytes()
    );

    let max_k = *KS.iter().max().unwrap_or(&10);

    // Ground truth per query, resolved once: recomputing it inside the depth
    // sweep would repeat an O(N) pass five times for no reason.
    let truths: Vec<Vec<u32>> = (0..queries.count)
        .map(|qi| {
            let q = &queries.data[qi * queries.dim..(qi + 1) * queries.dim];
            if is_subset {
                dataset_util::exact_top_l2(&base.data, base.count, base.dim, q, max_k)
            } else {
                truth
                    .row(qi)
                    .map(|r| r.iter().map(|x| *x as u32).collect())
                    .unwrap_or_default()
            }
        })
        .collect();

    // One scan per query serves every depth: the candidate list at depth `r` is
    // a prefix of the list at `max_depth`, since both are the same ranking
    // truncated. Rescanning per depth repeats a full N-vector pass for each —
    // four redundant passes over 8e9 code lookups at N=10^6.
    let depths: Vec<usize> = DEPTHS
        .iter()
        .copied()
        .filter(|r| *r <= base.count)
        .collect();
    let max_depth = depths.iter().copied().max().unwrap_or(100);
    let deep: Vec<Vec<u32>> = (0..queries.count)
        .map(|qi| {
            let q = &queries.data[qi * queries.dim..(qi + 1) * queries.dim];
            pipeline.stage_one(q, max_depth)
        })
        .collect();

    let mut rows = Vec::new();
    for &r in depths.iter() {
        let mut stage_one = [0f64; KS.len()];
        let mut two_stage = [0f64; KS.len()];
        // Fraction of true neighbours anywhere in the depth-R candidate list.
        // This is the ceiling rerank can reach: a neighbour the scan never
        // offered cannot be recovered, so the gap between `present` and
        // `two_stage` is rerank's loss and the gap below 1 is the scan's.
        let mut present = [0f64; KS.len()];
        for (qi, gt) in truths.iter().enumerate() {
            let q = &queries.data[qi * queries.dim..(qi + 1) * queries.dim];
            let candidates = &deep[qi][..r.min(deep[qi].len())];
            let reranked = pipeline.rerank(&base.data, base.dim, q, candidates, max_k);
            for (i, &k) in KS.iter().enumerate() {
                // Stage one is scored on the first `k` of the candidate list in
                // ADC order, not on the whole list. Scoring the full depth-`R`
                // list against a `k`-element truth measures whether the
                // neighbours are *present* at depth R, which is a different and
                // much easier question than whether they are in the top `k` —
                // and it makes stage one and two-stage identical whenever
                // rerank keeps the same set.
                let offered = &candidates[..k.min(candidates.len())];
                stage_one[i] += recall_at(offered, gt, k);
                two_stage[i] += recall_at(&reranked, gt, k);
                present[i] += recall_at(candidates, gt, k);
            }
        }
        let per = queries.count.max(1) as f64;
        let s1: Vec<f64> = stage_one.iter().map(|x| x / per).collect();
        let s2: Vec<f64> = two_stage.iter().map(|x| x / per).collect();
        let pr: Vec<f64> = present.iter().map(|x| x / per).collect();
        eprintln!(
            "R={r:<5} top-k@10 {:.4}   two-stage@10 {:.4}   present@10 {:.4}   rerank gain {:+.4}",
            s1[1],
            s2[1],
            pr[1],
            s2[1] - s1[1]
        );
        rows.push(json::obj(vec![
            ("r", json::i(r as i64)),
            ("stage_one_recall", json::floats(&s1)),
            ("two_stage_recall", json::floats(&s2)),
            ("present_in_candidates", json::floats(&pr)),
            (
                "rerank_gain",
                json::floats(
                    &s2.iter()
                        .zip(s1.iter())
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                ),
            ),
        ]));
    }

    // The synthetic figures the report states, carried alongside so the
    // comparison is in the file rather than in someone's memory.
    let reference = json::obj(vec![
        ("source", json::s("technical report v5, synthetic corpus")),
        ("config", json::s("D=128, m=16, b=8")),
        ("recall_at_10_r100", json::f(0.605)),
        ("recall_at_10_r500", json::f(0.934)),
    ]);

    let value = json::obj(vec![
        ("measurement", json::s("recall")),
        ("dataset", json::s(&base_path.display().to_string())),
        ("config", cfg.to_value(base.dim, base.count)),
        ("queries", json::i(queries.count as i64)),
        ("build_seconds", json::f(build_secs)),
        ("codebook_bytes", json::i(pipeline.codebook_bytes() as i64)),
        (
            "payload_bytes_per_vector",
            json::i(pipeline.payload_bytes() as i64),
        ),
        (
            "k_values",
            json::ints(&KS.iter().map(|x| *x as i64).collect::<Vec<_>>()),
        ),
        (
            "ground_truth",
            json::s(if is_subset {
                "recomputed for the subset — the shipped truth indexes the full corpus"
            } else {
                "shipped with the dataset"
            }),
        ),
        ("synthetic_reference", reference),
        ("by_depth", Value::List(rows)),
    ]);
    json::write_measurement(&out_name, &value).map_err(|e| format!("{e}"))
}
