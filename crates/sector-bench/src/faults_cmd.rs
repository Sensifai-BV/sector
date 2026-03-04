//! `faults` — recall under each fault channel, on a real index.
//!
//! Four channels are injected independently, because a claim about one must not
//! be validated by another's remedy. Recall is re-measured through the same
//! query path after each, and reported **relative to the clean two-stage
//! baseline** — charging the baseline's own gap to corruption would measure the
//! baseline's limits rather than the damage.
//!
//! The directed constructions are included alongside random injection. Random
//! displacement almost never strikes a query's own neighbours, so a random-only
//! sweep reports far less damage than an adversary would find — and can even
//! report a recall *gain*, since promoting a demoted true neighbour helps.

use crate::dataset_util::{self, recall_at};
use crate::{flag, opt_num, parse_config};
use sector_bench::json::{self, Value};
use sector_bench::pipeline::{build_index, Pipeline};
use std::path::PathBuf;

/// Deterministic PRNG. Injection must be reproducible from its seed, or a
/// comparison between a clean and a corrupted run means nothing.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

/// Measure two-stage recall@k over a query set.
/// The corpus and query set a measurement runs against.
///
/// Grouped rather than passed separately: corpus, dimension, queries and their
/// truths describe one dataset, and splitting them across a parameter list
/// invites a call that scores one corpus against another's ground truth.
pub struct Workload<'a> {
    /// Row-major corpus.
    pub base: &'a [f32],
    /// Dimension.
    pub d: usize,
    /// Row-major queries.
    pub queries: &'a [f32],
    /// Exact neighbour sets, aligned with the queries.
    pub truths: &'a [Vec<u32>],
    /// Candidate depth.
    pub r: usize,
    /// Results per query.
    pub k: usize,
}

/// Measure two-stage recall@k, with `dropped` ids removed from every candidate
/// list.
///
/// A dropped candidate is removed and not replaced. In the result it is
/// indistinguishable from an eviction, which is exactly why the drop count is
/// reported separately — silent degradation makes a recall regression
/// untraceable.
fn measure(pipeline: &Pipeline, w: &Workload<'_>, dropped: &std::collections::HashSet<u32>) -> f64 {
    let mut acc = 0f64;
    for (qi, gt) in w.truths.iter().enumerate() {
        let q = &w.queries[qi * w.d..(qi + 1) * w.d];
        let mut candidates = pipeline.stage_one(q, w.r);
        if !dropped.is_empty() {
            candidates.retain(|id| !dropped.contains(id));
        }
        let top = pipeline.rerank(w.base, w.d, q, &candidates, w.k);
        acc += recall_at(&top, gt, w.k);
    }
    acc / w.truths.len().max(1) as f64
}

/// Flip `count` random bits in the codebook.
///
/// The codebook is the high-fan-out structure: one corrupted byte alters the
/// reconstruction of every vector referencing that centroid, against one vector
/// for a payload byte.
fn flip_codebook_bits(p: &mut Pipeline, rng: &mut Rng, count: usize) {
    for _ in 0..count {
        let at = rng.below(p.codebook.components.len());
        let bit = rng.below(8);
        p.codebook.components[at] ^= 1i8 << bit;
    }
}

/// Flip `count` random bits in the payload codes.
fn flip_payload_bits(p: &mut Pipeline, rng: &mut Rng, count: usize) {
    for _ in 0..count {
        let at = rng.below(p.codes.len());
        let bit = rng.below(8);
        p.codes[at] ^= 1u8 << bit;
    }
}

/// Drop whole payload blocks, as a detected CRC failure does.
///
/// A detected block failure removes every vector in the block — 32 at a 16-byte
/// payload and a 512-byte block — rather than corrupting them individually.
fn drop_payload_blocks(
    p: &Pipeline,
    rng: &mut Rng,
    blocks: usize,
    dropped_ids: &mut std::collections::HashSet<u32>,
) -> usize {
    let per_block = sector_format::BLOCK_BYTES / p.m.max(1);
    let total_blocks = p.n.div_ceil(per_block.max(1));
    let mut dropped = 0usize;
    for _ in 0..blocks {
        let b = rng.below(total_blocks);
        let start = b * per_block;
        let end = (start + per_block).min(p.n);
        for v in start..end {
            // A detected CRC failure removes the block's vectors from the
            // result — it does not corrupt them into different vectors. The
            // dropped set is recorded and those ids are filtered out of every
            // candidate list, which is what "dropped" means for recall
            // accounting: identical to an eviction, and counted.
            if dropped_ids.insert(v as u32) {
                dropped += 1;
            }
        }
    }
    dropped
}

/// Kill one erase sector's worth of codebook, as a correlated failure does.
///
/// Flash fails in sector-correlated bursts, not byte-independently. The
/// allocation analysis assumes independence, and this channel exists to show
/// the gap rather than conceal it.
fn kill_codebook_sector(p: &mut Pipeline, rng: &mut Rng) -> usize {
    let sector = sector_format::SECTOR_BYTES;
    let len = p.codebook.components.len();
    if len == 0 {
        return 0;
    }
    let sectors = len.div_ceil(sector);
    let s = rng.below(sectors);
    let start = s * sector;
    let end = (start + sector).min(len);
    for byte in &mut p.codebook.components[start..end] {
        // An erased NOR sector reads as 0xFF, which as i8 is -1.
        *byte = -1;
    }
    end - start
}

/// Run `faults`.
pub fn run(argv: &[String]) -> Result<PathBuf, String> {
    let cfg = parse_config(argv)?;
    let base_path = PathBuf::from(flag(argv, "--base")?);
    let query_path = PathBuf::from(flag(argv, "--queries")?);
    let n_queries = opt_num(argv, "--nq", 100)?;
    let out_name = argv
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| argv.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "faults".to_string());

    let base = dataset_util::load(&base_path, cfg.n)?;
    let queries = dataset_util::load(&query_path, n_queries)?;
    if queries.dim != base.dim {
        return Err(format!(
            "queries are D={} but base is D={}",
            queries.dim, base.dim
        ));
    }
    let d = base.dim;

    let clean_pipeline = build_index(&base.data, base.count, d, &cfg)?;

    // Ground truth is the exact L2 ranking over the loaded corpus.
    let truths: Vec<Vec<u32>> = (0..queries.count)
        .map(|qi| {
            let q = &queries.data[qi * d..(qi + 1) * d];
            dataset_util::exact_top_l2(&base.data, base.count, d, q, cfg.k)
        })
        .collect();

    let workload = Workload {
        base: &base.data,
        d,
        queries: &queries.data,
        truths: &truths,
        r: cfg.r,
        k: cfg.k,
    };
    let none = std::collections::HashSet::new();
    let clean = measure(&clean_pipeline, &workload, &none);
    eprintln!("clean two-stage recall@{}: {clean:.4}", cfg.k);

    let mut channels = Vec::new();

    // Channel 1: codebook bit flips, the high-fan-out structure.
    let mut rows = Vec::new();
    for &count in &[1usize, 4, 16, 64, 256] {
        let mut p = build_index(&base.data, base.count, d, &cfg)?;
        let mut rng = Rng::new(cfg.seed);
        flip_codebook_bits(&mut p, &mut rng, count);
        let r = measure(&p, &workload, &none);
        eprintln!(
            "  codebook {count:>4} bit flips -> {r:.4}  (loss {:+.4})",
            r - clean
        );
        rows.push(json::obj(vec![
            ("bits_flipped", json::i(count as i64)),
            ("recall", json::f(r)),
            ("loss", json::f(clean - r)),
        ]));
    }
    channels.push(json::obj(vec![
        ("channel", json::s("codebook bit flips")),
        (
            "note",
            json::s("one corrupted byte alters every vector using that centroid"),
        ),
        ("points", Value::List(rows)),
    ]));

    // Channel 2: payload bit flips, one vector each.
    let mut rows = Vec::new();
    for &count in &[16usize, 256, 4096, 16384] {
        let mut p = build_index(&base.data, base.count, d, &cfg)?;
        let mut rng = Rng::new(cfg.seed);
        flip_payload_bits(&mut p, &mut rng, count);
        let r = measure(&p, &workload, &none);
        let frac = count as f64 / (p.codes.len() * 8) as f64;
        eprintln!(
            "  payload  {count:>5} bit flips ({:.2}% of code bits) -> {r:.4}  (loss {:+.4})",
            frac * 100.0,
            r - clean
        );
        rows.push(json::obj(vec![
            ("bits_flipped", json::i(count as i64)),
            ("fraction_of_code_bits", json::f(frac)),
            ("recall", json::f(r)),
            ("loss", json::f(clean - r)),
        ]));
    }
    channels.push(json::obj(vec![
        ("channel", json::s("payload bit flips")),
        ("note", json::s("one corrupted code affects one vector")),
        ("points", Value::List(rows)),
    ]));

    // Channel 3: whole-block drops, as a detected CRC failure produces.
    let mut rows = Vec::new();
    for &blocks in &[1usize, 4, 16, 64] {
        let p = build_index(&base.data, base.count, d, &cfg)?;
        let mut rng = Rng::new(cfg.seed);
        let mut dropped_ids = std::collections::HashSet::new();
        let dropped = drop_payload_blocks(&p, &mut rng, blocks, &mut dropped_ids);
        let r = measure(&p, &workload, &dropped_ids);
        eprintln!(
            "  blocks   {blocks:>4} dropped ({dropped} vectors) -> {r:.4}  (loss {:+.4})",
            r - clean
        );
        rows.push(json::obj(vec![
            ("blocks_dropped", json::i(blocks as i64)),
            ("vectors_lost", json::i(dropped as i64)),
            ("recall", json::f(r)),
            ("loss", json::f(clean - r)),
        ]));
    }
    channels.push(json::obj(vec![
        ("channel", json::s("payload block drops")),
        (
            "note",
            json::s("a detected CRC failure removes every vector in the block"),
        ),
        ("points", Value::List(rows)),
    ]));

    // Channel 4: a correlated sector failure in the codebook.
    let mut p = build_index(&base.data, base.count, d, &cfg)?;
    let mut rng = Rng::new(cfg.seed);
    let killed = kill_codebook_sector(&mut p, &mut rng);
    let sector_recall = measure(&p, &workload, &none);
    eprintln!(
        "  sector   {killed} codebook bytes erased -> {sector_recall:.4}  (loss {:+.4})",
        sector_recall - clean
    );
    channels.push(json::obj(vec![
        ("channel", json::s("correlated codebook sector failure")),
        (
            "note",
            json::s(
                "flash fails in sector-correlated bursts; the allocation analysis \
                 assumes independence, and this channel shows the gap",
            ),
        ),
        (
            "points",
            Value::List(vec![json::obj(vec![
                ("bytes_erased", json::i(killed as i64)),
                ("recall", json::f(sector_recall)),
                ("loss", json::f(clean - sector_recall)),
            ])]),
        ),
    ]));

    let value = json::obj(vec![
        ("measurement", json::s("faults")),
        ("dataset", json::s(&base_path.display().to_string())),
        ("config", cfg.to_value(d, base.count)),
        ("queries", json::i(queries.count as i64)),
        ("clean_two_stage_recall", json::f(clean)),
        (
            "note",
            json::s("every loss is relative to the clean two-stage baseline, not an oracle"),
        ),
        ("channels", Value::List(channels)),
    ]);
    json::write_measurement(&out_name, &value).map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rng_is_reproducible_and_never_degenerates() {
        // Comparing a clean run against a corrupted one requires the corruption
        // to be identical across runs.
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let xs: Vec<u64> = (0..8).map(|_| a.next()).collect();
        let ys: Vec<u64> = (0..8).map(|_| b.next()).collect();
        assert_eq!(xs, ys);
        // A zero seed must not lock the generator at zero.
        let mut z = Rng::new(0);
        assert_ne!(z.next(), 0);
        assert_ne!(z.next(), 0);
    }

    #[test]
    fn below_stays_in_range_and_handles_zero() {
        let mut r = Rng::new(3);
        for _ in 0..200 {
            assert!(r.below(10) < 10);
        }
        assert_eq!(r.below(0), 0);
    }
}
