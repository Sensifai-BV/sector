//! `sector stats` — what a volume costs and what it will do.
//!
//! `inspect` describes the image's layout; this describes its *behaviour*: bytes
//! resident per worker, bytes read per query, and the latency breakdown by phase.
//! An operator sizing a Pi deployment needs the second set.
//!
//! # Measured, not derived
//!
//! Every figure here comes from running queries and reading the counters the
//! backend and the engine keep. The alternative — computing `R * rerank_bytes`
//! and calling it the per-query read volume — is wrong in a way that matters on
//! managed storage: a 128 B record is fetched as a 512 B block because that is
//! the CRC granularity, so the real figure is 4x the naive one. That ratio is
//! the point of measuring rather than predicting.
//!
//! # Why the warm-up pass is excluded
//!
//! The first pass over a volume populates the page cache, and on a Pi that is a
//! read per block through the flash translation layer. Including it would report
//! a cold-cache figure as if it were steady state. Counters are reset after the
//! warm-up and the reported numbers are from the measured passes, with the
//! warm-up cost reported separately so the difference is visible rather than
//! hidden.

use std::path::PathBuf;
use std::time::Instant;

use sector_hal::{Edge, Instrument, Phase};
use sector_os::json::Json;
use sector_os::search::Searcher;
use sector_os::FileFlash;

/// `stats` arguments.
pub struct Args {
    /// Volume to profile.
    pub image: PathBuf,
    /// Queries to run. Synthetic when absent.
    pub queries: Option<PathBuf>,
    /// Queries to time.
    pub count: usize,
    /// Results per query.
    pub k: usize,
    /// Emit JSON.
    pub json: bool,
}

/// Parse `stats` arguments.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    Ok(Args {
        image: PathBuf::from(crate::flag(argv, "--image")?),
        queries: argv
            .iter()
            .position(|a| a == "--queries")
            .and_then(|i| argv.get(i + 1))
            .map(PathBuf::from),
        count: crate::opt_num(argv, "--count", 50)?,
        k: crate::opt_num(argv, "--k", 10)?,
        json: argv.iter().any(|a| a == "--json"),
    })
}

/// Wall-clock time per phase, in nanoseconds.
#[derive(Default)]
struct PhaseTimer {
    entered: [Option<Instant>; 5],
    total_ns: [u128; 5],
    entries: [u32; 5],
}

impl Instrument for PhaseTimer {
    fn cycles(&self) -> u64 {
        0
    }
    fn mark(&mut self, phase: Phase, edge: Edge) {
        let i = sector_core::metrics::phase_index(phase);
        match edge {
            Edge::Enter => {
                self.entered[i] = Some(Instant::now());
                self.entries[i] += 1;
            }
            Edge::Leave => {
                if let Some(at) = self.entered[i].take() {
                    self.total_ns[i] += at.elapsed().as_nanos();
                }
            }
        }
    }
}

impl PhaseTimer {
    /// Mean nanoseconds per entry for `phase`.
    fn mean_ns(&self, phase: usize) -> f64 {
        if self.entries[phase] == 0 {
            return 0.0;
        }
        self.total_ns[phase] as f64 / self.entries[phase] as f64
    }
}

/// Names in the order [`sector_core::metrics::phase_index`] uses.
const PHASE_NAMES: [&str; 5] = ["rotate", "table", "scan", "rerank", "finalize"];

/// Everything measured.
struct Measured {
    n: usize,
    d: usize,
    m: usize,
    centroids: usize,
    payload_bytes: usize,
    rerank_bytes: usize,
    /// Flash cost of one stored vector, in hundredths of a byte.
    ///
    /// From `Geometry`, not recomputed here: deriving it locally is what made this
    /// command and `inspect` report 160 B and 168 B for the same volume.
    stored_centi: usize,
    resident: usize,
    depth: usize,
    queries: usize,
    /// Backend counters over the measured passes.
    reads: u64,
    bytes: u64,
    blocks_touched: u64,
    straddling: u64,
    /// The warm-up pass, reported separately.
    warmup_reads: u64,
    warmup_bytes: u64,
    total_ns: u128,
    timer: PhaseTimer,
    dropped: u32,
    verified: u32,
    scanned: u32,
}

/// Run the measurement.
pub fn run(args: Args) -> Result<(), String> {
    let mut searcher: Searcher<FileFlash> =
        Searcher::open(&args.image, None).map_err(|e| format!("{e}"))?;
    let g = *searcher.geometry();

    // Queries: from a file, or synthesised deterministically so a run without a
    // dataset still measures the same thing twice.
    let queries: Vec<Vec<f32>> = match &args.queries {
        Some(path) => {
            let mut reader =
                sector_build::dataset::VecsReader::open(path).map_err(|e| format!("{e:?}"))?;
            if reader.layout().dim as usize != g.d {
                return Err(format!(
                    "queries are D={} but the volume is D={}",
                    reader.layout().dim,
                    g.d
                ));
            }
            let take = args.count.min(reader.len());
            let mut out = Vec::with_capacity(take);
            let mut buf = vec![0f32; g.d];
            for _ in 0..take {
                match reader.next_f32(&mut buf).map_err(|e| format!("{e:?}"))? {
                    Some(_) => out.push(buf.clone()),
                    None => break,
                }
            }
            out
        }
        None => (0..args.count)
            .map(|q| {
                (0..g.d)
                    .map(|j| (((q * 31 + j * 17) % 97) as f32) - 48.0)
                    .collect()
            })
            .collect(),
    };
    if queries.is_empty() {
        return Err("no queries to run".into());
    }

    // Warm-up: one full pass, counted separately. On a Pi this is where the page
    // cache fills, and folding it into the reported figures would present a
    // cold-cache number as steady state.
    let (warmup_reads, warmup_bytes) = {
        let mut sink = PhaseTimer::default();
        for q in &queries {
            let quantized =
                sector_os::search::quantize_query(q, g.d).map_err(|e| format!("{e}"))?;
            searcher
                .search_instrumented(&quantized, args.k, &mut sink)
                .map_err(|e| format!("{e}"))?;
        }
        // Read the backend's counters through a fresh handle on the same file:
        // the searcher owns its handles, so its own counters are what matter and
        // are read below. This block exists to warm the cache.
        // The searcher's own counters after the warm-up pass: this is the cost
        // of filling the page cache, reported separately so the steady-state
        // figures below are not a cold-cache measurement in disguise.
        let s = searcher.backend_stats();
        (s.reads, s.bytes)
    };

    searcher.reset_backend_stats();
    let mut timer = PhaseTimer::default();
    let mut dropped = 0u32;
    let mut verified = 0u32;
    let mut scanned = 0u32;

    let start = Instant::now();
    for q in &queries {
        let quantized = sector_os::search::quantize_query(q, g.d).map_err(|e| format!("{e}"))?;
        let a = searcher
            .search_instrumented(&quantized, args.k, &mut timer)
            .map_err(|e| format!("{e}"))?;
        dropped += a.stats.rerank.dropped;
        verified += a.stats.rerank.blocks_verified;
        scanned = a.stats.scan.scanned;
    }
    let total_ns = start.elapsed().as_nanos();
    let backend = searcher.backend_stats();

    let measured = Measured {
        n: g.n,
        d: g.d,
        m: g.m,
        centroids: g.centroids,
        payload_bytes: g.payload_bytes,
        rerank_bytes: g.rerank_bytes,
        stored_centi: g.stored_bytes_per_vector_centi(),
        resident: searcher.resident_bytes(),
        depth: searcher.depth(),
        queries: queries.len(),
        reads: backend.reads,
        bytes: backend.bytes,
        blocks_touched: backend.blocks_touched,
        straddling: backend.straddling_reads,
        warmup_reads,
        warmup_bytes,
        total_ns,
        timer,
        dropped,
        verified,
        scanned,
    };

    if args.json {
        print!("{}", render_json(&args, &measured));
    } else {
        render_text(&args, &measured);
    }
    Ok(())
}

fn render_text(args: &Args, s: &Measured) {
    let q = s.queries as f64;
    println!("volume     {}", args.image.display());
    println!(
        "profile    D={} m={} 2^b={} N={} R={}",
        s.d, s.m, s.centroids, s.n, s.depth
    );
    println!();

    println!("memory (per worker)");
    println!("  resident            {} B", s.resident);
    println!(
        "  stored per vector   {}.{:02} B  (payload + rerank + CRC share)",
        s.stored_centi / 100,
        s.stored_centi % 100
    );
    println!("  volume working set  {} B", s.n * s.stored_centi / 100);
    println!();

    println!("latency ({} queries, warm)", s.queries);
    println!(
        "  total per query     {:.1} us",
        s.total_ns as f64 / q / 1000.0
    );
    for (i, name) in PHASE_NAMES.iter().enumerate() {
        println!("  {name:<19} {:.1} us", s.timer.mean_ns(i) / 1000.0);
    }
    println!();

    println!("storage per query");
    println!("  reads               {:.1}", s.reads as f64 / q);
    println!("  bytes               {:.1}", s.bytes as f64 / q);
    println!("  device blocks       {:.1}", s.blocks_touched as f64 / q);
    println!("  straddling reads    {:.1}", s.straddling as f64 / q);
    // The amplification the CRC granularity creates. A rerank fetch reads a whole
    // 512 B block to score a record smaller than it, and this ratio is what a
    // per-candidate byte estimate misses.
    let fetched = s.depth as f64 * s.rerank_bytes as f64;
    if fetched > 0.0 {
        let amplification = (s.bytes as f64 / q) / fetched;
        println!("  read amplification  {amplification:.2}x vs R x rerank_bytes");
    }
    println!();

    println!("verification");
    println!(
        "  blocks verified     {:.1} per query",
        s.verified as f64 / q
    );
    println!("  candidates dropped  {}", s.dropped);
    println!("  vectors scanned     {}", s.scanned);
    println!();

    println!("warm-up (excluded above)");
    println!("  reads               {}", s.warmup_reads);
    println!("  bytes               {}", s.warmup_bytes);
}

fn render_json(args: &Args, s: &Measured) -> String {
    let q = s.queries as f64;
    let mut j = Json::new();
    j.object(|o| {
        o.str("image", &args.image.display().to_string());
        o.object("profile", |p| {
            p.uint("d", s.d as u64);
            p.uint("m", s.m as u64);
            p.uint("centroids", s.centroids as u64);
            p.uint("n", s.n as u64);
            p.uint("r", s.depth as u64);
            p.uint("payload_bytes", s.payload_bytes as u64);
            p.uint("rerank_bytes", s.rerank_bytes as u64);
        });
        o.object("memory", |m| {
            m.uint("resident_bytes_per_worker", s.resident as u64);
            m.uint("stored_bytes_per_vector", s.stored_centi as u64 / 100);
        });
        o.object("latency_us", |l| {
            l.float("total_per_query", s.total_ns as f64 / q / 1000.0);
            for (i, name) in PHASE_NAMES.iter().enumerate() {
                l.float(name, s.timer.mean_ns(i) / 1000.0);
            }
        });
        o.object("storage_per_query", |st| {
            st.float("reads", s.reads as f64 / q);
            st.float("bytes", s.bytes as f64 / q);
            st.float("device_blocks", s.blocks_touched as f64 / q);
            st.float("straddling_reads", s.straddling as f64 / q);
            let fetched = s.depth as f64 * s.rerank_bytes as f64;
            if fetched > 0.0 {
                st.float("read_amplification", (s.bytes as f64 / q) / fetched);
            }
        });
        o.object("verification", |v| {
            v.float("blocks_verified_per_query", s.verified as f64 / q);
            v.uint("candidates_dropped", s.dropped as u64);
            v.uint("vectors_scanned", s.scanned as u64);
        });
        o.object("warmup", |w| {
            w.uint("reads", s.warmup_reads);
            w.uint("bytes", s.warmup_bytes);
        });
        o.uint("queries", s.queries as u64);
    });
    j.finish()
}
