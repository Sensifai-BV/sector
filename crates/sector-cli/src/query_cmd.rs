//! `sector query` — run queries against an image on the host.
//!
//! The host path and the device path must return identical results on the same
//! image. This command is the host half of that comparison; the device half is
//! the firmware's UART shell running the same queries.
//!
//! # This command runs the engine, and previously did not
//!
//! It calls `sector_core::query` through `sector_os::Searcher` — the same mount,
//! the same scan, the same CRC-verified rerank the firmware executes.
//!
//! An earlier version of this file reimplemented stage one here: it scored every
//! vector with an `f32` inner product against the dequantized codebook, sorted
//! all `N`, truncated to `r` and then to `k`, and returned. It therefore
//! performed no rerank, verified no CRC, and could not drop a candidate, while
//! its own documentation claimed to be byte-identical to the device path. A
//! host/device recall discrepancy would have been attributed to hardware.
//!
//! `sector_bench::pipeline` records having made and fixed the same mistake. This
//! is the second instance, which is why the round-trip test now asserts that
//! stage two actually ran — a claim of equivalence that nothing checks decays
//! back to this.

use std::path::PathBuf;

use sector_os::search::{SearchError, Searcher};
use sector_os::FileFlash;

/// Parsed `query` arguments.
pub struct Args {
    /// Image to query.
    pub image: PathBuf,
    /// Query vectors in `.fvecs`.
    pub queries: PathBuf,
    /// Results per query.
    pub k: usize,
    /// Candidate depth. 0 uses the image's own value.
    pub r: usize,
    /// Queries to run. 0 means all.
    pub limit: usize,
}

/// Parse `query` flags.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    Ok(Args {
        image: PathBuf::from(crate::flag(argv, "--image")?),
        queries: PathBuf::from(crate::flag(argv, "--queries")?),
        k: crate::opt_num(argv, "--k", 10)?,
        r: crate::opt_num(argv, "--r", 0)?,
        limit: crate::opt_num(argv, "--limit", 0)?,
    })
}

/// One query's result.
pub struct Answer {
    /// Query index.
    pub query: usize,
    /// Result ids, best first.
    pub ids: Vec<u32>,
    /// Their scores, from the exact rescore against the rerank copy.
    pub scores: Vec<i32>,
    /// Candidates dropped on a CRC mismatch.
    ///
    /// Reported per query because a drop and an eviction are indistinguishable
    /// in the result: without this number a recall regression cannot be
    /// attributed to corruption.
    pub dropped: u32,
    /// Blocks whose CRC was verified.
    pub blocks_verified: u32,
    /// Vectors stage one examined. Equal to `N` on a complete scan.
    pub scanned: u32,
}

/// Run `query`, returning the answers so a test can compare two paths.
pub fn answers(args: &Args) -> Result<Vec<Answer>, String> {
    let depth = if args.r == 0 { None } else { Some(args.r) };
    let mut searcher: Searcher<FileFlash> = Searcher::open(&args.image, depth).map_err(describe)?;
    let d = searcher.geometry().d;

    let mut reader =
        sector_build::dataset::VecsReader::open(&args.queries).map_err(|e| format!("{e:?}"))?;
    if reader.layout().dim as usize != d {
        return Err(format!(
            "queries are D={} but the image is D={d}",
            reader.layout().dim
        ));
    }
    let count = if args.limit == 0 {
        reader.len()
    } else {
        args.limit.min(reader.len())
    };

    let mut out = Vec::with_capacity(count);
    let mut q = vec![0f32; d];
    for qi in 0..count {
        if reader
            .next_f32(&mut q)
            .map_err(|e| format!("{e:?}"))?
            .is_none()
        {
            break;
        }
        let a = searcher.search(&q, args.k).map_err(describe)?;
        out.push(Answer {
            query: qi,
            ids: a.ids,
            scores: a.scores,
            dropped: a.stats.rerank.dropped,
            blocks_verified: a.stats.rerank.blocks_verified,
            scanned: a.stats.scan.scanned,
        });
    }
    Ok(out)
}

/// Render a search error, keeping the underlying cause.
fn describe(e: SearchError) -> String {
    e.to_string()
}

/// Run `query` and print the answers.
pub fn run(args: Args) -> Result<(), String> {
    let out = answers(&args)?;
    for a in &out {
        let pairs: Vec<String> = a
            .ids
            .iter()
            .zip(a.scores.iter())
            .map(|(id, s)| format!("{id}:{s}"))
            .collect();
        println!("q{:<4} {}", a.query, pairs.join(" "));
    }

    // Drops are surfaced rather than buried: a volume returning fewer than `k`
    // results is reporting damage, and a caller that does not see the count
    // reads it as poor recall.
    let dropped: u32 = out.iter().map(|a| a.dropped).sum();
    let verified: u32 = out.iter().map(|a| a.blocks_verified).sum();
    println!("\n{} queries, k={}", out.len(), args.k);
    println!("{verified} blocks CRC-verified, {dropped} candidates dropped");
    if dropped > 0 {
        println!("a non-zero drop count means corruption; run `sector verify --image`");
    }

    // A scan that did not reach every vector reduces recall without failing, so
    // the shortfall is reported rather than left to be inferred from the answers.
    // Every query scans the whole corpus, so any two disagreeing — or any one
    // below `N` — means the payload reader stopped early.
    if let Some(short) = out.iter().find(|a| a.scanned != out[0].scanned) {
        println!(
            "warning: query {} scanned {} vectors where query 0 scanned {}",
            short.query, short.scanned, out[0].scanned
        );
    }
    if let Some(a) = out.first() {
        println!("{} vectors scanned per query", a.scanned);
    }
    Ok(())
}
