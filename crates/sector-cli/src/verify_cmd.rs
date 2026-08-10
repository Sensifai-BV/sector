//! `sector verify` and `sector repair` — integrity on demand.
//!
//! The engine checks a block only when a query selects a candidate in it, so
//! damage in a cold region stays undetected for as long as nothing looks there.
//! These commands sweep without waiting for a query, which is what a cron entry
//! or a pre-flight check needs.
//!
//! # Exit codes carry the verdict
//!
//! `verify` exits 0 on a clean volume and 1 on a damaged one, so a monitoring
//! check is `sector verify --image X || alert`. A read error exits 2, because
//! "could not check" and "checked and found damage" call for different responses
//! and collapsing them into one code loses that.
//!
//! # Damage is reported in vectors, not blocks
//!
//! A block count is not actionable. What an operator needs is how many vectors
//! now return wrong candidates, how many candidates will be dropped, and whether
//! the volume can be repaired in place or must be rebuilt from the corpus. Those
//! three numbers have different magnitudes for the same one flipped byte,
//! depending on which region it landed in.

use std::path::PathBuf;

use sector_format::region::RegionKind;
use sector_os::json::Json;
use sector_os::verify::{verify, CodebookStatus, VerifyReport};
use sector_os::{FileFlash, HostVolume};

/// `verify` arguments.
pub struct Args {
    /// Volume to check.
    pub image: PathBuf,
    /// Emit JSON rather than text.
    pub json: bool,
}

/// Parse `verify` arguments.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    Ok(Args {
        image: PathBuf::from(crate::flag(argv, "--image")?),
        json: argv.iter().any(|a| a == "--json"),
    })
}

/// Sweep the volume and report. Returns the number of damaged regions.
pub fn run(args: Args) -> Result<usize, String> {
    let mut flash = FileFlash::open(&args.image).map_err(|e| format!("{e}"))?;
    let volume = HostVolume::mount(&mut flash, None).map_err(|e| format!("{e}"))?;
    let report = verify(&mut flash, &volume).map_err(|e| format!("{e}"))?;

    let damaged = report
        .regions
        .iter()
        .filter(|r| !r.bad_blocks.is_empty())
        .count();

    if args.json {
        print!("{}", render_json(&args, &report));
    } else {
        render_text(&args, &report);
    }
    Ok(damaged)
}

/// Render the report as text.
fn render_text(args: &Args, r: &VerifyReport) {
    println!("volume     {}", args.image.display());
    println!("vectors    {}", r.n);
    println!();
    println!("region checks");
    for region in &r.regions {
        let how = if region.checked {
            "crc"
        } else {
            // The codebook: compared against its replica, which detects a
            // disagreement without attributing it. Naming the method keeps a
            // reader from reading this as a checksum result.
            "replica comparison"
        };
        let verdict = if region.bad_blocks.is_empty() {
            "ok".to_string()
        } else {
            format!(
                "{} of {} blocks bad",
                region.bad_blocks.len(),
                region.blocks
            )
        };
        println!(
            "  {:<18} {:<20} {}",
            format!("{:?}", region.kind),
            how,
            verdict
        );
    }

    println!();
    println!("consequences");
    let codes = r.vectors_with_bad_codes();
    let drops = r.candidates_that_will_drop();
    let recon = r.vectors_with_bad_reconstruction();
    println!("  vectors with wrong codes        {codes}");
    println!("  candidates that will be dropped {drops}");
    println!("  vectors misreconstructed        {recon}");

    if codes > 0 {
        // The regime that does not announce itself: a payload CRC failure is
        // detected at scan time and the scan does not stop, so these vectors
        // return a wrong candidate set with nothing in the result to say so.
        println!("  note: payload damage is silent in query results");
    }

    println!();
    match &r.codebook {
        CodebookStatus::Agree => println!("codebook   both copies identical"),
        CodebookStatus::NoReplica => println!("codebook   no replica region to compare against"),
        CodebookStatus::Disagree { blocks } => {
            println!("codebook   copies differ in {} block(s)", blocks.len());
            // There is no codebook CRC, so neither copy can be shown correct.
            // Guessing would repair a good block from a bad one half the time.
            println!("           neither copy can be shown correct: the format stores no");
            println!("           codebook CRC, so a disagreement is not attributable.");
            println!("           rebuild from the source corpus rather than repairing.");
        }
    }

    println!();
    if r.is_clean() {
        println!("verdict    clean");
    } else {
        println!("verdict    damaged");
    }
}

/// Render the report as JSON.
fn render_json(args: &Args, r: &VerifyReport) -> String {
    let mut j = Json::new();
    j.object(|o| {
        o.str("image", &args.image.display().to_string());
        o.uint("vectors", r.n as u64);
        o.bool("clean", r.is_clean());
        o.array("regions", |a| {
            for region in &r.regions {
                a.object(|ro| {
                    ro.str("kind", &format!("{:?}", region.kind));
                    ro.uint("blocks", region.blocks as u64);
                    ro.bool("crc_checked", region.checked);
                    ro.uints("bad_blocks", region.bad_blocks.iter().map(|b| *b as u64));
                });
            }
        });
        o.object("consequences", |c| {
            c.uint(
                "vectors_with_wrong_codes",
                r.vectors_with_bad_codes() as u64,
            );
            c.uint("candidates_dropped", r.candidates_that_will_drop() as u64);
            c.uint(
                "vectors_misreconstructed",
                r.vectors_with_bad_reconstruction() as u64,
            );
            c.bool("payload_damage_is_silent", r.vectors_with_bad_codes() > 0);
        });
        o.object("codebook", |c| match &r.codebook {
            CodebookStatus::Agree => {
                c.str("status", "agree");
                c.bool("attributable", false);
            }
            CodebookStatus::NoReplica => {
                c.str("status", "no_replica");
                c.bool("attributable", false);
            }
            CodebookStatus::Disagree { blocks } => {
                c.str("status", "disagree");
                c.uints("blocks", blocks.iter().map(|b| *b as u64));
                // No codebook CRC exists, so a disagreement cannot be resolved.
                c.bool("attributable", false);
                c.str("remedy", "rebuild from the source corpus");
            }
        });
    });
    j.finish()
}

/// `repair` arguments.
pub struct RepairArgs {
    /// Volume to repair.
    pub image: PathBuf,
    /// Report what would change without writing.
    pub dry_run: bool,
}

/// Parse `repair` arguments.
pub fn parse_repair(argv: &[String]) -> Result<RepairArgs, String> {
    Ok(RepairArgs {
        image: PathBuf::from(crate::flag(argv, "--image")?),
        dry_run: argv.iter().any(|a| a == "--dry-run"),
    })
}

/// Attempt in-place repair.
///
/// # What this can and cannot do
///
/// Nothing, at present, and it says so rather than pretending.
///
/// Repair needs two things: damage localised to one copy, and a way to tell
/// which copy is good. The format provides the first — replicas exist — and not
/// the second, because there is no codebook CRC. Payload and rerank have CRCs and
/// no replicas, so their damage is precisely localised and unrecoverable.
///
/// So the honest surface is a command that diagnoses and refuses. Writing a
/// repair that picks the primary copy on a coin flip would convert
/// single-copy damage into total loss half the time, and a tool that reports
/// success on a volume it has not fixed is worse than one that reports nothing.
pub fn run_repair(args: RepairArgs) -> Result<(), String> {
    let mut flash = FileFlash::open(&args.image).map_err(|e| format!("{e}"))?;
    let volume = HostVolume::mount(&mut flash, None).map_err(|e| format!("{e}"))?;
    let report = verify(&mut flash, &volume).map_err(|e| format!("{e}"))?;

    if report.is_clean() {
        println!("nothing to repair: {} is clean", args.image.display());
        return Ok(());
    }

    println!("volume     {}", args.image.display());
    println!();

    let payload_bad = report
        .bad_in(RegionKind::Payload)
        .map(|b| b.len())
        .unwrap_or(0);
    let rerank_bad = report
        .bad_in(RegionKind::Rerank)
        .map(|b| b.len())
        .unwrap_or(0);

    if payload_bad > 0 || rerank_bad > 0 {
        println!("unrepairable damage");
        if payload_bad > 0 {
            println!(
                "  payload  {payload_bad} block(s), {} vector(s) with wrong codes",
                report.vectors_with_bad_codes()
            );
        }
        if rerank_bad > 0 {
            println!(
                "  rerank   {rerank_bad} block(s), {} candidate(s) will drop",
                report.candidates_that_will_drop()
            );
        }
        // These regions carry a CRC and no replica: damage is localised exactly
        // and there is no second copy to restore from. The CRC bought detection,
        // not correction, which is the trade the format makes deliberately —
        // a payload replica would double the volume.
        println!("  these regions have a CRC and no replica: detectable, not recoverable");
    }

    if let CodebookStatus::Disagree { blocks } = &report.codebook {
        println!();
        println!("codebook   copies differ in {} block(s)", blocks.len());
        println!("  the format stores no codebook CRC, so neither copy can be shown");
        println!("  correct. repairing from the wrong copy would turn one-copy damage");
        println!("  into total loss, so this is refused rather than guessed.");
    }

    println!();
    println!("remedy     rebuild the volume from its source corpus:");
    println!(
        "           sector build --input <corpus> --out {}",
        args.image.display()
    );
    if args.dry_run {
        println!("(dry run: nothing was written, and nothing would have been)");
    }

    Err("no repair is possible for this damage".to_string())
}
