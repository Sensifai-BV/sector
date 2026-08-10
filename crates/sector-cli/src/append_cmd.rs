//! `sector append` and `sector vectors` — add and read back stored vectors.
//!
//! # Append is insert-only
//!
//! There is no delete and no update: an id, once written, is permanent. A validity
//! bitmap would add a per-candidate lookup to the scan loop, and insert-only growth
//! does not need one. `sector build` is how a vector is removed.
//!
//! # A short batch is padded, and the padding is reported
//!
//! Appends advance by whole blocks in both regions at once, so the batch size must
//! be a multiple of the append unit — 32 ids at `D=128, m=16`. A shorter batch is
//! padded with zero vectors and the count is printed, because silently rounding up
//! would leave the operator believing ids they did not supply hold their data.
//! Padding vectors are real stored vectors that will be returned by queries; they
//! are not phantoms.

use std::path::PathBuf;

use sector_os::json::Json;
use sector_os::{append, capacity, ingest, FileFlash, HostVolume};

/// `append` arguments.
pub struct Args {
    /// Volume to extend.
    pub image: PathBuf,
    /// Vectors to append, `.fvecs`.
    pub input: PathBuf,
    /// Emit JSON.
    pub json: bool,
    /// Report what would happen and write nothing.
    pub dry_run: bool,
}

/// `vectors` arguments.
pub struct VectorsArgs {
    /// Volume to read.
    pub image: PathBuf,
    /// Single id, if given.
    pub id: Option<u32>,
    /// First id of a range.
    pub from: u32,
    /// Ids to list.
    pub count: usize,
    /// Emit JSON.
    pub json: bool,
}

/// Parse `append` flags.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    Ok(Args {
        image: PathBuf::from(crate::flag(argv, "--image")?),
        input: PathBuf::from(crate::flag(argv, "--input")?),
        json: argv.iter().any(|a| a == "--json"),
        dry_run: argv.iter().any(|a| a == "--dry-run"),
    })
}

/// Parse `vectors` flags.
pub fn parse_vectors(argv: &[String]) -> Result<VectorsArgs, String> {
    let id = argv
        .iter()
        .position(|a| a == "--id")
        .and_then(|i| argv.get(i + 1))
        .map(|v| {
            v.parse::<u32>()
                .map_err(|_| "--id must be a number".to_string())
        })
        .transpose()?;
    Ok(VectorsArgs {
        image: PathBuf::from(crate::flag(argv, "--image")?),
        id,
        from: crate::opt_num(argv, "--from", 0)? as u32,
        count: crate::opt_num(argv, "--count", 10)?,
        json: argv.iter().any(|a| a == "--json"),
    })
}

/// Run `append`.
pub fn run(args: Args) -> Result<(), String> {
    let (geometry, manifest) = {
        let mut f = FileFlash::open(&args.image).map_err(|e| format!("{e}"))?;
        let v = HostVolume::mount(&mut f, None).map_err(|e| format!("{e}"))?;
        (v.geometry, v.manifest)
    };
    let unit = ingest::append_unit(&geometry);
    let (payload_slots, rerank_slots) = capacity(&args.image).map_err(|e| format!("{e}"))?;

    // The builder's own reader, rather than a second .fvecs parser here: two
    // parsers for one format would eventually disagree about a malformed file.
    let mut reader =
        sector_build::dataset::VecsReader::open(&args.input).map_err(|e| format!("{e:?}"))?;
    // The dimension is a property of the file, checked once. `next_f32` returns the
    // record's index, not its width.
    let found = reader.layout().dim as usize;
    if found != geometry.d {
        return Err(format!(
            "{} has {found}-component vectors, volume expects {}",
            args.input.display(),
            geometry.d
        ));
    }
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let mut row = vec![0f32; geometry.d];
    while reader
        .next_f32(&mut row)
        .map_err(|e| format!("{e:?}"))?
        .is_some()
    {
        vectors.push(row.clone());
    }
    let supplied = vectors.len();
    if supplied == 0 {
        return Err(format!("{} contains no vectors", args.input.display()));
    }

    // Pad to the append unit. Reported, never silent.
    let padded = supplied.next_multiple_of(unit) - supplied;
    for _ in 0..padded {
        vectors.push(vec![0.0f32; geometry.d]);
    }

    if args.dry_run {
        let head = manifest.n.next_multiple_of(unit as u32);
        println!("volume     {}", args.image.display());
        println!("would append");
        println!("  supplied          {supplied} vectors");
        println!("  padding           {padded} zero vectors (append unit is {unit})");
        println!("  first id          {head}");
        if head > manifest.n {
            println!(
                "  gap created       ids {}..{} would be absent ({} ids)",
                manifest.n,
                head,
                head - manifest.n
            );
        }
        println!("  new extent        {}", head + vectors.len() as u32);
        println!("capacity");
        println!("  payload slots     {payload_slots}");
        println!("  rerank slots      {rerank_slots}  (binds first)");
        println!("(dry run: nothing was written)");
        return Ok(());
    }

    let report = append(&args.image, &vectors).map_err(|e| format!("{e}"))?;

    if args.json {
        let mut j = Json::new();
        j.object(|o| {
            o.uint("first_id", report.first_id as u64);
            o.uint("count", report.count as u64);
            o.uint("supplied", supplied as u64);
            o.uint("padded", padded as u64);
            o.uint("n", report.n as u64);
            o.object("gap", |g| {
                g.uint("from", report.gap.0 as u64);
                g.uint("to", report.gap.1 as u64);
                g.uint("ids", (report.gap.1 - report.gap.0) as u64);
            });
            o.object("blocks", |b| {
                b.uint("payload", report.payload_blocks as u64);
                b.uint("rerank", report.rerank_blocks as u64);
            });
            o.object("drift", |d| {
                d.uint("appended", report.drift.appended as u64);
                d.uint("built", report.drift.built as u64);
                d.uint("appended_ppm", report.drift.appended_ppm());
                d.uint("mean_error_x1024", report.drift.mean_error_x1024());
                d.bool("warrants_rebuild", report.drift.warrants_rebuild(100_000));
            });
            o.object("remaining", |r| {
                r.uint("payload_slots", report.remaining.0 as u64);
                r.uint("rerank_slots", report.remaining.1 as u64);
            });
        });
        print!("{}", j.finish());
    } else {
        println!(
            "appended   {} vectors at id {}",
            report.count, report.first_id
        );
        if padded > 0 {
            println!(
                "  padding           {padded} zero vectors added to reach the {unit}-vector unit"
            );
            println!("                    these are stored and will be returned by queries");
        }
        println!(
            "  blocks written    {} payload, {} rerank",
            report.payload_blocks, report.rerank_blocks
        );
        println!("  extent            {} ids", report.n);
        let (from, to) = report.gap;
        if to > from {
            println!("  absent ids        {from}..{to} ({} ids)", to - from);
            println!("                    the built corpus did not end on a block boundary;");
            println!("                    these ids are addressable and not stored.");
        }
        println!("drift");
        println!(
            "  appended          {} of {} built",
            report.drift.appended, report.drift.built
        );
        println!("  appended share    {} ppm", report.drift.appended_ppm());
        println!(
            "  mean error        {}/1024",
            report.drift.mean_error_x1024()
        );
        // Appended vectors were encoded against a codebook trained without them, so
        // recall on them is not the recall the build measured.
        if report.drift.warrants_rebuild(100_000) {
            println!("  NOTE              appended vectors exceed 10% of the corpus.");
            println!("                    recall on them is not what the build measured;");
            println!("                    rebuild from the full corpus to restore it.");
        }
        println!("remaining");
        println!("  payload slots     {}", report.remaining.0);
        println!("  rerank slots      {}  (binds first)", report.remaining.1);
    }
    Ok(())
}

/// Run `vectors`.
pub fn run_vectors(args: VectorsArgs) -> Result<(), String> {
    let mut flash = FileFlash::open(&args.image).map_err(|e| format!("{e}"))?;
    let volume = HostVolume::mount(&mut flash, None).map_err(|e| format!("{e}"))?;
    let g = volume.geometry;
    let m = volume.manifest;

    let ids: Vec<u32> = match args.id {
        Some(id) => vec![id],
        None => (args.from..args.from.saturating_add(args.count as u32)).collect(),
    };

    let mut rows = Vec::new();
    for id in ids {
        // `holds` is the authority, not `id < n`: a volume with a gap has
        // addressable ids that are not stored.
        if !m.holds(id) {
            rows.push((id, None, if id < m.n { "absent" } else { "out of range" }));
            continue;
        }
        let Some(off) = g.rerank.offset_of(id as usize) else {
            rows.push((id, None, "out of range"));
            continue;
        };
        let mut rec = vec![0u8; g.rerank_bytes];
        sector_hal::NorFlash::read(&mut flash, volume.rerank_base() + off as u32, &mut rec)
            .map_err(|e| format!("{e}"))?;
        rows.push((id, Some(rec), "stored"));
    }

    if args.json {
        let mut j = Json::new();
        j.object(|o| {
            o.array("vectors", |a| {
                for (id, rec, status) in &rows {
                    a.object(|v| {
                        v.uint("id", *id as u64);
                        v.str("status", status);
                        match rec {
                            // int8 as stored, not rescaled to float: the record's
                            // scale is per-vector and is not in the manifest, so a
                            // float here would carry a scale nothing can verify.
                            Some(r) => v.ints("record", r.iter().map(|b| *b as i8 as i64)),
                            None => v.null("record"),
                        }
                    });
                }
            });
            o.object("volume", |v| {
                v.uint("n", m.n as u64);
                v.uint("stored", m.stored() as u64);
                v.uint("built_n", m.built_n as u64);
                v.uint("appended", m.appended() as u64);
                v.object("gap", |gg| {
                    gg.uint("from", m.gap().0 as u64);
                    gg.uint("to", m.gap().1 as u64);
                });
            });
        });
        print!("{}", j.finish());
    } else {
        println!(
            "volume     {} ids addressable, {} stored, {} appended",
            m.n,
            m.stored(),
            m.appended()
        );
        let (from, to) = m.gap();
        if to > from {
            println!("absent     ids {from}..{to}");
        }
        println!();
        for (id, rec, status) in &rows {
            match rec {
                Some(r) => {
                    let head: Vec<String> =
                        r.iter().take(8).map(|b| (*b as i8).to_string()).collect();
                    println!(
                        "  {id:<8} {status:<12} [{}{}]",
                        head.join(" "),
                        if r.len() > 8 { " ..." } else { "" }
                    );
                }
                None => println!("  {id:<8} {status}"),
            }
        }
    }
    Ok(())
}
