//! `sector query` — run queries against an image on the host.
//!
//! The host path and the device path must return identical results on the same
//! image. This command is the host half of that comparison; the device half is
//! the firmware's UART shell running the same queries.

use sector_format::manifest::{self, Manifest, MANIFEST_BYTES};
use sector_format::region::RegionKind;
use sector_format::BLOCK_BYTES;
use std::path::PathBuf;

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
    /// Their scores.
    pub scores: Vec<i32>,
}

/// Run `query`, returning the answers so a test can compare two paths.
pub fn answers(args: &Args) -> Result<Vec<Answer>, String> {
    let bytes = std::fs::read(&args.image).map_err(|e| format!("{e}"))?;
    if bytes.len() < 2 * MANIFEST_BYTES {
        return Err("image shorter than two manifest slots".into());
    }
    let mut a = [0u8; MANIFEST_BYTES];
    let mut b = [0u8; MANIFEST_BYTES];
    a.copy_from_slice(&bytes[..MANIFEST_BYTES]);
    b.copy_from_slice(&bytes[MANIFEST_BYTES..2 * MANIFEST_BYTES]);
    let m: Manifest = manifest::select(&a, &b).map_err(|e| format!("{e:?}"))?;

    let d = m.d as usize;
    let mv = m.m as usize;
    let k_centroids = 1usize << m.b;
    let ds = d / mv.max(1);
    let r = if args.r == 0 { m.r as usize } else { args.r };

    let cb_region = m
        .table
        .get(RegionKind::Codebook)
        .ok_or("image has no codebook region")?;
    let payload_region = m
        .table
        .get(RegionKind::Payload)
        .ok_or("image has no payload region")?;

    // Codebook components, as stored.
    let cb_len = mv * k_centroids * ds;
    let cb_at = cb_region.base as usize;
    let codebook: Vec<i8> = bytes
        .get(cb_at..cb_at + cb_len)
        .ok_or("codebook region runs past the image")?
        .iter()
        .map(|x| *x as i8)
        .collect();

    // Payload codes, unpacked from their blocks.
    let per_block = BLOCK_BYTES / mv.max(1);
    let n = m.n as usize;
    let mut codes = vec![0u8; n * mv];
    for v in 0..n {
        let at =
            payload_region.base as usize + (v / per_block) * BLOCK_BYTES + (v % per_block) * mv;
        let src = bytes
            .get(at..at + mv)
            .ok_or("payload region runs past the image")?;
        codes[v * mv..(v + 1) * mv].copy_from_slice(src);
    }

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
        // Stage one: score every vector against its reconstruction.
        let mut scored: Vec<(i32, u32)> = (0..n)
            .map(|v| {
                let mut s = 0i32;
                for j in 0..mv {
                    let c = codes[v * mv + j] as usize;
                    let at = (j * k_centroids + c) * ds;
                    for i in 0..ds {
                        let comp = codebook.get(at + i).copied().unwrap_or(0) as i32;
                        let qc = q.get(j * ds + i).copied().unwrap_or(0.0) as i32;
                        s += comp * qc;
                    }
                }
                (s, v as u32)
            })
            .collect();
        // Ties break by id: irreproducible ordering makes a host/device
        // comparison meaningless.
        scored.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
        scored.truncate(r.min(scored.len()));
        scored.truncate(args.k.min(scored.len()));

        out.push(Answer {
            query: qi,
            ids: scored.iter().map(|(_, id)| *id).collect(),
            scores: scored.iter().map(|(s, _)| *s).collect(),
        });
    }
    Ok(out)
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
    println!("\n{} queries, k={}", out.len(), args.k);
    Ok(())
}
