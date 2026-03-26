//! `sector inspect` — dump an image's manifest, regions and computed budgets.
//!
//! Budgets are recomputed from the manifest rather than read back from stored
//! header fields. The recurring failure in this project has been a
//! configuration that looks reasonable and does not fit; recomputing catches it
//! before the device does.

use sector_format::manifest::{self, Manifest, MANIFEST_BYTES};
use sector_format::region::RegionKind;
use sector_format::SECTOR_BYTES;
use std::path::PathBuf;

/// Parsed `inspect` arguments.
pub struct Args {
    /// Image to read.
    pub image: PathBuf,
}

/// Parse `inspect` flags.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    Ok(Args {
        image: PathBuf::from(crate::flag(argv, "--image")?),
    })
}

/// Run `inspect`.
pub fn run(args: Args) -> Result<(), String> {
    let bytes = std::fs::read(&args.image).map_err(|e| format!("{e}"))?;
    if bytes.len() < 2 * MANIFEST_BYTES {
        return Err(format!(
            "image is {} B, shorter than two manifest slots ({} B)",
            bytes.len(),
            2 * MANIFEST_BYTES
        ));
    }

    let mut a = [0u8; MANIFEST_BYTES];
    let mut b = [0u8; MANIFEST_BYTES];
    a.copy_from_slice(&bytes[..MANIFEST_BYTES]);
    b.copy_from_slice(&bytes[MANIFEST_BYTES..2 * MANIFEST_BYTES]);
    let m: Manifest = manifest::select(&a, &b).map_err(|e| format!("{e:?}"))?;

    println!("image      {} ({} B)", args.image.display(), bytes.len());
    println!(
        "profile    D={} m={} b={} N={} R={} (sequence {})",
        m.d, m.m, m.b, m.n, m.r, m.sequence
    );

    println!("\nregions");
    println!(
        "  {:<18} {:>10} {:>10} {:>8}  protection",
        "kind", "base", "bytes", "blocks"
    );
    for r in &m.table.regions {
        println!(
            "  {:<18} {:>10} {:>10} {:>8}  {:?}",
            format!("{:?}", r.kind),
            r.base,
            r.byte_len(),
            r.blocks,
            r.protection
        );
    }

    // Recomputed, not read back.
    let codebook_bytes = (1usize << m.b) * m.d as usize;
    let payload_bytes = m.m as usize;
    let rerank_bytes = m.d as usize;
    println!("\nbudget (recomputed from the profile)");
    println!("  codebook            {codebook_bytes} B  (2^b * D, independent of N)");
    println!("  payload per vector  {payload_bytes} B");
    println!(
        "  rerank per vector   {rerank_bytes} B  ({}x payload)",
        rerank_bytes / payload_bytes.max(1)
    );
    println!(
        "  stored per vector   {} B",
        payload_bytes + rerank_bytes + 8 // CRC share at 512 B blocks, both regions
    );

    // A truncated image keeps a verifying manifest — the digest covers the
    // region table, not the regions themselves — so the extent check is what
    // catches truncation. Refusing here is the point: reporting a full
    // inspection of an image whose regions run past its end would describe a
    // volume that does not exist.
    let declared: u64 = m.table.regions.iter().map(|r| r.byte_len()).sum();
    let end = m.table.regions.iter().map(|r| r.end()).max().unwrap_or(0);
    if end > bytes.len() as u64 {
        return Err(format!(
            "image is {} B but its regions extend to {end} B — truncated or corrupt",
            bytes.len()
        ));
    }
    println!(
        "\nconsistency\n  regions sum to {declared} B, image is {} B",
        bytes.len()
    );
    let aligned = m
        .table
        .regions
        .iter()
        .all(|r| (r.base as usize).is_multiple_of(SECTOR_BYTES));
    println!("  every region sector-aligned: {aligned}");

    let cb = m.table.get(RegionKind::Codebook);
    let rep = m.table.get(RegionKind::CodebookReplica);
    if let (Some(cb), Some(rep)) = (cb, rep) {
        let disjoint = (cb.base as usize) / SECTOR_BYTES != (rep.base as usize) / SECTOR_BYTES;
        println!("  codebook replica in a different erase sector: {disjoint}");
    }
    Ok(())
}
