//! `sector build` — train, encode and emit a volume image.
//!
//! The whole offline pipeline in one command: load, train, optimise labels,
//! quantize, encode, emit. Parameters that change the image are echoed to
//! stdout, so a build is reproducible from its own output.

use sector_build::dataset::VecsReader;
use sector_build::emit::{emit, Image};
use sector_build::encode::{encode, quantize};
use sector_build::label_opt::{optimise, permute_centroids, permute_codes};
use sector_build::train::{train, TrainConfig};
use std::path::PathBuf;

/// Parsed `build` arguments.
pub struct Args {
    /// Corpus in `.fvecs` / `.bvecs`.
    pub input: PathBuf,
    /// Destination image path.
    pub out: PathBuf,
    /// Subspaces.
    pub m: usize,
    /// Bits per code.
    pub b: usize,
    /// Candidate depth recorded in the manifest.
    pub r: usize,
    /// Vectors to read. 0 means the whole file.
    pub limit: usize,
    /// Training seed.
    pub seed: u64,
    /// Codebook copies, including the primary.
    pub copies: usize,
    /// Vector slots to leave erased for later appends.
    pub reserve: usize,
}

/// Parse `build` flags.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    Ok(Args {
        input: PathBuf::from(crate::flag(argv, "--input")?),
        out: PathBuf::from(crate::flag(argv, "--out")?),
        m: crate::opt_num(argv, "--m", 16)?,
        b: crate::opt_num(argv, "--b", 8)?,
        r: crate::opt_num(argv, "--r", 100)?,
        limit: crate::opt_num(argv, "--limit", 0)?,
        seed: crate::opt_num(argv, "--seed", 42)? as u64,
        copies: crate::opt_num(argv, "--copies", 2)?,
        reserve: crate::opt_num(argv, "--reserve", 0)?,
    })
}

/// Run `build`.
pub fn run(args: Args) -> Result<(), String> {
    let mut reader = VecsReader::open(&args.input).map_err(|e| format!("{e:?}"))?;
    let layout = reader.layout();
    let d = layout.dim as usize;
    let n = if args.limit == 0 {
        layout.count
    } else {
        args.limit.min(layout.count)
    };
    if !d.is_multiple_of(args.m) {
        return Err(format!("d={d} is not divisible by m={}", args.m));
    }

    let mut corpus = vec![0f32; n * d];
    let mut row = vec![0f32; d];
    for v in 0..n {
        match reader.next_f32(&mut row).map_err(|e| format!("{e:?}"))? {
            Some(_) => {
                corpus[v * d..(v + 1) * d].copy_from_slice(&row);
            }
            None => break,
        }
    }

    println!("input      {} ({n} vectors, D={d})", args.input.display());
    let cfg = TrainConfig {
        d,
        m: args.m,
        b: args.b,
        iterations: 25,
        seed: args.seed,
    };
    let (books, report) = train(&corpus, n, cfg).map_err(|e| format!("{e:?}"))?;
    println!(
        "train      m={} b={} seed={} mse={:.4} imbalance={:.2}x",
        args.m,
        args.b,
        args.seed,
        report.mean_mse(),
        report.imbalance()
    );

    let (mut codes, pops) = encode(&corpus, n, d, &books);

    // Label optimisation before quantization: the permutation is lossless, and
    // applying it to codebook and codes together is what keeps it so.
    let mut relabelled = Vec::with_capacity(books.len());
    for (j, book) in books.iter().enumerate() {
        let perm = optimise(book, args.b as u32, 8);
        permute_codes(&mut codes, args.m, j, &perm.map);
        relabelled.push(permute_centroids(book, &perm.map));
        if j == 0 {
            println!(
                "relabel    displacement {:.1} -> {:.1} ({:.2}x, {} swaps)",
                perm.before,
                perm.after,
                perm.reduction(),
                perm.swaps
            );
        }
    }

    let quantized = quantize(&relabelled, 127);
    println!(
        "quantize   codebook {} B (independent of N), skew {:.2}x",
        quantized.byte_len(),
        pops.skew_x1024() as f32 / 1024.0
    );

    // Rerank records: the corpus narrowed to i8 bytes.
    let rerank: Vec<u8> = corpus
        .iter()
        .map(|v| (v / 2.0).clamp(-128.0, 127.0) as i8 as u8)
        .collect();

    let mut image = Vec::new();
    let emitted = emit(
        &Image {
            codebook: &quantized,
            codes: &codes,
            rerank: &rerank,
            n,
            d,
            r: args.r as u32,
            copies: args.copies,
            reserve: args.reserve,
        },
        &mut image,
    )
    .map_err(|e| format!("{e:?}"))?;

    std::fs::write(&args.out, &image).map_err(|e| format!("{e}"))?;
    println!(
        "emit       {} ({} B, {} payload blocks, {} rerank blocks, {} codebook copies)",
        args.out.display(),
        emitted.bytes,
        emitted.payload_blocks,
        emitted.rerank_blocks,
        emitted.codebook_copies
    );
    Ok(())
}
