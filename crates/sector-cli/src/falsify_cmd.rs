//! `sector falsify` — run the falsification suite and report each verdict.
//!
//! Exits non-zero when a claim is refuted, so a refutation fails CI rather than
//! being recorded in a log nobody reads. A refutation is a result to report,
//! not a test to fix.

use sector_build::allocate::Rate;
use sector_build::encode::encode;
use sector_build::train::{train, TrainConfig};
use sector_sim::recall::{exact_top_ids, Encoded};
use sector_sim::sweep::{run_all, Instance, Verdict};

const D: usize = 16;
const M: usize = 2;
const N: usize = 1500;
const B: usize = 4;

/// Run every claim. Returns the number refuted.
pub fn run() -> Result<usize, String> {
    let mut corpus = vec![0f32; N * D];
    for v in 0..N {
        let c = if v % 10 < 6 {
            0
        } else if v % 10 < 9 {
            1
        } else {
            2
        };
        for j in 0..D {
            corpus[v * D + j] = (c as f32) * 30.0 + (((v * 13 + j * 7) % 41) as f32) - 20.0;
        }
    }
    let cfg = TrainConfig {
        d: D,
        m: M,
        b: B,
        iterations: 40,
        seed: 9,
    };
    let (books, _) = train(&corpus, N, cfg).map_err(|e| format!("{e:?}"))?;
    let (codes, _) = encode(&corpus, N, D, &books);
    let mut centroids = vec![0f32; M * (1 << B) * (D / M)];
    for (j, book) in books.iter().enumerate() {
        let at = j * (1 << B) * (D / M);
        centroids[at..at + book.centroids.len()].copy_from_slice(&book.centroids);
    }

    let queries: Vec<Vec<f32>> = (0..8)
        .map(|i| {
            (0..D)
                .map(|j| ((i * 5 + j * 3) % 31) as f32 - 15.0)
                .collect()
        })
        .collect();
    let data = Encoded {
        corpus: &corpus,
        n: N,
        d: D,
        codes: &codes,
        m: M,
        centroids: &centroids,
        k: 1 << B,
    };
    let truths: Vec<Vec<u32>> = queries.iter().map(|q| exact_top_ids(data, q, 10)).collect();

    let inst = Instance {
        data,
        queries: &queries,
        truths: &truths,
        centroids: 1 << B,
        r: 100,
        k: 10,
    };
    let rates = vec![
        Rate {
            parity_per_256: 0,
            residual_ppb: 1_000_000,
        },
        Rate {
            parity_per_256: 32,
            residual_ppb: 100_000,
        },
        Rate {
            parity_per_256: 64,
            residual_ppb: 10_000,
        },
        Rate {
            parity_per_256: 128,
            residual_ppb: 1_000,
        },
    ];

    let findings = run_all(&inst, &[100, 50, 20, 5], &[4096; 4], &rates, 12, 12);
    println!("configuration D={D} m={M} b={B} N={N} R=100 k=10\n");
    println!(
        "  {:<20} {:<13} {:>9}  evidence",
        "claim", "verdict", "looseness"
    );
    let mut refuted = 0usize;
    for f in &findings {
        if f.is_refutation() {
            refuted += 1;
        }
        let loose = match f.looseness {
            Some(l) => format!("{l:.2}x"),
            None => "-".to_string(),
        };
        println!(
            "  {:<20} {:<13} {:>9}  {}",
            f.claim,
            match f.verdict {
                Verdict::Held => "held",
                Verdict::Refuted => "REFUTED",
                Verdict::Inconclusive => "inconclusive",
            },
            loose,
            f.evidence
        );
    }
    Ok(refuted)
}
