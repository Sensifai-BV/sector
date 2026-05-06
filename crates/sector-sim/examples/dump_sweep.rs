//! Dump the signed displacement sweep: measured loss against the two-sided
//! depth-aware bound, at both signs.
//!
//! Feeds `measurements/sweep.png`. Kept as an example so the figure is
//! regenerable without running the suite, and so the plotted numbers are the
//! ones the library computes.

use sector_build::encode::encode;
use sector_build::train::{train, TrainConfig};
use sector_sim::corrupt::{
    affected_of, apply, bound, depths_of, looseness, worst_case_for, Construction, Sign,
};
use sector_sim::recall::{exact_top_ids, top_ids, Encoded};

const D: usize = 16;
const M: usize = 2;
const N: usize = 1500;
const B: usize = 4;
const R: usize = 100;
const K: usize = 10;

fn main() {
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
    let (books, _) = train(&corpus, N, cfg).unwrap();
    let (codes, _) = encode(&corpus, N, D, &books);
    // Flatten the trained codebooks so stage one scores reconstructions rather
    // than exact inner products; otherwise clean recall is identically 1.
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
    // Ground truth is the exact ranking; the candidate list is approximate.
    let truths: Vec<Vec<u32>> = queries.iter().map(|q| exact_top_ids(data, q, K)).collect();

    let magnitudes = [100.0f32, 200.0, 400.0, 600.0, 800.0, 1000.0, 1400.0, 2000.0];

    println!("{{");
    println!("  \"d\": {D}, \"m\": {M}, \"n\": {N}, \"b\": {B}, \"r\": {R}, \"k\": {K},");
    println!("  \"queries\": {},", queries.len());
    print!("  \"points\": [");
    let mut first = true;
    let mut clean_total = 0f32;

    for sign in [Sign::Inflate, Sign::Deflate] {
        for &magnitude in magnitudes.iter() {
            let mut loss = 0f32;
            let mut bnd = 0f32;
            let mut intruders = 0u32;
            for (qi, (q, t)) in queries.iter().zip(truths.iter()).enumerate() {
                // Aim must match sign: inflation damages via intruders, so it
                // targets a centroid the query's neighbours avoid; deflation
                // targets the one they use.
                let aim = worst_case_for(sign, qi);
                let c = sector_sim::corrupt::resolve(aim, data, 0, 1 << B, &truths);
                let dmg = apply(
                    data,
                    q,
                    t,
                    Construction {
                        aim,
                        sign,
                        magnitude,
                        subspace: 0,
                    },
                    c,
                    R,
                    K,
                );
                let clean = top_ids(data, q, R);
                let depths = depths_of(&clean, t, K, R);
                let affected = match sign {
                    Sign::Deflate => affected_of(data, t, K, 0, c),
                    Sign::Inflate => vec![false; K],
                };
                loss += dmg.loss();
                bnd += bound(&depths, &affected, dmg.intruders, R, K);
                intruders += dmg.intruders;
                clean_total += dmg.clean_recall;
            }
            let nq = queries.len() as f32;
            let (loss, bnd) = (loss / nq, bnd / nq);
            let l = looseness(bnd, loss).unwrap_or(f32::NAN);
            print!(
                "{}{{\"sign\":\"{}\",\"magnitude\":{magnitude},\"loss\":{loss:.6},\"bound\":{bnd:.6},\"intruders\":{intruders},\"looseness\":{}}}",
                if first { "" } else { "," },
                if matches!(sign, Sign::Inflate) { "inflate" } else { "deflate" },
                if l.is_nan() { "null".to_string() } else { format!("{l:.4}") }
            );
            first = false;
        }
    }
    println!("],");
    let n_points = (magnitudes.len() * 2 * queries.len()) as f32;
    println!("  \"clean_recall\": {:.6}", clean_total / n_points);
    println!("}}");
}
