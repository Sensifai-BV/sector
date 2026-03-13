//! Dump measured criticality weights and the achievable-rate envelope as JSON.
//!
//! Feeds `measurements/criticality.png`. Kept as an example rather than a test
//! so the figure is regenerable without running the suite, and so the numbers
//! plotted are the ones the library computes.

use sector_build::allocate::{allocate, envelope, group_weights, Rate};
use sector_build::criticality::{measure, Encoded, Probe, Sweep};
use sector_build::encode::encode;
use sector_build::train::{train, TrainConfig};

const D: usize = 16;
const M: usize = 2;
const N: usize = 2000;
const B: usize = 5;

fn main() {
    let mut data = vec![0f32; N * D];
    for v in 0..N {
        // Three unequal clusters, so populations and exposures are skewed.
        let c = if v % 10 < 6 {
            0
        } else if v % 10 < 9 {
            1
        } else {
            2
        };
        for j in 0..D {
            data[v * D + j] = (c as f32) * 30.0 + (((v * 13 + j * 7) % 41) as f32) - 20.0;
        }
    }

    let cfg = TrainConfig {
        d: D,
        m: M,
        b: B,
        iterations: 40,
        seed: 9,
    };
    let (books, _) = train(&data, N, cfg).unwrap();
    let (codes, pops) = encode(&data, N, D, &books);

    let queries: Vec<Vec<f32>> = (0..12)
        .map(|i| {
            (0..D)
                .map(|j| ((i * 5 + j * 3) % 31) as f32 - 15.0)
                .collect()
        })
        .collect();
    // Ground truth by exhaustive inner product, the same metric the sweep uses.
    let truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            let mut s: Vec<(f32, u32)> = (0..N)
                .map(|v| {
                    let mut acc = 0f32;
                    for (a, b) in data[v * D..(v + 1) * D].iter().zip(q.iter()) {
                        acc += a * b;
                    }
                    (acc, v as u32)
                })
                .collect();
            s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            s.into_iter().take(10).map(|(_, id)| id).collect()
        })
        .collect();
    let ps: Vec<Probe<'_>> = queries
        .iter()
        .zip(truths.iter())
        .map(|(q, t)| Probe {
            vector: q,
            truth: t,
        })
        .collect();

    let w = measure(
        Encoded {
            corpus: &data,
            n: N,
            d: D,
            codes: &codes,
            m: M,
        },
        0,
        &books[0],
        &ps,
        Sweep {
            r: 100,
            k: 10,
            delta: 600.0,
        },
    );

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
        Rate {
            parity_per_256: 256,
            residual_ppb: 100,
        },
    ];
    let group_bytes = vec![4096usize; 4];
    let gw = group_weights(&w, 4);
    let hull = envelope(&rates, 4096);
    // Several budgets: a single budget can land on a uniform assignment by
    // arithmetic coincidence, which would misrepresent what the weighting does.
    let budgets = [1024usize, 2048, 3000, 4096, 6144];
    let allocs: Vec<_> = budgets
        .iter()
        .map(|b| allocate(&gw, &group_bytes, &rates, *b))
        .collect();
    let alloc = allocate(&gw, &group_bytes, &rates, 4096);

    println!("{{");
    println!(
        "  \"d\": {D}, \"m\": {M}, \"n\": {N}, \"b\": {B}, \"r\": 100, \"k\": 10, \"delta\": 600.0,"
    );
    print!("  \"population\": [");
    for c in 0..(1usize << B) {
        print!("{}{}", if c > 0 { "," } else { "" }, pops.get(0, c));
    }
    println!("],");
    print!("  \"inflate\": [");
    for (i, e) in w.per_centroid.iter().enumerate() {
        print!("{}{}", if i > 0 { "," } else { "" }, e.inflate_loss);
    }
    println!("],");
    print!("  \"deflate\": [");
    for (i, e) in w.per_centroid.iter().enumerate() {
        print!("{}{}", if i > 0 { "," } else { "" }, e.deflate_loss);
    }
    println!("],");
    print!("  \"envelope\": [");
    for (i, r) in hull.iter().enumerate() {
        print!(
            "{}[{},{}]",
            if i > 0 { "," } else { "" },
            r.cost(4096),
            r.residual_ppb
        );
    }
    println!("],");
    print!("  \"rates\": [");
    for (i, r) in rates.iter().enumerate() {
        print!(
            "{}[{},{}]",
            if i > 0 { "," } else { "" },
            r.cost(4096),
            r.residual_ppb
        );
    }
    println!("],");
    print!("  \"group_weights\": [");
    for (i, g) in gw.iter().enumerate() {
        print!("{}{}", if i > 0 { "," } else { "" }, g);
    }
    println!("],");
    print!("  \"chosen\": [");
    for (i, a) in alloc.assignments.iter().enumerate() {
        print!(
            "{}[{},{}]",
            if i > 0 { "," } else { "" },
            a.cost,
            a.rate.residual_ppb
        );
    }
    println!("],");
    print!("  \"by_budget\": [");
    for (i, (b, a)) in budgets.iter().zip(allocs.iter()).enumerate() {
        print!(
            "{}{{\"budget\":{b},\"spent\":{},\"points\":[",
            if i > 0 { "," } else { "" },
            a.spent
        );
        for (j, x) in a.assignments.iter().enumerate() {
            print!(
                "{}[{},{}]",
                if j > 0 { "," } else { "" },
                x.cost,
                x.rate.residual_ppb
            );
        }
        print!("]}}");
    }
    println!("],");
    println!("  \"budget\": {}, \"spent\": {}", alloc.budget, alloc.spent);
    println!("}}");
}
