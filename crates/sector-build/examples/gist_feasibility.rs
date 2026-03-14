//! Enumerate the (m, b) grid at GIST's D=960 against the T0 and T1 budgets.
//!
//! Uses `Profile`'s own `const fn` arithmetic rather than restating it, so a
//! change to the budget rules cannot leave this analysis quietly wrong.
//!
//! Emits JSON on stdout.

use sector_format::profile::{Profile, T0, T1};

fn main() {
    const D: usize = 960;
    // Divisors of 960 that give a subspace dimension worth quantizing. m must
    // divide D, and ds below 4 wastes a codebook on almost no information.
    let ms = [
        8usize, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 120,
    ];
    let bs = [4usize, 6, 8];

    println!("{{");
    println!("  \"dimension\": {D},");
    println!("  \"note\": \"codebook is 2^b * D bytes and is independent of m; m sets payload bytes and ADC table size\",");

    for (tier_name, tier) in [("T0", T0), ("T1", T1)] {
        println!("  \"{tier_name}\": {{");
        println!("    \"ram_budget\": {},", tier.ram_budget);
        println!("    \"grid\": [");
        let mut first = true;
        for &m in &ms {
            if !D.is_multiple_of(m) {
                continue;
            }
            for &b in &bs {
                let p = Profile { d: D, m, b, ..tier };
                let fixed = p.fixed_bytes();
                let fits = fixed < p.ram_budget;
                let resident = if fits { p.resident_vectors() } else { 0 };
                if !first {
                    println!(",");
                }
                first = false;
                print!(
                    "      {{\"m\": {m}, \"b\": {b}, \"ds\": {}, \"codebook\": {}, \
                     \"adc_table\": {}, \"payload\": {}, \"fixed\": {fixed}, \
                     \"fits\": {fits}, \"resident_vectors\": {resident}}}",
                    p.ds(),
                    p.codebook_bytes(),
                    p.adc_table_bytes(),
                    p.payload_bytes(),
                );
            }
        }
        println!("\n    ]");
        println!("  }},");
    }

    // The claim under test: dimension, not subspace count, is what makes 8-bit
    // codes unaffordable. The codebook is 2^b * D with no m in it.
    println!("  \"codebook_by_dimension\": [");
    let dims = [128usize, 256, 512, 768, 960];
    for (i, &d) in dims.iter().enumerate() {
        let cb8 = 256 * d;
        let cb4 = 16 * d;
        print!(
            "    {{\"d\": {d}, \"codebook_b8\": {cb8}, \"codebook_b4\": {cb4}, \
             \"b8_fits_t0\": {}}}",
            cb8 + 8 * 1024 < T0.ram_budget
        );
        if i + 1 < dims.len() {
            println!(",");
        } else {
            println!();
        }
    }
    println!("  ]");
    println!("}}");
}
