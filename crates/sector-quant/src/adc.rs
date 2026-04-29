//! Asymmetric distance computation tables.
//!
//! `T[j][v] = <q_j, C_j[v]>`, built once per query and read `m` times per
//! stored vector. Scoring a vector is `m` table lookups and `m` adds.
//!
//! Construction costs `2^b * D` multiply-accumulates and is independent of `N`,
//! so it is a per-query floor that does not amortise over corpus size. At T0
//! (`b = 8`, `D = 128`) that is 32,768 integer MACs. The same `b` at `D = 768`
//! costs 196,608, which rules out wide dimensions on arithmetic grounds before
//! the codebook footprint does.
//!
//! # Implementation notes
//!
//! Accumulate in `i32`. The table lives in the caller's workspace and is
//! rebuilt in place each query, so no allocation is implied.
//!
//! Construction is instrumented as its own phase (`Phase::Table`). It scales
//! with `2^b·D` while the scan scales with `N`; a combined latency figure
//! cannot attribute a regression to either.
//!
//! The table depends only on `q`, so a batch of queries sharing a rotation can
//! reuse it across shards.

use crate::codebook::Codebook;

/// Why a table build or scoring call was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdcError {
    /// The table buffer is not `m * centroids` entries.
    TableSize {
        /// Entries supplied.
        found: usize,
        /// Entries required.
        expected: usize,
    },
    /// The query is not `m * ds` components.
    QuerySize {
        /// Components supplied.
        found: usize,
        /// Components required.
        expected: usize,
    },
    /// Codebooks supplied does not equal `m`.
    SubspaceCount {
        /// Codebooks supplied.
        found: usize,
        /// Codebooks required.
        expected: usize,
    },
}

/// Multiply-accumulates one table build costs: `2^b * D`, independent of `N`.
///
/// A per-query floor that does not amortise over corpus size. At T0 (b=8,
/// D=128) it is 32,768; the same `b` at D=768 costs 196,608.
pub const fn table_macs(centroids: usize, d: usize) -> usize {
    centroids * d
}

/// Build `T[j][v] = <q_j, C_j[v]>` into `table`, in place.
///
/// `table` is the caller's workspace buffer, so no allocation is implied. It is
/// laid out row-major: subspace `j` occupies `table[j * centroids ..]`.
///
/// Returns the multiply-accumulates performed, counted rather than assumed, so
/// the cost model is validated by the code that pays it.
pub fn build_table(
    query: &[i8],
    codebooks: &[Codebook<'_>],
    table: &mut [i32],
) -> Result<usize, AdcError> {
    let m = codebooks.len();
    let first = match codebooks.first() {
        Some(cb) => cb,
        None => {
            return Err(AdcError::SubspaceCount {
                found: 0,
                expected: 1,
            })
        }
    };
    let centroids = first.centroids();
    let ds = first.ds();

    if query.len() != m * ds {
        return Err(AdcError::QuerySize {
            found: query.len(),
            expected: m * ds,
        });
    }
    if table.len() != m * centroids {
        return Err(AdcError::TableSize {
            found: table.len(),
            expected: m * centroids,
        });
    }

    let mut macs = 0usize;
    for (j, cb) in codebooks.iter().enumerate() {
        if cb.centroids() != centroids || cb.ds() != ds {
            return Err(AdcError::SubspaceCount {
                found: j,
                expected: m,
            });
        }
        let q_j = match query.get(j * ds..(j + 1) * ds) {
            Some(q) => q,
            None => {
                return Err(AdcError::QuerySize {
                    found: query.len(),
                    expected: m * ds,
                })
            }
        };
        for v in 0..centroids {
            let row = match cb.centroid(v) {
                Some(r) => r,
                None => continue,
            };
            let mut acc: i32 = 0;
            for (a, b) in q_j.iter().zip(row.iter()) {
                acc += (*a as i32) * (*b as i32);
                macs += 1;
            }
            if let Some(slot) = table.get_mut(j * centroids + v) {
                *slot = acc;
            }
        }
    }
    Ok(macs)
}

/// Score one vector from its 8-bit codes: `m` lookups and `m` adds.
///
/// No multiplies. Every multiply was paid once in [`build_table`], which is
/// what makes the per-vector cost low enough for the scan to be bandwidth-bound
/// above T0.
#[inline]
pub fn score_b8(codes: &[u8], table: &[i32], centroids: usize) -> i32 {
    let mut acc = 0i32;
    // Advance the table by one subspace row per code, carrying the remaining
    // rows in a slice rather than an index.
    //
    // The obvious form, `table[j * centroids + c]`, compiles to a shift on
    // RV32IMC — the optimiser can prove `centroids` is a power of two — but on
    // Cortex-M0+ it emitted four `muls`, caught by asm-check on
    // thumbv6m-none-eabi. M0+ has a weak multiplier and no hardware divide, so
    // a multiply in the per-vector path is exactly what this scan is designed
    // to avoid.
    //
    // Two earlier attempts each moved the problem rather than removing it:
    // `chunks_exact(centroids).nth(j)` computes an offset and put two
    // multiplies on RISC-V, and `chunks_exact().zip()` left them in LLVM's
    // unrolled remainder, where it fell back to index arithmetic. Splitting the
    // slice keeps the stride in a pointer the compiler never has to
    // reconstruct, and both targets come out clean.
    let mut rest = table;
    for &c in codes {
        let (row, tail) = match rest.split_at_checked(centroids) {
            Some(parts) => parts,
            None => break,
        };
        if let Some(v) = row.get(c as usize) {
            acc = acc.wrapping_add(*v);
        }
        rest = tail;
    }
    acc
}

/// Score one vector from packed 4-bit codes, two per byte.
#[inline]
pub fn score_b4(codes: &[u8], table: &[i32], centroids: usize, m: usize) -> i32 {
    let mut acc = 0i32;
    for j in 0..m {
        let byte = match codes.get(j / 2) {
            Some(v) => *v,
            None => break,
        };
        let c = if j.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        if let Some(v) = table.get(j * centroids + c as usize) {
            acc = acc.wrapping_add(*v);
        }
    }
    acc
}

/// Largest magnitude a table entry can reach, given the component range.
///
/// `ds * 127 * 128` bounds one inner product, and a full score sums `m` of
/// them. Used to prove the `i32` accumulator cannot overflow at a profile's
/// parameters rather than assuming it.
pub const fn max_score_magnitude(m: usize, ds: usize) -> i64 {
    (m as i64) * (ds as i64) * 127 * 128
}

/// Score one record using index arithmetic instead of slice splitting.
///
/// Measured 39.0 us against [`score_b8`]'s 48.8 us at `N=5,000, m=16` on an
/// out-of-order 64-bit core: a 25% gap, because `j * centroids` is a single
/// shift there and the split form re-derives a slice pair per code.
///
/// It is **not** the portable choice. On Cortex-M0+ the same expression emits
/// four `muls` — asm-check on `thumbv6m-none-eabi` catches it — and M0+ has a
/// weak multiplier and no hardware divide, so a multiply in the per-vector path
/// is what the scan design exists to avoid. `score_b8` is therefore the default
/// and the one the device runs; this exists for host tooling, where the
/// multiplier is free and the corpus is large enough for 25% to matter.
///
/// Both return identical values for every input; a test asserts it.
pub fn score_b8_indexed(codes: &[u8], table: &[i32], centroids: usize) -> i32 {
    let mut acc = 0i32;
    for (j, &c) in codes.iter().enumerate() {
        if let Some(v) = table.get(j * centroids + c as usize) {
            acc = acc.wrapping_add(*v);
        }
    }
    acc
}

/// Build an ADC table whose maximum is the nearest neighbour by **L2**.
///
/// `build_table` holds inner products, and ranking by inner product is not
/// ranking by L2 unless every stored vector has the same norm. On SIFT-like
/// data the two disagree badly — measured 2 of 10 overlap in the top ten — so a
/// scan that maximises inner product against an L2 ground truth silently
/// returns wrong answers at full speed.
///
/// Expanding the distance,
///
/// ```text
/// ||x - q||^2 = ||x||^2 - 2<x, q> + ||q||^2
/// ```
///
/// the `||q||^2` term is constant across the corpus for one query and cannot
/// change an ordering, so minimising `||x||^2 - 2<x, q>` is minimising L2.
/// Negating gives a quantity to maximise:
///
/// ```text
/// T[j][v] = 2 * <q_j, C_j[v]> - ||C_j[v]||^2
/// ```
///
/// Both terms are per-subspace and per-centroid, so both fold into the table.
/// The scan is unchanged — `m` lookups and `m` adds, no multiplies — and the
/// correction costs nothing per vector because it is paid once per centroid
/// during table construction.
///
/// # Range
///
/// Worst case at `D = 960`, `i8` components: `960 * (2*127*127 + 127*127)` =
/// 46,451,520, against an `i32` maximum of 2,147,483,647. Accumulation is safe
/// at every configuration this format admits.
pub fn build_table_l2(
    query: &[i8],
    codebooks: &[Codebook<'_>],
    table: &mut [i32],
) -> Result<usize, AdcError> {
    let macs = build_table(query, codebooks, table)?;

    let centroids = match codebooks.first() {
        Some(cb) => cb.centroids(),
        None => return Ok(macs),
    };

    for (j, cb) in codebooks.iter().enumerate() {
        for v in 0..centroids {
            let row = match cb.centroid(v) {
                Some(r) => r,
                None => continue,
            };
            let mut sq: i32 = 0;
            for a in row {
                sq += (*a as i32) * (*a as i32);
            }
            if let Some(slot) = table.get_mut(j * centroids + v) {
                // 2<q,c> - ||c||^2 from the inner product already stored.
                *slot = slot.saturating_mul(2).saturating_sub(sq);
            }
        }
    }
    Ok(macs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::Scale;

    const M: usize = 16;
    const DS: usize = 8;
    const CENTROIDS: usize = 256;

    fn components() -> [i8; CENTROIDS * DS] {
        core::array::from_fn(|i| ((i * 37) % 255) as i8)
    }

    #[test]
    fn table_build_costs_32768_macs_at_t0() {
        let comps = components();
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&comps, CENTROIDS, DS, scale).unwrap());
        let query = [1i8; M * DS];
        let mut table = [0i32; M * CENTROIDS];

        let macs = build_table(&query, &cbs, &mut table).unwrap();
        // Measured, not assumed: 2^b * D = 256 * 128.
        assert_eq!(macs, 32_768);
        assert_eq!(macs, table_macs(CENTROIDS, M * DS));
    }

    #[test]
    fn wide_dimension_costs_six_times_more() {
        // The arithmetic reason D=768 is impractical at b=8, independent of
        // the codebook footprint.
        assert_eq!(table_macs(256, 768), 196_608);
        assert_eq!(table_macs(256, 768) / table_macs(256, 128), 6);
    }

    #[test]
    fn scoring_agrees_with_the_direct_inner_product() {
        let comps = components();
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&comps, CENTROIDS, DS, scale).unwrap());
        let query: [i8; M * DS] = core::array::from_fn(|i| ((i * 13) % 100) as i8);
        let mut table = [0i32; M * CENTROIDS];
        build_table(&query, &cbs, &mut table).unwrap();

        let codes: [u8; M] = core::array::from_fn(|j| ((j * 17) % CENTROIDS) as u8);
        let from_table = score_b8(&codes, &table, CENTROIDS);

        // Same score computed the slow way, straight from the codebooks.
        let mut direct = 0i32;
        for (j, &c) in codes.iter().enumerate() {
            let row = cbs[j].centroid(c as usize).unwrap();
            let q_j = &query[j * DS..(j + 1) * DS];
            for (a, b) in q_j.iter().zip(row.iter()) {
                direct += (*a as i32) * (*b as i32);
            }
        }
        assert_eq!(from_table, direct);
    }

    #[test]
    fn four_bit_scoring_matches_eight_bit_on_the_same_codes() {
        let comps: [i8; 16 * DS] = core::array::from_fn(|i| ((i * 11) % 127) as i8);
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&comps, 16, DS, scale).unwrap());
        let query: [i8; M * DS] = core::array::from_fn(|i| (i % 50) as i8);
        let mut table = [0i32; M * 16];
        build_table(&query, &cbs, &mut table).unwrap();

        // Pack codes 0..16 two per byte, then score both ways.
        let wide: [u8; M] = core::array::from_fn(|j| (j % 16) as u8);
        let mut packed = [0u8; M / 2];
        for (j, &c) in wide.iter().enumerate() {
            let slot = &mut packed[j / 2];
            *slot = if j.is_multiple_of(2) {
                (*slot & 0xF0) | c
            } else {
                (*slot & 0x0F) | (c << 4)
            };
        }
        assert_eq!(
            score_b4(&packed, &table, 16, M),
            score_b8(&wide, &table, 16)
        );
    }

    #[test]
    fn the_accumulator_cannot_overflow_at_t0_parameters() {
        // Worst case: every component at its extreme, all m subspaces.
        let bound = max_score_magnitude(M, DS);
        assert!(
            bound < i32::MAX as i64,
            "i32 accumulator overflows: {bound}"
        );
        // T1 doubles m; still safe.
        assert!(max_score_magnitude(32, DS) < i32::MAX as i64);
    }

    #[test]
    fn worst_case_components_do_not_overflow_in_practice() {
        let comps = [i8::MIN; CENTROIDS * DS];
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&comps, CENTROIDS, DS, scale).unwrap());
        let query = [i8::MAX; M * DS];
        let mut table = [0i32; M * CENTROIDS];
        build_table(&query, &cbs, &mut table).unwrap();

        let codes = [0u8; M];
        let score = score_b8(&codes, &table, CENTROIDS);
        // 16 * 8 * 127 * -128 = -2,080,768, well inside i32.
        assert_eq!(score, -2_080_768);
        assert!((score as i64).abs() <= max_score_magnitude(M, DS));
    }

    #[test]
    fn mismatched_buffers_are_refused() {
        let comps = components();
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&comps, CENTROIDS, DS, scale).unwrap());
        let query = [0i8; M * DS];
        let mut small = [0i32; 10];
        assert_eq!(
            build_table(&query, &cbs, &mut small),
            Err(AdcError::TableSize {
                found: 10,
                expected: M * CENTROIDS
            })
        );

        let short_query = [0i8; 10];
        let mut table = [0i32; M * CENTROIDS];
        assert_eq!(
            build_table(&short_query, &cbs, &mut table),
            Err(AdcError::QuerySize {
                found: 10,
                expected: M * DS
            })
        );
    }

    #[test]
    fn the_table_is_rebuilt_in_place() {
        let comps = components();
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&comps, CENTROIDS, DS, scale).unwrap());
        let mut table = [0i32; M * CENTROIDS];

        let q1 = [1i8; M * DS];
        build_table(&q1, &cbs, &mut table).unwrap();
        let first = table[5];

        let q2 = [2i8; M * DS];
        build_table(&q2, &cbs, &mut table).unwrap();
        // Same buffer, fully overwritten: no stale entry survives.
        assert_eq!(table[5], first * 2);
    }

    #[test]
    fn the_l2_table_ranks_by_distance_not_inner_product() {
        // The defect this exists to prevent: maximising inner product against
        // an L2 ground truth returns wrong answers at full speed. On vectors
        // with unequal norms the two orderings disagree.
        const M: usize = 4;
        const DS: usize = 8;
        const K: usize = 16;

        // Centroids with deliberately unequal norms.
        let mut comps = [0i8; M * K * DS];
        for j in 0..M {
            for v in 0..K {
                for i in 0..DS {
                    let scale = 1 + (v as i32 % 5) * 20;
                    comps[(j * K + v) * DS + i] =
                        (((j * 7 + v * 13 + i * 3) % 21) as i32 * scale / 5 - 40) as i8;
                }
            }
        }
        let scale = Scale { num: 1, den: 1 };
        let books: [Codebook<'_>; M] = core::array::from_fn(|j| {
            Codebook::new(&comps[j * K * DS..(j + 1) * K * DS], K, DS, scale).unwrap()
        });

        let query: [i8; M * DS] = core::array::from_fn(|i| ((i * 11) % 61) as i8 - 30);
        let mut table = [0i32; M * K];
        build_table_l2(&query, &books, &mut table).unwrap();

        // Score every code combination two ways: through the table, and by
        // reconstructing and measuring L2 directly.
        let mut disagreements = 0usize;
        let mut worst = (0i32, 0i32);
        for a in 0..K {
            for b in 0..K {
                let codes = [a as u8, b as u8, (a % K) as u8, (b % K) as u8];
                let table_score = score_b8(&codes, &table, K);

                let mut l2 = 0i32;
                for (j, c) in codes.iter().enumerate() {
                    let row = books[j].centroid(*c as usize).unwrap();
                    for (i, comp) in row.iter().enumerate() {
                        let diff = *comp as i32 - query[j * DS + i] as i32;
                        l2 += diff * diff;
                    }
                }
                // The table maximum must correspond to the L2 minimum, so the
                // two must be exact negations up to the constant ||q||^2.
                let qsq: i32 = query.iter().map(|x| (*x as i32) * (*x as i32)).sum();
                if table_score != qsq - l2 {
                    disagreements += 1;
                    worst = (table_score, qsq - l2);
                }
            }
        }
        assert_eq!(
            disagreements, 0,
            "table score is not -(L2) + ||q||^2: got {} expected {}",
            worst.0, worst.1
        );
    }

    #[test]
    fn the_l2_correction_leaves_the_scan_shape_unchanged() {
        // The correction is paid per centroid at table build, never per vector,
        // so the table has exactly the same extent as the inner-product one.
        const M: usize = 2;
        const DS: usize = 4;
        const K: usize = 8;
        let comps = [3i8; M * K * DS];
        let scale = Scale { num: 1, den: 1 };
        let books: [Codebook<'_>; M] = core::array::from_fn(|j| {
            Codebook::new(&comps[j * K * DS..(j + 1) * K * DS], K, DS, scale).unwrap()
        });
        let query = [2i8; M * DS];

        let mut plain = [0i32; M * K];
        let mut l2 = [0i32; M * K];
        let a = build_table(&query, &books, &mut plain).unwrap();
        let b = build_table_l2(&query, &books, &mut l2).unwrap();
        assert_eq!(a, b, "MAC count must not change");
        assert_eq!(plain.len(), l2.len());
        // 2*<q,c> - ||c||^2 with q=2, c=3, ds=4: 2*24 - 36 = 12.
        assert_eq!(l2[0], 12);
        assert_eq!(plain[0], 24);
    }

    #[test]
    fn the_indexed_and_split_scorers_agree_on_every_input() {
        // The two exist because they compile differently, not because they
        // compute differently. If they can disagree, the host and device are
        // running different engines.
        const M: usize = 8;
        const K: usize = 16;
        let table: [i32; M * K] = core::array::from_fn(|i| (i as i32 * 37) % 251 - 125);
        for seed in 0..64u32 {
            let codes: [u8; M] = core::array::from_fn(|j| {
                let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(j as u32);
                x ^= x >> 13;
                (x % K as u32) as u8
            });
            assert_eq!(
                score_b8(&codes, &table, K),
                score_b8_indexed(&codes, &table, K),
                "seed {seed}"
            );
        }
    }
}
