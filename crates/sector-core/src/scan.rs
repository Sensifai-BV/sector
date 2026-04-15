//! Stage one: sequential payload scan.
//!
//! `score_i = sum_j T[j][c_ij]` per stored vector — `m` table lookups and `m`
//! adds, no multiplies. The multiplies were paid once during table
//! construction, which leaves this stage bandwidth-bound at T1 and above.
//!
//! At T0 the codes are RAM-resident. Where they exceed RAM, an execute-in-place
//! backend borrows them from the mapped NOR window, so the loop performs no I/O
//! calls and no copies either way.
//!
//! # Inner-loop rules
//!
//! Threshold-test against the current heap minimum before attempting
//! insertion. Most vectors never qualify, and the test keeps them out of the
//! sift path.
//!
//! At `b = 8` a code is one byte: a load and an add per subspace. The `b = 4`
//! path unpacks two codes per byte with a shift and a mask rather than doubling
//! the bytes streamed.
//!
//! Payload CRCs are not verified here. Verification covers the `R` candidates
//! in stage two; verifying all `N` blocks costs more than the scan and protects
//! bytes whose corruption stage two catches.

use crate::heap::{Candidate, Heap};
use sector_quant::adc;

/// What a scan pass measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanStats {
    /// Vectors examined.
    pub scanned: u32,
    /// Vectors that passed the threshold test and were offered to the heap.
    pub offered: u32,
    /// Incumbents displaced.
    pub evicted: u32,
}

/// Scan `codes` into `heap`, choosing the loop shape by target width.
///
/// The two shapes are proven bit-identical by test. Which is faster is decided
/// on **instruction count**, measured off the shipped binaries:
///
/// | target | scalar | four-wide |
/// |---|---:|---:|
/// | RV32IMC (ESP32-C3) | 8.00 instr/code | 5.75 (-28%) |
/// | Xtensa (ESP32-S3) | 8.00 instr/code | 6.25 (-22%) |
///
/// On a core that retires near one instruction per cycle, fewer instructions is
/// faster, so four-wide is selected on 32-bit targets. On a 64-bit
/// out-of-order host the scalar shape measured faster (39.0 us against 40.1 us
/// at N=5,000), because such a core already overlaps the serial accumulator
/// chain and the extra chains only cost register pressure.
///
/// # What is not claimed
///
/// No measured device speed-up. Emulator runs appeared to show 1.26x on the C3
/// and 1.08x on the S3 for the same source; the two disagree, an
/// instruction-count model fitted to one mispredicts the other by 10%, and the
/// emulator's own timer was shown to report host wall-clock time rather than
/// emulated cycles. Those figures measured an interpreter. The instruction
/// counts above are static facts about the machine code and do not depend on
/// that timebase; the speed-up on real silicon is unmeasured.
#[inline]
pub fn scan_b8_auto(
    codes: &[u8],
    first_id: u32,
    payload_bytes: usize,
    table: &[i32],
    centroids: usize,
    heap: &mut Heap<'_>,
) -> ScanStats {
    #[cfg(target_pointer_width = "64")]
    {
        scan_b8(codes, first_id, payload_bytes, table, centroids, heap)
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        scan_b8_x4(codes, first_id, payload_bytes, table, centroids, heap)
    }
}

/// One record's ADC score, using whichever scorer suits the target.
///
/// The two forms are numerically identical — a test in `sector-quant` asserts
/// it over every input — and differ only in what they compile to. Index
/// arithmetic is one shift on a core with a strong multiplier and measured 25%
/// faster there; on Cortex-M0+ it emits multiplies, which is what the scan
/// design exists to avoid. Choosing per target gives the device the portable
/// form without taxing host tooling.
#[inline(always)]
fn score_one(record: &[u8], table: &[i32], centroids: usize) -> i32 {
    #[cfg(target_pointer_width = "64")]
    {
        adc::score_b8_indexed(record, table, centroids)
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        adc::score_b8(record, table, centroids)
    }
}

/// Scan `codes` into `heap`, one `payload_bytes`-strided record per vector.
///
/// `m` table lookups and `m` adds per vector, no multiplies. Payload CRCs are
/// not verified here: verifying `N` blocks per query costs more than the scan
/// itself, and stage two is where a corrupted byte would change an answer.
///
/// `codes` may be borrowed from a memory-mapped window, in which case the loop
/// performs no I/O calls and no copies.
pub fn scan_b8(
    codes: &[u8],
    first_id: u32,
    payload_bytes: usize,
    table: &[i32],
    centroids: usize,
    heap: &mut Heap<'_>,
) -> ScanStats {
    let mut stats = ScanStats::default();
    if payload_bytes == 0 {
        return stats;
    }
    for (i, record) in codes.chunks_exact(payload_bytes).enumerate() {
        let id = first_id + i as u32;
        let score = score_one(record, table, centroids);
        stats.scanned += 1;
        // Threshold-test before insertion: most vectors never qualify, and the
        // test keeps them out of the sift path entirely.
        if !heap.would_accept(score, id) {
            continue;
        }
        stats.offered += 1;
        if heap.push(Candidate { score, id }).is_some() {
            stats.evicted += 1;
        }
    }
    stats
}

/// Scan packed 4-bit codes, two per byte.
pub fn scan_b4(
    codes: &[u8],
    first_id: u32,
    payload_bytes: usize,
    table: &[i32],
    centroids: usize,
    m: usize,
    heap: &mut Heap<'_>,
) -> ScanStats {
    let mut stats = ScanStats::default();
    if payload_bytes == 0 {
        return stats;
    }
    for (i, record) in codes.chunks_exact(payload_bytes).enumerate() {
        let id = first_id + i as u32;
        let score = adc::score_b4(record, table, centroids, m);
        stats.scanned += 1;
        if !heap.would_accept(score, id) {
            continue;
        }
        stats.offered += 1;
        if heap.push(Candidate { score, id }).is_some() {
            stats.evicted += 1;
        }
    }
    stats
}

/// Scan four vectors per iteration, keeping four independent accumulator
/// chains.
///
/// `scan_b8` accumulates `m` table lookups into one register, so each add waits
/// on the previous one and the core issues roughly one add per cycle regardless
/// of how many it could retire. Four vectors give four independent chains and
/// fill those slots.
///
/// # Why not NEON
///
/// The inner operation is a gather: `table[j * centroids + code]` with a
/// data-dependent index. NEON's table-lookup instructions (`tbl`, `tbx`)
/// address at most four registers — 64 bytes — while a `b=8` table is
/// `m * 256 * 4` bytes, 16 KiB at `m=16`. There is no NEON gather for this
/// shape, so the width has to come from independent chains rather than from
/// vector lanes. That keeps the scan portable and inside `forbid(unsafe_code)`.
///
/// Results are bit-identical to [`scan_b8`]: same arithmetic, same order of
/// insertion, only the interleaving differs.
///
/// # Measured: no faster on a Cortex-A72 or an Apple core
///
/// 40.1 us against `scan_b8`'s 39.0 us at `N=5,000, m=16`. The scalar loop
/// already runs at roughly 1.6 cycles per lookup-and-add, which is a scalar
/// load-add's throughput, so there is no stall for a second chain to fill. The
/// benchmark uses [`scan_b8`]; this is kept because the equivalence is proven
/// and an in-order core with a longer load-use penalty may still benefit, and
/// because deleting it would lose the evidence that width was tried.
pub fn scan_b8_x4(
    codes: &[u8],
    first_id: u32,
    payload_bytes: usize,
    table: &[i32],
    centroids: usize,
    heap: &mut Heap<'_>,
) -> ScanStats {
    let mut stats = ScanStats::default();
    if payload_bytes == 0 {
        return stats;
    }

    let stride = payload_bytes * 4;
    let mut base = 0usize;
    let mut id = first_id;

    // Four at a time while a full group remains.
    while base + stride <= codes.len() {
        let (mut a, mut b, mut c, mut d) = (0i32, 0i32, 0i32, 0i32);
        for j in 0..payload_bytes {
            let row = j * centroids;
            // Four independent loads and adds; no chain waits on another.
            a = a.wrapping_add(lookup(table, row, codes[base + j]));
            b = b.wrapping_add(lookup(table, row, codes[base + payload_bytes + j]));
            c = c.wrapping_add(lookup(table, row, codes[base + 2 * payload_bytes + j]));
            d = d.wrapping_add(lookup(table, row, codes[base + 3 * payload_bytes + j]));
        }
        for (k, score) in [a, b, c, d].into_iter().enumerate() {
            let vid = id + k as u32;
            stats.scanned += 1;
            if !heap.would_accept(score, vid) {
                continue;
            }
            stats.offered += 1;
            if heap.push(Candidate { score, id: vid }).is_some() {
                stats.evicted += 1;
            }
        }
        base += stride;
        id += 4;
    }

    // Tail: fewer than four records left.
    let tail = &codes[base..];
    let rest = scan_b8(tail, id, payload_bytes, table, centroids, heap);
    stats.scanned += rest.scanned;
    stats.offered += rest.offered;
    stats.evicted += rest.evicted;
    stats
}

/// One table lookup, returning zero for an out-of-range entry.
///
/// Mirrors `adc::score_b8`, which skips entries a malformed table cannot
/// supply rather than panicking: a corrupted code must degrade the score, not
/// stop the device.
#[inline(always)]
fn lookup(table: &[i32], row: usize, code: u8) -> i32 {
    match table.get(row + code as usize) {
        Some(v) => *v,
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_quant::codebook::{Codebook, Scale};

    const M: usize = 16;
    const DS: usize = 8;
    const CENTROIDS: usize = 256;
    const PI: usize = M; // b=8: one byte per subspace

    fn table_for(query: &[i8], comps: &[i8], out: &mut [i32]) {
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(comps, CENTROIDS, DS, scale).unwrap());
        adc::build_table(query, &cbs, out).unwrap();
    }

    fn fixture() -> ([i8; CENTROIDS * DS], [i8; M * DS], [i32; M * CENTROIDS]) {
        let comps: [i8; CENTROIDS * DS] = core::array::from_fn(|i| ((i * 37) % 101) as i8 - 50);
        let query: [i8; M * DS] = core::array::from_fn(|i| ((i * 13) % 61) as i8 - 30);
        let mut table = [0i32; M * CENTROIDS];
        table_for(&query, &comps, &mut table);
        (comps, query, table)
    }

    #[test]
    fn the_scan_finds_the_true_top_k() {
        let (_, _, table) = fixture();
        const N: usize = 500;
        let codes: [u8; N * PI] = core::array::from_fn(|i| ((i * 31) % 256) as u8);

        let mut s = [0i32; 10];
        let mut i = [0u32; 10];
        let mut heap = Heap::new(&mut s, &mut i, 10).unwrap();
        let stats = scan_b8(&codes, 0, PI, &table, CENTROIDS, &mut heap);
        assert_eq!(stats.scanned, N as u32);

        // Brute-force the same corpus and compare the top 10.
        let mut all: [(i32, u32); N] = core::array::from_fn(|v| {
            let rec = &codes[v * PI..(v + 1) * PI];
            (adc::score_b8(rec, &table, CENTROIDS), v as u32)
        });
        all.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        let mut out = [Candidate { score: 0, id: 0 }; 10];
        let n = heap.drain_sorted(&mut out);
        assert_eq!(n, 10);
        for (got, want) in out.iter().zip(all.iter()) {
            assert_eq!((got.score, got.id), *want);
        }
    }

    #[test]
    fn the_threshold_test_rejects_most_of_the_corpus() {
        // The reason the scan is cheap: the sift path runs rarely. Reported
        // rather than assumed, since a threshold that admits everything would
        // still be correct and much slower.
        let (_, _, table) = fixture();
        const N: usize = 2_000;
        let codes: [u8; N * PI] = core::array::from_fn(|i| ((i * 53) % 256) as u8);

        let mut s = [0i32; 100];
        let mut i = [0u32; 100];
        let mut heap = Heap::new(&mut s, &mut i, 100).unwrap();
        let stats = scan_b8(&codes, 0, PI, &table, CENTROIDS, &mut heap);

        assert_eq!(stats.scanned, N as u32);
        assert!(
            stats.offered * 4 < stats.scanned,
            "threshold admitted {} of {}",
            stats.offered,
            stats.scanned
        );
        assert_eq!(stats.evicted, stats.offered - 100);
    }

    #[test]
    fn ids_are_offset_by_the_shard_base() {
        // A scan over a block range reports absolute vector ids, so a dropped
        // block's ids match what the layout says it held.
        let (_, _, table) = fixture();
        let codes: [u8; 32 * PI] = core::array::from_fn(|i| ((i * 7) % 256) as u8);

        let mut s = [0i32; 4];
        let mut i = [0u32; 4];
        let mut heap = Heap::new(&mut s, &mut i, 4).unwrap();
        scan_b8(&codes, 3_200, PI, &table, CENTROIDS, &mut heap);

        let mut out = [Candidate { score: 0, id: 0 }; 4];
        let n = heap.drain_sorted(&mut out);
        for c in &out[..n] {
            assert!((3_200..3_232).contains(&c.id), "id {} out of range", c.id);
        }
    }

    #[test]
    fn a_partial_trailing_record_is_ignored() {
        // `chunks_exact` drops a short tail rather than scoring a truncated
        // record, which would produce a plausible wrong score.
        let (_, _, table) = fixture();
        let codes = [0u8; PI * 3 + 5];
        let mut s = [0i32; 8];
        let mut i = [0u32; 8];
        let mut heap = Heap::new(&mut s, &mut i, 8).unwrap();
        let stats = scan_b8(&codes, 0, PI, &table, CENTROIDS, &mut heap);
        assert_eq!(stats.scanned, 3);
    }

    #[test]
    fn scanning_in_block_ranges_matches_one_pass() {
        // The scan is called per block on a buffered backend and over the whole
        // region on an XIP one. Both must give the same result.
        let (_, _, table) = fixture();
        const N: usize = 256;
        let codes: [u8; N * PI] = core::array::from_fn(|i| ((i * 29) % 256) as u8);

        let mut s1 = [0i32; 16];
        let mut i1 = [0u32; 16];
        let mut whole = Heap::new(&mut s1, &mut i1, 16).unwrap();
        scan_b8(&codes, 0, PI, &table, CENTROIDS, &mut whole);
        let mut a = [Candidate { score: 0, id: 0 }; 16];
        let na = whole.drain_sorted(&mut a);

        let mut s2 = [0i32; 16];
        let mut i2 = [0u32; 16];
        let mut chunked = Heap::new(&mut s2, &mut i2, 16).unwrap();
        // 512 B blocks hold 32 vectors at PI=16.
        for (b, block) in codes.chunks_exact(32 * PI).enumerate() {
            scan_b8(block, (b * 32) as u32, PI, &table, CENTROIDS, &mut chunked);
        }
        let mut c = [Candidate { score: 0, id: 0 }; 16];
        let nc = chunked.drain_sorted(&mut c);

        assert_eq!(na, nc);
        assert_eq!(a, c);
    }

    #[test]
    fn four_bit_and_eight_bit_scans_agree_on_the_same_codes() {
        let comps: [i8; 16 * DS] = core::array::from_fn(|i| ((i * 11) % 127) as i8);
        let query: [i8; M * DS] = core::array::from_fn(|i| (i % 50) as i8);
        let scale = Scale::new(1, 1000).unwrap();
        let cbs: [Codebook<'_>; M] =
            core::array::from_fn(|_| Codebook::new(&comps, 16, DS, scale).unwrap());
        let mut table = [0i32; M * 16];
        adc::build_table(&query, &cbs, &mut table).unwrap();

        const N: usize = 64;
        let wide: [u8; N * M] = core::array::from_fn(|i| (i % 16) as u8);
        let mut packed = [0u8; N * (M / 2)];
        for v in 0..N {
            for j in 0..M {
                let c = wide[v * M + j];
                let slot = &mut packed[v * (M / 2) + j / 2];
                *slot = if j.is_multiple_of(2) {
                    (*slot & 0xF0) | c
                } else {
                    (*slot & 0x0F) | (c << 4)
                };
            }
        }

        let mut s1 = [0i32; 8];
        let mut i1 = [0u32; 8];
        let mut h8 = Heap::new(&mut s1, &mut i1, 8).unwrap();
        scan_b8(&wide, 0, M, &table, 16, &mut h8);
        let mut a = [Candidate { score: 0, id: 0 }; 8];
        h8.drain_sorted(&mut a);

        let mut s2 = [0i32; 8];
        let mut i2 = [0u32; 8];
        let mut h4 = Heap::new(&mut s2, &mut i2, 8).unwrap();
        scan_b4(&packed, 0, M / 2, &table, 16, M, &mut h4);
        let mut b = [Candidate { score: 0, id: 0 }; 8];
        h4.drain_sorted(&mut b);

        assert_eq!(a, b);
    }

    #[test]
    fn the_four_wide_scan_is_bit_identical_to_the_scalar_one() {
        // The interleaved version exists for speed only. If it can return a
        // different answer it is not an optimisation, so the two are compared
        // on their full output rather than on a recall average.
        const M: usize = 16;
        const K: usize = 256;
        const N: usize = 1000;

        let mut table = [0i32; M * K];
        for (i, slot) in table.iter_mut().enumerate() {
            // Deterministic spread with both signs, so ties and evictions occur.
            *slot = ((i as i32).wrapping_mul(2_654_435_761u32 as i32) >> 11) % 10_000;
        }
        let mut codes = [0u8; N * M];
        for (i, slot) in codes.iter_mut().enumerate() {
            *slot = ((i * 37 + i / 7) % K) as u8;
        }

        let mut s1 = [0i32; 100];
        let mut i1 = [0u32; 100];
        let mut s4 = [0i32; 100];
        let mut i4 = [0u32; 100];
        let mut o1 = [Candidate { score: 0, id: 0 }; 100];
        let mut o4 = [Candidate { score: 0, id: 0 }; 100];

        for cap in [1usize, 10, 100] {
            let mut h1 = Heap::new(&mut s1[..cap], &mut i1[..cap], cap).unwrap();
            let a = scan_b8(&codes, 0, M, &table, K, &mut h1);
            let n1 = h1.drain_sorted(&mut o1[..cap]);

            let mut h4 = Heap::new(&mut s4[..cap], &mut i4[..cap], cap).unwrap();
            let b = scan_b8_x4(&codes, 0, M, &table, K, &mut h4);
            let n4 = h4.drain_sorted(&mut o4[..cap]);

            assert_eq!(a, b, "stats differ at cap {cap}");
            assert_eq!(n1, n4);
            assert_eq!(o1[..n1], o4[..n4], "results differ at cap {cap}");
        }
    }

    #[test]
    fn the_four_wide_scan_handles_a_tail_that_is_not_a_multiple_of_four() {
        // 1001 records: 250 full groups and one left over. An off-by-one in the
        // tail would drop or double-count the last vectors.
        const M: usize = 4;
        const K: usize = 8;
        const N: usize = 1001;

        let table: [i32; M * K] = core::array::from_fn(|i| (i as i32 * 13) % 97);
        let codes: [u8; N * M] = core::array::from_fn(|i| ((i * 5) % K) as u8);

        let cap = 16;
        let mut s1 = [0i32; 16];
        let mut i1 = [0u32; 16];
        let mut h1 = Heap::new(&mut s1, &mut i1, cap).unwrap();
        let a = scan_b8(&codes, 0, M, &table, K, &mut h1);

        let mut s4 = [0i32; 16];
        let mut i4 = [0u32; 16];
        let mut h4 = Heap::new(&mut s4, &mut i4, cap).unwrap();
        let b = scan_b8_x4(&codes, 0, M, &table, K, &mut h4);

        assert_eq!(a.scanned, N as u32);
        assert_eq!(a, b);

        let mut o1 = [Candidate { score: 0, id: 0 }; 16];
        let mut o4 = [Candidate { score: 0, id: 0 }; 16];
        h1.drain_sorted(&mut o1);
        h4.drain_sorted(&mut o4);
        assert_eq!(o1, o4);
    }

    #[test]
    fn the_auto_scan_matches_the_shape_it_selects() {
        // scan_b8_auto must be a dispatch and nothing more. If it ever diverges
        // from the shape it selects, host and device results differ with no
        // other symptom.
        const M: usize = 8;
        const K: usize = 16;
        const N: usize = 40;
        let table: [i32; M * K] = core::array::from_fn(|i| (i as i32 * 53) % 401 - 200);
        let codes: [u8; N * M] = core::array::from_fn(|i| ((i * 7) % K) as u8);

        let mut sa = [0i32; 8];
        let mut ia = [0u32; 8];
        let mut ha = Heap::new(&mut sa, &mut ia, 8).unwrap();
        let auto = scan_b8_auto(&codes, 0, M, &table, K, &mut ha);
        let mut ca = [Candidate { score: 0, id: 0 }; 8];
        ha.drain_sorted(&mut ca);

        let mut sb = [0i32; 8];
        let mut ib = [0u32; 8];
        let mut hb = Heap::new(&mut sb, &mut ib, 8).unwrap();
        let direct = if cfg!(target_pointer_width = "64") {
            scan_b8(&codes, 0, M, &table, K, &mut hb)
        } else {
            scan_b8_x4(&codes, 0, M, &table, K, &mut hb)
        };
        let mut cb = [Candidate { score: 0, id: 0 }; 8];
        hb.drain_sorted(&mut cb);

        assert_eq!(auto, direct);
        assert_eq!(ca, cb);
    }
}
