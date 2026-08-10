//! Tier profiles.
//!
//! A profile fixes the quantization and retrieval parameters for one target
//! class. Every derived quantity is a `const fn` and every feasibility check is
//! a `const` assertion, so an unrealisable configuration fails the build.
//!
//! # Codebook size drives the configuration
//!
//! The codebook is `2^b * D * s` bytes and does not depend on `m`. At `s = 1`
//! (int8) this puts `b = 8` inside T0's 192 KiB budget for any `D <= 384`, and
//! out of reach at `D = 768`, where it is the entire budget. Dimension, not
//! subspace count, decides whether 8-bit codes are affordable.
//!
//! At equal payload size *and* equal dimension, 8-bit codes measure 1.35x the
//! recall of 4-bit ones: recall@10 at D=128, m=16, b=8 is 0.605 at R=100 and
//! 0.934 at R=500, against 0.4485 and 0.788 for D=128, m=32, b=4 at the same
//! 16 B/vector payload.
//!
//! A 2.5x figure appears in earlier documents. It compares 0.605 against 0.243,
//! which is a D=256, m=32, b=4 measurement — it varies dimension as well as
//! code width, so it does not support an equal-payload claim.

/// Quantization and retrieval parameters for one target class.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    /// Vector dimension.
    pub d: usize,
    /// PQ subspaces.
    pub m: usize,
    /// Bits per code; `2^b` centroids per subspace.
    pub b: usize,
    /// Bytes per stored codebook component. 1 = int8 fixed point.
    pub cb_bytes: usize,
    /// Bytes per rerank-copy component. 1 = int8.
    pub rerank_bytes: usize,
    /// Bytes per ADC table accumulator. 2 = i16, 4 = i32.
    pub adc_bytes: usize,
    /// Candidate depth for stage two.
    pub r: usize,
    /// Neighbours returned.
    pub k: usize,
    /// Resident RAM budget for the whole index.
    pub ram_budget: usize,
    /// Stack reserved for firmware outside the engine. Excluded from the index
    /// budget so the residual figure is the space codes may actually occupy.
    pub stack_reserve: usize,
}

impl Profile {
    /// Subspace dimension, `D / m`.
    pub const fn ds(&self) -> usize {
        self.d / self.m
    }

    /// Centroids per subspace, `2^b`.
    pub const fn centroids(&self) -> usize {
        1 << self.b
    }

    /// Payload bytes per vector, `m * b / 8`.
    pub const fn payload_bytes(&self) -> usize {
        self.m * self.b / 8
    }

    /// Codebook bytes, `2^b * D * s`. Independent of `N` and of `m`.
    pub const fn codebook_bytes(&self) -> usize {
        self.centroids() * self.d * self.cb_bytes
    }

    /// ADC table bytes, `m * 2^b` accumulators.
    pub const fn adc_table_bytes(&self) -> usize {
        self.m * self.centroids() * self.adc_bytes
    }

    /// Multiply-accumulates to build the ADC table: `2^b * D` per query,
    /// independent of `N`.
    pub const fn table_macs(&self) -> usize {
        self.centroids() * self.d
    }

    /// Candidate heap bytes: `(i32 score, u32 id)` per slot.
    pub const fn heap_bytes(&self) -> usize {
        self.r * 8
    }

    /// Rerank-copy bytes per vector.
    pub const fn rerank_bytes_per_vec(&self) -> usize {
        self.d * self.rerank_bytes
    }

    /// Resident bytes that do not scale with `N`: codebook, ADC table, heap,
    /// and the reserved stack.
    pub const fn fixed_bytes(&self) -> usize {
        self.codebook_bytes() + self.adc_table_bytes() + self.heap_bytes() + self.stack_reserve
    }

    /// Budget left for PQ codes after the fixed set.
    ///
    /// The codebook replica is charged to flash, not here: one working copy is
    /// resident and repair reads the replica from NOR on CRC mismatch. Holding
    /// both in RAM would cost 32 KiB of code budget, about 2,000 vectors, to
    /// avoid a fault-path flash read.
    pub const fn code_budget(&self) -> usize {
        self.ram_budget - self.fixed_bytes()
    }

    /// Vectors whose codes fit in RAM alongside the fixed set.
    pub const fn resident_vectors(&self) -> usize {
        self.code_budget() / self.payload_bytes()
    }

    /// Rerank copies that fit in `flash_bytes` after two codebook copies.
    pub const fn rerank_capacity(&self, flash_bytes: usize) -> usize {
        (flash_bytes - 2 * self.codebook_bytes()) / self.rerank_bytes_per_vec()
    }
}

/// T0 — ESP32-C3, 160 MHz, no FPU, 4 MB NOR with execute-in-place.
///
/// `D = 128` matches SIFT and the common output width of edge embedding models.
/// It admits `b = 8` in a 32 KiB codebook. Measured on full SIFT1M
/// (`N = 1,000,000`, 200 queries, shipped ground truth): recall@10 of 0.9605 at
/// `R = 100` and 0.9975 at `R = 500`. The synthetic corpus gave 0.605 and 0.934
/// at the same configuration, so it understated real embeddings by 0.36 at the
/// operating point.
///
/// At equal payload and equal dimension a `b = 4` configuration
/// (`m = 32`) measures 0.4485 at `R = 100` on the synthetic corpus.
///
/// At 16 B/vector the codes are RAM-resident: roughly 9,000 vectors, with
/// rerank copies in NOR. This differs from the large-`D` case, where codes
/// stream from the mapped window because they cannot fit.
pub const T0: Profile = Profile {
    d: 128,
    m: 16,
    b: 8,
    cb_bytes: 1,
    rerank_bytes: 1,
    adc_bytes: 2,
    r: 500,
    k: 10,
    ram_budget: 192 * 1024,
    stack_reserve: 8 * 1024,
};

/// T1 — ESP32-S3, 240 MHz, PSRAM, 16 MB NOR.
///
/// PSRAM removes the resident constraint, so `m` doubles for finer subspace
/// resolution at 32 B/vector. The codebook is unchanged at `2^b * D`.
pub const T1: Profile = Profile {
    d: 128,
    m: 32,
    b: 8,
    cb_bytes: 1,
    rerank_bytes: 1,
    adc_bytes: 4,
    r: 500,
    k: 10,
    ram_budget: 6 * 1024 * 1024,
    stack_reserve: 32 * 1024,
};

/// T0 at GIST-class dimension, where the codebook forces narrower codes.
///
/// Retained as the boundary case rather than a deployment target. `2^8 * 768`
/// is 196,608 B against T0's 192 KiB budget, so `b = 8` does not fit; the exact
/// cutoff is `D = 735`, above which no `m` makes it fit.
///
/// Measured on GIST1M at D=960, N=100,000, 200 queries: **0.579 at `R = 100`**
/// with `b = 4`, rising to 0.9425 at `R = 1000`. An earlier figure of 0.243 at
/// D=768 is superseded — the measured value is more than double it at a higher
/// dimension.
///
/// `b = 6` is the better boundary configuration and is not represented here:
/// at `m = 120` it fits T0 with a 61,440 B codebook and reaches 0.940 at
/// `R = 100`. A corpus at this dimension is therefore usable on T0, contrary to
/// the earlier reading that it must reduce dimension or move to T1.
pub const T0_WIDE: Profile = Profile {
    d: 768,
    m: 32,
    b: 4,
    ..T0
};

/// T2 — 32-bit ARM Linux: Pi 1, Pi Zero, Pi Zero W, Pi 2, and any later board
/// running a 32-bit userland.
///
/// # What sets each parameter
///
/// `adc_bytes = 2` is set by the ARM1176 in BCM2835, the weakest member: 16 KiB
/// of L1 data cache, and no usable L2 (BCM2835's 128 KiB L2 is wired to the
/// VideoCore). The ADC table is indexed `m` times per vector with no locality,
/// so it must stay cache-resident or every lookup is a miss. At `m = 16, b = 8`
/// an i16 table is 8,192 B — half of L1, leaving room for the payload stream.
/// An i32 table would be 16,384 B, the entire L1, and would evict the codes it
/// is being used to score.
///
/// `m = 16` follows from the same bound. Doubling `m` to T1's 32 doubles the
/// table with it.
///
/// `r = 100` is set by storage, not by RAM, and is the parameter that separates
/// this tier from T0. Stage two fetches one rerank record per candidate; on a
/// microSD or eMMC part behind a flash translation layer that is a random read
/// the FTL services at block granularity, where T0's raw NOR services it as a
/// load instruction from a mapped window. Candidate depth is therefore charged
/// at a rate a mapped-NOR tier does not pay, and the recall this costs — 0.605
/// at `R = 100` against 0.934 at `R = 500`, measured at `D = 128, m = 16,
/// b = 8` — is the price of the storage rather than of the processor.
///
/// # `ram_budget` is a floor, not a target
///
/// 32 MiB is what every member of the tier has, including a 256 MB Pi 1 Model A
/// after BCM2835's minimum 64 MB GPU carve-out. It is not what a 1 GB Pi 2
/// should use. The figure `resident_vectors()` reports here is the capacity
/// guaranteed across the tier; a board with more memory raises it at runtime
/// and the volume's own `N` is what binds in practice. Reading this constant as
/// a per-board maximum understates a Pi 2 by a factor of 30.
pub const T2: Profile = Profile {
    d: 128,
    m: 16,
    b: 8,
    cb_bytes: 1,
    rerank_bytes: 1,
    adc_bytes: 2,
    r: 100,
    k: 10,
    ram_budget: 32 * 1024 * 1024,
    // A hosted process, unlike T0's bare-metal single stack: glibc's main
    // thread grows to `ulimit -s`, 8 MiB by default. 2 MiB is the reserve for
    // everything outside the index — the daemon's per-connection state and the
    // C library's own arenas — on the assumption the engine is not running on
    // the main thread.
    stack_reserve: 2 * 1024 * 1024,
};

/// T3 — 64-bit ARM Linux: Pi Zero 2 W, Pi 2 v1.2, Pi 3, Pi 4, Pi 5.
///
/// # What sets each parameter
///
/// Every member is Cortex-A53 or later with at least 512 KiB of shared L2, so
/// the constraint that fixed T2's `adc_bytes` is gone: `m = 32` with i32
/// accumulators is a 32 KiB table, L2-resident on an A53 and inside a single
/// A76 core's private L2 on a Pi 5. Finer subspace resolution at 32 B/vector
/// follows, matching T1.
///
/// `r = 500` returns to the depth T2 could not afford. The justification is
/// memory rather than storage: at 128 B/vector the rerank copies for a
/// million-vector corpus are 128 MB, so on every board in this tier except the
/// Zero 2 W they can be held resident and stage two stops touching the FTL at
/// all. That is the tier's actual advantage, and it is worth stating precisely
/// because it is easy to mis-attribute: the gain comes from having enough RAM
/// to eliminate the storage path, not from the processor being faster.
///
/// # `ram_budget` is a floor, not a target
///
/// 64 MiB is what a 512 MB Pi Zero 2 W can give the index. A 16 GB Pi 5 is
/// three orders of magnitude away from that and is not described by this
/// constant; see [`T2`] on why the figure is a floor.
pub const T3: Profile = Profile {
    d: 128,
    m: 32,
    b: 8,
    cb_bytes: 1,
    rerank_bytes: 1,
    adc_bytes: 4,
    r: 500,
    k: 10,
    ram_budget: 64 * 1024 * 1024,
    stack_reserve: 8 * 1024 * 1024,
};

// ---------------------------------------------------------------------------
// Feasibility. Checked at compile time; a profile that does not fit is a build
// error, not a runtime failure.
// ---------------------------------------------------------------------------

#[allow(clippy::manual_is_multiple_of)] // `is_multiple_of` is not const on stable
const _: () = assert!(T0.d % T0.m == 0, "D must partition into m subspaces");
#[allow(clippy::manual_is_multiple_of)]
const _: () = assert!(T1.d % T1.m == 0, "D must partition into m subspaces");
#[allow(clippy::manual_is_multiple_of)]
const _: () = assert!(
    T0_WIDE.d % T0_WIDE.m == 0,
    "D must partition into m subspaces"
);

const _: () = assert!(T0.b == 4 || T0.b == 8, "only 4- and 8-bit codes are packed");
const _: () = assert!(
    T0.fixed_bytes() < T0.ram_budget,
    "T0 fixed set leaves no room for codes"
);
const _: () = assert!(
    T0.resident_vectors() > 8_000,
    "T0 resident capacity regressed"
);
const _: () = assert!(
    T1.fixed_bytes() < T1.ram_budget,
    "T1 fixed set exceeds its budget"
);

#[allow(clippy::manual_is_multiple_of)]
const _: () = assert!(T2.d % T2.m == 0, "D must partition into m subspaces");
#[allow(clippy::manual_is_multiple_of)]
const _: () = assert!(T3.d % T3.m == 0, "D must partition into m subspaces");
const _: () = assert!(
    T2.fixed_bytes() < T2.ram_budget,
    "T2 fixed set exceeds the tier's floor budget"
);
const _: () = assert!(
    T3.fixed_bytes() < T3.ram_budget,
    "T3 fixed set exceeds the tier's floor budget"
);

// The cache bound that fixes T2's accumulator width. ARM1176 in BCM2835 has
// 16 KiB of L1 data cache and no L2 available to the CPU, and the ADC table is
// indexed with no locality, so a table exceeding half of L1 evicts the codes it
// scores. This is the constraint stated in T2's documentation, asserted so a
// later change to `m` or `adc_bytes` fails the build rather than silently
// costing a cache miss per lookup.
const _: () = {
    const ARM1176_L1_DATA_BYTES: usize = 16 * 1024;
    assert!(
        T2.adc_table_bytes() * 2 <= ARM1176_L1_DATA_BYTES,
        "T2 ADC table exceeds half of ARM1176 L1 data cache"
    );
    let widened = Profile { adc_bytes: 4, ..T2 };
    assert!(
        widened.adc_table_bytes() == ARM1176_L1_DATA_BYTES,
        "at m=16, b=8 an i32 ADC table is exactly ARM1176's L1 data cache"
    );
};

// T3's table is sized against L2 rather than L1: 32 KiB at m=32 with i32
// accumulators, against the 512 KiB shared L2 of the weakest member (A53).
const _: () = {
    const A53_MIN_L2_BYTES: usize = 512 * 1024;
    assert!(
        T3.adc_table_bytes() <= A53_MIN_L2_BYTES / 8,
        "T3 ADC table is too large a share of the weakest member's L2"
    );
};

// The tiers must remain distinguishable in the parameters that define them.
// T2 and T3 differing only in `ram_budget` would mean the split describes the
// boards rather than the engine's configuration, and the profile would not be
// carrying its weight.
const _: () = assert!(
    T2.m != T3.m && T2.adc_bytes != T3.adc_bytes && T2.r != T3.r,
    "T2 and T3 must differ in subspace count, accumulator width and depth"
);

// The dimension boundary that forces the code width, asserted so the constraint
// is checked rather than described.
const _: () = {
    let wide_b8 = Profile { b: 8, ..T0_WIDE };
    assert!(
        wide_b8.codebook_bytes() == T0.ram_budget,
        "at D=768 an 8-bit int8 codebook is exactly the T0 budget"
    );
    let wide_b8_f32 = Profile {
        b: 8,
        cb_bytes: 4,
        ..T0_WIDE
    };
    assert!(
        wide_b8_f32.codebook_bytes() == 4 * T0.ram_budget,
        "at D=768 an 8-bit f32 codebook is four times the T0 budget"
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t0_fits_and_holds_target_capacity() {
        assert_eq!(T0.ds(), 8);
        assert_eq!(T0.centroids(), 256);
        assert_eq!(T0.payload_bytes(), 16);
        assert_eq!(T0.codebook_bytes(), 32 * 1024);
        assert_eq!(T0.adc_table_bytes(), 8 * 1024);
        assert_eq!(T0.table_macs(), 32_768);
        assert_eq!(T0.heap_bytes(), 4_000);
        // 32 KiB codebook + 8 KiB table + 4000 B heap + 8 KiB stack = 51.9 KiB
        assert_eq!(T0.fixed_bytes(), 53_152);
        assert_eq!(T0.resident_vectors(), 8_966);
        assert_eq!(T0.rerank_bytes_per_vec(), 128);
        assert_eq!(T0.rerank_bytes_per_vec() / T0.payload_bytes(), 8);
    }

    #[test]
    fn t0_rerank_copies_fit_nor() {
        assert!(T0.rerank_capacity(4 * 1024 * 1024) > 30_000);
    }

    #[test]
    fn t2_table_stays_within_the_weakest_l1() {
        // The tier's binding constraint. 8,192 B is half of ARM1176's 16 KiB
        // L1 data cache; the codes being scored occupy the other half.
        assert_eq!(T2.adc_table_bytes(), 8 * 1024);
        assert_eq!(T2.payload_bytes(), 16);
        assert_eq!(T2.codebook_bytes(), 32 * 1024);
    }

    #[test]
    fn t3_doubles_subspace_resolution_over_t2() {
        // Twice the subspaces at twice the accumulator width: a 32 KiB table,
        // which needs the L2 that no T2 member has.
        assert_eq!(T3.adc_table_bytes(), 4 * T2.adc_table_bytes());
        assert_eq!(T3.payload_bytes(), 32);
        // The codebook is 2^b * D and does not depend on m, so it is unchanged.
        assert_eq!(T3.codebook_bytes(), T2.codebook_bytes());
    }

    #[test]
    fn pi_tier_floors_hold_a_useful_corpus() {
        // The floor budgets must leave room for a corpus worth serving, or the
        // tier is describing a board that cannot run the engine.
        assert!(
            T2.resident_vectors() > 1_800_000,
            "T2 floor holds {} vectors",
            T2.resident_vectors()
        );
        assert!(
            T3.resident_vectors() > 1_700_000,
            "T3 floor holds {} vectors",
            T3.resident_vectors()
        );
    }

    #[test]
    fn a_million_vector_rerank_set_fits_t3_but_not_t2() {
        // The claim in T3's documentation: at 128 B/vector, a million rerank
        // copies are 128 MB — resident on a 1 GB+ board, and twice T2's entire
        // floor budget. This is what buys R=500 back.
        let million = 1_000_000 * T3.rerank_bytes_per_vec();
        assert_eq!(million, 128_000_000);
        assert!(million > 2 * T2.ram_budget);
    }

    #[test]
    fn every_tier_fits_its_own_budget() {
        // The const assertions already enforce this at compile time. Repeated
        // as a test so a reader sees the property named, and so the failure
        // message identifies which tier regressed.
        for (name, p) in [("T0", T0), ("T1", T1), ("T2", T2), ("T3", T3)] {
            assert!(
                p.fixed_bytes() < p.ram_budget,
                "{name} fixed set {} B exceeds budget {} B",
                p.fixed_bytes(),
                p.ram_budget
            );
            assert!(p.resident_vectors() > 0, "{name} holds no vectors");
        }
    }

    #[test]
    fn wide_dimension_forces_four_bit_codes() {
        // 2^8 * 768 = 192 KiB, the whole budget, hence b=4 at this dimension.
        assert_eq!(Profile { b: 8, ..T0_WIDE }.codebook_bytes(), 192 * 1024);
        assert_eq!(T0_WIDE.codebook_bytes(), 12 * 1024);
        // The cost of that: payload parity but 2.5x worse measured recall.
        assert_eq!(T0_WIDE.payload_bytes(), T0.payload_bytes());
    }

    #[test]
    fn t1_widens_subspaces_not_codebook() {
        assert_eq!(T1.codebook_bytes(), T0.codebook_bytes());
        assert_eq!(T1.payload_bytes(), 32);
        assert_eq!(T1.table_macs(), T0.table_macs());
    }
}
