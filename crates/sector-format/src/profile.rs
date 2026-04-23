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
//! At equal payload size 8-bit codes are worth roughly 2.5x the recall of
//! 4-bit ones: recall@10 at D=128, m=16, b=8 is 0.605 at R=100 and 0.934 at
//! R=500, against 0.243 and 0.577 for a b=4 configuration of the same
//! 16 B/vector payload.

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
/// It admits `b = 8` in a 32 KiB codebook, which measures recall@10 of 0.605 at
/// `R = 100` and 0.934 at `R = 500` on the synthetic corpus, against 0.243 and
/// 0.577 for a `b = 4` configuration of identical payload size.
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
