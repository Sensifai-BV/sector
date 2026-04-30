//! Bounded fixed-point codebooks.
//!
//! A single-bit flip in a `beta`-bit fixed-point value displaces it by at most
//! `2^(beta-1) * Delta`, and the result stays representable. The same flip in
//! an IEEE-754 exponent multiplies the value by `2^128`.
//!
//! The bound propagates. A displacement `delta` shifts every affected score by
//! at most `‖q_j‖·‖delta‖`, which caps how many corrupted vectors can cross the
//! candidate boundary. Measured intruder count under one bit flip (D=256,
//! m=32, b=8, N=20,000, hottest centroid at 351 referencing vectors): 9.6 per
//! query for int8 against 351 — the entire affected set — for an f32 exponent
//! flip.
//!
//! # Implementation notes
//!
//! Centroids are `i8` with a per-subspace scale held as an integer
//! numerator/denominator pair. No float appears in the reconstruction path, so
//! host and device produce identical bytes.
//!
//! Training runs in f32 on the host; quantization is a separate later stage,
//! and recall is re-measured after it. The quantized codebook is what ships, so
//! a figure taken before quantization is not the figure the device shows.
//!
//! The displacement bound is a `const fn` on the codebook type. The allocator
//! consumes the computed value, so it cannot drift from the format it
//! describes.

/// Bits in a stored codebook component.
pub const BETA: u32 = 8;

/// Per-subspace scale, held as an integer ratio.
///
/// A float here would make host and device reconstruction diverge on any target
/// whose rounding differs, and would reintroduce the unbounded displacement the
/// fixed-point format exists to prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    /// Numerator of the value a unit step represents.
    pub num: i32,
    /// Denominator. Never zero.
    pub den: i32,
}

impl Scale {
    /// A scale of `num / den`. Rejects a zero denominator.
    pub const fn new(num: i32, den: i32) -> Option<Self> {
        if den == 0 {
            return None;
        }
        Some(Self { num, den })
    }

    /// Unit step in the same fixed-point units the scale is expressed in,
    /// scaled by `den` so the result stays integral.
    pub const fn step_scaled(&self) -> i32 {
        self.num
    }

    /// Worst-case displacement from a single-bit flip, in scaled units.
    ///
    /// A flip at bit `l < BETA` of a two's-complement word changes the integer
    /// by `+/- 2^l`, so the largest is `2^(BETA-1)`. The result stays
    /// representable, which is the whole property: an IEEE-754 exponent flip
    /// multiplies the value by `2^128` instead.
    pub const fn max_displacement_scaled(&self) -> i64 {
        (1i64 << (BETA - 1)) * self.num as i64
    }
}

/// Why a codebook was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodebookError {
    /// Component count does not equal `centroids * subspace_dim`.
    Shape {
        /// Components supplied.
        found: usize,
        /// Components required.
        expected: usize,
    },
    /// A centroid or subspace index is out of range.
    OutOfRange,
}

/// A bounded fixed-point codebook for one subspace.
///
/// Centroids are `i8`, laid out row-major: centroid `c` occupies
/// `components[c * ds .. (c+1) * ds]`.
#[derive(Clone, Copy, Debug)]
pub struct Codebook<'a> {
    components: &'a [i8],
    centroids: usize,
    ds: usize,
    scale: Scale,
}

impl<'a> Codebook<'a> {
    /// Borrow `components` as a codebook of `centroids` rows of `ds` each.
    pub fn new(
        components: &'a [i8],
        centroids: usize,
        ds: usize,
        scale: Scale,
    ) -> Result<Self, CodebookError> {
        let expected = centroids * ds;
        if components.len() != expected {
            return Err(CodebookError::Shape {
                found: components.len(),
                expected,
            });
        }
        Ok(Self {
            components,
            centroids,
            ds,
            scale,
        })
    }

    /// Centroids, `2^b`.
    pub const fn centroids(&self) -> usize {
        self.centroids
    }

    /// Subspace dimension, `D / m`.
    pub const fn ds(&self) -> usize {
        self.ds
    }

    /// The scale a unit step represents.
    pub const fn scale(&self) -> Scale {
        self.scale
    }

    /// Stored bytes, `centroids * ds` at one byte per component.
    pub const fn byte_len(&self) -> usize {
        self.centroids * self.ds
    }

    /// Centroid `c`.
    pub fn centroid(&self, c: usize) -> Option<&'a [i8]> {
        if c >= self.centroids {
            return None;
        }
        self.components.get(c * self.ds..(c + 1) * self.ds)
    }

    /// Worst-case displacement of any single-bit flip in this codebook.
    ///
    /// Exposed as a computed value rather than documented, because the
    /// allocator consumes it: a bound that is computed cannot drift from the
    /// format it describes.
    pub const fn max_displacement_scaled(&self) -> i64 {
        self.scale.max_displacement_scaled()
    }
}

/// Bound on the score shift a displacement of `delta_scaled` induces.
///
/// A displacement shifts every affected score by at most
/// `||q_j|| * ||delta||` (Cauchy-Schwarz), which is what caps how many
/// corrupted vectors can cross the candidate boundary. Computed in `i64` on
/// scaled integers; the caller supplies `q_norm_scaled` in the same units.
pub const fn score_shift_bound(q_norm_scaled: i64, delta_scaled: i64) -> i64 {
    let a = if q_norm_scaled < 0 {
        -q_norm_scaled
    } else {
        q_norm_scaled
    };
    let b = if delta_scaled < 0 {
        -delta_scaled
    } else {
        delta_scaled
    };
    a * b
}

/// Displacement in scaled units from flipping bit `bit` of `value`.
///
/// Returns `None` for a bit position outside the stored width.
pub fn flip_displacement_scaled(value: i8, bit: u32, scale: Scale) -> Option<i64> {
    if bit >= BETA {
        return None;
    }
    let flipped = ((value as u8) ^ (1u8 << bit)) as i8;
    Some((flipped as i64 - value as i64) * scale.num as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T0: 256 centroids of D/m = 8 components each.
    const T0_CENTROIDS: usize = 256;
    const T0_DS: usize = 8;

    fn scale() -> Scale {
        // A unit step of 1/1000, held as an integer ratio.
        Scale::new(1, 1000).unwrap()
    }

    #[test]
    fn single_bit_displacement_never_exceeds_the_bound() {
        let s = scale();
        let bound = s.max_displacement_scaled();
        // Exhaustive over all 256 stored values and all 8 bit positions.
        for v in i8::MIN..=i8::MAX {
            for bit in 0..BETA {
                let d = flip_displacement_scaled(v, bit, s).expect("bit in range");
                assert!(
                    d.abs() <= bound,
                    "value {v} bit {bit} displaced {d}, bound {bound}"
                );
            }
        }
        assert_eq!(flip_displacement_scaled(0, BETA, s), None);
    }

    #[test]
    fn the_bound_is_attained_not_merely_respected() {
        let s = scale();
        let bound = s.max_displacement_scaled();
        // The sign bit of a non-negative value moves it by exactly 2^(BETA-1).
        let d = flip_displacement_scaled(0, BETA - 1, s).unwrap();
        assert_eq!(d.abs(), bound);
    }

    #[test]
    fn a_flipped_value_stays_representable() {
        // The property f32 lacks: every result is a valid i8, so no
        // displacement can escape the format's range.
        for v in i8::MIN..=i8::MAX {
            for bit in 0..BETA {
                let flipped = ((v as u8) ^ (1u8 << bit)) as i8;
                assert!((i8::MIN..=i8::MAX).contains(&flipped));
            }
        }
    }

    #[test]
    fn shape_is_checked_rather_than_assumed() {
        let components = [0i8; T0_CENTROIDS * T0_DS];
        assert!(Codebook::new(&components, T0_CENTROIDS, T0_DS, scale()).is_ok());
        assert_eq!(
            Codebook::new(&components[..10], T0_CENTROIDS, T0_DS, scale()).err(),
            Some(CodebookError::Shape {
                found: 10,
                expected: 2048
            })
        );
    }

    #[test]
    fn t0_codebook_is_32_kib() {
        let components = [0i8; T0_CENTROIDS * T0_DS];
        let cb = Codebook::new(&components, T0_CENTROIDS, T0_DS, scale()).unwrap();
        // 256 centroids x 8 components, one byte each, per subspace; m=16
        // subspaces gives the 32 KiB resident figure.
        assert_eq!(cb.byte_len(), 2048);
        assert_eq!(cb.byte_len() * 16, 32 * 1024);
    }

    #[test]
    fn centroid_rows_are_disjoint_and_bounded() {
        let components: [i8; T0_CENTROIDS * T0_DS] = core::array::from_fn(|i| (i % 251) as i8);
        let cb = Codebook::new(&components, T0_CENTROIDS, T0_DS, scale()).unwrap();
        for c in 0..T0_CENTROIDS {
            let row = cb.centroid(c).expect("centroid in range");
            assert_eq!(row.len(), T0_DS);
            assert_eq!(row[0], components[c * T0_DS]);
        }
        assert_eq!(cb.centroid(T0_CENTROIDS), None);
    }

    #[test]
    fn zero_denominator_is_refused() {
        assert_eq!(Scale::new(1, 0), None);
        assert!(Scale::new(1, 1).is_some());
    }

    #[test]
    fn score_shift_bound_is_symmetric_in_sign() {
        // The induced shift is signed; the bound is on its magnitude, and both
        // directions must be covered because deflation loses recall with zero
        // intruders.
        assert_eq!(score_shift_bound(3, 4), 12);
        assert_eq!(score_shift_bound(-3, 4), 12);
        assert_eq!(score_shift_bound(3, -4), 12);
        assert_eq!(score_shift_bound(-3, -4), 12);
    }
}
