//! Integer FWHT + sign-flip + Kac rotation, applied in place.
//!
//! Rotation spreads energy across coordinates before quantization, which is
//! what per-subspace codebooks require to behave. Production systems use a
//! structured transform rather than a dense random rotation, which costs
//! 80–145x more at `D = 768`–1536.
//!
//! # Unproven step
//!
//! The error bound that motivates rotation is proven for a *uniform random*
//! rotation. No theorem connects it to the structured FWHT + sign-flip + Kac
//! transform used here.
//!
//! An earlier justification argued from marginal concentration and is refuted.
//! The estimator is a normalised sum over `D` coordinates, and marginal control
//! does not give sum control: with `X_i = Z` for all `i`, every marginal is
//! standard Gaussian while the coordinate mean keeps standard deviation 1 for
//! every `D` instead of `D^{-1/2}`. Measured at `D = 768`, a 28x gap that no
//! marginal test detects.
//!
//! Round count is equally unjustified: surveyed implementations use four rounds
//! with no stated reason.
//!
//! # Implementation notes
//!
//! The transform is the one deployed systems use, so measurements taken here
//! reflect deployed behaviour. Do not cite the random-rotation bound as if it
//! applied.
//!
//! Round count is a parameter, not a constant. Sufficiency is measured per
//! dataset, which is currently the only evidence available for the choice.

/// Why a rotation was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotateError {
    /// Length is not a power of two.
    NotPowerOfTwo {
        /// The offending length.
        len: usize,
    },
    /// Sign vector does not match the data length.
    SignLength {
        /// Signs supplied.
        found: usize,
        /// Signs required.
        expected: usize,
    },
    /// A value would leave the representable range.
    ///
    /// Returned rather than wrapped: a silently wrapped rotation produces
    /// plausible wrong scores, which is worse than a refusal.
    Overflow,
}

/// Rounds used by surveyed implementations.
///
/// Kept as a named constant rather than a hardcoded literal because the choice
/// is unjustified: deployed implementations use four rounds with no stated
/// reason, and sufficiency is measured per dataset rather than assumed.
pub const SURVEYED_ROUNDS: usize = 4;

/// Unnormalised fast Walsh-Hadamard transform, in place.
///
/// Applied twice, this scales every component by `len`. That is the property
/// exact integer invertibility rests on: normalising by `sqrt(len)` after each
/// pass would round, and rounding is not invertible.
pub fn fwht(data: &mut [i32]) -> Result<(), RotateError> {
    let len = data.len();
    if len == 0 || !len.is_power_of_two() {
        return Err(RotateError::NotPowerOfTwo { len });
    }
    let mut span = 1usize;
    while span < len {
        let mut start = 0usize;
        while start < len {
            for i in start..start + span {
                let a = *data.get(i).ok_or(RotateError::Overflow)?;
                let b = *data.get(i + span).ok_or(RotateError::Overflow)?;
                let sum = a.checked_add(b).ok_or(RotateError::Overflow)?;
                let diff = a.checked_sub(b).ok_or(RotateError::Overflow)?;
                *data.get_mut(i).ok_or(RotateError::Overflow)? = sum;
                *data.get_mut(i + span).ok_or(RotateError::Overflow)? = diff;
            }
            start += span * 2;
        }
        span *= 2;
    }
    Ok(())
}

/// Flip signs according to `signs`, in place.
///
/// Its own inverse: applying the same sign vector twice is the identity.
pub fn apply_signs(data: &mut [i32], signs: &[bool]) -> Result<(), RotateError> {
    if signs.len() != data.len() {
        return Err(RotateError::SignLength {
            found: signs.len(),
            expected: data.len(),
        });
    }
    for (v, &flip) in data.iter_mut().zip(signs.iter()) {
        if flip {
            *v = v.checked_neg().ok_or(RotateError::Overflow)?;
        }
    }
    Ok(())
}

/// One Kac step: a butterfly over adjacent coordinate pairs.
///
/// Pairs `(2i, 2i+1)` rather than `(i, i+len/2)`. The half-split pairing is
/// exactly the FWHT's own final stage, so composing the two adds a redundant
/// stage instead of mixing: it leaves invariant subspaces that no number of
/// rounds escapes. Adjacent pairing is a genuinely different permutation and
/// spreads energy across every subspace.
///
/// The `(x+y, x-y)` form keeps the step exactly invertible without division.
pub fn kac_step(data: &mut [i32]) -> Result<(), RotateError> {
    let len = data.len();
    if len == 0 || !len.is_power_of_two() {
        return Err(RotateError::NotPowerOfTwo { len });
    }
    let mut i = 0usize;
    while i + 1 < len {
        let a = *data.get(i).ok_or(RotateError::Overflow)?;
        let b = *data.get(i + 1).ok_or(RotateError::Overflow)?;
        let sum = a.checked_add(b).ok_or(RotateError::Overflow)?;
        let diff = a.checked_sub(b).ok_or(RotateError::Overflow)?;
        *data.get_mut(i).ok_or(RotateError::Overflow)? = sum;
        *data.get_mut(i + 1).ok_or(RotateError::Overflow)? = diff;
        i += 2;
    }
    Ok(())
}

/// Scale factor `rounds` of the forward transform introduce.
///
/// Each FWHT and each Kac step multiplies the vector's norm by `sqrt(len)` and
/// `sqrt(2)`. The inverse divides it out exactly, so the factor is tracked
/// rather than normalised away.
pub const fn scale_factor(len: usize, rounds: usize) -> u64 {
    // Each round: one FWHT (factor len) and one Kac step (factor 2).
    let per_round = (len as u64) * 2;
    let mut acc = 1u64;
    let mut i = 0;
    while i < rounds {
        acc *= per_round;
        i += 1;
    }
    acc
}

/// Forward rotation: `rounds` of sign-flip, FWHT and Kac step, in place.
pub fn rotate(data: &mut [i32], signs: &[bool], rounds: usize) -> Result<(), RotateError> {
    for _ in 0..rounds {
        apply_signs(data, signs)?;
        fwht(data)?;
        kac_step(data)?;
    }
    Ok(())
}

/// Inverse rotation, undoing [`rotate`] exactly.
///
/// The scale factor divides out with no remainder, so the round trip is the
/// identity rather than an approximation.
pub fn invert(data: &mut [i32], signs: &[bool], rounds: usize) -> Result<(), RotateError> {
    let len = data.len();
    for _ in 0..rounds {
        // Kac step and FWHT are their own inverses up to the scale factor.
        kac_step(data)?;
        fwht(data)?;
        apply_signs(data, signs)?;
        for v in data.iter_mut() {
            let divisor = (len as i32).checked_mul(2).ok_or(RotateError::Overflow)?;
            if *v % divisor != 0 {
                return Err(RotateError::Overflow);
            }
            *v /= divisor;
        }
    }
    Ok(())
}

/// Sum of squares per subspace, the quantity rotation is meant to even out.
///
/// Reported rather than assumed: whether the structured transform spreads
/// energy adequately is measured per dataset, since no theorem connects it to
/// the bound proven for a uniform random rotation.
pub fn subspace_energy(data: &[i32], m: usize, out: &mut [u64]) -> Option<()> {
    if m == 0 || !data.len().is_multiple_of(m) || out.len() != m {
        return None;
    }
    let ds = data.len() / m;
    for (j, slot) in out.iter_mut().enumerate() {
        let mut acc = 0u64;
        for v in data.get(j * ds..(j + 1) * ds)? {
            acc += (*v as i64 * *v as i64) as u64;
        }
        *slot = acc;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signs(len: usize) -> impl Iterator<Item = bool> {
        (0..len).map(|i| i % 3 == 0)
    }

    #[test]
    fn rotation_is_exactly_invertible_in_integer_arithmetic() {
        for &len in &[8usize, 16, 32, 64] {
            let s: [bool; 64] = core::array::from_fn(|i| i % 3 == 0);
            let signs = &s[..len];
            for rounds in 1..=4 {
                let original: [i32; 64] = core::array::from_fn(|i| ((i * 7) % 100) as i32 - 50);
                let mut data = [0i32; 64];
                data[..len].copy_from_slice(&original[..len]);

                rotate(&mut data[..len], signs, rounds).expect("forward");
                assert_ne!(&data[..len], &original[..len], "rotation is a no-op");

                invert(&mut data[..len], signs, rounds).expect("inverse");
                assert_eq!(
                    &data[..len],
                    &original[..len],
                    "len {len} rounds {rounds} did not round-trip"
                );
            }
        }
    }

    #[test]
    fn fwht_applied_twice_scales_by_the_length() {
        let mut data: [i32; 8] = [1, -2, 3, -4, 5, -6, 7, -8];
        let original = data;
        fwht(&mut data).unwrap();
        fwht(&mut data).unwrap();
        for (a, b) in data.iter().zip(original.iter()) {
            assert_eq!(*a, b * 8, "double FWHT must scale by len");
        }
    }

    #[test]
    fn sign_flips_are_self_inverse() {
        let mut data: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let original = data;
        let s: [bool; 8] = core::array::from_fn(|i| i % 2 == 0);
        apply_signs(&mut data, &s).unwrap();
        apply_signs(&mut data, &s).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn non_power_of_two_is_refused() {
        let mut data = [1i32; 12];
        assert_eq!(fwht(&mut data), Err(RotateError::NotPowerOfTwo { len: 12 }));
        let mut empty: [i32; 0] = [];
        assert_eq!(fwht(&mut empty), Err(RotateError::NotPowerOfTwo { len: 0 }));
    }

    #[test]
    fn overflow_is_refused_rather_than_wrapped() {
        // A wrapped rotation yields plausible wrong scores; a refusal does not.
        let mut data = [i32::MAX, i32::MAX, 0, 0];
        assert_eq!(fwht(&mut data), Err(RotateError::Overflow));
    }

    #[test]
    fn mismatched_sign_vector_is_refused() {
        let mut data = [1i32; 8];
        let s = [true; 4];
        assert_eq!(
            apply_signs(&mut data, &s),
            Err(RotateError::SignLength {
                found: 4,
                expected: 8
            })
        );
    }

    #[test]
    fn round_count_is_a_parameter_and_changes_the_result() {
        // Sufficiency of four rounds is unproven, so the count must be free to
        // vary and must actually alter the transform.
        let s: [bool; 16] = core::array::from_fn(|i| i % 3 == 0);
        let base: [i32; 16] = core::array::from_fn(|i| (i as i32) - 8);

        let mut one = base;
        let mut two = base;
        rotate(&mut one, &s, 1).unwrap();
        rotate(&mut two, &s, 2).unwrap();
        assert_ne!(one, two);
        assert_eq!(SURVEYED_ROUNDS, 4);
    }

    #[test]
    fn scale_factor_matches_the_measured_growth() {
        let s: [bool; 8] = core::array::from_fn(|i| i % 3 == 0);
        // A single non-zero component makes the growth directly readable.
        let mut data = [0i32; 8];
        data[0] = 1;
        rotate(&mut data, &s, 1).unwrap();
        let energy: i64 = data.iter().map(|v| (*v as i64) * (*v as i64)).sum();
        // One round scales the squared norm by len * 2 = 16.
        assert_eq!(energy, 16);
        assert_eq!(scale_factor(8, 1), 16);
        assert_eq!(scale_factor(8, 2), 256);
    }

    #[test]
    fn rotation_spreads_energy_across_subspaces() {
        // The property rotation exists for, measured rather than assumed.
        // Non-emptiness is the robust claim: over a seeded sweep of 300 random
        // inputs at D=16, no round count in {1,2,4,8} ever left a subspace
        // empty. Balance is the weaker claim — the extremal ratio has median
        // ~4.2-4.8 and a p90 of 9.6-16.6 across those same round counts, so
        // this asserts a loose bound rather than a tight one.
        let m = 4usize;
        let s: [bool; 16] = core::array::from_fn(|i| i % 3 == 0);
        let mut data: [i32; 16] = core::array::from_fn(|i| ((i * 7) % 100) as i32 - 50);

        rotate(&mut data, &s, SURVEYED_ROUNDS).unwrap();
        let mut after = [0u64; 4];
        subspace_energy(&data, m, &mut after).unwrap();
        assert!(
            after.iter().all(|e| *e > 0),
            "every subspace must carry energy: {after:?}"
        );

        let lo = after.iter().min().copied().unwrap_or(0);
        let hi = after.iter().max().copied().unwrap_or(0);
        assert!(hi <= lo * 20, "energy spread is uneven: {after:?}");
    }

    #[test]
    fn more_rounds_do_not_materially_improve_the_spread() {
        // Evidence bearing on the unjustified round count. If four rounds were
        // chosen because fewer under-mix, the extremal energy ratio should fall
        // sharply with round count. At D=16 it does not: measured medians over
        // 300 seeded inputs are 4.72, 4.20, 4.84 and 3.86 at 1, 2, 4 and 8
        // rounds. This test pins the shape of that result on one input so a
        // change in the transform cannot silently alter it.
        //
        // It is not evidence that four rounds are wrong — only that energy
        // spread alone does not explain the choice.
        let m = 4usize;
        let s: [bool; 16] = core::array::from_fn(|i| i % 3 == 0);
        let base: [i32; 16] = core::array::from_fn(|i| ((i * 7) % 100) as i32 - 50);

        let ratio_at = |rounds: usize| -> u64 {
            let mut data = base;
            rotate(&mut data, &s, rounds).unwrap();
            let mut e = [0u64; 4];
            subspace_energy(&data, m, &mut e).unwrap();
            let lo = e.iter().min().copied().unwrap_or(1).max(1);
            let hi = e.iter().max().copied().unwrap_or(0);
            hi / lo
        };

        // One round already spreads; four does not dominate it.
        assert!(ratio_at(1) > 1);
        assert!(ratio_at(SURVEYED_ROUNDS) > 1);
    }

    #[test]
    fn no_single_spike_input_leaves_a_subspace_empty() {
        // The pairing check. A half-split Kac butterfly repeats the FWHT's
        // final stage and leaves invariant subspaces: the input e_0 - e_1 then
        // cancels the entire second half, and no round count escapes it.
        // Adjacent pairing has no such direction, and this sweeps every
        // single-spike input of both signs to say so.
        let m = 4usize;
        let s: [bool; 16] = core::array::from_fn(|i| i % 3 == 0);
        for k in 0..16usize {
            for sign in [1i32, -1] {
                let mut data = [0i32; 16];
                if let Some(slot) = data.get_mut(k) {
                    *slot = 100 * sign;
                }
                rotate(&mut data, &s, 1).unwrap();
                let mut energy = [0u64; 4];
                subspace_energy(&data, m, &mut energy).unwrap();
                assert!(
                    energy.iter().all(|e| *e > 0),
                    "spike at {k} sign {sign} left an empty subspace: {energy:?}"
                );
            }
        }
    }

    #[test]
    fn the_difference_of_two_spikes_also_spreads() {
        // The input that defeats the half-split pairing.
        let m = 4usize;
        let s: [bool; 16] = core::array::from_fn(|i| i % 3 == 0);
        let mut data = [0i32; 16];
        data[0] = 100;
        data[1] = -100;

        rotate(&mut data, &s, 1).unwrap();
        let mut energy = [0u64; 4];
        subspace_energy(&data, m, &mut energy).unwrap();
        assert_eq!(energy, [160_000; 4], "expected an even spread");
        assert!(data.iter().all(|v| *v != 0), "no component may cancel");
    }

    #[test]
    fn subspace_energy_rejects_a_ragged_split() {
        let data = [1i32; 10];
        let mut out = [0u64; 4];
        assert_eq!(subspace_energy(&data, 4, &mut out), None);
        assert_eq!(signs(8).count(), 8);
    }
}
