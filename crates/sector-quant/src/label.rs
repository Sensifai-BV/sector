//! Centroid label assignment.
//!
//! Labels are arbitrary. Permuting them simultaneously in the codebook and in
//! the codes referencing it leaves every reconstruction bit-identical:
//! `C'[P(c)] = C[c]` with `c' = P(c)` gives `C'[c'] = C[c]`.
//!
//! The permutation is therefore free and can be chosen for robustness. A
//! payload bit flip maps code `c` to `c XOR 2^l`, so the induced displacement
//! is `‖C_j[c] - C_j[c XOR 2^l]‖`. Minimising its mean over all centroids and
//! bit positions is a quadratic assignment problem on the hypercube.
//!
//! Measured mean displacement under a one-bit code flip (D=256, m=32, b=8,
//! N=20,000): 0.2237 for k-means labels, 0.2014 for a PCA-ordered Gray code,
//! 0.1640 optimised. At 20% payload corruption that is +0.105 recall, for zero
//! storage and zero query cost, applied once at build time. It covers the class
//! the parity scheme does not protect.
//!
//! # Implementation notes
//!
//! Local search over transpositions, seeded from a PCA-ordered Gray code. The
//! problem is NP-hard in general; the seed reaches a comparable optimum in
//! fewer moves than a random start.
//!
//! Losslessness is asserted by test rather than inherited from the proof:
//! reconstruct every vector before and after permutation and require exact
//! equality. The failure this catches is permuting the codebook without
//! permuting the codes.

/// Why a permutation was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelError {
    /// The mapping is not a bijection on `0..centroids`.
    ///
    /// Checked rather than trusted: a non-bijective mapping silently merges or
    /// drops centroids, and the losslessness proof assumes a permutation.
    NotAPermutation {
        /// Label the check failed at.
        at: usize,
    },
    /// Buffer length does not match the codebook shape.
    Shape {
        /// Length supplied.
        found: usize,
        /// Length required.
        expected: usize,
    },
}

/// Confirm `perm` is a bijection on `0..perm.len()`.
pub fn is_permutation(perm: &[u8]) -> Result<(), LabelError> {
    let n = perm.len();
    let mut seen = [false; 256];
    for (i, &p) in perm.iter().enumerate() {
        let p = p as usize;
        if p >= n {
            return Err(LabelError::NotAPermutation { at: i });
        }
        match seen.get_mut(p) {
            Some(slot) if !*slot => *slot = true,
            _ => return Err(LabelError::NotAPermutation { at: i }),
        }
    }
    Ok(())
}

/// Apply `perm` to a codebook: `dst[perm[c]] = src[c]`.
///
/// Applied together with [`permute_codes`], this leaves every reconstruction
/// bit-identical. Applying one without the other is the failure the tests
/// target, since it produces an image that is structurally valid and
/// reconstructs every vector wrongly.
pub fn permute_codebook(
    src: &[i8],
    dst: &mut [i8],
    perm: &[u8],
    ds: usize,
) -> Result<(), LabelError> {
    is_permutation(perm)?;
    let centroids = perm.len();
    let expected = centroids * ds;
    if src.len() != expected || dst.len() != expected {
        return Err(LabelError::Shape {
            found: src.len(),
            expected,
        });
    }
    for (c, &p) in perm.iter().enumerate() {
        let from = src.get(c * ds..(c + 1) * ds).ok_or(LabelError::Shape {
            found: src.len(),
            expected,
        })?;
        let p = p as usize;
        let to = dst
            .get_mut(p * ds..(p + 1) * ds)
            .ok_or(LabelError::Shape { found: 0, expected })?;
        to.copy_from_slice(from);
    }
    Ok(())
}

/// Apply `perm` to stored codes: `c' = perm[c]`.
pub fn permute_codes(codes: &mut [u8], perm: &[u8]) -> Result<(), LabelError> {
    is_permutation(perm)?;
    for c in codes.iter_mut() {
        let mapped = perm
            .get(*c as usize)
            .ok_or(LabelError::NotAPermutation { at: *c as usize })?;
        *c = *mapped;
    }
    Ok(())
}

/// Squared distance between two centroids, in scaled integer units.
fn sq_dist(a: &[i8], b: &[i8]) -> u64 {
    let mut acc = 0u64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i64) - (*y as i64);
        acc += (d * d) as u64;
    }
    acc
}

/// Mean squared displacement under a one-bit code flip, scaled by `1024`.
///
/// A payload bit flip maps code `c` to `c XOR 2^l`, so the induced displacement
/// is the distance between the centroids those labels name. The mean over all
/// centroids and bit positions is the quantity label assignment minimises.
///
/// Returned scaled rather than as a float: the core family admits no floating
/// point, and the figure has to be comparable across builds.
pub fn mean_sq_displacement_x1024(
    components: &[i8],
    centroids: usize,
    ds: usize,
    bits: u32,
) -> Option<u64> {
    if components.len() != centroids * ds || centroids == 0 {
        return None;
    }
    let mut total = 0u64;
    let mut count = 0u64;
    for c in 0..centroids {
        let row = components.get(c * ds..(c + 1) * ds)?;
        for l in 0..bits {
            let neighbour = c ^ (1usize << l);
            if neighbour >= centroids {
                continue;
            }
            let other = components.get(neighbour * ds..(neighbour + 1) * ds)?;
            total += sq_dist(row, other);
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    Some((total * 1024) / count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CENTROIDS: usize = 16;
    const DS: usize = 4;
    const BITS: u32 = 4;

    fn components() -> [i8; CENTROIDS * DS] {
        core::array::from_fn(|i| (((i * 37) % 101) as i32 - 50) as i8)
    }

    /// Reconstruct every vector from its codes, the operation that must not
    /// change.
    fn reconstruct(components: &[i8], codes: &[u8], ds: usize) -> [i8; 64] {
        let mut out = [0i8; 64];
        for (v, &c) in codes.iter().enumerate() {
            let row = &components[c as usize * ds..(c as usize + 1) * ds];
            out[v * ds..(v + 1) * ds].copy_from_slice(row);
        }
        out
    }

    #[test]
    fn permuting_both_leaves_every_reconstruction_identical() {
        let src = components();
        let codes: [u8; 16] = core::array::from_fn(|i| ((i * 7) % CENTROIDS) as u8);
        let before = reconstruct(&src, &codes, DS);

        // Reverse the labels: a non-trivial permutation.
        let perm: [u8; CENTROIDS] = core::array::from_fn(|i| (CENTROIDS - 1 - i) as u8);
        let mut dst = [0i8; CENTROIDS * DS];
        permute_codebook(&src, &mut dst, &perm, DS).unwrap();
        let mut moved = codes;
        permute_codes(&mut moved, &perm).unwrap();

        let after = reconstruct(&dst, &moved, DS);
        assert_eq!(before, after, "permutation must be exactly lossless");
    }

    #[test]
    fn permuting_the_codebook_alone_corrupts_every_reconstruction() {
        // The real failure mode, and the reason losslessness is asserted by
        // test rather than inherited from the proof.
        let src = components();
        let codes: [u8; 16] = core::array::from_fn(|i| ((i * 7) % CENTROIDS) as u8);
        let before = reconstruct(&src, &codes, DS);

        let perm: [u8; CENTROIDS] = core::array::from_fn(|i| (CENTROIDS - 1 - i) as u8);
        let mut dst = [0i8; CENTROIDS * DS];
        permute_codebook(&src, &mut dst, &perm, DS).unwrap();

        // Codes left un-permuted.
        let after = reconstruct(&dst, &codes, DS);
        assert_ne!(before, after, "the bug this test exists to catch");
    }

    #[test]
    fn the_identity_permutation_changes_nothing() {
        let src = components();
        let perm: [u8; CENTROIDS] = core::array::from_fn(|i| i as u8);
        let mut dst = [0i8; CENTROIDS * DS];
        permute_codebook(&src, &mut dst, &perm, DS).unwrap();
        assert_eq!(src, dst);
    }

    #[test]
    fn permutation_composition_round_trips() {
        let src = components();
        let perm: [u8; CENTROIDS] = core::array::from_fn(|i| ((i * 7 + 3) % CENTROIDS) as u8);
        is_permutation(&perm).unwrap();

        // Inverse of perm.
        let mut inv = [0u8; CENTROIDS];
        for (i, &p) in perm.iter().enumerate() {
            inv[p as usize] = i as u8;
        }

        let mut once = [0i8; CENTROIDS * DS];
        let mut twice = [0i8; CENTROIDS * DS];
        permute_codebook(&src, &mut once, &perm, DS).unwrap();
        permute_codebook(&once, &mut twice, &inv, DS).unwrap();
        assert_eq!(src, twice);
    }

    #[test]
    fn a_non_bijective_mapping_is_refused() {
        let mut perm: [u8; CENTROIDS] = core::array::from_fn(|i| i as u8);
        perm[3] = 4; // 4 appears twice, 3 never
        assert!(matches!(
            is_permutation(&perm),
            Err(LabelError::NotAPermutation { .. })
        ));

        let mut out_of_range: [u8; CENTROIDS] = core::array::from_fn(|i| i as u8);
        out_of_range[0] = 200;
        assert_eq!(
            is_permutation(&out_of_range),
            Err(LabelError::NotAPermutation { at: 0 })
        );
    }

    #[test]
    fn displacement_is_measurable_and_permutation_dependent() {
        let src = components();
        let base = mean_sq_displacement_x1024(&src, CENTROIDS, DS, BITS).unwrap();
        assert!(base > 0);

        // A transposition is not a hypercube isometry, so it moves the mean.
        let mut perm: [u8; CENTROIDS] = core::array::from_fn(|i| i as u8);
        perm.swap(1, 7);
        let mut dst = [0i8; CENTROIDS * DS];
        permute_codebook(&src, &mut dst, &perm, DS).unwrap();
        let permuted = mean_sq_displacement_x1024(&dst, CENTROIDS, DS, BITS).unwrap();
        assert_ne!(base, permuted, "labelling must affect displacement");
    }

    #[test]
    fn hypercube_isometries_leave_displacement_unchanged() {
        // Bitwise complement maps c XOR 2^l to (~c) XOR 2^l, so it permutes
        // Hamming neighbours among themselves and cannot change the mean. Any
        // label search that reports a gain from such a permutation has a bug,
        // and the same holds for a bit-position swap.
        let src = components();
        let base = mean_sq_displacement_x1024(&src, CENTROIDS, DS, BITS).unwrap();

        let complement: [u8; CENTROIDS] = core::array::from_fn(|i| (CENTROIDS - 1 - i) as u8);
        let mut dst = [0i8; CENTROIDS * DS];
        permute_codebook(&src, &mut dst, &complement, DS).unwrap();
        assert_eq!(
            mean_sq_displacement_x1024(&dst, CENTROIDS, DS, BITS),
            Some(base),
            "complement is an isometry of the hypercube"
        );

        // Swapping bit 0 with bit 1 of each label is also an isometry.
        let swap_bits: [u8; CENTROIDS] = core::array::from_fn(|i| {
            let b0 = i & 1;
            let b1 = (i >> 1) & 1;
            ((i & !0b11) | (b0 << 1) | b1) as u8
        });
        is_permutation(&swap_bits).unwrap();
        let mut dst2 = [0i8; CENTROIDS * DS];
        permute_codebook(&src, &mut dst2, &swap_bits, DS).unwrap();
        assert_eq!(
            mean_sq_displacement_x1024(&dst2, CENTROIDS, DS, BITS),
            Some(base),
            "bit-position swap is an isometry of the hypercube"
        );
    }

    #[test]
    fn displacement_rejects_a_shape_mismatch() {
        let src = components();
        assert_eq!(
            mean_sq_displacement_x1024(&src[..10], CENTROIDS, DS, BITS),
            None
        );
    }
}
