//! `sector-quant` — integer-only product quantization.
//!
//! Fixed-point by construction. A single-bit flip in a `beta`-bit fixed-point
//! value displaces it by at most `2^(beta-1) * Delta` and the result stays
//! representable; the same flip in an IEEE-754 exponent multiplies the value by
//! `2^128`. Measured recall cost of one flipped bit in one codebook entry
//! (D=256, m=32, b=8, N=20,000, hottest centroid, clean two-stage baseline
//! 0.637): 0.0005 for int8, 0.246 for f32.
//!
//! A float here is a correctness defect rather than a portability nuisance.
//! `deny(clippy::float_arithmetic)` enforces it.
//!
//! Arithmetic cost is the second reason: the T0 core has no FPU, and a software
//! float multiply costs 30–100 cycles against 2–4 for the integer path. Table
//! construction is `2^b * D` multiply-accumulates per query, independent of `N`.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![deny(clippy::float_arithmetic)]

/// Asymmetric distance computation tables: `T[j][v] = <q_j, C_j[v]>`.
pub mod adc;
/// Bounded fixed-point codebooks and their displacement bound (Prop. D).
pub mod codebook;
/// Centroid label assignment (Prop. E): permuting labels is exactly lossless,
/// so labels are chosen to minimise Hamming-neighbour displacement.
pub mod label;
/// Integer FWHT + sign-flip + Kac rotation, applied in place.
///
/// Whether the structured transform inherits the error bound proven for a
/// uniform random rotation is Open Problem 1 of the report and is *not* settled
/// here. The implementation is used because deployed systems use it; the
/// unproven step is recorded, not papered over.
pub mod rotate;

/// Non-inlined wrappers for assembly inspection. Not built by default.
#[cfg(feature = "asm-probe")]
pub mod probe;
