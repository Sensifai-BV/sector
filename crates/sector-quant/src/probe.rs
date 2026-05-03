//! Non-inlined wrappers for inspecting emitted assembly.
//!
//! The scan's inner loop must contain no multiplies: every multiply is paid
//! once during table construction. That is a claim about generated code, so it
//! is checked against generated code rather than asserted.
//!
//! The scoring functions are `#[inline]`, so a library build emits no
//! standalone symbol for them. These wrappers are `#[inline(never)]`, which
//! forces a symbol `make asm-check` can disassemble. Names are left mangled —
//! `#[no_mangle]` counts as unsafe under this crate's `forbid(unsafe_code)`,
//! and the checker matches on the demangled substring instead.
//!
//! Behind the `asm-probe` feature, so no device build carries them.

use crate::adc;

/// Wrapper around [`adc::score_b8`].
#[inline(never)]
pub fn probe_score_b8(codes: &[u8], table: &[i32], centroids: usize) -> i32 {
    adc::score_b8(codes, table, centroids)
}

/// Wrapper around [`adc::score_b4`].
#[inline(never)]
pub fn probe_score_b4(codes: &[u8], table: &[i32], centroids: usize, m: usize) -> i32 {
    adc::score_b4(codes, table, centroids, m)
}
