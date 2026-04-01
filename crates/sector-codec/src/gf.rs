//! GF(2^8) log/antilog tables.
//!
//! 512 bytes of rodata reducing field multiplication to two lookups and an add:
//! `a * b = antilog[(log[a] + log[b]) mod 255]`, with zero handled separately.
//! Generated for the AES polynomial `0x11D`.
//!
//! # Implementation notes
//!
//! Tables are built by `const fn` at compile time rather than shipped as a
//! literal or computed at mount. They occupy flash either way, and a computed
//! table cannot be transcribed wrongly.
//!
//! They stay in rodata. At T0 every resident byte competes with the codebook,
//! and these are read-only.

/// Field-generating polynomial, `x^8 + x^4 + x^3 + x^2 + 1`.
pub const POLYNOMIAL: u16 = 0x11D;

/// Generator of the multiplicative group.
pub const GENERATOR: u8 = 0x02;

const fn build_tables() -> ([u8; 256], [u8; 256]) {
    let mut log = [0u8; 256];
    let mut antilog = [0u8; 256];
    let mut x: u16 = 1;
    let mut i = 0usize;
    while i < 255 {
        antilog[i] = x as u8;
        log[x as usize] = i as u8;
        // x *= GENERATOR, reduced by the field polynomial.
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= POLYNOMIAL;
        }
        i += 1;
    }
    // antilog[255] wraps to antilog[0]; log[0] is undefined and stays 0,
    // which `mul` never consults because it short-circuits on a zero operand.
    antilog[255] = 1;
    (log, antilog)
}

const TABLES: ([u8; 256], [u8; 256]) = build_tables();

/// Discrete logarithm base [`GENERATOR`]. Undefined at 0.
pub static LOG: [u8; 256] = TABLES.0;

/// Inverse of [`LOG`].
pub static ANTILOG: [u8; 256] = TABLES.1;

#[inline]
fn log_of(a: u8) -> usize {
    match LOG.get(a as usize) {
        Some(v) => *v as usize,
        None => 0,
    }
}

#[inline]
fn antilog_of(i: usize) -> u8 {
    match ANTILOG.get(i % 255) {
        Some(v) => *v,
        None => 0,
    }
}

/// Field addition, which is XOR.
#[inline]
pub const fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Field multiplication by table lookup.
#[inline]
pub fn mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    antilog_of(log_of(a) + log_of(b))
}

/// Field division. Returns `None` when `b` is zero.
#[inline]
pub fn div(a: u8, b: u8) -> Option<u8> {
    if b == 0 {
        return None;
    }
    if a == 0 {
        return Some(0);
    }
    Some(antilog_of(255 + log_of(a) - log_of(b)))
}

/// Multiplicative inverse. Returns `None` at zero.
#[inline]
pub fn inv(a: u8) -> Option<u8> {
    div(1, a)
}

/// `a^n`.
pub fn pow(a: u8, n: usize) -> u8 {
    if a == 0 {
        return u8::from(n == 0);
    }
    antilog_of((log_of(a) * n) % 255)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_nonzero_element_has_an_inverse() {
        for a in 1..=255u8 {
            let i = inv(a).expect("nonzero has an inverse");
            assert_eq!(mul(a, i), 1, "inverse failed for {a}");
        }
        assert_eq!(inv(0), None);
    }

    #[test]
    fn multiplication_is_commutative_and_associative() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                assert_eq!(mul(a, b), mul(b, a));
            }
        }
        // Associativity over a strided sweep; the full cube is 16.7M triples.
        for a in (0..=255u8).step_by(7) {
            for b in (0..=255u8).step_by(11) {
                for c in (0..=255u8).step_by(13) {
                    assert_eq!(mul(mul(a, b), c), mul(a, mul(b, c)));
                }
            }
        }
    }

    #[test]
    fn multiplication_distributes_over_addition() {
        for a in (0..=255u8).step_by(5) {
            for b in (0..=255u8).step_by(7) {
                for c in (0..=255u8).step_by(11) {
                    assert_eq!(mul(a, add(b, c)), add(mul(a, b), mul(a, c)));
                }
            }
        }
    }

    #[test]
    fn identities_hold() {
        for a in 0..=255u8 {
            assert_eq!(mul(a, 1), a);
            assert_eq!(mul(a, 0), 0);
            assert_eq!(add(a, a), 0);
            assert_eq!(add(a, 0), a);
        }
    }

    #[test]
    fn division_inverts_multiplication() {
        for a in 0..=255u8 {
            for b in 1..=255u8 {
                let q = div(a, b).expect("nonzero divisor");
                assert_eq!(mul(q, b), a);
            }
        }
        assert_eq!(div(1, 0), None);
    }

    #[test]
    fn the_generator_enumerates_the_whole_multiplicative_group() {
        let mut seen = [false; 256];
        let mut x = 1u8;
        for _ in 0..255 {
            assert!(!seen[x as usize], "generator repeats before 255 steps");
            seen[x as usize] = true;
            x = mul(x, GENERATOR);
        }
        assert_eq!(x, 1, "generator has order 255");
        assert!(seen.iter().skip(1).all(|s| *s));
    }

    #[test]
    fn powers_agree_with_repeated_multiplication() {
        for a in (0..=255u8).step_by(9) {
            let mut acc = 1u8;
            for n in 0..12 {
                assert_eq!(pow(a, n), acc, "a={a} n={n}");
                acc = mul(acc, a);
            }
        }
        assert_eq!(pow(0, 0), 1);
        assert_eq!(pow(0, 3), 0);
    }

    #[test]
    fn tables_are_mutual_inverses() {
        for i in 0..255usize {
            assert_eq!(log_of(antilog_of(i)), i);
        }
    }
}
