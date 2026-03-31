//! Block CRC-32 and the verify-or-drop decision.
//!
//! An MDS code corrects `n - k` erasures but only `floor((n - k) / 2)` unknown
//! errors. Silent flash corruption is an error channel; the CRC localises
//! damage and converts it to an erasure channel, restoring the full `n - k`
//! figure. The CRC is a precondition of the coding arithmetic, not an economy
//! measure.
//!
//! Granularity sets the cost. Per-vector CRC-32 against a 16-byte payload is
//! 25% overhead, against the ~0.8% the whole protection scheme spends. At
//! 512-byte blocks it is `4/512 = 0.78%`; at 4 KiB sectors, 0.10%.
//!
//! A *detected* block failure drops every vector in the block — 32 at a 16-byte
//! payload. `Delta_payload = 1` holds per corrupted byte, not per detected
//! failure.
//!
//! # Placement and cost
//!
//! CRCs are stored out of line in a block-parallel array, so the payload region
//! stays exactly `pi`-strided and the scan needs no per-vector address
//! arithmetic.
//!
//! Verification is lazy: only the `R` candidates surviving stage one, never
//! every scanned vector. Verifying `N` blocks per query costs more than the
//! scan, and stage two is where a corrupted byte would change an answer.
//!
//! Table-driven, standard polynomial. On a 160 MHz core a 512-byte block is
//! negligible against the flash read it accompanies.

/// Reflected CRC-32 polynomial (IEEE 802.3), the standard choice.
pub const POLYNOMIAL: u32 = 0xEDB8_8320;

/// Initial and final register value.
const SEED: u32 = 0xFFFF_FFFF;

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut n = 0usize;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                POLYNOMIAL ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
}

/// Byte-at-a-time lookup table, 1 KiB in rodata.
static TABLE: [u32; 256] = build_table();

/// CRC-32 over `bytes`.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = SEED;
    for &byte in bytes {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        // The mask bounds `idx` to 0..256, so this lookup cannot be out of range.
        let entry = match TABLE.get(idx) {
            Some(v) => *v,
            None => 0,
        };
        crc = entry ^ (crc >> 8);
    }
    crc ^ SEED
}

/// Whether `bytes` matches `expected`.
///
/// Returned rather than asserted: a mismatch is countable degradation on the
/// query path, not a fault the engine may panic on.
pub fn verify(bytes: &[u8], expected: u32) -> bool {
    crc32(bytes) == expected
}

/// Overhead of one CRC per `block_bytes`, in parts per million of stored bytes.
///
/// Expressed in ppm because the figure is an integer ratio and the core family
/// admits no floating point. At 512 B blocks this is 7,812 ppm (0.78%); at
/// 4 KiB, 976 ppm (0.10%); per-vector against a 16 B payload, 250,000 ppm (25%).
pub const fn overhead_ppm(block_bytes: usize) -> usize {
    (crate::CRC_BYTES * 1_000_000) / block_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answers() {
        // The check value every CRC-32 implementation is expected to reproduce.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    }

    #[test]
    fn detects_every_single_bit_flip_in_a_block() {
        let block = [0xA5u8; 512];
        let clean = crc32(&block);
        for byte in 0..block.len() {
            for bit in 0..8 {
                let mut damaged = block;
                match damaged.get_mut(byte) {
                    Some(b) => *b ^= 1 << bit,
                    None => unreachable!(),
                }
                assert_ne!(
                    crc32(&damaged),
                    clean,
                    "missed flip at byte {byte} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn overhead_matches_the_stated_figures() {
        // 4/512 = 0.78%, the granularity the payload region uses.
        assert_eq!(overhead_ppm(512), 7_812);
        // 4/4096 = 0.10% at sector granularity.
        assert_eq!(overhead_ppm(4096), 976);
        // 4/16 = 25%, the per-vector alternative against a T0 payload.
        assert_eq!(overhead_ppm(16), 250_000);
    }

    #[test]
    fn verify_agrees_with_crc32() {
        let data = b"sector volume block";
        assert!(verify(data, crc32(data)));
        assert!(!verify(data, crc32(data) ^ 1));
    }
}
