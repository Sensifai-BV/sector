//! Payload region: `pi`-strided PQ codes with a block-parallel CRC array.
//!
//! The object the scan streams: `pi = m * b / 8` bytes per vector, 16 B at T0.
//! Strictly strided with no per-vector header, so scanning needs no address
//! computation beyond an increment.
//!
//! # Layout rules
//!
//! CRCs live in a separate parallel array. Inline CRCs break the stride,
//! forcing per-vector address arithmetic into the hottest loop in the system to
//! protect bytes stage one does not verify.
//!
//! At `b = 8` a code is one byte and the stride is `m`. At `b = 4` two codes
//! pack per byte; the unpack is a shift and a mask, cheaper than the doubled
//! traffic of byte-per-code storage.
//!
//! A dropped 512 B block removes 32 vectors at a 16-byte payload. Recall
//! accounting treats those as evictions, and they are counted.

use crate::{BLOCK_BYTES, SECTOR_BYTES};
use sector_codec::CRC_BYTES;

/// Placement of `N` vectors of `pi`-byte codes into fixed-size blocks.
///
/// Vectors never straddle a block boundary. A block is the CRC unit, and a
/// vector split across two blocks would be lost when either failed, doubling
/// its exposure for no gain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadLayout {
    /// Payload bytes per vector, `m * b / 8`.
    pub payload_bytes: usize,
    /// Bytes per block.
    pub block_bytes: usize,
    /// Vectors stored.
    pub n: usize,
}

impl PayloadLayout {
    /// Layout for `n` vectors at `payload_bytes` each, at [`BLOCK_BYTES`].
    pub const fn new(payload_bytes: usize, n: usize) -> Self {
        Self {
            payload_bytes,
            block_bytes: BLOCK_BYTES,
            n,
        }
    }

    /// Vectors per block, `floor(block_bytes / payload_bytes)`.
    ///
    /// 32 at a 512 B block and a 16 B payload: the number a detected block
    /// failure removes.
    pub const fn vectors_per_block(&self) -> usize {
        self.block_bytes / self.payload_bytes
    }

    /// Bytes at the end of each block too small to hold another vector.
    pub const fn slack_bytes(&self) -> usize {
        self.block_bytes - self.vectors_per_block() * self.payload_bytes
    }

    /// Blocks needed for `n` vectors.
    pub const fn blocks(&self) -> usize {
        self.n.div_ceil(self.vectors_per_block())
    }

    /// Payload region extent, rounded up to a whole erase sector.
    pub const fn region_bytes(&self) -> usize {
        (self.blocks() * self.block_bytes).next_multiple_of(SECTOR_BYTES)
    }

    /// CRC array extent, rounded up to a whole erase sector.
    pub const fn crc_region_bytes(&self) -> usize {
        (self.blocks() * CRC_BYTES).next_multiple_of(SECTOR_BYTES)
    }

    /// Block holding vector `id`, or `None` past the end.
    pub const fn block_of(&self, id: usize) -> Option<usize> {
        if id >= self.n {
            return None;
        }
        Some(id / self.vectors_per_block())
    }

    /// Byte offset of vector `id` from the start of the region.
    pub const fn offset_of(&self, id: usize) -> Option<usize> {
        if id >= self.n {
            return None;
        }
        let per = self.vectors_per_block();
        Some((id / per) * self.block_bytes + (id % per) * self.payload_bytes)
    }

    /// Vector ids held in `block`, as a half-open range clamped to `n`.
    ///
    /// The range a detected block failure removes.
    pub const fn ids_in_block(&self, block: usize) -> Option<(usize, usize)> {
        if block >= self.blocks() {
            return None;
        }
        let per = self.vectors_per_block();
        let start = block * per;
        let end = if start + per < self.n {
            start + per
        } else {
            self.n
        };
        Some((start, end))
    }
}

/// Unpack the `j`-th 4-bit code from a `pi`-byte record.
///
/// Two codes per byte: low nibble first. Cheaper than a byte per code, which
/// would double the bytes the scan streams.
#[inline]
pub fn code_b4(record: &[u8], j: usize) -> Option<u8> {
    let byte = *record.get(j / 2)?;
    Some(if j.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    })
}

/// Pack a 4-bit code into the `j`-th position of a record.
pub fn set_code_b4(record: &mut [u8], j: usize, code: u8) -> Option<()> {
    let byte = record.get_mut(j / 2)?;
    let code = code & 0x0F;
    *byte = if j.is_multiple_of(2) {
        (*byte & 0xF0) | code
    } else {
        (*byte & 0x0F) | (code << 4)
    };
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T0: D=128, m=16, b=8, so 16 B per vector.
    const T0_PI: usize = 16;

    #[test]
    fn dropped_block_removes_exactly_32_vectors_at_t0() {
        let l = PayloadLayout::new(T0_PI, 8_966);
        assert_eq!(l.vectors_per_block(), 32);
        assert_eq!(l.slack_bytes(), 0);
        let (start, end) = l.ids_in_block(0).unwrap();
        assert_eq!(end - start, 32);
        let (start, end) = l.ids_in_block(100).unwrap();
        assert_eq!((start, end - start), (3_200, 32));
    }

    #[test]
    fn final_block_is_partial_and_clamped() {
        let l = PayloadLayout::new(T0_PI, 8_966);
        // 8966 = 280 * 32 + 6, so the last block holds 6 vectors.
        assert_eq!(l.blocks(), 281);
        let (start, end) = l.ids_in_block(280).unwrap();
        assert_eq!((start, end), (8_960, 8_966));
        assert_eq!(l.ids_in_block(281), None);
    }

    #[test]
    fn offsets_are_strided_and_never_cross_a_block() {
        let l = PayloadLayout::new(T0_PI, 1_000);
        for id in 0..l.n {
            let off = l.offset_of(id).unwrap();
            let block = l.block_of(id).unwrap();
            let within = off - block * l.block_bytes;
            assert!(within + T0_PI <= l.block_bytes, "vector {id} straddles");
        }
        assert_eq!(l.offset_of(1_000), None);
    }

    #[test]
    fn a_vector_maps_into_the_block_that_claims_it() {
        let l = PayloadLayout::new(T0_PI, 500);
        for block in 0..l.blocks() {
            let (start, end) = l.ids_in_block(block).unwrap();
            for id in start..end {
                assert_eq!(l.block_of(id), Some(block));
            }
        }
    }

    #[test]
    fn regions_are_sector_aligned() {
        let l = PayloadLayout::new(T0_PI, 8_966);
        assert_eq!(l.region_bytes() % SECTOR_BYTES, 0);
        assert_eq!(l.crc_region_bytes() % SECTOR_BYTES, 0);
        // 281 blocks x 512 B = 143.9 KiB, rounded to 36 sectors.
        assert_eq!(l.region_bytes(), 36 * SECTOR_BYTES);
        // 281 CRCs x 4 B = 1124 B, one sector.
        assert_eq!(l.crc_region_bytes(), SECTOR_BYTES);
    }

    #[test]
    fn slack_appears_when_the_payload_does_not_divide_the_block() {
        // T0_WIDE: m=32, b=4 gives 16 B, but m=24,b=4 gives 12 B: 512/12 = 42 r 8.
        let l = PayloadLayout::new(12, 100);
        assert_eq!(l.vectors_per_block(), 42);
        assert_eq!(l.slack_bytes(), 8);
    }

    #[test]
    fn four_bit_codes_round_trip_over_every_value() {
        let mut record = [0u8; 16]; // m=32 at b=4
        for j in 0..32 {
            for code in 0..16u8 {
                set_code_b4(&mut record, j, code).unwrap();
                assert_eq!(code_b4(&record, j), Some(code));
            }
        }
        assert_eq!(code_b4(&record, 32), None);
    }

    #[test]
    fn packing_one_code_leaves_its_neighbour_intact() {
        let mut record = [0u8; 16];
        for j in 0..32 {
            set_code_b4(&mut record, j, (j as u8) & 0x0F).unwrap();
        }
        for j in 0..32 {
            assert_eq!(code_b4(&record, j), Some((j as u8) & 0x0F));
        }
    }
}
