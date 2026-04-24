//! Region descriptors: base, length, block size, protection class.
//!
//! A volume is a small set of contiguous regions — codebook with replicas,
//! payload, rerank copy, CRC arrays — each with its own block size and
//! protection policy. The descriptor binds a region to an execute-in-place
//! window or a buffered read path without either choice reaching the query
//! code.
//!
//! # Alignment rules
//!
//! Align every region to an erase sector. Flash fails in sector-correlated
//! bursts, so a region straddling a boundary shares a failure domain with its
//! neighbour and cannot bound its own failure probability, which invalidates
//! the allocation computed over it.
//!
//! Store lengths in blocks with the block size in the descriptor. The CRC
//! array's extent is then derivable rather than stored twice, and two stored
//! copies of one fact eventually disagree.

use crate::{BLOCK_BYTES, SECTOR_BYTES};

/// Number of regions in a volume, fixed by the layout.
pub const REGION_COUNT: usize = 6;

/// Encoded size of one region descriptor.
pub const REGION_DESC_BYTES: usize = 16;

/// What a region holds. The discriminant is the on-flash encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionKind {
    /// Working copy of the PQ codebook.
    Codebook = 1,
    /// Replicas of the codebook, interleaved across independent erase sectors.
    CodebookReplica = 2,
    /// `pi`-strided PQ codes.
    Payload = 3,
    /// Block-parallel CRC array covering the payload.
    PayloadCrc = 4,
    /// Higher-precision copy used by stage two.
    Rerank = 5,
    /// Block-parallel CRC array covering the rerank copy.
    RerankCrc = 6,
}

impl RegionKind {
    /// Decode a stored discriminant.
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Codebook),
            2 => Some(Self::CodebookReplica),
            3 => Some(Self::Payload),
            4 => Some(Self::PayloadCrc),
            5 => Some(Self::Rerank),
            6 => Some(Self::RerankCrc),
            _ => None,
        }
    }
}

/// How a region is repaired once the CRC has localised damage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Protection {
    /// CRC only. A failed block is dropped, and its vectors degrade.
    Detect = 0,
    /// Repair from a replica: the T0/T1 codebook default.
    Replicate = 1,
    /// Repair from RS parity, for structures large enough to earn the decode.
    ReedSolomon = 2,
}

impl Protection {
    /// Decode a stored discriminant.
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Detect),
            1 => Some(Self::Replicate),
            2 => Some(Self::ReedSolomon),
            _ => None,
        }
    }
}

/// Why a descriptor was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionError {
    /// Base address is not erase-sector aligned.
    Unaligned {
        /// The offending base address.
        base: u32,
    },
    /// Byte extent is not a whole number of erase sectors.
    ExtentNotSectorMultiple {
        /// The offending byte extent.
        bytes: u64,
    },
    /// Block size is not a divisor of the erase sector.
    BlockSize {
        /// The offending block size.
        block_bytes: u32,
    },
    /// Extent overflows the address space.
    Overflow,
    /// Two regions share bytes.
    Overlap {
        /// Index of the earlier region in the table.
        first: usize,
        /// Index of the region that overlaps it.
        second: usize,
    },
    /// A stored discriminant is not a known value.
    UnknownDiscriminant,
}

/// Base, extent, block size and protection policy for one contiguous region.
///
/// Length is stored in blocks with the block size alongside, so the CRC array's
/// extent is derivable rather than stored a second time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionDesc {
    /// What the region holds.
    pub kind: RegionKind,
    /// How it is repaired.
    pub protection: Protection,
    /// Byte offset from the start of the volume. Erase-sector aligned.
    pub base: u32,
    /// Bytes per block within this region.
    pub block_bytes: u32,
    /// Length in blocks.
    pub blocks: u32,
}

impl RegionDesc {
    /// Byte extent, `blocks * block_bytes`.
    pub const fn byte_len(&self) -> u64 {
        (self.blocks as u64) * (self.block_bytes as u64)
    }

    /// One past the last byte.
    pub const fn end(&self) -> u64 {
        self.base as u64 + self.byte_len()
    }

    /// Reject a descriptor that cannot bound its own failure probability.
    ///
    /// A region straddling an erase-sector boundary shares a failure domain
    /// with its neighbour, which invalidates the allocation computed over it.
    /// Both the base and the extent must therefore be sector-quantised.
    #[allow(clippy::manual_is_multiple_of)] // `is_multiple_of` is not const on stable
    pub const fn validate(&self) -> Result<(), RegionError> {
        if self.base as usize % SECTOR_BYTES != 0 {
            return Err(RegionError::Unaligned { base: self.base });
        }
        if self.block_bytes == 0 || SECTOR_BYTES % self.block_bytes as usize != 0 {
            return Err(RegionError::BlockSize {
                block_bytes: self.block_bytes,
            });
        }
        let bytes = self.byte_len();
        if bytes % SECTOR_BYTES as u64 != 0 {
            return Err(RegionError::ExtentNotSectorMultiple { bytes });
        }
        if self.end() > u32::MAX as u64 {
            return Err(RegionError::Overflow);
        }
        Ok(())
    }

    /// Byte offset of `block` within the volume, or `None` if out of range.
    pub const fn block_offset(&self, block: u32) -> Option<u32> {
        if block >= self.blocks {
            return None;
        }
        Some(self.base + block * self.block_bytes)
    }

    /// Encode to the fixed 16-byte on-flash form, little-endian.
    pub const fn encode(&self) -> [u8; REGION_DESC_BYTES] {
        let base = self.base.to_le_bytes();
        let blk = self.block_bytes.to_le_bytes();
        let n = self.blocks.to_le_bytes();
        [
            self.kind as u8,
            self.protection as u8,
            0,
            0,
            base[0],
            base[1],
            base[2],
            base[3],
            blk[0],
            blk[1],
            blk[2],
            blk[3],
            n[0],
            n[1],
            n[2],
            n[3],
        ]
    }

    /// Decode the on-flash form. Rejects unknown discriminants rather than
    /// guessing, so an unrecognised layout cannot become a wrong answer.
    pub const fn decode(raw: &[u8; REGION_DESC_BYTES]) -> Result<Self, RegionError> {
        let kind = match RegionKind::from_u8(raw[0]) {
            Some(k) => k,
            None => return Err(RegionError::UnknownDiscriminant),
        };
        let protection = match Protection::from_u8(raw[1]) {
            Some(p) => p,
            None => return Err(RegionError::UnknownDiscriminant),
        };
        Ok(Self {
            kind,
            protection,
            base: u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
            block_bytes: u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]),
            blocks: u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]),
        })
    }
}

/// The volume's regions, in table order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionTable {
    /// Descriptors, one per [`RegionKind`].
    pub regions: [RegionDesc; REGION_COUNT],
}

impl RegionTable {
    /// Encoded size of the whole table.
    pub const ENCODED_BYTES: usize = REGION_COUNT * REGION_DESC_BYTES;

    /// Validate every descriptor and check the set is disjoint.
    ///
    /// Overlap is checked here rather than per descriptor because it is a
    /// property of the table: a descriptor cannot know its neighbours.
    pub fn validate(&self) -> Result<(), RegionError> {
        for r in &self.regions {
            r.validate()?;
        }
        for (i, a) in self.regions.iter().enumerate() {
            for (j, b) in self.regions.iter().enumerate().skip(i + 1) {
                if (a.base as u64) < b.end() && (b.base as u64) < a.end() {
                    return Err(RegionError::Overlap {
                        first: i,
                        second: j,
                    });
                }
            }
        }
        Ok(())
    }

    /// The descriptor for `kind`.
    pub fn get(&self, kind: RegionKind) -> Option<&RegionDesc> {
        self.regions.iter().find(|r| r.kind == kind)
    }

    /// Encode the table into `out`, returning the bytes written.
    pub fn encode(&self, out: &mut [u8]) -> Option<usize> {
        if out.len() < Self::ENCODED_BYTES {
            return None;
        }
        for (i, r) in self.regions.iter().enumerate() {
            let enc = r.encode();
            let start = i * REGION_DESC_BYTES;
            out.get_mut(start..start + REGION_DESC_BYTES)?
                .copy_from_slice(&enc);
        }
        Some(Self::ENCODED_BYTES)
    }

    /// Decode a table from `raw`.
    pub fn decode(raw: &[u8]) -> Result<Self, RegionError> {
        let mut regions = [RegionDesc {
            kind: RegionKind::Codebook,
            protection: Protection::Detect,
            base: 0,
            block_bytes: BLOCK_BYTES as u32,
            blocks: 0,
        }; REGION_COUNT];
        for (i, slot) in regions.iter_mut().enumerate() {
            let start = i * REGION_DESC_BYTES;
            let chunk = raw
                .get(start..start + REGION_DESC_BYTES)
                .ok_or(RegionError::Overflow)?;
            let mut buf = [0u8; REGION_DESC_BYTES];
            buf.copy_from_slice(chunk);
            *slot = RegionDesc::decode(&buf)?;
        }
        Ok(Self { regions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(kind: RegionKind, base: u32, blocks: u32) -> RegionDesc {
        RegionDesc {
            kind,
            protection: Protection::Detect,
            base,
            block_bytes: BLOCK_BYTES as u32,
            blocks,
        }
    }

    #[test]
    fn descriptor_round_trips() {
        let d = RegionDesc {
            kind: RegionKind::Payload,
            protection: Protection::Replicate,
            base: 3 * SECTOR_BYTES as u32,
            block_bytes: BLOCK_BYTES as u32,
            blocks: 64,
        };
        assert_eq!(RegionDesc::decode(&d.encode()), Ok(d));
    }

    #[test]
    fn unaligned_base_is_rejected() {
        let d = desc(RegionKind::Payload, SECTOR_BYTES as u32 + 512, 8);
        assert_eq!(
            d.validate(),
            Err(RegionError::Unaligned {
                base: SECTOR_BYTES as u32 + 512
            })
        );
    }

    #[test]
    fn extent_must_be_whole_sectors() {
        // 8 blocks of 512 B is 4 KiB, exactly one sector; 7 is not.
        assert!(desc(RegionKind::Payload, 0, 8).validate().is_ok());
        assert_eq!(
            desc(RegionKind::Payload, 0, 7).validate(),
            Err(RegionError::ExtentNotSectorMultiple { bytes: 3584 })
        );
    }

    #[test]
    fn block_size_must_divide_the_sector() {
        let d = RegionDesc {
            block_bytes: 768,
            ..desc(RegionKind::Payload, 0, 8)
        };
        assert_eq!(
            d.validate(),
            Err(RegionError::BlockSize { block_bytes: 768 })
        );
    }

    #[test]
    fn unknown_discriminant_is_refused_not_guessed() {
        let mut raw = desc(RegionKind::Payload, 0, 8).encode();
        raw[0] = 99;
        assert_eq!(
            RegionDesc::decode(&raw),
            Err(RegionError::UnknownDiscriminant)
        );
    }

    #[test]
    fn overlap_is_caught_at_table_level() {
        let sector = SECTOR_BYTES as u32;
        let mut regions = [desc(RegionKind::Codebook, 0, 8); REGION_COUNT];
        regions[1] = desc(RegionKind::CodebookReplica, sector, 8);
        regions[2] = desc(RegionKind::Payload, 2 * sector, 8);
        regions[3] = desc(RegionKind::PayloadCrc, 3 * sector, 8);
        regions[4] = desc(RegionKind::Rerank, 4 * sector, 8);
        regions[5] = desc(RegionKind::RerankCrc, 5 * sector, 8);
        let table = RegionTable { regions };
        assert_eq!(table.validate(), Ok(()));

        // Grow the codebook into its replica's sector.
        let mut clashing = table;
        clashing.regions[0].blocks = 16;
        assert_eq!(
            clashing.validate(),
            Err(RegionError::Overlap {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn block_offset_is_bounded() {
        let d = desc(RegionKind::Payload, SECTOR_BYTES as u32, 8);
        assert_eq!(d.block_offset(0), Some(SECTOR_BYTES as u32));
        assert_eq!(d.block_offset(7), Some(SECTOR_BYTES as u32 + 7 * 512));
        assert_eq!(d.block_offset(8), None);
    }

    #[test]
    fn table_round_trips() {
        let sector = SECTOR_BYTES as u32;
        let mut regions = [desc(RegionKind::Codebook, 0, 8); REGION_COUNT];
        regions[1] = desc(RegionKind::CodebookReplica, sector, 8);
        regions[2] = desc(RegionKind::Payload, 2 * sector, 8);
        regions[3] = desc(RegionKind::PayloadCrc, 3 * sector, 8);
        regions[4] = desc(RegionKind::Rerank, 4 * sector, 8);
        regions[5] = desc(RegionKind::RerankCrc, 5 * sector, 8);
        let table = RegionTable { regions };

        let mut buf = [0u8; RegionTable::ENCODED_BYTES];
        assert_eq!(table.encode(&mut buf), Some(RegionTable::ENCODED_BYTES));
        assert_eq!(RegionTable::decode(&buf), Ok(table));
    }
}
