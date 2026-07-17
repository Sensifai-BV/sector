//! `NorFlash` and `Xip` backends over the ESP32 flash controller.
//!
//! # Status
//!
//! The types and their contracts are here; the register access is not. Bringing
//! this up means adding `esp-hal` as a dependency and filling in the three
//! `NorFlash` methods against `esp_storage::FlashStorage`.
//!
//! The contracts are written down now because the simulator enforces them
//! already, and a backend that quietly violates one — a second program to a page
//! without an intervening erase, an unaligned erase — produces corruption that
//! looks like a codec bug.

use sector_hal::{NorFlash, Xip};

/// Why a flash operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashError {
    /// The controller rejected the operation.
    Controller,
    /// The access fell outside the volume partition.
    OutOfPartition {
        /// Byte offset within the partition.
        offset: u32,
    },
    /// A program was not page-aligned, or an erase not sector-aligned.
    Misaligned {
        /// Byte offset within the partition.
        offset: u32,
    },
    /// The backend is a stub and has no hardware behind it.
    ///
    /// Every method returns this until bring-up. Returning an error is
    /// deliberate: a stub that silently succeeded would let the mount path
    /// "work" against zeroed flash and report a volume that does not exist.
    NotImplemented,
}

/// Flash backend over the `sector` partition.
///
/// Offsets are partition-relative. The partition base is added here and appears
/// nowhere above this type, so no absolute flash address reaches the engine.
pub struct Esp32Flash {
    /// Partition base, from `partition::VOLUME_OFFSET`.
    base: u32,
    /// Partition extent.
    size: u32,
}

impl Esp32Flash {
    /// A backend over the volume partition.
    pub const fn new() -> Self {
        Self {
            base: crate::partition::VOLUME_OFFSET,
            size: crate::partition::VOLUME_SIZE,
        }
    }

    /// Absolute flash address of a partition-relative offset.
    const fn absolute(&self, offset: u32, len: u32) -> Result<u32, FlashError> {
        match offset.checked_add(len) {
            Some(end) if end <= self.size => Ok(self.base + offset),
            _ => Err(FlashError::OutOfPartition { offset }),
        }
    }
}

impl Default for Esp32Flash {
    fn default() -> Self {
        Self::new()
    }
}

impl NorFlash for Esp32Flash {
    type Error = FlashError;

    fn page_size(&self) -> usize {
        crate::partition::PAGE_BYTES as usize
    }

    fn sector_size(&self) -> usize {
        crate::partition::SECTOR_BYTES as usize
    }

    fn capacity(&self) -> u32 {
        self.size
    }

    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), FlashError> {
        let _abs = self.absolute(addr, buf.len() as u32)?;
        Err(FlashError::NotImplemented)
    }

    fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), FlashError> {
        let page = crate::partition::PAGE_BYTES;
        if !addr.is_multiple_of(page) || !(buf.len() as u32).is_multiple_of(page) {
            return Err(FlashError::Misaligned { offset: addr });
        }
        let _abs = self.absolute(addr, buf.len() as u32)?;
        Err(FlashError::NotImplemented)
    }

    fn erase(&mut self, sector_addr: u32) -> Result<(), FlashError> {
        let sector = crate::partition::SECTOR_BYTES;
        if !sector_addr.is_multiple_of(sector) {
            return Err(FlashError::Misaligned {
                offset: sector_addr,
            });
        }
        let _abs = self.absolute(sector_addr, sector)?;
        Err(FlashError::NotImplemented)
    }
}

impl Xip for Esp32Flash {
    /// Borrow from the memory-mapped window.
    ///
    /// Returns `None` until bring-up. The window must be *probed*, not inferred
    /// from a feature flag: whether a range is mapped is a runtime property of
    /// the partition layout, and guessing wrong gives either a silent 100x
    /// fallback or a borrow of unmapped memory.
    fn window(&self, _addr: u32, _len: usize) -> Option<&[u8]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accesses_past_the_partition_are_refused_before_the_stub() {
        // Bounds are checked first, so bring-up does not have to add them and
        // the error names the real problem rather than NotImplemented.
        let mut f = Esp32Flash::new();
        let mut buf = [0u8; 16];
        assert_eq!(
            f.read(crate::partition::VOLUME_SIZE, &mut buf),
            Err(FlashError::OutOfPartition {
                offset: crate::partition::VOLUME_SIZE
            })
        );
    }

    #[test]
    fn alignment_contracts_are_enforced_by_the_backend() {
        // The simulator enforces these already; a backend that does not would
        // produce corruption resembling a codec bug.
        let mut f = Esp32Flash::new();
        assert_eq!(
            f.program(128, &[0u8; 256]),
            Err(FlashError::Misaligned { offset: 128 })
        );
        assert_eq!(
            f.program(0, &[0u8; 100]),
            Err(FlashError::Misaligned { offset: 0 })
        );
        assert_eq!(f.erase(512), Err(FlashError::Misaligned { offset: 512 }));
    }

    #[test]
    fn a_well_formed_access_reaches_the_stub_and_says_so() {
        // The stub reports NotImplemented rather than succeeding: a silent
        // success would let mount "work" against zeroed flash.
        let mut f = Esp32Flash::new();
        let mut buf = [0u8; 256];
        assert_eq!(f.read(0, &mut buf), Err(FlashError::NotImplemented));
        assert_eq!(f.program(0, &[0u8; 256]), Err(FlashError::NotImplemented));
        assert_eq!(f.erase(0), Err(FlashError::NotImplemented));
    }

    #[test]
    fn the_window_is_absent_until_it_is_probed() {
        let f = Esp32Flash::new();
        assert!(f.window(0, 16).is_none());
    }

    #[test]
    fn geometry_matches_the_partition_table() {
        let f = Esp32Flash::new();
        assert_eq!(f.page_size(), 256);
        assert_eq!(f.sector_size(), 4096);
        assert_eq!(f.capacity(), crate::partition::VOLUME_SIZE);
    }
}
