//! Raspberry Pi Pico (RP2040) port of SECTOR.
//!
//! Supplies a QSPI NOR backend and a cycle instrument; the engine crates build
//! unchanged for `thumbv6m-none-eabi`.
//!
//! # Why this part matters to the design
//!
//! Cortex-M0+ has a weak multiplier and **no hardware divide**. SECTOR's scan
//! is `m` table lookups and `m` adds with no multiplies by construction, and
//! `make asm-check` enforces that on this target as well as on RV32IMC — the
//! property was written for a part like this one and was, until it was checked
//! here, false on it.

#![no_std]

use rp2040_hal::rom_data;
use sector_hal::{Edge, Instrument, NorFlash, Phase};

/// NOR page size: the programming granularity.
pub const PAGE_BYTES: usize = 256;

/// NOR erase sector, and SECTOR's protection-group alignment.
pub const SECTOR_BYTES: usize = 4096;

/// Where the XIP window maps flash into the address space.
const XIP_BASE: u32 = 0x1000_0000;

/// Why a flash operation was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashError {
    /// Address or length outside the mapped region.
    OutOfBounds,
    /// A program or erase call that was not correctly aligned.
    Unaligned,
}

/// A SECTOR volume region in the Pico's QSPI flash.
///
/// Reads go through the XIP window, so a scan streams from flash at memory
/// speed with no driver call per block — the same execute-in-place path the
/// format's zero-copy design assumes. Writes go through the boot ROM's flash
/// routines, which is the only supported way to program this part while code is
/// executing from it.
pub struct PicoFlash {
    base: u32,
    len: u32,
}

impl PicoFlash {
    /// Map a volume at `[base, base + len)`, both offsets from the start of
    /// flash rather than absolute addresses.
    ///
    /// Returns `None` unless the region is sector-aligned at both ends. An
    /// unaligned region cannot bound its own failure probability, because
    /// erase — and therefore correlated failure — happens in whole sectors.
    pub fn new(base: u32, len: u32, capacity: u32) -> Option<Self> {
        if !(base as usize).is_multiple_of(SECTOR_BYTES)
            || !(len as usize).is_multiple_of(SECTOR_BYTES)
        {
            return None;
        }
        if base.checked_add(len)? > capacity {
            return None;
        }
        Some(Self { base, len })
    }

    /// Borrow the whole region through the XIP window.
    ///
    /// This is what makes the scan zero-copy on this part: the codes are read
    /// directly out of flash rather than copied into RAM first, which is the
    /// only way a corpus larger than 264 KiB is scannable at all here.
    pub fn xip(&self) -> &'static [u8] {
        // The XIP window is a fixed hardware mapping of flash into the address
        // space; the region was bounds-checked against capacity in `new`.
        unsafe {
            core::slice::from_raw_parts((XIP_BASE + self.base) as *const u8, self.len as usize)
        }
    }
}

impl NorFlash for PicoFlash {
    type Error = FlashError;

    fn page_size(&self) -> usize {
        PAGE_BYTES
    }

    fn sector_size(&self) -> usize {
        SECTOR_BYTES
    }

    fn capacity(&self) -> u32 {
        self.len
    }

    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), Self::Error> {
        let end = addr as u64 + buf.len() as u64;
        if end > self.len as u64 {
            return Err(FlashError::OutOfBounds);
        }
        let src = &self.xip()[addr as usize..addr as usize + buf.len()];
        buf.copy_from_slice(src);
        Ok(())
    }

    /// Program a page-aligned, page-sized run onto erased flash.
    ///
    /// Interrupts are masked for the duration: the boot ROM routines run with
    /// the XIP cache disabled, so any code fetched from flash while they are
    /// executing would fault.
    fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), Self::Error> {
        if !(addr as usize).is_multiple_of(PAGE_BYTES) || !buf.len().is_multiple_of(PAGE_BYTES) {
            return Err(FlashError::Unaligned);
        }
        if addr as u64 + buf.len() as u64 > self.len as u64 {
            return Err(FlashError::OutOfBounds);
        }
        let offset = self.base + addr;
        cortex_m::interrupt::free(|_| unsafe {
            rom_data::connect_internal_flash();
            rom_data::flash_exit_xip();
            rom_data::flash_range_program(offset, buf.as_ptr(), buf.len());
            rom_data::flash_flush_cache();
            rom_data::flash_enter_cmd_xip();
        });
        Ok(())
    }

    fn erase(&mut self, sector_addr: u32) -> Result<(), Self::Error> {
        if !(sector_addr as usize).is_multiple_of(SECTOR_BYTES) {
            return Err(FlashError::Unaligned);
        }
        if sector_addr as u64 + SECTOR_BYTES as u64 > self.len as u64 {
            return Err(FlashError::OutOfBounds);
        }
        let offset = self.base + sector_addr;
        cortex_m::interrupt::free(|_| unsafe {
            rom_data::connect_internal_flash();
            rom_data::flash_exit_xip();
            // 0x20 is the sector-erase opcode this family expects.
            rom_data::flash_range_erase(offset, SECTOR_BYTES, SECTOR_BYTES as u32, 0x20);
            rom_data::flash_flush_cache();
            rom_data::flash_enter_cmd_xip();
        });
        Ok(())
    }
}

/// Phase instrument backed by the RP2040's microsecond timer.
///
/// The timer runs at a fixed 1 MHz regardless of the system clock, so a
/// measurement stays comparable across clock configurations — a cycle counter
/// would not, and comparing runs at different clocks is the mistake it exists
/// to prevent.
pub struct TimerInstrument {
    marks: u32,
}

impl Default for TimerInstrument {
    fn default() -> Self {
        Self::new()
    }
}

impl TimerInstrument {
    pub const fn new() -> Self {
        Self { marks: 0 }
    }

    /// Phase boundaries seen, for checking that a run marked every phase it
    /// claims to have measured.
    pub const fn marks(&self) -> u32 {
        self.marks
    }
}

impl Instrument for TimerInstrument {
    fn cycles(&self) -> u64 {
        // TIMERAWL is a free-running 1 MHz counter; reading the low word alone
        // is enough for phases far shorter than its 71-minute wrap.
        const TIMERAWL: *const u32 = 0x4005_4028 as *const u32;
        unsafe { core::ptr::read_volatile(TIMERAWL) as u64 }
    }

    fn mark(&mut self, phase: Phase, edge: Edge) {
        self.marks = self.marks.saturating_add(1);
        let _ = (phase, edge);
    }
}
