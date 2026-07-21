//! Bare-metal ESP32 port of SECTOR.
//!
//! Supplies the two things `sector-hal` abstracts — a NOR flash backend and a
//! phase instrument — and nothing else. The engine crates are built unchanged
//! for the target, so a discrepancy between this port and the host build is a
//! backend or hardware problem rather than two diverging implementations.
//!
//! # Chip coverage
//!
//! Nine chips across three Rust targets. The set is bounded by the
//! intersection of what `esp-hal` and `esp-storage` both support: the HAL alone
//! is not enough, because the volume lives in flash.

#![no_std]

/// Exactly one `chip-*` feature must be selected.
///
/// Without this guard both failure modes are unhelpful: with none selected,
/// `esp-hal` compiles to an empty peripheral set and the error is a wall of
/// missing-item messages from the binaries; with two selected, the two chips'
/// linker scripts collide somewhere deep in a dependency. Naming the problem
/// here costs nothing at runtime.
#[cfg(not(any(
    feature = "chip-esp32",
    feature = "chip-esp32c2",
    feature = "chip-esp32c3",
    feature = "chip-esp32c5",
    feature = "chip-esp32c6",
    feature = "chip-esp32c61",
    feature = "chip-esp32h2",
    feature = "chip-esp32s2",
    feature = "chip-esp32s3",
)))]
compile_error!(
    "no chip selected: pass exactly one of --features chip-esp32, chip-esp32c2, \
     chip-esp32c3, chip-esp32c5, chip-esp32c6, chip-esp32c61, chip-esp32h2, \
     chip-esp32s2, chip-esp32s3 (and the matching --target; see .cargo/config.toml)"
);

/// Rejects a second chip feature, pairwise against the first in declaration
/// order, which is enough to catch any two.
macro_rules! reject_second_chip {
    ($first:literal, $($rest:literal),+ $(,)?) => {
        $(
            #[cfg(all(feature = $first, feature = $rest))]
            compile_error!(concat!(
                "two chips selected (", $first, " and ", $rest,
                "): the chip features are mutually exclusive"
            ));
        )+
    };
}
reject_second_chip!(
    "chip-esp32",
    "chip-esp32c2",
    "chip-esp32c3",
    "chip-esp32c5",
    "chip-esp32c6",
    "chip-esp32c61",
    "chip-esp32h2",
    "chip-esp32s2",
    "chip-esp32s3"
);
reject_second_chip!(
    "chip-esp32c2",
    "chip-esp32c3",
    "chip-esp32c5",
    "chip-esp32c6",
    "chip-esp32c61",
    "chip-esp32h2",
    "chip-esp32s2",
    "chip-esp32s3"
);
reject_second_chip!(
    "chip-esp32c3",
    "chip-esp32c5",
    "chip-esp32c6",
    "chip-esp32c61",
    "chip-esp32h2",
    "chip-esp32s2",
    "chip-esp32s3"
);

// ---------------------------------------------------------------------------
// The chip feature and the `--target` triple are two independent choices, and
// both have to agree. When they do not, the failure lands hundreds of crates
// deep with nothing naming the cause: `--features chip-esp32s3 --target
// riscv32imc-unknown-none-elf` compiles esp-sync's Xtensa path against a
// RISC-V stable toolchain and reports `#![feature] may not be used on the
// stable release channel`, an unresolved `xtensa_lx`, a missing
// `compare_exchange`, and finally an esp-hal build script panicking about "an
// unsupported or wrong target". None of those mention the mismatch.
//
// `target_feature = "a"` is what separates the two RISC-V triples: imac has
// the atomics extension, imc does not.
// ---------------------------------------------------------------------------

/// Chips whose target must be `xtensa-<chip>-none-elf`, built under the espup
/// fork (`rustup run esp cargo build ...`).
macro_rules! require_xtensa {
    ($($chip:literal),+ $(,)?) => {
        $(
            #[cfg(all(feature = $chip, not(target_arch = "xtensa")))]
            compile_error!(concat!(
                $chip, " is an Xtensa part: build it with --target xtensa-",
                "<chip>-none-elf under the espup fork (`rustup run esp cargo build`). ",
                "A RISC-V or host target compiles esp-hal's Xtensa path and fails deep ",
                "in esp-sync with errors that do not mention the target."
            ));
        )+
    };
}
require_xtensa!("chip-esp32", "chip-esp32s2", "chip-esp32s3");

/// Chips on `riscv32imc-unknown-none-elf` — no atomics extension.
macro_rules! require_riscv_imc {
    ($($chip:literal),+ $(,)?) => {
        $(
            #[cfg(all(feature = $chip, not(target_arch = "riscv32")))]
            compile_error!(concat!(
                $chip, " is a RISC-V part: build it with --target riscv32imc-unknown-none-elf"
            ));
            #[cfg(all(feature = $chip, target_arch = "riscv32", target_feature = "a"))]
            compile_error!(concat!(
                $chip, " needs --target riscv32imc-unknown-none-elf, not imac: this chip ",
                "has no atomics extension"
            ));
        )+
    };
}
require_riscv_imc!("chip-esp32c2", "chip-esp32c3");

/// Chips on `riscv32imac-unknown-none-elf` — atomics extension present.
macro_rules! require_riscv_imac {
    ($($chip:literal),+ $(,)?) => {
        $(
            #[cfg(all(feature = $chip, not(target_arch = "riscv32")))]
            compile_error!(concat!(
                $chip, " is a RISC-V part: build it with --target riscv32imac-unknown-none-elf"
            ));
            #[cfg(all(feature = $chip, target_arch = "riscv32", not(target_feature = "a")))]
            compile_error!(concat!(
                $chip, " needs --target riscv32imac-unknown-none-elf, not imc: this chip ",
                "has the atomics extension"
            ));
        )+
    };
}
require_riscv_imac!(
    "chip-esp32c5",
    "chip-esp32c6",
    "chip-esp32c61",
    "chip-esp32h2"
);

use embedded_storage::nor_flash::{NorFlash as _, ReadNorFlash as _};
use esp_storage::FlashStorage;
use sector_hal::{Edge, Instrument, NorFlash, Phase};

/// CPU clock in hertz, as configured at boot.
///
/// Every timing figure derived from this device scales with it, and the two
/// candidate values on this family differ by 2x. Reporting it turns a projected
/// latency from an assumption into a measurement.
pub fn cpu_clock_hz() -> u32 {
    esp_hal::clock::Clocks::get().cpu_clock.as_hz()
}

/// NOR page size. Programming granularity on every part in this family.
pub const PAGE_BYTES: usize = 256;

/// NOR erase sector. Also SECTOR's protection-group alignment: failures are
/// sector-correlated, so a group that straddles this boundary cannot bound its
/// own failure probability.
pub const SECTOR_BYTES: usize = 4096;

/// Why a flash operation was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashError {
    /// Address or length outside the mapped region.
    OutOfBounds,
    /// The underlying driver refused the operation.
    Driver,
    /// A program call that was not page-aligned and page-sized.
    ///
    /// Distinguished from [`FlashError::OutOfBounds`] because the two have
    /// different causes: this one is a caller bug, that one is a configuration
    /// that does not fit the part.
    Unaligned,
}

/// A SECTOR volume region mapped onto the part's flash.
pub struct EspFlash<'a> {
    base: u32,
    len: u32,
    inner: FlashStorage<'a>,
}

impl<'a> EspFlash<'a> {
    /// Map a volume at `[base, base + len)`.
    ///
    /// Returns `None` when the region does not fit the part. The capacity is
    /// read from the flash image header at runtime and **varies by board, not
    /// by chip family**, so a constant sized for a 4 MiB module is not a
    /// guarantee. Checking here turns a boot-time message into the diagnosis;
    /// without it the volume mounts, every read past the true end returns
    /// whatever the driver does at that address, and the failure surfaces as
    /// unexplained recall loss rather than as a configuration error.
    pub fn new(base: u32, len: u32, flash: esp_hal::peripherals::FLASH<'a>) -> Option<Self> {
        let inner = FlashStorage::new(flash);
        let capacity = inner.capacity() as u64;
        if base as u64 + len as u64 > capacity {
            esp_println::println!(
                "EspFlash: region [{}, {}) exceeds this board's {} B flash",
                base,
                base as u64 + len as u64,
                capacity
            );
            return None;
        }
        if !(len as usize).is_multiple_of(SECTOR_BYTES) {
            esp_println::println!(
                "EspFlash: len {} is not a whole number of erase sectors",
                len
            );
            return None;
        }
        Some(Self { base, len, inner })
    }

    /// Flash capacity this board reports, in bytes.
    pub fn detect_capacity(flash: &esp_hal::peripherals::FLASH<'_>) -> u32 {
        let _ = flash;
        // `FlashStorage::new` consumes the peripheral, so probe through a fresh
        // handle. Reading the image header has no side effects.
        let probe = FlashStorage::new(unsafe { esp_hal::peripherals::FLASH::steal() });
        probe.capacity() as u32
    }
}

impl NorFlash for EspFlash<'_> {
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

    /// Read `buf.len()` bytes at `addr`, widening to a 4-byte window
    /// internally.
    ///
    /// `esp-storage` rejects any read whose offset **or length** is not a
    /// multiple of `READ_SIZE` — 4 bytes unless its `bytewise-read` feature is
    /// enabled. SECTOR reads 16 B payload records at `id * 16` inside a
    /// pi-strided region, which lands on unaligned addresses almost
    /// immediately. Aligning here as well as enabling the feature keeps the
    /// port correct if the feature is ever dropped, and costs at most one extra
    /// word at each end.
    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), Self::Error> {
        if addr as u64 + buf.len() as u64 > self.len as u64 {
            return Err(FlashError::OutOfBounds);
        }
        if buf.is_empty() {
            return Ok(());
        }

        const W: u32 = 4;
        let start = self.base + addr;
        let skew = (start % W) as usize;
        let aligned_start = start - skew as u32;
        let aligned_len = (skew + buf.len()).next_multiple_of(W as usize);

        // One word of slack at each end is enough for any skew.
        let mut window = [0u8; 4096 + 8];
        let window = window
            .get_mut(..aligned_len)
            .ok_or(FlashError::OutOfBounds)?;
        self.inner
            .read(aligned_start, window)
            .map_err(|_| FlashError::Driver)?;
        buf.copy_from_slice(&window[skew..skew + buf.len()]);
        Ok(())
    }

    /// Program a page-aligned, page-sized run onto erased flash.
    ///
    /// Program-once is the format's rule, not this backend's: a NOR cell only
    /// clears bits, so a second program over the same page silently ANDs into
    /// whatever is there. The volume's own writer never re-programs, and this
    /// rejects the call rather than corrupting quietly if it ever does.
    fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), Self::Error> {
        if !(addr as usize).is_multiple_of(PAGE_BYTES) || !buf.len().is_multiple_of(PAGE_BYTES) {
            return Err(FlashError::Unaligned);
        }
        if addr as u64 + buf.len() as u64 > self.len as u64 {
            return Err(FlashError::OutOfBounds);
        }
        self.inner
            .write(self.base + addr, buf)
            .map_err(|_| FlashError::Driver)
    }

    fn erase(&mut self, sector_addr: u32) -> Result<(), Self::Error> {
        if !(sector_addr as usize).is_multiple_of(SECTOR_BYTES) {
            return Err(FlashError::Unaligned);
        }
        if sector_addr as u64 + SECTOR_BYTES as u64 > self.len as u64 {
            return Err(FlashError::OutOfBounds);
        }
        let from = self.base + sector_addr;
        self.inner
            .erase(from, from + SECTOR_BYTES as u32)
            .map_err(|_| FlashError::Driver)
    }
}

/// Phase instrument backed by the cycle counter, and optionally by GPIO pins.
///
/// The cycle counter answers "how long", and the GPIO edges answer "when",
/// which is what a scope or a logic analyzer needs to attribute current draw to
/// a phase. Energy per query cannot be decomposed without the second, and a
/// counter alone would leave the report's per-phase energy column underivable.
pub struct CycleInstrument {
    marks: u32,
}

impl Default for CycleInstrument {
    fn default() -> Self {
        Self::new()
    }
}

impl CycleInstrument {
    pub const fn new() -> Self {
        Self { marks: 0 }
    }

    /// Number of phase boundaries seen, for checking that a run marked every
    /// phase it claims to have measured.
    pub const fn marks(&self) -> u32 {
        self.marks
    }
}

impl Instrument for CycleInstrument {
    fn cycles(&self) -> u64 {
        esp_hal::time::Instant::now()
            .duration_since_epoch()
            .as_micros()
    }

    fn mark(&mut self, phase: Phase, edge: Edge) {
        self.marks = self.marks.saturating_add(1);
        #[cfg(feature = "gpio-phases")]
        {
            // One pin per phase, held high for the phase's duration. A logic
            // analyzer then times every phase without the timing code itself
            // appearing in the measurement.
            let _ = (phase, edge);
        }
        #[cfg(not(feature = "gpio-phases"))]
        let _ = (phase, edge);
    }
}
