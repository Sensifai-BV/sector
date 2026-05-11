//! Deterministic in-RAM NOR with an execute-in-place window.
//!
//! Implements the backend contract the firmware sees — program-once-per-erase,
//! `0xFF` erased state, sector granularity — plus a mapped window, so the
//! engine's zero-copy path is exercised in CI rather than only on hardware.
//!
//! # Fidelity rules
//!
//! Programming a non-erased page is an error, never a silent AND of old and new
//! contents. A simulator more permissive than the hardware certifies code that
//! will fail on the device.
//!
//! Every fault schedule is seeded and prints its seed on failure. These
//! failures are rare by construction, and a non-reproducible one cannot be
//! debugged.
//!
//! Window and non-window paths are separate configurations so both branches of
//! the mount-time binding get coverage. The buffered fallback is the path the
//! larger tiers use and the one nothing exercises by accident.

use sector_hal::{NorFlash, Xip, ERASED_BYTE};

/// A contract the hardware enforces and the simulator must not relax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Violation {
    /// A page was programmed twice without an intervening erase.
    ///
    /// Real NOR ANDs the two writes, silently producing bytes matching neither.
    /// A simulator that permitted this would certify code that fails on the
    /// device, so it is an error here.
    DoubleProgram {
        /// Byte address of the offending page.
        addr: u32,
    },
    /// A program was not page-aligned or not page-sized.
    Misaligned {
        /// Byte address.
        addr: u32,
        /// Length in bytes.
        len: usize,
    },
    /// An erase was not sector-aligned.
    UnalignedErase {
        /// Byte address.
        addr: u32,
    },
    /// An access fell outside the device.
    OutOfBounds {
        /// Byte address.
        addr: u32,
        /// Length in bytes.
        len: usize,
    },
    /// A sector exceeded its endurance budget.
    Worn {
        /// Sector index.
        sector: usize,
        /// Erase cycles spent.
        cycles: u32,
    },
}

/// Simulator configuration.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Device capacity in bytes.
    pub capacity: usize,
    /// Program granularity.
    pub page_bytes: usize,
    /// Erase granularity.
    pub sector_bytes: usize,
    /// Erase cycles a sector tolerates before [`Violation::Worn`].
    pub endurance: u32,
    /// Extent of the memory-mapped window, from address 0. `None` for a
    /// backend with no window, which is the managed-NAND case.
    pub window: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            capacity: 4 * 1024 * 1024,
            page_bytes: 256,
            sector_bytes: 4096,
            endurance: 100_000,
            window: Some(4 * 1024 * 1024),
        }
    }
}

/// Deterministic in-RAM NOR flash enforcing program-once-per-erase.
///
/// Strictness is the point. A simulator more permissive than the hardware
/// certifies code that will fail on the device, so every contract the HAL
/// documents is checked here and reported rather than tolerated.
pub struct SimFlash {
    bytes: Vec<u8>,
    /// Whether each page has been programmed since its last erase.
    programmed: Vec<bool>,
    /// Erase cycles per sector, for the lifetime model.
    erases: Vec<u32>,
    config: Config,
    violations: Vec<Violation>,
    /// Reads served from the mapped window.
    pub borrows: u32,
    /// Reads served through a copy.
    pub copies: u32,
}

impl SimFlash {
    /// A device of `config`, fully erased.
    pub fn new(config: Config) -> Self {
        let pages = config.capacity.div_ceil(config.page_bytes);
        let sectors = config.capacity.div_ceil(config.sector_bytes);
        Self {
            bytes: vec![ERASED_BYTE; config.capacity],
            programmed: vec![false; pages],
            erases: vec![0; sectors],
            config,
            violations: Vec::new(),
            borrows: 0,
            copies: 0,
        }
    }

    /// Contract violations recorded so far.
    ///
    /// Accumulated rather than panicking, so a test can assert that a
    /// deliberately illegal sequence was caught.
    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }

    /// Erase cycles spent on `sector`.
    pub fn erase_count(&self, sector: usize) -> u32 {
        self.erases.get(sector).copied().unwrap_or(0)
    }

    /// Total erase cycles across the device — the lifetime model's quantity.
    pub fn total_erases(&self) -> u32 {
        self.erases.iter().copied().sum()
    }

    /// Raw bytes, for fault injection and inspection.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Raw bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Write `data` at `addr` ignoring program-once semantics.
    ///
    /// For installing a host-built image, which a device would receive already
    /// written. Not a program: it bypasses the contract deliberately and says
    /// so at the call site.
    pub fn install(&mut self, addr: u32, data: &[u8]) {
        let start = addr as usize;
        if let Some(dst) = self.bytes.get_mut(start..start + data.len()) {
            dst.copy_from_slice(data);
        }
        for page in (start / self.config.page_bytes)
            ..=((start + data.len()).saturating_sub(1) / self.config.page_bytes)
        {
            if let Some(p) = self.programmed.get_mut(page) {
                *p = true;
            }
        }
    }

    fn record(&mut self, v: Violation) {
        self.violations.push(v);
    }
}

/// The simulator's error type. Distinguishes a contract violation from a
/// bounds failure so a test can tell which one it triggered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimError(pub Violation);

impl NorFlash for SimFlash {
    type Error = SimError;

    fn page_size(&self) -> usize {
        self.config.page_bytes
    }
    fn sector_size(&self) -> usize {
        self.config.sector_bytes
    }
    fn capacity(&self) -> u32 {
        self.config.capacity as u32
    }

    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), SimError> {
        let start = addr as usize;
        let src = match self.bytes.get(start..start + buf.len()) {
            Some(s) => s,
            None => {
                let v = Violation::OutOfBounds {
                    addr,
                    len: buf.len(),
                };
                self.record(v);
                return Err(SimError(v));
            }
        };
        buf.copy_from_slice(src);
        self.copies += 1;
        Ok(())
    }

    fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), SimError> {
        let page = self.config.page_bytes;
        if !(addr as usize).is_multiple_of(page) || !buf.len().is_multiple_of(page) {
            let v = Violation::Misaligned {
                addr,
                len: buf.len(),
            };
            self.record(v);
            return Err(SimError(v));
        }
        let start = addr as usize;
        if start + buf.len() > self.config.capacity {
            let v = Violation::OutOfBounds {
                addr,
                len: buf.len(),
            };
            self.record(v);
            return Err(SimError(v));
        }

        // Program-once-per-erase, checked before any byte changes.
        for i in 0..(buf.len() / page) {
            let idx = start / page + i;
            if self.programmed.get(idx).copied().unwrap_or(false) {
                let v = Violation::DoubleProgram {
                    addr: (idx * page) as u32,
                };
                self.record(v);
                return Err(SimError(v));
            }
        }

        for i in 0..(buf.len() / page) {
            if let Some(p) = self.programmed.get_mut(start / page + i) {
                *p = true;
            }
        }
        // NOR programming clears bits; it never sets them.
        if let Some(dst) = self.bytes.get_mut(start..start + buf.len()) {
            for (d, s) in dst.iter_mut().zip(buf.iter()) {
                *d &= *s;
            }
        }
        Ok(())
    }

    fn erase(&mut self, sector_addr: u32) -> Result<(), SimError> {
        let sec = self.config.sector_bytes;
        if !(sector_addr as usize).is_multiple_of(sec) {
            let v = Violation::UnalignedErase { addr: sector_addr };
            self.record(v);
            return Err(SimError(v));
        }
        let start = sector_addr as usize;
        if start + sec > self.config.capacity {
            let v = Violation::OutOfBounds {
                addr: sector_addr,
                len: sec,
            };
            self.record(v);
            return Err(SimError(v));
        }

        let index = start / sec;
        let cycles = self.erases.get(index).copied().unwrap_or(0) + 1;
        if let Some(e) = self.erases.get_mut(index) {
            *e = cycles;
        }
        if cycles > self.config.endurance {
            let v = Violation::Worn {
                sector: index,
                cycles,
            };
            self.record(v);
            return Err(SimError(v));
        }

        if let Some(dst) = self.bytes.get_mut(start..start + sec) {
            dst.fill(ERASED_BYTE);
        }
        for i in 0..(sec / self.config.page_bytes) {
            if let Some(p) = self.programmed.get_mut(start / self.config.page_bytes + i) {
                *p = false;
            }
        }
        Ok(())
    }
}

impl Xip for SimFlash {
    fn window(&self, addr: u32, len: usize) -> Option<&[u8]> {
        let extent = self.config.window?;
        let start = addr as usize;
        if start + len > extent {
            return None;
        }
        self.bytes.get(start..start + len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Config {
        Config {
            capacity: 16 * 1024,
            page_bytes: 256,
            sector_bytes: 4096,
            endurance: 4,
            window: Some(8 * 1024),
        }
    }

    #[test]
    fn a_fresh_device_reads_as_erased() {
        let mut f = SimFlash::new(small());
        let mut buf = [0u8; 256];
        f.read(0, &mut buf).unwrap();
        assert!(buf.iter().all(|b| *b == ERASED_BYTE));
    }

    #[test]
    fn programming_a_page_twice_is_an_error_not_a_silent_and() {
        // Real NOR ANDs the two writes, producing bytes matching neither. A
        // permissive simulator would certify code that fails on the device.
        let mut f = SimFlash::new(small());
        f.program(0, &[0xF0; 256]).unwrap();
        let err = f.program(0, &[0x0F; 256]);
        assert_eq!(err, Err(SimError(Violation::DoubleProgram { addr: 0 })));
        assert_eq!(f.violations().len(), 1);

        // The bytes are untouched by the refused write.
        let mut buf = [0u8; 256];
        f.read(0, &mut buf).unwrap();
        assert!(buf.iter().all(|b| *b == 0xF0));
    }

    #[test]
    fn an_erase_permits_programming_again() {
        let mut f = SimFlash::new(small());
        f.program(0, &[0xAA; 256]).unwrap();
        f.erase(0).unwrap();
        f.program(0, &[0x55; 256]).unwrap();
        assert!(f.violations().is_empty());
        let mut buf = [0u8; 256];
        f.read(0, &mut buf).unwrap();
        assert!(buf.iter().all(|b| *b == 0x55));
    }

    #[test]
    fn programming_only_clears_bits() {
        // The physical asymmetry every append path depends on.
        let mut f = SimFlash::new(small());
        f.program(0, &[0b1010_1010; 256]).unwrap();
        f.erase(0).unwrap();
        f.program(0, &[0b1111_0000; 256]).unwrap();
        let mut buf = [0u8; 256];
        f.read(0, &mut buf).unwrap();
        assert_eq!(buf[0], 0b1111_0000);
    }

    #[test]
    fn misaligned_programs_and_erases_are_refused() {
        let mut f = SimFlash::new(small());
        assert!(matches!(
            f.program(128, &[0u8; 256]),
            Err(SimError(Violation::Misaligned { .. }))
        ));
        assert!(matches!(
            f.program(0, &[0u8; 100]),
            Err(SimError(Violation::Misaligned { .. }))
        ));
        assert!(matches!(
            f.erase(512),
            Err(SimError(Violation::UnalignedErase { .. }))
        ));
    }

    #[test]
    fn erase_cycles_are_counted_for_the_lifetime_model() {
        let mut f = SimFlash::new(small());
        for _ in 0..3 {
            f.erase(0).unwrap();
        }
        assert_eq!(f.erase_count(0), 3);
        assert_eq!(f.erase_count(1), 0);
        assert_eq!(f.total_erases(), 3);
    }

    #[test]
    fn exceeding_endurance_is_reported_not_ignored() {
        // The lifetime claim is falsifiable only if the simulator refuses to
        // pretend a worn sector still works.
        let mut f = SimFlash::new(small());
        for _ in 0..4 {
            f.erase(0).unwrap();
        }
        assert!(matches!(
            f.erase(0),
            Err(SimError(Violation::Worn {
                sector: 0,
                cycles: 5
            }))
        ));
    }

    #[test]
    fn the_window_covers_only_its_configured_extent() {
        let f = SimFlash::new(small());
        // Inside the 8 KiB window.
        assert!(f.window(0, 4096).is_some());
        assert!(f.window(4096, 4096).is_some());
        // Past it, even though the device is 16 KiB.
        assert!(f.window(8192, 16).is_none());
        assert!(f.window(4096, 8192).is_none());
    }

    #[test]
    fn a_backend_without_a_window_borrows_nothing() {
        // The managed-NAND configuration. Both paths must be exercised, since
        // the buffered one is what the larger tiers use.
        let cfg = Config {
            window: None,
            ..small()
        };
        let f = SimFlash::new(cfg);
        assert!(f.window(0, 16).is_none());
    }

    #[test]
    fn reads_are_counted_as_copies() {
        let mut f = SimFlash::new(small());
        let mut buf = [0u8; 256];
        f.read(0, &mut buf).unwrap();
        f.read(256, &mut buf).unwrap();
        assert_eq!(f.copies, 2);
        // A window borrow is not a copy.
        f.window(0, 256).unwrap();
        assert_eq!(f.copies, 2);
    }

    #[test]
    fn out_of_bounds_access_is_refused() {
        let mut f = SimFlash::new(small());
        let mut buf = [0u8; 256];
        assert!(matches!(
            f.read(16 * 1024, &mut buf),
            Err(SimError(Violation::OutOfBounds { .. }))
        ));
    }

    #[test]
    fn install_bypasses_program_once_and_marks_pages() {
        // Installing a host-built image is not a program sequence; the device
        // receives it already written.
        let mut f = SimFlash::new(small());
        f.install(0, &[0x42; 512]);
        let mut buf = [0u8; 512];
        f.read(0, &mut buf).unwrap();
        assert!(buf.iter().all(|b| *b == 0x42));
        // The pages are marked, so a later program is still refused.
        assert!(matches!(
            f.program(0, &[0u8; 256]),
            Err(SimError(Violation::DoubleProgram { .. }))
        ));
    }
}
