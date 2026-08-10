//! Memory-mapped backend: opt-in, and what its `Xip` claim does not mean.
//!
//! Implements [`NorFlash`] and [`Xip`], so [`sector_core::mount::mount_xip`]
//! binds the borrowing path and the engine's `is_zero_copy()` reports true.
//!
//! # The claim this makes, and why it is weaker here than on NOR
//!
//! [`Xip`] documents that a mapped window makes a fetch a load instruction: no
//! translation layer, no block granularity, no bounce buffer. On the raw NOR of
//! an ESP32 or RP2040 that is literally true — the part is wired into the address
//! space.
//!
//! On a Raspberry Pi it is not. The volume lives on microSD, USB or NVMe behind a
//! flash translation layer, and `mmap` does not change that: the *first* touch of
//! each page is a major fault that reads a block through the FTL, at the same
//! cost the buffered backend pays with `pread`. What `mmap` buys is where the
//! cost lands (a fault rather than a syscall) and that the page cache absorbs
//! repeat touches. What it does not buy is byte-addressable storage.
//!
//! This backend therefore exists to *quantify* the page cache's contribution,
//! not to claim the storage is something it is not. Reporting a Pi as
//! execute-in-place would erase the project's measured inversion — the smaller
//! tier executing stage two faster because raw NOR services a per-candidate
//! random read as a load — by making the larger tier look like it has the same
//! access cost. Run both backends on one board and the gap is the number.
//!
//! `FileFlash` is the default for that reason, and this module is behind the
//! `mmap` feature.
//!
//! # Fault accounting
//!
//! [`MappedFlash::fault_stats`] reports pages first-touched, at the granularity
//! [`crate::platform::page_size`] returns rather than an assumed 4096 — Pi OS on
//! Pi 5 uses 16 KiB pages. A first touch is counted by the backend when it
//! borrows a range it has not borrowed before, which is a lower bound on major
//! faults: the kernel may have prefetched, and `MAP_POPULATE` is deliberately not
//! used so the fault pattern reflects the access pattern.

use std::fs::File;
use std::path::Path;

use sector_hal::{NorFlash, Xip};

use crate::platform::page_size;
use crate::Error;

/// Page-touch accounting for a mapped volume.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaultStats {
    /// Distinct pages this backend has borrowed at least once.
    ///
    /// A lower bound on major faults: the kernel may prefetch, and pages may be
    /// evicted and re-faulted without the backend seeing it.
    pub pages_touched: u64,
    /// Window borrows served.
    pub borrows: u64,
    /// Borrows that fell outside the map and forced the buffered path.
    pub misses: u64,
    /// Page size the accounting is against.
    pub page_bytes: usize,
}

/// A SECTOR volume mapped into the address space.
///
/// The mapping is read-only and private, and lives as long as this value.
#[derive(Debug)]
pub struct MappedFlash {
    /// The mapped base address. Never null while this value exists.
    ptr: *const u8,
    len: usize,
    /// Kept so the mapping's backing file is not closed under it. `mmap` keeps
    /// its own reference, so this is belt-and-braces for clarity rather than
    /// necessity.
    _file: File,
    page_bytes: usize,
    /// One bit per page, tracking first touch.
    touched: Vec<u64>,
    stats: FaultStats,
}

// SAFETY: the mapping is read-only (`PROT_READ`, `MAP_PRIVATE`) and the raw
// pointer is only ever read through, never written and never freed except in
// `Drop`. Sharing a read-only mapping across threads is sound, which is what the
// daemon needs to serve concurrent queries from one volume.
unsafe impl Send for MappedFlash {}
unsafe impl Sync for MappedFlash {}

impl MappedFlash {
    /// Map `path` read-only.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        let needed = 2 * sector_format::manifest::MANIFEST_BYTES as u64;
        if len < needed {
            return Err(Error::TooSmall { len, needed });
        }
        if len > u32::MAX as u64 {
            return Err(Error::TooLarge { len });
        }

        let page_bytes = page_size();
        let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
        let ptr = map_read_only(fd, len as usize)?;

        let pages = (len as usize).div_ceil(page_bytes);
        Ok(Self {
            ptr,
            len: len as usize,
            _file: file,
            page_bytes,
            touched: vec![0u64; pages.div_ceil(64)],
            stats: FaultStats {
                page_bytes,
                ..FaultStats::default()
            },
        })
    }

    /// Volume length in bytes.
    pub const fn len(&self) -> u32 {
        self.len as u32
    }

    /// Whether the volume is empty. Never true; [`Self::open`] rejects a volume
    /// shorter than the header.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Page-touch counters.
    pub const fn fault_stats(&self) -> FaultStats {
        self.stats
    }

    /// Clear the counters and the first-touch record, so a measurement can
    /// exclude mount cost.
    ///
    /// This does not evict anything: the pages stay in the page cache, so a
    /// subsequent pass counts them as touched again but will not fault. That is
    /// the point — the difference between the first pass and the second is the
    /// cache's contribution.
    pub fn reset_stats(&mut self) {
        self.stats = FaultStats {
            page_bytes: self.page_bytes,
            ..FaultStats::default()
        };
        self.touched.fill(0);
    }

    /// Record a first touch of every page in `addr..addr + len`.
    fn mark_pages(&mut self, addr: usize, len: usize) {
        if len == 0 {
            return;
        }
        let first = addr / self.page_bytes;
        let last = (addr + len - 1) / self.page_bytes;
        for page in first..=last {
            let (word, bit) = (page / 64, page % 64);
            if let Some(slot) = self.touched.get_mut(word) {
                let mask = 1u64 << bit;
                if *slot & mask == 0 {
                    *slot |= mask;
                    self.stats.pages_touched += 1;
                }
            }
        }
    }

    /// Borrow `len` bytes at `addr` from the mapping, without accounting.
    fn slice(&self, addr: usize, len: usize) -> Option<&[u8]> {
        if addr.checked_add(len)? > self.len {
            return None;
        }
        // SAFETY: `addr + len <= self.len` was just checked, the mapping covers
        // `self.len` bytes from `self.ptr`, and it is alive for `&self`'s
        // lifetime — `Drop` is the only place it is unmapped. The mapping is
        // read-only and private, so no other process's write can invalidate the
        // bytes under the borrow.
        Some(unsafe { std::slice::from_raw_parts(self.ptr.add(addr), len) })
    }
}

impl Drop for MappedFlash {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` are exactly what `mmap` returned and this runs once,
        // at the end of the value's life, after which no borrow can exist.
        unsafe {
            munmap(self.ptr as *mut core::ffi::c_void, self.len);
        }
    }
}

impl NorFlash for MappedFlash {
    type Error = Error;

    fn page_size(&self) -> usize {
        sector_format::BLOCK_BYTES
    }

    fn sector_size(&self) -> usize {
        sector_format::SECTOR_BYTES
    }

    fn capacity(&self) -> u32 {
        self.len as u32
    }

    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), Self::Error> {
        let at = addr as usize;
        let Some(src) = self.slice(at, buf.len()) else {
            return Err(Error::OutOfBounds {
                addr,
                len: buf.len(),
            });
        };
        // The copy is a `memcpy` from the page cache rather than a syscall, which
        // is this backend's actual advantage on the buffered path.
        buf.copy_from_slice(src);
        self.mark_pages(at, buf.len());
        Ok(())
    }

    fn program(&mut self, _addr: u32, _buf: &[u8]) -> Result<(), Self::Error> {
        Err(Error::ReadOnly)
    }

    fn erase(&mut self, _sector_addr: u32) -> Result<(), Self::Error> {
        Err(Error::ReadOnly)
    }
}

impl Xip for MappedFlash {
    fn window(&self, addr: u32, len: usize) -> Option<&[u8]> {
        // `Xip::window` takes `&self`, so the counters cannot be updated from
        // here without interior mutability. Borrows are counted in
        // `window_counted`, which the adapters use; this path stays free of
        // accounting so the trait's cost claim is not distorted by measuring it.
        self.slice(addr as usize, len)
    }
}

impl MappedFlash {
    /// [`Xip::window`] with accounting.
    ///
    /// Separate from the trait method because the trait takes `&self` and the
    /// counters need `&mut`. A caller measuring fault behaviour uses this; the
    /// engine uses the trait and pays nothing.
    pub fn window_counted(&mut self, addr: u32, len: usize) -> Option<&[u8]> {
        let at = addr as usize;
        if at.checked_add(len).is_none_or(|end| end > self.len) {
            self.stats.borrows += 1;
            self.stats.misses += 1;
            return None;
        }
        self.stats.borrows += 1;
        self.mark_pages(at, len);
        self.slice(at, len)
    }
}

/// `mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0)`.
///
/// Declared locally rather than taken as a `libc` dependency: this and
/// `munmap` are the only two calls needed, and the workspace holds no external
/// dependencies. The constants are identical across Linux and the BSDs for these
/// three flags.
fn map_read_only(fd: i32, len: usize) -> Result<*const u8, Error> {
    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;

    unsafe extern "C" {
        fn mmap(
            addr: *mut core::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut core::ffi::c_void;
    }

    // SAFETY: a null hint lets the kernel choose the address, `len` is the file's
    // measured length, and the flags are a read-only private mapping. The return
    // value is checked against MAP_FAILED before use.
    let ptr = unsafe { mmap(core::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
    if ptr as isize == -1 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    Ok(ptr as *const u8)
}

unsafe extern "C" {
    fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::test_support::write_temp_volume;
    use crate::FileFlash;

    #[test]
    fn mapped_reads_match_the_file_byte_for_byte() {
        // The property that makes the two backends interchangeable: same bytes,
        // different cost. If this fails, no comparison between them is valid.
        let (_dir, path, image) = write_temp_volume("mapped_agree");
        let mut mapped = MappedFlash::open(&path).expect("map");
        let mut buffered = FileFlash::open(&path).expect("open");

        for at in [0u32, 512, 4096, 8192] {
            let mut a = [0u8; 512];
            let mut b = [0u8; 512];
            mapped.read(at, &mut a).expect("mapped read");
            buffered.read(at, &mut b).expect("buffered read");
            assert_eq!(a, b, "backends disagree at {at}");
            assert_eq!(&a[..], &image[at as usize..at as usize + 512]);
        }
    }

    #[test]
    fn the_window_borrows_rather_than_copying() {
        let (_dir, path, image) = write_temp_volume("mapped_window");
        let m = MappedFlash::open(&path).expect("map");
        let w = m.window(1024, 256).expect("window");
        assert_eq!(w, &image[1024..1280]);
        // Out of range yields None so the engine binds the buffered path rather
        // than borrowing unmapped memory.
        assert!(m.window(image.len() as u32 - 8, 64).is_none());
    }

    #[test]
    fn page_accounting_counts_each_page_once() {
        let (_dir, path, _) = write_temp_volume("mapped_pages");
        let mut m = MappedFlash::open(&path).expect("map");
        let page = m.fault_stats().page_bytes;
        assert!(page >= 4096);

        m.window_counted(0, 64).expect("first borrow");
        assert_eq!(m.fault_stats().pages_touched, 1);
        // Same page again: no new touch.
        m.window_counted(128, 64).expect("second borrow");
        assert_eq!(m.fault_stats().pages_touched, 1);
        assert_eq!(m.fault_stats().borrows, 2);
        // A borrow spanning a page boundary touches two.
        m.window_counted(page as u32 - 32, 64).expect("straddle");
        assert_eq!(m.fault_stats().pages_touched, 2);
    }

    #[test]
    fn a_miss_is_counted_and_does_not_mark_pages() {
        let (_dir, path, image) = write_temp_volume("mapped_miss");
        let mut m = MappedFlash::open(&path).expect("map");
        assert!(m.window_counted(image.len() as u32, 512).is_none());
        assert_eq!(m.fault_stats().misses, 1);
        assert_eq!(m.fault_stats().pages_touched, 0);
    }

    #[test]
    fn writes_are_refused() {
        let (_dir, path, _) = write_temp_volume("mapped_readonly");
        let mut m = MappedFlash::open(&path).expect("map");
        assert!(matches!(m.program(0, &[0u8; 512]), Err(Error::ReadOnly)));
        assert!(matches!(m.erase(0), Err(Error::ReadOnly)));
    }

    #[test]
    fn reset_clears_the_first_touch_record() {
        let (_dir, path, _) = write_temp_volume("mapped_reset");
        let mut m = MappedFlash::open(&path).expect("map");
        m.window_counted(0, 64).expect("borrow");
        assert_eq!(m.fault_stats().pages_touched, 1);
        m.reset_stats();
        assert_eq!(m.fault_stats().pages_touched, 0);
        assert_eq!(m.fault_stats().borrows, 0);
        // Page size survives the reset: it is a property of the kernel, not a
        // counter.
        assert!(m.fault_stats().page_bytes >= 4096);
    }
}
