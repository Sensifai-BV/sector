//! Positional-read backend: the honest default.
//!
//! Implements [`NorFlash`] and deliberately **not** [`Xip`], so
//! [`sector_core::mount::mount`] binds the buffered path and every stage-two
//! fetch is a real `read` syscall against the flash translation layer. That is
//! what a Raspberry Pi actually does, and it is the cost the project's
//! NOR-versus-managed-storage comparison is about.
//!
//! # Access accounting
//!
//! Reads are counted and bucketed by whether they crossed a device block
//! boundary, because the FTL's unit of work is a block and a 128 B record that
//! straddles two blocks costs two of them. The counters are the input to the
//! measurement campaign's per-query byte figures, and they count what was asked
//! of the kernel rather than what the engine wanted — a distinction that is the
//! whole point of measuring here rather than inferring from `R * D`.
//!
//! # `pread`, not seek-then-read
//!
//! [`FileExt::read_at`] does not disturb a shared file offset, so one open
//! volume can serve concurrent readers in the daemon without a lock and without
//! interleaved seeks returning each other's bytes.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use sector_hal::NorFlash;

use crate::Error;

/// Device block size assumed for access accounting.
///
/// 512 B is the logical sector every block device reports, and it matches
/// [`sector_format::BLOCK_BYTES`], so a straddle counted here is a straddle in
/// the format too. The physical erase block of an SD card is far larger — 4 MiB
/// is common — but that figure is unobservable from userspace and varies by
/// part, so accounting against it would be a guess dressed as a measurement.
pub const DEVICE_BLOCK_BYTES: usize = 512;

/// What a backend's reads cost, as asked of the kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AccessStats {
    /// `read` calls issued.
    pub reads: u64,
    /// Bytes requested.
    pub bytes: u64,
    /// Device blocks touched, counting a straddling read as more than one.
    ///
    /// This is the figure that tracks FTL work, and it is why the accounting
    /// exists: `bytes` alone understates the cost of a small random read by the
    /// ratio of the block size to the record size.
    pub blocks_touched: u64,
    /// Reads that crossed a device block boundary.
    pub straddling_reads: u64,
    /// Short reads returned by the kernel and retried.
    ///
    /// Non-zero means the file is being read from something that does not
    /// return full requests, which changes the per-read cost model.
    pub short_reads: u64,
}

impl std::ops::Add for AccessStats {
    type Output = Self;

    /// Sum two handles' counters.
    ///
    /// The daemon and the CLI read one volume through two handles, and the
    /// per-query cost is their total. Adding is well-defined for every field
    /// because each counts an event, not a rate.
    fn add(self, rhs: Self) -> Self {
        Self {
            reads: self.reads + rhs.reads,
            bytes: self.bytes + rhs.bytes,
            blocks_touched: self.blocks_touched + rhs.blocks_touched,
            straddling_reads: self.straddling_reads + rhs.straddling_reads,
            short_reads: self.short_reads + rhs.short_reads,
        }
    }
}

impl AccessStats {
    /// Mean device blocks per read, in hundredths.
    ///
    /// Reported as a fixed-point integer rather than a float so it can appear in
    /// the same JSON as the engine's integer counters without a float
    /// round-trip.
    pub const fn blocks_per_read_centi(&self) -> u64 {
        if self.reads == 0 {
            return 0;
        }
        self.blocks_touched * 100 / self.reads
    }
}

/// A SECTOR volume read with positional reads.
#[derive(Debug)]
pub struct FileFlash {
    file: File,
    len: u32,
    stats: AccessStats,
}

impl FileFlash {
    /// Open `path` read-only.
    ///
    /// Fails when the volume cannot hold the format's two manifest slots, or
    /// when it exceeds the 4 GiB the format's `u32` region bases can address.
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
        Ok(Self {
            file,
            len: len as u32,
            stats: AccessStats::default(),
        })
    }

    /// Volume length in bytes.
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether the volume is empty. Never true — [`Self::open`] rejects a
    /// volume too short for the header — but `clippy::len_without_is_empty`
    /// asks for it, and answering the lint is cheaper than an allow.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reads issued so far.
    pub const fn stats(&self) -> AccessStats {
        self.stats
    }

    /// Clear the access counters, so a measurement can exclude mount cost.
    pub fn reset_stats(&mut self) {
        self.stats = AccessStats::default();
    }

    /// Read exactly `buf.len()` bytes at `addr`, counting the access.
    fn read_counted(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), Error> {
        let end = (addr as u64).saturating_add(buf.len() as u64);
        if end > self.len as u64 {
            return Err(Error::OutOfBounds {
                addr,
                len: buf.len(),
            });
        }

        self.stats.reads += 1;
        self.stats.bytes += buf.len() as u64;
        if !buf.is_empty() {
            let first = addr as usize / DEVICE_BLOCK_BYTES;
            let last = (end as usize - 1) / DEVICE_BLOCK_BYTES;
            self.stats.blocks_touched += (last - first + 1) as u64;
            if last != first {
                self.stats.straddling_reads += 1;
            }
        }

        // `read_at` may return short. Loop rather than treating a short read as
        // an error: it is legal, and on a slow SD card under memory pressure it
        // happens. The count is kept because a non-zero value changes the
        // per-read cost model.
        let mut done = 0usize;
        while done < buf.len() {
            let n = self
                .file
                .read_at(&mut buf[done..], addr as u64 + done as u64)?;
            if n == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "volume ended mid-read",
                )));
            }
            if done + n < buf.len() {
                self.stats.short_reads += 1;
            }
            done += n;
        }
        Ok(())
    }
}

impl NorFlash for FileFlash {
    type Error = Error;

    fn page_size(&self) -> usize {
        sector_format::BLOCK_BYTES
    }

    fn sector_size(&self) -> usize {
        sector_format::SECTOR_BYTES
    }

    fn capacity(&self) -> u32 {
        self.len
    }

    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), Self::Error> {
        self.read_counted(addr, buf)
    }

    fn program(&mut self, _addr: u32, _buf: &[u8]) -> Result<(), Self::Error> {
        Err(Error::ReadOnly)
    }

    fn erase(&mut self, _sector_addr: u32) -> Result<(), Self::Error> {
        Err(Error::ReadOnly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::test_support::write_temp_volume;

    #[test]
    fn reads_return_the_bytes_written() {
        let (_dir, path, image) = write_temp_volume("file_roundtrip");
        let mut f = FileFlash::open(&path).expect("open");
        assert_eq!(f.capacity(), image.len() as u32);

        let mut buf = [0u8; 256];
        f.read(1024, &mut buf).expect("read");
        assert_eq!(&buf[..], &image[1024..1280]);
    }

    #[test]
    fn a_read_past_the_end_is_refused_rather_than_truncated() {
        let (_dir, path, image) = write_temp_volume("file_oob");
        let mut f = FileFlash::open(&path).expect("open");
        let mut buf = [0u8; 64];
        let at = image.len() as u32 - 32;
        assert!(matches!(
            f.read(at, &mut buf),
            Err(Error::OutOfBounds { .. })
        ));
    }

    #[test]
    fn block_accounting_charges_a_straddle_to_two_blocks() {
        // The reason the counter exists: a 128 B record crossing a boundary
        // costs the FTL two blocks, and a byte count cannot show that.
        let (_dir, path, _) = write_temp_volume("file_straddle");
        let mut f = FileFlash::open(&path).expect("open");

        let mut buf = [0u8; 128];
        f.read(0, &mut buf).expect("aligned read");
        assert_eq!(f.stats().blocks_touched, 1);
        assert_eq!(f.stats().straddling_reads, 0);

        f.read(DEVICE_BLOCK_BYTES as u32 - 64, &mut buf)
            .expect("straddling read");
        assert_eq!(f.stats().blocks_touched, 3);
        assert_eq!(f.stats().straddling_reads, 1);
        assert_eq!(f.stats().reads, 2);
        assert_eq!(f.stats().bytes, 256);
        // 3 blocks over 2 reads.
        assert_eq!(f.stats().blocks_per_read_centi(), 150);
    }

    #[test]
    fn writes_are_refused() {
        // A host tool must not be able to corrupt a volume whose write-ordering
        // guarantees assume it is immutable.
        let (_dir, path, _) = write_temp_volume("file_readonly");
        let mut f = FileFlash::open(&path).expect("open");
        assert!(matches!(f.program(0, &[0u8; 512]), Err(Error::ReadOnly)));
        assert!(matches!(f.erase(0), Err(Error::ReadOnly)));
    }

    #[test]
    fn a_file_too_short_for_the_header_is_refused_at_open() {
        let dir = std::env::temp_dir().join("sector_os_short");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("short.sector");
        std::fs::write(&path, [0u8; 16]).expect("write");
        assert!(matches!(
            FileFlash::open(&path),
            Err(Error::TooSmall { .. })
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stats_reset_so_mount_cost_can_be_excluded() {
        let (_dir, path, _) = write_temp_volume("file_reset");
        let mut f = FileFlash::open(&path).expect("open");
        let mut buf = [0u8; 64];
        f.read(0, &mut buf).expect("read");
        assert_eq!(f.stats().reads, 1);
        f.reset_stats();
        assert_eq!(f.stats(), AccessStats::default());
        assert_eq!(AccessStats::default().blocks_per_read_centi(), 0);
    }
}
