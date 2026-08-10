//! `sector-os` — POSIX storage backends and platform detection.
//!
//! The Linux half of the portability seam. `sector-core` reaches storage through
//! [`sector_hal::NorFlash`] and [`sector_hal::Xip`]; this crate implements those
//! over a file, so the same engine that runs on bare metal serves a volume on a
//! Raspberry Pi with no change to the query path.
//!
//! # Two backends, and why the choice is not a performance detail
//!
//! [`file::FileFlash`] reads with `pread` and implements `NorFlash` alone.
//! [`mapped::MappedFlash`] maps the volume and implements `NorFlash` **and**
//! `Xip`, which the engine treats as a claim that a fetch is a load instruction.
//!
//! On a Pi that claim is not true in the way it is on raw NOR. Every Pi reads
//! its volume through a flash translation layer — microSD, USB, or NVMe — so the
//! first touch of a mapped page is a block read the FTL services, not a load.
//! `mmap` moves where that cost is paid (a page fault instead of a `read` call)
//! and lets the page cache absorb repeats; it does not make the storage
//! byte-addressable.
//!
//! This matters beyond tidiness because the project reports an inversion — the
//! smaller tier executing stage two faster than the larger one, because raw NOR
//! services a per-candidate random read as a load while managed storage pays the
//! FTL penalty. A mapped backend that reported itself as execute-in-place would
//! erase that effect from the measurement rather than measure it.
//!
//! So `FileFlash` is the default and `mmap` is a non-default feature. Running
//! both on one board turns the question into a number: the gap between them is
//! the page cache's contribution, and it is reported rather than assumed.
//!
//! # What is not here
//!
//! No writes. `NorFlash::program` and `erase` return
//! [`Error::ReadOnly`]: a SECTOR volume is built offline by `sector-build` and
//! served read-only, and the append path is a firmware concern. A backend that
//! silently accepted writes would let a host tool corrupt a volume that the
//! format's write-ordering rules assume is immutable.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod file;
pub mod ingest;
pub mod json;
#[cfg(feature = "mmap")]
pub mod mapped;
pub mod platform;
pub mod search;
pub mod source;
pub mod verify;
pub mod volume;

pub use file::FileFlash;
pub use ingest::{append, capacity, AppendError, AppendReport};
#[cfg(feature = "mmap")]
pub use mapped::MappedFlash;
pub use search::{Answer, Searcher};
pub use volume::{Geometry, HostVolume};

/// Why a backend operation failed.
#[derive(Debug)]
pub enum Error {
    /// The underlying file could not be read.
    Io(std::io::Error),
    /// An access fell outside the volume.
    OutOfBounds {
        /// Byte address.
        addr: u32,
        /// Length in bytes.
        len: usize,
    },
    /// A write was attempted. Host-side volumes are read-only; see the crate
    /// documentation.
    ReadOnly,
    /// The volume is smaller than the format's fixed header.
    TooSmall {
        /// Bytes present.
        len: u64,
        /// Bytes the format requires.
        needed: u64,
    },
    /// The volume does not fit the 32-bit address space the format uses.
    ///
    /// `RegionDesc::base` is a `u32`, so a volume beyond 4 GiB cannot be
    /// addressed by the format regardless of the host's word size.
    TooLarge {
        /// Bytes present.
        len: u64,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::OutOfBounds { addr, len } => {
                write!(f, "access at {addr} for {len} B falls outside the volume")
            }
            Self::ReadOnly => write!(f, "host volumes are read-only"),
            Self::TooSmall { len, needed } => {
                write!(f, "volume is {len} B, shorter than the {needed} B header")
            }
            Self::TooLarge { len } => write!(
                f,
                "volume is {len} B, beyond the 4 GiB the format can address"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
