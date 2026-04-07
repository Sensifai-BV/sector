//! Exhaustive error enum.
//!
//! Storage faults and integrity faults are distinct kinds and callers react
//! differently: a failed read can be retried, while a block that failed its CRC
//! is dropped and counted. One variant for both makes the difference invisible.
//!
//! # Error families
//!
//! *Operational* — backend read, program or erase failed; region out of range.
//! Retryable.
//!
//! *Integrity* — block CRC mismatch, manifest digest mismatch, unrepairable
//! region. Countable degradation.
//!
//! *Configuration* — unknown format version, profile the device cannot host.
//! Refusal at mount.
//!
//! No panicking paths in the core: no `unwrap`, no `expect`, no indexing that
//! can panic outside tests. On a device with no operating system a panic is a
//! reset, and a reset during an append is a torn write. Errors propagate
//! through this enum.
//!
//! The enum is `#[non_exhaustive]`, so adding a variant is not a breaking
//! change downstream.

use crate::mount::MountError;
use crate::workspace::WorkspaceError;
use sector_format::region::RegionKind;

/// Every way a query or maintenance operation can fail.
///
/// Storage faults and integrity faults are distinct variants because callers
/// react differently: a failed read can be retried, while a block that failed
/// its CRC is dropped and counted. One variant for both makes the difference
/// invisible at the call site.
///
/// There are no panicking paths in this crate. On a device with no operating
/// system a panic is a reset, and a reset during an append is a torn write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    // --- Operational: the storage layer failed. Retryable. ---
    /// A backend read failed.
    Read {
        /// Byte offset the read was attempted at.
        addr: u32,
    },
    /// A backend program failed.
    Program {
        /// Byte offset the program was attempted at.
        addr: u32,
    },
    /// A backend erase failed.
    Erase {
        /// Sector base address.
        addr: u32,
    },
    /// An access fell outside a region's extent.
    OutOfRange {
        /// The region.
        kind: RegionKind,
        /// Byte offset requested.
        offset: u32,
    },

    // --- Integrity: the bytes are wrong. Countable degradation. ---
    /// A block failed its CRC.
    BlockCrc {
        /// Block index within its region.
        block: u32,
    },
    /// A region could not be repaired from replicas or parity.
    ///
    /// For the codebook this is fatal, because one corrupted byte perturbs
    /// `N / 2^b` vectors. For payload or rerank it degrades gracefully — the
    /// asymmetry follows the fan-out, not a policy preference.
    Unrepairable {
        /// The region.
        kind: RegionKind,
    },

    // --- Configuration: refused at mount. ---
    /// The volume could not be mounted.
    Mount(MountError),
    /// A workspace buffer is too small for the profile.
    Workspace(WorkspaceError),

    // --- Programming errors surfaced as values, never panics. ---
    /// An output buffer is too small for the result.
    OutputTooSmall {
        /// Capacity supplied.
        found: usize,
        /// Capacity required.
        expected: usize,
    },
}

impl Error {
    /// Whether retrying the same operation could succeed.
    ///
    /// Only the operational family is retryable. An integrity fault will
    /// reproduce, and a configuration fault is a property of the image.
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::Read { .. } | Error::Program { .. } | Error::Erase { .. }
        )
    }

    /// Whether this is degradation to be counted rather than a failure.
    ///
    /// A dropped block reduces recall; it does not fail the query.
    pub const fn is_degradation(&self) -> bool {
        matches!(self, Error::BlockCrc { .. })
    }

    /// Whether the volume cannot serve queries at all.
    pub const fn is_fatal(&self) -> bool {
        matches!(
            self,
            Error::Mount(_) | Error::Workspace(_) | Error::Unrepairable { .. }
        )
    }
}

impl From<MountError> for Error {
    fn from(e: MountError) -> Self {
        Error::Mount(e)
    }
}

impl From<WorkspaceError> for Error {
    fn from(e: WorkspaceError) -> Self {
        Error::Workspace(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_format::manifest::ManifestError;

    #[test]
    fn the_three_families_are_distinguishable() {
        let operational = Error::Read { addr: 4096 };
        let integrity = Error::BlockCrc { block: 7 };
        let configuration = Error::Mount(MountError::Manifest(ManifestError::NoValidSlot));

        assert!(operational.is_retryable());
        assert!(!operational.is_degradation());
        assert!(!operational.is_fatal());

        assert!(!integrity.is_retryable());
        assert!(integrity.is_degradation());
        assert!(!integrity.is_fatal());

        assert!(!configuration.is_retryable());
        assert!(configuration.is_fatal());
    }

    #[test]
    fn codebook_and_payload_damage_differ_in_severity() {
        // A lost codebook block perturbs N/2^b vectors; a lost payload block
        // loses the 32 it held. Only the first is fatal.
        let codebook = Error::Unrepairable {
            kind: RegionKind::Codebook,
        };
        let payload = Error::BlockCrc { block: 3 };
        assert!(codebook.is_fatal());
        assert!(!payload.is_fatal());
        assert!(payload.is_degradation());
    }

    #[test]
    fn conversions_preserve_the_underlying_fault() {
        let w = WorkspaceError::TooSmall {
            which: crate::workspace::Buffer::Heap,
            found: 10,
            expected: 500,
        };
        assert_eq!(Error::from(w), Error::Workspace(w));
    }
}
