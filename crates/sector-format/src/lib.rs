//! `sector-format` — the on-flash byte layout.
//!
//! One crate owns every offset, magic byte and size assertion, so a layout
//! change is a single reviewable diff and firmware and host builder cannot
//! drift apart. All integers are little-endian.
//!
//! A SECTOR volume is written once by the host builder, then read and appended
//! to by the device. It is not a general mutable store: PQ training, label
//! optimisation and criticality measurement are corpus-global operations, and
//! the build is offline for that reason.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![deny(clippy::float_arithmetic)]

pub mod codebook_blk;
pub mod group;
pub mod manifest;
pub mod payload_blk;
pub mod region;
pub mod rerank_blk;

/// Tier profiles: the const-generic parameter sets of report Table 1.
pub mod profile;

/// ASCII "SECT" — first four bytes of every volume manifest.
pub const MAGIC_VOLUME: [u8; 4] = *b"SECT";

/// On-flash format version. Any layout change bumps this; a device refuses to
/// mount a volume it does not recognise rather than mis-reading it.
pub const FORMAT_VERSION: u16 = 2;

/// Block granularity for CRC and for the payload/rerank regions.
///
/// Chosen at 512 B because the CRC cost is `4/512 = 0.78%` of stored bytes,
/// against 25% for a per-vector CRC on a 16-byte payload. The consequence is
/// stated rather than hidden: a *detected* block failure drops every vector in
/// the block — 32 of them at a 16-byte payload — so `Delta_payload = 1` holds
/// per corrupted *byte*, not per detected failure.
pub const BLOCK_BYTES: usize = 512;

/// Bytes of CRC-32 carried per block, stored out-of-line in a block-parallel
/// array so the payload array stays exactly `pi`-strided and scannable without
/// per-vector arithmetic.
pub const BLOCK_CRC_BYTES: usize = 4;

/// Erase-sector size assumed for protection-group alignment. Groups align to
/// this because flash failures are sector-correlated; a group straddling a
/// sector boundary cannot bound its own failure probability.
pub const SECTOR_BYTES: usize = 4096;
