//! `sector-codec` — detection and repair.
//!
//! Two mechanisms, kept separate because they are not interchangeable.
//!
//! # Detection
//!
//! An MDS code corrects `n - k` *erasures* (locations known) but only
//! `floor((n - k) / 2)` unknown *errors*. Silent flash corruption is an error
//! channel. A per-block CRC localises the damage and converts the channel to an
//! erasure channel, restoring the full `n - k` figure. At 512-byte granularity
//! it costs `4/512 = 0.78%` of stored bytes; a per-vector CRC against a 16-byte
//! payload costs 25%.
//!
//! # Repair
//!
//! Chosen per scale. At the report's T0 configuration (D=768, b=4, int8) a full
//! second copy of the 12 KiB codebook costs 6.2% of the 192 KiB index budget
//! against 3.1% for RS(12,8) parity — the same order, with no GF(2^8)
//! arithmetic. Replication is the default at T0/T1; RS applies to structures
//! large enough that halving the parity bytes outweighs the decode cost.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![deny(clippy::float_arithmetic)]

/// Bytes of CRC carried per protected block.
pub const CRC_BYTES: usize = 4;

/// CRC-32 over a block, and the verify-or-drop decision.
pub mod crc;
/// GF(2^8) log/antilog tables (512 B, in rodata).
pub mod gf;
/// N-copy replication with CRC-directed copy selection.
pub mod replicate;
/// Systematic Reed-Solomon over GF(2^8), erasure-only decode.
pub mod rs;
