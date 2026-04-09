//! `sector-core` — the heapless query engine.
//!
//! `no_std`, no `alloc`, no panicking paths, no hardware names. Every buffer
//! the engine touches lives in a caller-provided [`workspace::Workspace`] sized
//! at compile time from the tier profile, so peak RAM is a linker symbol.
//!
//! # The query path
//!
//! Two-stage retrieval is mandatory. Single-stage PQ recall is unusable at
//! every configuration measured — 0.043 at m=32, b=4, D=256, N=20,000. The
//! stages are
//!
//! 1. rotate the query in place (integer FWHT);
//! 2. build the ADC table, `2^b * D` multiply-accumulates, independent of `N`;
//! 3. scan the compressed payload — `m` table lookups and `m` adds per vector,
//!    no multiplies — into a bounded heap of `R` candidates;
//! 4. verify each candidate's block CRC, drop on mismatch, and rescore the
//!    survivors against the higher-precision rerank copy;
//! 5. drain the heap for the top `k`.
//!
//! Stage 4 streams from flash. On a backend implementing [`sector_hal::Xip`] it
//! borrows from the memory-mapped window, so the steady-state query performs no
//! copies and no allocation.
//!
//! # What this crate does not do
//!
//! It does not train codebooks, optimise labels, measure criticality, or solve
//! the protection allocation. Those are corpus-global and offline; they live in
//! `sector-build` and arrive as a finished volume image.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![deny(clippy::float_arithmetic)]

pub mod append;
pub mod error;
pub mod heap;
pub mod metrics;
pub mod mount;
pub mod query;
pub mod repair;
pub mod rerank;
pub mod scan;
pub mod scrub;
pub mod workspace;
