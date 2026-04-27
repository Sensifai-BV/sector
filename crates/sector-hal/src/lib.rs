//! `sector-hal` — the portability seam.
//!
//! Trait definitions only: no logic, no dependencies, no concrete hardware.
//! `sector-core` reaches storage, time and instrumentation exclusively through
//! these traits, so no chip name appears above this crate.
//!
//! # Scope
//!
//! SECTOR is read-dominated, single-writer and built offline, so the trait set
//! is smaller than a general embedded store needs:
//!
//! - no entropy source — nothing in the query path is randomised;
//! - no monotonic counter — no rollback adversary is in scope;
//! - no async flash — the hot path on an [`Xip`] backend performs no block I/O.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![deny(clippy::float_arithmetic)]

/// The erased state of a NOR cell. Programming clears bits; only an erase sets
/// them. A region reading all-`0xFF` is either never-written or freshly erased,
/// which is what makes an append head recognisable without a separate journal.
pub const ERASED_BYTE: u8 = 0xFF;

/// NOR flash with program-once-per-erase semantics.
///
/// The contract, which every implementation and the simulator must honour:
///
/// 1. `program` may only touch pages currently in the erased state. Programming
///    a page twice without an intervening `erase` is an error, not a silent
///    AND of the two writes.
/// 2. `addr` and `buf.len()` are page-aligned and page-sized multiples for
///    `program`; `erase` takes a sector base address.
/// 3. Return of `Ok` from `program`/`erase` means durable. There is no flush.
/// 4. A power loss inside `program` may leave the page partially written; the
///    block CRC is what detects it (`sector-codec`). A power loss inside
///    `erase` may leave the sector partially erased.
///
/// `read` exists for every backend. On a backend that also implements [`Xip`],
/// the query hot path does not call it — see that trait's documentation for why
/// the distinction is the load-bearing one in this design.
pub trait NorFlash {
    /// Backend error type.
    type Error: core::fmt::Debug;

    /// Program granularity in bytes (NOR page, typically 256).
    fn page_size(&self) -> usize;
    /// Erase granularity in bytes (NOR sector, typically 4096).
    ///
    /// This is also the unit at which SECTOR allocates protection: failures are
    /// sector-correlated, so a protection group that does not align to this
    /// boundary cannot bound its own failure probability.
    fn sector_size(&self) -> usize;
    /// Total addressable capacity in bytes.
    fn capacity(&self) -> u32;

    /// Copy `buf.len()` bytes starting at `addr` into `buf`.
    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), Self::Error>;
    /// Program a page-aligned, page-sized run. Only legal on erased pages.
    fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), Self::Error>;
    /// Erase one sector, returning it to [`ERASED_BYTE`].
    fn erase(&mut self, sector_addr: u32) -> Result<(), Self::Error>;
}

/// A byte-addressable, memory-mapped view of flash — execute-in-place.
///
/// This trait is the reason the smallest target is the fastest one for stage two. Raw NOR is
/// mapped into the address space, so a read is a load instruction: no FTL, no
/// 4 KiB block granularity, no random-read penalty, and no bounce buffer. The
/// consequence for the engine is structural rather than incremental — with
/// [`window`] the scan and rerank stages borrow their operands in place and the
/// hot loop performs *zero* I/O calls and *zero* copies, which is what makes a
/// heapless fixed-workspace query path achievable at all.
///
/// Managed NAND behind an FTL (microSD, eMMC) cannot implement this. Such a
/// backend implements [`NorFlash`] alone, and the engine falls back to a
/// buffered read path whose cost is dominated by the random-read penalty the
/// FTL imposes. That fallback is not a degraded corner case to be hidden: it is
/// the measured inversion this project reports, where the larger tier performs
/// the same access pattern far more slowly than the smaller one. Keeping the
/// capability in the type system means a backend cannot quietly claim it.
///
/// Implementors must guarantee the returned slice is stable and coherent for
/// the borrow's lifetime — no concurrent programming of the same range.
pub trait Xip {
    /// Borrow `len` bytes at `addr` directly from the mapped window.
    ///
    /// Returns `None` when the range falls outside the mapped window; the
    /// caller then falls back to [`NorFlash::read`]. A backend with no mapped
    /// window must leave this trait unimplemented rather than implement it
    /// returning `None`, so that the engine binds the buffered path at mount
    /// instead of testing per access.
    fn window(&self, addr: u32, len: usize) -> Option<&[u8]>;
}

/// Coarse monotonic time source, milliseconds since an arbitrary origin.
///
/// Used only for reporting and for the measurement harness. Nothing in the
/// engine's control flow depends on wall-clock time; determinism is a
/// requirement, so a missing clock degrades reporting, never behaviour.
pub trait Clock {
    /// Milliseconds since an implementation-defined origin. Must not go
    /// backwards.
    fn now_ms(&self) -> u64;
}

/// The stages of a query, as reported to an [`Instrument`].
///
/// The split is not decorative: the report's cost model attributes query energy
/// to these five stages separately, and two of them (`Table`, `Rerank`) are the
/// terms that decide the tier configuration. A measurement that reports only a
/// total cannot falsify that model, so the engine marks each boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// In-place integer rotation of the query vector.
    Rotate,
    /// Building the asymmetric-distance lookup table. Cost is `2^b * D`
    /// multiply-accumulates and is independent of `N` — at T0 this dominates
    /// everything else, which is the second independent reason for `b = 4`.
    Table,
    /// Sequential scan of the compressed payload: `m` lookups and `m` adds per
    /// vector, no multiplies.
    Scan,
    /// Two-stage rescoring of the `R` candidates against the higher-precision
    /// copy, including block-CRC verification and drop-on-mismatch.
    Rerank,
    /// Heap drain and result materialisation.
    Finalize,
}

/// Edge direction for a phase marker.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Edge {
    /// Phase entered.
    Enter,
    /// Phase left.
    Leave,
}

/// Sink for on-device measurement.
///
/// The default implementation is [`NoInstrument`], which compiles to nothing.
/// The firmware measurement build supplies a GPIO-toggling implementation whose
/// edges are what a current probe on the supply rail is aligned against — the
/// report's energy-per-query figures are only attributable to a phase because
/// this boundary exists in the engine rather than around it.
pub trait Instrument {
    /// Free-running cycle counter, for host-side timing without a scope.
    fn cycles(&self) -> u64;
    /// Emit a phase boundary marker.
    fn mark(&mut self, phase: Phase, edge: Edge);
}

/// Zero-cost no-op instrumentation: the default for production builds.
#[derive(Clone, Copy, Default, Debug)]
pub struct NoInstrument;

impl Instrument for NoInstrument {
    #[inline(always)]
    fn cycles(&self) -> u64 {
        0
    }
    #[inline(always)]
    fn mark(&mut self, _phase: Phase, _edge: Edge) {}
}
