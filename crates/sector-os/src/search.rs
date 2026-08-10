//! The host search runtime: one mounted volume, answering queries.
//!
//! Both the CLI and the daemon go through [`Searcher`], so a REST answer and a
//! `sector query` answer are the same bytes from the same code. Two entry points
//! into one engine would eventually disagree, and the disagreement would show up
//! as a recall difference nobody could attribute.
//!
//! # What a `Searcher` owns
//!
//! The resident codebook, the fixed workspace, and **two** backend handles. Two
//! because [`sector_core::query::query`] borrows a payload source and a rerank
//! source mutably at the same time; on a device those are one flash controller
//! reached twice, and here they are two file descriptors on one volume. That is
//! also why [`crate::file::FileFlash`] reads with `pread` — two handles sharing a
//! seek offset would return each other's bytes.
//!
//! Nothing is allocated per query. The workspace is bound once at construction,
//! which is what makes a `Searcher` per worker thread the daemon's unit of
//! concurrency: `N` workers cost `N * fixed_bytes()`, a figure that is known
//! before the process starts.
//!
//! # Query quantization
//!
//! The engine scores integers, so an `f32` query is quantized to `i8` on the way
//! in. The scale is **one factor for the whole query**, taken from its own
//! largest magnitude. Scaling every component of a query by the same positive
//! constant multiplies every score by that constant, so the ranking is
//! unchanged — this is exact for ranking, not an approximation.
//!
//! What is *not* handled here is the codebook's own scaling.
//! `sector_build::encode::quantize` gives each subspace a scale from that
//! subspace's extent, and `sector_quant::adc::build_table` sums raw integer
//! products without consulting it, so subspace `j` enters the score weighted by
//! its `num_j`. On an isotropic corpus the weights agree and the score is the
//! intended inner product; on an anisotropic one it is not, and the scales are
//! not stored in the manifest so no reader can correct for it. That is a
//! pre-existing format defect, recorded in `docs/DEVELOPMENT_STATE.md`, and it is
//! named here because this is where a reader will wonder about it.

use std::path::Path;

use sector_core::heap::Candidate;
use sector_core::metrics::Metrics;
use sector_core::query::{query, QueryStats};
use sector_core::workspace::Workspace;
use sector_hal::{Instrument, NorFlash};
use sector_quant::codebook::{Codebook, Scale};

use crate::source::{PayloadReader, RerankReader};
use crate::volume::{Geometry, HostVolume, MountError};
use crate::Error;

/// Blocks the payload reader buffers per read.
///
/// 64 blocks is 32 KiB: it amortises the read over roughly 2,000 vectors at
/// T0's 16 B payload, and stays inside the order of magnitude of the ADC table
/// so the two do not evict each other on the smallest tier.
pub const PAYLOAD_BLOCKS_PER_READ: usize = 64;

/// A backend this crate can open from a path.
///
/// Exists so [`Searcher`] is generic over the storage path without knowing which
/// one it has — the property that makes a buffered-versus-mapped comparison a
/// measurement rather than a rewrite.
pub trait OpenBackend: NorFlash + Sized {
    /// Open the volume at `path`.
    fn open_volume(path: &Path) -> Result<Self, Error>;

    /// Name for reporting, so a measurement records which path produced it.
    fn backend_name() -> &'static str;

    /// Whether this backend implements [`sector_hal::Xip`].
    fn is_mapped() -> bool;
}

impl OpenBackend for crate::file::FileFlash {
    fn open_volume(path: &Path) -> Result<Self, Error> {
        Self::open(path)
    }
    fn backend_name() -> &'static str {
        "file"
    }
    fn is_mapped() -> bool {
        false
    }
}

#[cfg(feature = "mmap")]
impl OpenBackend for crate::mapped::MappedFlash {
    fn open_volume(path: &Path) -> Result<Self, Error> {
        Self::open(path)
    }
    fn backend_name() -> &'static str {
        "mmap"
    }
    fn is_mapped() -> bool {
        true
    }
}

/// A backend that counts what its reads cost.
///
/// Separate from [`OpenBackend`] because the mapped backend's cost is in page
/// faults rather than reads, and reporting a fault count in a field named `reads`
/// would make the two backends' figures look comparable when they are not. A
/// mapped backend implements this with its own accounting and the caller reads
/// [`crate::mapped::MappedFlash::fault_stats`] for the part that differs.
pub trait HasAccessStats {
    /// Read counters.
    fn access_stats(&self) -> crate::file::AccessStats;
    /// Clear them.
    fn reset_access_stats(&mut self);
}

impl HasAccessStats for crate::file::FileFlash {
    fn access_stats(&self) -> crate::file::AccessStats {
        self.stats()
    }
    fn reset_access_stats(&mut self) {
        self.reset_stats();
    }
}

#[cfg(feature = "mmap")]
impl HasAccessStats for crate::mapped::MappedFlash {
    /// A mapped backend's `read` is a `memcpy` from the page cache, so `reads`
    /// and `bytes` are real but do not represent syscalls. The fault count — the
    /// figure that corresponds to storage work — is in
    /// [`crate::mapped::MappedFlash::fault_stats`], and is deliberately not
    /// folded in here.
    fn access_stats(&self) -> crate::file::AccessStats {
        let f = self.fault_stats();
        crate::file::AccessStats {
            reads: f.borrows,
            bytes: 0,
            blocks_touched: f.pages_touched,
            straddling_reads: 0,
            short_reads: 0,
        }
    }
    fn reset_access_stats(&mut self) {
        self.reset_stats();
    }
}

/// One query's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    /// Result ids, best first.
    pub ids: Vec<u32>,
    /// Their exact-rescored scores.
    pub scores: Vec<i32>,
    /// What the query cost.
    pub stats: QueryStats,
}

/// A mounted volume, ready to answer queries.
pub struct Searcher<F: OpenBackend> {
    volume: HostVolume,
    payload_flash: F,
    rerank_flash: F,
    codebook: Vec<i8>,
    adc: Vec<i32>,
    heap_scores: Vec<i32>,
    heap_ids: Vec<u32>,
    rotation: Vec<i32>,
    bounce: Vec<u8>,
    /// Candidate depth in use, which may be below the image's.
    r: usize,
}

impl<F: OpenBackend> Searcher<F> {
    /// Mount the volume at `path` and bind a workspace.
    ///
    /// `r` overrides the image's candidate depth; `None` uses the image's own.
    /// A depth above the image's is refused rather than clamped, because a
    /// silently reduced depth is a silently reduced recall.
    pub fn open(path: &Path, r: Option<usize>) -> Result<Self, SearchError> {
        let mut mount_flash = F::open_volume(path).map_err(SearchError::Backend)?;
        let volume = HostVolume::mount(&mut mount_flash, None).map_err(SearchError::Mount)?;
        drop(mount_flash);

        let g = volume.geometry;
        let depth = match r {
            None => g.r,
            Some(0) => g.r,
            Some(want) if want > sector_core::query::MAX_R => {
                return Err(SearchError::DepthTooLarge {
                    want,
                    limit: sector_core::query::MAX_R,
                })
            }
            Some(want) => want,
        };

        let codebook = volume.codebook.iter().map(|b| *b as i8).collect();
        Ok(Self {
            payload_flash: F::open_volume(path).map_err(SearchError::Backend)?,
            rerank_flash: F::open_volume(path).map_err(SearchError::Backend)?,
            codebook,
            adc: vec![0i32; g.m * g.centroids],
            heap_scores: vec![0i32; depth],
            heap_ids: vec![0u32; depth],
            rotation: vec![0i32; g.d],
            bounce: vec![0u8; sector_format::BLOCK_BYTES],
            r: depth,
            volume,
        })
    }

    /// The volume's geometry.
    pub const fn geometry(&self) -> &Geometry {
        &self.volume.geometry
    }

    /// The mounted volume.
    pub const fn volume(&self) -> &HostVolume {
        &self.volume
    }

    /// Candidate depth in use.
    pub const fn depth(&self) -> usize {
        self.r
    }

    /// Which backend this searcher reads through.
    pub fn backend_name(&self) -> &'static str {
        F::backend_name()
    }

    /// Access counters from both handles, summed.
    ///
    /// Summed rather than reported separately because the two handles serve one
    /// query between them: stage one reads through the payload handle and stage
    /// two through the rerank handle, and the per-query cost is their total.
    /// [`crate::file::FileFlash::stats`] on each is available for a per-stage
    /// split when that is what is wanted.
    pub fn backend_stats(&self) -> crate::file::AccessStats
    where
        F: HasAccessStats,
    {
        self.payload_flash.access_stats() + self.rerank_flash.access_stats()
    }

    /// Clear both handles' counters, so mount and warm-up costs can be excluded.
    pub fn reset_backend_stats(&mut self)
    where
        F: HasAccessStats,
    {
        self.payload_flash.reset_access_stats();
        self.rerank_flash.reset_access_stats();
    }

    /// The volume's manifest, including the id-gap fields.
    pub const fn manifest(&self) -> &sector_format::manifest::Manifest {
        &self.volume.manifest
    }

    /// Stored records for `ids`, `None` where an id is not held.
    ///
    /// `None` covers two cases the caller must distinguish — an id inside the
    /// volume's gap, and an id past its extent — so the manifest is exposed
    /// alongside rather than folded into this result.
    ///
    /// Reads through the rerank handle, so it perturbs that handle's access
    /// counters: a `/stats` scrape taken after an enumeration will include these
    /// reads. That is honest rather than convenient — hiding them would make the
    /// per-query figures understate what the process actually asked of the disk.
    pub fn records(&mut self, ids: &[u32]) -> Result<Vec<Option<Vec<u8>>>, SearchError> {
        let g = self.volume.geometry;
        let base = self.volume.rerank_base();
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            // `holds` is the authority, not `id < n`: an appended volume has
            // addressable ids that were never written.
            if !self.volume.manifest.holds(id) {
                out.push(None);
                continue;
            }
            let Some(off) = g.rerank.offset_of(id as usize) else {
                out.push(None);
                continue;
            };
            let mut rec = vec![0u8; g.rerank_bytes];
            sector_hal::NorFlash::read(&mut self.rerank_flash, base + off as u32, &mut rec)
                .map_err(|_| SearchError::Internal("rerank read failed"))?;
            out.push(Some(rec));
        }
        Ok(out)
    }

    /// Resident bytes this searcher holds, excluding the backend's own buffers.
    ///
    /// The figure a deployment multiplies by its worker count. Reported rather
    /// than estimated: the daemon's memory claim has to be checkable.
    pub fn resident_bytes(&self) -> usize {
        self.codebook.len()
            + self.adc.len() * 4
            + self.heap_scores.len() * 4
            + self.heap_ids.len() * 4
            + self.rotation.len() * 4
            + self.bounce.len()
    }

    /// Answer one `f32` query.
    pub fn search(&mut self, q: &[f32], k: usize) -> Result<Answer, SearchError> {
        let quantized = quantize_query(q, self.volume.geometry.d)?;
        self.search_quantized(&quantized, k)
    }

    /// Answer one already-quantized query.
    ///
    /// The daemon accepts `i8` vectors directly, which skips a float round-trip
    /// for a client that already holds quantized embeddings.
    pub fn search_quantized(&mut self, q: &[i8], k: usize) -> Result<Answer, SearchError> {
        let mut sink = NoInstrument;
        self.search_instrumented(q, k, &mut sink)
    }

    /// Answer one quantized query, marking phase boundaries into `instrument`.
    pub fn search_instrumented<I: Instrument>(
        &mut self,
        q: &[i8],
        k: usize,
        instrument: &mut I,
    ) -> Result<Answer, SearchError> {
        let g = self.volume.geometry;
        if q.len() != g.d {
            return Err(SearchError::Dimension {
                found: q.len(),
                expected: g.d,
            });
        }
        let k = k.clamp(1, sector_core::query::MAX_R);

        // Per-subspace views over the resident codebook. A unit scale: the ADC
        // table sums raw integer products, so passing a scale here would not
        // change the score — see the module documentation.
        let unit = Scale::new(1, 1).ok_or(SearchError::Internal("unit scale"))?;
        let per_book = g.centroids * g.ds;
        let mut books = Vec::with_capacity(g.m);
        for j in 0..g.m {
            let slice = self
                .codebook
                .get(j * per_book..(j + 1) * per_book)
                .ok_or(SearchError::Internal("codebook shorter than the geometry"))?;
            books.push(
                Codebook::new(slice, g.centroids, g.ds, unit)
                    .map_err(|_| SearchError::Internal("codebook view"))?,
            );
        }

        let mut ws = Workspace {
            adc_table: &mut self.adc,
            heap_scores: &mut self.heap_scores,
            heap_ids: &mut self.heap_ids,
            rotation: &mut self.rotation,
            bounce: &mut self.bounce,
            scrub_cursor: 0,
        };
        // The gap is passed to the reader so the scan never sees the padding slots
        // an append left behind: their codes sit in a CRC-valid block and would
        // otherwise occupy candidate slots real vectors could have used.
        let gap = self.volume.manifest.gap();
        let mut payload = PayloadReader::with_gap(
            &mut self.payload_flash,
            self.volume.payload_base(),
            &g,
            PAYLOAD_BLOCKS_PER_READ,
            (gap.0 as usize, gap.1 as usize),
        );
        let mut rerank = RerankReader::new(
            &mut self.rerank_flash,
            self.volume.rerank_base(),
            self.volume.rerank_crc_base(),
            &g,
        );

        let mut out = vec![Candidate { score: 0, id: 0 }; k];
        let mut metrics = Metrics::default();
        let stats = query(
            q,
            &books,
            // No rotation: the image is emitted unrotated, so applying one here
            // would score against a basis the codebook was not trained in.
            &[],
            0,
            &mut payload,
            &mut rerank,
            &mut ws,
            instrument,
            g.payload_bytes,
            k,
            &mut out,
            &mut metrics,
        )
        .map_err(|e| SearchError::Query(format!("{e:?}")))?;

        let taken = stats.returned as usize;
        Ok(Answer {
            ids: out[..taken].iter().map(|c| c.id).collect(),
            scores: out[..taken].iter().map(|c| c.score).collect(),
            stats,
        })
    }
}

/// Quantize an `f32` query to `i8` with one scale for the whole vector.
///
/// Uniform scaling multiplies every score by the same positive constant, so the
/// ranking is exactly preserved. An all-zero query is returned as zeros rather
/// than dividing by zero; it scores every vector equally, which is the honest
/// answer to a query carrying no information.
pub fn quantize_query(q: &[f32], d: usize) -> Result<Vec<i8>, SearchError> {
    if q.len() != d {
        return Err(SearchError::Dimension {
            found: q.len(),
            expected: d,
        });
    }
    if let Some(bad) = q.iter().position(|x| !x.is_finite()) {
        return Err(SearchError::NonFinite { at: bad });
    }
    let extent = q.iter().fold(0f32, |a, x| a.max(x.abs()));
    if extent == 0.0 {
        return Ok(vec![0i8; d]);
    }
    Ok(q.iter()
        .map(|x| ((x / extent) * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect())
}

/// An instrument that records nothing, for the un-instrumented path.
struct NoInstrument;

impl Instrument for NoInstrument {
    fn cycles(&self) -> u64 {
        0
    }
    fn mark(&mut self, _phase: sector_hal::Phase, _edge: sector_hal::Edge) {}
}

/// Why a search could not be performed.
#[derive(Debug)]
pub enum SearchError {
    /// The volume could not be opened.
    Backend(Error),
    /// The volume could not be mounted.
    Mount(MountError),
    /// The query's dimension does not match the volume's.
    Dimension {
        /// Components supplied.
        found: usize,
        /// Components the volume needs.
        expected: usize,
    },
    /// A query component was NaN or infinite.
    ///
    /// Refused rather than clamped: a NaN would quantize to zero and silently
    /// drop that component from the score.
    NonFinite {
        /// Index of the offending component.
        at: usize,
    },
    /// The requested candidate depth exceeds the engine's fixed buffers.
    DepthTooLarge {
        /// Depth requested.
        want: usize,
        /// Largest the engine supports.
        limit: usize,
    },
    /// The engine refused the query.
    Query(String),
    /// An invariant this crate is responsible for did not hold.
    Internal(&'static str),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(e) => write!(f, "{e}"),
            Self::Mount(e) => write!(f, "{e}"),
            Self::Dimension { found, expected } => {
                write!(f, "query is D={found} but the volume is D={expected}")
            }
            Self::NonFinite { at } => write!(f, "query component {at} is not finite"),
            Self::DepthTooLarge { want, limit } => {
                write!(
                    f,
                    "candidate depth {want} exceeds the engine's limit of {limit}"
                )
            }
            Self::Query(e) => write!(f, "query failed: {e}"),
            Self::Internal(what) => write!(f, "internal error: {what}"),
        }
    }
}

impl std::error::Error for SearchError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::test_support::{build_image_and_corpus, TempDir};
    use crate::FileFlash;

    const D: usize = 32;
    const M: usize = 4;
    const N: usize = 300;

    fn volume(tag: &str) -> (TempDir, std::path::PathBuf, Vec<f32>) {
        let dir = TempDir::new(tag);
        let (image, corpus) = build_image_and_corpus(D, M, N);
        let path = dir.path().join("volume.sector");
        std::fs::write(&path, &image).expect("write");
        (dir, path, corpus)
    }

    #[test]
    fn a_searcher_answers_and_reports_what_it_cost() {
        let (_dir, path, corpus) = volume("search");
        let mut s: Searcher<FileFlash> = Searcher::open(&path, None).expect("open");
        assert_eq!(s.geometry().n, N);
        assert_eq!(s.backend_name(), "file");

        let a = s.search(&corpus[..D], 10).expect("search");
        assert_eq!(a.ids.len(), 10);
        assert_eq!(a.scores.len(), 10);
        assert_eq!(a.stats.scan.scanned as usize, N);
        assert_eq!(a.stats.rerank.dropped, 0, "clean volume dropped candidates");
        // Scores come back sorted descending.
        assert!(a.scores.windows(2).all(|w| w[0] >= w[1]), "{:?}", a.scores);
    }

    #[test]
    fn uniform_query_scaling_does_not_change_the_ranking() {
        // The property that licenses per-query quantization: scaling a query by a
        // positive constant scales every score by it, so the order is fixed.
        let (_dir, path, corpus) = volume("scaling");
        let mut s: Searcher<FileFlash> = Searcher::open(&path, None).expect("open");

        let q: Vec<f32> = corpus[..D].to_vec();
        let scaled: Vec<f32> = q.iter().map(|x| x * 37.5).collect();
        let a = s.search(&q, 10).expect("search");
        let b = s.search(&scaled, 10).expect("search scaled");
        assert_eq!(a.ids, b.ids, "ranking changed under uniform scaling");
    }

    #[test]
    fn a_dimension_mismatch_is_refused() {
        let (_dir, path, _) = volume("dim");
        let mut s: Searcher<FileFlash> = Searcher::open(&path, None).expect("open");
        let err = s.search(&[0.0; D + 1], 10).unwrap_err();
        assert!(matches!(err, SearchError::Dimension { .. }), "{err}");
    }

    #[test]
    fn a_non_finite_component_is_refused_rather_than_quantized_to_zero() {
        // A NaN clamps to zero, which would drop that component from every score
        // without saying so.
        let (_dir, path, _) = volume("nan");
        let mut q = vec![1.0f32; D];
        q[7] = f32::NAN;
        let mut s: Searcher<FileFlash> = Searcher::open(&path, None).expect("open");
        assert!(matches!(
            s.search(&q, 10).unwrap_err(),
            SearchError::NonFinite { at: 7 }
        ));
        q[7] = f32::INFINITY;
        assert!(matches!(
            s.search(&q, 10).unwrap_err(),
            SearchError::NonFinite { at: 7 }
        ));
    }

    #[test]
    fn an_all_zero_query_is_answered_rather_than_dividing_by_zero() {
        let (_dir, path, _) = volume("zero");
        let mut s: Searcher<FileFlash> = Searcher::open(&path, None).expect("open");
        let a = s.search(&[0.0; D], 5).expect("search");
        assert_eq!(a.ids.len(), 5);
        // Every score is zero, so the tie-break by id decides the order.
        assert!(a.scores.iter().all(|s| *s == 0), "{:?}", a.scores);
    }

    #[test]
    fn a_depth_beyond_the_engines_buffers_is_refused_not_clamped() {
        let (_dir, path, _) = volume("depth");
        // `Searcher` holds two open backends and is deliberately not `Debug`, so
        // the error is matched rather than unwrapped.
        match Searcher::<FileFlash>::open(&path, Some(sector_core::query::MAX_R + 1)) {
            Err(SearchError::DepthTooLarge { want, limit }) => {
                assert_eq!(want, sector_core::query::MAX_R + 1);
                assert_eq!(limit, sector_core::query::MAX_R);
            }
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("a depth beyond the engine's buffers was accepted"),
        }
    }

    #[test]
    fn resident_bytes_accounts_for_every_buffer() {
        let (_dir, path, _) = volume("resident");
        let s: Searcher<FileFlash> = Searcher::open(&path, Some(64)).expect("open");
        let g = *s.geometry();
        let expected = g.centroids * g.d          // codebook, int8
            + g.m * g.centroids * 4               // ADC table, i32
            + 64 * 4 + 64 * 4                     // heap scores and ids
            + g.d * 4                             // rotation scratch
            + sector_format::BLOCK_BYTES; // bounce
        assert_eq!(s.resident_bytes(), expected);
    }

    #[test]
    fn quantization_uses_the_full_i8_range() {
        // A query quantized into a fraction of the range throws away precision
        // the format paid for.
        let q: Vec<f32> = (0..8).map(|i| (i as f32) * 0.125).collect();
        let out = quantize_query(&q, 8).expect("quantize");
        assert_eq!(out.iter().copied().max(), Some(127));
        let neg: Vec<f32> = q.iter().map(|x| -x).collect();
        let out = quantize_query(&neg, 8).expect("quantize");
        assert_eq!(out.iter().copied().min(), Some(-127));
    }
}
