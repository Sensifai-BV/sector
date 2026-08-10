//! Volume geometry, resolved once at mount.
//!
//! The engine's query call needs the region bases, the block layouts and the
//! codebook as borrowed slices. Computing those per query would repeat the same
//! division for every candidate, so they are resolved here and the result is
//! immutable for the volume's lifetime.
//!
//! # Why the codebook is copied and the rest is not
//!
//! The codebook is resident by design at every tier — `2^b * D` bytes,
//! independent of `N` — and the engine borrows it for the whole ADC table build.
//! It is read once at mount and held.
//!
//! Payload and rerank bytes are not copied. They are reached through
//! [`crate::source`] adapters, which is what lets the buffered and mapped
//! backends differ in cost while the engine's code path is identical.

use sector_format::manifest::{Manifest, MANIFEST_BYTES};
use sector_format::payload_blk::PayloadLayout;
use sector_format::profile::Profile;
use sector_format::region::{RegionDesc, RegionKind};
use sector_format::rerank_blk::RerankLayout;
use sector_hal::NorFlash;

use crate::Error;

/// Everything about a volume's shape that the query path needs.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Vector dimension.
    pub d: usize,
    /// PQ subspaces.
    pub m: usize,
    /// Bits per code.
    pub b: usize,
    /// Centroids per subspace, `2^b`.
    pub centroids: usize,
    /// Subspace dimension, `d / m`.
    pub ds: usize,
    /// Vectors stored.
    pub n: usize,
    /// Candidate depth the image was built for.
    pub r: usize,
    /// Payload bytes per vector.
    pub payload_bytes: usize,
    /// Rerank bytes per vector.
    pub rerank_bytes: usize,
    /// Payload block placement.
    pub payload: PayloadLayout,
    /// Rerank block placement.
    pub rerank: RerankLayout,
}

impl Geometry {
    /// Flash bytes a stored vector costs, in hundredths of a byte.
    ///
    /// Payload record, rerank record, and this vector's share of the two CRC
    /// arrays. Hundredths because the CRC share is genuinely fractional — a 512 B
    /// block carries one 4 B CRC and holds several records, so the per-vector share
    /// is `4 / records_per_block` and is rarely an integer. At `D = 128, m = 32`
    /// the exact figure is 161.25 B; rounding it to 161 in the calculation would
    /// misreport a million-vector corpus by 250 kB.
    ///
    /// # Why this lives here rather than in each command
    ///
    /// It was computed twice, and the two disagreed. `inspect` added a hardcoded
    /// `+ 8` for "the CRC share", which is wrong at every shipped profile — the
    /// true share is 1.25 B at `m = 32` and 1.125 B at `m = 16` — and `stats`
    /// omitted the CRC arrays entirely. So the same volume was reported as 168 B
    /// and 160 B per vector by two commands in the same tool, and the correct
    /// answer was neither.
    ///
    /// Reporting a figure twice from two derivations is the defect; one function
    /// both callers use is the fix.
    ///
    /// # Marginal, not amortised
    ///
    /// This is what one *more* vector costs, which is the figure for sizing a
    /// corpus. It is deliberately not the region bytes divided by `N`: a volume
    /// built with `--reserve` sizes its regions for the reserve too, so that
    /// division charges the reserve to the stored vectors and overstates them —
    /// 194.56 B against a true 161.25 B on a 20,000-vector volume reserving 4,096.
    pub const fn stored_bytes_per_vector_centi(&self) -> usize {
        let vpb = self.payload.vectors_per_block();
        let rpb = self.rerank.records_per_block();
        let mut centi = (self.payload_bytes + self.rerank_bytes) * 100;
        // 4 B of CRC per block, shared by the records in it. A region with no
        // records per block (a record wider than a block) carries a whole CRC per
        // block it spans, which `blocks_per_record` already accounts for.
        // `checked_div` rather than a guarded `/`: a zero here means a record
        // wider than a block, whose CRC cost `blocks_per_record` already carries.
        centi += match 400usize.checked_div(vpb) {
            Some(v) => v,
            None => 0,
        };
        centi += match 400usize.checked_div(rpb) {
            Some(v) => v,
            None => 0,
        };
        centi
    }

    /// Flash bytes a stored vector costs, rounded down.
    ///
    /// [`Self::stored_bytes_per_vector_centi`] when the fraction matters.
    pub const fn stored_bytes_per_vector(&self) -> usize {
        self.stored_bytes_per_vector_centi() / 100
    }

    /// Derive geometry from a verified manifest.
    pub fn of(m: &Manifest) -> Self {
        let d = m.d as usize;
        let subspaces = m.m as usize;
        let b = m.b as usize;
        // The payload record is what the image actually stores per vector. At
        // b=8 that is one byte per subspace; at b=4 two codes share a byte.
        let payload_bytes = subspaces * b / 8;
        let rerank_bytes = d * m.cb_bytes as usize;
        Self {
            d,
            m: subspaces,
            b,
            centroids: 1usize << b,
            ds: d / subspaces.max(1),
            n: m.n as usize,
            r: m.r as usize,
            payload_bytes,
            rerank_bytes,
            payload: PayloadLayout::new(payload_bytes, m.n as usize),
            rerank: RerankLayout::new(rerank_bytes, m.n as usize),
        }
    }

    /// Codebook bytes, `2^b * D * s`.
    pub const fn codebook_bytes(&self) -> usize {
        self.centroids * self.d
    }
}

/// A mounted host volume: manifest, geometry, resident codebook, region bases.
///
/// Generic over the backend so the same type serves the buffered and mapped
/// paths; the backend is what differs in cost, not this.
pub struct HostVolume {
    /// The verified manifest.
    pub manifest: Manifest,
    /// Resolved geometry.
    pub geometry: Geometry,
    /// The resident codebook, as stored bytes.
    pub codebook: Vec<u8>,
    payload: RegionDesc,
    payload_crc: RegionDesc,
    rerank: RegionDesc,
    rerank_crc: RegionDesc,
}

impl HostVolume {
    /// Read the manifest, select the live slot, and resolve everything the query
    /// path needs.
    ///
    /// `profile` is checked against the image the same way the firmware checks
    /// it: an image this host cannot serve is refused at mount rather than
    /// failing per query. Passing `None` skips the check, which is what the CLI
    /// does when inspecting an arbitrary image.
    pub fn mount<F: NorFlash>(flash: &mut F, profile: Option<&Profile>) -> Result<Self, MountError>
    where
        F::Error: std::fmt::Debug,
    {
        let mut slot_a = [0u8; MANIFEST_BYTES];
        let mut slot_b = [0u8; MANIFEST_BYTES];

        // With no device profile to enforce, the check still has to run — it is
        // what validates the region table — so it runs against a profile read
        // from the image itself. `check_profile` compares for equality, so a
        // profile of zeros would reject every image rather than accept any.
        let permissive = match profile {
            Some(p) => *p,
            None => profile_from_image(flash, &mut slot_a, &mut slot_b)?,
        };
        let volume = sector_core::mount::mount(flash, &permissive, &mut slot_a, &mut slot_b)
            .map_err(MountError::Mount)?;

        let manifest = volume.manifest;
        let geometry = Geometry::of(&manifest);

        let region = |kind: RegionKind| -> Result<RegionDesc, MountError> {
            manifest
                .table
                .get(kind)
                .copied()
                .ok_or(MountError::MissingRegion(kind))
        };
        let cb = region(RegionKind::Codebook)?;
        let mut codebook = vec![0u8; geometry.codebook_bytes()];
        flash
            .read(cb.base, &mut codebook)
            .map_err(|e| MountError::Backend(format!("{e:?}")))?;

        Ok(Self {
            manifest,
            geometry,
            codebook,
            payload: region(RegionKind::Payload)?,
            payload_crc: region(RegionKind::PayloadCrc)?,
            rerank: region(RegionKind::Rerank)?,
            rerank_crc: region(RegionKind::RerankCrc)?,
        })
    }

    /// Base address of the payload region.
    pub const fn payload_base(&self) -> u32 {
        self.payload.base
    }
    /// Base address of the payload CRC array.
    pub const fn payload_crc_base(&self) -> u32 {
        self.payload_crc.base
    }
    /// Base address of the rerank region.
    pub const fn rerank_base(&self) -> u32 {
        self.rerank.base
    }
    /// Base address of the rerank CRC array.
    pub const fn rerank_crc_base(&self) -> u32 {
        self.rerank_crc.base
    }
}

/// Read the manifest and build the profile it describes.
///
/// `sector_core::mount` checks the image against a device profile, which is
/// correct on a device: an image the hardware cannot host must be refused rather
/// than mis-read. A host tool inspecting an arbitrary image has no such
/// constraint, and refusing an image because it was built for another tier would
/// make the tool useless for its main case.
///
/// So the parameters come from the image and the check passes by construction.
/// It is not a no-op: `mount` also validates the region table and rejects a
/// manifest whose digest fails, and both still apply.
///
/// The manifest is read twice — here and inside `mount`. Two reads of one erase
/// sector at startup is not worth an API that could hand `mount` a
/// pre-verified manifest and let the two paths diverge.
fn profile_from_image<F: NorFlash>(
    flash: &mut F,
    slot_a: &mut [u8; MANIFEST_BYTES],
    slot_b: &mut [u8; MANIFEST_BYTES],
) -> Result<Profile, MountError>
where
    F::Error: std::fmt::Debug,
{
    flash
        .read(sector_format::manifest::SLOT_A_OFFSET, slot_a)
        .map_err(|e| MountError::Backend(format!("{e:?}")))?;
    flash
        .read(sector_format::manifest::SLOT_B_OFFSET, slot_b)
        .map_err(|e| MountError::Backend(format!("{e:?}")))?;
    let m = sector_format::manifest::select(slot_a, slot_b)
        .map_err(|e| MountError::Mount(sector_core::mount::MountError::Manifest(e)))?;

    Ok(Profile {
        d: m.d as usize,
        m: m.m as usize,
        b: m.b as usize,
        cb_bytes: m.cb_bytes as usize,
        rerank_bytes: m.cb_bytes as usize,
        // Not stored in the manifest and not checked by `mount`. The values the
        // engine actually uses come from the workspace the caller binds.
        adc_bytes: 4,
        // `mount` refuses an image whose depth exceeds the device's, so the
        // permissive value is the image's own.
        r: m.r as usize,
        k: 0,
        ram_budget: 0,
        stack_reserve: 0,
    })
}

/// Why a volume could not be mounted.
#[derive(Debug)]
pub enum MountError {
    /// The engine's mount refused the image.
    Mount(sector_core::mount::MountError),
    /// A region the query path needs is absent from the table.
    MissingRegion(RegionKind),
    /// The backend failed while reading.
    Backend(String),
}

impl std::fmt::Display for MountError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mount(e) => write!(f, "mount refused the image: {e:?}"),
            Self::MissingRegion(k) => write!(f, "image has no {k:?} region"),
            Self::Backend(e) => write!(f, "backend error: {e}"),
        }
    }
}

impl std::error::Error for MountError {}

impl From<Error> for MountError {
    fn from(e: Error) -> Self {
        Self::Backend(e.to_string())
    }
}

/// Volume construction for tests, shared across this crate's modules.
///
/// Public because the integration tests and the CLI's round-trip test build the
/// same shape, and three copies of an image builder would drift.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    /// A temp directory that removes itself, so a failing test does not leave
    /// megabyte images behind.
    pub struct TempDir(std::path::PathBuf);

    impl TempDir {
        /// Create a uniquely named directory under the system temp path.
        pub fn new(tag: &str) -> Self {
            // Process id plus a counter: two tests in one binary must not
            // collide, and `std` has no temp-dir facility.
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEQ: AtomicU32 = AtomicU32::new(0);
            let n = SEQ.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("sector_os_{tag}_{}_{n}", std::process::id()));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }
        /// The directory's path.
        pub fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    /// Build a small but real volume image: trained codebook, encoded corpus,
    /// rerank copies, CRC arrays, manifest.
    ///
    /// Real rather than synthetic bytes because the tests assert that CRC
    /// verification passes and that two backends agree — both of which a
    /// hand-written byte pattern would satisfy without exercising the format.
    pub fn build_image(d: usize, m: usize, n: usize) -> Vec<u8> {
        let (image, _) = build_image_and_corpus(d, m, n);
        image
    }

    /// As [`build_image`], also returning the corpus so a test can compute a
    /// brute-force reference and check recall.
    pub fn build_image_and_corpus(d: usize, m: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
        build_image_inner(d, m, n, 0)
    }

    /// As [`build_image_and_corpus`], with room reserved for `reserve` appends.
    ///
    /// The append tests need a volume whose payload and rerank regions have erased
    /// blocks past the built corpus. `reserve` is in vector slots; the emitter
    /// rounds it up to whole blocks in both regions.
    pub fn build_reserved_image(
        d: usize,
        m: usize,
        n: usize,
        reserve: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        build_image_inner(d, m, n, reserve)
    }

    fn build_image_inner(d: usize, m: usize, n: usize, reserve: usize) -> (Vec<u8>, Vec<f32>) {
        use sector_build::emit::{emit, Image};
        use sector_build::encode::{encode, quantize};
        use sector_build::train::{train, TrainConfig};

        // # Why this corpus is isotropic across subspaces
        //
        // Every component is drawn from the same range, so each subspace's
        // extent — and therefore its quantization scale `num_j` — comes out
        // near-identical. That is deliberate, and it is working around a defect
        // rather than exercising the system fully.
        //
        // `sector_quant::adc::build_table` sums raw integer products of the
        // stored `i8` components and never consults `Codebook::scale()`. The
        // scales are per subspace (`sector_build::encode::quantize` takes each
        // subspace's own extent), so when subspaces differ in energy, subspace
        // `j` enters the score weighted by `num_j` and the ranking is on a
        // metric no one chose. Measured on a synthetic anisotropic corpus at
        // `D = 32, m = 4, b = 6, N = 2000`: recall@10 falls to 0.060 with
        // per-subspace scales against 0.178 for a single shared scale at 8x
        // energy spread, and 0.050 against 0.122 at 47x.
        //
        // The scales are also not stored in the image, so no reader can correct
        // for this after the fact. Fixing it means either one global scale or
        // scales in the manifest behind a `FORMAT_VERSION` bump, which is a
        // format decision affecting the firmware and the report's recall
        // figures — out of scope here and recorded in
        // `docs/DEVELOPMENT_STATE.md`.
        //
        // An isotropic corpus keeps the defect off the path these tests are
        // measuring (backend equivalence and end-to-end mount/scan/rerank),
        // which is what they exist to check.
        let mut corpus = vec![0f32; n * d];
        for v in 0..n {
            for j in 0..d {
                // Clustered rather than uniform so training produces populated
                // centroids instead of a degenerate codebook, and every
                // component shares one range.
                corpus[v * d + j] =
                    (((v * 37 + j * 11) % 89) as f32) * 0.75 + ((v % 7) as f32) * 9.0;
            }
        }

        let cfg = TrainConfig {
            d,
            m,
            b: 8,
            iterations: 6,
            seed: 0x5EC7,
        };
        let (books, _) = train(&corpus, n, cfg).expect("train");
        let quantized = quantize(&books, 127);
        let (codes, _) = encode(&corpus, n, d, &books);

        // Rerank copies must be quantized with the *same* per-subspace scale the
        // codebook uses. `sector_core::rerank::exact_score` takes the integer
        // inner product of the quantized query against these bytes, so a rerank
        // copy on a different scale would rank on a different metric than stage
        // one and the two stages would disagree by construction.
        let ds = d / m;
        let mut rerank = vec![0u8; n * d];
        for v in 0..n {
            for j in 0..m {
                let scale = quantized.scales[j];
                for i in 0..ds {
                    let x = corpus[v * d + j * ds + i];
                    let q = ((x / scale.num as f32) * scale.den as f32)
                        .round()
                        .clamp(i8::MIN as f32, i8::MAX as f32) as i8;
                    rerank[v * d + j * ds + i] = q as u8;
                }
            }
        }

        let image = Image {
            codebook: &quantized,
            codes: &codes,
            rerank: &rerank,
            n,
            d,
            r: 100,
            copies: 2,
            // No reserve: these images exercise the query path, and a reserved
            // build is a different geometry with its own tests in sector-build.
            reserve,
        };
        let mut out = Vec::new();
        emit(&image, &mut out).expect("emit");
        (out, corpus)
    }

    /// Write a small volume to a temp file, returning the directory guard, the
    /// path, and the image bytes.
    pub fn write_temp_volume(tag: &str) -> (TempDir, std::path::PathBuf, Vec<u8>) {
        let dir = TempDir::new(tag);
        let image = build_image(32, 4, 200);
        let path = dir.path().join("volume.sector");
        std::fs::write(&path, &image).expect("write volume");
        (dir, path, image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_format::payload_blk::PayloadLayout;
    use sector_format::rerank_blk::RerankLayout;

    fn geometry(m: usize, payload_bytes: usize) -> Geometry {
        Geometry {
            d: 128,
            m,
            b: 8,
            centroids: 256,
            ds: 128 / m,
            n: 1000,
            r: 500,
            payload_bytes,
            rerank_bytes: 128,
            payload: PayloadLayout::new(payload_bytes, 1000),
            rerank: RerankLayout::new(128, 1000),
        }
    }

    #[test]
    fn stored_bytes_counts_the_crc_share_exactly() {
        // The figure `inspect` and `stats` used to derive independently, and
        // disagree on: one added a hardcoded +8 for "the CRC share" and the other
        // omitted the CRC arrays entirely, so the same volume was reported as 168 B
        // and 160 B by two commands in one tool and neither was right.
        //
        // At D=128 m=32 b=8: payload 32 B (16 per block, so 0.25 B of CRC),
        // rerank 128 B (4 per block, so 1.00 B). 161.25 B.
        let g = geometry(32, 32);
        assert_eq!(g.stored_bytes_per_vector_centi(), 16125);
        assert_eq!(g.stored_bytes_per_vector(), 161);
        assert_ne!(
            g.stored_bytes_per_vector(),
            168,
            "the hardcoded +8 that shipped must not come back"
        );
    }

    #[test]
    fn the_crc_share_tracks_the_profile_rather_than_being_a_constant() {
        // Why a constant could not have been right at every profile: the share is
        // 4 B divided by the records in a block, so it moves with the record size.
        // A 16 B payload packs 32 per block and pays 0.125 B against a 32 B
        // payload's 0.25 B.
        let narrow = geometry(16, 16);
        let wide = geometry(32, 32);
        assert_eq!(narrow.stored_bytes_per_vector_centi(), 14512);
        assert_ne!(
            narrow.stored_bytes_per_vector_centi(),
            wide.stored_bytes_per_vector_centi(),
            "the CRC share must differ between profiles, or one constant would do"
        );
    }

    #[test]
    fn stored_bytes_is_marginal_not_amortised() {
        // It reports what one MORE vector costs, which is the figure for sizing a
        // corpus. Dividing a reserved volume's region bytes by N charges the
        // reserve to the stored vectors: measured at 194.56 B against a true
        // 161.25 B on a 20,000-vector volume reserving 4,096.
        let g = geometry(32, 32);
        let marginal = g.stored_bytes_per_vector_centi();
        // Doubling N must not change the per-vector figure.
        let mut bigger = g;
        bigger.n = 2000;
        bigger.payload = PayloadLayout::new(32, 2000);
        bigger.rerank = RerankLayout::new(128, 2000);
        assert_eq!(bigger.stored_bytes_per_vector_centi(), marginal);
    }
}
