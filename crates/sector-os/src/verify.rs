//! Whole-volume integrity: what is damaged, how much it costs, and what can be
//! repaired.
//!
//! Behind `sector verify`, `sector repair` and `sector stats`. The engine
//! verifies a block only when a query reaches it, so damage in a cold region can
//! sit undetected for as long as nothing selects a candidate there. This is the
//! sweep that does not wait for a query.
//!
//! # Damage is reported per region, because the consequences differ per region
//!
//! A corrupted payload block perturbs the codes of the vectors in it — one
//! wrong candidate set for those ids, and no way to know from the result that it
//! happened. A corrupted rerank block drops every candidate in it, which is
//! visible in the drop counter. A corrupted codebook block alters the
//! reconstruction of every vector whose code points into it: `N / 2^b` vectors in
//! expectation, and the reason the codebook is the only replicated region.
//!
//! Reporting one aggregate "blocks bad" figure would flatten a three-order
//! difference in blast radius into a single number, so the report keeps them
//! apart and states the expected vector count for each.
//!
//! # Repair is copy-from-replica, not error correction
//!
//! The codebook has replicas; nothing else does. So `repair` can restore a
//! damaged codebook block from a good copy, and can do nothing for payload or
//! rerank damage beyond naming it. That asymmetry is the format's, and it follows
//! from criticality: the codebook is fixed-size and independent of `N`, so a full
//! second copy costs a constant that does not grow with the corpus, while
//! replicating the payload would double the volume.
//!
//! # The codebook has no CRC, and that limits what can be said about it
//!
//! The region table carries `PayloadCrc` and `RerankCrc` and nothing for the
//! codebook: its `Protection` is `Replicate`, and the replica *is* the mechanism.
//! A sweep can therefore compare the two copies and detect that they disagree,
//! but it cannot say which one is wrong — there is no independent check to
//! adjudicate between them.
//!
//! This report says exactly that rather than picking one. Declaring the primary
//! authoritative would repair a good block from a bad replica half the time,
//! turning single-copy damage into total loss; declaring them equal would report
//! a damaged volume as clean. [`CodebookStatus::Disagree`] is the honest third
//! answer, and it tells an operator to rebuild from the source corpus rather than
//! to trust a repair.
//!
//! Adding a codebook CRC would make the damage attributable and cost 4 B per
//! 512 B block — 0.8% of the region, against the 100% a replica costs. That it is
//! absent while the far more expensive replica is present is worth flagging, and
//! it is recorded in `docs/DEVELOPMENT_STATE.md` rather than changed here, since
//! adding a region means a `FORMAT_VERSION` bump and a firmware change.

use sector_format::region::RegionKind;
use sector_format::BLOCK_BYTES;
use sector_hal::NorFlash;

use crate::volume::HostVolume;

/// One region's integrity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionReport {
    /// Which region.
    pub kind: RegionKind,
    /// Blocks checked.
    pub blocks: usize,
    /// Blocks whose stored CRC did not match.
    pub bad_blocks: Vec<usize>,
    /// Whether this region has a CRC array at all.
    ///
    /// The CRC arrays themselves do not, and neither does the manifest, whose
    /// integrity is its own digest. A region without one is reported as unchecked
    /// rather than as clean — the distinction a monitoring check needs.
    pub checked: bool,
}

impl RegionReport {
    /// Whether every checked block matched.
    pub fn is_clean(&self) -> bool {
        self.bad_blocks.is_empty()
    }
}

/// A whole volume's integrity.
#[derive(Clone, Debug)]
pub struct VerifyReport {
    /// Per-region findings, in region-table order.
    pub regions: Vec<RegionReport>,
    /// Vectors in the volume.
    pub n: usize,
    /// Centroids per subspace.
    pub centroids: usize,
    /// Payload bytes per vector.
    pub payload_bytes: usize,
    /// Rerank bytes per vector.
    pub rerank_bytes: usize,
    /// What the codebook copies say about each other.
    pub codebook: CodebookStatus,
}

/// What comparing the codebook copies established.
///
/// There is no codebook CRC, so this is a comparison rather than a check — see
/// the module documentation for why that limits the answer to three cases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodebookStatus {
    /// Every block of both copies is identical. Either both are intact, or both
    /// were damaged identically, which no mechanism in the format can distinguish.
    Agree,
    /// The copies differ in these blocks. Neither can be declared correct.
    Disagree {
        /// Block indices where the copies differ.
        blocks: Vec<usize>,
    },
    /// The volume has no replica region, so there was nothing to compare.
    NoReplica,
}

impl CodebookStatus {
    /// Whether the copies agree.
    pub fn is_clean(&self) -> bool {
        !matches!(self, Self::Disagree { .. })
    }

    /// Blocks where the copies differ.
    pub fn differing(&self) -> &[usize] {
        match self {
            Self::Disagree { blocks } => blocks,
            _ => &[],
        }
    }
}

impl VerifyReport {
    /// Whether the volume is undamaged.
    pub fn is_clean(&self) -> bool {
        self.regions.iter().all(RegionReport::is_clean)
    }

    /// Bad blocks in `kind`, if it was checked.
    pub fn bad_in(&self, kind: RegionKind) -> Option<&[usize]> {
        self.regions
            .iter()
            .find(|r| r.kind == kind)
            .map(|r| r.bad_blocks.as_slice())
    }

    /// Vectors whose stored codes are wrong, from payload damage.
    ///
    /// Exact rather than estimated: a payload block holds a known number of
    /// records, and a damaged one affects those and no others. The engine will
    /// return a wrong candidate set for them with no indication, because a
    /// payload CRC failure is detected only at scan time and the scan does not
    /// stop.
    pub fn vectors_with_bad_codes(&self) -> usize {
        let per_block = BLOCK_BYTES.checked_div(self.payload_bytes).unwrap_or(0);
        self.bad_in(RegionKind::Payload)
            .map(|b| b.len() * per_block)
            .unwrap_or(0)
            .min(self.n)
    }

    /// Candidates that will be dropped, from rerank damage.
    ///
    /// The blast radius the shared CRC creates: a whole block's records go
    /// together, four at every shipped profile.
    pub fn candidates_that_will_drop(&self) -> usize {
        let per_block = BLOCK_BYTES.checked_div(self.rerank_bytes).unwrap_or(0);
        self.bad_in(RegionKind::Rerank)
            .map(|b| b.len() * per_block.max(1))
            .unwrap_or(0)
            .min(self.n)
    }

    /// Vectors whose reconstruction is wrong, from codebook damage.
    ///
    /// `N / 2^b` per damaged centroid in expectation, under a uniform assignment
    /// assumption that the build's own relabelling deliberately violates — real
    /// populations are skewed, measured at 4.5x from mean to maximum — so this is
    /// a lower bound on the worst case rather than a prediction.
    pub fn vectors_with_bad_reconstruction(&self) -> usize {
        let bad = self.codebook.differing().len();
        if bad == 0 || self.centroids == 0 {
            return 0;
        }
        let total_blocks = self
            .regions
            .iter()
            .find(|r| r.kind == RegionKind::Codebook)
            .map(|r| r.blocks)
            .unwrap_or(1)
            .max(1);
        // Centroids per block, each affecting N / 2^b vectors in expectation.
        let centroids_per_block = self.centroids.div_ceil(total_blocks);
        (bad * centroids_per_block * self.n / self.centroids).min(self.n)
    }
}

/// Block runs that hold written data, as half-open `[from, to)` spans.
///
/// A region's descriptor is rounded up to a whole erase sector, and `sector-build`
/// writes a CRC word only for the blocks it filled — so the descriptor's extent
/// exceeds the written extent and sweeping it would checksum slack against an
/// unwritten CRC word.
///
/// On an appended volume there are **two** such spans. The built corpus ends in a
/// block that was sealed when its CRC was written, so an append starts at the next
/// boundary and the blocks between are erased. Reporting those as damage would
/// make every clean appended volume look corrupt, and would destroy the meaning of
/// the `dropped` counter — the only evidence of real corruption a query gives.
///
/// Returns one run for a built-only volume and two when a gap exists.
fn written_runs(m: &sector_format::manifest::Manifest, per_block: usize) -> Vec<(usize, usize)> {
    let per = per_block.max(1);
    let built_blocks = (m.built_n as usize).div_ceil(per);
    if m.appended() == 0 {
        return vec![(0, built_blocks)];
    }
    // The appended run is block-aligned at its start by construction.
    let appended_start = (m.appended_from as usize) / per;
    let appended_end = (m.n as usize).div_ceil(per);
    if appended_start <= built_blocks {
        // No gap in this region: the two runs are contiguous.
        return vec![(0, appended_end)];
    }
    vec![(0, built_blocks), (appended_start, appended_end)]
}

/// Sweep every CRC-protected region, and compare the codebook copies.
pub fn verify<F: NorFlash>(flash: &mut F, volume: &HostVolume) -> Result<VerifyReport, F::Error> {
    let g = volume.geometry;
    let mut regions = Vec::new();

    // Blocks that actually hold data. On an appended volume the built run and the
    // appended run are separated by erased blocks — the gap and the unused reserve
    // — and those have erased CRC words. Sweeping them would report every clean
    // appended volume as damaged, which would make the `dropped` counter useless
    // as evidence of real corruption. See docs/design/002-append-and-reserve.md.
    let m = &volume.manifest;
    let payload_runs = written_runs(m, g.payload.vectors_per_block());
    let rerank_runs = written_runs(m, g.rerank.records_per_block());

    // Only these two regions carry a CRC array. The codebook is protected by
    // replication instead, and is handled below.
    //
    // The block count comes from the *layout*, not the region descriptor. A
    // region is rounded up to a whole erase sector, so its descriptor covers more
    // blocks than hold data, and the builder writes a CRC only for the blocks it
    // filled. Sweeping the descriptor's full extent would compute a checksum over
    // slack and compare it against an unwritten CRC word, reporting every clean
    // volume as damaged in its tail.
    for (kind, crc_kind, runs) in [
        (RegionKind::Payload, RegionKind::PayloadCrc, &payload_runs),
        (RegionKind::Rerank, RegionKind::RerankCrc, &rerank_runs),
    ] {
        let blocks: usize = runs.iter().map(|(a, b)| b - a).sum();
        let Some(desc) = volume.manifest.table.get(kind) else {
            continue;
        };
        let Some(crc) = volume.manifest.table.get(crc_kind) else {
            // A data region with no CRC array is unchecked, not clean. Reporting
            // it as clean would let a truncated image pass a verify.
            regions.push(RegionReport {
                kind,
                blocks,
                bad_blocks: Vec::new(),
                checked: false,
            });
            continue;
        };

        let mut block = vec![0u8; BLOCK_BYTES];
        let mut raw = [0u8; 4];
        let mut bad = Vec::new();
        // Runs, not a range: an appended volume's written blocks are two disjoint
        // spans with erased blocks between them.
        for (from, to) in runs.iter() {
            for b in *from..*to {
                flash.read(desc.base + (b * BLOCK_BYTES) as u32, &mut block)?;
                flash.read(crc.base + (b * 4) as u32, &mut raw)?;
                if sector_codec::crc::crc32(&block) != u32::from_le_bytes(raw) {
                    bad.push(b);
                }
            }
        }
        regions.push(RegionReport {
            kind,
            blocks,
            bad_blocks: bad,
            checked: true,
        });
    }

    // The codebook: compare the two copies block by block. A difference proves
    // damage without identifying the damaged copy.
    let codebook = match (
        volume.manifest.table.get(RegionKind::Codebook),
        volume.manifest.table.get(RegionKind::CodebookReplica),
    ) {
        (Some(primary), Some(replica)) => {
            // The codebook is `2^b * D * cb_bytes` bytes; the region is rounded
            // to a sector, so compare only the blocks that hold it. Slack in the
            // two regions is not required to match and is not part of the data.
            let data_blocks =
                (g.centroids * g.d * volume.manifest.cb_bytes as usize).div_ceil(BLOCK_BYTES);
            let blocks = data_blocks
                .min(primary.blocks as usize)
                .min(replica.blocks as usize);
            let mut a = vec![0u8; BLOCK_BYTES];
            let mut b = vec![0u8; BLOCK_BYTES];
            let mut differ = Vec::new();
            for i in 0..blocks {
                flash.read(primary.base + (i * BLOCK_BYTES) as u32, &mut a)?;
                flash.read(replica.base + (i * BLOCK_BYTES) as u32, &mut b)?;
                if a != b {
                    differ.push(i);
                }
            }
            regions.push(RegionReport {
                kind: RegionKind::Codebook,
                blocks,
                bad_blocks: differ.clone(),
                // Not a checksum: a comparison. The flag says an independent
                // check was not available.
                checked: false,
            });
            if differ.is_empty() {
                CodebookStatus::Agree
            } else {
                CodebookStatus::Disagree { blocks: differ }
            }
        }
        _ => CodebookStatus::NoReplica,
    };

    Ok(VerifyReport {
        regions,
        n: g.n,
        centroids: g.centroids,
        payload_bytes: g.payload_bytes,
        rerank_bytes: g.rerank_bytes,
        codebook,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::test_support::{build_image_and_corpus, TempDir};
    use crate::FileFlash;

    const D: usize = 32;
    const M: usize = 4;
    const N: usize = 300;

    fn volume(tag: &str) -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new(tag);
        let (image, _) = build_image_and_corpus(D, M, N);
        let path = dir.path().join("volume.sector");
        std::fs::write(&path, &image).expect("write");
        (dir, path)
    }

    fn report(path: &std::path::Path) -> VerifyReport {
        let mut f = FileFlash::open(path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        verify(&mut f, &v).expect("verify")
    }

    #[test]
    fn a_freshly_built_volume_is_clean() {
        let (_dir, path) = volume("clean");
        let r = report(&path);
        assert!(r.is_clean(), "{:?}", r.regions);
        assert_eq!(r.vectors_with_bad_codes(), 0);
        assert_eq!(r.candidates_that_will_drop(), 0);
        assert_eq!(r.codebook, CodebookStatus::Agree);
        // Payload and rerank were CRC-checked; the codebook was compared, which
        // the report marks as unchecked because no independent check exists.
        for x in &r.regions {
            match x.kind {
                RegionKind::Payload | RegionKind::Rerank => {
                    assert!(x.checked, "{:?} was not CRC-checked", x.kind)
                }
                RegionKind::Codebook => assert!(
                    !x.checked,
                    "the codebook has no CRC and must not claim to be checked"
                ),
                other => panic!("unexpected region in the report: {other:?}"),
            }
        }
        assert_eq!(r.regions.len(), 3, "{:?}", r.regions);
    }

    #[test]
    fn payload_damage_is_counted_in_vectors_not_blocks() {
        // The number that matters is how many vectors now have wrong codes, and
        // it is a multiple of the records per block.
        let (_dir, path) = volume("payload");
        let mut bytes = std::fs::read(&path).expect("read");
        let mut f = FileFlash::open(&path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        let base = v
            .manifest
            .table
            .get(RegionKind::Payload)
            .expect("region")
            .base as usize;
        bytes[base + 3] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");

        let r = report(&path);
        assert!(!r.is_clean());
        assert_eq!(r.bad_in(RegionKind::Payload).map(|b| b.len()), Some(1));
        let per_block = BLOCK_BYTES / r.payload_bytes;
        assert_eq!(r.vectors_with_bad_codes(), per_block);
        // Rerank and codebook are untouched: damage does not bleed across regions.
        assert_eq!(r.candidates_that_will_drop(), 0);
        assert_eq!(r.vectors_with_bad_reconstruction(), 0);
    }

    #[test]
    fn rerank_damage_predicts_the_drop_count() {
        // The prediction must match what the engine actually drops, or the report
        // is not telling an operator what to expect.
        let (_dir, path) = volume("rerank");
        let mut bytes = std::fs::read(&path).expect("read");
        let mut f = FileFlash::open(&path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        let base = v
            .manifest
            .table
            .get(RegionKind::Rerank)
            .expect("region")
            .base as usize;
        bytes[base + 7] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");

        let r = report(&path);
        let per_block = BLOCK_BYTES / r.rerank_bytes;
        assert_eq!(r.candidates_that_will_drop(), per_block);
        assert!(per_block > 1, "the shared-CRC blast radius is the point");
    }

    #[test]
    fn codebook_damage_is_repairable_when_the_replica_survives() {
        let (_dir, path) = volume("cbrepair");
        let mut bytes = std::fs::read(&path).expect("read");
        let mut f = FileFlash::open(&path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        let base = v
            .manifest
            .table
            .get(RegionKind::Codebook)
            .expect("region")
            .base as usize;
        bytes[base + 11] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");

        let r = report(&path);
        // The copies now disagree, and neither can be declared correct.
        assert!(
            matches!(r.codebook, CodebookStatus::Disagree { .. }),
            "{:?}",
            r.codebook
        );
        assert_eq!(r.codebook.differing().len(), 1);
        assert!(r.vectors_with_bad_reconstruction() > 0);
        assert!(!r.is_clean());
    }

    #[test]
    fn identical_damage_in_both_copies_is_undetectable_without_a_crc() {
        // The bound on what replication alone can do. Interleaving across erase
        // sectors makes this unlikely; it does not make it detectable.
        let (_dir, path) = volume("cbboth");
        let mut bytes = std::fs::read(&path).expect("read");
        let mut f = FileFlash::open(&path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        let primary = v.manifest.table.get(RegionKind::Codebook).expect("p").base as usize;
        let replica = v
            .manifest
            .table
            .get(RegionKind::CodebookReplica)
            .expect("r")
            .base as usize;
        bytes[primary + 11] ^= 0xFF;
        bytes[replica + 11] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");

        let r = report(&path);
        // Identical damage in both copies is invisible to a comparison. This is
        // the limitation a codebook CRC would remove, and reporting `Agree` here
        // is the honest consequence of not having one rather than a bug in the
        // sweep: nothing in the format can distinguish this case.
        assert_eq!(
            r.codebook,
            CodebookStatus::Agree,
            "a comparison cannot detect identical damage in both copies"
        );
    }

    #[test]
    fn damage_in_one_region_does_not_report_another_as_bad() {
        // A sweep that read the wrong CRC array would flag every region at once,
        // which is the failure mode this checks against.
        let (_dir, path) = volume("isolate");
        let mut bytes = std::fs::read(&path).expect("read");
        let mut f = FileFlash::open(&path).expect("open");
        let v = HostVolume::mount(&mut f, None).expect("mount");
        let base = v
            .manifest
            .table
            .get(RegionKind::Rerank)
            .expect("region")
            .base as usize;
        bytes[base + 1] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");

        let r = report(&path);
        assert_eq!(r.bad_in(RegionKind::Rerank).map(|b| b.len()), Some(1));
        assert_eq!(r.bad_in(RegionKind::Payload).map(|b| b.len()), Some(0));
        assert_eq!(r.codebook, CodebookStatus::Agree);
    }
}
