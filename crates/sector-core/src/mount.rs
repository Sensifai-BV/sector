//! Volume mount: manifest verification, region binding, XIP window probe.
//!
//! Mount probes the backend's execute-in-place window and binds each region to
//! either a borrow or a buffered read, so the query path never branches on
//! backend kind.
//!
//! # Mount order
//!
//! Verify the manifest before trusting any field in it. Refuse a profile the
//! device cannot host rather than degrading: a `b = 8` codebook at `D = 768` is
//! the entire T0 RAM budget, and the refusal belongs at mount rather than
//! mid-query.
//!
//! Probe the XIP window rather than inferring it from a feature flag. Whether
//! an address range is memory-mapped is a runtime property of the partition
//! layout, and a wrong assumption gives either a silent 100x fallback or a
//! borrow of unmapped memory.
//!
//! Report the bound path. NOR against managed NAND is the largest single factor
//! in query latency, and a measurement that cannot name the path it exercised
//! is not attributable.

use sector_format::manifest::{self, Manifest, ManifestError, MANIFEST_BYTES};
use sector_format::profile::Profile;
use sector_format::region::{RegionDesc, RegionKind};
use sector_hal::{NorFlash, Xip};

/// Why a mount was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MountError {
    /// Neither manifest slot verified, or the image is malformed.
    Manifest(ManifestError),
    /// A backend read failed.
    Backend,
    /// The image declares parameters this device cannot host.
    ///
    /// Refused here rather than degraded: a `b = 8` codebook at `D = 768` is
    /// the entire T0 RAM budget, and discovering that mid-query is worse than
    /// discovering it at mount.
    ProfileMismatch {
        /// Field that disagrees.
        field: ProfileField,
        /// Value the image declares.
        image: u32,
        /// Value the device is built for.
        device: u32,
    },
    /// A region named by the manifest is absent.
    MissingRegion {
        /// The absent region.
        kind: RegionKind,
    },
}

/// Names the profile field a mismatch was found in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileField {
    /// Vector dimension.
    Dimension,
    /// PQ subspaces.
    Subspaces,
    /// Bits per code.
    CodeBits,
    /// Bytes per codebook component.
    CodebookBytes,
    /// Rerank candidate depth.
    RerankDepth,
}

/// How a region's bytes reach the engine.
///
/// Bound once at mount so the query path never branches on backend kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// Borrowed in place from the memory-mapped window: no copies, no I/O
    /// calls in the hot loop.
    Borrowed,
    /// Read through a bounce buffer. The path the larger tiers use, and the
    /// one nothing exercises by accident.
    Buffered,
}

/// A mounted volume: verified manifest, bound regions, known access path.
#[derive(Clone, Copy, Debug)]
pub struct Volume {
    /// The verified manifest.
    pub manifest: Manifest,
    /// How payload bytes are reached.
    pub payload_binding: Binding,
    /// How rerank bytes are reached.
    pub rerank_binding: Binding,
    /// How codebook bytes are reached.
    pub codebook_binding: Binding,
}

impl Volume {
    /// The descriptor for `kind`, or a `MissingRegion` error.
    pub fn region(&self, kind: RegionKind) -> Result<&RegionDesc, MountError> {
        self.manifest
            .table
            .get(kind)
            .ok_or(MountError::MissingRegion { kind })
    }

    /// Whether the steady-state query performs no copies.
    ///
    /// True only when payload and rerank both borrow. Reported rather than
    /// inferred: NOR against managed NAND is the largest single factor in query
    /// latency, and a measurement that cannot name the path it exercised is not
    /// attributable to one.
    pub const fn is_zero_copy(&self) -> bool {
        matches!(self.payload_binding, Binding::Borrowed)
            && matches!(self.rerank_binding, Binding::Borrowed)
    }
}

/// Check the image's parameters against the device's profile.
///
/// Verified before any region is touched: a manifest that verifies its digest
/// still describes an image this device may be unable to host.
fn check_profile(m: &Manifest, p: &Profile) -> Result<(), MountError> {
    let checks = [
        (ProfileField::Dimension, m.d as u32, p.d as u32),
        (ProfileField::Subspaces, m.m as u32, p.m as u32),
        (ProfileField::CodeBits, m.b as u32, p.b as u32),
        (
            ProfileField::CodebookBytes,
            m.cb_bytes as u32,
            p.cb_bytes as u32,
        ),
    ];
    for (field, image, device) in checks {
        if image != device {
            return Err(MountError::ProfileMismatch {
                field,
                image,
                device,
            });
        }
    }
    // A deeper candidate list than the workspace heap can hold would silently
    // truncate stage two, so it is a refusal rather than a clamp.
    if m.r as usize > p.r {
        return Err(MountError::ProfileMismatch {
            field: ProfileField::RerankDepth,
            image: m.r,
            device: p.r as u32,
        });
    }
    Ok(())
}

/// Read both manifest slots and select the live one.
fn read_manifest<F: NorFlash>(
    flash: &mut F,
    slot_a: &mut [u8],
    slot_b: &mut [u8],
) -> Result<Manifest, MountError> {
    flash
        .read(manifest::SLOT_A_OFFSET, slot_a)
        .map_err(|_| MountError::Backend)?;
    flash
        .read(manifest::SLOT_B_OFFSET, slot_b)
        .map_err(|_| MountError::Backend)?;
    manifest::select(slot_a, slot_b).map_err(MountError::Manifest)
}

/// Mount a volume from a backend with no memory-mapped window.
///
/// Every region binds to the buffered path.
pub fn mount<F: NorFlash>(
    flash: &mut F,
    profile: &Profile,
    slot_a: &mut [u8; MANIFEST_BYTES],
    slot_b: &mut [u8; MANIFEST_BYTES],
) -> Result<Volume, MountError> {
    let manifest = read_manifest(flash, slot_a, slot_b)?;
    check_profile(&manifest, profile)?;
    let volume = Volume {
        manifest,
        payload_binding: Binding::Buffered,
        rerank_binding: Binding::Buffered,
        codebook_binding: Binding::Buffered,
    };
    // Fail here rather than at first query if a region the engine needs is
    // absent from the table.
    for kind in [
        RegionKind::Codebook,
        RegionKind::Payload,
        RegionKind::PayloadCrc,
        RegionKind::Rerank,
        RegionKind::RerankCrc,
    ] {
        volume.region(kind)?;
    }
    Ok(volume)
}

/// Mount a volume from a backend with a memory-mapped window.
///
/// Each region is probed against the window rather than assumed mapped.
/// Whether an address range is memory-mapped is a runtime property of the
/// partition layout, and a wrong assumption gives either a silent fallback at
/// roughly 100x the cost or a borrow of unmapped memory.
pub fn mount_xip<F: NorFlash + Xip>(
    flash: &mut F,
    profile: &Profile,
    slot_a: &mut [u8; MANIFEST_BYTES],
    slot_b: &mut [u8; MANIFEST_BYTES],
) -> Result<Volume, MountError> {
    let manifest = read_manifest(flash, slot_a, slot_b)?;
    check_profile(&manifest, profile)?;

    let probe = |kind: RegionKind| -> Result<Binding, MountError> {
        let r = manifest
            .table
            .get(kind)
            .ok_or(MountError::MissingRegion { kind })?;
        let len = usize::try_from(r.byte_len()).map_err(|_| MountError::Backend)?;
        Ok(match flash.window(r.base, len) {
            Some(_) => Binding::Borrowed,
            None => Binding::Buffered,
        })
    };

    let volume = Volume {
        manifest,
        payload_binding: probe(RegionKind::Payload)?,
        rerank_binding: probe(RegionKind::Rerank)?,
        codebook_binding: probe(RegionKind::Codebook)?,
    };
    for kind in [RegionKind::PayloadCrc, RegionKind::RerankCrc] {
        volume.region(kind)?;
    }
    Ok(volume)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sector_format::profile::T0;
    use sector_format::region::{Protection, RegionTable, REGION_COUNT};
    use sector_format::{BLOCK_BYTES, SECTOR_BYTES};

    const IMAGE_BYTES: usize = 64 * 1024;

    /// A test backend over a fixed byte array, with an optional mapped window.
    struct TestFlash {
        bytes: [u8; IMAGE_BYTES],
        window: Option<(u32, usize)>,
        reads: usize,
    }

    impl TestFlash {
        fn new(window: Option<(u32, usize)>) -> Self {
            Self {
                bytes: [0xFF; IMAGE_BYTES],
                window,
                reads: 0,
            }
        }
    }

    impl NorFlash for TestFlash {
        type Error = ();
        fn page_size(&self) -> usize {
            256
        }
        fn sector_size(&self) -> usize {
            SECTOR_BYTES
        }
        fn capacity(&self) -> u32 {
            IMAGE_BYTES as u32
        }
        fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), ()> {
            self.reads += 1;
            let start = addr as usize;
            let src = self.bytes.get(start..start + buf.len()).ok_or(())?;
            buf.copy_from_slice(src);
            Ok(())
        }
        fn program(&mut self, addr: u32, buf: &[u8]) -> Result<(), ()> {
            let start = addr as usize;
            let dst = self.bytes.get_mut(start..start + buf.len()).ok_or(())?;
            dst.copy_from_slice(buf);
            Ok(())
        }
        fn erase(&mut self, sector_addr: u32) -> Result<(), ()> {
            let start = sector_addr as usize;
            let dst = self.bytes.get_mut(start..start + SECTOR_BYTES).ok_or(())?;
            dst.fill(0xFF);
            Ok(())
        }
    }

    impl Xip for TestFlash {
        fn window(&self, addr: u32, len: usize) -> Option<&[u8]> {
            let (base, size) = self.window?;
            let start = addr.checked_sub(base)? as usize;
            if start + len > size {
                return None;
            }
            self.bytes.get(addr as usize..addr as usize + len)
        }
    }

    fn table() -> RegionTable {
        let kinds = [
            RegionKind::Codebook,
            RegionKind::CodebookReplica,
            RegionKind::Payload,
            RegionKind::PayloadCrc,
            RegionKind::Rerank,
            RegionKind::RerankCrc,
        ];
        let mut regions = [RegionDesc {
            kind: RegionKind::Codebook,
            protection: Protection::Replicate,
            base: 0,
            block_bytes: BLOCK_BYTES as u32,
            blocks: 8,
        }; REGION_COUNT];
        for (i, slot) in regions.iter_mut().enumerate() {
            slot.kind = kinds[i];
            slot.base = manifest::MANIFEST_RESERVED_BYTES + (i as u32) * SECTOR_BYTES as u32;
        }
        RegionTable { regions }
    }

    fn manifest_for(profile: &Profile, sequence: u64) -> Manifest {
        Manifest {
            sequence,
            d: profile.d as u16,
            m: profile.m as u16,
            b: profile.b as u16,
            cb_bytes: profile.cb_bytes as u16,
            n: 1_000,
            r: profile.r as u32,
            table: table(),
        }
    }

    fn install(flash: &mut TestFlash, m: &Manifest, slot: u32) {
        let mut buf = [0u8; MANIFEST_BYTES];
        m.encode(&mut buf).unwrap();
        flash.program(slot, &buf).unwrap();
    }

    #[test]
    fn a_valid_image_mounts_buffered_without_a_window() {
        let mut flash = TestFlash::new(None);
        install(&mut flash, &manifest_for(&T0, 1), manifest::SLOT_A_OFFSET);

        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        let v = mount(&mut flash, &T0, &mut a, &mut b).expect("mount");
        assert_eq!(v.payload_binding, Binding::Buffered);
        assert!(!v.is_zero_copy());
        assert_eq!(v.manifest.sequence, 1);
    }

    #[test]
    fn a_mapped_window_binds_the_borrowed_path() {
        // Window covering the whole image.
        let mut flash = TestFlash::new(Some((0, IMAGE_BYTES)));
        install(&mut flash, &manifest_for(&T0, 3), manifest::SLOT_B_OFFSET);

        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        let v = mount_xip(&mut flash, &T0, &mut a, &mut b).expect("mount");
        assert_eq!(v.payload_binding, Binding::Borrowed);
        assert_eq!(v.rerank_binding, Binding::Borrowed);
        assert!(v.is_zero_copy());
    }

    #[test]
    fn a_partial_window_binds_each_region_on_its_own_evidence() {
        // Window covers only the first three regions.
        let cut = manifest::MANIFEST_RESERVED_BYTES as usize + 3 * SECTOR_BYTES;
        let mut flash = TestFlash::new(Some((0, cut)));
        install(&mut flash, &manifest_for(&T0, 1), manifest::SLOT_A_OFFSET);

        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        let v = mount_xip(&mut flash, &T0, &mut a, &mut b).expect("mount");
        // Codebook and payload are inside; rerank is not.
        assert_eq!(v.codebook_binding, Binding::Borrowed);
        assert_eq!(v.payload_binding, Binding::Borrowed);
        assert_eq!(v.rerank_binding, Binding::Buffered);
        assert!(!v.is_zero_copy(), "a partial window is not zero-copy");
    }

    #[test]
    fn a_profile_the_device_cannot_host_is_refused_at_mount() {
        let mut flash = TestFlash::new(None);
        let mut m = manifest_for(&T0, 1);
        m.d = 768; // the configuration whose codebook is the whole budget
        install(&mut flash, &m, manifest::SLOT_A_OFFSET);

        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        assert_eq!(
            mount(&mut flash, &T0, &mut a, &mut b).err(),
            Some(MountError::ProfileMismatch {
                field: ProfileField::Dimension,
                image: 768,
                device: 128,
            })
        );
    }

    #[test]
    fn a_deeper_candidate_list_than_the_heap_is_refused_not_clamped() {
        let mut flash = TestFlash::new(None);
        let mut m = manifest_for(&T0, 1);
        m.r = T0.r as u32 + 1;
        install(&mut flash, &m, manifest::SLOT_A_OFFSET);

        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        assert!(matches!(
            mount(&mut flash, &T0, &mut a, &mut b),
            Err(MountError::ProfileMismatch {
                field: ProfileField::RerankDepth,
                ..
            })
        ));
    }

    #[test]
    fn an_erased_device_refuses_rather_than_mounting_empty() {
        let mut flash = TestFlash::new(None);
        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        assert_eq!(
            mount(&mut flash, &T0, &mut a, &mut b).err(),
            Some(MountError::Manifest(ManifestError::NoValidSlot))
        );
    }

    #[test]
    fn a_torn_install_mounts_the_previous_image() {
        let mut flash = TestFlash::new(None);
        install(&mut flash, &manifest_for(&T0, 4), manifest::SLOT_A_OFFSET);
        install(&mut flash, &manifest_for(&T0, 5), manifest::SLOT_B_OFFSET);

        // Interrupt slot B mid-write: valid prefix, erased tail.
        flash.bytes[manifest::SLOT_B_OFFSET as usize + 64
            ..manifest::SLOT_B_OFFSET as usize + MANIFEST_BYTES]
            .fill(0xFF);

        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        let v = mount(&mut flash, &T0, &mut a, &mut b).expect("falls back");
        assert_eq!(v.manifest.sequence, 4);
    }

    #[test]
    fn mount_reads_both_slots_exactly_once() {
        // Mount cost is fixed and small: the manifest is the only object at a
        // known address, and nothing else is read to establish the layout.
        let mut flash = TestFlash::new(None);
        install(&mut flash, &manifest_for(&T0, 1), manifest::SLOT_A_OFFSET);
        let mut a = [0u8; MANIFEST_BYTES];
        let mut b = [0u8; MANIFEST_BYTES];
        mount(&mut flash, &T0, &mut a, &mut b).unwrap();
        assert_eq!(flash.reads, 2);
    }
}
