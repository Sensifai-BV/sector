//! Board and ABI detection.
//!
//! What `sector doctor` reports, and what the daemon logs at startup. The
//! failure this exists to prevent is quiet: an ARMv7 binary installs cleanly on
//! a Pi Zero and dies with `SIGILL` at the first ARMv7-only instruction, which
//! may be inside a code path that does not run until a query arrives.
//!
//! # Detection order
//!
//! The board is identified from `/proc/cpuinfo`'s `Revision` field first and
//! `/proc/device-tree/model` second. The revision code is authoritative: it
//! encodes model and SoC in a documented bit layout, where the model string is
//! marketing text that does not distinguish a Pi 2 v1.1 (Cortex-A7, ARMv7) from
//! a v1.2 (Cortex-A53, ARMv8) — two boards that need different binaries.
//!
//! # The ABI is a compile-time fact, the board is a runtime one
//!
//! [`Abi::current`] reports what this binary was built for, read from `cfg`.
//! [`Board::detect`] reports what it is running on. Comparing the two is the
//! check that matters, and neither alone is sufficient.

/// The instruction-set baseline a binary was compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    /// ARMv6 + VFPv2, 32-bit. `arm-unknown-linux-*eabihf`.
    Armv6Hf,
    /// ARMv7-A + VFPv3, 32-bit. `armv7-unknown-linux-*eabihf`.
    Armv7Hf,
    /// ARMv8-A, 64-bit. `aarch64-unknown-linux-*`.
    Aarch64,
    /// 64-bit x86, for development hosts.
    X86_64,
    /// Anything else.
    Other,
}

impl Abi {
    /// The ABI this binary was compiled for.
    ///
    /// `target_feature = "v7"` is what separates the two 32-bit ARM triples:
    /// both report `target_arch = "arm"`, and the triple name is not visible to
    /// `cfg`.
    pub const fn current() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Aarch64
        }
        #[cfg(target_arch = "x86_64")]
        {
            Self::X86_64
        }
        #[cfg(all(target_arch = "arm", target_feature = "v7"))]
        {
            Self::Armv7Hf
        }
        #[cfg(all(target_arch = "arm", not(target_feature = "v7")))]
        {
            Self::Armv6Hf
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "arm")))]
        {
            Self::Other
        }
    }

    /// The release artifact that carries this ABI.
    pub const fn artifact(&self) -> &'static str {
        match self {
            Self::Armv6Hf => "sector-linux-armv6hf",
            Self::Armv7Hf => "sector-linux-armv7hf",
            Self::Aarch64 => "sector-linux-arm64",
            Self::X86_64 => "sector-linux-amd64",
            Self::Other => "(built from source)",
        }
    }

    /// Whether a binary of this ABI can execute on `board`.
    ///
    /// Downward compatibility only: an ARMv6 binary runs on every Pi, an ARMv7
    /// binary does not run on a Pi 1 or Zero, and an aarch64 binary needs both a
    /// 64-bit core and a 64-bit userland.
    pub const fn runs_on(&self, board: &Board) -> bool {
        match self {
            Self::Armv6Hf => board.arch.is_arm(),
            Self::Armv7Hf => matches!(board.arch, Arch::Armv7 | Arch::Armv8),
            Self::Aarch64 => matches!(board.arch, Arch::Armv8),
            Self::X86_64 => matches!(board.arch, Arch::Other),
            Self::Other => true,
        }
    }
}

impl core::fmt::Display for Abi {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Armv6Hf => "armv6 + vfpv2 (32-bit)",
            Self::Armv7Hf => "armv7-a + vfpv3 (32-bit)",
            Self::Aarch64 => "armv8-a (64-bit)",
            Self::X86_64 => "x86_64",
            Self::Other => "unknown",
        };
        f.write_str(s)
    }
}

/// The instruction-set generation of a board's CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// ARM1176, BCM2835. Pi 1, Zero, Zero W, Compute Module 1.
    Armv6,
    /// Cortex-A7, BCM2836. Pi 2 Model B v1.1 only.
    Armv7,
    /// Cortex-A53 or later. Everything from the Pi 3 onward, plus Pi 2 v1.2 and
    /// the Zero 2 W.
    Armv8,
    /// Not a Raspberry Pi.
    Other,
}

impl Arch {
    /// Whether this is any ARM generation.
    pub const fn is_arm(&self) -> bool {
        matches!(self, Self::Armv6 | Self::Armv7 | Self::Armv8)
    }

    /// The tier profile this generation maps to.
    ///
    /// ARMv6 and ARMv7 are T2, ARMv8 is T3 — see
    /// `docs/design/001-pi-tier-profiles.md` for why the split follows ISA
    /// generation, which is also where the cache facts change.
    pub const fn tier(&self) -> &'static str {
        match self {
            Self::Armv6 | Self::Armv7 => "T2",
            Self::Armv8 => "T3",
            Self::Other => "host",
        }
    }
}

/// What board this process is running on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    /// Marketing name from the device tree, when present.
    pub model: Option<String>,
    /// Raw revision code from `/proc/cpuinfo`, when present.
    pub revision: Option<u32>,
    /// SoC name decoded from the revision code.
    pub soc: Option<&'static str>,
    /// Instruction-set generation.
    pub arch: Arch,
    /// Whether the running userland is 64-bit.
    ///
    /// Distinct from `arch`: a Pi 4 running 32-bit Raspberry Pi OS is an ARMv8
    /// board that needs the ARMv7 binary, and this is the field that says so.
    pub userland_64bit: bool,
}

impl Board {
    /// Detect the board from procfs and the device tree.
    ///
    /// Never fails: an unreadable procfs yields a `Board` with `arch` set from
    /// the compile-time architecture, which is the best available answer on a
    /// non-Linux host or inside a container without `/proc`.
    pub fn detect() -> Self {
        let model = read_device_tree_model();
        let revision = read_revision();
        let (soc, revision_arch) = revision.map(decode_revision).unwrap_or((None, None));

        // Revision code first, model string second, compile-time last. The
        // revision distinguishes a Pi 2 v1.1 from a v1.2; the model string does
        // not.
        let arch = revision_arch
            .or_else(|| model.as_deref().and_then(arch_from_model))
            .unwrap_or(match Abi::current() {
                Abi::Armv6Hf => Arch::Armv6,
                Abi::Armv7Hf => Arch::Armv7,
                Abi::Aarch64 => Arch::Armv8,
                _ => Arch::Other,
            });

        Self {
            model,
            revision,
            soc,
            arch,
            userland_64bit: cfg!(target_pointer_width = "64"),
        }
    }

    /// Whether this is a recognised Raspberry Pi.
    pub const fn is_raspberry_pi(&self) -> bool {
        self.revision.is_some() && self.arch.is_arm()
    }

    /// The artifact a user should install on this board.
    ///
    /// The 32-bit answer depends on the userland, not the core: a Pi 4 running
    /// 32-bit Raspberry Pi OS takes the ARMv7 build even though its CPU is
    /// ARMv8.
    pub const fn recommended_artifact(&self) -> Abi {
        match self.arch {
            Arch::Armv6 => Abi::Armv6Hf,
            Arch::Armv7 => Abi::Armv7Hf,
            Arch::Armv8 => {
                if self.userland_64bit {
                    Abi::Aarch64
                } else {
                    Abi::Armv7Hf
                }
            }
            Arch::Other => Abi::current(),
        }
    }

    /// Whether the running binary matches what this board should run.
    ///
    /// A mismatch is not necessarily fatal — an ARMv6 binary on a Pi 5 runs
    /// correctly, just without the newer ISA — so this reports two states rather
    /// than one.
    pub const fn abi_status(&self) -> AbiStatus {
        let current = Abi::current();
        if !current.runs_on(self) {
            return AbiStatus::Incompatible;
        }
        // `Abi` has no const PartialEq, so compare discriminants by hand.
        if current as u8 == self.recommended_artifact() as u8 {
            AbiStatus::Match
        } else {
            AbiStatus::Suboptimal
        }
    }
}

/// How the running binary's ABI relates to the board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbiStatus {
    /// The binary is the recommended one for this board.
    Match,
    /// The binary runs but a better-matched artifact exists.
    Suboptimal,
    /// The binary cannot execute correctly on this board. On a 32-bit board
    /// running an ARMv7 build this means `SIGILL`, possibly not until a query
    /// reaches the offending instruction.
    Incompatible,
}

/// Read `/proc/device-tree/model`, trimming the trailing NUL the device tree
/// stores.
fn read_device_tree_model() -> Option<String> {
    let raw = std::fs::read("/proc/device-tree/model").ok()?;
    let text = String::from_utf8_lossy(&raw);
    let trimmed = text.trim_end_matches('\0').trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Read the `Revision` field from `/proc/cpuinfo`.
fn read_revision() -> Option<u32> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim() == "Revision" {
            return u32::from_str_radix(value.trim(), 16).ok();
        }
    }
    None
}

/// Decode a Raspberry Pi revision code into its SoC and ISA generation.
///
/// The new-style encoding (bit 23 set) packs the processor id in bits 12–15:
/// 0 = BCM2835 (ARM1176), 1 = BCM2836 (Cortex-A7), 2 = BCM2837 (Cortex-A53),
/// 3 = BCM2711 (Cortex-A72), 4 = BCM2712 (Cortex-A76).
///
/// Old-style codes predate the Pi 2 and are all BCM2835, so any code without
/// bit 23 is ARMv6.
fn decode_revision(revision: u32) -> (Option<&'static str>, Option<Arch>) {
    let new_style = revision & (1 << 23) != 0;
    if !new_style {
        return (Some("BCM2835"), Some(Arch::Armv6));
    }
    match (revision >> 12) & 0xF {
        0 => (Some("BCM2835"), Some(Arch::Armv6)),
        1 => (Some("BCM2836"), Some(Arch::Armv7)),
        2 => (Some("BCM2837"), Some(Arch::Armv8)),
        3 => (Some("BCM2711"), Some(Arch::Armv8)),
        4 => (Some("BCM2712"), Some(Arch::Armv8)),
        // An unrecognised processor id is a board newer than this table. Report
        // the ISA as unknown rather than guessing: a wrong guess here is a
        // SIGILL, and `None` falls through to the model string.
        _ => (None, None),
    }
}

/// Infer the ISA generation from a device-tree model string.
///
/// The fallback when no revision code is available. Deliberately conservative:
/// "Raspberry Pi 2" alone cannot distinguish v1.1 from v1.2, so it reports the
/// older of the two and the binary that runs on both.
fn arch_from_model(model: &str) -> Option<Arch> {
    let m = model.to_ascii_lowercase();
    if !m.contains("raspberry pi") {
        return None;
    }
    // Order matters: "Zero 2" must be tested before "Zero", and "Pi 2" must not
    // match "Pi 2 W"-style names that do not exist but would be mis-bucketed by
    // a looser test.
    if m.contains("zero 2") {
        return Some(Arch::Armv8);
    }
    if m.contains("zero") {
        return Some(Arch::Armv6);
    }
    for (needle, arch) in [
        ("pi 5", Arch::Armv8),
        ("pi 500", Arch::Armv8),
        ("pi 4", Arch::Armv8),
        ("pi 400", Arch::Armv8),
        ("pi 3", Arch::Armv8),
        ("compute module 5", Arch::Armv8),
        ("compute module 4", Arch::Armv8),
        ("compute module 3", Arch::Armv8),
        ("pi 2", Arch::Armv7),
        ("compute module 1", Arch::Armv6),
        ("model a", Arch::Armv6),
        ("model b", Arch::Armv6),
    ] {
        if m.contains(needle) {
            return Some(arch);
        }
    }
    None
}

/// The kernel's page size, from `sysconf(_SC_PAGESIZE)`.
///
/// Read rather than assumed: Raspberry Pi OS on Pi 5 uses 16 KiB pages where
/// every other Pi configuration uses 4 KiB. The mapped backend reports fault
/// granularity, and a hardcoded 4096 would misreport it by 4x on exactly one
/// popular configuration.
///
/// Falls back to 4096 when the call fails, which cannot happen for
/// `_SC_PAGESIZE` on Linux but is not worth a panic.
pub fn page_size() -> usize {
    // Declared here rather than taken as a `libc` dependency: this is the only
    // libc call in the crate, and the workspace holds no external dependencies.
    // `_SC_PAGESIZE` is 30 on Linux and 29 on macOS/BSD.
    #[cfg(target_os = "linux")]
    const SC_PAGESIZE: i32 = 30;
    #[cfg(not(target_os = "linux"))]
    const SC_PAGESIZE: i32 = 29;

    unsafe extern "C" {
        fn sysconf(name: i32) -> i64;
    }
    // SAFETY: `sysconf` is a pure query with no pointer arguments and no
    // side effects. It returns -1 on an unrecognised name, which is handled.
    let n = unsafe { sysconf(SC_PAGESIZE) };
    if n > 0 {
        n as usize
    } else {
        4096
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_style_revisions_are_all_armv6() {
        // Every board predating the Pi 2 is BCM2835. A 0x0002 is a Model B rev1.
        assert_eq!(
            decode_revision(0x0002),
            (Some("BCM2835"), Some(Arch::Armv6))
        );
        assert_eq!(
            decode_revision(0x000D),
            (Some("BCM2835"), Some(Arch::Armv6))
        );
    }

    #[test]
    fn new_style_revisions_decode_to_their_soc() {
        // Real revision codes, one per SoC generation.
        // 0x900092 Zero v1.2, 0xa01041 Pi 2 v1.1, 0xa02082 Pi 3B,
        // 0xc03111 Pi 4B 4GB, 0xc04170 Pi 5 8GB.
        assert_eq!(
            decode_revision(0x900092),
            (Some("BCM2835"), Some(Arch::Armv6))
        );
        assert_eq!(
            decode_revision(0xa01041),
            (Some("BCM2836"), Some(Arch::Armv7))
        );
        assert_eq!(
            decode_revision(0xa02082),
            (Some("BCM2837"), Some(Arch::Armv8))
        );
        assert_eq!(
            decode_revision(0xc03111),
            (Some("BCM2711"), Some(Arch::Armv8))
        );
        assert_eq!(
            decode_revision(0xc04170),
            (Some("BCM2712"), Some(Arch::Armv8))
        );
    }

    #[test]
    fn the_pi_2_revisions_that_differ_by_soc_decode_differently() {
        // The case a model string cannot answer: both are "Raspberry Pi 2
        // Model B", and they need different binaries.
        let v11 = decode_revision(0xa01041);
        let v12 = decode_revision(0xa22042);
        assert_eq!(v11.1, Some(Arch::Armv7));
        assert_eq!(v12.1, Some(Arch::Armv8));
        assert_ne!(v11.0, v12.0);
    }

    #[test]
    fn an_unknown_processor_id_reports_unknown_rather_than_guessing() {
        // A board newer than this table: bit 23 set (new-style) with processor
        // id 7, which no SoC uses yet. Guessing an ISA here would be a SIGILL.
        assert_eq!(decode_revision(0x0080_7000), (None, None));
        // Bit 23 clear is old-style regardless of the other bits, and every
        // old-style board is BCM2835.
        assert_eq!(
            decode_revision(0x0007_0000),
            (Some("BCM2835"), Some(Arch::Armv6))
        );
    }

    #[test]
    fn model_strings_bucket_zero_2_apart_from_zero() {
        // "Zero 2" contains "Zero", so order of testing is load-bearing.
        assert_eq!(
            arch_from_model("Raspberry Pi Zero 2 W Rev 1.0"),
            Some(Arch::Armv8)
        );
        assert_eq!(
            arch_from_model("Raspberry Pi Zero W Rev 1.1"),
            Some(Arch::Armv6)
        );
        assert_eq!(
            arch_from_model("Raspberry Pi 5 Model B Rev 1.0"),
            Some(Arch::Armv8)
        );
        assert_eq!(
            arch_from_model("Raspberry Pi 2 Model B Rev 1.1"),
            Some(Arch::Armv7)
        );
        assert_eq!(arch_from_model("Some Other SBC"), None);
    }

    #[test]
    fn armv6_runs_everywhere_and_aarch64_does_not() {
        let zero = Board {
            model: None,
            revision: Some(0x900092),
            soc: Some("BCM2835"),
            arch: Arch::Armv6,
            userland_64bit: false,
        };
        let pi5 = Board {
            model: None,
            revision: Some(0xc04170),
            soc: Some("BCM2712"),
            arch: Arch::Armv8,
            userland_64bit: true,
        };
        assert!(Abi::Armv6Hf.runs_on(&zero));
        assert!(Abi::Armv6Hf.runs_on(&pi5));
        assert!(!Abi::Armv7Hf.runs_on(&zero));
        assert!(!Abi::Aarch64.runs_on(&zero));
        assert!(Abi::Aarch64.runs_on(&pi5));
    }

    #[test]
    fn a_64_bit_board_on_a_32_bit_userland_takes_the_armv7_build() {
        // The case that catches people out: the CPU is ARMv8 but the userland
        // cannot load a 64-bit binary.
        let pi4_32 = Board {
            model: Some("Raspberry Pi 4 Model B Rev 1.4".into()),
            revision: Some(0xc03111),
            soc: Some("BCM2711"),
            arch: Arch::Armv8,
            userland_64bit: false,
        };
        assert_eq!(pi4_32.recommended_artifact(), Abi::Armv7Hf);
        let pi4_64 = Board {
            userland_64bit: true,
            ..pi4_32.clone()
        };
        assert_eq!(pi4_64.recommended_artifact(), Abi::Aarch64);
    }

    #[test]
    fn tiers_follow_isa_generation() {
        assert_eq!(Arch::Armv6.tier(), "T2");
        assert_eq!(Arch::Armv7.tier(), "T2");
        assert_eq!(Arch::Armv8.tier(), "T3");
    }

    #[test]
    fn page_size_is_a_power_of_two_and_at_least_4k() {
        let p = page_size();
        assert!(p >= 4096, "page size {p}");
        assert!(p.is_power_of_two(), "page size {p} is not a power of two");
    }

    #[test]
    fn detection_never_panics_on_a_host_without_procfs() {
        // This test runs on macOS in development, where neither file exists.
        let b = Board::detect();
        assert_eq!(b.userland_64bit, cfg!(target_pointer_width = "64"));
    }
}
