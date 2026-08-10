# PLATFORMS.md

Every board SECTOR ships a binary for, and the one Rust target triple that
serves it. Three triples cover every Raspberry Pi ever made.

## Why the mapping is not obvious

Two boards with the same name can need different binaries, and two boards a
generation apart can share one. The Pi Zero and the Pi Zero 2 W differ by ISA
*generation*, not by speed: the first is ARMv6 and cannot execute an ARMv7
instruction at all. The Pi 2 changed SoC mid-life — a v1.1 is Cortex-A7
(ARMv7-A), a v1.2 is Cortex-A53 (ARMv8-A) — so "Pi 2" alone does not determine
the answer.

This is also why Raspberry Pi OS is a separate distribution rather than Debian
armhf. Debian's own `armhf` port requires ARMv7; a Pi 1 or Zero cannot run it.
Raspberry Pi OS 32-bit is built for ARMv6 + VFPv2 precisely so one image covers
the whole 32-bit family.

## Model matrix

| Board | SoC | Core | ISA | RAM | Triple | Tier |
|---|---|---|---|---:|---|---|
| Pi 1 Model A / A+ | BCM2835 | ARM1176JZF-S | ARMv6Z + VFPv2 | 256 MB | `arm-unknown-linux-musleabihf` | T2 |
| Pi 1 Model B / B+ | BCM2835 | ARM1176JZF-S | ARMv6Z + VFPv2 | 512 MB | `arm-unknown-linux-musleabihf` | T2 |
| Pi Zero / Zero W / Zero WH | BCM2835 | ARM1176JZF-S | ARMv6Z + VFPv2 | 512 MB | `arm-unknown-linux-musleabihf` | T2 |
| Compute Module 1 | BCM2835 | ARM1176JZF-S | ARMv6Z + VFPv2 | 512 MB | `arm-unknown-linux-musleabihf` | T2 |
| Pi 2 Model B **v1.1** | BCM2836 | Cortex-A7 ×4 | ARMv7-A + VFPv4/NEON | 1 GB | `armv7-unknown-linux-musleabihf` | T2 |
| Pi 2 Model B **v1.2** | BCM2837 | Cortex-A53 ×4 | ARMv8-A | 1 GB | `aarch64-unknown-linux-musl` | T3 |
| Pi Zero 2 W | BCM2710A1 (RP3A0) | Cortex-A53 ×4 | ARMv8-A | 512 MB | `aarch64-unknown-linux-musl` | T3 |
| Pi 3 Model A+ | BCM2837B0 | Cortex-A53 ×4 | ARMv8-A | 512 MB | `aarch64-unknown-linux-musl` | T3 |
| Pi 3 Model B | BCM2837 | Cortex-A53 ×4 | ARMv8-A | 1 GB | `aarch64-unknown-linux-musl` | T3 |
| Pi 3 Model B+ | BCM2837B0 | Cortex-A53 ×4 | ARMv8-A | 1 GB | `aarch64-unknown-linux-musl` | T3 |
| Compute Module 3 / 3+ | BCM2837 | Cortex-A53 ×4 | ARMv8-A | 1 GB | `aarch64-unknown-linux-musl` | T3 |
| Pi 4 Model B | BCM2711 | Cortex-A72 ×4 | ARMv8-A | 1–8 GB | `aarch64-unknown-linux-musl` | T3 |
| Pi 400 | BCM2711 | Cortex-A72 ×4 | ARMv8-A | 4 GB | `aarch64-unknown-linux-musl` | T3 |
| Compute Module 4 | BCM2711 | Cortex-A72 ×4 | ARMv8-A | 1–8 GB | `aarch64-unknown-linux-musl` | T3 |
| Pi 5 | BCM2712 | Cortex-A76 ×4 | ARMv8.2-A | 2–16 GB | `aarch64-unknown-linux-musl` | T3 |
| Pi 500 / Compute Module 5 | BCM2712 | Cortex-A76 ×4 | ARMv8.2-A | 2–16 GB | `aarch64-unknown-linux-musl` | T3 |

A 64-bit-capable board running a 32-bit userland takes the 32-bit binary. The
kernel's word size does not decide this; the userland's does. `sector doctor`
reports both and names the artifact to install.

## Identifying a board at runtime

`sector doctor` reads `/proc/device-tree/model` for the marketing name and the
`Revision` field of `/proc/cpuinfo` for the board revision, which encodes the
model and SoC unambiguously where the name does not. It then reports whether the
running binary's ISA baseline matches the hardware.

The failure this prevents is quiet: an ARMv7 binary installs cleanly on a Pi
Zero and dies with `SIGILL` at the first ARMv7-only instruction, which may be
inside a code path that does not run until a query arrives. The release workflow
asserts the armv6 artifact declares `Tag_CPU_arch: ARM v6` in its ELF attributes
rather than trusting the triple's name.

## Static linking

Release binaries are static musl builds. One artifact per ABI runs on Raspberry
Pi OS, Ubuntu, Debian, Alpine and Yocto without a runtime dependency, and there
is no glibc version floor to track — a Pi 1 running an old Raspbian and a Pi 5
running current Ubuntu take the same two artifacts between them.

`aarch64-unknown-linux-gnu` is also published for anyone who needs to link
against a system library; it carries a glibc floor and the musl build should be
preferred otherwise.

## Page size is a runtime fact

Raspberry Pi OS on Pi 5 uses a 16 KiB kernel page size; every other Pi
configuration and every Ubuntu build uses 4 KiB. `sector-os` reads
`sysconf(_SC_PAGESIZE)` rather than assuming, because the mapped backend reports
fault granularity and a hardcoded 4096 would misreport it by 4x on exactly one
popular configuration.

## Storage, and what it means for the measurement

Every board here reads its volume through a flash translation layer — microSD on
all of them, USB or NVMe on Pi 4 and Pi 5. That is the opposite of the T0/T1
case, where raw NOR is memory-mapped and a rerank fetch is a load instruction.

The consequence is the inversion this project reports: the larger tier performs
the same access pattern more slowly, because stage two's per-candidate random
read is serviced at FTL block granularity instead of by the memory system. It is
a measured result, not a caveat, and the default `FileFlash` backend is
deliberately the honest one — see `crates/sector-os` for why the `mmap` backend
is opt-in and what its `Xip` implementation does and does not claim.

## Tiers

`T2` and `T3` are defined in `crates/sector-format/src/profile.rs`, with their
derivation in `docs/design/001-pi-tier-profiles.md`. Their `ram_budget` figures
are **floors for the tier**, not per-board maximums: T2's 32 MiB is what a
256 MB Pi 1 Model A can guarantee after the GPU carve-out, and reading it as a
Pi 2's capacity understates that board by a factor of 30.