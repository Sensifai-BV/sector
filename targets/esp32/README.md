# targets/esp32

Firmware for the two hardware tiers. **Nothing here has run on a device**, and
that is the point of precondition P1: every power and bandwidth figure in the
report is an estimate until this code runs against a shunt and a scope.

## What this closes, and what it does not

This directory closes as much of P1 as is possible without hardware: the
backends, the partition layout, the shell, and the measurement binaries all
exist and compile. It does **not** close P1. The campaign needs:

- an ESP32-C3 (T0) and an ESP32-S3 (T1),
- a current shunt and an oscilloscope for the per-phase energy trace,
- a host-built volume image flashed to the `sector` partition.

## Tiers

| Tier | Part | ISA | Target triple | Toolchain |
|---|---|---|---|---|
| T0 | ESP32-C3 | RISC-V | `riscv32imc-unknown-none-elf` | upstream stable |
| T1 | ESP32-S3 | Xtensa | `xtensa-esp32s3-none-elf` | `esp-rs` fork of rustc |

T1 needs a forked compiler because Xtensa is not an upstream LLVM target.
Install it with `espup install`; upstream `rustup` cannot produce this target.

## Expected build warning

`make build-t0` links with

    rust-lld: cannot find entry symbol _start; not setting start address

This is correct for the current state. The reset vector and `_start` come from
`esp-hal`'s runtime, which is not yet a dependency — the skeleton must build
without hardware, and an unbuildable skeleton is worse than none. The warning
disappears when `esp-hal` is added at bring-up; until then it is the honest
signal that this image will not boot.

## The measurement that matters

The cost model attributes query energy to Rotate, Table, Scan, Rerank and
Finalize **separately**. A measurement reporting only a total cannot falsify it,
which is why `Instrument` toggles a GPIO at each phase boundary and the five
edges must be individually distinguishable on the trace.

The specific number under test is the T0 rerank estimate of **1.92 ms at
R=100**. It is stated as refutable: if the measurement disagrees, the estimate
is withdrawn rather than the measurement being requestioned.
