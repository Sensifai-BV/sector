#!/usr/bin/env bash
# Build the ESP32 firmware for one chip, with the correct target chosen for you.
#
#   scripts/build_chip.sh esp32c3 [extra cargo args...]
#
# The chip feature and the `--target` triple are independent choices that both
# have to be right, and getting them wrong fails hundreds of crates deep with
# nothing naming the cause. `--features chip-esp32s3 --target
# riscv32imc-unknown-none-elf` reports:
#
#   error[E0554]: `#![feature]` may not be used on the stable release channel
#   error[E0433]: cannot find module or crate `xtensa_lx` in this scope
#   error[E0599]: no method named `compare_exchange` found for `Atomic<T>`
#   ... Seems you are building for an unsupported or wrong target
#
# Four errors, none of them the actual problem. A build.rs check cannot help:
# cargo compiles esp-sync and esp-hal in parallel with this crate's own build
# script, so they fail first. The pairing has to be decided before cargo runs,
# which is what this script is for.
set -uo pipefail

cd "$(dirname "$0")/../targets/esp32" || exit 1

# chip:target. Xtensa needs the espup fork for asm_experimental_arch; the two
# RISC-V triples differ by the atomics extension, which decides whether
# esp-sync's spinlock has a compare_exchange to call.
#
# A flat list rather than an associative array: macOS ships bash 3.2 as
# /bin/bash, where `declare -A` does not exist and fails with "unbound
# variable" on every lookup.
CHIPS="
esp32c2:riscv32imc-unknown-none-elf
esp32c3:riscv32imc-unknown-none-elf
esp32c5:riscv32imac-unknown-none-elf
esp32c6:riscv32imac-unknown-none-elf
esp32c61:riscv32imac-unknown-none-elf
esp32h2:riscv32imac-unknown-none-elf
esp32:xtensa-esp32-none-elf
esp32s2:xtensa-esp32s2-none-elf
esp32s3:xtensa-esp32s3-none-elf
"

target_for() {
  echo "$CHIPS" | while IFS=: read -r c t; do
    [ "$c" = "$1" ] && echo "$t"
  done
}

usage() {
  echo "usage: scripts/build_chip.sh <chip> [cargo args...]"
  echo
  echo "chip          target"
  echo "$CHIPS" | while IFS=: read -r c t; do
    [ -n "$c" ] && printf '  %-10s  %s\n' "$c" "$t"
  done
  echo
  echo "All nine at once: scripts/build_matrix.sh"
}

if [ $# -lt 1 ]; then
  usage >&2
  exit 2
fi

CHIP="$1"; shift
T=$(target_for "$CHIP")
if [ -z "$T" ]; then
  echo "unknown chip: $CHIP" >&2
  echo >&2
  usage >&2
  exit 2
fi

# Xtensa needs the espup toolchain, and there are three ways it can be present
# but unusable. Each is detected and reported separately, because they need
# different fixes and the raw failures name none of them.
PREFIX=""
BUILD_STD=""
case "$T" in xtensa-*)
  sysroot=$(rustup run esp rustc --print sysroot 2>/dev/null || true)
  if [ -z "$sysroot" ]; then
    echo "sector-esp32: no 'esp' toolchain. Install: cargo install espup && espup install" >&2
    exit 3
  fi
  PREFIX="rustup run esp"

  # (1) No precompiled core for the target. espup reuses an existing install
  # when it finds one and reports success, so `espup install` can print
  # "Installation successfully completed!" while lib/rustlib holds only
  # rust-std-<host> and no rust-std-xtensa-*. The build then fails with
  # "can't find crate for `core`" hundreds of crates deep. rust-src is present
  # on that path, so -Zbuild-std=core compiles core from source instead of
  # requiring a reinstall.
  if [ ! -d "$sysroot/lib/rustlib/$T" ]; then
    if [ -f "$sysroot/lib/rustlib/src/rust/library/core/Cargo.toml" ]; then
      echo "note: no precompiled core for $T; building it from source (-Zbuild-std=core)" >&2
      BUILD_STD="-Zbuild-std=core"
    else
      echo "sector-esp32: 'esp' toolchain has neither a precompiled core for $T" >&2
      echo "  nor rust-src to build one. Force a clean install:" >&2
      echo "    espup uninstall && espup install" >&2
      exit 3
    fi
  fi

  # (2) Linker not on PATH. espup writes ~/export-esp.sh for this, but on the
  # reuse path it can leave the file empty, and the GCC directory is versioned
  # so it cannot be hardcoded. Find it under the toolchain instead.
  if ! command -v "${CHIP%%-*}" >/dev/null 2>&1 &&
     ! command -v "xtensa-${CHIP}-elf-gcc" >/dev/null 2>&1; then
    gcc_bin=$(find "$sysroot/xtensa-esp-elf" -type d -name bin 2>/dev/null | head -1)
    if [ -n "$gcc_bin" ] && [ -x "$gcc_bin/xtensa-${CHIP}-elf-gcc" ]; then
      PATH="$gcc_bin:$PATH"
      export PATH
    else
      echo "sector-esp32: linker xtensa-${CHIP}-elf-gcc not found under $sysroot" >&2
      echo "  espup installs it; if ~/export-esp.sh is empty, re-run: espup install" >&2
      exit 3
    fi
  fi
  ;;
esac

echo "building $CHIP for $T"
# PREFIX is deliberately unquoted: it is either empty or two words.
exec $PREFIX cargo build --release $BUILD_STD --features "chip-$CHIP" --target "$T" "$@"
