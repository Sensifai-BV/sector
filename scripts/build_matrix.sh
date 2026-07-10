#!/usr/bin/env bash
# Build the ESP32 firmware for every chip the port claims to support.
#
# A chip feature and a Rust target are separate choices and both must be right,
# so each row carries both. Xtensa parts need the espup fork and are skipped
# with a stated reason rather than reported as failures when it is absent —
# "not attempted" and "attempted and failed" are different claims.
set -uo pipefail

# Absolute, resolved before the cd: $0 is relative, so `dirname "$0"` no longer
# points at scripts/ once the working directory changes.
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPTS/../targets/esp32" || exit 1

RISCV_IMC="riscv32imc-unknown-none-elf"     # c2, c3
RISCV_IMAC="riscv32imac-unknown-none-elf"   # c5, c6, c61, h2

ROWS=(
  "esp32c2:$RISCV_IMC:riscv"
  "esp32c3:$RISCV_IMC:riscv"
  "esp32c5:$RISCV_IMAC:riscv"
  "esp32c6:$RISCV_IMAC:riscv"
  "esp32c61:$RISCV_IMAC:riscv"
  "esp32h2:$RISCV_IMAC:riscv"
  "esp32:xtensa-esp32-none-elf:xtensa"
  "esp32s2:xtensa-esp32s2-none-elf:xtensa"
  "esp32s3:xtensa-esp32s3-none-elf:xtensa"
)

OUT="../../measurements/build_matrix.csv"
mkdir -p "$(dirname "$OUT")"
echo "chip,arch,target,build,bench_bytes,measure_flash_bytes,note" > "$OUT"

# Whether an Xtensa build is possible is build_chip.sh's decision, not a second
# probe here: it handles the case where espup installed GCC and LLVM but no
# precompiled core (it falls back to -Zbuild-std) and locates the versioned
# linker directory. Duplicating that logic would let the two disagree, and this
# script would report SKIP for chips that build.
have_xtensa=0
if rustup run esp rustc --print sysroot >/dev/null 2>&1; then
  have_xtensa=1
fi

fail=0
for row in "${ROWS[@]}"; do
  IFS=: read -r chip target arch <<< "$row"

  if [ "$arch" = "xtensa" ] && [ "$have_xtensa" -eq 0 ]; then
    echo "$chip,$arch,$target,SKIP,,,espup Xtensa toolchain not installed" >> "$OUT"
    printf '%-10s SKIP  (no espup toolchain)\n' "$chip"
    continue
  fi

  # Through build_chip.sh: it owns the chip->target table, so the pairing is
  # defined in one place rather than repeated here where the two could drift.
  if "$SCRIPTS/build_chip.sh" "$chip" > /tmp/build_"$chip".log 2>&1; then
    b=$(stat -f%z "target/$target/release/bench" 2>/dev/null \
        || stat -c%s "target/$target/release/bench" 2>/dev/null || echo 0)
    m=$(stat -f%z "target/$target/release/measure_flash" 2>/dev/null \
        || stat -c%s "target/$target/release/measure_flash" 2>/dev/null || echo 0)
    echo "$chip,$arch,$target,PASS,$b,$m," >> "$OUT"
    printf '%-10s PASS  bench=%s B\n' "$chip" "$b"
  else
    reason=$(grep -m1 '^error' /tmp/build_"$chip".log | tr ',' ';' | cut -c1-90)
    echo "$chip,$arch,$target,FAIL,,,\"$reason\"" >> "$OUT"
    printf '%-10s FAIL  %s\n' "$chip" "$reason"
    fail=1
  fi
done

echo
echo "wrote $OUT"
exit $fail
