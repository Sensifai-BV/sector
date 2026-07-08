#!/usr/bin/env bash
# Assert the scan's inner loop emits no multiply on the T0 target.
#
# RV32IMC names its multiplies mul, mulh, mulhsu, mulhu. Any of them inside a
# scoring wrapper means a multiply reached the per-vector path, which breaks
# the cost claim the scan design rests on: m table lookups and m adds, with
# every multiply paid once during table construction.
#
# Cortex-M0+ (thumbv6m, the RP2040) is checked by the same rule with a different
# instruction set. It has a single-cycle `muls` and NO hardware divide, so a
# multiply that reached the per-vector path costs more there than on RV32IMC —
# which makes the Pico the sharpest test of the claim, not a softer one.
set -euo pipefail

TARGET=${ASM_TARGET:-riscv32imc-unknown-none-elf}
ASM=$(ls -t target/${TARGET}/release/deps/sector_quant-*.s 2>/dev/null | head -1 || true)

if [[ -z "${ASM}" ]]; then
  echo "asm-check: no assembly for ${TARGET}; run the cargo rustc step first" >&2
  exit 1
fi

# Per-architecture multiply mnemonics. Matching the wrong set would let a
# multiply through silently, so the target picks the pattern explicitly rather
# than one regex trying to cover both.
case "${TARGET}" in
  riscv32*) MUL_RE='^[[:space:]]+mul(h|hsu|hu)?[[:space:]]' ;;
  thumbv6m*|thumbv7*) MUL_RE='^[[:space:]]+(muls?|mla|mls|smull|umull)[[:space:]]' ;;
  *) echo "asm-check: no multiply pattern for ${TARGET}" >&2; exit 1 ;;
esac

echo "asm-check: target ${TARGET}"
status=0
for sym in probe_score_b8 probe_score_b4; do
  # Symbols are mangled; match the label line carrying the function name.
  label=$(grep -oE "^_[A-Za-z0-9_]*${sym}[A-Za-z0-9_]*:" "${ASM}" | head -1 | tr -d ':')
  if [[ -z "${label}" ]]; then
    echo "asm-check: symbol ${sym} not found in ${ASM}" >&2
    status=1
    continue
  fi

  body=$(awk -v s="${label}:" '$0==s{f=1} f{print} f&&/\.Lfunc_end/{exit}' "${ASM}")
  hits=$(printf '%s\n' "${body}" | grep -cE "${MUL_RE}" || true)
  if [[ "${hits}" -ne 0 ]]; then
    echo "asm-check: ${sym} contains ${hits} multiply instruction(s)" >&2
    printf '%s\n' "${body}" | grep -nE "${MUL_RE}" >&2
    status=1
  else
    echo "asm-check: ${sym} — no multiplies"
  fi
done
exit "${status}"
