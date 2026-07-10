#!/usr/bin/env bash
# Run each firmware on its emulated devkit board and record the result.
#
# Three outcomes are distinguished, because collapsing them would misreport what
# was tested:
#   PASS  the scenario ran and every assertion held
#   FAIL  the board booted the image and an assertion did not hold
#   SKIP  no Wokwi board exists for the chip, or its toolchain is unavailable
#         -- build-verified only, never presented as an emulation result
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
CLI=".tools/wokwi/bin/wokwi-cli"
WOKWI="targets/esp32/wokwi"
OUT="measurements/wokwi_matrix.csv"

if [ ! -x "$CLI" ]; then
  echo "wokwi-cli not found at $CLI" >&2
  exit 1
fi
if [ -z "${WOKWI_CLI_TOKEN:-}" ]; then
  echo "WOKWI_CLI_TOKEN is not set" >&2
  exit 1
fi

# chip:target:board — chips with no Wokwi board are absent and reported SKIP.
ROWS=(
  "esp32c3:riscv32imc-unknown-none-elf:board-esp32-c3-devkitm-1"
  "esp32c6:riscv32imac-unknown-none-elf:board-esp32-c6-devkitc-1"
  "esp32h2:riscv32imac-unknown-none-elf:board-esp32-h2-devkitm-1"
  "esp32:xtensa-esp32-none-elf:board-esp32-devkit-c-v4"
  "esp32s2:xtensa-esp32s2-none-elf:board-esp32-s2-devkitm-1"
  "esp32s3:xtensa-esp32s3-none-elf:board-esp32-s3-devkitc-1"
)
NO_BOARD=(esp32c2 esp32c5 esp32c61)

mkdir -p "$(dirname "$OUT")" measurements/wokwi
echo "chip,board,result,scan_us_per_query,scan_ns_per_vector,scan_x4_us_per_query,capacity_vectors,evidence" > "$OUT"

for chip in "${NO_BOARD[@]}"; do
  echo "$chip,(no Wokwi board),SKIP,,,,,build-verified only" >> "$OUT"
  printf '%-9s SKIP  no Wokwi board\n' "$chip"
done

for row in "${ROWS[@]}"; do
  IFS=: read -r chip target board <<< "$row"
  elf="targets/esp32/target/$target/release/bench"

  if [ ! -f "$elf" ]; then
    echo "$chip,$board,SKIP,,,,,firmware not built (toolchain unavailable)" >> "$OUT"
    printf '%-9s SKIP  firmware not built\n' "$chip"
    continue
  fi

  # wokwi.toml points at one elf; write it per chip rather than keeping six.
  cat > "$WOKWI/wokwi.toml" <<TOML
[wokwi]
version = 1
elf = "../target/$target/release/bench"
firmware = "../target/$target/release/bench"
TOML

  log="measurements/wokwi/$chip.log"
  if "$CLI" "$WOKWI" \
       --diagram-file "diagram-$chip.json" \
       --scenario "bench.scenario.yaml" \
       --timeout 60000 \
       --serial-log-file "$log" > "measurements/wokwi/$chip.out" 2>&1; then
    res=PASS
  else
    res=FAIL
  fi

  # Pull the measured values out of the serial log, whatever the verdict: a
  # failing run's numbers are still evidence about where it failed.
  get() { grep -m1 "^$1=" "$log" 2>/dev/null | cut -d= -f2 | tr -d '\r'; }
  us=$(get scan_us_per_query)
  ns=$(get scan_ns_per_vector)
  x4=$(get scan_x4_us_per_query)
  cap=$(get capacity_vectors)

  if [ "$res" = FAIL ]; then
    why=$(tail -3 "measurements/wokwi/$chip.out" | tr '\n' ' ' | tr ',' ';' | cut -c1-90)
  else
    why="scan + budgets asserted against profile arithmetic"
  fi

  echo "$chip,$board,$res,$us,$ns,$x4,$cap,\"$why\"" >> "$OUT"
  printf '%-9s %s  scan=%sus/query x4=%sus\n' "$chip" "$res" "${us:-?}" "${x4:-?}"
done

echo
echo "wrote $OUT"
