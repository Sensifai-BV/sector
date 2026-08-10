#!/usr/bin/env bash
# Run the Pico firmware on the emulated board and record the result.
#
# Separate from wokwi_matrix.sh because this is one board and one target, not a
# nine-chip sweep: that script rewrites wokwi.toml per chip, which has nothing to
# do here.
#
# Three outcomes are distinguished, because collapsing them would misreport what
# was tested:
#   PASS  the scenario ran and every assertion held
#   FAIL  the board booted the image and an assertion did not hold
#   SKIP  the firmware is not built, or wokwi.com is unreachable
#         -- build-verified only, never presented as an emulation result
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
CLI=".tools/wokwi/bin/wokwi-cli"
WOKWI="targets/rp2040/wokwi"
ELF="targets/rp2040/target/thumbv6m-none-eabi/release/bench"
OUT="measurements/wokwi_pico.csv"

mkdir -p "$(dirname "$OUT")" measurements/wokwi
echo "board,result,cpu_clock_hz,timebase_ok,calib_ns_per_iter,scan_us_per_query,scan_x4_us_per_query,evictions_total,evidence" > "$OUT"

emit() { echo "wokwi-pi-pico,$1,$2,$3,$4,$5,$6,$7,\"$8\"" >> "$OUT"; }

if [ ! -x "$CLI" ]; then
  emit SKIP "" "" "" "" "" "" "wokwi-cli not found at $CLI"
  echo "pico      SKIP  wokwi-cli not found" ; exit 0
fi
if [ -z "${WOKWI_CLI_TOKEN:-}" ]; then
  emit SKIP "" "" "" "" "" "" "WOKWI_CLI_TOKEN is not set"
  echo "pico      SKIP  WOKWI_CLI_TOKEN is not set" ; exit 0
fi
if [ ! -f "$ELF" ]; then
  emit SKIP "" "" "" "" "" "" "firmware not built: cd targets/rp2040 && cargo build --release"
  echo "pico      SKIP  firmware not built" ; exit 0
fi

log="measurements/wokwi/pico.log"
if "$CLI" "$WOKWI" \
     --diagram-file "diagram-pico.json" \
     --scenario "bench.scenario.yaml" \
     --timeout 90000 \
     --serial-log-file "$log" > "measurements/wokwi/pico.out" 2>&1; then
  res=PASS
else
  res=FAIL
fi

# Pull the measured values out of the serial log whatever the verdict: a failing
# run's numbers are still evidence about where it failed.
get() { grep -m1 "^$1=" "$log" 2>/dev/null | cut -d= -f2 | tr -d '\r'; }
clk=$(get cpu_clock_hz)
tb=$(get timebase_ok)
cal=$(get calib_ns_per_iter)
us=$(get scan_us_per_query)
x4=$(get scan_x4_us_per_query)
ev=$(get evictions_total)

# An unreachable simulator is not a failed assertion, and recording it as FAIL
# would put a fabricated verdict in the results table.
if [ "$res" = FAIL ] && grep -qi "ENOTFOUND\|ECONNREFUSED\|getaddrinfo\|Error connecting" "measurements/wokwi/pico.out"; then
  emit SKIP "" "" "" "" "" "" "wokwi.com unreachable from this host"
  echo "pico      SKIP  wokwi.com unreachable" ; exit 0
fi

if [ "$res" = FAIL ]; then
  why=$(tail -3 "measurements/wokwi/pico.out" | tr '\n' ' ' | tr ',' ';' | cut -c1-90)
else
  why="scan + budgets asserted against profile arithmetic"
fi

emit "$res" "$clk" "$tb" "$cal" "$us" "$x4" "$ev" "$why"
printf 'pico      %s  clk=%s timebase_ok=%s scan=%sus x4=%sus\n' \
  "$res" "${clk:-?}" "${tb:-?}" "${us:-?}" "${x4:-?}"

echo
echo "wrote $OUT"
