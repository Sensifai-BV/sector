.PHONY: all fmt fmt-check lint test nostd asm-check asm-check-riscv asm-check-thumb chip build-t0 lint-t0 test-t0 build-pico wokwi-pico deny check-all clean

all: check-all

fmt:
	cargo fmt --all
	cd targets/esp32 && cargo fmt --all
	cd targets/rp2040 && cargo fmt --all
	cd targets/esp32 && cargo fmt --all

fmt-check:
	cargo fmt --all --check
	cd targets/esp32 && cargo fmt --all --check
	cd targets/rp2040 && cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

# The no_std guarantee: the four core-family crates must build with no std,
# no alloc, on a target that has neither.
nostd:
	cargo build -p sector-hal -p sector-quant -p sector-codec -p sector-format -p sector-core \
		--no-default-features --target thumbv7em-none-eabihf

# The no-multiply guarantee: the scan's inner loop must contain no multiply
# instruction on the T0 target. Every multiply is paid once during table
# construction, and the per-vector cost claim depends on that holding in the
# emitted code rather than in the source.
asm-check: asm-check-riscv asm-check-thumb

asm-check-riscv:
	@cargo rustc -p sector-quant --release --features asm-probe \
		--target riscv32imc-unknown-none-elf -- --emit asm -C opt-level=3 >/dev/null
	@scripts/asm-check.sh

# Cortex-M0+ (RP2040). Its multiplier is weak and it has no hardware divide, so
# a multiply reaching the per-vector path costs more here than on RV32IMC.
asm-check-thumb:
	@cargo rustc -p sector-quant --release --features asm-probe \
		--target thumbv6m-none-eabi -- --emit asm -C opt-level=3 >/dev/null
	@ASM_TARGET=thumbv6m-none-eabi scripts/asm-check.sh

# T0 firmware (ESP32-C3, RISC-V). T1 (ESP32-S3) is Xtensa: `rustup run esp make build-t1`.
# Through build_chip.sh so the chip and target cannot disagree: passing them
# separately fails hundreds of crates deep with errors that never mention the
# target. See scripts/build_chip.sh.
build-t0:
	scripts/build_chip.sh esp32c3
	cd targets/rp2040 && cargo build --release

# Lint the firmware against its real target. On the host the `no_main` binaries
# would be built for a std target where `panic = "abort"` does not apply, so
# they fail to compile for reasons that say nothing about the code.
lint-t0:
	cd targets/esp32 && cargo clippy --release --features chip-esp32c3 \
		--target riscv32imc-unknown-none-elf -- -D warnings
	cd targets/rp2040 && cargo clippy --release -- -D warnings

# One chip, correct target chosen for you: make chip CHIP=esp32c6
chip:
	scripts/build_chip.sh $(CHIP)

# The firmware's host-testable half: partition arithmetic, the instrument's
# pulse encoding, shell parsing, and the measurement record conversions. These
# run without hardware, which is where an off-by-one in the phase encoding
# would otherwise cost an oscilloscope session to find.
# The firmware crates have no host tests: their binaries are `no_main` and
# their backends call chip peripherals, so `cargo test` on the host cannot
# link them. What they are checked by is `build-t0` and `lint-t0` against the
# real targets, plus `build_matrix.sh` across all nine chips.
test-t0:
	@echo "firmware: no host tests by design; see build-t0, lint-t0, scripts/build_matrix.sh"

deny:
	cargo deny check

# Pico (RP2040) benchmark firmware. Separate target because this crate is
# excluded from the root workspace: it has its own .cargo/config.toml pinning
# thumbv6m-none-eabi.
build-pico:
	cd targets/rp2040 && cargo build --release

# Emulated Pico run. Needs WOKWI_CLI_TOKEN and network reach to wokwi.com, so it
# is invoked explicitly rather than from check-all -- a check that SKIPs when its
# prerequisites are absent would report green without testing anything.
wokwi-pico: build-pico
	scripts/wokwi_pico.sh

check-all: fmt-check lint test test-t0 nostd asm-check build-t0 lint-t0 deny
	@echo "All checks passed."

clean:
	cargo clean
	cd targets/esp32 && cargo clean
