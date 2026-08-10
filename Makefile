.PHONY: all fmt fmt-check lint test nostd asm-check asm-check-riscv asm-check-thumb chip build-t0 lint-t0 test-t0 build-pico wokwi-pico deny check-all clean \
        build-pi check-pi isa-check isa-check-negative test-cross test-cross-if-available \
        selftest serve install-service

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

# `--all-features` is not optional here. sector-os's mmap backend and
# test-support, and the integration tests gated on them, are invisible to a
# default-feature run — check-all would report green on code it never compiled.
lint:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace --all-features

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

# ---------------------------------------------------------------------------
# Raspberry Pi (T2/T3)
# ---------------------------------------------------------------------------

# Three ABIs cover every Pi ever made; docs/PLATFORMS.md maps model to artifact.
# Static musl, so one binary per ABI runs on Raspberry Pi OS, Debian, Ubuntu,
# Alpine and Yocto with no runtime dependency. The linker flags live in
# .cargo/config.toml, so no C toolchain, sysroot or apt package is involved and
# this builds identically here and in CI.
PI_TARGETS := arm-unknown-linux-musleabihf armv7-unknown-linux-musleabihf aarch64-unknown-linux-musl

build-pi:
	@for t in $(PI_TARGETS); do \
		echo "=== $$t"; \
		rustup target add $$t >/dev/null 2>&1 || true; \
		cargo build --release -p sector-cli --target $$t || exit 1; \
	done

# Cross-compile the whole workspace with every feature. Stricter than build-pi:
# it covers sector-serve and both sector-os backends, not just the binary.
check-pi:
	@for t in $(PI_TARGETS); do \
		printf '%-34s' "$$t"; \
		rustup target add $$t >/dev/null 2>&1 || true; \
		if cargo build --workspace --all-features --target $$t --quiet; \
			then echo "OK"; else echo "FAIL"; exit 1; fi; \
	done

# The ISA baseline every Pi artifact must declare, read numerically out of
# .ARM.attributes. A binary built for a newer ISA installs cleanly on an older
# board and then dies with SIGILL at the first unsupported instruction, so the
# triple's name is not evidence — the ELF attributes are.
#
# Depends on build-pi rather than assuming the binaries exist: checking a stale
# artifact from a previous build would report on code that is no longer there.
isa-check: build-pi
	python3 scripts/check_isa.py target/arm-unknown-linux-musleabihf/release/sector v6
	python3 scripts/check_isa.py target/armv7-unknown-linux-musleabihf/release/sector v7
	python3 scripts/check_isa.py target/aarch64-unknown-linux-musl/release/sector aarch64

# Prove the check fails when it should. A decoder bug that makes it silently pass
# is worse than no check, and that has happened once.
isa-check-negative: build-pi
	@if python3 scripts/check_isa.py target/arm-unknown-linux-musleabihf/release/sector v7 >/dev/null 2>&1; \
		then echo "FAIL: an ARMv6 binary was accepted as ARMv7"; exit 1; fi
	@if python3 scripts/check_isa.py target/armv7-unknown-linux-musleabihf/release/sector v6 >/dev/null 2>&1; \
		then echo "FAIL: an ARMv7 binary was accepted as ARMv6"; exit 1; fi
	@echo "isa-check rejects a wrong baseline."

# Run the workspace tests on each Pi ABI's own arithmetic. Requires qemu-user,
# which is present in CI and on Linux dev boxes but not on macOS.
#
# This target FAILS rather than skipping when qemu is missing: a check that
# reports green without running anything is worse than one that is honestly
# unavailable. Use `make test-cross-if-available` in a script that must tolerate
# either.
test-cross:
	@command -v qemu-arm-static >/dev/null 2>&1 || { \
		echo "test-cross needs qemu-user-static (apt install qemu-user-static)."; \
		echo "It is unavailable on macOS; CI runs this leg on Linux."; exit 1; }
	CARGO_TARGET_ARM_UNKNOWN_LINUX_MUSLEABIHF_RUNNER=qemu-arm-static \
		cargo test --workspace --all-features --target arm-unknown-linux-musleabihf
	CARGO_TARGET_ARMV7_UNKNOWN_LINUX_MUSLEABIHF_RUNNER=qemu-arm-static \
		cargo test --workspace --all-features --target armv7-unknown-linux-musleabihf
	CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER=qemu-aarch64-static \
		cargo test --workspace --all-features --target aarch64-unknown-linux-musl

test-cross-if-available:
	@if command -v qemu-arm-static >/dev/null 2>&1; then $(MAKE) test-cross; \
	else echo "SKIP test-cross: no qemu-user-static (not counted as a pass)"; fi

# End-to-end on the host: build a volume, query it, check recall against brute
# force, corrupt a block and confirm detection. No dataset, no network.
selftest:
	cargo run --release -p sector-cli -- selftest

# Run the daemon against a throwaway volume, for manual poking.
serve:
	cargo run --release -p sector-cli -- build --input /tmp/sector-demo.fvecs \
		--out /tmp/sector-demo.sector --m 16 --b 8 --reserve 4096 2>/dev/null || \
		{ echo "no corpus at /tmp/sector-demo.fvecs; make a .fvecs first"; exit 1; }
	cargo run --release -p sector-cli -- serve --image /tmp/sector-demo.sector \
		--listen 127.0.0.1:8642 --workers 2

install-service:
	@echo "Run this on the Pi, from an unpacked release archive:"
	@echo "  sudo ./install.sh --image /path/to/volume.sector"
	@echo
	@echo "install.sh runs 'sector doctor' first and refuses a binary that cannot"
	@echo "run on the board. Packaging lives in packaging/."

# ---------------------------------------------------------------------------
# The gate.
#
# check-pi, isa-check and isa-check-negative are in: they need no network, no
# token and no emulator, so there is no reason for them to be optional and every
# reason for a Pi break to surface here rather than at tag time.
#
# test-cross is NOT in: it needs qemu, and a target that SKIPs on a missing
# prerequisite would let check-all report green without having run it. CI runs
# that leg on Linux where the prerequisite is real.
# ---------------------------------------------------------------------------
check-all: fmt-check lint test test-t0 nostd asm-check build-t0 lint-t0 \
           check-pi isa-check isa-check-negative deny
	@echo "All checks passed."

clean:
	cargo clean
	cd targets/esp32 && cargo clean
