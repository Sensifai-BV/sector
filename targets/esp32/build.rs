//! Reject a chip/target mismatch before any dependency is compiled.
//!
//! The chip feature and the `--target` triple are independent choices and both
//! have to agree. A `compile_error!` in `lib.rs` cannot catch a disagreement,
//! because `esp-hal` and `esp-sync` are compiled *first* and fail there — so
//! the diagnosis has to happen in a build script, which cargo runs before the
//! dependency graph is built.
//!
//! What the mismatch looks like without this check, for
//! `--features chip-esp32s3 --target riscv32imc-unknown-none-elf`:
//!
//! ```text
//! error[E0554]: `#![feature]` may not be used on the stable release channel
//! error[E0433]: cannot find module or crate `xtensa_lx` in this scope
//! error[E0599]: no method named `compare_exchange` found for `Atomic<T>`
//! ... Seems you are building for an unsupported or wrong target
//! ```
//!
//! Four errors, hundreds of crates deep, none of which names the cause.

use std::env;

/// `(feature suffix, required target triple)`.
///
/// The three architectures differ in ways that cannot be papered over: Xtensa
/// needs the espup toolchain fork for `asm_experimental_arch`, and the two
/// RISC-V triples differ by the atomics extension, which decides whether
/// `esp-sync`'s spinlock has a `compare_exchange` to call.
const CHIPS: &[(&str, &str)] = &[
    ("esp32c2", "riscv32imc-unknown-none-elf"),
    ("esp32c3", "riscv32imc-unknown-none-elf"),
    ("esp32c5", "riscv32imac-unknown-none-elf"),
    ("esp32c6", "riscv32imac-unknown-none-elf"),
    ("esp32c61", "riscv32imac-unknown-none-elf"),
    ("esp32h2", "riscv32imac-unknown-none-elf"),
    ("esp32", "xtensa-esp32-none-elf"),
    ("esp32s2", "xtensa-esp32s2-none-elf"),
    ("esp32s3", "xtensa-esp32s3-none-elf"),
];

/// Flash offset of the mock rerank corpus, mirroring `bench.rs`.
const CORPUS_BASE: u64 = 0x0020_0000;

/// Where the application image starts on this family.
const APP_BASE: u64 = 0x0001_0000;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bin/bench.rs");

    // The bench firmware erases and programs a region at CORPUS_BASE. If the
    // image ever grows past that offset, the benchmark overwrites the code it is
    // running from, and the fault looks like a corpus bug rather than a layout
    // one. The image size is not known here (this runs before linking), so the
    // check is on the headroom the layout assumes: warn loudly if the gap is
    // smaller than a typical debug-symbol growth step.
    let headroom = CORPUS_BASE - APP_BASE;
    if headroom < 1 << 20 {
        println!(
            "cargo:warning=bench corpus at {:#x} leaves only {} KiB for the image; \
             move CORPUS_BASE or shrink the region",
            CORPUS_BASE,
            headroom / 1024
        );
    }

    let target = env::var("TARGET").unwrap_or_default();
    let selected: Vec<&str> = CHIPS
        .iter()
        .map(|(chip, _)| *chip)
        .filter(|chip| env::var(format!("CARGO_FEATURE_CHIP_{}", chip.to_uppercase())).is_ok())
        .collect();

    // Zero or several chips are also reported here rather than by `lib.rs`.
    // With none selected, `esp-hal` fails first with an empty peripheral set;
    // with two, their linker scripts collide inside a dependency.
    if selected.is_empty() {
        fail(&format!(
            "no chip selected.\n\
             Pass exactly one chip feature and its matching target:\n\
             {}",
            usage()
        ));
    }
    if selected.len() > 1 {
        fail(&format!(
            "{} chips selected ({}). The chip features are mutually exclusive.",
            selected.len(),
            selected.join(", ")
        ));
    }

    let chip = selected[0];
    let want = CHIPS
        .iter()
        .find(|(c, _)| *c == chip)
        .map(|(_, t)| *t)
        .expect("chip came from CHIPS");

    if target != want {
        let extra = if want.starts_with("xtensa") {
            "\nXtensa targets need the espup toolchain fork:\n  \
             rustup run esp cargo build --release --features chip-{chip} --target {want}"
                .replace("{chip}", chip)
                .replace("{want}", want)
        } else {
            String::new()
        };
        fail(&format!(
            "chip/target mismatch.\n\
             \x20 --features chip-{chip}  requires  --target {want}\n\
             \x20 got --target {got}\n\
             \n\
             Without this check the build fails hundreds of crates deep in \
             esp-sync and esp-hal with errors that do not mention the target \
             (`#![feature] may not be used on the stable release channel`, \
             an unresolved `xtensa_lx`, a missing `compare_exchange`).\
             {extra}\n\
             \n{}",
            usage(),
            chip = chip,
            want = want,
            got = if target.is_empty() {
                "(unset)"
            } else {
                &target
            },
            extra = extra,
        ));
    }
}

fn usage() -> String {
    let mut out = String::from("Valid combinations:\n");
    for (chip, target) in CHIPS {
        out.push_str(&format!("  --features chip-{chip:<9} --target {target}\n"));
    }
    out.push_str("\nOr build every chip at once: scripts/build_matrix.sh");
    out
}

/// Emit the message as a cargo error and stop.
///
/// `cargo:warning=` per line keeps the text visible: a build-script panic
/// prints its payload once and truncates awkwardly, and this needs to be
/// readable.
fn fail(msg: &str) -> ! {
    for line in msg.lines() {
        println!("cargo:warning={line}");
    }
    panic!(
        "sector-esp32: {}",
        msg.lines().next().unwrap_or("misconfigured")
    );
}
