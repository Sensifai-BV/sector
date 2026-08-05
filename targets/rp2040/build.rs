//! Puts `memory.x` where the linker will find it.
//!
//! `cortex-m-rt`'s `link.x` includes `memory.x` by name from the link search
//! path, which is the build output directory rather than the crate root.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    File::create(out.join("memory.x"))
        .expect("write memory.x")
        .write_all(include_bytes!("memory.x"))
        .expect("write memory.x");
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=build.rs");
}
