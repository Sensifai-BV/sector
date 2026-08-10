//! `sector selftest` — prove the binary works on this machine.
//!
//! Builds a volume in a temporary directory, queries it, checks recall against
//! brute force, corrupts a block and checks that the corruption is detected. No
//! dataset and no network, so it runs on a freshly installed Pi and under
//! `qemu-user` in CI.
//!
//! # What this catches that a unit test cannot
//!
//! Unit tests ran on the build machine. This runs on the target, which is where
//! the interesting failures live: a wrong-ABI binary that starts and then hits an
//! unsupported instruction, an endianness or alignment assumption that only shows
//! on ARM, a page size that differs from the builder's. It is the check the
//! release workflow runs under emulation for every artifact, so a cross-build that
//! compiles but does not execute fails the release instead of shipping.
//!
//! # Why recall is checked against brute force rather than a fixed number
//!
//! A hardcoded expected recall would encode the build machine's arithmetic, and
//! the first legitimate change to training or quantization would break it for
//! reasons unrelated to the platform. Comparing against a brute-force ranking
//! computed here, with the engine's own scoring function, tests the property that
//! must hold on every platform: the two-stage pipeline recovers most of what an
//! exhaustive scan would return.

use sector_hal::NorFlash;
use sector_os::json::Json;
use sector_os::search::Searcher;
use sector_os::verify::verify;
use sector_os::{FileFlash, HostVolume};

/// `selftest` arguments.
pub struct Args {
    /// Corpus size. Small by default so this finishes on a Pi 1.
    pub n: usize,
    /// Emit JSON.
    pub json: bool,
    /// Keep the temporary volume for inspection.
    pub keep: bool,
}

/// Parse `selftest` arguments.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    Ok(Args {
        n: crate::opt_num(argv, "--n", 512)?,
        json: argv.iter().any(|a| a == "--json"),
        keep: argv.iter().any(|a| a == "--keep"),
    })
}

/// One check's outcome.
struct Check {
    name: &'static str,
    passed: bool,
    detail: String,
}

const D: usize = 32;
const M: usize = 4;
const K: usize = 10;

/// Run every check. Returns the number that failed.
pub fn run(args: Args) -> Result<usize, String> {
    let dir = std::env::temp_dir().join(format!("sector_selftest_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{e}"))?;
    let image = dir.join("selftest.sector");

    let mut checks = Vec::new();

    // 1. Build, through the same image builder `sector-os`'s own tests use, so a
    // selftest failure and a unit-test failure point at the same code. It also
    // keeps this command's output clean: `build_cmd::run` prints a training
    // report, which would be noise in a check whose output gets parsed.
    //
    // The corpus is isotropic across subspaces on purpose — every component
    // shares one range, so the per-subspace quantization scales agree. See
    // `sector_os::volume::test_support` for the defect that makes it matter.
    let n = args.n.max(64);
    let (image_bytes, corpus) = sector_os::volume::test_support::build_image_and_corpus(D, M, n);
    let built = std::fs::write(&image, &image_bytes);
    checks.push(Check {
        name: "build",
        passed: built.is_ok(),
        detail: match &built {
            Ok(()) => format!("{n} vectors, D={D} m={M}, {} B", image_bytes.len()),
            Err(e) => format!("{e}"),
        },
    });
    if built.is_err() {
        return finish(args, checks, &dir);
    }

    // 2. Mount and query.
    let mut verified = 0u32;
    match Searcher::<FileFlash>::open(&image, None) {
        Err(e) => checks.push(Check {
            name: "mount",
            passed: false,
            detail: format!("{e}"),
        }),
        Ok(mut searcher) => {
            checks.push(Check {
                name: "mount",
                passed: true,
                detail: format!("{} resident bytes", searcher.resident_bytes()),
            });

            // Rerank records, for the brute-force reference. Read through a
            // second handle so the searcher's counters are untouched.
            let reference = match brute_force_reference(&image, &corpus, n) {
                Ok(r) => r,
                Err(e) => {
                    checks.push(Check {
                        name: "reference",
                        passed: false,
                        detail: e,
                    });
                    return finish(args, checks, &dir);
                }
            };

            let probes = [0usize, n / 4, n / 2, n - 1];
            let mut overlap = 0usize;
            let mut ok = true;
            for &v in &probes {
                match searcher.search(&corpus[v * D..(v + 1) * D], K) {
                    Err(e) => {
                        ok = false;
                        checks.push(Check {
                            name: "query",
                            passed: false,
                            detail: format!("{e}"),
                        });
                        break;
                    }
                    Ok(a) => {
                        verified += a.stats.rerank.blocks_verified;
                        if a.stats.scan.scanned as usize != n {
                            ok = false;
                            checks.push(Check {
                                name: "query",
                                passed: false,
                                detail: format!("scanned {} of {n} vectors", a.stats.scan.scanned),
                            });
                            break;
                        }
                        let want = &reference[&v];
                        overlap += a.ids.iter().filter(|id| want.contains(id)).count();
                    }
                }
            }
            if ok {
                checks.push(Check {
                    name: "query",
                    passed: true,
                    detail: format!("{} queries, full scan each", probes.len()),
                });
                let recall = overlap as f64 / (probes.len() * K) as f64;
                checks.push(Check {
                    name: "recall",
                    // 0.5 against brute force: stage one keeps only R candidates
                    // by quantized score, so perfect agreement is not what the
                    // design claims. A floor well below 1.0 tests the platform
                    // rather than the algorithm's precision.
                    passed: recall >= 0.5,
                    detail: format!("{recall:.3} at k={K} vs brute force"),
                });
                checks.push(Check {
                    name: "crc verification ran",
                    passed: verified > 0,
                    detail: format!("{verified} blocks verified"),
                });
            }
        }
    }

    // 3. Corruption is detected. The property the whole protection design rests
    // on, checked on this machine's own arithmetic rather than assumed from the
    // build host's.
    match corruption_is_detected(&image) {
        Ok(dropped) => checks.push(Check {
            name: "corruption detected",
            passed: dropped > 0,
            detail: format!("{dropped} candidates dropped after one flipped byte"),
        }),
        Err(e) => checks.push(Check {
            name: "corruption detected",
            passed: false,
            detail: e,
        }),
    }

    // 4. A clean volume sweeps clean, so the sweep is not reporting damage that
    // is not there.
    match sweep_is_clean(&image) {
        Ok(true) => checks.push(Check {
            name: "clean sweep",
            passed: true,
            detail: "every CRC matched".into(),
        }),
        Ok(false) => checks.push(Check {
            name: "clean sweep",
            passed: false,
            detail: "a freshly built volume reported damage".into(),
        }),
        Err(e) => checks.push(Check {
            name: "clean sweep",
            passed: false,
            detail: e,
        }),
    }

    finish(args, checks, &dir)
}

/// Print the results and clean up. Returns the failure count.
fn finish(args: Args, checks: Vec<Check>, dir: &std::path::Path) -> Result<usize, String> {
    let failed = checks.iter().filter(|c| !c.passed).count();

    if args.json {
        let mut j = Json::new();
        j.object(|o| {
            o.array("checks", |a| {
                for c in &checks {
                    a.object(|co| {
                        co.str("name", c.name);
                        co.bool("passed", c.passed);
                        co.str("detail", &c.detail);
                    });
                }
            });
            o.uint("failed", failed as u64);
            o.bool("ok", failed == 0);
            o.str("abi", &sector_os::platform::Abi::current().to_string());
        });
        print!("{}", j.finish());
    } else {
        for c in &checks {
            println!(
                "{}  {:<24} {}",
                if c.passed { "ok  " } else { "FAIL" },
                c.name,
                c.detail
            );
        }
        println!();
        if failed == 0 {
            println!("selftest passed on {}", sector_os::platform::Abi::current());
        } else {
            println!("{failed} check(s) failed");
        }
    }

    if !args.keep {
        let _ = std::fs::remove_dir_all(dir);
    } else {
        println!("volume kept in {}", dir.display());
    }
    Ok(failed)
}

/// Brute-force top-K per probe query, using the engine's own scoring function.
fn brute_force_reference(
    image: &std::path::Path,
    corpus: &[f32],
    n: usize,
) -> Result<std::collections::BTreeMap<usize, Vec<u32>>, String> {
    let mut f = FileFlash::open(image).map_err(|e| format!("{e}"))?;
    let v = HostVolume::mount(&mut f, None).map_err(|e| format!("{e}"))?;
    let g = v.geometry;

    let mut records = vec![vec![0u8; g.rerank_bytes]; n];
    for (id, rec) in records.iter_mut().enumerate() {
        let off = g
            .rerank
            .offset_of(id)
            .ok_or_else(|| format!("no rerank offset for id {id}"))?;
        f.read(v.rerank_base() + off as u32, rec)
            .map_err(|e| format!("{e}"))?;
    }

    let mut out = std::collections::BTreeMap::new();
    for &probe in &[0usize, n / 4, n / 2, n - 1] {
        let q = sector_os::search::quantize_query(&corpus[probe * D..(probe + 1) * D], g.d)
            .map_err(|e| format!("{e}"))?;
        let mut scored: Vec<(i32, u32)> = records
            .iter()
            .enumerate()
            .map(|(id, rec)| (sector_core::rerank::exact_score(&q, rec), id as u32))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        out.insert(probe, scored.iter().take(K).map(|(_, id)| *id).collect());
    }
    Ok(out)
}

/// Flip one rerank byte in a copy of the volume and count the resulting drops.
fn corruption_is_detected(image: &std::path::Path) -> Result<u32, String> {
    let mut bytes = std::fs::read(image).map_err(|e| format!("{e}"))?;
    let base = {
        let mut f = FileFlash::open(image).map_err(|e| format!("{e}"))?;
        let v = HostVolume::mount(&mut f, None).map_err(|e| format!("{e}"))?;
        v.rerank_base() as usize
    };
    bytes[base + 3] ^= 0xFF;

    let corrupted = image.with_extension("corrupt");
    std::fs::write(&corrupted, &bytes).map_err(|e| format!("{e}"))?;

    let mut s: Searcher<FileFlash> =
        Searcher::open(&corrupted, None).map_err(|e| format!("{e}"))?;
    let d = s.geometry().d;
    // A query that reaches the damaged block: id 0 lives in it, so a query near
    // the corpus's first vector will select it.
    let q: Vec<f32> = (0..d).map(|j| ((j * 11 % 89) as f32) * 0.75).collect();
    let a = s.search(&q, K).map_err(|e| format!("{e}"))?;
    let dropped = a.stats.rerank.dropped;
    let _ = std::fs::remove_file(&corrupted);
    Ok(dropped)
}

/// Whether a full sweep finds the volume clean.
fn sweep_is_clean(image: &std::path::Path) -> Result<bool, String> {
    let mut f = FileFlash::open(image).map_err(|e| format!("{e}"))?;
    let v = HostVolume::mount(&mut f, None).map_err(|e| format!("{e}"))?;
    let r = verify(&mut f, &v).map_err(|e| format!("{e}"))?;
    Ok(r.is_clean())
}
