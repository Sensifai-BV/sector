//! `sector` — command-line tools for SECTOR volumes.
//!
//! | Subcommand | Purpose |
//! |---|---|
//! | `build` | train, encode and protect a corpus into a volume image |
//! | `append` | add vectors to a volume built with `--reserve` |
//! | `vectors` | read back a stored record, or enumerate a range |
//! | `inspect` | dump manifest, regions, protection groups and budgets |
//! | `query` | run queries against a volume image through the engine |
//! | `stats` | measure what a volume costs: resident bytes, latency by phase, reads per query |
//! | `verify` | sweep every CRC-protected region and report damage in vectors |
//! | `repair` | diagnose damage and say whether it can be repaired |
//! | `doctor` | report the board, the binary's ABI, and whether they match |
//! | `selftest` | build, query and corrupt a volume on this machine |
//! | `serve` | run the query daemon on a Unix socket and HTTP |
//! | `falsify` | run the falsification suite, reporting pass or refuted per claim |
//!
//! # Implementation notes
//!
//! `query` calls `sector_core::query` through `sector_os::Searcher` — the same
//! mount, scan and CRC-verified rerank the firmware runs — so a host/device
//! recall discrepancy means a backend or hardware problem rather than two
//! diverging implementations. That equivalence is now asserted by the round-trip
//! test rather than only claimed here: an earlier version of `query_cmd`
//! reimplemented stage one in floating point and made this claim while
//! performing no rerank at all.
//!
//! `inspect` prints computed budgets (resident bytes, stored bytes per vector,
//! capacity at this volume's profile) rather than only the stored header
//! fields. The recurring failure in this project has been a configuration that
//! looks reasonable and does not fit, and recomputing the arithmetic catches it
//! before the device does.
//!
//! # Exit codes
//!
//! `0` success, `1` a substantive negative result — a refuted claim, a damaged
//! volume, a failed check — and `2` a suboptimal-but-working state, which at
//! present only `doctor` reports. Every command that can answer "no" does so
//! through the exit code as well as its output, so a monitoring check or a
//! packaging script does not have to parse prose.
//!
//! `--json` is available on every reporting command for the same reason: a check
//! that scrapes human-readable columns breaks the first time a column is added.

mod append_cmd;
mod build_cmd;
mod doctor_cmd;
mod falsify_cmd;
mod inspect_cmd;
mod query_cmd;
mod selftest_cmd;
mod serve_cmd;
mod stats_cmd;
mod verify_cmd;

use std::process::ExitCode;

/// Parsed command line.
enum Command {
    Version,
    Append(append_cmd::Args),
    Vectors(append_cmd::VectorsArgs),
    Build(build_cmd::Args),
    Inspect(inspect_cmd::Args),
    Query(query_cmd::Args),
    Stats(stats_cmd::Args),
    Verify(verify_cmd::Args),
    Repair(verify_cmd::RepairArgs),
    Doctor(doctor_cmd::Args),
    Selftest(selftest_cmd::Args),
    Serve(serve_cmd::Args),
    Falsify,
    Help,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match parse(&argv) {
        Ok(Command::Build(a)) => report(build_cmd::run(a)),
        Ok(Command::Inspect(a)) => report(inspect_cmd::run(a)),
        Ok(Command::Version) => {
            println!(
                "sector {} (on-disk format {})",
                env!("CARGO_PKG_VERSION"),
                sector_format::FORMAT_VERSION
            );
            ExitCode::SUCCESS
        }
        Ok(Command::Append(a)) => report(append_cmd::run(a)),
        Ok(Command::Vectors(a)) => report(append_cmd::run_vectors(a)),
        Ok(Command::Query(a)) => report(query_cmd::run(a)),
        Ok(Command::Stats(a)) => report(stats_cmd::run(a)),
        Ok(Command::Serve(a)) => report(serve_cmd::run(a)),
        // A damaged volume is a result, not an error: the sweep succeeded and
        // found something. Exit 1 so `sector verify || alert` works.
        Ok(Command::Verify(a)) => match verify_cmd::run(a) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(damaged) => {
                eprintln!("{damaged} region(s) damaged");
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!("error: {e}");
                // 2 distinguishes "could not check" from "checked and found
                // damage": they call for different responses.
                ExitCode::from(2)
            }
        },
        Ok(Command::Repair(a)) => report(verify_cmd::run_repair(a)),
        // `doctor` has a three-way verdict, so it does not go through `report`.
        Ok(Command::Doctor(a)) => match doctor_cmd::run(a) {
            Ok(code) => ExitCode::from(code),
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::Selftest(a)) => match selftest_cmd::run(a) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(failed) => {
                eprintln!("{failed} check(s) failed");
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        // `falsify` exits non-zero on a refuted claim, so CI fails loudly
        // rather than recording a refutation in a log nobody reads.
        Ok(Command::Falsify) => match falsify_cmd::run() {
            Ok(0) => ExitCode::SUCCESS,
            Ok(refuted) => {
                eprintln!("{refuted} claim(s) refuted");
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::Help) => {
            print_usage();
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}\n");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn report(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse(argv: &[String]) -> Result<Command, String> {
    let Some(first) = argv.first() else {
        return Ok(Command::Help);
    };
    let rest = &argv[1..];
    match first.as_str() {
        "append" => append_cmd::parse(rest).map(Command::Append),
        "vectors" => append_cmd::parse_vectors(rest).map(Command::Vectors),
        "build" => build_cmd::parse(rest).map(Command::Build),
        "inspect" => inspect_cmd::parse(rest).map(Command::Inspect),
        "query" => query_cmd::parse(rest).map(Command::Query),
        "stats" => stats_cmd::parse(rest).map(Command::Stats),
        "verify" => verify_cmd::parse(rest).map(Command::Verify),
        "repair" => verify_cmd::parse_repair(rest).map(Command::Repair),
        "doctor" => doctor_cmd::parse(rest).map(Command::Doctor),
        "selftest" => selftest_cmd::parse(rest).map(Command::Selftest),
        "serve" => serve_cmd::parse(rest).map(Command::Serve),
        "falsify" => Ok(Command::Falsify),
        "help" | "-h" | "--help" => Ok(Command::Help),
        // `--version` is the one flag every packaging script, CI smoke test and
        // distribution check reaches for first. Its absence is not cosmetic: the
        // release workflow's post-build check and install.sh both probe a binary
        // before trusting it, and a binary that exits non-zero on `--version`
        // looks broken.
        //
        // It reports the on-disk FORMAT_VERSION alongside the crate version,
        // because those are independent compatibility axes: a reader refuses an
        // unknown format outright, so "which sector am I running" and "which
        // volumes can it read" are two different questions and an operator
        // holding a volume needs the second one answered.
        "version" | "-V" | "--version" => Ok(Command::Version),
        other => Err(format!("unknown subcommand `{other}`")),
    }
}

fn print_usage() {
    println!("sector — command-line tools for SECTOR volumes\n");
    println!("BUILD AND INSPECT");
    println!("  sector build    --input <.fvecs> --out <image> [--m N --b N --r N --seed N]");
    println!("  sector inspect  --image <image>");
    println!("  sector append   --image <image> --input <.fvecs> [--dry-run] [--json]");
    println!("  sector vectors  --image <image> [--id N | --from N --count N] [--json]");
    println!();
    println!("SEARCH");
    println!("  sector query    --image <image> --queries <.fvecs> [--k N --r N --limit N]");
    println!("  sector stats    --image <image> [--queries <.fvecs> --count N --k N] [--json]");
    println!();
    println!("INTEGRITY");
    println!("  sector verify   --image <image> [--json]");
    println!("  sector repair   --image <image> [--dry-run]");
    println!();
    println!("PLATFORM");
    println!("  sector doctor   [--json]");
    println!("  sector selftest [--n N] [--json] [--keep]");
    println!("  sector falsify");
    println!();
    println!("DAEMON");
    println!("  sector serve    --image <image> [--socket <path>] [--listen <addr:port>]");
    println!("                  [--workers N] [--k N] [--r N]");
    println!();
    println!("EXIT CODES");
    println!("  0  success");
    println!("  1  a negative result: refuted claim, damaged volume, failed check");
    println!("  2  works but suboptimal (`doctor`), or could not check (`verify`)");
}

/// Read a required `--name value` flag.
pub fn flag<'a>(argv: &'a [String], name: &str) -> Result<&'a str, String> {
    argv.iter()
        .position(|a| a == name)
        .and_then(|i| argv.get(i + 1))
        .map(|s| s.as_str())
        .ok_or_else(|| format!("missing required flag {name}"))
}

/// Read an optional numeric flag.
pub fn opt_num(argv: &[String], name: &str, default: usize) -> Result<usize, String> {
    match argv.iter().position(|a| a == name) {
        None => Ok(default),
        Some(i) => argv
            .get(i + 1)
            .ok_or_else(|| format!("{name} needs a value"))?
            .parse()
            .map_err(|_| format!("{name} must be a number")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn no_arguments_prints_help_rather_than_failing() {
        assert!(matches!(parse(&[]), Ok(Command::Help)));
    }

    #[test]
    fn an_unknown_subcommand_is_an_error_not_a_silent_default() {
        assert!(parse(&args(&["frobnicate"])).is_err());
    }

    #[test]
    fn a_missing_required_flag_names_itself() {
        let a = args(&["--out", "x"]);
        let err = flag(&a, "--input").unwrap_err();
        assert!(err.contains("--input"), "unhelpful message: {err}");
    }

    #[test]
    fn optional_numeric_flags_default_and_parse() {
        let a = args(&["--k", "25"]);
        assert_eq!(opt_num(&a, "--k", 10).unwrap(), 25);
        assert_eq!(opt_num(&a, "--r", 100).unwrap(), 100);
        let bad = args(&["--k", "many"]);
        assert!(opt_num(&bad, "--k", 10).is_err());
    }
}
