//! `sector` — command-line tools for SECTOR volumes.
//!
//! | Subcommand | Purpose |
//! |---|---|
//! | `build` | train, encode and protect a corpus into a volume image |
//! | `inspect` | dump manifest, regions, protection groups and budgets |
//! | `query` | run queries against a volume image on the host |
//! | `falsify` | run the falsification suite, reporting pass or refuted per claim |
//!
//! # Implementation notes
//!
//! `query` is byte-identical to the device path — same mount code, same scan,
//! same rerank — so a host/device recall discrepancy means a backend or
//! hardware problem rather than two diverging implementations.
//!
//! `inspect` prints computed budgets (resident bytes, stored bytes per vector,
//! capacity at this volume's profile) rather than only the stored header
//! fields. The recurring failure in this project has been a configuration that
//! looks reasonable and does not fit, and recomputing the arithmetic catches it
//! before the device does.

mod build_cmd;
mod falsify_cmd;
mod inspect_cmd;
mod query_cmd;

use std::process::ExitCode;

/// Parsed command line.
enum Command {
    Build(build_cmd::Args),
    Inspect(inspect_cmd::Args),
    Query(query_cmd::Args),
    Falsify,
    Help,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match parse(&argv) {
        Ok(Command::Build(a)) => report(build_cmd::run(a)),
        Ok(Command::Inspect(a)) => report(inspect_cmd::run(a)),
        Ok(Command::Query(a)) => report(query_cmd::run(a)),
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
        "build" => build_cmd::parse(rest).map(Command::Build),
        "inspect" => inspect_cmd::parse(rest).map(Command::Inspect),
        "query" => query_cmd::parse(rest).map(Command::Query),
        "falsify" => Ok(Command::Falsify),
        "help" | "-h" | "--help" => Ok(Command::Help),
        other => Err(format!("unknown subcommand `{other}`")),
    }
}

fn print_usage() {
    println!("sector — command-line tools for SECTOR volumes\n");
    println!("USAGE:");
    println!("  sector build   --input <.fvecs> --out <image> [--d N --m N --b N --seed N]");
    println!("  sector inspect --image <image>");
    println!("  sector query   --image <image> --queries <.fvecs> [--k N --r N]");
    println!("  sector falsify");
    println!();
    println!("`falsify` exits non-zero when a claim is refuted.");
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
