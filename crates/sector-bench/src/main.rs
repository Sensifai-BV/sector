//! `sector-bench` — five-axis benchmark harness.
//!
//! | Subcommand | Axis |
//! |---|---|
//! | `recall` | recall against a dataset's shipped ground truth |
//! | `perf` | per-phase latency and bytes, and the energy model's inputs |
//! | `budget` | peak RSS and image bytes against the profile's predictions |
//! | `faults` | recall under each fault channel at increasing rates |
//!
//! Every subcommand writes JSON to `measurements/`. The plotting reads those
//! files rather than recomputing anything, so a figure cannot disagree with the
//! measurement it draws.

use std::process::ExitCode;

mod budget_cmd;
mod dataset_util;
mod faults_cmd;
mod perf_cmd;
mod recall_cmd;

use sector_bench::Config;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(verb) = argv.first().map(|s| s.as_str()) else {
        usage();
        return ExitCode::SUCCESS;
    };
    let rest = &argv[1..];

    let result = match verb {
        "recall" => recall_cmd::run(rest),
        "perf" => perf_cmd::run(rest),
        "budget" => budget_cmd::run(rest),
        "faults" => faults_cmd::run(rest),
        "help" | "-h" | "--help" => {
            usage();
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown subcommand `{other}`")),
    };

    match result {
        Ok(path) => {
            println!("\nwrote {}", path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    println!("sector-bench — five-axis benchmark harness\n");
    println!("USAGE:");
    println!("  sector-bench recall --base <.fvecs> --queries <.fvecs> --truth <.ivecs> [--n N --m N --b N]");
    println!("  sector-bench perf   --base <.fvecs> --queries <.fvecs> [--n N --m N --b N --r N]");
    println!("  sector-bench budget --base <.fvecs> [--n N --m N --b N]");
    println!("  sector-bench faults --base <.fvecs> --queries <.fvecs> --truth <.ivecs> [--n N --seed N]");
    println!();
    println!("Results are written to measurements/<name>.json.");
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

/// Parse the shared configuration flags.
pub fn parse_config(argv: &[String]) -> Result<Config, String> {
    Ok(Config {
        n: opt_num(argv, "--n", 0)?,
        m: opt_num(argv, "--m", 16)?,
        b: opt_num(argv, "--b", 8)?,
        r: opt_num(argv, "--r", 100)?,
        k: opt_num(argv, "--k", 10)?,
        seed: opt_num(argv, "--seed", 42)? as u64,
        train_n: opt_num(argv, "--train-n", 100_000)?,
    })
}
