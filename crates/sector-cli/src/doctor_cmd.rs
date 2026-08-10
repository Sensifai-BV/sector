//! `sector doctor` — what this machine is, and whether this binary belongs on it.
//!
//! The first thing to run after installing on a Pi, and the check that catches
//! the failure mode the release matrix exists to prevent: an ARMv7 binary
//! installs cleanly on a Pi Zero and dies with `SIGILL` at the first ARMv7-only
//! instruction, which may be inside a code path that does not execute until a
//! query arrives. A binary that starts is not a binary that runs.
//!
//! # Exit codes
//!
//! 0 when the running binary matches the board, 1 when a better-matched artifact
//! exists, 2 when the binary cannot execute correctly here. A packaging script
//! can therefore gate on `sector doctor` and get a three-way answer rather than
//! parsing prose.

use sector_os::json::Json;
use sector_os::platform::{page_size, Abi, AbiStatus, Board};

/// `doctor` arguments.
pub struct Args {
    /// Emit JSON.
    pub json: bool,
}

/// Parse `doctor` arguments.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    Ok(Args {
        json: argv.iter().any(|a| a == "--json"),
    })
}

/// Report the platform. Returns the exit code the verdict implies.
pub fn run(args: Args) -> Result<u8, String> {
    let board = Board::detect();
    let abi = Abi::current();
    let status = board.abi_status();

    if args.json {
        let mut j = Json::new();
        j.object(|o| {
            o.object("board", |b| {
                b.opt_str("model", board.model.as_deref());
                match board.revision {
                    Some(r) => b.str("revision", &format!("{r:x}")),
                    None => b.null("revision"),
                }
                b.opt_str("soc", board.soc);
                b.str("arch", &format!("{:?}", board.arch).to_lowercase());
                b.bool("userland_64bit", board.userland_64bit);
                b.bool("raspberry_pi", board.is_raspberry_pi());
                b.str("tier", board.arch.tier());
            });
            o.object("binary", |x| {
                x.str("abi", &abi.to_string());
                x.str("artifact", abi.artifact());
            });
            o.object("recommended", |x| {
                let want = board.recommended_artifact();
                x.str("abi", &want.to_string());
                x.str("artifact", want.artifact());
            });
            o.object("system", |x| {
                x.uint("page_size", page_size() as u64);
            });
            o.str(
                "verdict",
                match status {
                    AbiStatus::Match => "match",
                    AbiStatus::Suboptimal => "suboptimal",
                    AbiStatus::Incompatible => "incompatible",
                },
            );
        });
        print!("{}", j.finish());
    } else {
        println!("board");
        println!(
            "  model               {}",
            board.model.as_deref().unwrap_or("(not reported)")
        );
        match board.revision {
            Some(r) => println!("  revision            {r:x}"),
            None => println!("  revision            (not reported)"),
        }
        println!("  soc                 {}", board.soc.unwrap_or("(unknown)"));
        println!("  isa                 {:?}", board.arch);
        println!(
            "  userland            {}-bit",
            if board.userland_64bit { 64 } else { 32 }
        );
        println!("  tier                {}", board.arch.tier());
        println!();

        println!("binary");
        println!("  built for           {abi}");
        println!("  artifact            {}", abi.artifact());
        println!();

        println!("system");
        println!("  page size           {} B", page_size());
        // Read rather than assumed: Pi OS on Pi 5 uses 16 KiB pages and every
        // other Pi configuration uses 4 KiB, so a hardcoded 4096 would misreport
        // fault granularity by 4x on exactly one popular setup.
        println!();

        let want = board.recommended_artifact();
        match status {
            AbiStatus::Match => println!("verdict    ok — this binary matches this board"),
            AbiStatus::Suboptimal => {
                println!("verdict    runs, but {} is a better match", want.artifact());
                println!("           this binary works; the recommended one uses the");
                println!("           board's own instruction set.");
            }
            AbiStatus::Incompatible => {
                println!("verdict    WRONG BINARY for this board");
                println!("           install {} instead.", want.artifact());
                // The specific danger: it may already appear to work.
                println!("           an incompatible binary can start and then fail with");
                println!("           SIGILL when a query reaches an unsupported instruction.");
            }
        }
    }

    Ok(match status {
        AbiStatus::Match => 0,
        AbiStatus::Suboptimal => 1,
        AbiStatus::Incompatible => 2,
    })
}
