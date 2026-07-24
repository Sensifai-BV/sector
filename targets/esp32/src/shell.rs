//! UART command shell.
//!
//! The device half of the host/device round-trip. The host runs
//! `sector query --image <img>`; the device runs `q <index>` against the same
//! image and must return the same ids and scores. A discrepancy then means a
//! backend or hardware problem rather than two diverging implementations.
//!
//! Commands are line-oriented and the output is machine-readable, because a
//! human-friendly format that a host-side comparison has to parse loosely is
//! where a real difference hides.

/// A parsed command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    /// Mount the volume and print its profile.
    Mount,
    /// Run query `index` from the flashed query set.
    Query {
        /// Query index.
        index: u32,
    },
    /// Print per-phase counters from the last query.
    Metrics,
    /// Verify every block CRC and report failures.
    Scrub,
    /// Print the partition and region map.
    Info,
}

/// Why a command line was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// The line was empty.
    Empty,
    /// The verb is not a command.
    UnknownVerb,
    /// A required argument was missing or not a number.
    BadArgument,
}

/// Parse one line.
pub fn parse(line: &str) -> Result<Cmd, ParseError> {
    let mut parts = line.split_whitespace();
    let verb = parts.next().ok_or(ParseError::Empty)?;
    match verb {
        "mount" | "m" => Ok(Cmd::Mount),
        "query" | "q" => {
            let arg = parts.next().ok_or(ParseError::BadArgument)?;
            let index = arg.parse::<u32>().map_err(|_| ParseError::BadArgument)?;
            Ok(Cmd::Query { index })
        }
        "metrics" => Ok(Cmd::Metrics),
        "scrub" => Ok(Cmd::Scrub),
        "info" | "i" => Ok(Cmd::Info),
        _ => Err(ParseError::UnknownVerb),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_parse_with_and_without_their_short_forms() {
        assert_eq!(parse("mount"), Ok(Cmd::Mount));
        assert_eq!(parse("m"), Ok(Cmd::Mount));
        assert_eq!(parse("q 7"), Ok(Cmd::Query { index: 7 }));
        assert_eq!(parse("query 0"), Ok(Cmd::Query { index: 0 }));
        assert_eq!(parse("info"), Ok(Cmd::Info));
        assert_eq!(parse("metrics"), Ok(Cmd::Metrics));
        assert_eq!(parse("scrub"), Ok(Cmd::Scrub));
    }

    #[test]
    fn a_malformed_line_is_rejected_rather_than_defaulted() {
        // Defaulting a bad query index to 0 would silently compare the wrong
        // query against the host and report a false discrepancy.
        assert_eq!(parse(""), Err(ParseError::Empty));
        assert_eq!(parse("   "), Err(ParseError::Empty));
        assert_eq!(parse("frobnicate"), Err(ParseError::UnknownVerb));
        assert_eq!(parse("q"), Err(ParseError::BadArgument));
        assert_eq!(parse("q seven"), Err(ParseError::BadArgument));
        assert_eq!(parse("q -1"), Err(ParseError::BadArgument));
    }

    #[test]
    fn leading_and_trailing_whitespace_is_tolerated() {
        // Serial terminals append carriage returns and stray spaces.
        assert_eq!(parse("  q 3  "), Ok(Cmd::Query { index: 3 }));
        assert_eq!(parse("\tmount\r"), Ok(Cmd::Mount));
    }
}
