//! `budget` — memory and disk, measured against the profile's predictions.
//!
//! Every quantity here has a `const fn` prediction from the tier profile. The
//! point is the comparison: a measurement without its prediction cannot falsify
//! anything, and the profile's central claim — that peak RAM is a linker symbol
//! rather than a hope — is only meaningful if a measurement can contradict it.
//!
//! Peak RSS is read during a **query**, not during the build. The build runs on
//! a host with a heap and an FPU by design; the resident claim is about the
//! query path.

use crate::dataset_util;
use crate::{flag, parse_config};
use sector_bench::json::{self, Value};
use sector_bench::pipeline::build_index;
use sector_bench::report::{Claim, Claims};
use std::path::PathBuf;

/// Peak resident set size in bytes, or `None` where the platform does not
/// report it.
///
/// Read from the OS rather than estimated by summing allocations: an estimate
/// would agree with the prediction by construction and could not falsify it.
fn peak_rss_bytes() -> Option<u64> {
    // Linux: VmHWM is the high-water mark, which is what "peak" means here.
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

/// Run `budget`.
pub fn run(argv: &[String]) -> Result<PathBuf, String> {
    let cfg = parse_config(argv)?;
    let base_path = PathBuf::from(flag(argv, "--base")?);
    let out_name = argv
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| argv.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "budget".to_string());

    let base = dataset_util::load(&base_path, cfg.n)?;
    let d = base.dim;
    let n = base.count;
    if !d.is_multiple_of(cfg.m) {
        return Err(format!("D={d} is not divisible by m={}", cfg.m));
    }

    let pipeline = build_index(&base.data, n, d, &cfg)?;

    // Predictions, from the profile arithmetic rather than from the objects.
    let predicted_codebook = (1usize << cfg.b) * d; // 2^b * D * s, s = 1 for int8
                                                    // m * b / 8. The earlier form was `cfg.m` with a comment noting it held
                                                    // "at b = 8" -- and since the measured side made the same assumption, the
                                                    // claim held vacuously at b = 4 while both sides were wrong by 2x. A claim
                                                    // that cannot fail is not a check.
    let predicted_payload = cfg.m * cfg.b / 8;
    let predicted_rerank = d; // int8 rerank record
    let block_bytes = sector_format::BLOCK_BYTES;
    let vectors_per_payload_block = block_bytes / predicted_payload.max(1);
    let payload_blocks = n.div_ceil(vectors_per_payload_block.max(1));
    let rerank_per_block = block_bytes / predicted_rerank.max(1);
    let rerank_blocks = n.div_ceil(rerank_per_block.max(1));
    // Out-of-line CRC array: 4 bytes per block, both regions.
    let crc_bytes = (payload_blocks + rerank_blocks) * sector_codec::CRC_BYTES;
    let replica_bytes = predicted_codebook; // one replica at the default

    let image_bytes = predicted_codebook
        + replica_bytes
        + payload_blocks * block_bytes
        + rerank_blocks * block_bytes
        + crc_bytes;
    let stored_per_vector = image_bytes as f64 / n.max(1) as f64;
    let protection_bytes = replica_bytes + crc_bytes;

    // Measure peak RSS across a query pass, after the build has released its
    // working set as far as it will.
    let mut sink = 0usize;
    for qi in 0..64.min(n) {
        let q = &base.data[qi * d..(qi + 1) * d];
        let candidates = pipeline.stage_one(q, cfg.r);
        sink = sink.wrapping_add(candidates.len());
    }
    let measured_rss = peak_rss_bytes();

    let mut claims = Claims::new();
    claims.push(Claim::new(
        "codebook size",
        predicted_codebook as f64,
        pipeline.codebook_bytes() as f64,
        0.0,
        "bytes",
    ));
    claims.push(Claim::new(
        "payload per vector",
        predicted_payload as f64,
        pipeline.payload_bytes() as f64,
        0.0,
        "bytes",
    ));
    // Stored size: what a device writes to flash, at m * b / 8 per vector.
    claims.push(Claim::new(
        "payload stored",
        (n * predicted_payload) as f64,
        (n * pipeline.payload_bytes()) as f64,
        0.0,
        "bytes",
    ));
    // Resident size: what this host benchmark holds, one byte per code. It
    // differs from the stored size at b < 8 by design -- unpacking nibbles per
    // lookup costs more host time than the memory saves -- and reporting either
    // as the other would misstate one of the two claims.
    claims.push(Claim::new(
        "codes array resident",
        pipeline.codes_bytes(n) as f64,
        pipeline.codes.len() as f64,
        0.0,
        "bytes",
    ));

    println!("{}", claims.table());
    if !claims.refuted().is_empty() {
        for c in claims.refuted() {
            eprintln!(
                "REFUTED: {} predicted {} measured {}",
                c.name, c.predicted, c.measured
            );
        }
    }

    let value = json::obj(vec![
        ("measurement", json::s("budget")),
        ("dataset", json::s(&base_path.display().to_string())),
        ("config", cfg.to_value(d, n)),
        ("claims", claims.to_value()),
        (
            "disk",
            json::obj(vec![
                ("codebook_bytes", json::i(predicted_codebook as i64)),
                ("codebook_replica_bytes", json::i(replica_bytes as i64)),
                ("payload_blocks", json::i(payload_blocks as i64)),
                ("rerank_blocks", json::i(rerank_blocks as i64)),
                ("crc_bytes", json::i(crc_bytes as i64)),
                ("image_bytes", json::i(image_bytes as i64)),
                ("stored_bytes_per_vector", json::f(stored_per_vector)),
                (
                    "protection_fraction",
                    json::f(protection_bytes as f64 / image_bytes.max(1) as f64),
                ),
                (
                    "note",
                    json::s(
                        "protection = one codebook replica + out-of-line CRC arrays; \
                         the replica is a fixed cost independent of N, so its share \
                         falls as N grows",
                    ),
                ),
            ]),
        ),
        (
            "memory",
            json::obj(vec![
                (
                    "peak_rss_bytes",
                    match measured_rss {
                        Some(v) => json::i(v as i64),
                        None => Value::Num(f64::NAN),
                    },
                ),
                (
                    "note",
                    json::s(
                        "host process RSS, which includes the resident corpus and the \
                         builder's working set — NOT the device workspace figure. The \
                         no_std workspace claim is checked by sector-core's const-fn \
                         assertions, not by this number.",
                    ),
                ),
                ("query_sink", json::i(sink as i64)),
            ]),
        ),
    ]);
    json::write_measurement(&out_name, &value).map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_is_read_from_the_os_where_available() {
        // On Linux this must return a plausible figure; elsewhere it returns
        // None rather than a fabricated one.
        // On Linux VmHWM must be present and plausible; elsewhere the
        // function returns None rather than fabricating a figure.
        let got = peak_rss_bytes();
        if cfg!(target_os = "linux") {
            let v = got.expect("Linux must report VmHWM");
            assert!(v > 1024, "implausible RSS {v}");
        } else {
            assert!(got.is_none_or(|v| v > 1024));
        }
    }
}
