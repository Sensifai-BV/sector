//! The REST surface: routes, request decoding, response shapes.
//!
//! | Route | Method | Purpose |
//! |---|---|---|
//! | `/health` | GET | liveness, no volume access |
//! | `/ready` | GET | readiness: the volume is mounted and answering |
//! | `/info` | GET | geometry, tier, board, resident bytes |
//! | `/stats` | GET | counters since start |
//! | `/search` | POST | one or more queries |
//! | `/vectors` | GET | enumerate stored ids and their records |
//! | `/vectors/{id}` | GET | one stored record |
//!
//! # Why there is no `POST /vectors`
//!
//! Not for the reason it first appears. An append does **not** disturb a mounted
//! reader: it programs only erased blocks, and it installs the new manifest into
//! the *spare* slot, leaving the live one untouched. Every byte a worker has
//! already read stays valid, its CRCs still match, and its `n` still describes a
//! corpus that is really there. A worker that has not remounted is **stale, not
//! wrong** — it answers from the corpus it mounted and cannot see the new vectors.
//! That alone would be a tolerable trade.
//!
//! The blocker is concurrency, and it is specific. With a fixed worker pool every
//! worker is a potential writer, and two appends racing corrupt the *bookkeeping*
//! rather than the data:
//!
//! - `find_head` is a read-then-write with no lock. Two appends that observe the
//!   same erased head both target the same block. The data collision itself fails
//!   loudly — NOR programming cannot set a cleared bit, so the second write is
//!   refused rather than silently blended — so this part is safe by accident of the
//!   medium.
//! - The manifest install is not. Both appends read sequence `S`, both compute the
//!   same spare slot, and both erase it before programming `S + 1`. The second
//!   erase destroys the first's manifest, and
//!   [`sector_format::manifest::select`] breaks an equal-sequence tie by slot
//!   position rather than by recency. So one append's vectors are written, durable,
//!   CRC-valid, and described by no live manifest — orphaned, while its caller was
//!   told it succeeded.
//!
//! `sector_os::append` closes that race with an exclusive file lock held across the
//! whole operation, so concurrent appends are refused rather than serialised — one
//! writer at a time, enforced. What it does *not* give the daemon is a remount
//! barrier: after an ingest, each worker keeps answering from the corpus it mounted
//! until something restarts it, so the pool would disagree about how many vectors
//! exist for as long as it ran. Every answer would be correct; the fleet would not
//! be coherent.
//!
//! Making `POST` worth having therefore needs a remount protocol across workers —
//! quiescing queries, re-reading the manifest, rebinding each workspace — which is a
//! design worth doing deliberately rather than acquiring by adding a route. So
//! ingest is `sector append`, and the daemon is restarted or pointed at the new
//! file.
//!
//! `GET` is served here because reading a stored record needs none of that.
//!
//! One further note for the mapped backend: [`sector_os::MappedFlash`] maps the
//! file's length at open, so appended bytes would appear inside an existing mapping
//! without the manifest that describes them — a worker could read codes for ids its
//! own `n` says do not exist. The daemon uses the buffered backend, so this is a
//! constraint on any future in-process ingest rather than a live hazard.
//!
//! # Request bodies
//!
//! `/search` accepts two encodings, chosen by `Content-Type`:
//!
//! - `application/octet-stream`: `f32` little-endian, `D` per query, back to
//!   back. A `1000 × 128` batch is 512 KB and parses with one length check. This
//!   is the encoding a client library should use.
//! - `text/plain`: whitespace- or comma-separated decimal numbers, one query per
//!   line. For `curl` and for debugging.
//!
//! There is no JSON *input*. Output is JSON because a consumer expects it, but
//! accepting JSON would mean writing a parser, and a parser is attackable in ways
//! a writer is not — for a body whose entire content is `D` numbers, that trade
//! is not worth making. The binary encoding is smaller and faster anyway; the text
//! one covers the case JSON would have served.
//!
//! # Scores are integers
//!
//! Scores come back as the engine's `i32` inner products, not normalised floats.
//! Dividing by a scale to produce a float would imply a similarity metric with
//! units, and the ADC score does not have one — it is monotone in similarity and
//! that is all it claims. A client ranking by it gets the right order; a client
//! reading it as cosine similarity would be wrong, and an integer makes that
//! obvious.

use sector_os::json::Json;
use sector_os::platform::Board;
use sector_os::search::{Answer, HasAccessStats, OpenBackend, SearchError, Searcher};

use crate::http::{Request, Status};

/// A decoded batch of queries.
#[derive(Debug)]
pub struct Batch {
    /// Quantized queries, `D` components each.
    pub queries: Vec<Vec<f32>>,
}

/// Decode a `/search` body.
pub fn decode_batch(req: &Request, d: usize) -> Result<Batch, (Status, String)> {
    if req.body.is_empty() {
        return Err((Status::BadRequest, "empty body".into()));
    }
    let ct = req.content_type.as_deref().unwrap_or("");

    // Text is the explicit opt-in; binary is the default, including for a client
    // that sends no Content-Type at all, because that is the encoding a library
    // uses and a missing header should not silently change how bytes are read.
    if ct.starts_with("text/plain") {
        decode_text(&req.body, d)
    } else {
        decode_binary(&req.body, d)
    }
}

/// `f32` little-endian, `D` per query.
fn decode_binary(body: &[u8], d: usize) -> Result<Batch, (Status, String)> {
    let stride = d * 4;
    if stride == 0 || !body.len().is_multiple_of(stride) {
        return Err((
            Status::BadRequest,
            format!(
                "body is {} B, not a multiple of D*4 = {stride} B",
                body.len()
            ),
        ));
    }
    let queries = body
        .chunks_exact(stride)
        .map(|chunk| {
            chunk
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect();
    Ok(Batch { queries })
}

/// Decimal numbers, one query per line.
fn decode_text(body: &[u8], d: usize) -> Result<Batch, (Status, String)> {
    let text = std::str::from_utf8(body)
        .map_err(|_| (Status::BadRequest, "body is not valid UTF-8".to_string()))?;
    let mut queries = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut q = Vec::with_capacity(d);
        for tok in line.split([',', ' ', '\t']).filter(|t| !t.is_empty()) {
            let v: f32 = tok.parse().map_err(|_| {
                (
                    Status::BadRequest,
                    format!("line {}: `{tok}` is not a number", i + 1),
                )
            })?;
            q.push(v);
        }
        if q.len() != d {
            return Err((
                Status::BadRequest,
                format!("line {} has {} components, expected {d}", i + 1, q.len()),
            ));
        }
        queries.push(q);
    }
    if queries.is_empty() {
        return Err((Status::BadRequest, "no queries in body".into()));
    }
    Ok(Batch { queries })
}

/// Render `/search` results.
pub fn search_response(answers: &[Answer], elapsed_us: f64) -> String {
    let mut j = Json::new();
    j.object(|o| {
        o.array("results", |a| {
            for (i, ans) in answers.iter().enumerate() {
                a.object(|r| {
                    r.uint("query", i as u64);
                    r.uints("ids", ans.ids.iter().map(|x| *x as u64));
                    // Integer inner products, not normalised similarities: the ADC
                    // score is monotone in similarity and carries no units.
                    r.ints("scores", ans.scores.iter().map(|x| *x as i64));
                    r.object("stats", |s| {
                        s.uint("scanned", ans.stats.scan.scanned as u64);
                        s.uint("candidates", ans.stats.rerank.candidates as u64);
                        // Dropped candidates are reported per query, not summed
                        // away: a drop and an eviction are indistinguishable in
                        // the ids, so this is the only signal of corruption.
                        s.uint("dropped", ans.stats.rerank.dropped as u64);
                        s.uint("blocks_verified", ans.stats.rerank.blocks_verified as u64);
                    });
                });
            }
        });
        o.uint("queries", answers.len() as u64);
        o.float("elapsed_us", elapsed_us);
        let dropped: u32 = answers.iter().map(|a| a.stats.rerank.dropped).sum();
        o.uint("dropped_total", dropped as u64);
        // Surfaced at the top level so a client does not have to walk every
        // result to notice the volume is damaged.
        o.bool("degraded", dropped > 0);
    });
    j.finish()
}

/// Render `/info`.
pub fn info_response<F: OpenBackend>(
    searcher: &Searcher<F>,
    workers: usize,
    image: &str,
) -> String {
    let g = *searcher.geometry();
    let board = Board::detect();
    let mut j = Json::new();
    j.object(|o| {
        o.str("image", image);
        o.str("backend", searcher.backend_name());
        o.object("geometry", |x| {
            x.uint("d", g.d as u64);
            x.uint("m", g.m as u64);
            x.uint("centroids", g.centroids as u64);
            x.uint("n", g.n as u64);
            x.uint("r", searcher.depth() as u64);
            x.uint("payload_bytes", g.payload_bytes as u64);
            x.uint("rerank_bytes", g.rerank_bytes as u64);
        });
        o.object("memory", |x| {
            x.uint(
                "resident_bytes_per_worker",
                searcher.resident_bytes() as u64,
            );
            x.uint("workers", workers as u64);
            // The figure a deployment sizes against, stated rather than left to
            // be multiplied out.
            x.uint(
                "resident_bytes_total",
                (searcher.resident_bytes() * workers) as u64,
            );
        });
        o.object("platform", |x| {
            x.opt_str("model", board.model.as_deref());
            x.opt_str("soc", board.soc);
            x.str("arch", &format!("{:?}", board.arch).to_lowercase());
            x.str("tier", board.arch.tier());
            x.str("abi", &sector_os::platform::Abi::current().to_string());
            x.uint("page_size", sector_os::platform::page_size() as u64);
        });
        o.str("version", env!("CARGO_PKG_VERSION"));
    });
    j.finish()
}

/// Render `/vectors` and `/vectors/{id}`.
///
/// Records come back as the stored `i8` bytes, not rescaled floats. The record's
/// quantization scale is per-vector and is not in the manifest, so a float here
/// would carry a scale no reader can verify — see the `no scales in the manifest`
/// note in `sector_quant::adc`.
pub fn vectors_response<F: OpenBackend>(
    searcher: &mut Searcher<F>,
    ids: &[u32],
) -> Result<String, (Status, String)> {
    let rows = searcher
        .records(ids)
        .map_err(|e| (Status::ServerError, e.to_string()))?;
    let m = *searcher.manifest();

    let mut j = Json::new();
    j.object(|o| {
        o.array("vectors", |a| {
            for (id, rec) in ids.iter().zip(rows.iter()) {
                a.object(|v| {
                    v.uint("id", *id as u64);
                    v.str(
                        "status",
                        match rec {
                            Some(_) => "stored",
                            // `absent` and `out of range` are different facts: the
                            // first is a gap inside the volume, the second is past
                            // its end.
                            None if *id < m.n => "absent",
                            None => "out of range",
                        },
                    );
                    match rec {
                        Some(r) => v.ints("record", r.iter().map(|b| *b as i8 as i64)),
                        None => v.null("record"),
                    }
                });
            }
        });
        o.object("volume", |v| {
            v.uint("n", m.n as u64);
            v.uint("stored", m.stored() as u64);
            v.uint("built_n", m.built_n as u64);
            v.uint("appended", m.appended() as u64);
            v.object("gap", |g| {
                g.uint("from", m.gap().0 as u64);
                g.uint("to", m.gap().1 as u64);
            });
        });
    });
    Ok(j.finish())
}

/// Storage counters, summed across every worker.
///
/// # Why these are not read from the serving worker's own backend
///
/// Each worker holds its own `Searcher` with its own two backend handles, so a
/// worker's access counters describe only the requests *it* served. `/stats`
/// previously read them from whichever worker happened to accept the `/stats`
/// request — which, with more than one worker, is usually not the one that did the
/// work.
///
/// Measured on a Pi 4 with two workers: after 40 concurrent `/ready` requests and
/// a `/search` that read 19.8 MB, `/stats` reported **0 reads and 0 bytes**,
/// because the scrape landed on the idle worker. The same daemon with one worker
/// reported the reads correctly, so the bug was invisible in the single-worker case
/// and in every in-memory test.
///
/// A monitoring check reading zero I/O from a daemon that is serving traffic would
/// conclude the volume was fully cached. Workers therefore fold their deltas into
/// this shared total, and `/stats` reports the sum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Storage {
    /// `read` calls issued.
    pub reads: u64,
    /// Bytes requested.
    pub bytes: u64,
    /// Device blocks touched, counting a straddling read more than once.
    pub blocks_touched: u64,
    /// Reads that crossed a device block boundary.
    pub straddling_reads: u64,
    /// Short reads the kernel returned and the backend retried.
    pub short_reads: u64,
}

impl Storage {
    /// Add one worker's counters.
    pub fn add(&mut self, a: &sector_os::file::AccessStats) {
        self.reads += a.reads;
        self.bytes += a.bytes;
        self.blocks_touched += a.blocks_touched;
        self.straddling_reads += a.straddling_reads;
        self.short_reads += a.short_reads;
    }
}

/// Counters a daemon accumulates.
#[derive(Debug, Default)]
pub struct Counters {
    /// Requests served, by outcome.
    pub requests: u64,
    /// Queries answered.
    pub queries: u64,
    /// Requests rejected before reaching the engine.
    pub rejected: u64,
    /// Candidates dropped across every query.
    pub dropped: u64,
    /// Cumulative query time.
    pub query_us: f64,
    /// Storage counters summed across every worker; see [`Storage`].
    pub storage: Storage,
}

/// Render `/stats`.
pub fn stats_response<F>(searcher: &Searcher<F>, counters: &Counters, uptime_s: u64) -> String
where
    F: OpenBackend + HasAccessStats,
{
    // The shared total, not this worker's handle. Reading the serving worker's own
    // counters reported 0 bytes on a two-worker daemon that had just read 19.8 MB,
    // because the scrape landed on the idle worker — see `Storage`.
    let access = counters.storage;
    let _ = searcher;
    let mut j = Json::new();
    j.object(|o| {
        o.uint("uptime_s", uptime_s);
        o.object("requests", |x| {
            x.uint("served", counters.requests);
            x.uint("rejected", counters.rejected);
        });
        o.object("queries", |x| {
            x.uint("answered", counters.queries);
            x.float(
                "mean_us",
                if counters.queries == 0 {
                    0.0
                } else {
                    counters.query_us / counters.queries as f64
                },
            );
            x.uint("candidates_dropped", counters.dropped);
        });
        o.object("storage", |x| {
            x.uint("reads", access.reads);
            x.uint("bytes", access.bytes);
            x.uint("device_blocks", access.blocks_touched);
            x.uint("straddling_reads", access.straddling_reads);
            // Non-zero means the volume is being read from something that returns
            // partial requests, which changes the per-read cost model.
            x.uint("short_reads", access.short_reads);
        });
    });
    j.finish()
}

/// Render a search error.
///
/// A dimension mismatch or a non-finite component is the client's fault and gets
/// 400; anything else is the server's and gets 500. Returning 500 for a bad
/// request would make a client retry a request that can never succeed.
pub fn search_error(e: &SearchError) -> (Status, String) {
    match e {
        SearchError::Dimension { .. } | SearchError::NonFinite { .. } => {
            (Status::BadRequest, e.to_string())
        }
        SearchError::DepthTooLarge { .. } => (Status::BadRequest, e.to_string()),
        _ => (Status::ServerError, e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(ct: Option<&str>, body: &[u8]) -> Request {
        Request {
            method: "POST".into(),
            path: "/search".into(),
            query: String::new(),
            body: body.to_vec(),
            keep_alive: true,
            content_type: ct.map(|s| s.to_string()),
        }
    }

    #[test]
    fn a_binary_batch_decodes_to_the_right_shape() {
        let mut body = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            body.extend_from_slice(&v.to_le_bytes());
        }
        let b = decode_batch(&req(Some("application/octet-stream"), &body), 3).expect("decode");
        assert_eq!(b.queries.len(), 2);
        assert_eq!(b.queries[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(b.queries[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn a_missing_content_type_is_treated_as_binary() {
        // A library that forgets the header must not have its bytes reinterpreted
        // as text.
        let mut body = Vec::new();
        for v in [1.0f32, 2.0] {
            body.extend_from_slice(&v.to_le_bytes());
        }
        let b = decode_batch(&req(None, &body), 2).expect("decode");
        assert_eq!(b.queries[0], vec![1.0, 2.0]);
    }

    #[test]
    fn a_binary_body_of_the_wrong_length_is_rejected_with_the_expected_stride() {
        // Truncating to a whole number of queries would answer a different
        // question than the client asked.
        let (status, detail) = decode_batch(&req(None, &[0u8; 9]), 2).unwrap_err();
        assert_eq!(status, Status::BadRequest);
        assert!(detail.contains("D*4 = 8"), "{detail}");
    }

    #[test]
    fn text_bodies_accept_commas_spaces_and_tabs() {
        let b = decode_batch(&req(Some("text/plain"), b"1, 2 3\n4\t5,6\n"), 3).expect("decode");
        assert_eq!(b.queries.len(), 2);
        assert_eq!(b.queries[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn a_text_line_of_the_wrong_width_names_the_line() {
        let (status, detail) =
            decode_batch(&req(Some("text/plain"), b"1 2 3\n4 5\n"), 3).unwrap_err();
        assert_eq!(status, Status::BadRequest);
        assert!(detail.contains("line 2"), "{detail}");
    }

    #[test]
    fn a_non_numeric_token_names_itself() {
        let (_, detail) = decode_batch(&req(Some("text/plain"), b"1 two 3\n"), 3).unwrap_err();
        assert!(detail.contains("`two`"), "{detail}");
    }

    #[test]
    fn an_empty_body_is_rejected() {
        assert_eq!(
            decode_batch(&req(None, b""), 4).unwrap_err().0,
            Status::BadRequest
        );
    }

    #[test]
    fn a_client_error_is_400_and_a_server_error_is_500() {
        // A 500 on a malformed request would make a client retry forever.
        assert_eq!(
            search_error(&SearchError::Dimension {
                found: 3,
                expected: 4
            })
            .0,
            Status::BadRequest
        );
        assert_eq!(
            search_error(&SearchError::NonFinite { at: 2 }).0,
            Status::BadRequest
        );
        assert_eq!(
            search_error(&SearchError::Internal("x")).0,
            Status::ServerError
        );
    }

    #[test]
    fn a_degraded_response_is_flagged_at_the_top_level() {
        // So a client notices corruption without walking every result.
        use sector_core::query::QueryStats;
        let mut stats = QueryStats::default();
        stats.rerank.dropped = 4;
        let answers = vec![Answer {
            ids: vec![1, 2],
            scores: vec![10, 5],
            stats,
        }];
        let text = search_response(&answers, 123.0);
        assert!(text.contains(r#""degraded":true"#), "{text}");
        assert!(text.contains(r#""dropped_total":4"#), "{text}");
        assert!(text.contains(r#""scores":[10,5]"#), "{text}");
    }

    #[test]
    fn a_clean_response_is_not_flagged() {
        let answers = vec![Answer {
            ids: vec![7],
            scores: vec![99],
            stats: sector_core::query::QueryStats::default(),
        }];
        let text = search_response(&answers, 1.0);
        assert!(text.contains(r#""degraded":false"#), "{text}");
    }
}
