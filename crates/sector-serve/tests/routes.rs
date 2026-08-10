//! Every route, driven over an in-memory stream.
//!
//! The companion to `daemon.rs`, and the reason it exists: some sandboxes forbid
//! `bind`, and a test that skips reports green without checking anything. These
//! tests exercise the same request handler the sockets use — routing, parsing,
//! keep-alive, every limit and every status code — with no socket at all, so they
//! run everywhere including under `qemu-user` in CI.
//!
//! What they do *not* cover is the transport: accept, the worker pool, and the
//! shutdown path. `daemon.rs` covers those where binding is permitted, and the
//! division is deliberate rather than a gap.

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use sector_os::search::Searcher;
use sector_os::volume::test_support::{build_image_and_corpus, TempDir};
use sector_os::FileFlash;
use sector_serve::server::Duplex;

const D: usize = 32;
const M: usize = 4;
const N: usize = 300;

/// A bidirectional in-memory stream: the request to read, the response captured.
///
/// `dup()` returns a handle sharing the same buffers, which is what the server
/// does with a socket's `try_clone`.
#[derive(Clone)]
struct MemStream {
    input: Arc<Mutex<std::io::Cursor<Vec<u8>>>>,
    output: Arc<Mutex<Vec<u8>>>,
}

impl MemStream {
    fn new(request: &[u8]) -> Self {
        Self {
            input: Arc::new(Mutex::new(std::io::Cursor::new(request.to_vec()))),
            output: Arc::new(Mutex::new(Vec::new())),
        }
    }
    fn response(&self) -> String {
        String::from_utf8_lossy(&self.output.lock().expect("lock")).to_string()
    }
}

impl Read for MemStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.input.lock().expect("lock").read(buf)
    }
}

impl Write for MemStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.output.lock().expect("lock").write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Duplex for MemStream {
    fn dup(&self) -> std::io::Result<Self> {
        Ok(self.clone())
    }
}

/// A volume and a searcher over it.
struct Fixture {
    _dir: TempDir,
    searcher: Searcher<FileFlash>,
    corpus: Vec<f32>,
    image: String,
    /// The counter set this worker folds into. Shared in the real daemon, which is
    /// the property `stats_reports_storage_read_by_a_different_worker` exercises.
    counters: std::sync::Arc<std::sync::Mutex<sector_serve::api::Counters>>,
}

fn fixture(tag: &str) -> Fixture {
    let dir = TempDir::new(tag);
    let (image, corpus) = build_image_and_corpus(D, M, N);
    let path = dir.path().join("volume.sector");
    std::fs::write(&path, &image).expect("write volume");
    let searcher = Searcher::open(&path, None).expect("open");
    Fixture {
        _dir: dir,
        searcher,
        corpus,
        image: path.display().to_string(),
        counters: std::sync::Arc::new(std::sync::Mutex::new(Default::default())),
    }
}

impl Fixture {
    /// Serve one raw request and return the full response.
    fn serve(&mut self, raw: &[u8]) -> String {
        let stream = MemStream::new(raw);
        // Through the shared-counter entry point, so a worker's storage deltas
        // accumulate across requests the way the daemon's do rather than resetting
        // per connection.
        sector_serve::server::serve_connection_with(
            stream.clone(),
            &mut self.searcher,
            10,
            &self.image,
            2,
            &self.counters,
        );
        stream.response()
    }

    /// This worker's counter set, for handing to another worker.
    fn counters_snapshot(&self) -> std::sync::Arc<std::sync::Mutex<sector_serve::api::Counters>> {
        std::sync::Arc::clone(&self.counters)
    }

    /// Serve from someone else's counter set: what a second daemon worker does.
    fn adopt_counters(
        &mut self,
        shared: std::sync::Arc<std::sync::Mutex<sector_serve::api::Counters>>,
    ) {
        self.counters = shared;
    }

    fn get(&mut self, path: &str) -> String {
        self.serve(
            format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(),
        )
    }

    fn post(&mut self, path: &str, content_type: &str, body: &[u8]) -> String {
        let mut raw = format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: {content_type}\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        raw.extend_from_slice(body);
        self.serve(&raw)
    }

    /// A binary body for the first `count` corpus vectors.
    fn binary(&self, count: usize) -> Vec<u8> {
        let mut body = Vec::new();
        for &x in &self.corpus[..count * D] {
            body.extend_from_slice(&x.to_le_bytes());
        }
        body
    }
}

/// Extract a JSON number by key.
fn number(json: &str, key: &str) -> Option<f64> {
    let at = json.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = &json[at..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The `ids` array of the first result.
fn first_ids(json: &str) -> String {
    let at = json.find("\"ids\":").expect("ids in response");
    let rest = &json[at..];
    rest[..rest.find(']').expect("close bracket") + 1].to_string()
}

#[test]
fn health_answers_without_touching_the_volume() {
    let mut f = fixture("r_health");
    let r = f.get("/health");
    assert!(r.starts_with("HTTP/1.1 200 OK"), "{r}");
    assert!(r.contains("Content-Type: application/json"), "{r}");
    assert!(r.contains(r#""status":"ok""#), "{r}");
}

#[test]
fn ready_runs_a_real_query() {
    // Liveness and readiness are distinct: only a query proves a mounted volume
    // can answer.
    let mut f = fixture("r_ready");
    let r = f.get("/ready");
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(r.contains(r#""ready":true"#), "{r}");
}

#[test]
fn info_reports_geometry_the_tier_and_the_resident_total() {
    let mut f = fixture("r_info");
    let r = f.get("/info");
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert_eq!(number(&r, "n"), Some(N as f64), "{r}");
    assert_eq!(number(&r, "d"), Some(D as f64), "{r}");
    assert_eq!(number(&r, "m"), Some(M as f64), "{r}");
    assert_eq!(number(&r, "workers"), Some(2.0), "{r}");
    // The figure a deployment sizes against, stated rather than left to be
    // multiplied out by the reader.
    let per = number(&r, "resident_bytes_per_worker").expect("per worker");
    assert_eq!(number(&r, "resident_bytes_total"), Some(per * 2.0), "{r}");
    // The tier this board maps to, so an operator can check it against the
    // profile the volume was built for.
    assert!(r.contains(r#""tier":"#), "{r}");
    assert!(r.contains(r#""backend":"file""#), "{r}");
}

#[test]
fn a_binary_search_returns_ranked_ids_and_proves_stage_two_ran() {
    let mut f = fixture("r_bin");
    let body = f.binary(1);
    let r = f.post("/search?k=5", "application/octet-stream", &body);
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert_eq!(number(&r, "queries"), Some(1.0), "{r}");
    assert!(r.contains(r#""degraded":false"#), "{r}");
    // Every vector scanned: a payload reader that stopped early would silently
    // reduce recall rather than fail.
    assert_eq!(number(&r, "scanned"), Some(N as f64), "{r}");
    // A CRC was checked, which is the observable evidence of stage two.
    assert!(
        number(&r, "blocks_verified").unwrap_or(0.0) > 0.0,
        "stage two did not run: {r}"
    );
    // Scores are integers, not floats: the ADC score carries no units.
    assert!(!first_ids(&r).contains('.'), "{r}");
}

#[test]
fn the_two_body_encodings_return_identical_ids() {
    // `curl` and a client library must agree about what the same query means.
    let mut f = fixture("r_encodings");
    let body = f.binary(1);
    let from_bin = f.post("/search?k=5", "application/octet-stream", &body);

    let text: String = f.corpus[..D]
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let from_txt = f.post("/search?k=5", "text/plain", text.as_bytes());
    assert!(from_txt.starts_with("HTTP/1.1 200"), "{from_txt}");
    assert_eq!(
        first_ids(&from_bin),
        first_ids(&from_txt),
        "the two encodings disagree"
    );
}

#[test]
fn a_missing_content_type_is_read_as_binary() {
    // A library that forgets the header must not have its bytes reinterpreted.
    let mut f = fixture("r_noct");
    let body = f.binary(1);
    let mut raw = format!(
        "POST /search?k=3 HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    raw.extend_from_slice(&body);
    let r = f.serve(&raw);
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert_eq!(number(&r, "queries"), Some(1.0), "{r}");
}

#[test]
fn a_batch_returns_one_result_per_query() {
    let mut f = fixture("r_batch");
    let body = f.binary(4);
    let r = f.post("/search?k=3", "application/octet-stream", &body);
    assert_eq!(number(&r, "queries"), Some(4.0), "{r}");
    // Four `"query":` keys, one per result.
    assert_eq!(r.matches(r#""query":"#).count(), 4, "{r}");
}

#[test]
fn k_is_honoured_and_defaults_when_absent() {
    let mut f = fixture("r_k");
    let body = f.binary(1);
    let r = f.post("/search?k=3", "application/octet-stream", &body);
    let ids = first_ids(&r);
    assert_eq!(ids.matches(',').count(), 2, "expected 3 ids: {ids}");
    // No k: the daemon's default, 10 here.
    let r = f.post("/search", "application/octet-stream", &body);
    let ids = first_ids(&r);
    assert_eq!(ids.matches(',').count(), 9, "expected 10 ids: {ids}");
}

#[test]
fn a_body_of_the_wrong_length_is_400_and_names_the_stride() {
    // Truncating to a whole number of queries would answer a different question
    // than the client asked.
    let mut f = fixture("r_badlen");
    let r = f.post("/search", "application/octet-stream", &[0u8; 9]);
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
    assert!(r.contains("D*4 = 128"), "{r}");
}

#[test]
fn a_malformed_k_is_rejected_rather_than_defaulted() {
    // Answering `k=ten` with k=10 hides a client bug.
    let mut f = fixture("r_badk");
    let body = f.binary(1);
    let r = f.post("/search?k=ten", "application/octet-stream", &body);
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
}

#[test]
fn a_query_of_the_wrong_dimension_is_400_not_500() {
    // A client error must not look like a server error, or the client retries a
    // request that can never succeed.
    let mut f = fixture("r_dim");
    let r = f.post("/search", "text/plain", b"1 2 3\n");
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
    assert!(r.contains("expected 32"), "{r}");
}

#[test]
fn a_non_finite_component_is_refused() {
    // NaN would quantize to zero and silently drop that component from the score.
    let mut f = fixture("r_nan");
    let mut q: Vec<String> = (0..D).map(|_| "1".to_string()).collect();
    q[5] = "NaN".to_string();
    let r = f.post("/search", "text/plain", q.join(" ").as_bytes());
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
}

#[test]
fn an_unknown_route_is_404_and_a_wrong_method_is_405() {
    // The distinction tells a client whether to fix the URL or the verb.
    let mut f = fixture("r_routes");
    assert!(f.get("/nope").starts_with("HTTP/1.1 404"));
    assert!(f.get("/search").starts_with("HTTP/1.1 405"));
    assert!(f
        .post("/health", "text/plain", b"x")
        .starts_with("HTTP/1.1 405"));
    assert!(f
        .post("/info", "text/plain", b"x")
        .starts_with("HTTP/1.1 405"));
}

#[test]
fn an_oversized_content_length_is_413_before_the_body_arrives() {
    // The allocation defence: refused on the announced length, so the bytes are
    // never reserved. This request announces a huge body and sends none.
    let mut f = fixture("r_413");
    let raw = format!(
        "POST /search HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        sector_serve::http::MAX_BODY_BYTES + 1
    );
    let r = f.serve(raw.as_bytes());
    assert!(r.starts_with("HTTP/1.1 413"), "{r}");
    // A protocol rejection closes the connection: the stream position is no
    // longer trustworthy.
    assert!(r.contains("Connection: close"), "{r}");
}

#[test]
fn chunked_encoding_is_411_rather_than_decoded() {
    // Chunked decoding is where hand-written parsers get request smuggling wrong.
    let mut f = fixture("r_411");
    let r =
        f.serve(b"POST /search HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n");
    assert!(r.starts_with("HTTP/1.1 411"), "{r}");
}

#[test]
fn too_many_headers_are_refused_even_when_each_is_tiny() {
    // A byte cap alone permits thousands of one-byte headers.
    let mut f = fixture("r_431");
    let mut raw = String::from("GET /health HTTP/1.1\r\n");
    for i in 0..sector_serve::http::MAX_HEADERS + 5 {
        raw.push_str(&format!("h{i}: v\r\n"));
    }
    raw.push_str("\r\n");
    let r = f.serve(raw.as_bytes());
    assert!(r.starts_with("HTTP/1.1 431"), "{r}");
}

#[test]
fn a_garbage_request_line_is_400() {
    let mut f = fixture("r_400");
    let r = f.serve(b"nonsense\r\n\r\n");
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
}

#[test]
fn keep_alive_serves_several_requests_on_one_connection() {
    // Two requests written up front: a server that closed after the first would
    // return one response.
    let mut f = fixture("r_keepalive");
    let raw = "GET /health HTTP/1.1\r\nHost: x\r\n\r\n\
               GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
    let r = f.serve(raw.as_bytes());
    assert_eq!(
        r.matches("HTTP/1.1 200").count(),
        2,
        "keep-alive served {} responses: {r}",
        r.matches("HTTP/1.1 200").count()
    );
    assert!(r.contains("Connection: keep-alive"), "{r}");
    // The last response closes, because the client asked it to.
    assert!(r.trim_end().ends_with('}'), "{r}");
    assert!(r.contains("Connection: close"), "{r}");
}

#[test]
fn http_1_0_closes_after_one_response() {
    let mut f = fixture("r_http10");
    let r = f.serve(b"GET /health HTTP/1.0\r\n\r\nGET /health HTTP/1.0\r\n\r\n");
    assert_eq!(
        r.matches("HTTP/1.1 200").count(),
        1,
        "1.0 must close after one response: {r}"
    );
}

#[test]
fn stats_reports_counters_and_storage() {
    let mut f = fixture("r_stats");
    let r = f.get("/stats");
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    // Present even at zero, so a scrape does not have to handle a missing key.
    assert!(r.contains(r#""answered":"#), "{r}");
    assert!(r.contains(r#""device_blocks":"#), "{r}");
    assert!(r.contains(r#""short_reads":"#), "{r}");
}

#[test]
fn a_corrupted_volume_degrades_rather_than_failing_the_request() {
    // Corruption reduces the answer; it does not take the service down. The
    // `degraded` flag is how a client notices without walking every result.
    let dir = TempDir::new("r_degraded");
    let (mut image, corpus) = build_image_and_corpus(D, M, N);
    let path = dir.path().join("volume.sector");
    std::fs::write(&path, &image).expect("write");

    // Locate the rerank region from the manifest rather than assuming an offset,
    // so a layout change fails this test loudly instead of corrupting padding.
    let base = {
        let mut fl = FileFlash::open(&path).expect("open");
        let v = sector_os::HostVolume::mount(&mut fl, None).expect("mount");
        v.rerank_base() as usize
    };
    image[base + 3] ^= 0xFF;
    std::fs::write(&path, &image).expect("rewrite");

    let mut f = Fixture {
        _dir: dir,
        searcher: Searcher::open(&path, None).expect("open"),
        corpus,
        image: path.display().to_string(),
        counters: std::sync::Arc::new(std::sync::Mutex::new(Default::default())),
    };
    let body = f.binary(1);
    let r = f.post("/search?k=10", "application/octet-stream", &body);
    assert!(
        r.starts_with("HTTP/1.1 200"),
        "corruption must degrade, not fail: {r}"
    );
    assert!(r.contains(r#""degraded":true"#), "{r}");
    assert!(
        number(&r, "dropped_total").unwrap_or(0.0) > 0.0,
        "drops were not reported: {r}"
    );
}

#[test]
fn a_stored_record_is_readable_by_id() {
    let mut f = fixture("r_vec_one");
    let r = f.get("/vectors/7");
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(r.contains(r#""id":7"#), "{r}");
    assert!(r.contains(r#""status":"stored""#), "{r}");
    // int8 as stored, not rescaled: the record's scale is per-vector and absent
    // from the manifest, so a float would carry a scale nothing can verify.
    assert!(r.contains(r#""record":["#), "{r}");
}

#[test]
fn an_id_past_the_extent_is_reported_rather_than_erroring() {
    // A 404 would conflate "no such route" with "no such vector".
    let mut f = fixture("r_vec_oob");
    let r = f.get(&format!("/vectors/{}", N + 100));
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(r.contains(r#""status":"out of range""#), "{r}");
    assert!(r.contains(r#""record":null"#), "{r}");
}

#[test]
fn a_non_numeric_id_is_400() {
    let mut f = fixture("r_vec_badid");
    let r = f.get("/vectors/abc");
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
}

#[test]
fn enumerating_reports_the_volume_extent_and_gap() {
    let mut f = fixture("r_vec_range");
    let r = f.get("/vectors?from=10&count=4");
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert_eq!(r.matches(r#""id":"#).count(), 4, "{r}");
    // `stored` and `n` differ on an appended volume; both are reported so a client
    // never has to infer one from the other.
    assert_eq!(number(&r, "stored"), Some(N as f64), "{r}");
    assert_eq!(number(&r, "n"), Some(N as f64), "{r}");
}

#[test]
fn an_oversized_enumeration_is_refused() {
    // `count` drives an allocation of count * rerank_bytes, so one request must not
    // be able to ask for the whole corpus.
    let mut f = fixture("r_vec_cap");
    let r = f.get("/vectors?count=100000");
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
    assert!(r.contains("limit"), "{r}");
}

#[test]
fn writing_vectors_over_http_is_refused_with_405() {
    // Ingest would mutate the volume under handles that have already mounted it.
    // Refused explicitly rather than 404, so the answer is "not here" rather than
    // "no such thing".
    let mut f = fixture("r_vec_write");
    for r in [
        f.post("/vectors", "application/octet-stream", &[0u8; 4]),
        f.serve(b"DELETE /vectors HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"),
    ] {
        assert!(r.starts_with("HTTP/1.1 405"), "{r}");
    }
}

#[test]
fn stats_reports_storage_read_by_a_different_worker() {
    // The Pi-4 bug this guards. Each worker holds its own backend handles, so
    // `/stats` reading the *serving* worker's counters reported 0 reads and 0 bytes
    // on a two-worker daemon that had just read 19.8 MB — the scrape landed on the
    // idle worker. A monitoring check seeing zero I/O from a busy daemon would
    // conclude the volume was fully cached.
    //
    // Reproduced without threads: the shared Counters is the contract, so a delta
    // folded by one worker must be visible to a scrape rendered by another.
    let mut worker_a = fixture("r_stats_agg_a");
    let mut worker_b = fixture("r_stats_agg_b");

    // Worker A does real work.
    let query = [0.25f32; D];
    let body: Vec<u8> = query.iter().flat_map(|f| f.to_le_bytes()).collect();
    let r = worker_a.post("/search", "application/octet-stream", &body);
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");

    let a_reads = number(&worker_a.get("/stats"), "reads").unwrap_or(0.0);
    assert!(a_reads > 0.0, "the worker that searched reported no reads");

    // Worker B, which has served nothing but a scrape, must still see A's work
    // once A's delta is in the shared total.
    let shared = worker_a.counters_snapshot();
    worker_b.adopt_counters(shared);
    let b_reads = number(&worker_b.get("/stats"), "reads").unwrap_or(-1.0);
    assert!(
        b_reads >= a_reads,
        "a scrape on an idle worker reported {b_reads} reads against {a_reads} done"
    );
}

#[test]
fn ready_does_not_scan_the_corpus() {
    // Readiness used to run a full `search()`, measured on a Pi 4 at 1,020 reads and
    // 901 KB of SD traffic per call — 10.8 MB/min at a 5 s probe interval, and
    // growing with N, which is backwards for a check meant to be polled.
    //
    // One record read is O(1) and is what readiness actually has to establish: an
    // unmountable volume cannot reach this code, so what is left to prove is that
    // storage still answers.
    let mut f = fixture("r_ready_cost");
    let before = number(&f.get("/stats"), "reads").unwrap_or(0.0);

    let r = f.get("/ready");
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(r.contains(r#""ready":true"#), "{r}");

    let after = number(&f.get("/stats"), "reads").unwrap_or(0.0);
    let cost = after - before;
    // A full scan of N vectors is tens of reads at minimum; a record probe is a
    // handful. The bound is deliberately loose — the property is O(1) vs O(N), not
    // an exact count.
    assert!(
        cost < 32.0,
        "/ready cost {cost} reads; a readiness probe must not scale with the corpus"
    );
}

#[test]
fn every_response_is_json_including_errors() {
    // One shape for a client to parse, whatever happened.
    let mut f = fixture("r_json");
    for r in [
        f.get("/health"),
        f.get("/nope"),
        f.post("/search", "application/octet-stream", &[0u8; 3]),
    ] {
        assert!(
            r.contains("Content-Type: application/json"),
            "not JSON: {r}"
        );
        let body = r.split("\r\n\r\n").nth(1).unwrap_or("");
        assert!(body.trim_start().starts_with('{'), "not an object: {body}");
        assert!(body.trim_end().ends_with('}'), "not an object: {body}");
    }
}
