//! The daemon end to end: bind, serve every route, shut down cleanly.
//!
//! Uses a real TCP listener on an ephemeral port rather than a mocked stream, so
//! the accept loop, the worker pool, keep-alive and the shutdown path are all
//! exercised. A test against an in-memory stream would pass with a broken accept
//! loop.
//!
//! # Skipping is not passing
//!
//! Some sandboxes forbid `bind`. When that happens these tests print why and
//! return rather than reporting green on something they did not check — but the
//! *reason* is verified: only a permission error is tolerated, and any other bind
//! failure fails the test. A blanket skip on any error would hide a regression in
//! the bind path itself.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use sector_os::volume::test_support::{build_image_and_corpus, TempDir};
use sector_serve::{Config, Server};

const D: usize = 32;
const M: usize = 4;
const N: usize = 300;

/// A running daemon on an ephemeral port.
struct Daemon {
    addr: std::net::SocketAddr,
    handle: sector_serve::server::ShutdownHandle,
    thread: Option<std::thread::JoinHandle<()>>,
    _dir: TempDir,
}

/// Start a daemon on `path`, or `None` when the sandbox forbids binding.
fn spawn(path: std::path::PathBuf, dir: TempDir, workers: usize) -> Option<Daemon> {
    let config = Config {
        image: path,
        socket: None,
        // Port 0: the kernel chooses, so parallel tests cannot collide.
        listen: Some("127.0.0.1:0".to_string()),
        workers,
        k: 10,
        r: None,
    };
    let mut server: Server<sector_os::FileFlash> = match Server::new(config) {
        Ok(s) => s,
        Err(sector_serve::ServerError::Bind(what, e))
            if e.kind() == std::io::ErrorKind::PermissionDenied =>
        {
            eprintln!("skipping: this environment forbids binding {what}: {e}");
            return None;
        }
        Err(e) => panic!("daemon failed to start: {e}"),
    };
    let addr = server.local_addr().expect("bound address");
    let handle = server.shutdown_handle();
    let thread = std::thread::spawn(move || {
        server.run().expect("serve");
    });
    // The listener is bound before `new` returns, so a connection cannot be
    // refused; workers may still be spawning, which the read timeout covers.
    std::thread::sleep(Duration::from_millis(120));
    Some(Daemon {
        addr,
        handle,
        thread: Some(thread),
        _dir: dir,
    })
}

impl Daemon {
    /// A daemon serving a freshly built volume, with its corpus.
    fn start(tag: &str) -> Option<(Self, Vec<f32>)> {
        let dir = TempDir::new(tag);
        let (image, corpus) = build_image_and_corpus(D, M, N);
        let path = dir.path().join("volume.sector");
        std::fs::write(&path, &image).expect("write volume");
        spawn(path, dir, 2).map(|d| (d, corpus))
    }

    /// Send a raw request and return (status line, body).
    fn request(&self, raw: &[u8]) -> (String, String) {
        let mut stream = TcpStream::connect(self.addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("timeout");
        stream.write_all(raw).expect("write");
        stream.flush().expect("flush");

        let mut reader = BufReader::new(stream);
        let mut status = String::new();
        reader.read_line(&mut status).expect("status line");

        // Headers, then exactly Content-Length bytes: reading to EOF would hang on
        // a keep-alive connection the server has not closed.
        let mut len = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("header");
            let t = line.trim_end();
            if t.is_empty() {
                break;
            }
            if let Some((k, v)) = t.split_once(':') {
                if k.trim().eq_ignore_ascii_case("content-length") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
        }
        let mut body = vec![0u8; len];
        reader.read_exact(&mut body).expect("body");
        (
            status.trim_end().to_string(),
            String::from_utf8_lossy(&body).to_string(),
        )
    }

    fn get(&self, path: &str) -> (String, String) {
        self.request(
            format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(),
        )
    }

    fn post(&self, path: &str, content_type: &str, body: &[u8]) -> (String, String) {
        let mut raw = format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: {content_type}\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        raw.extend_from_slice(body);
        self.request(&raw)
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.handle.shutdown();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Extract a JSON number by key, without pulling in a parser.
fn number(json: &str, key: &str) -> Option<f64> {
    let at = json.find(&format!("\"{key}\":"))? + key.len() + 3;
    let rest = &json[at..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The `ids` array of the first result, as text.
fn first_ids(json: &str) -> String {
    let at = json.find("\"ids\":").expect("ids");
    let rest = &json[at..];
    rest[..rest.find(']').expect("close bracket") + 1].to_string()
}

#[test]
fn health_answers_without_touching_the_volume() {
    let Some((d, _)) = Daemon::start("health") else {
        return;
    };
    let (status, body) = d.get("/health");
    assert!(status.starts_with("HTTP/1.1 200"), "{status}");
    assert!(body.contains(r#""status":"ok""#), "{body}");
}

#[test]
fn ready_runs_a_real_query() {
    // Liveness and readiness are distinct: a mounted volume that cannot answer is
    // not ready, and only a query proves it can.
    let Some((d, _)) = Daemon::start("ready") else {
        return;
    };
    let (status, body) = d.get("/ready");
    assert!(status.starts_with("HTTP/1.1 200"), "{status}");
    assert!(body.contains(r#""ready":true"#), "{body}");
}

#[test]
fn info_reports_geometry_and_the_resident_total() {
    let Some((d, _)) = Daemon::start("info") else {
        return;
    };
    let (status, body) = d.get("/info");
    assert!(status.starts_with("HTTP/1.1 200"), "{status}");
    assert_eq!(number(&body, "n"), Some(N as f64), "{body}");
    assert_eq!(number(&body, "d"), Some(D as f64), "{body}");
    assert_eq!(number(&body, "workers"), Some(2.0), "{body}");
    // The figure a deployment sizes against, stated rather than left to be
    // multiplied out.
    let per = number(&body, "resident_bytes_per_worker").expect("per worker");
    let total = number(&body, "resident_bytes_total").expect("total");
    assert_eq!(total, per * 2.0, "{body}");
}

#[test]
fn a_binary_search_returns_ranked_ids() {
    let Some((d, corpus)) = Daemon::start("search_bin") else {
        return;
    };
    let mut body = Vec::new();
    for &x in &corpus[..D] {
        body.extend_from_slice(&x.to_le_bytes());
    }
    let (status, text) = d.post("/search?k=5", "application/octet-stream", &body);
    assert!(status.starts_with("HTTP/1.1 200"), "{status} {text}");
    assert_eq!(number(&text, "queries"), Some(1.0), "{text}");
    assert!(text.contains(r#""degraded":false"#), "{text}");
    // Every vector must be scanned, or the payload reader stopped early.
    assert_eq!(number(&text, "scanned"), Some(N as f64), "{text}");
    // Stage two ran: a CRC was checked.
    assert!(
        number(&text, "blocks_verified").unwrap_or(0.0) > 0.0,
        "stage two did not run: {text}"
    );
}

#[test]
fn a_text_search_returns_the_same_ids_as_the_binary_one() {
    // The two encodings must be interchangeable, or `curl` and a client library
    // disagree about what the same query means.
    let Some((d, corpus)) = Daemon::start("search_txt") else {
        return;
    };
    let q = &corpus[..D];

    let mut bin = Vec::new();
    for &x in q {
        bin.extend_from_slice(&x.to_le_bytes());
    }
    let (_, from_bin) = d.post("/search?k=5", "application/octet-stream", &bin);

    let text: String = q
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let (status, from_txt) = d.post("/search?k=5", "text/plain", text.as_bytes());
    assert!(status.starts_with("HTTP/1.1 200"), "{status} {from_txt}");
    assert_eq!(
        first_ids(&from_bin),
        first_ids(&from_txt),
        "the two encodings disagree"
    );
}

#[test]
fn a_batch_of_queries_returns_one_result_each() {
    let Some((d, corpus)) = Daemon::start("batch") else {
        return;
    };
    let mut body = Vec::new();
    for v in 0..4usize {
        for &x in &corpus[v * D..(v + 1) * D] {
            body.extend_from_slice(&x.to_le_bytes());
        }
    }
    let (status, text) = d.post("/search?k=3", "application/octet-stream", &body);
    assert!(status.starts_with("HTTP/1.1 200"), "{status}");
    assert_eq!(number(&text, "queries"), Some(4.0), "{text}");
}

#[test]
fn a_body_of_the_wrong_length_is_400_not_500() {
    // A client error must not look like a server error, or the client retries a
    // request that can never succeed.
    let Some((d, _)) = Daemon::start("badlen") else {
        return;
    };
    let (status, text) = d.post("/search", "application/octet-stream", &[0u8; 9]);
    assert!(status.starts_with("HTTP/1.1 400"), "{status} {text}");
    assert!(text.contains("D*4"), "{text}");
}

#[test]
fn a_malformed_k_is_rejected_rather_than_defaulted() {
    let Some((d, corpus)) = Daemon::start("badk") else {
        return;
    };
    let mut body = Vec::new();
    for &x in &corpus[..D] {
        body.extend_from_slice(&x.to_le_bytes());
    }
    let (status, _) = d.post("/search?k=ten", "application/octet-stream", &body);
    assert!(status.starts_with("HTTP/1.1 400"), "{status}");
}

#[test]
fn an_unknown_route_is_404_and_a_wrong_method_is_405() {
    // The distinction tells a client whether to fix the URL or the verb.
    let Some((d, _)) = Daemon::start("routes") else {
        return;
    };
    let (status, _) = d.get("/nope");
    assert!(status.starts_with("HTTP/1.1 404"), "{status}");
    let (status, _) = d.get("/search");
    assert!(status.starts_with("HTTP/1.1 405"), "{status}");
    let (status, _) = d.post("/health", "text/plain", b"x");
    assert!(status.starts_with("HTTP/1.1 405"), "{status}");
}

#[test]
fn an_oversized_content_length_is_413_before_a_body_is_sent() {
    // The allocation defence: the request announces a huge body and sends none.
    let Some((d, _)) = Daemon::start("toolarge") else {
        return;
    };
    let raw = format!(
        "POST /search HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        sector_serve::http::MAX_BODY_BYTES + 1
    );
    let (status, _) = d.request(raw.as_bytes());
    assert!(status.starts_with("HTTP/1.1 413"), "{status}");
}

#[test]
fn chunked_encoding_is_411() {
    let Some((d, _)) = Daemon::start("chunked") else {
        return;
    };
    let raw = "POST /search HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n";
    let (status, _) = d.request(raw.as_bytes());
    assert!(status.starts_with("HTTP/1.1 411"), "{status}");
}

#[test]
fn keep_alive_serves_several_requests_on_one_connection() {
    // The property a client library depends on, and the one that makes the read
    // timeout necessary.
    let Some((d, _)) = Daemon::start("keepalive") else {
        return;
    };
    let mut stream = TcpStream::connect(d.addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("timeout");

    for i in 0..3 {
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            .expect("write");
        stream.flush().expect("flush");
        let mut buf = vec![0u8; 1024];
        let mut total = 0;
        loop {
            let n = stream.read(&mut buf[total..]).expect("read");
            if n == 0 {
                break;
            }
            total += n;
            if String::from_utf8_lossy(&buf[..total]).contains("}") {
                break;
            }
        }
        let text = String::from_utf8_lossy(&buf[..total]);
        assert!(text.starts_with("HTTP/1.1 200"), "request {i}: {text}");
        assert!(
            text.contains("Connection: keep-alive"),
            "request {i}: {text}"
        );
    }
}

#[test]
fn stats_accumulates_across_requests() {
    let Some((d, corpus)) = Daemon::start("stats") else {
        return;
    };
    let mut body = Vec::new();
    for &x in &corpus[..D] {
        body.extend_from_slice(&x.to_le_bytes());
    }
    for _ in 0..3 {
        d.post("/search?k=5", "application/octet-stream", &body);
    }
    let (status, text) = d.get("/stats");
    assert!(status.starts_with("HTTP/1.1 200"), "{status}");
    assert!(
        number(&text, "answered").unwrap_or(0.0) >= 3.0,
        "queries were not counted: {text}"
    );
    assert_eq!(number(&text, "candidates_dropped"), Some(0.0), "{text}");
}

#[test]
fn a_corrupted_volume_is_reported_as_degraded_rather_than_failing() {
    // Corruption degrades the answer; it does not take the service down. The
    // `degraded` flag is how a client notices without walking every result.
    let dir = TempDir::new("degraded");
    let (mut image, corpus) = build_image_and_corpus(D, M, N);
    let path = dir.path().join("volume.sector");
    std::fs::write(&path, &image).expect("write");

    // Locate the rerank region from the manifest rather than assuming an offset.
    let base = {
        let mut f = sector_os::FileFlash::open(&path).expect("open");
        let v = sector_os::HostVolume::mount(&mut f, None).expect("mount");
        v.rerank_base() as usize
    };
    image[base + 3] ^= 0xFF;
    std::fs::write(&path, &image).expect("rewrite");

    let Some(d) = spawn(path, dir, 1) else {
        return;
    };
    let mut body = Vec::new();
    for &x in &corpus[..D] {
        body.extend_from_slice(&x.to_le_bytes());
    }
    let (status, text) = d.post("/search?k=10", "application/octet-stream", &body);
    assert!(
        status.starts_with("HTTP/1.1 200"),
        "corruption must degrade, not fail: {status}"
    );
    assert!(text.contains(r#""degraded":true"#), "{text}");
    assert!(
        number(&text, "dropped_total").unwrap_or(0.0) > 0.0,
        "drops were not reported: {text}"
    );
}

#[test]
fn a_daemon_with_no_listener_is_refused_at_startup() {
    // Better than starting and answering nothing.
    let dir = TempDir::new("nolistener");
    let (image, _) = build_image_and_corpus(D, M, 64);
    let path = dir.path().join("volume.sector");
    std::fs::write(&path, &image).expect("write");

    let config = Config {
        image: path,
        socket: None,
        listen: None,
        workers: 1,
        k: 10,
        r: None,
    };
    // `Server` holds live listeners and is deliberately not `Debug`, so the
    // result is matched rather than unwrapped.
    match Server::<sector_os::FileFlash>::new(config) {
        Err(sector_serve::ServerError::NoListener) => {}
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("a daemon with no listener was accepted"),
    }
}

#[test]
fn an_unmountable_volume_fails_at_startup_not_per_request() {
    // A daemon that starts on a broken volume 503s every request, which looks like
    // a runtime fault rather than the configuration error it is.
    let dir = TempDir::new("badvolume");
    let path = dir.path().join("not-a-volume.sector");
    std::fs::write(&path, [0u8; 4096]).expect("write");

    let config = Config {
        image: path,
        socket: None,
        listen: Some("127.0.0.1:0".to_string()),
        workers: 1,
        k: 10,
        r: None,
    };
    match Server::<sector_os::FileFlash>::new(config) {
        Err(sector_serve::ServerError::Volume(_)) => {}
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("an unmountable volume was accepted"),
    }
}
