//! Listeners, the worker pool, and shutdown.
//!
//! # One connection per worker, for the connection's lifetime
//!
//! A worker accepts a connection and serves every request on it until the client
//! closes or a timeout fires. With keep-alive that means a client can hold a
//! worker, which is why the read timeout is not optional: without it one idle
//! connection removes a worker from the pool permanently, and `workers` idle
//! connections are a complete denial of service. The timeout is the bound that
//! makes a fixed pool safe.
//!
//! # Shutdown
//!
//! `SIGTERM` from systemd must not cut a query in half or leave the socket file
//! behind. A shutdown flag is set, the listeners are woken by connecting to them,
//! workers finish the request in flight and exit, and the socket path is removed
//! in `Drop`. A worker checks the flag between requests rather than mid-query: a
//! query is bounded work measured in milliseconds, and interrupting it would
//! return a partial answer.
//!
//! Signal handling itself is the caller's: this crate exposes
//! [`Server::shutdown_handle`] and `sector-cli` wires it to a signal, so nothing
//! here needs `unsafe` or a signal-handling dependency.

use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sector_os::search::{HasAccessStats, OpenBackend, Searcher};

use crate::api::{self, Counters};
use crate::http::{self, ReadError, Request, Status};

/// How long a connection may stay idle before it is closed.
///
/// The bound that makes a fixed worker pool safe: without it an idle keep-alive
/// connection holds a worker forever. 30 s is generous for a local client and
/// short enough that a stalled connection recovers without operator action.
pub const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a write may block before the connection is abandoned.
///
/// A client that stops reading mid-response would otherwise hold a worker in
/// `write`.
pub const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

/// Daemon configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// Volume to serve.
    pub image: PathBuf,
    /// Unix socket path, if any.
    pub socket: Option<PathBuf>,
    /// TCP address, if any.
    pub listen: Option<String>,
    /// Worker threads.
    pub workers: usize,
    /// Default results per query.
    pub k: usize,
    /// Candidate depth override.
    pub r: Option<usize>,
}

/// Why the daemon could not start.
#[derive(Debug)]
pub enum ServerError {
    /// No listener was configured.
    NoListener,
    /// The volume could not be opened or mounted.
    Volume(String),
    /// A listener could not be bound.
    Bind(String, std::io::Error),
    /// A socket path already exists and is in use.
    SocketInUse(PathBuf),
    /// A worker thread could not be started.
    Spawn(std::io::Error),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoListener => write!(f, "no listener configured: pass --socket or --listen"),
            Self::Volume(e) => write!(f, "{e}"),
            Self::Bind(what, e) => write!(f, "could not bind {what}: {e}"),
            Self::SocketInUse(p) => {
                write!(f, "{} exists and a daemon is listening on it", p.display())
            }
            Self::Spawn(e) => write!(f, "could not start a worker: {e}"),
        }
    }
}

impl std::error::Error for ServerError {}

/// Set to stop the daemon.
#[derive(Clone, Debug)]
pub struct ShutdownHandle {
    flag: Arc<AtomicBool>,
    socket: Option<PathBuf>,
    listen: Option<String>,
}

impl ShutdownHandle {
    /// Ask the daemon to stop after finishing in-flight requests.
    ///
    /// Wakes each blocked `accept` by connecting to it. Without that the listener
    /// would not notice the flag until the next client arrived, which on an idle
    /// daemon is never — systemd would then escalate to `SIGKILL` and the socket
    /// file would be left behind.
    pub fn shutdown(&self) {
        self.flag.store(true, Ordering::SeqCst);
        if let Some(p) = &self.socket {
            let _ = UnixStream::connect(p);
        }
        if let Some(a) = &self.listen {
            let _ = TcpStream::connect(a);
        }
    }

    /// Whether shutdown has been requested.
    pub fn is_shutting_down(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// A running daemon.
pub struct Server<F: OpenBackend + HasAccessStats + Send + 'static> {
    config: Config,
    shutdown: Arc<AtomicBool>,
    counters: Arc<Mutex<Counters>>,
    started: Instant,
    /// Kept so the socket file is removed when the daemon stops.
    /// Held for its `Drop`, never read: dropping it removes the socket file.
    /// A stale socket file makes the next start fail with "address in use" on a
    /// daemon that is not running.
    #[allow(dead_code, reason = "held for Drop, which removes the socket file")]
    socket_guard: Option<SocketGuard>,
    /// Resident bytes one worker holds, measured at startup.
    resident_per_worker: usize,
    /// Bound at construction, so a bind failure precedes the startup banner.
    unix: Option<UnixListener>,
    tcp: Option<TcpListener>,
    _backend: std::marker::PhantomData<F>,
}

/// Removes the socket file on drop.
///
/// A stale socket file makes the next start fail with "address in use" on a
/// daemon that is not running, which is a confusing failure to debug at 3 a.m.
struct SocketGuard(PathBuf);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl<F: OpenBackend + HasAccessStats + Send + 'static> Server<F> {
    /// Mount the volume, bind the listeners, and prepare to serve.
    ///
    /// Everything that can fail happens here, before the caller announces the
    /// daemon: an unmountable image, a socket already in use, a port that cannot
    /// be bound. A daemon that prints "listening on :8080" and then exits on
    /// `EADDRINUSE` has told the operator something false, and in a systemd unit
    /// that line is what lands in the journal.
    pub fn new(config: Config) -> Result<Self, ServerError> {
        if config.socket.is_none() && config.listen.is_none() {
            return Err(ServerError::NoListener);
        }

        let probe: Searcher<F> = Searcher::open(&config.image, config.r)
            .map_err(|e| ServerError::Volume(e.to_string()))?;
        let resident_per_worker = probe.resident_bytes();
        drop(probe);

        let mut socket_guard = None;
        let unix = match &config.socket {
            None => None,
            Some(path) => {
                // A leftover file from a crash must be replaced, but a live
                // daemon's socket must not be: connecting first distinguishes the
                // two, where an unconditional remove would silently steal the
                // socket from a running instance.
                if path.exists() {
                    if UnixStream::connect(path).is_ok() {
                        return Err(ServerError::SocketInUse(path.clone()));
                    }
                    let _ = std::fs::remove_file(path);
                }
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let l = UnixListener::bind(path)
                    .map_err(|e| ServerError::Bind(path.display().to_string(), e))?;
                socket_guard = Some(SocketGuard(path.clone()));
                Some(l)
            }
        };

        let tcp = match &config.listen {
            None => None,
            Some(addr) => {
                Some(TcpListener::bind(addr).map_err(|e| ServerError::Bind(addr.clone(), e))?)
            }
        };

        Ok(Self {
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            counters: Arc::new(Mutex::new(Counters::default())),
            started: Instant::now(),
            socket_guard,
            resident_per_worker,
            unix,
            tcp,
            _backend: std::marker::PhantomData,
        })
    }

    /// The TCP address actually bound.
    ///
    /// Not the configured string: binding port 0 asks the kernel to choose, so
    /// the configured value and the real one differ. A caller announcing the
    /// address must report this one.
    pub fn local_addr(&self) -> Option<std::net::SocketAddr> {
        self.tcp.as_ref().and_then(|l| l.local_addr().ok())
    }

    /// Resident bytes one worker holds.
    pub const fn resident_per_worker(&self) -> usize {
        self.resident_per_worker
    }

    /// Total resident bytes at this worker count.
    pub const fn resident_total(&self) -> usize {
        self.resident_per_worker * self.config.workers
    }

    /// A handle that stops the daemon.
    pub fn shutdown_handle(&self) -> ShutdownHandle {
        ShutdownHandle {
            flag: Arc::clone(&self.shutdown),
            socket: self.config.socket.clone(),
            listen: self.config.listen.clone(),
        }
    }

    /// Serve until shutdown.
    pub fn run(&mut self) -> Result<(), ServerError> {
        let unix = self.unix.as_ref();
        let tcp = self.tcp.as_ref();
        let workers = self.config.workers.max(1);
        let mut handles = Vec::new();

        // Each worker owns a Searcher and both listeners. Sharing one accept loop
        // and a queue would add a channel and a bottleneck for no gain: the
        // kernel already serialises accept across threads.
        for id in 0..workers {
            let image = self.config.image.clone();
            let depth = self.config.r;
            let k = self.config.k;
            let shutdown = Arc::clone(&self.shutdown);
            let counters = Arc::clone(&self.counters);
            let started = self.started;
            let total_workers = workers;
            // Each worker gets its own handle on the same listener. The kernel
            // serialises accept across them, so no queue or channel is needed.
            let unix = unix
                .map(|l| l.try_clone())
                .transpose()
                .map_err(ServerError::Spawn)?;
            let tcp = tcp
                .map(|l| l.try_clone())
                .transpose()
                .map_err(ServerError::Spawn)?;

            let handle = std::thread::Builder::new()
                .name(format!("sector-worker-{id}"))
                .spawn(move || {
                    let mut searcher: Searcher<F> = match Searcher::open(&image, depth) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("worker {id}: {e}");
                            return;
                        }
                    };
                    let ctx = WorkerCtx {
                        k,
                        image: image.display().to_string(),
                        workers: total_workers,
                        started,
                    };
                    serve_loop(&mut searcher, &ctx, &shutdown, &counters, unix, tcp);
                })
                .map_err(ServerError::Spawn)?;
            handles.push(handle);
        }

        for h in handles {
            let _ = h.join();
        }
        Ok(())
    }
}

/// Per-worker context that does not change.
struct WorkerCtx {
    k: usize,
    image: String,
    workers: usize,
    started: Instant,
}

/// Accept and serve until shutdown.
fn serve_loop<F>(
    searcher: &mut Searcher<F>,
    ctx: &WorkerCtx,
    shutdown: &AtomicBool,
    counters: &Mutex<Counters>,
    unix: Option<UnixListener>,
    tcp: Option<TcpListener>,
) where
    F: OpenBackend + HasAccessStats,
{
    // Non-blocking accept with a short sleep, so a worker notices shutdown even
    // when no client ever connects. A blocking accept would need the wake-up
    // connection to arrive per worker rather than once.
    if let Some(l) = &unix {
        let _ = l.set_nonblocking(true);
    }
    if let Some(l) = &tcp {
        let _ = l.set_nonblocking(true);
    }

    while !shutdown.load(Ordering::SeqCst) {
        let mut served = false;

        if let Some(l) = &unix {
            match l.accept() {
                Ok((stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
                    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
                    handle_connection(stream, searcher, ctx, shutdown, counters);
                    served = true;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => {}
            }
        }

        if let Some(l) = &tcp {
            match l.accept() {
                Ok((stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
                    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
                    let _ = stream.set_nodelay(true);
                    handle_connection(stream, searcher, ctx, shutdown, counters);
                    served = true;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => {}
            }
        }

        if !served {
            // 2 ms: below a human's perception of latency on connect, and low
            // enough that an idle worker costs roughly nothing on a Pi 1.
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

/// A stream this server can read and write independently.
///
/// Both `UnixStream` and `TcpStream` provide `try_clone`, which is what lets one
/// half be wrapped in a `BufReader` while the other writes. The trait exists so
/// the connection handler is written once for both transports rather than twice —
/// the Unix socket and the TCP listener serve the identical request handler, so a
/// route cannot behave differently depending on how a client reached it.
///
/// It is public so a test can drive the request handler over an in-memory stream.
/// That matters more than it looks: a sandbox that forbids `bind` would otherwise
/// leave routing, parsing, keep-alive and every limit untested, and a test that
/// skips reports green without checking anything.
pub trait Duplex: std::io::Read + std::io::Write + Sized {
    /// A second handle on the same connection.
    fn dup(&self) -> std::io::Result<Self>;
}

impl Duplex for UnixStream {
    fn dup(&self) -> std::io::Result<Self> {
        self.try_clone()
    }
}

impl Duplex for TcpStream {
    fn dup(&self) -> std::io::Result<Self> {
        self.try_clone()
    }
}

/// Serve every request on one connection until it closes or times out.
///
/// Public so a test can drive it over an in-memory stream rather than a socket.
/// The transport is the only thing a socket adds; routing, parsing, keep-alive and
/// every limit are exercised identically here.
pub fn serve_connection<S, F>(
    stream: S,
    searcher: &mut Searcher<F>,
    k: usize,
    image: &str,
    workers: usize,
) where
    S: Duplex,
    F: OpenBackend + HasAccessStats,
{
    let ctx = WorkerCtx {
        k,
        image: image.to_string(),
        workers,
        started: Instant::now(),
    };
    let shutdown = AtomicBool::new(false);
    let counters = Mutex::new(Counters::default());
    handle_connection(stream, searcher, &ctx, &shutdown, &counters);
}

/// As [`serve_connection`], against a caller-supplied counter set.
///
/// The real daemon shares one `Counters` across every worker, and that sharing is
/// load-bearing: a worker's own backend handles only count the requests it served,
/// so the storage totals `/stats` reports exist only in the shared set. Fresh
/// counters per connection — which [`serve_connection`] creates — make that
/// property untestable, and it is precisely the property that broke: on a
/// two-worker Pi 4 daemon, `/stats` reported 0 bytes after 19.8 MB of reads,
/// because the scrape landed on the idle worker.
///
/// This entry point exists so a test can drive two independent workers against one
/// counter set and assert the aggregation, with no threads and no socket.
pub fn serve_connection_with<S, F>(
    stream: S,
    searcher: &mut Searcher<F>,
    k: usize,
    image: &str,
    workers: usize,
    counters: &Mutex<Counters>,
) where
    S: Duplex,
    F: OpenBackend + HasAccessStats,
{
    let ctx = WorkerCtx {
        k,
        image: image.to_string(),
        workers,
        started: Instant::now(),
    };
    let shutdown = AtomicBool::new(false);
    handle_connection(stream, searcher, &ctx, &shutdown, counters);
}

/// Serve every request on one connection until it closes or times out.
fn handle_connection<S, F>(
    stream: S,
    searcher: &mut Searcher<F>,
    ctx: &WorkerCtx,
    shutdown: &AtomicBool,
    counters: &Mutex<Counters>,
) where
    S: Duplex,
    F: OpenBackend + HasAccessStats,
{
    let Ok(write_half) = stream.dup() else {
        return;
    };
    let mut reader = BufReader::new(stream);
    let mut writer = std::io::BufWriter::new(write_half);

    loop {
        let request = match http::read_request(&mut reader) {
            Ok(r) => r,
            // A closed connection is the normal end of keep-alive, not an error.
            Err(ReadError::Closed) => return,
            Err(ReadError::Rejected(status)) => {
                if let Ok(mut c) = counters.lock() {
                    c.rejected += 1;
                }
                let _ = http::write_error(&mut writer, status, "request rejected");
                return;
            }
            Err(ReadError::Io(_)) => return,
        };

        let keep_alive = request.keep_alive && !shutdown.load(Ordering::SeqCst);
        let (status, content_type, body) = route(&request, searcher, ctx, counters);

        // Fold this worker's storage deltas into the shared total, then reset its
        // handles. Each worker's own counters describe only the requests it served,
        // so `/stats` reading the serving worker's handles reported 0 bytes on a
        // two-worker daemon that had just read 19.8 MB — the scrape landed on the
        // idle worker. Measured on a Pi 4; invisible with one worker.
        //
        // Reset after folding so the delta is added exactly once. The reset is safe
        // because nothing else reads a worker's own handles for reporting.
        let delta = searcher.backend_stats();
        searcher.reset_backend_stats();
        if let Ok(mut c) = counters.lock() {
            c.requests += 1;
            c.storage.add(&delta);
        }
        if http::write_response(
            &mut writer,
            status,
            content_type,
            body.as_bytes(),
            keep_alive,
        )
        .is_err()
        {
            return;
        }
        if !keep_alive {
            return;
        }
    }
}

/// Dispatch one request.
///
/// Returns the status, content type and body. Every route answers JSON, including
/// errors, so a client has one shape to parse.
fn route<F>(
    request: &Request,
    searcher: &mut Searcher<F>,
    ctx: &WorkerCtx,
    counters: &Mutex<Counters>,
) -> (Status, &'static str, String)
where
    F: OpenBackend + HasAccessStats,
{
    const JSON: &str = "application/json";

    match (request.method.as_str(), request.path.as_str()) {
        // Liveness: answers without touching the volume, so it stays up even if
        // storage is wedged. Conflating it with readiness would make systemd
        // restart a daemon whose volume is merely slow.
        ("GET", "/health") => (Status::Ok, JSON, health_body()),

        // Readiness: touch storage and prove a record reads back.
        //
        // # Why this is not a full query
        //
        // It was. A readiness probe ran `search()`, which scans every vector —
        // measured on a Pi 4 at 1,020 reads and 901 KB of SD traffic per call. At a
        // 5-second probe interval that is 10.8 MB/min of reads competing with real
        // queries for the same flash translation layer, and it scales with `N`: the
        // probe gets more expensive as the corpus grows, which is backwards for a
        // check meant to be cheap enough to poll.
        //
        // What readiness actually has to establish is narrower. An unmountable
        // volume cannot get here — the daemon refuses to start on one — so what is
        // left to prove is that storage is still reachable and the volume still
        // reads. One record read does that in O(1) instead of O(N).
        //
        // For a full-pipeline check, run `sector selftest` or issue a real
        // `/search`; both are stronger, and neither is something to poll.
        ("GET", "/ready") => {
            // The lowest id the volume actually holds. Not simply 0: a volume whose
            // built corpus is empty but which has been appended to holds nothing
            // below `appended_from`, and probing an absent id would report a
            // healthy daemon as unready.
            let m = searcher.manifest();
            let probe_id = if m.built_n > 0 { 0 } else { m.appended_from };
            match searcher.records(&[probe_id]) {
                Ok(rows) if rows.first().map(Option::is_some) == Some(true) => {
                    (Status::Ok, JSON, ready_body(true, ""))
                }
                // An empty volume is mounted and serving; it is ready, and saying
                // otherwise would keep a correctly-deployed daemon out of rotation.
                Ok(_) if searcher.manifest().stored() == 0 => {
                    (Status::Ok, JSON, ready_body(true, "volume holds no vectors"))
                }
                Ok(_) => (
                    Status::Unavailable,
                    JSON,
                    ready_body(false, "the first stored record did not read back"),
                ),
                Err(e) => (Status::Unavailable, JSON, ready_body(false, &e.to_string())),
            }
        }

        ("GET", "/info") => (
            Status::Ok,
            JSON,
            api::info_response(searcher, ctx.workers, &ctx.image),
        ),

        ("GET", "/stats") => {
            let uptime = ctx.started.elapsed().as_secs();
            // Fold this worker's outstanding delta before rendering, or a scrape
            // reports the total as of the *previous* request on this worker.
            let delta = searcher.backend_stats();
            searcher.reset_backend_stats();
            let body = match counters.lock() {
                Ok(mut c) => {
                    c.storage.add(&delta);
                    api::stats_response(searcher, &c, uptime)
                }
                // A poisoned lock means another worker panicked. Report what this
                // worker can see rather than nothing, and say the total is partial
                // rather than presenting it as the daemon's.
                Err(_) => {
                    let mut partial = Counters::default();
                    partial.storage.add(&delta);
                    api::stats_response(searcher, &partial, uptime)
                }
            };
            (Status::Ok, JSON, body)
        }

        ("POST", "/search") => {
            let d = searcher.geometry().d;
            let k = match request.num_param("k", ctx.k) {
                Ok(k) => k,
                Err(status) => {
                    return (status, JSON, error_body(status, "k must be a number"));
                }
            };
            let batch = match api::decode_batch(request, d) {
                Ok(b) => b,
                Err((status, detail)) => {
                    if let Ok(mut c) = counters.lock() {
                        c.rejected += 1;
                    }
                    return (status, JSON, error_body(status, &detail));
                }
            };

            let start = Instant::now();
            let mut answers = Vec::with_capacity(batch.queries.len());
            for q in &batch.queries {
                match searcher.search(q, k) {
                    Ok(a) => answers.push(a),
                    Err(e) => {
                        let (status, detail) = api::search_error(&e);
                        return (status, JSON, error_body(status, &detail));
                    }
                }
            }
            let elapsed_us = start.elapsed().as_micros() as f64;

            if let Ok(mut c) = counters.lock() {
                c.queries += answers.len() as u64;
                c.query_us += elapsed_us;
                c.dropped += answers
                    .iter()
                    .map(|a| a.stats.rerank.dropped as u64)
                    .sum::<u64>();
            }
            (Status::Ok, JSON, api::search_response(&answers, elapsed_us))
        }

        // Enumerate stored ids, or read one back.
        ("GET", "/vectors") => {
            let from = match request.num_param("from", 0) {
                Ok(v) => v as u32,
                Err(status) => {
                    return (status, JSON, error_body(status, "from must be a number"))
                }
            };
            let count = match request.num_param("count", 10) {
                Ok(v) => v,
                Err(status) => {
                    return (status, JSON, error_body(status, "count must be a number"))
                }
            };
            // Capped so one request cannot ask the daemon to read the whole corpus
            // into a response body: `count` drives an allocation of
            // `count * rerank_bytes`, which at D=128 is 128 B per id.
            const MAX_ENUMERATE: usize = 1024;
            if count > MAX_ENUMERATE {
                return (
                    Status::BadRequest,
                    JSON,
                    error_body(
                        Status::BadRequest,
                        &format!("count above the {MAX_ENUMERATE} limit"),
                    ),
                );
            }
            let ids: Vec<u32> = (from..from.saturating_add(count as u32)).collect();
            match api::vectors_response(searcher, &ids) {
                Ok(body) => (Status::Ok, JSON, body),
                Err((status, detail)) => (status, JSON, error_body(status, &detail)),
            }
        }

        // A known path with the wrong method is 405, not 404: the distinction
        // tells a client whether to fix the URL or the verb.
        ("GET", "/search")
        | ("POST", "/health" | "/ready" | "/info" | "/stats")
        // No POST /vectors: ingest would mutate the volume under handles that have
        // already mounted it. See the `api` module for why that is deliberate.
        | ("POST" | "PUT" | "DELETE", "/vectors") => (
            Status::MethodNotAllowed,
            JSON,
            error_body(Status::MethodNotAllowed, "wrong method for this route"),
        ),

        ("GET", path) if path.starts_with("/vectors/") => {
            let tail = &path["/vectors/".len()..];
            match tail.parse::<u32>() {
                Ok(id) => match api::vectors_response(searcher, &[id]) {
                    Ok(body) => (Status::Ok, JSON, body),
                    Err((status, detail)) => (status, JSON, error_body(status, &detail)),
                },
                Err(_) => (
                    Status::BadRequest,
                    JSON,
                    error_body(Status::BadRequest, "id must be a number"),
                ),
            }
        }

        _ => (
            Status::NotFound,
            JSON,
            error_body(Status::NotFound, "no such route"),
        ),
    }
}

/// `/health` body.
fn health_body() -> String {
    let mut j = sector_os::json::Json::new();
    j.object(|o| {
        o.str("status", "ok");
        o.str("version", env!("CARGO_PKG_VERSION"));
    });
    j.finish()
}

/// `/ready` body.
fn ready_body(ready: bool, detail: &str) -> String {
    let mut j = sector_os::json::Json::new();
    j.object(|o| {
        o.bool("ready", ready);
        if !detail.is_empty() {
            o.str("detail", detail);
        }
    });
    j.finish()
}

/// A JSON error body.
fn error_body(status: Status, detail: &str) -> String {
    let mut j = sector_os::json::Json::new();
    j.object(|o| {
        o.uint("status", status.code() as u64);
        o.str("error", status.reason());
        o.str("detail", detail);
    });
    j.finish()
}
