//! `sector serve` — run the query daemon.
//!
//! Defaults to a Unix socket, because that is the shape a systemd service on a Pi
//! wants: filesystem permissions are the access control, and there is no port to
//! expose by accident. A TCP listener is opt-in via `--listen`, and the startup
//! banner says plainly that it carries no transport security.
//!
//! # Signals
//!
//! `SIGTERM` and `SIGINT` set the daemon's shutdown flag. Workers finish the
//! request in flight and exit, and the socket file is removed — which is what
//! makes a restart clean rather than failing on a stale path.
//!
//! Signal handling here is a plain `sigaction` with a handler that does one
//! relaxed atomic store and nothing else. That is the async-signal-safe subset:
//! no allocation, no locks, no I/O. The alternative — a signal-handling crate —
//! would break the workspace's zero-dependency property for a dozen lines of
//! code.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use sector_os::platform::{Abi, AbiStatus, Board};
use sector_serve::{Config, Server};

/// Set by the signal handler.
///
/// A `static` rather than a captured variable because a signal handler cannot
/// carry state: it is a bare `extern "C"` function.
static SIGNALLED: AtomicBool = AtomicBool::new(false);

/// `serve` arguments.
pub struct Args {
    /// Volume to serve.
    pub image: PathBuf,
    /// Unix socket path.
    pub socket: Option<PathBuf>,
    /// TCP address.
    pub listen: Option<String>,
    /// Worker threads.
    pub workers: usize,
    /// Default results per query.
    pub k: usize,
    /// Candidate depth override, 0 for the image's own.
    pub r: usize,
}

/// Default socket path.
///
/// Under `/run` because it is a tmpfs on every current Pi OS and Ubuntu: the
/// socket disappears on reboot, which is correct for a runtime object, and a
/// stale path cannot survive a power cut.
pub const DEFAULT_SOCKET: &str = "/run/sector/sector.sock";

/// Parse `serve` arguments.
pub fn parse(argv: &[String]) -> Result<Args, String> {
    let socket = argv
        .iter()
        .position(|a| a == "--socket")
        .and_then(|i| argv.get(i + 1))
        .map(PathBuf::from);
    let listen = argv
        .iter()
        .position(|a| a == "--listen")
        .and_then(|i| argv.get(i + 1))
        .cloned();

    Ok(Args {
        image: PathBuf::from(crate::flag(argv, "--image")?),
        // Only default the socket when no listener was named at all. Defaulting
        // it alongside an explicit `--listen` would open a socket the operator
        // did not ask for.
        socket: match (&socket, &listen) {
            (None, None) => Some(PathBuf::from(DEFAULT_SOCKET)),
            _ => socket,
        },
        listen,
        // One worker by default. On a Pi 1 that is the whole machine, and on a
        // larger board an operator who wants concurrency will say so — a default
        // of "number of cores" would quietly multiply the resident figure.
        workers: crate::opt_num(argv, "--workers", 1)?,
        k: crate::opt_num(argv, "--k", 10)?,
        r: crate::opt_num(argv, "--r", 0)?,
    })
}

/// Run the daemon.
pub fn run(args: Args) -> Result<(), String> {
    let config = Config {
        image: args.image.clone(),
        socket: args.socket.clone(),
        listen: args.listen.clone(),
        workers: args.workers.max(1),
        k: args.k,
        r: if args.r == 0 { None } else { Some(args.r) },
    };

    // The mapped backend is not offered here. A daemon holding a volume open for
    // its lifetime is exactly where the page cache would flatter the numbers, and
    // `sector stats --backend mmap` is the place to measure that difference
    // deliberately rather than have it become the deployment default.
    let mut server: Server<sector_os::FileFlash> =
        Server::new(config).map_err(|e| format!("{e}"))?;

    install_signal_handlers()?;
    let handle = server.shutdown_handle();

    // A watcher thread turns the signal flag into the daemon's shutdown, so the
    // handler itself stays limited to one atomic store.
    let watcher = handle.clone();
    std::thread::Builder::new()
        .name("sector-signal".into())
        .spawn(move || {
            while !watcher.is_shutting_down() {
                if SIGNALLED.load(Ordering::Relaxed) {
                    watcher.shutdown();
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        })
        .map_err(|e| format!("could not start the signal watcher: {e}"))?;

    banner(&args, &server);
    server.run().map_err(|e| format!("{e}"))?;
    println!("stopped");
    Ok(())
}

/// Print what is being served and where.
fn banner(args: &Args, server: &Server<sector_os::FileFlash>) {
    let board = Board::detect();
    println!("sector serve {}", env!("CARGO_PKG_VERSION"));
    println!("  volume     {}", args.image.display());
    println!(
        "  board      {} ({}, tier {})",
        board.model.as_deref().unwrap_or("unknown"),
        Abi::current(),
        board.arch.tier()
    );

    // The ABI check belongs in the startup path, not only in `doctor`: an
    // incompatible binary can start and then fail with SIGILL when a query
    // reaches an unsupported instruction, and a daemon that dies under load is
    // harder to diagnose than one that warns at boot.
    match board.abi_status() {
        AbiStatus::Match => {}
        AbiStatus::Suboptimal => println!(
            "  note       {} would be a better match for this board",
            board.recommended_artifact().artifact()
        ),
        AbiStatus::Incompatible => {
            println!("  WARNING    this binary's ABI does not match this board.");
            println!(
                "             install {} — this one may fail with SIGILL",
                board.recommended_artifact().artifact()
            );
            println!("             once a query reaches an unsupported instruction.");
        }
    }

    println!("  workers    {}", args.workers.max(1));
    println!(
        "  resident   {} B per worker, {} B total",
        server.resident_per_worker(),
        server.resident_total()
    );
    if let Some(s) = &args.socket {
        println!("  socket     {}", s.display());
    }
    if let Some(a) = &args.listen {
        println!("  listening  {a}");
        // Said plainly rather than left to be assumed.
        println!("             no TLS and no authentication: use a trusted network");
        println!("             or a reverse proxy that terminates TLS.");
    }
    // Every route, because the banner is where an operator looks first and a
    // partial list reads as the whole surface. `/vectors` was missing here while
    // being served, which is the kind of omission that has someone conclude the
    // endpoint does not exist.
    println!("  routes     GET  /health /ready /info /stats");
    println!("             GET  /vectors /vectors/{{id}}");
    println!("             POST /search");
}

/// Install `SIGTERM` and `SIGINT` handlers.
///
/// The handler stores one atomic and returns. Everything else — waking the
/// listeners, draining workers, removing the socket — happens on a normal thread,
/// because a signal handler may not allocate, lock, or perform I/O.
fn install_signal_handlers() -> Result<(), String> {
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;

    extern "C" fn on_signal(_sig: i32) {
        // The entire handler: one relaxed store. Async-signal-safe.
        SIGNALLED.store(true, Ordering::Relaxed);
    }

    unsafe extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
    }

    // SAFETY: `signal` is being given a real function pointer to an `extern "C"`
    // handler that performs only an atomic store. The returned previous handler is
    // discarded, which is correct: this process installs its handlers once at
    // startup and never restores them.
    unsafe {
        let h = on_signal as *const () as usize;
        signal(SIGTERM, h);
        signal(SIGINT, h);
    }
    Ok(())
}
