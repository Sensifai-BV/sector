//! `sector-serve` — the query daemon.
//!
//! Serves one mounted volume over a Unix socket and, optionally, TCP. Blocking
//! `std::net` with a fixed thread pool, no async runtime, no external
//! dependencies.
//!
//! # Why blocking threads rather than async
//!
//! The workload is read-dominated with a small fixed concurrency: each worker
//! holds a [`sector_os::Searcher`] whose buffers are allocated once, and a query
//! is CPU-bound integer arithmetic over a bounded number of blocking reads. An
//! async runtime's advantage is holding many idle connections cheaply, which this
//! service does not need — a Pi serving a handful of clients is not
//! connection-bound, it is arithmetic-bound.
//!
//! What the choice buys is the property the project cares about: `Cargo.lock`
//! still contains nothing but the `sector-*` crates, so the audit surface of the
//! thing facing the network is this crate plus the standard library.
//!
//! # Concurrency and memory
//!
//! One [`sector_os::Searcher`] per worker, created at startup, never reallocated.
//! Resident cost is `workers × Searcher::resident_bytes()` and is reported at
//! startup rather than estimated — a deployment on a 512 MB Zero 2 W needs that
//! figure to be a fact.
//!
//! Workers are a fixed pool, so a request arriving when every worker is busy
//! waits in the accept queue. That is a deliberate bound: an unbounded pool on a
//! Pi 1 would thrash rather than degrade.
//!
//! # Transport security
//!
//! There is none, and the default is a Unix socket for that reason — filesystem
//! permissions are the access control. A TCP listener is for a trusted network or
//! behind a reverse proxy that terminates TLS. Nothing here authenticates a
//! client, and this note is the warning rather than an implied guarantee.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod api;
pub mod http;
pub mod server;

pub use server::{Config, Server, ServerError};
