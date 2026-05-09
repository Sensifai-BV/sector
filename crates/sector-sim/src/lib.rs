//! `sector-sim` — fault-injecting flash simulator and claim-validation harness.
//!
//! Turns the report's claims into tests that can fail. Each falsification
//! criterion of protocol P2 gets an executable able to return *refuted*, and
//! the corruption experiments run against real embeddings rather than the
//! synthetic corpus the current numbers rest on.
//!
//! Corruption sweeps are signed and directed rather than random. A first
//! attempt to reproduce a known defect failed because random displacements
//! rarely strike a query's own neighbours; a directed construction then
//! produced a 0.127 recall loss the bound predicted to be zero.

pub mod corrupt;
pub mod fault;
pub mod recall;
pub mod sim_flash;
pub mod sweep;
