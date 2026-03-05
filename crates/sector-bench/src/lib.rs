//! Benchmark harness for SECTOR on real datasets.
//!
//! Every measurement emits machine-readable JSON to `measurements/`, and the
//! plotting reads those files. Nothing is recomputed in the plotting code: a
//! figure that reimplements the measurement it draws can disagree with it, and
//! the disagreement is invisible.
//!
//! # What each axis can and cannot measure here
//!
//! Recall, latency, memory and disk are measured directly. **Energy is not.**
//! The measurement host has no current sensor, and an SBC figure would not
//! transfer to an MCU two orders of magnitude below it in draw. What is
//! measured instead is the energy model's inputs — cycles and bytes per phase —
//! leaving two platform constants to be filled in from a hardware measurement.

pub mod json;
pub mod pipeline;
pub mod report;
pub mod timing;

/// Configuration shared by every subcommand, echoed into every result file.
///
/// A number without its configuration is a reporting failure, so this travels
/// with the measurement rather than being remembered.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Vectors used. 0 means the whole file.
    pub n: usize,
    /// Subspaces.
    pub m: usize,
    /// Bits per code.
    pub b: usize,
    /// Candidate depth.
    pub r: usize,
    /// Results per query.
    pub k: usize,
    /// Training seed.
    pub seed: u64,
    /// Vectors used to train the codebook. 0 means all of them.
    pub train_n: usize,
}

impl Config {
    /// JSON form, embedded in every measurement file.
    pub fn to_value(&self, d: usize, n_actual: usize) -> json::Value {
        json::obj(vec![
            ("d", json::i(d as i64)),
            ("m", json::i(self.m as i64)),
            ("b", json::i(self.b as i64)),
            ("n", json::i(n_actual as i64)),
            ("r", json::i(self.r as i64)),
            ("k", json::i(self.k as i64)),
            ("seed", json::i(self.seed as i64)),
            ("train_n", json::i(self.train_n as i64)),
        ])
    }
}
