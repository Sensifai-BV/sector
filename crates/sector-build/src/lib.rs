//! `sector-build` — the host-side index builder.
//!
//! Everything corpus-global happens here, offline, on a machine with a heap and
//! an FPU. The output is a finished volume image the device only reads, which
//! is what lets `sector-core` stay heapless and integer-only.
//!
//! # Pipeline
//!
//! 1. train PQ codebooks;
//! 2. optimise centroid labels — lossless, worth +0.105 recall under 20%
//!    payload corruption at zero storage and query cost;
//! 3. quantize codebooks to bounded fixed point — caps single-bit displacement,
//!    cutting measured intruder count 37x;
//! 4. encode the corpus, recording per-centroid populations;
//! 5. measure depth-aware exposure to obtain criticality weights;
//! 6. solve the protection allocation over the convex hull of achievable rates;
//! 7. emit payload, rerank copy, protected codebook, block CRCs, manifest.
//!
//! Steps 2 and 3 cost nothing and outperform parity per byte spent.

pub mod allocate;
pub mod criticality;
pub mod dataset;
pub mod emit;
pub mod encode;
pub mod label_opt;
pub mod train;
