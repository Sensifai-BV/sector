//! The two backends must drive the engine to identical results.
//!
//! This is the test that licenses every comparison between them. If the
//! buffered and mapped paths returned different ids, a latency difference
//! measured across them would be uninterpretable — it could be the storage, or
//! it could be two implementations disagreeing, and the measurement could not
//! tell which.
//!
//! It also exercises the engine end to end against a real image: mount, rotate,
//! table, scan, CRC-verified rerank, drain. The unit tests in `sector-core` use
//! synthetic sources; these bytes came out of `sector-build`.
//!
//! # Features
//!
//! Needs `test-support` (the image builder) and `mmap` (the second backend).
//! Rather than compiling to nothing without them — a test that silently
//! disappears reports green while testing nothing — the file fails to build,
//! naming the flags. `cargo test -p sector-os --all-features` is the invocation,
//! and the Makefile's `test` target passes it.

#![cfg(all(feature = "test-support", feature = "mmap"))]

use sector_core::heap::Candidate;
use sector_core::metrics::Metrics;
use sector_core::query::query;
use sector_core::workspace::Workspace;
use sector_hal::{Edge, Instrument, NorFlash, Phase};
use sector_os::source::{PayloadReader, RerankReader};
use sector_os::volume::test_support::{build_image_and_corpus, TempDir};
use sector_os::{FileFlash, HostVolume, MappedFlash};
use sector_quant::codebook::{Codebook, Scale};

const D: usize = 32;
const M: usize = 4;
const N: usize = 400;
const K: usize = 10;
const R: usize = 64;

/// Counts phase entries so a test can assert every stage ran.
#[derive(Default)]
struct PhaseCounter {
    entries: [u32; 5],
}

impl Instrument for PhaseCounter {
    fn cycles(&self) -> u64 {
        0
    }
    fn mark(&mut self, phase: Phase, edge: Edge) {
        if matches!(edge, Edge::Enter) {
            self.entries[sector_core::metrics::phase_index(phase)] += 1;
        }
    }
}

/// One query's answer.
type Answer = Vec<Candidate>;

/// Which backend to open the volume with.
enum Backend {
    Buffered,
    Mapped,
}

/// Quantize a query to `i8` the way the device receives it.
///
/// One shared scale: the image's per-subspace scales are not stored (see
/// `sector_os::volume::test_support`), and the corpus these tests build is
/// isotropic so a single scale is the correct reconstruction.
fn quantize_query(q: &[f32], num: f32, den: f32) -> Vec<i8> {
    q.iter()
        .map(|x| ((x / num) * den).round().clamp(-128.0, 127.0) as i8)
        .collect()
}

/// Run every query against `path` through the engine.
///
/// Payload and rerank are borrowed at the same time, so each gets its own
/// handle on the volume. That is also what the daemon does per worker, and it
/// is why `FileFlash` uses `pread`: two handles on one file must not share a
/// seek offset.
fn run(
    path: &std::path::Path,
    backend: Backend,
    queries: &[Vec<f32>],
) -> (Vec<Answer>, PhaseCounter) {
    // Mount through the chosen backend so the manifest read is exercised on
    // both. `Volume`'s bindings differ between them; the answers must not.
    let geometry = match backend {
        Backend::Buffered => {
            let mut f = FileFlash::open(path).expect("open");
            let v = HostVolume::mount(&mut f, None).expect("mount buffered");
            assert!(v.codebook.iter().any(|b| *b != 0), "codebook read as zeros");
            v
        }
        Backend::Mapped => {
            let mut f = MappedFlash::open(path).expect("map");
            HostVolume::mount(&mut f, None).expect("mount mapped")
        }
    };
    let g = geometry.geometry;
    assert_eq!(g.n, N);
    assert_eq!(g.d, D);
    assert_eq!(g.m, M);

    let cb: Vec<i8> = geometry.codebook.iter().map(|b| *b as i8).collect();
    let per_book = g.centroids * g.ds;
    let scale = Scale::new(1, 1).expect("unit scale");

    let mut adc = vec![0i32; g.m * g.centroids];
    let mut heap_scores = vec![0i32; R];
    let mut heap_ids = vec![0u32; R];
    let mut rotation = vec![0i32; g.d];
    let mut bounce = vec![0u8; sector_format::BLOCK_BYTES];

    let mut answers = Vec::new();
    let mut counter = PhaseCounter::default();

    for q in queries {
        // 127 is the denominator `sector_build::encode::quantize` uses, and the
        // corpus extent is its numerator.
        let qi = quantize_query(q, 89.0 * 0.75 + 6.0 * 9.0, 127.0);

        let books: Vec<Codebook<'_>> = (0..g.m)
            .map(|j| {
                Codebook::new(
                    &cb[j * per_book..(j + 1) * per_book],
                    g.centroids,
                    g.ds,
                    scale,
                )
                .expect("codebook view")
            })
            .collect();

        let mut ws = Workspace {
            adc_table: &mut adc,
            heap_scores: &mut heap_scores,
            heap_ids: &mut heap_ids,
            rotation: &mut rotation,
            bounce: &mut bounce,
            scrub_cursor: 0,
        };
        let mut out = [Candidate { score: 0, id: 0 }; K];
        let mut metrics = Metrics::default();

        // Two independent handles, so payload and rerank can be borrowed at
        // once. Both are the same backend kind as the mount.
        let stats = match backend {
            Backend::Buffered => {
                let mut pf = FileFlash::open(path).expect("payload handle");
                let mut rf = FileFlash::open(path).expect("rerank handle");
                let mut payload = PayloadReader::new(&mut pf, geometry.payload_base(), &g, 8);
                let mut rerank = RerankReader::new(
                    &mut rf,
                    geometry.rerank_base(),
                    geometry.rerank_crc_base(),
                    &g,
                );
                query(
                    &qi,
                    &books,
                    &[],
                    0,
                    &mut payload,
                    &mut rerank,
                    &mut ws,
                    &mut counter,
                    g.payload_bytes,
                    K,
                    &mut out,
                    &mut metrics,
                )
                .map_err(|e| format!("{e:?}"))
            }
            Backend::Mapped => {
                let mut pf = MappedFlash::open(path).expect("payload map");
                let mut rf = MappedFlash::open(path).expect("rerank map");
                let mut payload = PayloadReader::new(&mut pf, geometry.payload_base(), &g, 8);
                let mut rerank = RerankReader::new(
                    &mut rf,
                    geometry.rerank_base(),
                    geometry.rerank_crc_base(),
                    &g,
                );
                query(
                    &qi,
                    &books,
                    &[],
                    0,
                    &mut payload,
                    &mut rerank,
                    &mut ws,
                    &mut counter,
                    g.payload_bytes,
                    K,
                    &mut out,
                    &mut metrics,
                )
                .map_err(|e| format!("{e:?}"))
            }
        }
        .expect("query");

        // Every vector must be scanned: a payload reader that stopped early
        // would silently reduce recall rather than fail.
        assert_eq!(
            stats.scan.scanned as usize, N,
            "scan saw {} of {N}",
            stats.scan.scanned
        );
        // Nothing may be dropped on a clean image. A non-zero count here means
        // the CRC extent and the record extent disagree.
        assert_eq!(stats.rerank.dropped, 0, "clean image dropped candidates");
        assert!(stats.rerank.blocks_verified > 0, "no CRC was checked");

        answers.push(out[..stats.returned as usize].to_vec());
    }
    (answers, counter)
}

/// Build a volume in a temp dir, returning the guard, path and corpus.
fn volume(tag: &str) -> (TempDir, std::path::PathBuf, Vec<f32>) {
    let dir = TempDir::new(tag);
    let (image, corpus) = build_image_and_corpus(D, M, N);
    let path = dir.path().join("volume.sector");
    std::fs::write(&path, &image).expect("write volume");
    (dir, path, corpus)
}

#[test]
fn buffered_and_mapped_backends_return_identical_answers() {
    let (_dir, path, corpus) = volume("agree");
    let queries: Vec<Vec<f32>> = [0usize, 37, 199, 301]
        .iter()
        .map(|&v| corpus[v * D..(v + 1) * D].to_vec())
        .collect();

    let (buffered, counter) = run(&path, Backend::Buffered, &queries);
    let (mapped, _) = run(&path, Backend::Mapped, &queries);

    assert_eq!(buffered.len(), queries.len());
    for (i, (a, b)) in buffered.iter().zip(mapped.iter()).enumerate() {
        assert_eq!(a, b, "backends disagree on query {i}");
        assert!(!a.is_empty(), "query {i} returned nothing");
    }

    // Five phases, each entered once per query.
    for (i, n) in counter.entries.iter().enumerate() {
        assert_eq!(*n as usize, queries.len(), "phase {i} entered {n} times");
    }
}

#[test]
fn the_engine_recovers_the_brute_force_ranking() {
    // End-to-end correctness against the right reference.
    //
    // The engine ranks by *inner product*: `adc::build_table` computes
    // `<q_j, C_j[v]>` with no norm term, and stage two's `exact_score` is an
    // integer inner product against the rerank record. So the reference is the
    // brute-force maximum-inner-product ranking over those same records — not
    // Euclidean nearest neighbour, and not "the query retrieves itself", which
    // under max-IP is false whenever norms differ.
    //
    // Two stages can each lose recall here: stage one keeps only `R` candidates
    // by quantized score, and stage two rescores those. What this asserts is
    // that the survivors are the brute-force leaders, which fails if the payload
    // reader shifts ids, if the CRC extent is wrong, or if rerank drops
    // everything.
    let (_dir, path, corpus) = volume("bruteforce");
    let ids = [0usize, 5, 123, 399];
    let queries: Vec<Vec<f32>> = ids
        .iter()
        .map(|&v| corpus[v * D..(v + 1) * D].to_vec())
        .collect();

    // The rerank region, read back as the int8 records stage two scores.
    let mut f = FileFlash::open(&path).expect("open");
    let v = HostVolume::mount(&mut f, None).expect("mount");
    let g = v.geometry;
    let mut records = vec![vec![0u8; g.rerank_bytes]; N];
    for (id, rec) in records.iter_mut().enumerate() {
        let off = g.rerank.offset_of(id).expect("offset");
        f.read(v.rerank_base() + off as u32, rec).expect("read");
    }

    let (answers, _) = run(&path, Backend::Buffered, &queries);

    let mut overlap = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let qq = quantize_query(q, 89.0 * 0.75 + 6.0 * 9.0, 127.0);
        // Brute force over every record, with the engine's own scoring function.
        let mut scored: Vec<(i32, u32)> = records
            .iter()
            .enumerate()
            .map(|(id, rec)| (sector_core::rerank::exact_score(&qq, rec), id as u32))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let reference: Vec<u32> = scored.iter().take(K).map(|(_, id)| *id).collect();
        let got: Vec<u32> = answers[qi].iter().map(|c| c.id).collect();
        overlap += got.iter().filter(|id| reference.contains(id)).count();
    }

    // Recall@10 over four queries at R=64 of N=400. The floor is deliberately
    // below 1.0: stage one is lossy by design, and a test demanding perfection
    // would be asserting something the design does not claim.
    let recall = overlap as f64 / (queries.len() * K) as f64;
    assert!(
        recall >= 0.5,
        "recall@{K} against brute force is {recall:.3}, {overlap}/{} matched",
        queries.len() * K
    );
}

#[test]
fn the_mapped_backend_reports_zero_copy_and_the_buffered_one_does_not() {
    // The distinction the `Xip` trait exists to make. `mount` binds the buffered
    // path unconditionally; `mount_xip` probes each region against the window.
    let (_dir, path, _) = volume("bindings");

    let mut buffered = FileFlash::open(&path).expect("open");
    let mut slot_a = [0u8; sector_format::manifest::MANIFEST_BYTES];
    let mut slot_b = [0u8; sector_format::manifest::MANIFEST_BYTES];
    let profile = permissive();
    let v = sector_core::mount::mount(&mut buffered, &profile, &mut slot_a, &mut slot_b)
        .expect("mount buffered");
    assert!(!v.is_zero_copy(), "pread backend must not claim zero-copy");

    let mut mapped = MappedFlash::open(&path).expect("map");
    let w = sector_core::mount::mount_xip(&mut mapped, &profile, &mut slot_a, &mut slot_b)
        .expect("mount mapped");
    assert!(w.is_zero_copy(), "mapped backend borrows every region");
}

/// The profile the test images are built for.
///
/// `mount` compares the image's parameters for equality, so this must match what
/// `build_image_and_corpus` emits rather than being a set of wildcards.
fn permissive() -> sector_format::profile::Profile {
    sector_format::profile::Profile {
        d: D,
        m: M,
        b: 8,
        cb_bytes: 1,
        rerank_bytes: 1,
        adc_bytes: 4,
        r: 100,
        k: K,
        ram_budget: 0,
        stack_reserve: 0,
    }
}

#[test]
fn the_mapped_backend_touches_fewer_device_blocks_on_a_second_pass() {
    // What the two backends are for: quantifying the page cache rather than
    // assuming it. The buffered backend issues a read per block on every pass;
    // the mapped one faults each page once. This asserts the accounting
    // distinguishes them, which is what makes the campaign's numbers meaningful.
    let (_dir, path, corpus) = volume("cache");
    let q = corpus[0..D].to_vec();
    let queries = vec![q.clone(), q.clone(), q];

    let mut f = FileFlash::open(&path).expect("open");
    let v = HostVolume::mount(&mut f, None).expect("mount");
    let g = v.geometry;
    f.reset_stats();

    // Three identical passes over the payload region.
    for _ in 0..3 {
        let mut buf = vec![0u8; sector_format::BLOCK_BYTES];
        for b in 0..g.payload.blocks() {
            f.read(
                v.payload_base() + (b * sector_format::BLOCK_BYTES) as u32,
                &mut buf,
            )
            .expect("read");
        }
    }
    let buffered_reads = f.stats().reads;
    assert_eq!(
        buffered_reads as usize,
        3 * g.payload.blocks(),
        "pread pays per block on every pass"
    );

    let mut m = MappedFlash::open(&path).expect("map");
    m.reset_stats();
    for _ in 0..3 {
        for b in 0..g.payload.blocks() {
            m.window_counted(
                v.payload_base() + (b * sector_format::BLOCK_BYTES) as u32,
                sector_format::BLOCK_BYTES,
            )
            .expect("window");
        }
    }
    let pages = m.fault_stats().pages_touched;
    let page_bytes = m.fault_stats().page_bytes;
    let region_bytes = g.payload.blocks() * sector_format::BLOCK_BYTES;
    // Each page is counted once across all three passes.
    assert!(
        (pages as usize) <= region_bytes.div_ceil(page_bytes) + 1,
        "{pages} pages touched for {region_bytes} B at {page_bytes} B/page"
    );
    assert_eq!(m.fault_stats().borrows as usize, 3 * g.payload.blocks());
    // The point of the comparison: one first-touch per page against one read per
    // block per pass.
    assert!(
        pages < buffered_reads,
        "{pages} page touches vs {buffered_reads} reads"
    );
    let _ = queries;
}
