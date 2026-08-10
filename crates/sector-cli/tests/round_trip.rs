//! Host/device round-trip: the two query paths must agree byte for byte.
//!
//! A recall discrepancy between host and device should mean a backend or
//! hardware problem, never two diverging implementations. These tests build a
//! real image through the CLI's own pipeline, then check that reading it back
//! reproduces what was written and that the results are stable.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    // The integration harness sits next to the binary under test.
    let mut p = std::env::current_exe().expect("test binary path");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("sector")
}

/// A path unique to this process *and* this call.
///
/// Tests run in parallel and each builds its own image; a shared path would
/// have them deleting each other's files, which shows up as an unrelated
/// "no such file" failure.
fn tmp(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("sector_rt_{}_{n}_{name}", std::process::id()));
    p
}

/// Write a `.fvecs` file of `n` vectors in `d` dimensions.
fn write_fvecs(path: &PathBuf, n: usize, d: usize, seed: usize) {
    let mut f = std::fs::File::create(path).expect("create");
    for v in 0..n {
        f.write_all(&(d as u32).to_le_bytes()).unwrap();
        for j in 0..d {
            let x = (((v * 13 + j * 7 + seed) % 41) as f32) - 20.0;
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

struct Built {
    image: PathBuf,
    queries: PathBuf,
    corpus: PathBuf,
}

fn build_image() -> Built {
    let corpus = tmp("corpus.fvecs");
    let queries = tmp("queries.fvecs");
    let image = tmp("volume.img");
    write_fvecs(&corpus, 600, 16, 0);
    write_fvecs(&queries, 5, 16, 991);

    let out = Command::new(bin())
        .args([
            "build",
            "--input",
            corpus.to_str().unwrap(),
            "--out",
            image.to_str().unwrap(),
            "--m",
            "2",
            "--b",
            "4",
            "--r",
            "50",
        ])
        .output()
        .expect("run build");
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Built {
        image,
        queries,
        corpus,
    }
}

fn cleanup(b: &Built) {
    let _ = std::fs::remove_file(&b.image);
    let _ = std::fs::remove_file(&b.queries);
    let _ = std::fs::remove_file(&b.corpus);
}

#[test]
fn a_built_image_inspects_consistently() {
    // `inspect` recomputes the budget rather than echoing stored fields, so
    // agreement here means the emitted layout matches the profile arithmetic.
    let b = build_image();
    let out = Command::new(bin())
        .args(["inspect", "--image", b.image.to_str().unwrap()])
        .output()
        .expect("run inspect");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);

    assert!(text.contains("D=16 m=2 b=4"), "profile missing:\n{text}");
    assert!(
        text.contains("every region sector-aligned: true"),
        "alignment check failed:\n{text}"
    );
    assert!(
        text.contains("codebook replica in a different erase sector: true"),
        "replica placement failed:\n{text}"
    );
    cleanup(&b);
}

#[test]
fn the_same_image_and_queries_give_the_same_answers() {
    // Determinism is the precondition for comparing host against device: if
    // one path is not reproducible, a discrepancy proves nothing.
    let b = build_image();
    let run = || {
        let out = Command::new(bin())
            .args([
                "query",
                "--image",
                b.image.to_str().unwrap(),
                "--queries",
                b.queries.to_str().unwrap(),
                "--k",
                "10",
            ])
            .output()
            .expect("run query");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    assert_eq!(run(), run(), "two runs of one image diverged");
    cleanup(&b);
}

#[test]
fn results_carry_ids_and_scores_for_every_query() {
    let b = build_image();
    let out = Command::new(bin())
        .args([
            "query",
            "--image",
            b.image.to_str().unwrap(),
            "--queries",
            b.queries.to_str().unwrap(),
            "--k",
            "10",
        ])
        .output()
        .expect("run query");
    let text = String::from_utf8_lossy(&out.stdout);
    let rows: Vec<&str> = text.lines().filter(|l| l.starts_with('q')).collect();
    assert_eq!(rows.len(), 5, "expected one row per query:\n{text}");
    for row in &rows {
        let pairs: Vec<&str> = row.split_whitespace().skip(1).collect();
        assert_eq!(pairs.len(), 10, "expected k=10 results: {row}");
        for p in pairs {
            let (id, score) = p.split_once(':').expect("id:score");
            id.parse::<u32>().expect("numeric id");
            score.parse::<i32>().expect("numeric score");
        }
    }
    cleanup(&b);
}

#[test]
fn a_query_set_of_the_wrong_dimension_is_refused() {
    // Silently truncating or padding would produce plausible wrong answers.
    let b = build_image();
    let wrong = tmp("wrong_dim.fvecs");
    write_fvecs(&wrong, 3, 8, 5);
    let out = Command::new(bin())
        .args([
            "query",
            "--image",
            b.image.to_str().unwrap(),
            "--queries",
            wrong.to_str().unwrap(),
        ])
        .output()
        .expect("run query");
    assert!(
        !out.status.success(),
        "wrong-dimension queries were accepted"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("D="), "unhelpful error: {err}");
    let _ = std::fs::remove_file(&wrong);
    cleanup(&b);
}

#[test]
fn a_truncated_image_fails_to_mount_rather_than_mounting_partially() {
    let b = build_image();
    let truncated = tmp("truncated.img");
    let bytes = std::fs::read(&b.image).unwrap();
    std::fs::write(&truncated, &bytes[..bytes.len() / 3]).unwrap();

    let out = Command::new(bin())
        .args(["inspect", "--image", truncated.to_str().unwrap()])
        .output()
        .expect("run inspect");
    // Either the manifest fails to verify, or the regions run past the image.
    // Both are refusals; mounting partially is not.
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success() || !text.contains("consistency"),
        "a truncated image reported a full inspection"
    );
    let _ = std::fs::remove_file(&truncated);
    cleanup(&b);
}

#[test]
fn query_runs_stage_two_and_says_so() {
    // The regression guard for this file's own history.
    //
    // `query_cmd` used to reimplement stage one — an f32 inner product over every
    // vector, truncate to r, truncate to k, done — while its documentation
    // claimed to be byte-identical to the device path. It performed no rerank,
    // verified no CRC, and could not drop a candidate, so a host/device recall
    // discrepancy would have been blamed on hardware. `sector_bench::pipeline`
    // records having made the same mistake earlier.
    //
    // A claim of equivalence that nothing checks decays back to that, so this
    // asserts the observable evidence of stage two: CRC verification happened.
    let b = build_image();
    let out = Command::new(bin())
        .args([
            "query",
            "--image",
            b.image.to_str().unwrap(),
            "--queries",
            b.queries.to_str().unwrap(),
            "--k",
            "10",
        ])
        .output()
        .expect("run query");
    let text = String::from_utf8_lossy(&out.stdout);

    let verified: u32 = text
        .lines()
        .find_map(|l| l.strip_suffix(" candidates dropped"))
        .and_then(|l| l.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no CRC line in output:\n{text}"));
    assert!(
        verified > 0,
        "stage two verified no blocks, so it did not run:\n{text}"
    );
    assert!(
        text.contains("0 candidates dropped"),
        "a clean image dropped candidates:\n{text}"
    );
    cleanup(&b);
}

#[test]
fn corrupting_a_rerank_block_is_reported_as_drops_not_lost_recall() {
    // Detection is the point of the CRC, and the count is what makes damage
    // attributable: a dropped candidate and an evicted one are identical in the
    // result set, so without the counter this looks like poor recall.
    //
    // It also pins the blast radius. One flipped byte fails the whole 512 B
    // block, and every record sharing that block goes with it — the same
    // shared-structure argument the codebook makes, one level down.
    let b = build_image();

    // Locate the rerank region from `inspect` rather than assuming an offset, so
    // a layout change fails this test loudly instead of corrupting padding.
    let out = Command::new(bin())
        .args(["inspect", "--image", b.image.to_str().unwrap()])
        .output()
        .expect("run inspect");
    let text = String::from_utf8_lossy(&out.stdout);
    let base: usize = text
        .lines()
        .find(|l| l.trim_start().starts_with("Rerank "))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no Rerank region in inspect output:\n{text}"));

    let mut bytes = std::fs::read(&b.image).expect("read image");
    // A byte inside the first rerank block, which no other region shares.
    bytes[base + 5] ^= 0xFF;
    let corrupted = tmp("corrupted.img");
    std::fs::write(&corrupted, &bytes).expect("write corrupted");

    let out = Command::new(bin())
        .args([
            "query",
            "--image",
            corrupted.to_str().unwrap(),
            "--queries",
            b.queries.to_str().unwrap(),
            "--k",
            "10",
        ])
        .output()
        .expect("run query");
    assert!(
        out.status.success(),
        "corruption must degrade the result, not fail the query"
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        !text.contains("0 candidates dropped"),
        "a corrupted block was not detected:\n{text}"
    );
    assert!(
        text.contains("run `sector verify --image`"),
        "damage was reported without telling the operator what to do:\n{text}"
    );

    let _ = std::fs::remove_file(&corrupted);
    cleanup(&b);
}

#[test]
fn falsify_exits_zero_when_no_claim_is_refuted() {
    // And non-zero when one is, which is what puts a refutation in front of
    // someone rather than in a log.
    let out = Command::new(bin())
        .arg("falsify")
        .output()
        .expect("run falsify");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("margin bridge"), "suite did not run:\n{text}");
    assert!(text.contains("deflation channel"), "missing claim:\n{text}");
    assert!(
        out.status.success(),
        "a claim was refuted:\n{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
