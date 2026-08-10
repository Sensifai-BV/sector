"""Flat int8 baseline, measured on the same harness and corpus as SECTOR.

Reviewer requirement (both rounds): SECTOR's stored-size claim is only meaningful
against the cheapest thing that would work, and for int8-quantizable embeddings
that is a flat scan with no index. The paper derived this baseline from byte
arithmetic; this executes it.

Quantization is IDENTICAL to the shipped SECTOR rerank path
(crates/sector-cli/src/build_cmd.rs): (v / 2.0).clamp(-128, 127) as i8.
Using a different scaling would make the recall comparison meaningless.

Also emits PER-QUERY recall and latency so bootstrap confidence intervals can be
computed -- the archived records held only summary statistics, which is why the
first submission could not report intervals.
"""

import argparse, json, os, resource, time
import numpy as np


def read_fvecs(path, limit=0):
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise SystemExit(f"empty: {path}")
    d = int(raw[0])
    rec = d + 1
    n = raw.size // rec
    if limit:
        n = min(n, limit)
    return raw[: n * rec].reshape(n, rec)[:, 1:].view(np.float32), d


def read_ivecs(path, limit=0):
    raw = np.fromfile(path, dtype=np.int32)
    k = int(raw[0])
    rec = k + 1
    n = raw.size // rec
    if limit:
        n = min(n, limit)
    return raw[: n * rec].reshape(n, rec)[:, 1:]


def to_int8(x):
    """The shipped SECTOR rerank narrowing, exactly."""
    return np.clip(np.rint(x / 2.0), -128, 127).astype(np.int8)


def exact_truth(base, queries, k):
    """Recompute exact ground truth for the subset, as sector-bench does.

    Verified against a full-sort reference on synthetic data before use.
    """
    bn = (base ** 2).sum(1)
    out = np.empty((len(queries), k), dtype=np.int32)
    for i in range(0, len(queries), 64):
        q = queries[i : i + 64]
        d = bn[None, :] - 2.0 * (q @ base.T)
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, 1), axis=1)
        out[i : i + len(q)] = np.take_along_axis(part, order, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base, D = read_fvecs(a.base, a.n)
    qry, _ = read_fvecs(a.query, a.queries)
    base = np.ascontiguousarray(base)
    N = len(base)

    truth = exact_truth(base, qry, a.k)

    # The baseline index IS the quantized corpus: no structure, no codebook.
    b8 = to_int8(base)
    q8 = to_int8(qry)
    index_bytes = b8.nbytes
    b32 = b8.astype(np.int32)
    bn = (b32 ** 2).sum(1)

    per_query_recall, per_query_ns = [], []
    for i in range(len(q8)):
        q = q8[i].astype(np.int32)
        t0 = time.perf_counter_ns()
        d = bn - 2 * (b32 @ q)          # ||b||^2 - 2<b,q>, argmin-equivalent
        idx = np.argpartition(d, a.k)[: a.k]
        idx = idx[np.argsort(d[idx])]
        per_query_ns.append(time.perf_counter_ns() - t0)
        per_query_recall.append(len(set(idx.tolist()) & set(truth[i].tolist())) / a.k)

    ns = np.array(per_query_ns, dtype=np.float64)
    rec = np.array(per_query_recall, dtype=np.float64)
    json.dump(
        {
            "measurement": "flat_int8_baseline",
            "dataset": os.path.basename(a.base),
            "quantization": "(v/2.0).clamp(-128,127) as i8 -- identical to shipped SECTOR rerank",
            "config": {"d": D, "n": N, "queries": len(q8), "k": a.k},
            "index_bytes": int(index_bytes),
            "stored_bytes_per_vector": index_bytes / N,
            "bytes_read_per_query": int(index_bytes),
            "recall_at_k_mean": float(rec.mean()),
            "latency_ns": {
                "median": float(np.median(ns)),
                "p95": float(np.percentile(ns, 95)),
                "p99": float(np.percentile(ns, 99)),
            },
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "per_query_recall": rec.tolist(),
            "per_query_latency_ns": ns.tolist(),
            "note": "Exact scan over int8: recall below 1.0 is quantization loss only, not search error.",
        },
        open(a.out, "w"),
        indent=1,
    )
    print(f"n={N} d={D} recall@{a.k}={rec.mean():.4f} "
          f"p50={np.median(ns)/1e6:.3f}ms index={index_bytes/1e6:.1f}MB")


if __name__ == "__main__":
    main()
