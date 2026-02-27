"""Benchmark comparison engines on the same corpus SECTOR is measured on.

Every engine reads the same `.fvecs` slice, answers the same queries, and is
scored against the same shipped `.ivecs` ground truth that `sector-bench` uses.
An engine tuned on a different corpus or scored against a recomputed truth is
not a comparison.

Each engine contributes a recall-versus-speed CURVE rather than a single point:
a latency figure without its recall is meaningless, since any index can be made
arbitrarily fast by returning wrong answers.

Engines are never substituted. One that cannot be installed, or whose installed
form is not the engine it names, is reported with that reason.
"""

import argparse
import gc
import json
import os
import resource
import shutil
import sqlite3
import tempfile
import time

import numpy as np

# --------------------------------------------------------------------------
# Dataset loading — the same formats sector-build reads.
# --------------------------------------------------------------------------

def read_fvecs(path, limit=0):
    """Read a `.fvecs` file. Each record is a 4-byte dimension prefix then D f32."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    dim = int(raw[0])
    record = dim + 1
    if raw.size % record != 0:
        raise ValueError(
            f"{path}: {raw.size * 4} bytes is not a whole number of "
            f"{record * 4}-byte records — truncated file"
        )
    count = raw.size // record
    if limit:
        count = min(count, limit)
    view = raw.reshape(-1, record)[:count, 1:]
    return view.view(np.float32).copy(), dim


def read_ivecs(path, limit=0):
    """Read a `.ivecs` ground-truth file."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    record = dim + 1
    count = raw.size // record
    if limit:
        count = min(count, limit)
    return raw.reshape(-1, record)[:count, 1:].copy(), dim


def recall_at(found, truth, k):
    """Fraction of the first k of `truth` present in `found`."""
    if k == 0 or len(truth) == 0:
        return 0.0
    take = min(k, len(truth))
    hits = len(set(found[:max(k, len(found))]) & set(truth[:take].tolist()))
    return hits / take


def peak_rss_bytes():
    """Process high-water RSS. ru_maxrss is KiB on Linux, bytes on macOS."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v * 1024 if os.uname().sysname == "Linux" else v


def percentile(values, p):
    """Nearest-rank percentile: reports a latency some query actually took."""
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(1, int(np.ceil(p / 100.0 * len(s))))
    return s[min(rank, len(s)) - 1]


def summarize(name, kind, params, build_s, latencies, recalls, index_bytes, rss, note=""):
    """One row of the comparison, with everything a reader needs to judge it."""
    n = len(latencies)
    total = sum(latencies)
    return {
        "engine": name,
        "kind": kind,
        "params": params,
        "build_seconds": build_s,
        "queries": n,
        "latency_ms": {
            "p50": percentile(latencies, 50) * 1e3,
            "p95": percentile(latencies, 95) * 1e3,
            "p99": percentile(latencies, 99) * 1e3,
        },
        "qps": (n / total) if total > 0 else 0.0,
        "recall": {f"@{k}": v for k, v in recalls.items()},
        "index_bytes": index_bytes,
        "peak_rss_bytes": rss,
        "note": note,
    }


def dir_bytes(path):
    """Total bytes under `path`, so an index's real on-disk cost is measured."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------

def run_sqlite_vec(base, queries, truth, ks, k_max):
    """sqlite-vec: exact brute-force scan through a vec0 virtual table.

    No approximate index — vec0 scans every row — so this is the exact-search
    reference point, and its recall is 1.0 by construction. Its value in the
    comparison is the cost of exactness, not the accuracy.
    """
    import sqlite_vec

    d = base.shape[1]
    path = tempfile.mktemp(suffix=".db")
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    t0 = time.perf_counter()
    db.execute(f"create virtual table v using vec0(embedding float[{d}])")
    db.executemany(
        "insert into v(rowid, embedding) values (?, ?)",
        ((i, sqlite_vec.serialize_float32(base[i])) for i in range(len(base))),
    )
    db.commit()
    build_s = time.perf_counter() - t0

    latencies, recalls = [], {k: 0.0 for k in ks}
    for qi in range(len(queries)):
        blob = sqlite_vec.serialize_float32(queries[qi])
        t = time.perf_counter()
        rows = db.execute(
            "select rowid from v where embedding match ? order by distance limit ?",
            (blob, k_max),
        ).fetchall()
        latencies.append(time.perf_counter() - t)
        found = [r[0] for r in rows]
        for k in ks:
            recalls[k] += recall_at(found, truth[qi], k)

    for k in ks:
        recalls[k] /= max(1, len(queries))
    size = os.path.getsize(path)
    db.close()
    os.unlink(path)
    return [summarize(
        "sqlite-vec", "exact brute-force scan (vec0)", {},
        build_s, latencies, recalls, size, peak_rss_bytes(),
        "No ANN index: vec0 scans every row, so recall is exact by construction.",
    )]


def run_usearch(base, queries, truth, ks, k_max, sweep):
    """usearch: HNSW, swept over search expansion for a recall-speed curve."""
    from usearch.index import Index

    rows = []
    d = base.shape[1]
    t0 = time.perf_counter()
    idx = Index(ndim=d, metric="l2sq", dtype="f32", connectivity=16)
    idx.add(np.arange(len(base)), base)
    build_s = time.perf_counter() - t0

    for ef in sweep:
        # expansion_search is a property of the Index, not a search argument.
        # Setting it per query had no effect, which showed up as three
        # identical rows -- a sweep that does not sweep.
        idx.expansion_search = ef
        latencies, recalls = [], {k: 0.0 for k in ks}
        for qi in range(len(queries)):
            t = time.perf_counter()
            m = idx.search(queries[qi], k_max, exact=False)
            latencies.append(time.perf_counter() - t)
            found = list(m.keys)
            for k in ks:
                recalls[k] += recall_at(found, truth[qi], k)
        for k in ks:
            recalls[k] /= max(1, len(queries))
        rows.append(summarize(
            "usearch", "HNSW graph", {"connectivity": 16, "expansion_search": ef},
            build_s, latencies, recalls, idx.memory_usage, peak_rss_bytes(),
        ))
    return rows


def run_lancedb(base, queries, truth, ks, k_max, nprobes_sweep):
    """LanceDB IVF_PQ — the closest architectural comparator: also PQ, also disk."""
    import lancedb
    import pyarrow as pa

    d = base.shape[1]
    root = tempfile.mkdtemp()
    db = lancedb.connect(root)

    t0 = time.perf_counter()
    table = pa.table({
        "id": pa.array(np.arange(len(base), dtype=np.int64)),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(base.reshape(-1), type=pa.float32()), d
        ),
    })
    tbl = db.create_table("bench", data=table)
    # Partition count follows the usual sqrt(N) heuristic; PQ subvectors match
    # SECTOR's m so the two are configured comparably.
    partitions = max(1, int(np.sqrt(len(base))))
    sub_vectors = 16 if d % 16 == 0 else 8
    tbl.create_index(
        metric="l2",
        num_partitions=partitions,
        num_sub_vectors=sub_vectors,
        index_type="IVF_PQ",
    )
    build_s = time.perf_counter() - t0

    rows = []
    for nprobes in nprobes_sweep:
        latencies, recalls = [], {k: 0.0 for k in ks}
        for qi in range(len(queries)):
            t = time.perf_counter()
            res = (
                tbl.search(queries[qi])
                .nprobes(nprobes)
                .limit(k_max)
                .select(["id", "_distance"])
                .to_list()
            )
            latencies.append(time.perf_counter() - t)
            found = [r["id"] for r in res]
            for k in ks:
                recalls[k] += recall_at(found, truth[qi], k)
        for k in ks:
            recalls[k] /= max(1, len(queries))
        rows.append(summarize(
            "lancedb", "Lance columnar + IVF_PQ",
            {"num_partitions": partitions, "num_sub_vectors": sub_vectors,
             "nprobes": nprobes},
            build_s, latencies, recalls, dir_bytes(root), peak_rss_bytes(),
        ))
    shutil.rmtree(root, ignore_errors=True)
    return rows


def run_qdrant_local(base, queries, truth, ks, k_max):
    """Qdrant client in local mode.

    This is NOT the Qdrant engine. Local mode is a NumPy exact-scan
    reimplementation inside the Python client; the real engine is a Rust HNSW
    server. The row is emitted with that stated, because presenting it as
    "Qdrant" would attribute the client's fallback to the product.
    """
    from qdrant_client import QdrantClient, models

    d = base.shape[1]
    cl = QdrantClient(":memory:")
    t0 = time.perf_counter()
    cl.create_collection(
        "bench",
        vectors_config=models.VectorParams(size=d, distance=models.Distance.EUCLID),
    )
    step = 2000
    for i in range(0, len(base), step):
        chunk = base[i:i + step]
        cl.upsert("bench", points=[
            models.PointStruct(id=int(i + j), vector=chunk[j].tolist())
            for j in range(len(chunk))
        ])
    build_s = time.perf_counter() - t0

    latencies, recalls = [], {k: 0.0 for k in ks}
    for qi in range(len(queries)):
        t = time.perf_counter()
        res = cl.query_points("bench", query=queries[qi].tolist(), limit=k_max).points
        latencies.append(time.perf_counter() - t)
        found = [p.id for p in res]
        for k in ks:
            recalls[k] += recall_at(found, truth[qi], k)
    for k in ks:
        recalls[k] /= max(1, len(queries))
    return [summarize(
        "qdrant (local mode)", "NumPy exact scan in the Python client", {},
        build_s, latencies, recalls, 0, peak_rss_bytes(),
        "NOT the Qdrant engine. Local mode is a client-side exact-scan "
        "reimplementation; the product is a Rust HNSW server, whose binary "
        "could not be downloaded from this host.",
    )]


ENGINES = {
    "sqlite-vec": run_sqlite_vec,
    "usearch": run_usearch,
    "lancedb": run_lancedb,
    "qdrant": run_qdrant_local,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--n", type=int, default=0, help="corpus vectors; 0 = all")
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--out", default="measurements/baselines.json")
    ap.add_argument("--engines", default="sqlite-vec,usearch,lancedb,qdrant")
    args = ap.parse_args()

    base, d = read_fvecs(args.base, args.n)
    queries, dq = read_fvecs(args.queries, args.nq)
    if dq != d:
        raise SystemExit(f"queries are D={dq} but base is D={d}")
    truth, tk = read_ivecs(args.truth, args.nq)

    subset = args.n and args.n < len(np.fromfile(args.base, dtype=np.int32)) // (d + 1)
    if subset:
        # The shipped ground truth indexes the full corpus, so on a subset it
        # names vectors that are not present. Recomputing exactly is correct;
        # scoring against the shipped rows anyway would be a plausible number
        # that means nothing.
        print(f"subset of {len(base)}: recomputing exact ground truth", flush=True)
        truth = np.zeros((len(queries), 100), dtype=np.int32)
        for qi in range(len(queries)):
            dist = ((base - queries[qi]) ** 2).sum(axis=1)
            truth[qi] = np.argsort(dist, kind="stable")[:100]
        truth_source = "recomputed exactly for the subset"
    else:
        truth_source = "shipped with the dataset"

    ks = [1, 10, 100]
    k_max = 100
    print(f"base {len(base)} x {d}, {len(queries)} queries, truth k={truth.shape[1]}",
          flush=True)

    results, failures = [], []
    for name in args.engines.split(","):
        name = name.strip()
        fn = ENGINES.get(name)
        if fn is None:
            failures.append({"engine": name, "reason": "unknown engine"})
            continue
        print(f"### {name}", flush=True)
        gc.collect()
        try:
            if name == "usearch":
                rows = fn(base, queries, truth, ks, k_max, [64, 128, 256])
            elif name == "lancedb":
                rows = fn(base, queries, truth, ks, k_max, [1, 4, 16, 64])
            else:
                rows = fn(base, queries, truth, ks, k_max)
            for r in rows:
                print(f"  {r['params']}  recall@10={r['recall']['@10']:.4f}  "
                      f"p50={r['latency_ms']['p50']:.3f}ms  qps={r['qps']:.1f}",
                      flush=True)
            results.extend(rows)
        except Exception as e:  # an engine that fails is reported, not hidden
            import traceback
            failures.append({
                "engine": name,
                "reason": f"{type(e).__name__}: {e}"[:400],
                "traceback": traceback.format_exc()[-800:],
            })
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    out = {
        "measurement": "baselines",
        "dataset": args.base,
        "config": {"d": d, "n": len(base), "queries": len(queries),
                   "k_values": ks, "k_max": k_max},
        "ground_truth": truth_source,
        "results": results,
        "failures": failures,
        "unavailable": [{
            "engine": "DiskANN",
            "reason": "no aarch64 wheel on PyPI; source build needs Intel MKL, "
                      "which has no aarch64 target. Not benchmarked, not substituted.",
        }],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
