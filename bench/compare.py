"""Assemble the SECTOR-versus-baselines comparison from measured JSON.

Reads only what the harnesses wrote. Nothing is recomputed here: an analysis
that re-derives a number can disagree with the measurement it claims to
summarise, and the disagreement is invisible.

Two kinds of axis are reported, and they are kept apart on purpose.

*Measured axes* — recall, latency, throughput, index bytes, build time — are
numbers every engine produced under identical conditions.

*Capability axes* — microcontroller viability, flash-write awareness, energy
reporting — are properties no engine in this comparison provides, established
from documented behaviour rather than from measurement. They are labelled as
such, because presenting a documented absence as a measured zero would overstate
what was tested.
"""

import argparse
import json
import os


# Capability assessment. These are NOT measurements: each entry records what an
# engine's own documentation and source say it targets. Kept separate from the
# measured table so a reader can see which is which.
CAPABILITIES = {
    "sqlite-vec": {
        "mcu_viable": False,
        "mcu_note": "requires a filesystem and SQLite; no no_std build",
        "flash_aware": False,
        "flash_note": "SQLite page cache; no wear or write-cost model",
        "energy_reported": False,
    },
    "usearch": {
        "mcu_viable": False,
        "mcu_note": "C++ with allocation; HNSW graph is RAM-resident",
        "flash_aware": False,
        "flash_note": "memory-mapped index; no wear model",
        "energy_reported": False,
    },
    "lancedb": {
        "mcu_viable": False,
        "mcu_note": "Rust + Arrow runtime, object-store oriented",
        "flash_aware": False,
        "flash_note": "columnar IO for throughput, not erase-block wear",
        "energy_reported": False,
    },
    "qdrant (local mode)": {
        "mcu_viable": False,
        "mcu_note": "server product; local mode is a Python/NumPy fallback",
        "flash_aware": False,
        "flash_note": "mmap storage; no wear model",
        "energy_reported": False,
    },
    "SECTOR": {
        "mcu_viable": True,
        "mcu_note": "no_std, no alloc, integer-only; workspace sized by const fn",
        "flash_aware": True,
        "flash_note": "erase-sector-aligned regions, wear-aware scrub, program-once enforced",
        "energy_reported": "inputs only",
    },
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pareto(rows):
    """Rows not dominated on both recall@10 and queries/s."""
    keep = []
    for r in rows:
        rec, qps = r["recall_at_10"], r["qps"]
        if not any(
            o["recall_at_10"] >= rec and o["qps"] >= qps
            and (o["recall_at_10"] > rec or o["qps"] > qps)
            for o in rows
        ):
            keep.append(r)
    return sorted(keep, key=lambda r: r["recall_at_10"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--sector-recall", required=True)
    ap.add_argument("--sector-perf", required=True)
    ap.add_argument("--sector-budget", required=True)
    ap.add_argument("--out", default="measurements/comparison.json")
    args = ap.parse_args()

    base = load(args.baselines)
    if base is None:
        raise SystemExit(f"missing {args.baselines}")
    rec = load(args.sector_recall)
    perf = load(args.sector_perf)
    budget = load(args.sector_budget)

    rows = []
    for r in base["results"]:
        rows.append({
            "engine": r["engine"],
            "kind": r["kind"],
            "params": r["params"],
            "recall_at_1": r["recall"].get("@1", 0.0),
            "recall_at_10": r["recall"].get("@10", 0.0),
            "recall_at_100": r["recall"].get("@100", 0.0),
            "p50_ms": r["latency_ms"]["p50"],
            "p99_ms": r["latency_ms"]["p99"],
            "qps": r["qps"],
            "build_s": r["build_seconds"],
            "index_bytes": r["index_bytes"],
            "note": r.get("note", ""),
        })

    # SECTOR's own row, assembled from its three measurement files. Its R sweep
    # is the tuning knob, matching how every baseline is swept.
    if rec and perf:
        by_depth = {d["r"]: d for d in rec["by_depth"]}
        median_ms = perf["latency_ns"]["median"] / 1e6
        p99_ms = perf["latency_ns"]["p99"] / 1e6
        measured_r = perf["config"]["r"]
        for R, row in by_depth.items():
            # Latency was measured at one depth. Stage one is O(N) and depth-
            # independent; only the rerank term scales with R, and it is 0.05%
            # of the total here, so the scan-dominated latency is reused across
            # depths and the assumption is recorded rather than hidden.
            rows.append({
                "engine": "SECTOR",
                "kind": "two-stage PQ scan + rerank, no graph index",
                "params": {"R": R, "m": perf["config"]["m"], "b": perf["config"]["b"]},
                "recall_at_1": row["two_stage_recall"][0],
                "recall_at_10": row["two_stage_recall"][1],
                "recall_at_100": row["two_stage_recall"][2],
                "p50_ms": median_ms,
                "p99_ms": p99_ms,
                "qps": perf["throughput_qps"],
                "build_s": rec.get("build_seconds", 0.0),
                "index_bytes": (budget or {}).get("disk", {}).get("image_bytes", 0),
                "note": (
                    f"latency measured at R={measured_r} and reused across the R "
                    f"sweep: stage one is O(N) and depth-independent, and rerank "
                    f"is {perf['phases'][3]['share_permille']}permille of query time"
                ),
            })

    front = pareto(rows)

    caps = []
    for name in sorted({r["engine"] for r in rows}):
        c = CAPABILITIES.get(name, {})
        caps.append({
            "engine": name,
            "microcontroller_viable": c.get("mcu_viable"),
            "microcontroller_note": c.get("mcu_note", ""),
            "flash_write_aware": c.get("flash_aware"),
            "flash_note": c.get("flash_note", ""),
            "energy_reported": c.get("energy_reported"),
        })

    out = {
        "measurement": "comparison",
        "corpus": base["config"],
        "ground_truth": base["ground_truth"],
        "measured_axes": rows,
        "pareto_frontier": front,
        "capability_axes": {
            "basis": (
                "documented behaviour and source, NOT measured on this host — "
                "an absence established from documentation is not a measured zero"
            ),
            "engines": caps,
        },
        "unavailable": base.get("unavailable", []),
        "failures": base.get("failures", []),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"{'engine':<22} {'r@10':>7} {'p50 ms':>9} {'qps':>9} {'idx MB':>9}")
    for r in sorted(rows, key=lambda r: -r["recall_at_10"]):
        print(f"{r['engine']:<22} {r['recall_at_10']:>7.4f} {r['p50_ms']:>9.2f} "
              f"{r['qps']:>9.1f} {r['index_bytes']/1e6:>9.2f}")
    print(f"\nPareto frontier: {[(r['engine'], round(r['recall_at_10'],4)) for r in front]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
