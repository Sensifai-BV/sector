# Figures

Referenced by `../BENCHMARKS.md` with relative paths, so the report renders on
GitHub, in an editor, and offline.

Filenames name the content. The earlier names (`benchmark.png`,
`comparison_v2.png`) said nothing about what the figure showed, which is how a
superseded figure gets quoted.

| file | what it shows | source data |
|---|---|---|
| `faults_and_phase_scaling.png` | recall against damage fraction for four corruption channels; per-phase time against `N` | `measurements/sift_slice_faults.json`, `measurements/phase_n{1000,2500,5000}.json` |
| `corruption_sweep_bound.png` | measured loss against the two-sided perturbation bound, both signs | `measurements/` sweep dump |
| `criticality_allocation.png` | per-centroid signed exposure and the parity allocation at five budgets | `sector-build --example dump_criticality` |
| `engine_comparison_sift.png` | recall against p50 latency for five engines, and index size, SIFT1M N=100,000 | `measurements/baselines_100k.json`, `measurements/sector_100k_*.json` |
| `gist_dimension_recall.png` | recall against candidate depth at three dimension/width points; codebook size against `D` | `measurements/gist_*_recall.json`, `measurements/gist_feasibility.json` |

## Regenerating

Panel B of `faults_and_phase_scaling.png` is measured, not illustrative:

```sh
for N in 1000 2500 5000; do
  ./target/release/sector-bench perf --base <sift.fvecs> --queries <query.fvecs> \
    --n $N --nq 60 --m 16 --b 8 --r 100 --train-n 5000 --out phase_n$N
done
```

## Two figures are deliberately absent

`measurements/comparison.png` and `measurements/benchmark.png` are the
pre-optimisation versions of `engine_comparison_sift.png` and the phase panel.
They report SECTOR at 22.0 ms and a 93% scan share, both measured before the
benchmark harness was routed through the engine's own bounded-heap scan. They
are kept in `measurements/` as the record of what was measured when, and are not
referenced here: two figures differing only by stale numbers is how the wrong one
ends up in a table.
