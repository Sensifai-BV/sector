# Datasets

Not committed. Fetch into this directory:

| Dataset | N | D | Ground truth |
|---|---:|---:|---|
| SIFT1M | 1,000,000 | 128 | provided (`.ivecs`) |
| GIST1M | 1,000,000 | 960 | provided (`.ivecs`) |

Use the ground-truth neighbour sets that ship with each dataset rather than
recomputing them: recomputation invites a metric mismatch that makes every
recall number quietly wrong.

Closing the real-embedding precondition means re-running the **full corruption
sweep** here, not only a recall comparison. The specific claim needing real-data
evidence is that the label-optimisation gain survives outside synthetic data.

## Downloading Datasets

To download and extract the datasets, run the following script:

```bash
#!/bin/bash
set -e
cd "$HOME/sector-work/datasets"
for u in ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz \
         ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz; do
  f=$(basename "$u")
  echo "fetching $f"
  curl -sS --retry 3 --connect-timeout 30 -o "$f" "$u" || { echo "FAILED $f"; continue; }
  tar xzf "$f" && rm -f "$f" && echo "extracted $f"
done
du -sh . ; ls -R . | head -20
echo DATA_DONE
```

## Running Benchmarks

To run the benchmark tests with these datasets:

```bash
#!/bin/bash
# Wait for the baseline pass so the two do not contend for 4 cores.
while pgrep -f baselines.py > /dev/null; do sleep 20; done
sleep 5
cd ~/sector-work/sector
D=~/sector-work/datasets/sift
B=./target/release/sector-bench
# Same N and same query count as the baseline run, so the comparison is
# like-for-like rather than an extrapolation.
$B recall --base $D/sift_base.fvecs --queries $D/sift_query.fvecs \
  --truth $D/sift_groundtruth.ivecs --n 100000 --nq 200 --m 16 --b 8 \
  --train-n 100000 --out sector_100k_recall
$B perf --base $D/sift_base.fvecs --queries $D/sift_query.fvecs \
  --n 100000 --nq 200 --m 16 --b 8 --r 100 --train-n 100000 --out sector_100k_perf
$B budget --base $D/sift_base.fvecs --n 100000 --m 16 --b 8 \
  --train-n 100000 --out sector_100k_budget
echo SECTOR_100K_DONE
```
