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
