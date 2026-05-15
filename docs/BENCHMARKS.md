# SECTOR — benchmark report

Every number here was measured on a Raspberry Pi 4B (Ubuntu 24.04, aarch64,
4 cores, 7.8 GB RAM, SD-card root), single-threaded, against real SIFT1M
vectors. Each figure carries its configuration; a number without one is a
reporting failure.

Two preconditions gated this work. **P2 — recall on real embeddings — is
closed.** **P1 — joules per query and NOR timings — is not**, and cannot be
closed on this host: a Pi 4B draws 3-6 W against an ESP32-C3's 50-100 mW, so an
energy figure taken here would describe the wrong machine. What was measured
instead is the energy model's *inputs*.

---

## 1. Recall — P2, closed

Full SIFT1M, **N = 1,000,000**, D=128, m=16, b=8, 200 queries, scored against
the **shipped** ground truth (`sift_groundtruth.ivecs`, k=100). Recomputing
ground truth would admit a metric mismatch that biases every number in the same
direction, so it was never recomputed where the shipped rows apply.

| R | PQ only @10 | two-stage @10 | report (synthetic) | delta |
|---:|---:|---:|---:|---:|
| 100 | 0.5370 | **0.9655** | 0.605 | +0.3605 |
| 500 | 0.5370 | **0.9985** | 0.934 | +0.0645 |

Full sweep, two-stage recall@10: 0.5370 at R=10,
0.8975 at R=50, 0.9655
at R=100, 0.9985 at R=500. recall@1 reaches 1.0000
by R=100.

**The result runs opposite to the stated risk.** P2 existed because the
synthetic corpus's margin distribution was the property most likely to differ on
real data, and the expectation was that recall might come back materially worse.
It came back materially better — **+0.36 at R=100**. Every recall figure in the
report taken at this configuration understates what SIFT1M delivers.

Two caveats so the figure is not over-read. 200 queries gives a measurement
resolution of 1/(200x10) = 5e-4, so differences below that are ties. The
codebook was trained on the 100,000-vector learn set the dataset ships for the
purpose — standard practice, since the codebook is 2^b x D bytes regardless of
N; full-corpus k-means was measured at roughly 3.8 hours against 7 minutes for
the sample. All 1,000,000 vectors are encoded and scanned.

Index build: 429.7 s, peak RSS 2.1 GB.

---

## 2. Performance

At N = 1,000,000, R=100, 100 queries:

| quantity | value |
|---|---:|
| median latency | 304.9 ms |
| p95 | 307.7 ms |
| p99 | 308.8 ms |
| throughput | 3.25 queries/s |

Per phase, with the clock calibrated rather than trusted (37 ns resolution,
50 ns mark overhead, two overheads subtracted per phase):

| phase | mean | share |
|---|---:|---:|
| scan | 304.7 ms | **998/1000** |
| table build | 171.9 us | 0/1000 |
| rerank | 145.4 us | 0/1000 |
| rotate, finalize | below the 37 ns clock resolution | — |

**Both structural claims of the cost model hold; its conclusion does not.**
Table build is 2^b x D = 32,768 MACs independent of N — measured flat at ~8.6 us
across N=1,000 to 5,000 — and scan is linear in N, measured at 39/106/228 us
across the same range. But the report's expectation that **Table and Rerank
dominate** is false by three orders of magnitude: scan is 99.8% of query time
and table build is 0.06% of it. The scan share rose monotonically with corpus
size at every N measured (734/1000 at N=1,000 to 998/1000 at N=1,000,000), so
this is a trend and not one anomalous point.

The qualification stands: this is an out-of-order 1.5 GHz ARM core with the
corpus in RAM. On NOR flash at T0 the balance moves, because scan streams from
flash while table build does not. What the measurement establishes is that the
dominance claim is **not free-standing** — it depends on the storage path and on
N, and on this host at this scale it is wrong.

### An open defect in the harness, not the engine

`sector-bench`'s `Pipeline::stage_one` allocates `Vec::with_capacity(N)`, scores
every vector into it, then sorts all 1,000,000 entries and truncates to R.
`sector_core::scan_b8` has done the right thing since Phase 4 — a `would_accept`
threshold test against the heap minimum, then a bounded heap of R, never sorting
N. The harness reimplemented stage one instead of calling the engine, and that
reimplementation is what was timed.

The arithmetic agrees. Measured scan throughput is **53 MB/s** — 16.0 MB read per
query in 304.7 ms — against the 2-4 GB/s a single ARM
core reaches on sequential RAM. Roughly 38x below memory bandwidth is not what an
m-lookup, m-add inner loop costs; it is what sorting 10^6 elements in f32 costs.

Three fixes, in order: route the benchmark through `sector_core`'s real scan
path (which also removes a divergence risk between two implementations); score
in i32 rather than f32, since the device path is integer-only by design; then
NEON over the inner loop.

---

## 3. Memory and disk budgets

All predictions hold **exactly, at zero tolerance**, at N = 1,000,000:

| claim | predicted | measured | verdict |
|---|---:|---:|---|
| codebook (2^b x D) | 32,768 B | 32,768 B | holds |
| payload per vector (m x b / 8) | 16 B | 16 B | holds |
| codes array (N x m) | 16,000,000 B | 16,000,000 B | holds |

Image: 145.2 MB, 145.19 B/vector,
31,250 payload blocks, 250,000 rerank blocks.

**The 0.81% protection claim holds: 0.797% measured.**
An earlier run at N=5,000 measured 4.85%, which was not a discrepancy — the
codebook replica is a fixed 32 KiB cost independent of N, so its share falls as N
grows. Same arithmetic, two scales. The harness now reports the fraction with N
attached so the two cannot be confused.

---

## 4. Fault tolerance

Four channels injected independently into a real index, recall re-measured
through the same query path, every loss relative to the clean two-stage baseline
rather than an oracle. N=3,000, R=100, k=10, clean recall 1.0000.

| channel | dose | recall | loss |
|---|---|---:|---:|
| codebook bit flips | 64 bits | 0.9980 | 0.0020 |
| codebook bit flips | 256 bits | 0.9820 | 0.0180 |
| payload bit flips | 1.07% of code bits | 0.9700 | 0.0300 |
| payload bit flips | 4.27% | 0.8100 | 0.1900 |
| block drops | 512 vectors | 0.8760 | 0.1240 |
| block drops | 1,408 vectors | 0.5540 | 0.4460 |
| **correlated sector erase** | **4,096 codebook bytes** | **0.9960** | **0.0040** |

All four degrade monotonically, no cliffs.

**The sector result is the notable one.** Erasing one 4 KiB sector destroys
12.5% of the codebook — 256 centroids' worth of components — and costs 0.0040
recall. This is the depth-margin property measured rather than argued: a true
neighbour at median depth survives unless enough intruders displace it, so
corruption below that threshold is harmless regardless of how many vectors it
touched. The report predicted this; it is now observed on real embeddings.

Block drops are the most damaging channel per byte, which follows — a dropped
block removes its vectors outright, with no margin to absorb the loss.

![Recall against damage fraction for four corruption channels, and per-phase
time against corpus size after the scan optimisation.](figures/faults_and_phase_scaling.png)

The two figures behind the protection design, from the falsification suite and
the host builder:

![Measured recall loss against the two-sided perturbation bound, for both signs
of centroid displacement.](figures/corruption_sweep_bound.png)

Loss is plotted against the bound for **both** signs of displacement, because an
inflation-only sweep misses centroids whose exposure is entirely on the
deflating side.

![Per-centroid signed exposure, and the parity allocation it produces across
protection groups at five budgets.](figures/criticality_allocation.png)

That asymmetry is not a corner case: **58.3% of total measured weight sits on
centroids with zero inflation loss**, which an inflation-only measurement would
weight at zero. Groups are ordered by measured weight, and parity is
non-increasing in weight rank at every budget plotted — verified against each
row before rendering.

---

## 5. Comparison against established engines

SIFT1M, **N = 100,000**, D=128, 200 queries. Every engine reads the same corpus,
answers the same queries, and is scored against the same exact ground truth
recomputed for the subset. Each contributes a recall-versus-speed **curve**: a
latency figure without its recall is meaningless, since any index can be made
arbitrarily fast by returning wrong answers.

| engine | kind | recall@10 | p50 | index |
|---|---|---:|---:|---:|
| usearch | HNSW | 0.9945-0.9995 | 1.5-2.6 ms | 85.3 MB |
| LanceDB | IVF_PQ | 0.4055-0.9955 | 16.7-25.3 ms | 53.5 MB |
| **SECTOR** | two-stage PQ scan | 0.6420-1.0000 | **22.0 ms** † | **14.5 MB** |
| sqlite-vec | exact scan (vec0) | 1.0000 | 143 ms | 53.7 MB |
| Qdrant local mode | NumPy exact scan | 1.0000 | 244 ms | — |

† Superseded. This latency was mostly a benchmark artifact — see §7, where
SECTOR measures **4.90 ms** at the same recall. The row is kept because it is
what was measured at the time, not because it is the current figure.

**Where SECTOR loses.** usearch answers at 1.5 ms against SECTOR's
22.0 ms — **15x** — at comparable recall (0.9945 against
0.9910 at R=100). That is the honest result and it
is not a surprise: SECTOR scans exhaustively and generates no candidates, so a
graph index wins on latency at any N large enough for the graph to pay for
itself. Below T2 an index has no room to exist, which is where the contribution
is aimed.

**Where SECTOR wins.** Index size: 14.5 MB against 53.5-85.3 MB, **3.7-5.9x
smaller** than every baseline. On a part where flash is the binding constraint,
that is the axis that decides feasibility.

*(Figure superseded by the post-optimisation version in §7. SECTOR's latency
here was 22.0 ms; measured at 4.90 ms after the harness was routed through the
engine's own scan. Only the corrected figure is kept, so a stale row cannot be
quoted from this document.)*

### A correction to an earlier figure in this project

Before the matched run existed, SECTOR's N=100,000 latency was *extrapolated*
from the N=1,000,000 measurement by dividing by ten, since scan is linear in N,
giving 30.5 ms and a 21x gap. Measured directly it is 22.0 ms and
the gap is 15x — the extrapolation was 1.38x too pessimistic. The arithmetic was
right and the assumption was not: 1.6 MB of codes fits in cache and 16 MB does
not, so scan is linear in N only within a cache regime. **The measured 15x
supersedes the extrapolated 21x.**

### Engines not benchmarked, and why

**DiskANN** has no aarch64 wheel on PyPI, and a source build needs Intel MKL,
which has no aarch64 target. It is reported unavailable rather than replaced by
a stand-in — benchmarking a different graph index in its place would be a
comparison against something else.

**Qdrant's local mode is not the Qdrant engine.** `QdrantClient(":memory:")` is a
NumPy exact-scan reimplementation inside the Python client; the product is a Rust
HNSW server whose binary could not be downloaded from this host. Its row is
labelled on the row itself, because calling it plain "Qdrant" would attribute the
client's fallback to the product.

**sqlite-vec has no ANN index** — `vec0` scans exhaustively — so its recall is
1.0 by construction and its contribution is the cost of exactness, not accuracy.

### Axes no engine in this comparison addresses

These are **not measurements**. Each is established from an engine's documented
behaviour and source, and is labelled as such: presenting a documented absence as
a measured zero would overstate what was tested.

| engine | microcontroller viable | flash-write aware | reports energy |
|---|---|---|---|
| sqlite-vec | no — needs a filesystem and SQLite | no | no |
| usearch | no — RAM-resident graph, allocates | no | no |
| LanceDB | no — Rust + Arrow, object-store oriented | no | no |
| Qdrant | no — server product | no | no |
| **SECTOR** | **yes** — no_std, no alloc, integer-only | **yes** — sector-aligned, wear-aware scrub | inputs only |

---

## 6. What closed, what did not

**Closed.** P2 — recall on real embeddings, and the answer was better than
feared. All budget predictions at N=10^6. Fault tolerance across four channels
on real vectors. The comparison against four established engines.

**Not closed.** P1 — joules per query and NOR write/erase timings. This needs an
ESP32 plus a shunt and a scope; the firmware skeleton, the GPIO phase
instrument and the measurement binaries exist and build for riscv32imc, but
nothing has run on the hardware. What is available now is the energy model's
inputs — cycles and bytes per phase — leaving `P_active` and `E_per_byte` as the
two platform constants a hardware measurement must supply.

**Still running.** The baseline pass at full SIFT1M (usearch and LanceDB only —
the two exact-scan engines would take roughly 25 minutes per query set at 10^6
and add nothing beyond the cost of exactness already measured at 10^5). The
matched comparison at N=100,000 above is complete and does not depend on it.

**Deferred.** GIST1M at D=960. The dataset is on disk (5.4 GB) but at D=960 a
b=8 codebook is 245 KB, over the entire T0 192 KiB budget — which the profile
already asserts at compile time. Running it needs a feasible configuration
chosen first rather than a run at a configuration that cannot ship.

**Refuted by measurement.** The cost model's conclusion that Table and Rerank
dominate query cost. Its two structural claims survive; the conclusion drawn
from them does not.

---

## 7. Optimisation — 4.5x, and what it changed

The comparison in section 5 measured SECTOR at 22.0 ms and reported a 15x gap
against usearch's HNSW. **Most of that was the benchmark, not the engine.**

`sector-bench`'s `Pipeline::stage_one` allocated `Vec::with_capacity(N)`, scored
every vector into it, sorted all of them and truncated to `R`.
`sector_core::scan_b8` has done the right thing since Phase 4 — a threshold test
against the heap minimum, then a bounded heap of `R`, never sorting `N`. The
harness had reimplemented stage one instead of calling the engine, and that
reimplementation is what was timed.

Routing it through the engine, measured on the Pi at N=100,000 against the same
corpus, queries and ground truth:

| phase | before | after | factor |
|---|---:|---:|---:|
| scan | 21.71 ms | **4.70 ms** | 4.62x |
| total | 22.03 ms | **4.90 ms** | 4.50x |

Scan throughput rose from 73 MB/s to **340 MB/s**. Recall is unchanged within
the query-quantization difference — 0.9895 at R=100 against 0.9910, and 1.0000
at R=500 — which is the point: a faster wrong answer is not an improvement, and
the exact returned identifiers were locked before any scoring code was touched.

The same factor holds at full scale. At **N = 1,000,000**:

| quantity | before | after | factor |
|---|---:|---:|---:|
| scan | 304.7 ms | **69.3 ms** | 4.39x |
| total | 304.9 ms | **69.6 ms** | 4.38x |
| throughput | 3.28 q/s | **14.36 q/s** | 4.38x |
| scan bandwidth | 53 MB/s | **231 MB/s** | 4.4x |

Recall at 10^6 moves by less than the quantization difference: 0.9605 at R=100
against 0.9655, 0.9975 at R=500 against 0.9985. That the factor is the same at
both scales — 4.5x at 10^5, 4.4x at 10^6 — is what a constant-factor fix to a
linear scan should look like; a cache effect would differ between them.

Scan is still 996 permille of query time after the change, against 998 permille
before, so the cost model's conclusion stays refuted whether the scan is fast or
slow.

![Recall against p50 latency for five engines on SIFT1M at N=100,000, and
index size on disk.](figures/engine_comparison_sift.png)

### The standings

| engine | recall@10 | p50 | SECTOR relative |
|---|---:|---:|---|
| usearch (HNSW) | 0.9945 | 1.47 ms | 3.3x slower (was 15x) |
| **SECTOR** | 0.9895 | **4.90 ms** | — |
| LanceDB (IVF_PQ) | 0.9955 | 25.34 ms | **5.2x faster** |
| sqlite-vec (exact) | 1.0000 | 143 ms | **29x faster** |
| Qdrant local mode | 1.0000 | 243.7 ms | **50x faster** |

SECTOR now beats every baseline except the graph index while keeping the
smallest index — 14.5 MB against 53.5-85.3 MB.

The remaining 3.3x is structural rather than another constant factor waiting to
be found. SECTOR scans all `N` and generates no candidates, so a graph index
wins on latency wherever the graph can afford to exist. Below T2 it cannot,
which is the regime this engine is for.

### A correctness defect caught before it shipped

`sector_core`'s ADC table holds **inner products**, and ranking by inner product
is not ranking by L2 unless every stored vector has the same norm. On SIFT-like
data the two orderings share **2 of 10** in the top ten. Calling the fast path
naively would have destroyed recall at full speed and looked like a clean win.

`adc::build_table_l2` folds the correction in. Since `||q||^2` is constant
across the corpus for one query and cannot change an ordering, maximising
`2<x,q> - ||x||^2` is minimising L2, and both terms are per-subspace and
per-centroid:

```text
T[j][v] = 2 * <q_j, C_j[v]> - ||C_j[v]||^2
```

The scan is unchanged — `m` lookups, `m` adds, no multiplies — because the
correction is paid once per centroid at table build rather than per vector.

### Vector acceleration: measured, and it does not pay

NEON is not applicable to this loop. The inner operation is a gather with a
data-dependent index, and NEON's `tbl`/`tbx` address at most 64 bytes against a
16 KiB table. Four-wide interleaving (`scan_b8_x4`) was then tried and measured
**40.1 us against 39.0 us** — no gain, because the scalar loop already runs at
roughly 1.6 cycles per lookup-and-add. Both paths are proven bit-identical and
the variant is kept for in-order cores, where it has not yet been measured.

---

## 8. Microcontroller ports

| target | status | detail |
|---|---|---|
| ESP32 c2, c3, c5, c6, c61, h2 | builds | 834 KB - 1.14 MB firmware, RISC-V |
| ESP32, s2, s3 | not built | `espup` Xtensa toolchain cannot install here |
| RP2040 (Pico) | builds | 128,328 B, `thumbv6m-none-eabi` |

Capacity from the profile arithmetic, T0 at D=128, m=16, b=8 — 56 KiB resident,
144 B/vector stored:

| part | resident / SRAM | vectors in flash |
|---|---|---:|
| RP2040 Pico (2 MB) | 56 / 264 KiB | 14,108 |
| ESP32-C3 (4 MB) | 56 / 400 KiB | 28,672 |
| ESP32-S3 (8 MB) | 56 / 512 KiB | 57,799 |
| ESP32-S2 (16 MB) | 56 / 520 KiB | 116,053 |

### The no-multiply claim was false on Cortex-M0+

Extending `asm-check` to `thumbv6m-none-eabi` found a real defect. `score_b8`'s
`table[j * centroids + c]` compiles to a **shift** on RV32IMC, where the
optimiser can prove `centroids` is a power of two — and to **four `muls`** on
Cortex-M0+, inside the per-vector path, on a core with a weak multiplier and no
hardware divide. The claim was weakest exactly where it was most load-bearing.

Two fixes moved the problem instead of removing it, and both are recorded:
`chunks_exact().nth(j)` computes an offset and put two multiplies on RISC-V;
`chunks_exact().zip()` left them in LLVM's unrolled remainder. Splitting the
slice per row keeps the stride in a pointer the compiler never reconstructs, and
both targets come out clean — verified in the **linked Pico firmware**, not only
in the probe.

That form costs 25% on out-of-order cores, so both scorers ship and `scan`
selects by target width. A test asserts they agree on every input; if they could
disagree, host and device would be running different engines.

### First device run: ESP32-C3 on Wokwi

The emulator is unreachable from the build sandbox — `wokwi-cli` 0.26.1's
WebSocket client does not honour `HTTP(S)_PROXY`, and the Pi's network does not
resolve the domain — but the user ran the same binary from their own machine.

**Every budget prediction held exactly on emulated silicon**: 4,194,304 B flash,
144 B/vector stored, a 16,384 B ADC table, a 32,768 B codebook, 800 B of heap,
and 28,672 vectors of capacity — six for six at zero tolerance.
`evictions_total=6540` confirms the bounded heap did real work; zero would have
meant the threshold test rejected everything and the timing was meaningless.

**The four-wide scan appeared to pay on an in-order core** — later retracted, see below:

| core | `scan_b8` | `scan_b8_x4` | winner |
|---|---:|---:|---|
| Cortex-A72, out-of-order | 39.0 us | 40.1 us | scalar |
| RV32IMC, in-order (C3) | 31.51 ms | 25.02 ms | four-wide, 20.6% — *retracted, see below* |

`scan_b8_auto` now selects by target width, with a test asserting it is a
dispatch and nothing more.

**No absolute timing can be derived from this emulator, and none is quoted.**
A third run at a doubled CPU clock (`cpu_clock_hz` 80 MHz → 160 MHz, same
firmware) returned **byte-identical** timings — 31,514 µs, 25,020 µs, 152 µs,
396 µs, unchanged to 0.002%. Had the time source counted emulated device cycles
every figure would have halved. It reports wall-clock time of the emulation host.

This retracts three earlier analyses in this project, each more carefully
calibrated than the last: a 10x inflation factor inferred at an assumed 160 MHz,
a 5x factor recomputed at a measured 80 MHz, and a 15.8-cycles-per-lookup figure
derived from them. All were `ns_per_instruction × clock`, and the clock never
entered the measurement. **On-device query latency now joins joules-per-query as
a precondition needing physical hardware.** The calibration loops could not
detect this, because they used the same clock; what detected it was changing an
input that should have moved the output.

**A second run refined this.** The ratio between two loops looked safe, but the
S3 gave **1.082x** where the C3 gave **1.260x** for the same source. An ESP32-S2
run then supplied the missing control: S2 and S3 are the same instruction set at
the same emulated clock, with host speeds differing by exactly **2.000x** (500 vs
1,000 ns for an identical 8-instruction calibration loop). Normalising each by its
*own* calibration, with nothing fitted:

| platform | scalar | four-wide |
|---|---:|---:|
| S2 | 220.1 | 201.6 |
| S3 | 220.1 | 203.4 |

Instruction-equivalents per vector — agreeing to 0.0% and 0.9% across a 2x host
difference. **The calibration loop is a valid host-speed normaliser**, and the
C3-vs-Xtensa gap is not noise: it is a real instruction-count difference between
the two ISAs (four-wide is 5.75 instructions per code on RV32IMC, 6.25 on
Xtensa), which the emulator measured faithfully.

**What it still cannot measure is silicon.** Wokwi charges a flat cost per
instruction, modelling an IPC-1 machine with no cache misses and no load-use
stalls — and those stalls are the entire mechanism by which independent
accumulator chains help. So the ratio is a faithful measurement of *instruction
count*, obtainable more cheaply from the disassembly, and no evidence about
speed-up on hardware. Absolute wall-clock time remains unusable: the
per-instruction cost belongs to the emulation host.

What survives is only what does not depend on the timebase at all:

- **All six budget predictions** — byte counts, exact.
- **Functional verification** of the flash path: 16 sectors erased, 256 pages
  programmed, 100 candidates read back and scored, non-zero checksum.
- **Static instruction counts** off the shipped binaries: four-wide is 28% fewer
  instructions per code on RV32IMC and 22% fewer on Xtensa. This is what
  `scan_b8_auto` now rests on, and the doc comment says so — on a core retiring
  near one instruction per cycle, fewer instructions is faster, but the size of
  any real speed-up is unmeasured.
- **Nine-chip portability**: all nine ESP32 variants and the RP2040 build, and
  the scoring loop contains no multiply on RV32IMC, Xtensa, or Cortex-M0+.

`marks=0`: the firmware calls `scan` directly, so `query()`'s phase marks never
execute. On-device phase instrumentation remains untested.

---

## 9. GIST1M feasibility at D=960

The report claims **dimension, not subspace count, decides whether 8-bit codes
are affordable**. The codebook is `2^b · D` bytes — there is no `m` in that
expression — and the enumeration confirms it exactly.

| D | codebook at b=8 | b=8 fits T0 |
|---:|---:|---|
| 512 | 131,072 B | yes |
| 768 | 196,608 B | **no** |
| 960 | 245,760 B | **no** |

### Measured on GIST1M

D=960, N=100,000, 200 queries, shipped ground truth, two-stage recall@10:

| config | R=10 | R=50 | R=100 | R=500 | R=1000 |
|---|---:|---:|---:|---:|---:|
| T0, m=60, b=4 | 0.1870 | 0.4430 | 0.5790 | 0.8710 | 0.9425 |
| T0, m=120, b=6 | 0.4500 | 0.8510 | **0.9400** | 0.9975 | 1.0000 |

Two corrections to the report come out of this.

**The boundary recall figure was too pessimistic.** The report records 0.243 at
D=768, b=4, R=100. Measured here: **0.579 at D=960** — more than double, at a
*higher* dimension. Whatever produced 0.243 was not this pipeline on this data,
and the figure is superseded rather than reconciled.

**b=6 changes the practical conclusion.** The report's position — that a
GIST-class corpus "either reduces dimension before indexing or moves to T1" —
follows from treating b=4 as the only alternative to b=8. It is not. b=6 fits
T0 at m=120 with a 61,440 B codebook and reaches 0.940 at R=100, which is usable
on the tier the report says cannot host this dimension.

**Depth substitutes for code width, but not freely.** Going from R=100 to
R=1000 recovers +0.3635 at b=4 and only +0.0600 at b=6 — at b=6 there is little
left to recover. Depth is the cheaper axis, since it costs scan time only
through rerank, but at b=4 the depth required is 10x the operating point.

Budget claims held exactly at both configurations, zero tolerance.

![Recall against candidate depth at three dimension and code-width points, and
codebook size against dimension with T0's RAM budget marked.](figures/gist_dimension_recall.png)


### Against the baselines at D=960

Same corpus, same queries, exact ground truth recomputed for the subset:

| engine | recall@10 | p50 | index |
|---|---:|---:|---:|
| usearch (HNSW) | 0.9840 | 7.28 ms | 555.0 MB |
| **SECTOR (PQ, b=6, m=120)** | **0.9400** | **32.37 ms** | **105.1 MB** |
| LanceDB (IVF_PQ) | 0.7240 | 46.52 ms | 388.2 MB |
| sqlite-vec (exact) | 1.0000 | 919 ms | 388.0 MB |
| Qdrant local mode (exact) | 1.0000 | 1579 ms | — |

**LanceDB's IVF_PQ collapses at this dimension and SECTOR does not.** Both are
PQ methods, so the difference is not PQ-versus-graph:

| | D=128 | D=960 | change |
|---|---:|---:|---:|
| LanceDB | 0.9955 | 0.7240 | **-0.2715** |
| SECTOR | 0.9895 | 0.9400 | -0.0495 |

Against LanceDB, SECTOR is **+0.216 recall, 1.44x faster and 3.7x smaller** — it
wins on every axis at this dimension. Against the graph index it is 4.4x slower
at 0.044 lower recall, with an index 5.3x smaller.

SECTOR's 105.1 MB is dominated by the rerank copy (96 MB of int8 originals),
charged to flash rather than RAM; codes and codebook alone are 9.1 MB. Scan is
985 permille of query time here, the same shape as at D=128.

One accounting caveat, stated rather than quietly corrected: the Pi's binary
predates the `payload_bytes` fix, so its printed budget rows show 120 B/vector
where the correct value at m=120, b=6 is 90. Recall and latency are unaffected;
the index figure above uses the corrected value.

**The boundary is D=735.** From D=736 a b=8 codebook plus the stack reserve
cannot fit T0's 192 KiB at any `m`. Over the full grid at D=960: 32 of 48
configurations fit T0 and **b=8 fits in none of them**; all 48 fit T1, 16 at
b=8. Enumerated through `Profile`'s own `const fn` arithmetic rather than a
restatement, so a change to the budget rules cannot leave the analysis stale.

---

## 10. What is measured, what is not, and what was refuted

### Closed

- **P2 — recall on real embeddings.** Full SIFT1M, N=1,000,000, shipped ground
  truth: recall@10 0.9605 at R=100 and 0.9975 at R=500. Better than the
  synthetic corpus predicted, not worse.
- **Budgets.** Every prediction held at zero tolerance at both 10^5 and 10^6,
  and at both GIST configurations.
- **Fault tolerance.** Four channels, all monotone, no cliffs.
- **Comparison.** Four engines on identical corpora at two dimensions.
- **The no-multiply property**, now on two instruction sets rather than one.

### Not closed

- **P1 — joules per query.** Needs a physical board with a shunt and a scope.
- **On-device query latency.** Newly added to this list. Wokwi runs the real
  image and the real instructions but its time source reports the emulation
  host's wall clock: doubling the emulated CPU clock left every timing figure
  byte-identical. Three progressively more careful calibrations were built on the
  assumption that it measured device time, and all three are retracted. This
  needs the same physical board P1 does.
  The Pi cannot supply it: it is not an MCU, its only relevant sensor is an
  undervoltage detector rather than a current sensor, and it draws orders of
  magnitude more power than the target class. The energy model's *inputs*
  (cycles and bytes per phase) are measured; only the two platform constants are
  missing.
- **Wokwi emulation.** The harness is complete; the CLI cannot reach the
  service from either available host. Recorded as blocked rather than skipped —
  the firmware was never put in front of the emulator.
- **Xtensa chips** (esp32, s2, s3). `espup` cannot install in this environment.
  Build-unverified, not build-failed.
- **T1 at D=960, b=8.** Compute-bound at roughly 64x the b=4 training cost; not
  needed for either GIST conclusion.

### Refuted by measurement

1. **The cost model's dominance conclusion.** Table build and rerank were
   claimed to dominate; measured, scan is 985-996 permille of query time at
   every scale and dimension tested. Both of the model's *structural* claims
   hold — table build is independent of `N`, scan is linear in it — but the
   conclusion drawn from them is wrong by three orders of magnitude, because it
   was stated without the corpus size that decides it.
2. **The no-multiply property on Cortex-M0+.** False until this pass, on
   precisely the class of part the property exists for.
3. **The boundary recall figure.** 0.243 at D=768, b=4 is superseded by 0.579
   measured at D=960 — more than double, at a higher dimension.
4. **That a GIST-class corpus needs T1 or dimension reduction.** b=6 fits T0
   and reaches 0.940 at R=100.
5. **This project's own 21x latency claim**, twice: first corrected to a
   measured 15x, then shown to be mostly a benchmark artifact, and finally
   measured at 3.3x after the harness was routed through the engine.

### Where SECTOR loses

Against a graph index with RAM to spare, on latency, by 3.3x at D=128 and 4.4x
at D=960. That is structural: SECTOR scans all `N` and generates no candidates,
so a graph wins wherever a graph can afford to exist. What it buys is an index
3.7 to 5.9x smaller, a working set that fits a microcontroller, and — at high
dimension — recall that a comparable PQ index does not hold.
