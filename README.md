# SECTOR

**Criticality-aware approximate nearest-neighbour search for microcontroller-class devices.**

Rust · `no_std` · heapless · integer-only · zero allocation on the query path

---

## What this is

A vector search engine for devices with no operating system, no FPU, no heap,
and a power budget measured in joules per query. The reference target is an
ESP32-C3: 160 MHz, 390 KB RAM, 4 MB NOR flash.

It is not a smaller version of a server vector database. At this scale `N` is
10³–10⁶ so constant factors dominate asymptotics, storage is flash where writes
cost ~100x reads, and the binding budget is energy.

## The idea

Compression is mandatory at this scale — and compression introduces **shared
structure**. Every stored vector's reconstruction depends on a product
quantization codebook shared by the whole corpus. A corrupted payload byte
alters one vector; a corrupted codebook byte alters `N / 2^b` of them in
expectation. Uncompressed stores have no analogue of this, and the ANN
literature does not treat it.

Three consequences, and **two of them are free**:

| | Mechanism | Cost |
|---|---|---|
| Bounded formats cap worst-case damage | int8 fixed-point codebooks, never f32 | zero bytes, zero cycles |
| Centroid labels are arbitrary — so choose them | build-time permutation, provably lossless | zero bytes, zero cycles |
| Protection belongs where criticality is | measured per-centroid allocation | ~0.8% of stored bytes |

One bit flipped in an f32 codebook costs 0.246 recall — 39% of baseline. The
same bit in a bounded int8 codebook costs 0.0005.

## Configuration

Codebook size is `2^b · D` and does not depend on `m`, so 8-bit codes fit T0's
192 KiB budget at any `D ≤ 384` and are out of reach at `D = 768`. Since 8-bit
codes measure roughly 2.5x the recall of 4-bit ones at equal payload size, T0
targets `D = 128` — SIFT's dimension, and the common output width of edge
embedding models.

T0 (`D=128, m=16, b=8, int8, R=500`): 16 B/vector payload, 32 KiB codebook,
8 KiB ADC table, 51.9 KiB fixed, **8,966 vectors resident** in 192 KiB, with
128 B/vector rerank copies in NOR (~32,000 in 4 MB).

Two-stage retrieval is mandatory — single-stage PQ recall is unusable at every
configuration measured — and it is fastest on the *smallest* tier, because raw
NOR is byte-addressable and execute-in-place while a managed-NAND tier pays an
FTL random-read penalty for the same access pattern.

These figures are `const`-asserted in `sector-format::profile`. A configuration
that does not fit fails the build.

## Layout

```
crates/
  sector-hal/      traits: NorFlash, Xip, Clock, Instrument     no_std, zero deps
  sector-quant/    integer PQ: codebooks, ADC, rotation, labels no_std
  sector-codec/    CRC32, replication, RS(n,k) over GF(2^8)     no_std
  sector-format/   on-flash layout + tier profiles              no_std
  sector-core/     the heapless query engine                    no_std
  sector-build/    host-side index builder                      std
  sector-sim/      fault injection + claim validation           std, dev-only
  sector-cli/      build / inspect / query / falsify            std, dev-only
targets/esp32/     T0 and T1 firmware
```

## Status

Pre-implementation. Workspace scaffolded, traits and tier profiles written,
budgets compiler-enforced. Two blocking preconditions remain open: no code has
run on hardware, and every recall figure to date uses a synthetic corpus.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/TASKS.md](docs/TASKS.md).

## Building

```sh
make check-all     # fmt, clippy, test, no_std build, cargo-deny
make nostd         # the no_std guarantee on thumbv7em-none-eabihf
make build-t0      # ESP32-C3 firmware
```

## License

MIT OR Apache-2.0
