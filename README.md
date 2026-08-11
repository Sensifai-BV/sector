# SECTOR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21883870.svg)](https://doi.org/10.5281/zenodo.21883870)


SECTOR is an on-flash index format and integer-only, zero-allocation search engine designed specifically for approximate nearest-neighbour vector search on highly constrained microcontroller-class
devices. These edge devices—such as standalone wildlife cameras or industrial sensors—typically lack operating systems, floating-point units, and heaps, meaning conventional server-optimized
search architectures simply cannot run on them. To overcome strict limitations on RAM, flash storage, and energy consumption, SECTOR relies on product quantization (PQ) and fixed-point int8
arithmetic. This approach heavily compresses vector embeddings so that massive datasets, which would otherwise exceed device storage, can reside and be efficiently searched directly on the
microcontroller's limited flash memory.

The core innovation of the project is its theoretical and practical treatment of the index as a stored artifact vulnerable to corruption. Because microcontrollers often read raw NOR flash without
error-correcting code (ECC) protections, data corruption is a dominant failure mode. SECTOR identifies a critical asymmetry in compression: a corrupted payload byte damages only one vector,
whereas a corrupted codebook byte damages every vector that shares that centroid. To solve this, the engine employs a bounded max-heap to naturally evict false positives (intruders) and utilizes
zero-cost format-level defenses like bounded fixed-point storage and lossless centroid relabeling. The result is a highly robust and fast search engine that, when compared to a raw flat scan
baseline, is 5.5× faster, achieves higher recall, and can fit 6.1× more vectors into a strict 192 KiB RAM budget without wasting space on traditional uniform error correction.
  
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

|                                                | Mechanism                                 | Cost                    |
|------------------------------------------------|-------------------------------------------|-------------------------|
| Bounded formats cap worst-case damage          | int8 fixed-point codebooks, never f32     | zero bytes, zero cycles |
| Centroid labels are arbitrary — so choose them | build-time permutation, provably lossless | zero bytes, zero cycles |
| Protection belongs where criticality is        | measured per-centroid allocation          | ~0.8% of stored bytes   |

One bit flipped in an f32 codebook costs 0.246 recall — 39% of baseline. The
same bit in a bounded int8 codebook costs 0.0005.

## Configuration

Codebook size is `2^b · D` and does not depend on `m`, so 8-bit codes fit T0's
192 KiB budget at any `D ≤ 384` and are out of reach from `D = 736` up. Since
8-bit codes measure 1.35x the recall of 4-bit ones at equal payload *and* equal
dimension (0.605 against 0.4485 at `R = 100`), T0 targets `D = 128` — SIFT's
dimension, and the common output width of edge embedding models.

T0 (`D=128, m=16, b=8, int8, R=500`): 16 B/vector payload, 32 KiB codebook,
8 KiB ADC table, 51.9 KiB fixed, **8,966 vectors resident** in 192 KiB, with
128 B/vector rerank copies in NOR (~32,000 in 4 MB).

Two-stage retrieval is mandatory — single-stage PQ recall is unusable at every
configuration measured — and it is fastest on the *smallest* tier, because raw
NOR is byte-addressable and execute-in-place while a managed-NAND tier pays an
FTL random-read penalty for the same access pattern.

These figures are `const`-asserted in `sector-format::profile`. A configuration
that does not fit fails the build.

## Raspberry Pi

Every Pi ever made is covered by three static musl binaries.
maps each board to its artifact and
explains why the Zero and the Zero 2 W need different ones — they differ by ISA
generation, not by speed.

```sh
# On the Pi, from an unpacked release archive:
./sector doctor                 # what this board is, and whether this binary fits it
sudo ./install.sh --image volume.sector
systemctl enable --now sector
```

`doctor` exists because the failure it catches is quiet: a binary built for a
newer instruction set installs cleanly on an older board and then dies with
`SIGILL` at the first unsupported instruction, possibly not until a query
arrives. `install.sh` runs it first and refuses rather than installing something
that cannot work.

### The daemon

```sh
sector serve --image volume.sector --socket /run/sector/sector.sock
curl --unix-socket /run/sector/sector.sock http://localhost/info
```

| Route                       | Method | Purpose                               |
|-----------------------------|--------|---------------------------------------|
| `/health`                   | GET    | liveness, no volume access            |
| `/ready`                    | GET    | readiness — runs a real query         |
| `/info`                     | GET    | geometry, tier, board, resident bytes |
| `/stats`                    | GET    | counters since start                  |
| `/search`                   | POST   | one or more queries                   |
| `/vectors`, `/vectors/{id}` | GET    | stored records                        |

No TLS and no authentication, which is why a Unix socket is the default and
filesystem permissions are the access control. `--listen` is for a trusted
network or behind a reverse proxy.

### Adding vectors

```sh
sector build --input corpus.fvecs --out volume.sector --reserve 4096
sector append --image volume.sector --input new.fvecs --dry-run
sector append --image volume.sector --input new.fvecs
```

Insert-only: there is no delete and no update, because a validity bitmap would
add a per-candidate lookup to the scan loop the cost argument is about. `build`
is how a vector is removed. `--reserve` must be given at build time — without it
a volume is fixed at its built size. `--dry-run` reports the id gap an append
would create before writing anything.

## Benchmarks

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full benchmark report.

## Building

```sh
make check-all     # fmt, clippy, test, no_std, asm-check, Pi cross-builds, deny
make nostd         # the no_std guarantee on thumbv7em-none-eabihf
make build-t0      # ESP32-C3 firmware
make build-pi      # static binaries for all three Pi ABIs
make check-pi      # cross-compile the whole workspace, every feature, per ABI
make isa-check     # assert each ARM artifact declares the ISA baseline it claims
make test-cross    # run the suite under qemu-user (Linux; fails if qemu is absent)
make selftest      # end-to-end on this machine, no dataset needed
```

`test-cross` is deliberately outside `check-all`: it needs `qemu-user-static`, and
a target that skipped on a missing prerequisite would let `check-all` report green
without having run it. CI runs that leg on Linux, where the prerequisite is real.

## License

MIT OR Apache-2.0
