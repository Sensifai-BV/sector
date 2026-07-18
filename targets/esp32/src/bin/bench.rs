//! On-device benchmark: latency per phase, budgets, and fault behaviour.
//!
//! Runs the engine crates unchanged. The corpus is generated on-device from a
//! fixed seed rather than embedded, because a real corpus would not fit in the
//! image and the point here is the *scan*, whose cost depends on the number of
//! codes and their width, not on what the codes mean.
//!
//! Three stages, measuring different things:
//!
//! | stage | reads from | what it bounds |
//! |---|---|---|
//! | scan | SRAM | the `m`-lookup inner loop, no I/O |
//! | rerank | NOR flash | per-candidate flash latency at `R` records |
//! | write | NOR flash | erase and program cost of installing a corpus |
//!
//! The scan stage alone is not a device measurement of a *query*: T0 keeps the
//! 128 B rerank record in flash, not RAM, so a query pays `R` flash reads that a
//! RAM-only benchmark never sees. That was the gap this file had — it reported a
//! scan and called it a benchmark.
//!
//! The mock corpus is written to a real flash region and read back through
//! `EspFlash`, so the rerank figure includes whatever the part actually charges
//! for a read. It is generated on-device from a fixed seed rather than embedded:
//! a real corpus would not fit in the image, and scan cost depends on the number
//! of codes and their width rather than on what the codes mean.
//!
//! Output is one line per measurement, parseable by the Wokwi harness. Numbers
//! carry their configuration so a line lifted into a table cannot lose it.

#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::main;
use sector_core::heap::Heap;
use sector_core::scan;
use sector_esp32::CycleInstrument;
use sector_hal::{Instrument, NorFlash};

/// T0 profile: D=128, m=16, b=8.
const M: usize = 16;
const CENTROIDS: usize = 256;

/// Corpus size. Chosen to fit SRAM on the smallest part in the matrix while
/// staying large enough that per-vector cost dominates loop overhead.
const N: usize = 2000;

/// Candidate depth. The report's operating point.
const R: usize = 100;

/// Bytes per int8 rerank record: one per dimension at D=128.
const RERANK_BYTES: usize = 128;

/// Flash region for the mock rerank corpus.
///
/// Placed after the 1 MiB mark to stay clear of the bootloader, partition table
/// and the application image itself. Writing over the running image would fault
/// in a way that looks like a corpus bug.
const CORPUS_BASE: u32 = 0x0020_0000;

/// Vectors written to flash. Bounded by test time, not by capacity: at 128 B a
/// record, 512 vectors is 64 KiB, which is 16 erase sectors.
const FLASH_VECTORS: usize = 512;

/// Deterministic byte stream, so every chip scans an identical corpus and the
/// comparison across chips is of the silicon rather than of the data.
fn fill(buf: &mut [u8], seed: u32) {
    let mut x = seed;
    for slot in buf.iter_mut() {
        // xorshift32: no multiplies, so corpus generation cannot contaminate a
        // measurement of a scan that is meant to contain none.
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *slot = (x >> 24) as u8;
    }
}

#[main]
fn main() -> ! {
    // Run the part at its maximum clock, not the HAL default.
    //
    // The v2 run reported `cpu_clock_hz=80000000` on a C3 whose maximum is
    // 160 MHz: `Config::default()` leaves half the available performance unused,
    // and every latency figure measured under it overstates the cost by 2x. A
    // benchmark should measure the part as a deployment would configure it. The
    // reported clock is what makes this checkable rather than assumed.
    let peripherals =
        esp_hal::init(esp_hal::Config::default().with_cpu_clock(esp_hal::clock::CpuClock::max()));

    esp_println::println!("sector-bench-device v4");
    // The clock decides the cycles-per-instruction reading, and therefore every
    // projected figure. Assuming 160 MHz where the part booted at 80 would halve
    // the inferred emulator inflation and double every real-silicon estimate, so
    // it is reported rather than assumed.
    esp_println::println!("cpu_clock_hz={}", sector_esp32::cpu_clock_hz());
    esp_println::println!("config m={} b=8 centroids={} n={} r={}", M, CENTROIDS, N, R);

    // Flash capacity decides how many vectors this board could actually hold,
    // which is the number the capacity claim is about.
    let capacity = sector_esp32::EspFlash::detect_capacity(&peripherals.FLASH);
    // Stored bytes per vector: m*b/8 payload plus the int8 rerank record. At
    // b=8 the payload is m bytes; the general form is written out so a future
    // b<8 build does not silently report double, which is the accounting bug
    // that was found on the host side.
    let per_vector = M * 8 / 8 + 128;
    let codebook = CENTROIDS * 128;
    let holds = capacity.saturating_sub(codebook as u32 * 2) as usize / per_vector;
    esp_println::println!("flash_bytes={}", capacity);
    esp_println::println!("bytes_per_vector={}", per_vector);
    esp_println::println!("capacity_vectors={}", holds);

    // Resident working set, from the same const-fn arithmetic the host uses.
    let table_bytes = M * CENTROIDS * 4;
    let heap_bytes = R * (4 + 4);
    esp_println::println!("resident_table_bytes={}", table_bytes);
    esp_println::println!("resident_heap_bytes={}", heap_bytes);
    esp_println::println!("resident_codebook_bytes={}", codebook);

    static mut CODES: [u8; N * M] = [0; N * M];
    static mut TABLE: [i32; M * CENTROIDS] = [0; M * CENTROIDS];

    // Single-threaded firmware with no interrupt touching these buffers; the
    // references are taken once and never aliased.
    let codes = unsafe { &mut *core::ptr::addr_of_mut!(CODES) };
    let table = unsafe { &mut *core::ptr::addr_of_mut!(TABLE) };

    fill(codes, 0x5EC7_0201);
    let mut t = 0x1234_5678u32;
    for slot in table.iter_mut() {
        t ^= t << 13;
        t ^= t >> 17;
        t ^= t << 5;
        // Signed spread so the threshold test and evictions both exercise.
        *slot = (t >> 16) as i32 - 32768;
    }

    let inst = CycleInstrument::new();

    // Calibrate the timebase before trusting any absolute figure.
    //
    // The first Wokwi run reported 15,757 ns/vector, which is ~22x more than
    // the emitted instruction count allows at this clock: `probe_score_b8` is
    // 20 instructions for a whole 16-code record, so a scan cannot cost 158
    // cycles per lookup-and-add on an in-order core. Either the emulator's
    // microsecond timer does not track emulated instruction rate, or something
    // on the device path is genuinely slow. A fixed-work loop of known
    // instruction count separates the two: if this reports far more time than
    // its instructions can account for, the timebase is the problem and every
    // absolute figure below is scaled by the same factor. Ratios survive
    // either way.
    let cal_start = inst.cycles();
    let mut acc = 1u32;
    const CAL_ITERS: u32 = 100_000;
    for _ in 0..CAL_ITERS {
        // xorshift32: three shifts and three xors, no multiply, no memory.
        acc ^= acc << 13;
        acc ^= acc >> 17;
        acc ^= acc << 5;
    }
    let cal_us = inst.cycles() - cal_start;
    esp_println::println!("calib_iters={}", CAL_ITERS);
    esp_println::println!("calib_us={}", cal_us);
    // ~6 arithmetic ops per iteration, one cycle each on this core.
    esp_println::println!("calib_ns_per_iter={}", cal_us * 1000 / CAL_ITERS as u64);
    esp_println::println!("calib_sink={}", acc);

    // Second calibration, memory-bound: a strided walk over the code array with
    // one load and one add per step. The register-only loop above cannot tell
    // whether the emulator's slowdown is uniform across instruction types, and
    // every calibrated figure assumes it is. If these two disagree, the scan --
    // which is load-dominated -- must be scaled by THIS factor, not that one.
    let cal2_start = inst.cycles();
    let mut msum = 0u32;
    const CAL2_ITERS: u32 = 100_000;
    {
        let mut idx = 0usize;
        for _ in 0..CAL2_ITERS {
            msum = msum.wrapping_add(codes[idx] as u32);
            // 64-byte stride: crosses cache lines the way a payload scan does.
            idx = (idx + 64) % codes.len();
        }
    }
    let cal2_us = inst.cycles() - cal2_start;
    esp_println::println!("calib2_iters={}", CAL2_ITERS);
    esp_println::println!("calib2_us={}", cal2_us);
    esp_println::println!("calib2_ns_per_iter={}", cal2_us * 1000 / CAL2_ITERS as u64);
    esp_println::println!("calib2_sink={}", msum);

    let mut scores = [0i32; R];
    let mut ids = [0u32; R];

    // Warm pass: the first scan pays for cache fills that later ones do not,
    // and reporting that as the query cost would overstate it.
    {
        let mut heap = Heap::new(&mut scores, &mut ids, R).unwrap();
        scan::scan_b8(codes, 0, M, table, CENTROIDS, &mut heap);
    }

    const REPS: u32 = 20;
    let start = inst.cycles();
    let mut evicted = 0u32;
    for _ in 0..REPS {
        let mut heap = Heap::new(&mut scores, &mut ids, R).unwrap();
        let stats = scan::scan_b8(codes, 0, M, table, CENTROIDS, &mut heap);
        evicted = evicted.saturating_add(stats.evicted);
    }
    let elapsed = inst.cycles() - start;

    let per_query_us = elapsed / REPS as u64;
    esp_println::println!("scan_us_per_query={}", per_query_us);
    esp_println::println!("scan_ns_per_vector={}", per_query_us * 1000 / N as u64);
    esp_println::println!("evictions_total={}", evicted);

    // The four-wide variant exists for in-order cores, which is what this part
    // is. Measured on Wokwi ESP32-C3: 31.51 ms scalar against 25.02 ms
    // four-wide, a 20.6% saving, against no gain on the out-of-order Cortex-A72
    // where it was first tried. `scan::scan_b8_auto` now selects this shape on
    // 32-bit targets; both are still timed here so a future core can be checked
    // rather than assumed.
    let start = inst.cycles();
    for _ in 0..REPS {
        let mut heap = Heap::new(&mut scores, &mut ids, R).unwrap();
        scan::scan_b8_x4(codes, 0, M, table, CENTROIDS, &mut heap);
    }
    let elapsed_x4 = inst.cycles() - start;
    esp_println::println!("scan_x4_us_per_query={}", elapsed_x4 / REPS as u64);

    // ---------------------------------------------------------------------
    // Stage 2: install a mock corpus in NOR and rerank out of it.
    //
    // This is the part a RAM-only scan cannot report. T0 holds the 128 B int8
    // rerank record in flash, so a query pays R flash reads; that latency is a
    // property of the part, and it is what decides whether the two-stage design
    // fits a latency budget on this silicon.
    // ---------------------------------------------------------------------
    let region_bytes = (FLASH_VECTORS * RERANK_BYTES) as u32;

    // The corpus offset is reported so a log names the region that was written.
    //
    // There is no runtime check that it clears the application image: the
    // symbols available here (`_stack_start` and friends) are RAM addresses on
    // this family and cannot bound a flash image, and reading the partition
    // table would be a second source of truth for a layout the build already
    // fixes. The build script warns if the offset leaves under 1 MiB of
    // headroom, which is the check that can be made honestly.
    esp_println::println!("corpus_base=0x{:08x}", CORPUS_BASE);
    let Some(mut nor) = sector_esp32::EspFlash::new(CORPUS_BASE, region_bytes, peripherals.FLASH)
    else {
        esp_println::println!("FAIL corpus region does not fit this board");
        loop {
            core::hint::spin_loop();
        }
    };
    let sector_bytes = nor.sector_size();
    let page_bytes = nor.page_size();
    esp_println::println!("corpus_bytes={}", region_bytes);
    esp_println::println!("corpus_vectors={}", FLASH_VECTORS);

    // Erase, timed. Erase is the operation that consumes endurance, so its cost
    // is reported per sector rather than amortised into a write throughput.
    let sectors = region_bytes.div_ceil(sector_bytes as u32);
    let t = inst.cycles();
    for i in 0..sectors {
        if let Err(e) = nor.erase(i * sector_bytes as u32) {
            esp_println::println!("FAIL erase: {:?}", e);
            loop {
                core::hint::spin_loop();
            }
        }
    }
    let erase_us = inst.cycles() - t;
    // WARNING on these three figures. Under emulation a 4 KiB NOR sector erase
    // measured 28 us; real NOR takes 25-45 ms, because erase is charge pumping
    // rather than computation and no emulator speedup makes it ~900x faster.
    // Wokwi does not model program/erase latency. The lines are emitted because
    // completing them proves the write path works, and are tagged so a number
    // lifted into a table carries the caveat with it.
    esp_println::println!("flash_timing_model=emulated_not_physical");
    esp_println::println!("corpus_erase_us_total={}", erase_us);
    esp_println::println!("corpus_erase_us_per_sector={}", erase_us / sectors as u64);

    // Program, timed. Page-sized runs, which is the only legal granularity.
    let mut page = [0u8; 256];
    let pages = region_bytes as usize / page_bytes;
    let t = inst.cycles();
    for pi in 0..pages {
        let addr = (pi * page_bytes) as u32;
        fill(&mut page[..page_bytes], 0x9E37_0000 ^ pi as u32);
        if let Err(e) = nor.program(addr, &page[..page_bytes]) {
            esp_println::println!("FAIL program: {:?}", e);
            loop {
                core::hint::spin_loop();
            }
        }
    }
    let prog_us = inst.cycles() - t;
    esp_println::println!("corpus_program_us_total={}", prog_us);
    esp_println::println!("corpus_program_us_per_page={}", prog_us / pages as u64);
    esp_println::println!(
        "corpus_write_bytes_per_ms={}",
        (region_bytes as u64 * 1000) / (erase_us + prog_us).max(1)
    );

    // Rerank R candidates out of flash, timed. The CRCs are computed over the
    // bytes just read rather than stored: this stage measures read latency and
    // scoring, and a stored-CRC mismatch would abort the measurement for a
    // reason unrelated to timing. Corruption behaviour is measure_flash's job.
    let mut record = [0u8; RERANK_BYTES];
    let query: [i8; RERANK_BYTES] = core::array::from_fn(|i| (i as i8).wrapping_mul(3));
    let mut checksum = 0i64;
    let t = inst.cycles();
    for c in 0..R {
        // Stride by a prime so consecutive candidates land in different sectors,
        // matching a real candidate list rather than a sequential sweep.
        let id = ((c * 37) % FLASH_VECTORS) as u32;
        let addr = id * RERANK_BYTES as u32;
        if let Err(e) = nor.read(addr, &mut record) {
            esp_println::println!("FAIL read: {:?}", e);
            loop {
                core::hint::spin_loop();
            }
        }
        checksum += sector_core::rerank::exact_score(&query, &record) as i64;
    }
    let rerank_us = inst.cycles() - t;
    esp_println::println!("rerank_r={}", R);
    esp_println::println!("rerank_us_total={}", rerank_us);
    esp_println::println!("rerank_us_per_candidate={}", rerank_us / R as u64);
    esp_println::println!("rerank_bytes_read={}", R * RERANK_BYTES);
    // Non-zero proves records were read back rather than zeros, which is the
    // functional result this stage delivers on an emulator that does not model
    // flash timing.
    esp_println::println!("rerank_checksum={}", checksum);

    // Read latency against request size, so the per-candidate figure above can
    // be separated into a fixed cost and a per-byte cost. A 128 B record that
    // costs the same as 512 B means the fixed cost dominates, and the rerank
    // record could be widened for free.
    for len in [128usize, 256, 512, 1024] {
        let mut buf = [0u8; 1024];
        let reads = 32u32;
        let t = inst.cycles();
        for i in 0..reads {
            let addr = (i * 1024) % (region_bytes - 1024);
            let _ = nor.read(addr, &mut buf[..len]);
        }
        let us = inst.cycles() - t;
        esp_println::println!("read_us_per_{}b={}", len, us / reads as u64);
    }

    esp_println::println!("marks={}", inst.marks());
    esp_println::println!("DEVICE_BENCH_DONE");

    loop {
        core::hint::spin_loop();
    }
}
