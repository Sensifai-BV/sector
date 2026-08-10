//! On-device benchmark for the Pico: scan cost per vector, and the two scan
//! shapes compared on an in-order core.
//!
//! The comparison this exists for: `scan_b8_x4` keeps four independent
//! accumulator chains and measured **no faster** on out-of-order cores, because
//! the scalar loop already saturates a load-add's throughput there. Cortex-M0+
//! is in-order with a longer load-use penalty, which is where the extra chains
//! should finally pay. If they do not pay here either, the variant has no
//! remaining justification and should be deleted.
//!
//! # Timebase
//!
//! Two things must be set up before `TimerInstrument` reads anything, and
//! neither is a default:
//!
//! - `init_clocks_and_plls` brings the core to 125 MHz. Without it the part runs
//!   from the ~6.5 MHz ring oscillator and every latency figure is ~19x too
//!   large.
//! - That call also starts watchdog tick generation, which is what drives
//!   `clk_tick` and therefore the TIMER block. Without it TIMERAWL never
//!   advances, every interval reads zero, and the benchmark reports 0 ns per
//!   vector — an absence printed as a measurement.
//!
//! The reported `timebase_ok` line exists so that failure is visible in the log
//! rather than inferred from an implausible number.
//!
//! # What the emulator can and cannot measure
//!
//! Under Wokwi the elapsed-time figures are the emulation host's, not the
//! part's: the ESP32 runs established that Wokwi's timer tracks instructions
//! retired scaled by a host cost, and that doubling the emulated clock left
//! every timing byte-identical. So the absolute microsecond figures here are not
//! device latency. `calib_ns_per_iter` is emitted for the same reason it is on
//! the ESP32 side — a fixed-work loop of known instruction count, which lets a
//! reader normalise the scan figures by host speed and recover the ratio of
//! *work* between the two scan shapes.

#![no_std]
#![no_main]

use core::fmt::Write;

use cortex_m_rt::entry;
use panic_halt as _;
// fugit comes from the HAL's own re-export: taking it as a direct dependency
// would let its version drift from the one the HAL's clock types use.
use rp2040_hal::clocks::init_clocks_and_plls;
use rp2040_hal::fugit::RateExtU32;
use rp2040_hal::gpio::Pins;
use rp2040_hal::uart::{DataBits, StopBits, UartConfig, UartPeripheral};
use rp2040_hal::watchdog::Watchdog;
use rp2040_hal::{pac, Clock, Sio};
use sector_core::heap::Heap;
use sector_core::scan;
use sector_hal::Instrument;
use sector_rp2040::TimerInstrument;

/// Second-stage bootloader. The boot ROM checksums and copies this before any
/// of the firmware runs, so it must be present and first in the image.
#[link_section = ".boot2"]
#[used]
pub static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_W25Q080;

/// Pico module crystal. Sets the PLL input and the watchdog tick divisor.
const XOSC_CRYSTAL_FREQ: u32 = 12_000_000;

/// T0 profile: D=128, m=16, b=8.
const M: usize = 16;
const CENTROIDS: usize = 256;

/// Corpus size. 2,000 vectors is 32 KB of codes, which fits comfortably in the
/// Pico's 264 KiB alongside the 16 KiB ADC table — and is large enough that
/// per-vector cost dominates loop overhead.
const N: usize = 2000;

/// Candidate depth, the report's operating point.
const R: usize = 100;

/// Deterministic byte stream. xorshift32 uses no multiplies, so generating the
/// corpus cannot contaminate a measurement of a loop that is meant to contain
/// none.
fn fill(buf: &mut [u8], seed: u32) {
    let mut x = seed;
    for slot in buf.iter_mut() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *slot = (x >> 24) as u8;
    }
}

static mut CODES: [u8; N * M] = [0; N * M];
static mut TABLE: [i32; M * CENTROIDS] = [0; M * CENTROIDS];

#[entry]
fn main() -> ! {
    let mut pac = pac::Peripherals::take().unwrap();
    let mut watchdog = Watchdog::new(pac.WATCHDOG);
    let sio = Sio::new(pac.SIO);

    // Brings the core to 125 MHz and starts watchdog tick generation. The tick
    // is what clocks the TIMER block, so this call is load-bearing for the
    // measurement and not only for speed.
    //
    // Under `--features half-clock` the PLL is reconfigured to 62.5 MHz
    // afterwards. That is the timebase discriminator: a physical counter must
    // report ~2x the ns-per-iteration for the same instruction count, whereas a
    // timer reporting emulation-host wall clock reports the same number at both
    // settings, which is what the ESP32 runs did.
    #[cfg(not(feature = "half-clock"))]
    let clocks = init_clocks_and_plls(
        XOSC_CRYSTAL_FREQ,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .ok()
    .unwrap();

    // Half-clock build: the same tree, with clk_sys at 62.5 MHz.
    //
    // 1500 MHz VCO / 6 / 4 = 62.5 MHz, against the stock /6/2 = 125 MHz. Only the
    // system PLL's post-divider differs; the watchdog tick still comes off the
    // 12 MHz crystal, so the timer's 1 MHz reference is unchanged and the two
    // builds are directly comparable.
    #[cfg(feature = "half-clock")]
    let clocks = {
        use rp2040_hal::clocks::ClocksManager;
        use rp2040_hal::pll::{setup_pll_blocking, PLLConfig};
        use rp2040_hal::xosc::setup_xosc_blocking;

        const PLL_SYS_62_5MHZ: PLLConfig = PLLConfig {
            vco_freq: rp2040_hal::fugit::HertzU32::MHz(1500),
            refdiv: 1,
            post_div1: 6,
            post_div2: 4,
        };

        let xosc = setup_xosc_blocking(pac.XOSC, XOSC_CRYSTAL_FREQ.Hz())
            .ok()
            .unwrap();
        // The TIMER block is clocked from this tick; without it TIMERAWL never
        // advances, exactly as in the default path.
        watchdog.enable_tick_generation((XOSC_CRYSTAL_FREQ / 1_000_000) as u8);
        let mut clocks = ClocksManager::new(pac.CLOCKS);
        let pll_sys = setup_pll_blocking(
            pac.PLL_SYS,
            xosc.operating_frequency(),
            PLL_SYS_62_5MHZ,
            &mut clocks,
            &mut pac.RESETS,
        )
        .ok()
        .unwrap();
        let pll_usb = setup_pll_blocking(
            pac.PLL_USB,
            xosc.operating_frequency(),
            rp2040_hal::pll::common_configs::PLL_USB_48MHZ,
            &mut clocks,
            &mut pac.RESETS,
        )
        .ok()
        .unwrap();
        clocks.init_default(&xosc, &pll_sys, &pll_usb).ok().unwrap();
        clocks
    };

    let pins = Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    // GP0/GP1 are UART0 TX/RX, the pair the Wokwi diagram wires to the serial
    // monitor. 115200 8N1 to match the ESP32 harness, so one log parser serves
    // both.
    let uart_pins = (pins.gpio0.into_function(), pins.gpio1.into_function());
    let mut uart = UartPeripheral::new(pac.UART0, uart_pins, &mut pac.RESETS)
        .enable(
            UartConfig::new(115_200_u32.Hz(), DataBits::Eight, None, StopBits::One),
            clocks.peripheral_clock.freq(),
        )
        .unwrap();

    let _ = writeln!(uart, "sector-bench-pico v1");
    let _ = writeln!(uart, "cpu_clock_hz={}", clocks.system_clock.freq().to_Hz());
    let _ = writeln!(
        uart,
        "config m={} b=8 centroids={} n={} r={}",
        M, CENTROIDS, N, R
    );

    // Resident working set, from the same arithmetic the host uses. The ADC
    // table is m * 2^b * 4 = 16,384 B; the heap is R * (4 + 4) = 800 B. These
    // are what decide whether the profile fits the part's 264 KiB at all.
    let _ = writeln!(uart, "resident_table_bytes={}", M * CENTROIDS * 4);
    let _ = writeln!(uart, "resident_heap_bytes={}", R * (4 + 4));
    let _ = writeln!(uart, "resident_codes_bytes={}", N * M);

    // Single-threaded firmware with no interrupt touching these buffers; the
    // references are taken once here and never aliased.
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

    let inst = TimerInstrument::new();

    // The timer must be seen to advance before any interval is reported. If the
    // watchdog tick were not running this reads 0 and every figure below would
    // be a zero dressed as a measurement.
    let t0 = inst.cycles();
    let mut spin = 0u32;
    while inst.cycles() == t0 && spin < 1_000_000 {
        spin = spin.wrapping_add(1);
    }
    let _ = writeln!(
        uart,
        "timebase_ok={}",
        if inst.cycles() > t0 { 1 } else { 0 }
    );

    // Fixed-work timebase calibration: xorshift32, three shifts and three xors
    // per iteration, no multiply and no memory. Its instruction count is
    // checkable in the disassembly, which is what makes the scan figures
    // normalisable by host speed under emulation.
    let cal_start = inst.cycles();
    let mut acc = 1u32;
    const CAL_ITERS: u32 = 100_000;
    for _ in 0..CAL_ITERS {
        acc ^= acc << 13;
        acc ^= acc >> 17;
        acc ^= acc << 5;
    }
    let cal_us = inst.cycles() - cal_start;
    let _ = writeln!(uart, "calib_iters={}", CAL_ITERS);
    let _ = writeln!(uart, "calib_us={}", cal_us);
    let _ = writeln!(
        uart,
        "calib_ns_per_iter={}",
        cal_us * 1000 / CAL_ITERS as u64
    );
    // Printed so the optimiser cannot discard the loop that was just timed.
    let _ = writeln!(uart, "calib_sink={}", acc);

    let mut scores = [0i32; R];
    let mut ids = [0u32; R];

    // Warm pass: the first scan pays for fills later ones do not, and reporting
    // that as the query cost would overstate it.
    {
        let mut heap = Heap::new(&mut scores, &mut ids, R).unwrap();
        scan::scan_b8(codes, 0, M, table, CENTROIDS, &mut heap);
    }

    const REPS: u32 = 10;

    let start = inst.cycles();
    let mut evicted = 0u32;
    for _ in 0..REPS {
        let mut heap = Heap::new(&mut scores, &mut ids, R).unwrap();
        let stats = scan::scan_b8(codes, 0, M, table, CENTROIDS, &mut heap);
        evicted = evicted.saturating_add(stats.evicted);
    }
    let scalar_us = (inst.cycles() - start) / REPS as u64;

    let start = inst.cycles();
    for _ in 0..REPS {
        let mut heap = Heap::new(&mut scores, &mut ids, R).unwrap();
        scan::scan_b8_x4(codes, 0, M, table, CENTROIDS, &mut heap);
    }
    let x4_us = (inst.cycles() - start) / REPS as u64;

    let _ = writeln!(uart, "scan_us_per_query={}", scalar_us);
    let _ = writeln!(uart, "scan_ns_per_vector={}", scalar_us * 1000 / N as u64);
    // Non-zero evictions prove the bounded heap displaced incumbents: with R=100
    // against 2,000 scanned vectors it must have. Zero would mean the threshold
    // test rejected everything and the latency figure is meaningless.
    let _ = writeln!(uart, "evictions_total={}", evicted);
    let _ = writeln!(uart, "scan_x4_us_per_query={}", x4_us);
    let _ = writeln!(uart, "marks={}", inst.marks());
    let _ = writeln!(uart, "PICO_BENCH_DONE");

    loop {
        cortex_m::asm::nop();
    }
}
