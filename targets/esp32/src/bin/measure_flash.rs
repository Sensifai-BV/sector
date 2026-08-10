//! NOR timings and fault behaviour, measured on the part.
//!
//! Two things the host cannot supply. The **timings** — page program and sector
//! erase — are what the lifetime and append models are built on, and they are
//! datasheet numbers until something measures them. The **fault behaviour** is
//! whether a corrupted block is detected and dropped rather than silently
//! scored, which the simulator asserts and only real flash can confirm.
//!
//! This writes to flash, so it erases the region it uses first and stays inside
//! a region declared here rather than probing for free space.

#![no_std]
#![no_main]

esp_bootloader_esp_idf::esp_app_desc!();

use esp_backtrace as _;
use esp_hal::main;
use sector_codec::crc::{crc32, verify};
use sector_esp32::{CycleInstrument, EspFlash, PAGE_BYTES, SECTOR_BYTES};
use sector_hal::{Instrument, NorFlash};

/// Region used for the measurement. Placed well past the firmware image; the
/// constructor refuses it if the board's real capacity is smaller, which is a
/// per-board property rather than a per-chip one.
const REGION_BASE: u32 = 0x0020_0000;
const REGION_LEN: u32 = 64 * 1024;

#[main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default());
    esp_println::println!("sector-measure-flash v2");

    let capacity = EspFlash::detect_capacity(&peripherals.FLASH);
    esp_println::println!("flash_bytes={}", capacity);

    let mut flash = match EspFlash::new(REGION_BASE, REGION_LEN, peripherals.FLASH) {
        Some(f) => f,
        None => {
            esp_println::println!("FAIL region does not fit this board");
            loop {
                core::hint::spin_loop();
            }
        }
    };

    let inst = CycleInstrument::new();

    // Erase timing. One sector at a time, since erase is the operation the
    // endurance budget is spent in and the lifetime model counts.
    let mut erase_total = 0u64;
    const ERASES: u32 = 4;
    for i in 0..ERASES {
        let addr = i * SECTOR_BYTES as u32;
        let t = inst.cycles();
        match flash.erase(addr) {
            Ok(()) => erase_total += inst.cycles() - t,
            Err(e) => {
                esp_println::println!("FAIL erase at {}: {:?}", addr, e);
                loop {
                    core::hint::spin_loop();
                }
            }
        }
    }
    esp_println::println!("erase_us_per_sector={}", erase_total / ERASES as u64);

    // Program timing, one page at a time on freshly erased flash.
    let page = [0xA5u8; PAGE_BYTES];
    let mut prog_total = 0u64;
    const PAGES: u32 = 16;
    for i in 0..PAGES {
        let addr = i * PAGE_BYTES as u32;
        let t = inst.cycles();
        match flash.program(addr, &page) {
            Ok(()) => prog_total += inst.cycles() - t,
            Err(e) => {
                esp_println::println!("FAIL program at {}: {:?}", addr, e);
                loop {
                    core::hint::spin_loop();
                }
            }
        }
    }
    esp_println::println!("program_us_per_page={}", prog_total / PAGES as u64);

    // Read timing at block granularity, the unit the scan streams in.
    let mut buf = [0u8; 512];
    let t = inst.cycles();
    const READS: u32 = 64;
    for i in 0..READS {
        let _ = flash.read((i * 512) % (PAGES * PAGE_BYTES as u32), &mut buf);
    }
    let read_us = inst.cycles() - t;
    esp_println::println!("read_us_per_512b={}", read_us / READS as u64);

    // Round-trip: what was programmed must read back byte for byte. A backend
    // that silently drops or reorders bytes would show up as recall loss with
    // no other symptom.
    let mut check = [0u8; PAGE_BYTES];
    let ok = flash.read(0, &mut check).is_ok() && check == page;
    esp_println::println!("roundtrip_exact={}", if ok { 1 } else { 0 });

    // Detection: a CRC over a block must reject a single flipped bit. This is
    // the property the whole repair path rests on, checked against real flash
    // contents rather than a simulated buffer.
    let good = crc32(&check);
    let mut damaged = check;
    damaged[7] ^= 0x01;
    esp_println::println!(
        "crc_detects_single_bit_flip={}",
        if verify(&check, good) && !verify(&damaged, good) {
            1
        } else {
            0
        }
    );

    // What a second program over a written page actually does.
    //
    // The earlier version of this test printed a single bit for `after[0] ==
    // (0xA5 & 0x5A)`, i.e. `== 0x00`, under a comment claiming the write "must
    // be refused rather than ANDed". Three different outcomes all produce a
    // zero there — refused (page still 0xA5), accepted outright (0x5A, an
    // emulator not modelling NOR physics), or anything else — so the check
    // could not fail for the reason it named. `EspFlash::program` validates
    // alignment and bounds only; program-once is the *format's* invariant,
    // enforced by the volume writer never re-programming, not by this backend.
    //
    // So report the observed byte and the call's verdict, and let the reader
    // decide. On real NOR the answer is 0x00; anything else says the backend or
    // the emulator does not behave as the format assumes.
    let second = [0x5Au8; PAGE_BYTES];
    let accepted = flash.program(0, &second).is_ok();
    let mut after = [0u8; PAGE_BYTES];
    let _ = flash.read(0, &mut after);
    esp_println::println!("reprogram_accepted={}", if accepted { 1 } else { 0 });
    esp_println::println!("reprogram_first_byte=0x{:02x}", after[0]);
    esp_println::println!("reprogram_expected_and=0x{:02x}", 0xA5u8 & 0x5A);
    esp_println::println!(
        "reprogram_behaviour={}",
        match after[0] {
            0x00 => "anded_like_nor",
            0xA5 => "unchanged_write_ignored",
            0x5A => "overwritten_not_nor_physics",
            _ => "unexpected",
        }
    );

    esp_println::println!("MEASURE_FLASH_DONE");
    loop {
        core::hint::spin_loop();
    }
}
