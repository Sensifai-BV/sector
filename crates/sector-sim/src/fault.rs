//! Fault injection: bit flips, block corruption, torn writes, bad sectors.
//!
//! Four channels, each independently injectable, so a claim about one is not
//! validated by another's remedy.
//!
//! # Fault models
//!
//! Torn writes are consistently wrong, not noisily wrong: a partially written
//! sector holds a valid prefix followed by erased bytes. That is why parity
//! cannot repair one and an atomic version switch can, and a simulator
//! injecting noise here would validate the wrong remedy.
//!
//! Bad sectors are sector-correlated, not byte-independent. Independent block
//! failure is assumed by the allocation analysis and is false for flash; the
//! simulator demonstrates the gap rather than concealing it.
//!
//! Injection is deterministic given a seed, including which bytes are affected.
//! Comparing recall between a clean and a corrupted build requires the
//! corruption to be identical across runs.

use crate::sim_flash::SimFlash;
use sector_hal::ERASED_BYTE;

/// Deterministic PRNG.
///
/// Injection must be reproducible given a seed, including which bytes are
/// affected: comparing recall between a clean and a corrupted build requires
/// the corruption to be identical across runs.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. Prints as part of any failure so a run is
    /// reproducible from its output alone.
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Next 64 bits (xorshift64*).
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..bound`, or 0 when `bound` is 0.
    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}

/// What an injection did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Injected {
    /// Individual bits flipped.
    pub bits: u32,
    /// Whole blocks overwritten.
    pub blocks: u32,
    /// Writes truncated mid-sector.
    pub torn: u32,
    /// Sectors lost as a correlated burst.
    pub sectors: u32,
}

/// Flip `count` single bits uniformly within `range`.
pub fn flip_bits(
    flash: &mut SimFlash,
    range: core::ops::Range<usize>,
    count: u32,
    rng: &mut Rng,
) -> Injected {
    let span = range.end.saturating_sub(range.start);
    let mut injected = Injected::default();
    for _ in 0..count {
        let offset = range.start + rng.below(span);
        let bit = rng.below(8);
        if let Some(byte) = flash.bytes_mut().get_mut(offset) {
            *byte ^= 1 << bit;
            injected.bits += 1;
        }
    }
    injected
}

/// Overwrite whole blocks with pseudo-random bytes.
///
/// Distinct from bit flips because a block-scale event and a bit-scale event
/// have different consequences: the first removes every vector the block held,
/// the second displaces one.
pub fn corrupt_blocks(
    flash: &mut SimFlash,
    base: usize,
    block_bytes: usize,
    blocks: &[usize],
    rng: &mut Rng,
) -> Injected {
    let mut injected = Injected::default();
    for &b in blocks {
        let start = base + b * block_bytes;
        let mut fill = [0u8; 8];
        for slot in fill.iter_mut() {
            *slot = (rng.next_u64() & 0xFF) as u8;
        }
        if let Some(dst) = flash.bytes_mut().get_mut(start..start + block_bytes) {
            for (i, byte) in dst.iter_mut().enumerate() {
                *byte = fill[i % fill.len()];
            }
            injected.blocks += 1;
        }
    }
    injected
}

/// Truncate a write: a valid prefix followed by erased bytes.
///
/// Torn writes are *consistently* wrong, not noisily wrong. That is why parity
/// cannot repair one and an atomic version switch can, and a simulator
/// injecting noise here would validate the wrong remedy.
pub fn tear_write(flash: &mut SimFlash, start: usize, len: usize, written: usize) -> Injected {
    let mut injected = Injected::default();
    let tail = start + written.min(len);
    if let Some(dst) = flash.bytes_mut().get_mut(tail..start + len) {
        dst.fill(ERASED_BYTE);
        injected.torn += 1;
    }
    injected
}

/// Lose a whole erase sector as a correlated burst.
///
/// Flash fails in sector-correlated bursts. Independent block failure is what
/// the allocation analysis assumes and is false for this medium; the simulator
/// demonstrates the gap rather than concealing it.
pub fn kill_sector(flash: &mut SimFlash, sector_bytes: usize, sector: usize) -> Injected {
    let start = sector * sector_bytes;
    let mut injected = Injected::default();
    if let Some(dst) = flash.bytes_mut().get_mut(start..start + sector_bytes) {
        dst.fill(0x00);
        injected.sectors += 1;
    }
    injected
}

/// Blocks a sector failure takes out, given the layout.
///
/// The quantity that makes the independence assumption checkable: a policy
/// derived under independent block failure is exposed to this many
/// simultaneous losses.
pub const fn blocks_per_sector(sector_bytes: usize, block_bytes: usize) -> usize {
    sector_bytes / block_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim_flash::Config;
    use sector_codec::crc::{crc32, verify};

    fn flash() -> SimFlash {
        SimFlash::new(Config {
            capacity: 32 * 1024,
            page_bytes: 256,
            sector_bytes: 4096,
            endurance: 100_000,
            window: Some(32 * 1024),
        })
    }

    #[test]
    fn injection_is_reproducible_from_its_seed() {
        // Comparing recall between builds requires identical corruption, so
        // the same seed must produce byte-identical damage.
        let mut a = flash();
        let mut b = flash();
        a.install(0, &[0x5A; 1024]);
        b.install(0, &[0x5A; 1024]);

        flip_bits(&mut a, 0..1024, 20, &mut Rng::new(42));
        flip_bits(&mut b, 0..1024, 20, &mut Rng::new(42));
        assert_eq!(a.bytes()[..1024], b.bytes()[..1024]);

        // A different seed gives different damage.
        let mut c = flash();
        c.install(0, &[0x5A; 1024]);
        flip_bits(&mut c, 0..1024, 20, &mut Rng::new(43));
        assert_ne!(a.bytes()[..1024], c.bytes()[..1024]);
    }

    #[test]
    fn bit_flips_stay_inside_their_range() {
        let mut f = flash();
        f.install(0, &[0x00; 2048]);
        flip_bits(&mut f, 512..1024, 50, &mut Rng::new(7));
        assert!(f.bytes()[..512].iter().all(|b| *b == 0));
        assert!(f.bytes()[1024..2048].iter().all(|b| *b == 0));
        assert!(f.bytes()[512..1024].iter().any(|b| *b != 0));
    }

    #[test]
    fn a_corrupted_block_fails_its_crc() {
        let mut f = flash();
        let data = [0xA5u8; 512];
        f.install(0, &data);
        let crc = crc32(&data);
        assert!(verify(&f.bytes()[..512], crc));

        corrupt_blocks(&mut f, 0, 512, &[0], &mut Rng::new(1));
        assert!(!verify(&f.bytes()[..512], crc));
    }

    #[test]
    fn a_torn_write_is_consistently_wrong_not_noisily_wrong() {
        // The distinction the whole manifest design turns on. A torn sector
        // holds a valid prefix and an erased tail; every byte is a value the
        // medium can legitimately hold, so no code can tell it is incomplete.
        let mut f = flash();
        let data: [u8; 512] = core::array::from_fn(|i| (i % 251) as u8);
        f.install(0, &data);

        tear_write(&mut f, 0, 512, 100);
        assert_eq!(&f.bytes()[..100], &data[..100], "the prefix is intact");
        assert!(
            f.bytes()[100..512].iter().all(|b| *b == ERASED_BYTE),
            "the tail is erased, not noise"
        );

        // A CRC still detects it, which is why the CRC is a precondition of
        // the erasure arithmetic rather than an economy measure.
        assert!(!verify(&f.bytes()[..512], crc32(&data)));
    }

    #[test]
    fn a_sector_failure_takes_every_block_it_holds() {
        // Independent block failure is assumed by the allocation analysis and
        // is false for flash. Eight 512 B blocks share one 4 KiB sector.
        let mut f = flash();
        let data = [0x33u8; 4096];
        f.install(0, &data);
        let crcs: [u32; 8] = core::array::from_fn(|b| crc32(&data[b * 512..(b + 1) * 512]));

        kill_sector(&mut f, 4096, 0);
        let survivors = (0..8)
            .filter(|b| verify(&f.bytes()[b * 512..(b + 1) * 512], crcs[*b]))
            .count();
        assert_eq!(
            survivors, 0,
            "a sector burst is not eight independent events"
        );
        assert_eq!(blocks_per_sector(4096, 512), 8);
    }

    #[test]
    fn a_sector_failure_spares_its_neighbours() {
        let mut f = flash();
        f.install(0, &[0x77; 8192]);
        kill_sector(&mut f, 4096, 0);
        assert!(f.bytes()[..4096].iter().all(|b| *b == 0));
        assert!(f.bytes()[4096..8192].iter().all(|b| *b == 0x77));
    }

    #[test]
    fn the_four_channels_are_independently_injectable() {
        // A claim about one channel must not be validated by another's remedy,
        // so each is separately addressable and separately counted.
        let mut f = flash();
        f.install(0, &[0x11; 16384]);
        let mut rng = Rng::new(99);

        let a = flip_bits(&mut f, 0..512, 3, &mut rng);
        let b = corrupt_blocks(&mut f, 4096, 512, &[0, 1], &mut rng);
        let c = tear_write(&mut f, 8192, 512, 64);
        let d = kill_sector(&mut f, 4096, 3);

        assert_eq!((a.bits, a.blocks, a.torn, a.sectors), (3, 0, 0, 0));
        assert_eq!((b.bits, b.blocks, b.torn, b.sectors), (0, 2, 0, 0));
        assert_eq!((c.bits, c.blocks, c.torn, c.sectors), (0, 0, 1, 0));
        assert_eq!((d.bits, d.blocks, d.torn, d.sectors), (0, 0, 0, 1));
    }

    #[test]
    fn the_rng_never_degenerates_on_a_zero_seed() {
        let mut r = Rng::new(0);
        let first = r.next_u64();
        assert_ne!(first, 0);
        assert_ne!(r.next_u64(), first);
    }
}
