//! Adapters that feed the engine from a host backend.
//!
//! [`sector_core::query::query`] reaches payload codes through
//! [`PayloadSource`] and rerank records through [`RerankSource`]. These types
//! implement both over any [`NorFlash`], so one engine call serves the buffered
//! and mapped paths and a measured difference between them is a property of the
//! storage rather than of two diverging implementations.
//!
//! # Buffered, by construction
//!
//! Both adapters own a buffer and copy into it. That is the honest shape for a
//! `read`-based backend, and it is also what the mapped backend uses here: the
//! zero-copy path needs to hand the engine a borrow whose lifetime outlives the
//! call, which a `&mut self` method cannot express without threading the map's
//! lifetime through the trait. The mapped backend's advantage on this path is
//! that its `read` is a `memcpy` from the page cache rather than a syscall — it
//! is measured as such, and is not claimed to be zero-copy.
//!
//! # Run size and why it is a whole number of blocks
//!
//! The payload adapter serves one buffer-full at a time, sized to a whole number
//! of 512 B blocks. A run ending mid-block would hand the scan a partial record
//! and shift every subsequent vector id, so the buffer is block-aligned and the
//! last run is short rather than misaligned.

use sector_core::query::PayloadSource;
use sector_core::rerank::{Guarded, RerankSource};
use sector_format::BLOCK_BYTES;
use sector_hal::NorFlash;

use crate::volume::Geometry;

/// Payload codes, read in multi-block chunks and served one block at a time.
///
/// # Why a run is one block and not the whole chunk
///
/// The scan treats a run as contiguous records: it steps `payload_bytes` at a
/// time from `first_id`. Within a block that holds, but a block may end in slack
/// — `512 % payload_bytes` bytes that belong to no record — so records are *not*
/// contiguous across a block boundary in general. At T0's 16 B payload the slack
/// is zero and a multi-block run would happen to work; at `m = 120, b = 6` a
/// 90 B record leaves 62 B of slack per block, and a run spanning two blocks
/// would feed the scan 62 bytes of padding as if they were four vectors and
/// shift every subsequent id.
///
/// So the read is amortised over `blocks_per_read` blocks and the *serving* is
/// per block. The syscall count is the chunk's, the correctness is the block's.
pub struct PayloadReader<'f, F: NorFlash> {
    flash: &'f mut F,
    base: u32,
    /// Payload bytes per vector.
    record_bytes: usize,
    /// Vectors per block, excluding slack.
    per_block: usize,
    /// Blocks in the region that hold vectors.
    blocks: usize,
    /// Vectors in the volume.
    n: usize,
    /// Next block to serve.
    next_block: usize,
    buf: Vec<u8>,
    /// Blocks currently in `buf`, as a half-open range.
    held: (usize, usize),
    /// Ids absent from storage, as a half-open range.
    ///
    /// Empty on a built-only volume. See the gap discussion in `next_run`.
    gap: (usize, usize),
}

impl<'f, F: NorFlash> PayloadReader<'f, F> {
    /// A reader over `geometry`'s payload region, reading `blocks_per_read`
    /// blocks per syscall.
    ///
    /// 64 blocks is 32 KiB: it amortises the read over roughly 2,000 vectors at
    /// T0's 16 B payload, and stays within the ADC table's own order of
    /// magnitude so the two together do not evict each other on the smallest
    /// tier.
    pub fn new(flash: &'f mut F, base: u32, geometry: &Geometry, blocks_per_read: usize) -> Self {
        Self::with_gap(flash, base, geometry, blocks_per_read, (0, 0))
    }

    /// As [`Self::new`], skipping ids in `gap`.
    ///
    /// `gap` is `(built_n, appended_from)` from the manifest: the ids an append
    /// left addressable and unwritten.
    pub fn with_gap(
        flash: &'f mut F,
        base: u32,
        geometry: &Geometry,
        blocks_per_read: usize,
        gap: (usize, usize),
    ) -> Self {
        let chunk = blocks_per_read.max(1);
        Self {
            gap,
            flash,
            base,
            record_bytes: geometry.payload_bytes,
            per_block: geometry.payload.vectors_per_block().max(1),
            blocks: geometry.payload.blocks(),
            n: geometry.n,
            next_block: 0,
            buf: vec![0u8; chunk * BLOCK_BYTES],
            held: (0, 0),
        }
    }

    /// Blocks the buffer can hold.
    fn chunk_blocks(&self) -> usize {
        (self.buf.len() / BLOCK_BYTES).max(1)
    }
}

impl<F: NorFlash> PayloadSource for PayloadReader<'_, F> {
    type Error = F::Error;

    fn next_run(&mut self) -> Result<Option<(&[u8], u32)>, Self::Error> {
        let block = self.next_block;
        if block >= self.blocks {
            return Ok(None);
        }
        let first_id = block * self.per_block;
        if first_id >= self.n {
            return Ok(None);
        }

        // Skip blocks that lie entirely within the gap: they are erased, so reading
        // them would be a syscall for bytes no id claims.
        if self.gap.1 > self.gap.0
            && first_id >= self.gap.0
            && first_id + self.per_block <= self.gap.1
        {
            self.next_block += 1;
            return self.next_run();
        }

        // Refill when the wanted block is outside what the buffer holds.
        if block < self.held.0 || block >= self.held.1 {
            let want = self.chunk_blocks().min(self.blocks - block);
            let bytes = want * BLOCK_BYTES;
            self.flash.read(
                self.base + (block * BLOCK_BYTES) as u32,
                &mut self.buf[..bytes],
            )?;
            self.held = (block, block + want);
        }

        // The final block is partly empty when `n` is not a whole multiple of
        // the per-block count; serve only the records that exist.
        let mut take = self.per_block.min(self.n - first_id);

        // On an appended volume the built corpus's last block also holds the start
        // of the gap: those record slots are padding the builder wrote, not
        // vectors. Their codes are inside a block whose CRC is valid, so nothing
        // downstream can tell them apart — stage one would score them, they would
        // occupy candidate slots, and stage two would drop them when their erased
        // rerank block failed its CRC. Correct answers, at a recall cost paid
        // silently.
        //
        // Truncating the run here is the whole fix, and it costs nothing: the ids
        // never enter the heap.
        if self.gap.1 > self.gap.0 && first_id < self.gap.0 && first_id + take > self.gap.0 {
            take = self.gap.0 - first_id;
        }

        let at = (block - self.held.0) * BLOCK_BYTES;
        self.next_block += 1;

        // A block wholly inside the gap has nothing to serve. Skipping rather than
        // returning an empty run keeps the scan's run count equal to the number of
        // blocks that hold data.
        if take == 0 {
            return self.next_run();
        }

        Ok(Some((
            &self.buf[at..at + take * self.record_bytes],
            first_id as u32,
        )))
    }

    fn rewind(&mut self) {
        self.next_block = 0;
        // The buffer's contents stay valid, so a rewound scan re-reads only when
        // it leaves what is held. This is what makes repeated queries on a small
        // volume cost no syscalls after the first pass.
    }
}

/// Rerank records, fetched one candidate at a time.
///
/// This is the access pattern the tier comparison is about: `R` random reads of
/// a record each, which raw NOR services as loads from a mapped window and
/// managed storage services through its translation layer at block granularity.
///
/// The buffer holds whole blocks because that is the extent a CRC covers. At
/// every shipped profile a record is smaller than a block, so one fetch reads
/// 512 B to score 128 — a 4x read amplification that is a property of the
/// format's CRC granularity and is visible in [`crate::file::AccessStats`].
pub struct RerankReader<'f, F: NorFlash> {
    flash: &'f mut F,
    base: u32,
    crc_base: u32,
    layout: sector_format::rerank_blk::RerankLayout,
    /// Whole blocks containing the current record.
    blocks: Vec<u8>,
    /// CRCs of those blocks.
    crcs: Vec<u32>,
    /// Offset of the record within `blocks`.
    offset: usize,
    /// Record length.
    len: usize,
}

impl<'f, F: NorFlash> RerankReader<'f, F> {
    /// A reader over `geometry`'s rerank region.
    pub fn new(flash: &'f mut F, base: u32, crc_base: u32, geometry: &Geometry) -> Self {
        let layout = geometry.rerank;
        let span = layout.blocks_per_record().max(1);
        // A record can straddle two blocks even when it is smaller than one, so
        // the buffer holds one more block than the record spans.
        let capacity = (span + 1) * BLOCK_BYTES;
        Self {
            flash,
            base,
            crc_base,
            layout,
            blocks: vec![0u8; capacity],
            crcs: vec![0u32; span + 1],
            offset: 0,
            len: geometry.rerank_bytes,
        }
    }
}

impl<F: NorFlash> RerankSource for RerankReader<'_, F> {
    type Error = F::Error;

    fn record(&mut self, id: u32) -> Result<Option<Guarded<'_>>, Self::Error> {
        let Some(offset) = self.layout.offset_of(id as usize) else {
            return Ok(None);
        };
        let Some((first, last)) = self.layout.blocks_of(id as usize) else {
            return Ok(None);
        };
        let count = last - first;

        let bytes = (count * BLOCK_BYTES).min(self.blocks.len());
        self.flash.read(
            self.base + (first * BLOCK_BYTES) as u32,
            &mut self.blocks[..bytes],
        )?;

        // The CRC array is a parallel u32-per-block region, little-endian.
        let mut raw = [0u8; 4];
        for i in 0..count {
            self.flash
                .read(self.crc_base + ((first + i) * 4) as u32, &mut raw)?;
            if let Some(slot) = self.crcs.get_mut(i) {
                *slot = u32::from_le_bytes(raw);
            }
        }

        self.offset = offset - first * BLOCK_BYTES;
        Ok(Some(Guarded {
            blocks: &self.blocks[..bytes],
            offset: self.offset,
            len: self.len,
            crcs: &self.crcs[..count],
        }))
    }

    fn block_bytes(&self) -> usize {
        BLOCK_BYTES
    }
}
