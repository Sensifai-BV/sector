//! Systematic Reed–Solomon over GF(2^8), erasure-only decode.
//!
//! Erasure-only is valid because the block CRC localises corruption first. With
//! locations known, RS(n, k) recovers any `n - k` lost blocks; without
//! localisation the same code recovers `floor((n - k) / 2)`. An earlier version
//! of this design assumed the former on an unlocalised channel and overstated
//! its correction capability by a factor of two.
//!
//! # Scope
//!
//! At T0/T1 codebook sizes replication costs less in total. RS applies where
//! the protected structure is large enough for its byte saving to outweigh
//! GF(2^8) decode work.
//!
//! On managed NAND the FTL already runs BCH or LDPC underneath, so RS over a
//! vector store there duplicates existing ECC. Raw NOR has no such layer.
//!
//! RS does nothing against torn writes. A partially written sector is
//! consistently wrong rather than noisily wrong; the remedy is an atomic
//! version switch.
//!
//! # Implementation notes
//!
//! Systematic Cauchy or Vandermonde construction, table-driven GF(2^8) with the
//! 512-byte log/antilog tables in rodata, operating on caller-provided block
//! buffers. Systematic form leaves the data blocks directly readable in the
//! no-failure case, with no decode step.
//!
//! Erasure patterns up to `n - k` are enumerated exhaustively rather than
//! sampled. The pattern space at RS(12,8) is 4,096.

use crate::gf;

/// Largest total block count a code may use.
pub const MAX_N: usize = 16;

/// Largest data block count a code may use.
pub const MAX_K: usize = 12;

/// Why an encode or decode was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RsError {
    /// `k` is zero, or `n` exceeds `MAX_N`, or `k` exceeds `min(n, MAX_K)`.
    Parameters {
        /// Total blocks requested.
        n: usize,
        /// Data blocks requested.
        k: usize,
    },
    /// Fewer than `k` blocks survived, so the data is not recoverable.
    ///
    /// Returned rather than approximated: a code that returns wrong data on an
    /// uncorrectable pattern is worse than one that refuses.
    TooManyErasures {
        /// Surviving blocks.
        survivors: usize,
        /// Blocks needed.
        needed: usize,
    },
    /// Block buffers differ in length.
    RaggedBlocks,
    /// A caller-provided buffer is the wrong size.
    BufferSize,
}

/// Systematic RS(n, k) over GF(2^8) with a Cauchy parity matrix.
///
/// Cauchy rather than Vandermonde because every square submatrix of a Cauchy
/// matrix is invertible by construction, which is exactly the MDS property
/// erasure decode needs: any `k` surviving blocks suffice.
#[derive(Clone, Copy, Debug)]
pub struct Rs {
    n: usize,
    k: usize,
    /// Parity coefficients, `(n-k) x k`, row-major.
    parity: [[u8; MAX_K]; MAX_N],
}

impl Rs {
    /// Construct RS(n, k).
    pub fn new(n: usize, k: usize) -> Result<Self, RsError> {
        if k == 0 || n > MAX_N || k > MAX_K || k > n {
            return Err(RsError::Parameters { n, k });
        }
        // Cauchy: P[i][j] = 1 / (x_i + y_j), with the x and y sets disjoint.
        // Field addition is XOR, so x_i ^ y_j is never zero here.
        let mut parity = [[0u8; MAX_K]; MAX_N];
        for (i, row) in parity.iter_mut().take(n - k).enumerate() {
            let x = (i + k) as u8;
            for (j, cell) in row.iter_mut().take(k).enumerate() {
                let y = j as u8;
                *cell = gf::inv(gf::add(x, y)).unwrap_or(0);
            }
        }
        Ok(Self { n, k, parity })
    }

    /// Total blocks.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Data blocks.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Parity blocks, `n - k`. The erasure count the code recovers.
    pub const fn parity_blocks(&self) -> usize {
        self.n - self.k
    }

    /// Parity bytes per data byte, in parts per million.
    pub const fn overhead_ppm(&self) -> usize {
        ((self.n - self.k) * 1_000_000) / self.k
    }

    /// Generator-matrix row for block `index`: identity above `k`, Cauchy below.
    fn row(&self, index: usize) -> [u8; MAX_K] {
        let mut out = [0u8; MAX_K];
        if index < self.k {
            if let Some(cell) = out.get_mut(index) {
                *cell = 1;
            }
        } else if let Some(src) = self.parity.get(index - self.k) {
            out = *src;
        }
        out
    }

    /// Compute parity blocks from `data`.
    ///
    /// Systematic, so the data blocks are stored unchanged and are readable
    /// with no decode step in the no-failure case.
    pub fn encode(&self, data: &[&[u8]], parity_out: &mut [&mut [u8]]) -> Result<(), RsError> {
        if data.len() != self.k || parity_out.len() != self.parity_blocks() {
            return Err(RsError::BufferSize);
        }
        let block_len = match data.first() {
            Some(b) => b.len(),
            None => return Err(RsError::BufferSize),
        };
        if data.iter().any(|b| b.len() != block_len)
            || parity_out.iter().any(|b| b.len() != block_len)
        {
            return Err(RsError::RaggedBlocks);
        }

        for (i, out) in parity_out.iter_mut().enumerate() {
            let coeffs = match self.parity.get(i) {
                Some(c) => *c,
                None => return Err(RsError::BufferSize),
            };
            for byte in 0..block_len {
                let mut acc = 0u8;
                for (j, src) in data.iter().enumerate() {
                    let c = coeffs.get(j).copied().unwrap_or(0);
                    let v = src.get(byte).copied().unwrap_or(0);
                    acc = gf::add(acc, gf::mul(c, v));
                }
                if let Some(dst) = out.get_mut(byte) {
                    *dst = acc;
                }
            }
        }
        Ok(())
    }

    /// Reconstruct the `k` data blocks from surviving blocks.
    ///
    /// `survivors` pairs each surviving block's index in `0..n` with its bytes.
    /// Erasure-only: the caller has already localised damage with a CRC, which
    /// is what makes `n - k` recoveries valid rather than `floor((n-k)/2)`.
    pub fn decode(
        &self,
        survivors: &[(usize, &[u8])],
        data_out: &mut [&mut [u8]],
    ) -> Result<(), RsError> {
        if survivors.len() < self.k {
            return Err(RsError::TooManyErasures {
                survivors: survivors.len(),
                needed: self.k,
            });
        }
        if data_out.len() != self.k {
            return Err(RsError::BufferSize);
        }
        let block_len = match survivors.first() {
            Some((_, b)) => b.len(),
            None => return Err(RsError::BufferSize),
        };
        if survivors.iter().any(|(_, b)| b.len() != block_len)
            || data_out.iter().any(|b| b.len() != block_len)
        {
            return Err(RsError::RaggedBlocks);
        }

        // Take the first k survivors and invert their generator rows.
        let mut m = [[0u8; MAX_K]; MAX_K];
        for (r, (index, _)) in survivors.iter().take(self.k).enumerate() {
            if *index >= self.n {
                return Err(RsError::BufferSize);
            }
            if let Some(row) = m.get_mut(r) {
                *row = self.row(*index);
            }
        }
        let inv = invert(&mut m, self.k).ok_or(RsError::TooManyErasures {
            survivors: survivors.len(),
            needed: self.k,
        })?;

        for byte in 0..block_len {
            for (r, out) in data_out.iter_mut().enumerate() {
                let mut acc = 0u8;
                for (c, (_, src)) in survivors.iter().take(self.k).enumerate() {
                    let coeff = inv.get(r).and_then(|row| row.get(c)).copied().unwrap_or(0);
                    let v = src.get(byte).copied().unwrap_or(0);
                    acc = gf::add(acc, gf::mul(coeff, v));
                }
                if let Some(dst) = out.get_mut(byte) {
                    *dst = acc;
                }
            }
        }
        Ok(())
    }
}

/// Invert a `k x k` matrix over GF(2^8) by Gauss-Jordan elimination.
fn invert(m: &mut [[u8; MAX_K]; MAX_K], k: usize) -> Option<[[u8; MAX_K]; MAX_K]> {
    let mut inv = [[0u8; MAX_K]; MAX_K];
    for (i, row) in inv.iter_mut().take(k).enumerate() {
        *row.get_mut(i)? = 1;
    }

    for col in 0..k {
        // Find a row at or below `col` with a non-zero pivot.
        let pivot = (col..k).find(|&r| m.get(r).and_then(|row| row.get(col)) != Some(&0))?;
        if pivot != col {
            m.swap(pivot, col);
            inv.swap(pivot, col);
        }

        let p = *m.get(col)?.get(col)?;
        let p_inv = gf::inv(p)?;
        for j in 0..k {
            let a = *m.get(col)?.get(j)?;
            *m.get_mut(col)?.get_mut(j)? = gf::mul(a, p_inv);
            let b = *inv.get(col)?.get(j)?;
            *inv.get_mut(col)?.get_mut(j)? = gf::mul(b, p_inv);
        }

        for r in 0..k {
            if r == col {
                continue;
            }
            let factor = *m.get(r)?.get(col)?;
            if factor == 0 {
                continue;
            }
            for j in 0..k {
                let a = *m.get(col)?.get(j)?;
                let cur = *m.get(r)?.get(j)?;
                *m.get_mut(r)?.get_mut(j)? = gf::add(cur, gf::mul(factor, a));

                let b = *inv.get(col)?.get(j)?;
                let cur = *inv.get(r)?.get(j)?;
                *inv.get_mut(r)?.get_mut(j)? = gf::add(cur, gf::mul(factor, b));
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 8;
    const N: usize = 12;
    const K: usize = 8;

    fn corpus() -> [[u8; BLOCK]; K] {
        core::array::from_fn(|i| core::array::from_fn(|j| (i * 31 + j * 7 + 1) as u8))
    }

    fn encoded() -> (Rs, [[u8; BLOCK]; N]) {
        let rs = Rs::new(N, K).unwrap();
        let data = corpus();
        let mut parity = [[0u8; BLOCK]; N - K];
        {
            let d: [&[u8]; K] = core::array::from_fn(|i| &data[i][..]);
            let mut p: [&mut [u8]; N - K] = {
                let (a, b) = parity.split_at_mut(2);
                let (a0, a1) = a.split_at_mut(1);
                let (b0, b1) = b.split_at_mut(1);
                [
                    &mut a0[0][..],
                    &mut a1[0][..],
                    &mut b0[0][..],
                    &mut b1[0][..],
                ]
            };
            rs.encode(&d, &mut p).unwrap();
        }
        let mut all = [[0u8; BLOCK]; N];
        all[..K].copy_from_slice(&data);
        all[K..].copy_from_slice(&parity);
        (rs, all)
    }

    /// Recover from the blocks whose bit is clear in `erased_mask`.
    fn try_recover(
        rs: &Rs,
        all: &[[u8; BLOCK]; N],
        erased_mask: u16,
    ) -> Result<[[u8; BLOCK]; K], RsError> {
        let mut survivors: [(usize, &[u8]); N] = core::array::from_fn(|i| (i, &all[i][..]));
        let mut count = 0usize;
        for (i, block) in all.iter().enumerate() {
            if erased_mask & (1 << i) == 0 {
                survivors[count] = (i, &block[..]);
                count += 1;
            }
        }
        let mut out = [[0u8; BLOCK]; K];
        {
            let mut refs: [&mut [u8]; K] = {
                let (a, b) = out.split_at_mut(4);
                let (a01, a23) = a.split_at_mut(2);
                let (a0, a1) = a01.split_at_mut(1);
                let (a2, a3) = a23.split_at_mut(1);
                let (b01, b23) = b.split_at_mut(2);
                let (b0, b1) = b01.split_at_mut(1);
                let (b2, b3) = b23.split_at_mut(1);
                [
                    &mut a0[0][..],
                    &mut a1[0][..],
                    &mut a2[0][..],
                    &mut a3[0][..],
                    &mut b0[0][..],
                    &mut b1[0][..],
                    &mut b2[0][..],
                    &mut b3[0][..],
                ]
            };
            rs.decode(&survivors[..count], &mut refs)?;
        }
        Ok(out)
    }

    #[test]
    fn systematic_encoding_leaves_data_blocks_unchanged() {
        let (_, all) = encoded();
        let data = corpus();
        for i in 0..K {
            assert_eq!(all[i], data[i], "data block {i} was modified");
        }
    }

    #[test]
    fn every_erasure_pattern_is_enumerated_not_sampled() {
        let (rs, all) = encoded();
        let data = corpus();
        let mut recovered = 0usize;
        let mut refused = 0usize;

        // All 2^12 subsets of the 12 blocks.
        for mask in 0u16..(1 << N) {
            let erasures = mask.count_ones() as usize;
            let result = try_recover(&rs, &all, mask);
            if erasures <= N - K {
                let out = result.expect("pattern within n-k must recover");
                for i in 0..K {
                    assert_eq!(out[i], data[i], "mask {mask:#06x} block {i}");
                }
                recovered += 1;
            } else {
                assert!(
                    matches!(result, Err(RsError::TooManyErasures { .. })),
                    "mask {mask:#06x} with {erasures} erasures must refuse"
                );
                refused += 1;
            }
        }
        // C(12,0..4) = 1 + 12 + 66 + 220 + 495 = 794 recoverable patterns.
        assert_eq!(recovered, 794);
        assert_eq!(recovered + refused, 1 << N);
    }

    #[test]
    fn losing_every_parity_block_still_recovers() {
        let (rs, all) = encoded();
        let data = corpus();
        let mask = 0b1111_0000_0000; // all four parity blocks
        let out = try_recover(&rs, &all, mask).unwrap();
        for i in 0..K {
            assert_eq!(out[i], data[i]);
        }
    }

    #[test]
    fn losing_four_data_blocks_recovers_from_parity() {
        let (rs, all) = encoded();
        let data = corpus();
        let out = try_recover(&rs, &all, 0b0000_0000_1111).unwrap();
        for i in 0..K {
            assert_eq!(out[i], data[i]);
        }
    }

    #[test]
    fn bad_parameters_are_refused() {
        assert!(matches!(Rs::new(12, 0), Err(RsError::Parameters { .. })));
        assert!(matches!(Rs::new(8, 12), Err(RsError::Parameters { .. })));
        assert!(matches!(Rs::new(20, 8), Err(RsError::Parameters { .. })));
        assert!(Rs::new(12, 8).is_ok());
    }

    #[test]
    fn overhead_matches_the_stated_rate() {
        // RS(12,8): 4 parity per 8 data = 50% of the protected structure.
        assert_eq!(Rs::new(12, 8).unwrap().overhead_ppm(), 500_000);
        assert_eq!(Rs::new(12, 8).unwrap().parity_blocks(), 4);
    }

    #[test]
    fn ragged_blocks_are_refused() {
        let rs = Rs::new(12, 8).unwrap();
        let data = corpus();
        let short = [0u8; BLOCK - 1];
        let mut d: [&[u8]; K] = core::array::from_fn(|i| &data[i][..]);
        d[3] = &short[..];
        let mut parity = [[0u8; BLOCK]; N - K];
        let (a, b) = parity.split_at_mut(2);
        let (a0, a1) = a.split_at_mut(1);
        let (b0, b1) = b.split_at_mut(1);
        let mut p: [&mut [u8]; N - K] = [
            &mut a0[0][..],
            &mut a1[0][..],
            &mut b0[0][..],
            &mut b1[0][..],
        ];
        assert_eq!(rs.encode(&d, &mut p), Err(RsError::RaggedBlocks));
    }
}
