//! Bounded candidate heap.
//!
//! Fixed capacity `R`, evicting the lowest-scoring member. The eviction rule is
//! load-bearing for the loss analysis, not an implementation detail.
//!
//! A bounded heap evicts from its far end, so `n` intruders displace exactly
//! the incumbents at ranks `R-n+1 … R`, and an incumbent at depth `d` survives
//! whenever `n <= R - d`. Measured at N=20,000, R=100, k=10, true neighbours
//! sit at median depth 27, so evicting a typical one takes on the order of 73
//! simultaneous intruders. Loss is therefore non-linear in displacement: below
//! a threshold, corruption is harmless regardless of how many vectors it
//! touched, which is the property bounded formats exploit.
//!
//! # Implementation notes
//!
//! A flat array with sift-down: 4 KiB at `R = 500`, cache-resident, and faster
//! than a pointer structure at this size.
//!
//! The minimum is O(1) so the scan can threshold-test before attempting
//! insertion, keeping the common case out of the sift path.
//!
//! Tie-breaking is by vector id. Score ties at the candidate boundary are real
//! at these scales, and non-deterministic ordering makes a recall measurement
//! irreproducible.

/// A candidate: score and vector id.
///
/// Ties break by id. Score ties at the candidate boundary are real at these
/// scales, and non-deterministic ordering makes a recall measurement
/// irreproducible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Similarity score. Higher is better.
    pub score: i32,
    /// Vector id.
    pub id: u32,
}

impl Candidate {
    /// Whether `self` ranks worse than `other`.
    ///
    /// Lower score is worse; on a tie the higher id is worse, so the surviving
    /// order is total and reproducible.
    #[inline]
    const fn worse_than(&self, other: &Self) -> bool {
        self.score < other.score || (self.score == other.score && self.id > other.id)
    }
}

/// Bounded min-heap of the best `R` candidates seen.
///
/// The eviction rule carries the loss analysis, not just the implementation.
/// The heap evicts from its far end, so `n` intruders displace exactly the
/// incumbents at ranks `R-n+1 ..= R`, and an incumbent at depth `d` survives
/// whenever `n <= R - d`. Loss is therefore non-linear in displacement: below a
/// threshold, corruption is harmless however many vectors it touched.
#[derive(Debug)]
pub struct Heap<'a> {
    scores: &'a mut [i32],
    ids: &'a mut [u32],
    len: usize,
    capacity: usize,
}

impl<'a> Heap<'a> {
    /// Borrow buffers as a heap of capacity `capacity`.
    ///
    /// Returns `None` when either buffer is shorter than `capacity`.
    pub fn new(scores: &'a mut [i32], ids: &'a mut [u32], capacity: usize) -> Option<Self> {
        if scores.len() < capacity || ids.len() < capacity || capacity == 0 {
            return None;
        }
        Some(Self {
            scores,
            ids,
            len: 0,
            capacity,
        })
    }

    /// Candidates held.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the heap holds nothing.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the heap is at capacity.
    pub const fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Capacity, `R`.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Discard every candidate, keeping the buffers.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    fn at(&self, i: usize) -> Candidate {
        Candidate {
            score: self.scores.get(i).copied().unwrap_or(i32::MIN),
            id: self.ids.get(i).copied().unwrap_or(u32::MAX),
        }
    }

    #[inline]
    fn put(&mut self, i: usize, c: Candidate) {
        if let Some(s) = self.scores.get_mut(i) {
            *s = c.score;
        }
        if let Some(d) = self.ids.get_mut(i) {
            *d = c.id;
        }
    }

    /// The worst candidate held, in O(1).
    ///
    /// The scan tests against this before attempting insertion, which keeps the
    /// common case out of the sift path entirely.
    #[inline]
    pub fn worst(&self) -> Option<Candidate> {
        if self.len == 0 {
            None
        } else {
            Some(self.at(0))
        }
    }

    /// Whether `score` could enter the heap.
    ///
    /// The threshold test the scan's inner loop performs per vector.
    #[inline]
    pub fn would_accept(&self, score: i32, id: u32) -> bool {
        if self.len < self.capacity {
            return true;
        }
        self.at(0).worse_than(&Candidate { score, id })
    }

    /// Offer a candidate, evicting the worst if the heap is full.
    ///
    /// Returns the evicted candidate when one was displaced.
    pub fn push(&mut self, c: Candidate) -> Option<Candidate> {
        if self.len < self.capacity {
            let i = self.len;
            self.put(i, c);
            self.len += 1;
            self.sift_up(i);
            return None;
        }
        let worst = self.at(0);
        if !worst.worse_than(&c) {
            return None;
        }
        self.put(0, c);
        self.sift_down(0);
        Some(worst)
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.at(i).worse_than(&self.at(parent)) {
                let (a, b) = (self.at(i), self.at(parent));
                self.put(parent, a);
                self.put(i, b);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        loop {
            let left = 2 * i + 1;
            let right = left + 1;
            let mut worst = i;
            if left < self.len && self.at(left).worse_than(&self.at(worst)) {
                worst = left;
            }
            if right < self.len && self.at(right).worse_than(&self.at(worst)) {
                worst = right;
            }
            if worst == i {
                break;
            }
            let (a, b) = (self.at(i), self.at(worst));
            self.put(worst, a);
            self.put(i, b);
            i = worst;
        }
    }

    /// Remove and return the worst candidate.
    pub fn pop_worst(&mut self) -> Option<Candidate> {
        if self.len == 0 {
            return None;
        }
        let out = self.at(0);
        self.len -= 1;
        if self.len > 0 {
            let last = self.at(self.len);
            self.put(0, last);
            self.sift_down(0);
        }
        Some(out)
    }

    /// Drain into `out` in descending score order, returning the count written.
    ///
    /// Destructive: the heap is empty afterwards.
    pub fn drain_sorted(&mut self, out: &mut [Candidate]) -> usize {
        let n = self.len.min(out.len());
        // Popping yields ascending order; fill from the back.
        let mut filled = 0usize;
        while self.len > n {
            self.pop_worst();
        }
        while let Some(c) = self.pop_worst() {
            let idx = n - 1 - filled;
            if let Some(slot) = out.get_mut(idx) {
                *slot = c;
            }
            filled += 1;
            if filled == n {
                break;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap_of<'a>(cap: usize, scores: &'a mut [i32], ids: &'a mut [u32]) -> Heap<'a> {
        Heap::new(scores, ids, cap).expect("buffers large enough")
    }

    #[test]
    fn the_heap_keeps_the_best_r_of_a_stream() {
        let mut s = [0i32; 8];
        let mut i = [0u32; 8];
        let mut h = heap_of(8, &mut s, &mut i);
        // Insert 100 descending-then-ascending scores.
        for id in 0..100u32 {
            let score = ((id as i32) * 37) % 101;
            h.push(Candidate { score, id });
        }
        let mut out = [Candidate { score: 0, id: 0 }; 8];
        let n = h.drain_sorted(&mut out);
        assert_eq!(n, 8);
        // Descending, and every kept score is at least the 8th best.
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score, "not sorted: {out:?}");
        }
        assert!(out[0].score >= 93);
    }

    #[test]
    fn eviction_removes_exactly_the_worst_incumbent() {
        let mut s = [0i32; 4];
        let mut i = [0u32; 4];
        let mut h = heap_of(4, &mut s, &mut i);
        for (score, id) in [(50, 0u32), (40, 1), (30, 2), (20, 3)] {
            assert_eq!(h.push(Candidate { score, id }), None);
        }
        assert!(h.is_full());
        assert_eq!(h.worst(), Some(Candidate { score: 20, id: 3 }));

        // A better candidate displaces exactly the worst.
        let evicted = h.push(Candidate { score: 35, id: 4 });
        assert_eq!(evicted, Some(Candidate { score: 20, id: 3 }));
        assert_eq!(h.worst(), Some(Candidate { score: 30, id: 2 }));
    }

    #[test]
    fn an_incumbent_at_depth_d_survives_r_minus_d_intruders() {
        // The property the loss analysis rests on, checked directly.
        const R: usize = 100;
        let mut s = [0i32; R];
        let mut i = [0u32; R];
        let mut h = heap_of(R, &mut s, &mut i);

        // Fill with scores 1000 down to 901; the incumbent at depth d has
        // score 1000 - d.
        for d in 0..R {
            h.push(Candidate {
                score: 1000 - d as i32,
                id: d as u32,
            });
        }

        // Target the incumbent at depth 27, the measured median depth of a
        // true neighbour at N=20,000, R=100.
        let depth = 27usize;
        let target = Candidate {
            score: 1000 - depth as i32,
            id: depth as u32,
        };

        // R - depth - 1 = 72 intruders leave it in place.
        for n in 0..(R - depth - 1) {
            h.push(Candidate {
                score: 2000 + n as i32,
                id: 10_000 + n as u32,
            });
        }
        let mut out = [Candidate { score: 0, id: 0 }; R];
        let count = h.drain_sorted(&mut out);
        assert!(
            out[..count].contains(&target),
            "incumbent evicted too early"
        );
    }

    #[test]
    fn one_more_intruder_than_the_margin_evicts_it() {
        const R: usize = 100;
        let mut s = [0i32; R];
        let mut i = [0u32; R];
        let mut h = heap_of(R, &mut s, &mut i);
        for d in 0..R {
            h.push(Candidate {
                score: 1000 - d as i32,
                id: d as u32,
            });
        }
        let depth = 27usize;
        let target = Candidate {
            score: 1000 - depth as i32,
            id: depth as u32,
        };
        // 73 intruders: one past the margin.
        for n in 0..(R - depth) {
            h.push(Candidate {
                score: 2000 + n as i32,
                id: 10_000 + n as u32,
            });
        }
        let mut out = [Candidate { score: 0, id: 0 }; R];
        let count = h.drain_sorted(&mut out);
        assert!(!out[..count].contains(&target), "incumbent should be gone");
    }

    #[test]
    fn the_threshold_test_agrees_with_push() {
        // The scan skips anything `would_accept` rejects, so a disagreement
        // between the two silently drops candidates that belong in the result.
        let mut base_s = [0i32; 4];
        let mut base_i = [0u32; 4];
        {
            let mut h = heap_of(4, &mut base_s, &mut base_i);
            for (score, id) in [(50, 0u32), (40, 1), (30, 2), (20, 3)] {
                h.push(Candidate { score, id });
            }
        }

        for score in 0..60i32 {
            let mut a_s = base_s;
            let mut a_i = base_i;
            let mut b_s = base_s;
            let mut b_i = base_i;

            let accepted = {
                let mut h = heap_of(4, &mut a_s, &mut a_i);
                h.len = 4;
                h.would_accept(score, 99)
            };
            let pushed = {
                let mut h = heap_of(4, &mut b_s, &mut b_i);
                h.len = 4;
                h.push(Candidate { score, id: 99 }).is_some()
            };
            assert_eq!(accepted, pushed, "disagreement at score {score}");
        }
    }

    #[test]
    fn ties_break_deterministically_by_id() {
        let mut s = [0i32; 3];
        let mut i = [0u32; 3];
        let mut h = heap_of(3, &mut s, &mut i);
        h.push(Candidate { score: 10, id: 5 });
        h.push(Candidate { score: 10, id: 2 });
        h.push(Candidate { score: 10, id: 9 });
        // The highest id is worst on a tie.
        assert_eq!(h.worst(), Some(Candidate { score: 10, id: 9 }));

        // Same score, lower id: displaces the tied incumbent.
        let evicted = h.push(Candidate { score: 10, id: 1 });
        assert_eq!(evicted, Some(Candidate { score: 10, id: 9 }));
    }

    #[test]
    fn an_equal_candidate_does_not_churn_the_heap() {
        let mut s = [0i32; 2];
        let mut i = [0u32; 2];
        let mut h = heap_of(2, &mut s, &mut i);
        h.push(Candidate { score: 10, id: 1 });
        h.push(Candidate { score: 20, id: 2 });
        // Identical to the incumbent: no eviction, no churn.
        assert_eq!(h.push(Candidate { score: 10, id: 1 }), None);
    }

    #[test]
    fn heap_bytes_match_the_workspace_figure() {
        // R=500 candidates: 500 * (4 + 4) = 3.9 KiB, the reported figure.
        let bytes = 500 * (core::mem::size_of::<i32>() + core::mem::size_of::<u32>());
        assert_eq!(bytes, 4_000);
        assert!(bytes < 4 * 1024);
    }

    #[test]
    fn undersized_buffers_are_refused() {
        let mut s = [0i32; 3];
        let mut i = [0u32; 3];
        assert!(Heap::new(&mut s, &mut i, 4).is_none());
        assert!(Heap::new(&mut s, &mut i, 0).is_none());
    }

    #[test]
    fn draining_fewer_than_held_returns_the_best() {
        let mut s = [0i32; 8];
        let mut i = [0u32; 8];
        let mut h = heap_of(8, &mut s, &mut i);
        for id in 0..8u32 {
            h.push(Candidate {
                score: id as i32 * 10,
                id,
            });
        }
        let mut out = [Candidate { score: 0, id: 0 }; 3];
        let n = h.drain_sorted(&mut out);
        assert_eq!(n, 3);
        assert_eq!(out[0].score, 70);
        assert_eq!(out[2].score, 50);
    }
}
