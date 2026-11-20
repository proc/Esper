//! Lock-free shared transposition table.
//!
//! Stockfish's density: 32-byte clusters of three entries, each a 64-bit data
//! word plus a 16-bit check, so 10.7 bytes per entry instead of 16. The check is
//! the key's low 16 bits XOR a 16-bit fold of the data, so a key and data word
//! torn apart by racing threads fail it (as the old full-width `key ^ data`
//! scheme did) and no locking is needed. Like Stockfish, a 16-bit check lets an
//! unrelated position match about once in 65536 probes of a full cluster; moves
//! taken from the table are validated before use.

use crate::position::Move;
use std::sync::atomic::{AtomicU16, AtomicU64, AtomicU8, Ordering::Relaxed};

pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

const ENTRIES_PER_BUCKET: usize = 3;
const GEN_MASK: u64 = 31;

#[repr(C, align(32))]
struct Bucket {
    data: [AtomicU64; ENTRIES_PER_BUCKET],
    check: [AtomicU16; ENTRIES_PER_BUCKET],
    _pad: u16,
}

const _: () = assert!(std::mem::size_of::<Bucket>() == 32);

/// The 16-bit check stored for `key` with `data`.
#[inline(always)]
fn check16(key: u64, data: u64) -> u16 {
    (key as u16) ^ (data as u16) ^ ((data >> 16) as u16) ^ ((data >> 32) as u16) ^ ((data >> 48) as u16)
}

impl Bucket {
    #[inline(always)]
    fn matches(&self, i: usize, key: u64, data: u64) -> bool {
        data != 0 && self.check[i].load(Relaxed) == check16(key, data)
    }

    #[inline(always)]
    fn write(&self, i: usize, key: u64, data: u64) {
        self.data[i].store(data, Relaxed);
        self.check[i].store(check16(key, data), Relaxed);
    }
}

#[derive(Clone, Copy)]
pub struct TtData {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: u8,
    pub pv: bool,
}

pub struct TT {
    buckets: *mut Bucket,
    len: usize,
    gen: AtomicU8,
    /// Huge-page backing on Linux; otherwise the table comes from the global allocator.
    large: Option<crate::mem::LargeAlloc>,
}

unsafe impl Send for TT {}
unsafe impl Sync for TT {}

/// Stored depths are offset so that qsearch entries (depth 0) and entries that only
/// hold a static evaluation (`DEPTH_UNSEARCHED`) stay distinct, as in Stockfish.
const DEPTH_OFFSET: i32 = 3;
/// Depth of an entry that only records the static evaluation.
pub const DEPTH_UNSEARCHED: i32 = -2;

// data layout (low to high): move:16 score:16 eval:16 depth:8 bound:2 pv:1 gen:5
#[inline(always)]
fn pack(mv: Move, score: i32, eval: i32, depth: i32, bound: u8, pv: bool, gen: u8) -> u64 {
    (mv.0 as u64)
        | ((score as i16 as u16 as u64) << 16)
        | ((eval as i16 as u16 as u64) << 32)
        | (((depth + DEPTH_OFFSET).clamp(0, 255) as u64) << 48)
        | ((bound as u64) << 56)
        | ((pv as u64) << 58)
        | ((gen as u64 & GEN_MASK) << 59)
}

#[inline(always)]
fn unpack(d: u64) -> TtData {
    TtData {
        mv: Move(d as u16),
        score: (d >> 16) as u16 as i16 as i32,
        eval: (d >> 32) as u16 as i16 as i32,
        depth: ((d >> 48) & 0xFF) as i32 - DEPTH_OFFSET,
        bound: ((d >> 56) & 3) as u8,
        pv: (d >> 58) & 1 != 0,
    }
}

impl TT {
    pub fn new(mb: usize) -> TT {
        let len = (mb.max(1) * 1024 * 1024 / std::mem::size_of::<Bucket>()).max(1);
        let layout = std::alloc::Layout::array::<Bucket>(len).unwrap();
        // On Linux use 2 MB huge pages: probes land at random, so 4 KB pages
        // would miss the TLB on almost every one. The mapping arrives zeroed.
        if crate::mem::ENABLED {
            if let Some(large) = crate::mem::LargeAlloc::new(layout.size()) {
                let buckets = large.as_ptr() as *mut Bucket;
                return TT { buckets, len, gen: AtomicU8::new(0), large: Some(large) };
            }
        }
        // SAFETY: an all-zero bit pattern is a valid Bucket (atomics holding 0).
        let buckets = unsafe { std::alloc::alloc_zeroed(layout) as *mut Bucket };
        if buckets.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        TT { buckets, len, gen: AtomicU8::new(0), large: None }
    }

    #[inline(always)]
    fn bucket(&self, key: u64) -> &Bucket {
        let idx = ((key as u128 * self.len as u128) >> 64) as usize;
        unsafe { &*self.buckets.add(idx) }
    }

    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            _mm_prefetch(self.bucket(key) as *const Bucket as *const i8, _MM_HINT_T0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let p = self.bucket(key) as *const Bucket;
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
        }
    }

    pub fn clear_range(&self, start: usize, end: usize) {
        for i in start..end.min(self.len) {
            let b = unsafe { &*self.buckets.add(i) };
            for i in 0..ENTRIES_PER_BUCKET {
                b.data[i].store(0, Relaxed);
                b.check[i].store(0, Relaxed);
            }
        }
    }

    pub fn reset_gen(&self) {
        self.gen.store(0, Relaxed);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clear(&self) {
        self.clear_range(0, self.len);
        self.reset_gen();
    }

    pub fn new_search(&self) {
        self.gen.store((self.gen.load(Relaxed) + 1) & GEN_MASK as u8, Relaxed);
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<TtData> {
        let b = self.bucket(key);
        for i in 0..ENTRIES_PER_BUCKET {
            let data = b.data[i].load(Relaxed);
            if b.matches(i, key, data) {
                return Some(unpack(data));
            }
        }
        None
    }

    #[inline]
    pub fn store(&self, key: u64, mut mv: Move, score: i32, eval: i32, depth: i32, bound: u8, pv: bool) {
        let b = self.bucket(key);
        let gen = self.gen.load(Relaxed);
        let mut victim = 0;
        let mut worst = i32::MAX;
        for i in 0..ENTRIES_PER_BUCKET {
            let data = b.data[i].load(Relaxed);
            if data == 0 {
                victim = i;
                break;
            }
            let old = unpack(data);
            let old_gen = (data >> 59) as u8;
            if b.matches(i, key, data) {
                // Stockfish keeps a deeper entry from the same search unless the new
                // result is exact; even then a new move replaces the stored one.
                let keep = bound != BOUND_EXACT && old_gen == gen && depth + 2 * pv as i32 <= old.depth - 4;
                if keep {
                    let decisive = old.score.abs() >= crate::search::MATE_BOUND && old.score.abs() < crate::search::INF;
                    // Mate scores that keep being bypassed age out slowly.
                    let new_depth = if old.depth >= 5 && old.bound != BOUND_EXACT && decisive { old.depth - 1 } else { old.depth };
                    let new_mv = if mv.is_none() { old.mv } else { mv };
                    if new_mv != old.mv || new_depth != old.depth {
                        let d = pack(new_mv, old.score, old.eval, new_depth, old.bound, old.pv, old_gen);
                        b.write(i, key, d);
                    }
                    return;
                }
                if mv.is_none() {
                    mv = old.mv;
                }
                victim = i;
                break;
            }
            let age = (gen.wrapping_sub(old_gen) & GEN_MASK as u8) as i32;
            let quality = old.depth + DEPTH_OFFSET - 8 * age;
            if quality < worst {
                worst = quality;
                victim = i;
            }
        }
        let data = pack(mv, score, eval, depth, bound, pv, gen);
        b.write(victim, key, data);
    }

    /// Lower the stored depth of `key`'s entry (Stockfish's TTWriter::penalize).
    pub fn penalize(&self, key: u64, amount: i32) {
        let b = self.bucket(key);
        for i in 0..ENTRIES_PER_BUCKET {
            let data = b.data[i].load(Relaxed);
            if b.matches(i, key, data) {
                let e = unpack(data);
                let d = pack(e.mv, e.score, e.eval, (e.depth - amount).max(-DEPTH_OFFSET), e.bound, e.pv, (data >> 59) as u8);
                b.write(i, key, d);
                return;
            }
        }
    }

    /// Permille of the table used by the current search.
    pub fn hashfull(&self) -> usize {
        let gen = self.gen.load(Relaxed);
        let n = self.len.min(1000);
        let mut used = 0;
        for i in 0..n {
            let b = unsafe { &*self.buckets.add(i) };
            for e in 0..ENTRIES_PER_BUCKET {
                let data = b.data[e].load(Relaxed);
                if data != 0 && (data >> 59) as u8 == gen {
                    used += 1;
                }
            }
        }
        used * 1000 / (n * ENTRIES_PER_BUCKET)
    }
}

impl Drop for TT {
    fn drop(&mut self) {
        if self.large.is_some() {
            return; // the mapping is released when `large` drops
        }
        let layout = std::alloc::Layout::array::<Bucket>(self.len).unwrap();
        unsafe { std::alloc::dealloc(self.buckets as *mut u8, layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_probe_and_density() {
        let tt = TT::new(1);
        assert_eq!(tt.len() * ENTRIES_PER_BUCKET, 3 * 32768);
        let key = 0x1234_5678_9abc_def0u64;
        assert!(tt.probe(key).is_none());
        tt.store(key, Move(0x0abc), 123, -45, 9, BOUND_EXACT, true);
        let e = tt.probe(key).expect("stored entry");
        assert_eq!((e.mv.0, e.score, e.eval, e.depth, e.bound, e.pv), (0x0abc, 123, -45, 9, BOUND_EXACT, true));
        // Same cluster, different check bits: no match.
        assert!(tt.probe(key ^ 1).is_none());
        // A torn pair (data rewritten without its check) is rejected.
        let b = tt.bucket(key);
        let i = (0..ENTRIES_PER_BUCKET).find(|&i| b.matches(i, key, b.data[i].load(Relaxed))).unwrap();
        b.data[i].store(pack(Move(1), 7, 7, 3, BOUND_LOWER, false, 0), Relaxed);
        assert!(tt.probe(key).is_none());
    }
}
