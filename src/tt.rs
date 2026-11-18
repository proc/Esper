//! Lock-free shared transposition table.
//!
//! Each entry is two atomic words: `key ^ data` and `data`. A torn write from a
//! racing thread simply fails the key check on probe, so no locking is needed.

use crate::position::Move;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

const ENTRIES_PER_BUCKET: usize = 4;
const GEN_MASK: u64 = 31;

#[repr(C, align(64))]
struct Bucket([AtomicU64; 2 * ENTRIES_PER_BUCKET]);

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
}

unsafe impl Send for TT {}
unsafe impl Sync for TT {}

// data layout (low to high): move:16 score:16 eval:16 depth:8 bound:2 pv:1 gen:5
#[inline(always)]
fn pack(mv: Move, score: i32, eval: i32, depth: i32, bound: u8, pv: bool, gen: u8) -> u64 {
    (mv.0 as u64)
        | ((score as i16 as u16 as u64) << 16)
        | ((eval as i16 as u16 as u64) << 32)
        | ((depth.clamp(0, 255) as u64) << 48)
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
        depth: ((d >> 48) & 0xFF) as i32,
        bound: ((d >> 56) & 3) as u8,
        pv: (d >> 58) & 1 != 0,
    }
}

impl TT {
    pub fn new(mb: usize) -> TT {
        let len = (mb.max(1) * 1024 * 1024 / std::mem::size_of::<Bucket>()).max(1);
        let layout = std::alloc::Layout::array::<Bucket>(len).unwrap();
        // SAFETY: an all-zero bit pattern is a valid Bucket (atomics holding 0).
        let buckets = unsafe { std::alloc::alloc_zeroed(layout) as *mut Bucket };
        if buckets.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        TT { buckets, len, gen: AtomicU8::new(0) }
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
            for w in &b.0 {
                w.store(0, Relaxed);
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
            let data = b.0[2 * i + 1].load(Relaxed);
            if data != 0 && b.0[2 * i].load(Relaxed) ^ data == key {
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
            let data = b.0[2 * i + 1].load(Relaxed);
            if data == 0 {
                victim = i;
                break;
            }
            let old = unpack(data);
            if b.0[2 * i].load(Relaxed) ^ data == key {
                if mv.is_none() {
                    mv = old.mv;
                }
                // Keep deeper results from the same search unless this one is exact.
                if bound != BOUND_EXACT && depth + 4 <= old.depth && ((data >> 59) as u8) == gen {
                    return;
                }
                victim = i;
                break;
            }
            let age = (gen.wrapping_sub((data >> 59) as u8) & GEN_MASK as u8) as i32;
            let quality = old.depth - 8 * age;
            if quality < worst {
                worst = quality;
                victim = i;
            }
        }
        let data = pack(mv, score, eval, depth, bound, pv, gen);
        b.0[2 * victim].store(key ^ data, Relaxed);
        b.0[2 * victim + 1].store(data, Relaxed);
    }

    /// Permille of the table used by the current search.
    pub fn hashfull(&self) -> usize {
        let gen = self.gen.load(Relaxed);
        let n = self.len.min(250);
        let mut used = 0;
        for i in 0..n {
            let b = unsafe { &*self.buckets.add(i) };
            for e in 0..ENTRIES_PER_BUCKET {
                let data = b.0[2 * e + 1].load(Relaxed);
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
        let layout = std::alloc::Layout::array::<Bucket>(self.len).unwrap();
        unsafe { std::alloc::dealloc(self.buckets as *mut u8, layout) };
    }
}
