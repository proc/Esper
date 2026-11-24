//! NNUE evaluation for Stockfish's "threat inputs" network architecture
//! (file version 0x6A448AFA):
//!
//!   feature transformer (int8 weights, 1024 outputs per perspective) fed by
//!     * K32H3      king-bucketed piece-square features, in 3 material "blocks"
//!     * FullThreats  piece-attacks-piece features
//!     * PP_3Wide   pawn-pair features
//!   -> pairwise clipped multiply -> 1024 -> 32 -> 32 -> 1, with one of up to
//!   192 layer stacks selected from material configuration.
//!
//! The accumulator is kept as two lazily, incrementally updated components per
//! perspective: the piece-square part (refreshed through a per-king-square cache
//! when the own king moves or the material block changes) and the threat/pawn-pair
//! part (only refreshed when the own king crosses the board's centre files).
//! Incremental updates are derived by diffing consecutive positions of the search
//! stack, so no bookkeeping is needed in make-move.

#![cfg_attr(feature = "portable", allow(dead_code))]

use crate::bitboard::*;
use crate::position::*;
use crate::simd::{self, Level};
use std::sync::Arc;

pub const L1: usize = 1024;
const L2: usize = 32;
const L3: usize = 32;
const THREAT_DIMS: usize = 59808;
const PAIR_DIMS: usize = 96 * 95 / 2;
const PSQ_BLOCK_DIMS: usize = 32 * 704;
const PSQ_DIMS: usize = 3 * PSQ_BLOCK_DIMS;
const RAW_STACKS: usize = 192;
const VERSION: u32 = 0x6A44_8AFA;
const STACK_TABLE_HASH: u32 = 0x5A7A_B1E5;
const MAX_THREATS: usize = 256;

pub const DEFAULT_NET_NAME: &str = "nn-bddd9d57eb87.nnue";

#[cfg(feature = "embed")]
static EMBEDDED: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/nn-bddd9d57eb87.nnue"));

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Acc(pub [i16; L1]);

#[repr(C, align(64))]
struct LayerStack {
    /// First layer weights grouped by 4 consecutive inputs:
    /// `w0[input / 4][output * 4 + input % 4]`.
    w0: [[i8; 4 * L2]; L1 / 4],
    b0: [i32; L2],
    w1: [[i8; 2 * L2]; L3],
    /// `w1` re-laid out by groups of 4 inputs, `w1c[input / 4][output * 4 + input % 4]`,
    /// so all outputs accumulate in registers (no per-output horizontal sums).
    w1c: [[i8; 4 * L3]; 2 * L2 / 4],
    b1: [i32; L3],
    w2: [i8; 2 * L2 + 2 * L3],
    b2: i32,
}

struct ThreatLuts {
    /// `[attacker][attacked][from < to]`
    lut1: [[[u32; 2]; 16]; 16],
    /// `[piece type][from][to]`
    lut2: [[[u16; 64]; 64]; 7],
    pair_bb: [u64; 64],
}

struct MmapBacking(*const u8, usize);
unsafe impl Send for MmapBacking {}
unsafe impl Sync for MmapBacking {}

#[cfg(unix)]
impl Drop for MmapBacking {
    fn drop(&mut self) {
        extern "C" {
            fn munmap(addr: *mut core::ffi::c_void, len: usize) -> core::ffi::c_int;
        }
        unsafe {
            munmap(self.0 as *mut core::ffi::c_void, self.1);
        }
    }
}

pub struct Network {
    data: &'static [u8],
    _backing: Option<MmapBacking>,
    /// Huge-page copy of the weights (Linux), which `data` then points into.
    _large: Option<crate::mem::LargeAlloc>,
    biases: Acc,
    tp_off: usize,
    psq_off: usize,
    stacks: Vec<LayerStack>,
    stack_index: [u16; RAW_STACKS],
    luts: Box<ThreatLuts>,
    /// Set when the file is a small network (see `small.rs`); the fields above
    /// are then unused.
    pub small: Option<crate::small::SmallNet>,
    pub description: String,
}

struct Reader<'a> {
    d: &'a [u8],
    pos: usize,
}
impl Reader<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.d[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn i32(&mut self) -> i32 {
        self.u32() as i32
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes(self.d[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        v
    }
    fn i8(&mut self) -> i8 {
        let v = self.d[self.pos] as i8;
        self.pos += 1;
        v
    }
}

fn build_luts() -> Box<ThreatLuts> {
    let t = tables();
    let mut l = Box::new(ThreatLuts { lut1: [[[0; 2]; 16]; 16], lut2: [[[0; 64]; 64]; 7], pair_bb: [0; 64] });
    let pseudo = |pt: usize, sq: usize| match pt {
        KNIGHT => t.knight[sq],
        BISHOP => t.bishop_empty[sq],
        ROOK => t.rook_empty[sq],
        QUEEN => t.bishop_empty[sq] | t.rook_empty[sq],
        _ => t.king[sq],
    };
    let mut offsets = [0u32; 7];
    for pt in PAWN..=KING {
        let mut offset = 0u32;
        for from in 0..64 {
            for to in 0..64 {
                let attacks = if pt != PAWN {
                    pseudo(pt, from)
                } else if to < from {
                    t.pawn[BLACK][from]
                } else {
                    t.pawn[WHITE][from]
                };
                l.lut2[pt][from][to] = (offset + popcount((bit(to) - 1) & attacks)) as u16;
            }
            if pt != PAWN {
                offset += popcount(pseudo(pt, from));
            } else if (8..56).contains(&from) {
                offset += popcount(t.pawn[WHITE][from]);
            }
        }
        offsets[pt] = offset;
    }

    const VALID_TARGETS: [u32; 7] = [0, 4, 10, 8, 8, 10, 0];
    const MAP: [[i32; 6]; 6] = [
        [-1, 0, -1, 1, -1, -1],
        [0, 1, 2, 3, 4, -1],
        [0, 1, 2, 3, -1, -1],
        [0, 1, 2, 3, -1, -1],
        [0, 1, 2, 3, 4, -1],
        [-1, -1, -1, -1, -1, -1],
    ];
    let all_pieces: Vec<u8> = (0..2).flat_map(|c| (PAWN..=KING).map(move |pt| make_piece(c, pt))).collect();
    let mut cumulative = 0u32;
    for &attacker in &all_pieces {
        let at = piece_type(attacker);
        let per_piece = offsets[at];
        let valid = VALID_TARGETS[at];
        for &attacked in &all_pieces {
            let enemy = (attacker ^ attacked) == 8;
            let adt = piece_type(attacked);
            let map = MAP[at - 1][adt - 1];
            let semi_excluded = at == adt && (enemy || at != PAWN);
            let excluded = map < 0;
            let feature = if excluded {
                THREAT_DIMS as u32
            } else {
                cumulative + (piece_color(attacked) as u32 * (valid / 2) + map as u32) * per_piece
            };
            l.lut1[attacker as usize][attacked as usize][0] = feature;
            l.lut1[attacker as usize][attacked as usize][1] =
                if semi_excluded { THREAT_DIMS as u32 } else { feature };
        }
        cumulative += valid * per_piece;
    }
    assert_eq!(cumulative as usize, THREAT_DIMS);

    for s in 0..64 {
        let file = FILE_A << (s & 7);
        let files = file | ((file & !FILE_H) << 1) | ((file & !FILE_A) >> 1);
        l.pair_bb[s] = files & !(RANK_1 | RANK_8) & !bit(s);
    }
    l
}

impl Network {
    /// Like `from_bytes`, but on Linux the weights are first copied into huge
    /// pages: the threat weights are read at random, so with 4 KB pages most
    /// row loads would also miss the TLB.
    pub fn from_bytes_large(data: &'static [u8]) -> Result<Network, String> {
        if crate::mem::ENABLED {
            if let Some((large, copy)) = crate::mem::LargeAlloc::copy_of(data) {
                let mut net = Network::from_bytes(copy)?;
                net._large = Some(large);
                return Ok(net);
            }
        }
        Network::from_bytes(data)
    }

    pub fn from_bytes(data: &'static [u8]) -> Result<Network, String> {
        if crate::small::SmallNet::is_small_file(data) {
            let small = crate::small::SmallNet::from_bytes(data)?;
            return Ok(Network {
                data: &[],
                _backing: None,
                _large: None,
                biases: Acc([0; L1]),
                tp_off: 0,
                psq_off: 0,
                stacks: Vec::new(),
                stack_index: [0; RAW_STACKS],
                luts: build_luts(),
                description: small.description(),
                small: Some(small),
            });
        }
        if data.len() < 12 {
            return Err("file too short".into());
        }
        let mut r = Reader { d: data, pos: 0 };
        if r.u32() != VERSION {
            return Err("unsupported network file version".into());
        }
        let _hash = r.u32();
        let desc_len = r.u32() as usize;
        if data.len() < 12 + desc_len + 12 + 2 * RAW_STACKS {
            return Err("file too short".into());
        }
        let description = String::from_utf8_lossy(&data[12..12 + desc_len]).into_owned();
        r.pos += desc_len;
        if r.u32() != STACK_TABLE_HASH || r.u32() as usize != RAW_STACKS {
            return Err("unexpected layer stack table".into());
        }
        let stored = r.u32() as usize;
        if stored == 0 || stored > RAW_STACKS {
            return Err("bad layer stack count".into());
        }
        let mut fallback = [0u16; RAW_STACKS];
        for f in fallback.iter_mut() {
            *f = r.u16();
            if *f as usize >= RAW_STACKS {
                return Err("bad layer stack fallback".into());
            }
        }
        let mut stack_index = [0u16; RAW_STACKS];
        let mut n = 0u16;
        for s in 0..RAW_STACKS {
            if fallback[s] as usize == s {
                stack_index[s] = n;
                n += 1;
            }
        }
        if n as usize != stored {
            return Err("inconsistent layer stack table".into());
        }
        for s in 0..RAW_STACKS {
            let f = fallback[s] as usize;
            if fallback[f] as usize != f {
                return Err("inconsistent layer stack fallback".into());
            }
            stack_index[s] = stack_index[f];
        }

        let stack_bytes = 4 + 4 * L2 + L2 * L1 + 4 * L3 + L3 * 2 * L2 + 4 + (2 * L2 + 2 * L3);
        let ft_bytes = 4 + 2 * L1 + (THREAT_DIMS + PAIR_DIMS + PSQ_DIMS) * L1;
        if data.len() != r.pos + ft_bytes + stored * stack_bytes {
            return Err(format!(
                "unexpected file size {} (wanted {})",
                data.len(),
                r.pos + ft_bytes + stored * stack_bytes
            ));
        }

        let _ft_hash = r.u32();
        let mut biases = Acc([0; L1]);
        for b in biases.0.iter_mut() {
            *b = r.u16() as i16;
        }
        let tp_off = r.pos;
        r.pos += (THREAT_DIMS + PAIR_DIMS) * L1;
        let psq_off = r.pos;
        r.pos += PSQ_DIMS * L1;

        let mut stacks = Vec::with_capacity(stored);
        for _ in 0..stored {
            let _hash = r.u32();
            let mut s =
                LayerStack { w0: [[0; 4 * L2]; L1 / 4], b0: [0; L2], w1: [[0; 2 * L2]; L3], w1c: [[0; 4 * L3]; 2 * L2 / 4], b1: [0; L3], w2: [0; 128], b2: 0 };
            for b in s.b0.iter_mut() {
                *b = r.i32();
            }
            for o in 0..L2 {
                for i in 0..L1 {
                    s.w0[i / 4][o * 4 + i % 4] = r.i8();
                }
            }
            for b in s.b1.iter_mut() {
                *b = r.i32();
            }
            for o in 0..L3 {
                for i in 0..2 * L2 {
                    s.w1[o][i] = r.i8();
                    s.w1c[i / 4][o * 4 + i % 4] = s.w1[o][i];
                }
            }
            s.b2 = r.i32();
            for w in s.w2.iter_mut() {
                *w = r.i8();
            }
            stacks.push(s);
        }
        debug_assert_eq!(r.pos, data.len());

        Ok(Network {
            data,
            _backing: None,
            _large: None,
            biases,
            tp_off,
            psq_off,
            stacks,
            stack_index,
            luts: build_luts(),
            small: None,
            description,
        })
    }

    /// The network embedded in the executable (if built with the `embed` feature).
    pub fn embedded() -> Option<Result<Network, String>> {
        #[cfg(feature = "embed")]
        {
            Some(Network::from_bytes_large(EMBEDDED))
        }
        #[cfg(not(feature = "embed"))]
        {
            None
        }
    }

    pub fn from_file(path: &str) -> Result<Network, String> {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            const PROT_READ: core::ffi::c_int = 1;
            const MAP_SHARED: core::ffi::c_int = 1;
            extern "C" {
                fn mmap(
                    addr: *mut core::ffi::c_void,
                    len: usize,
                    prot: core::ffi::c_int,
                    flags: core::ffi::c_int,
                    fd: core::ffi::c_int,
                    offset: i64,
                ) -> *mut core::ffi::c_void;
            }
            let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
            let len = file.metadata().map_err(|e| format!("{path}: {e}"))?.len() as usize;
            if len == 0 {
                return Err(format!("{path}: empty file"));
            }
            let ptr = unsafe { mmap(core::ptr::null_mut(), len, PROT_READ, MAP_SHARED, file.as_raw_fd(), 0) };
            let failed = usize::MAX as *mut core::ffi::c_void;
            if ptr == failed {
                return Err(format!("{path}: mmap failed: {}", std::io::Error::last_os_error()));
            }
            let backing = MmapBacking(ptr as *const u8, len);
            // Hint the OS to asynchronously read ahead the mapped network pages.
            const MADV_WILLNEED: core::ffi::c_int = 3;
            extern "C" {
                fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: core::ffi::c_int) -> core::ffi::c_int;
            }
            unsafe {
                madvise(ptr, len, MADV_WILLNEED);
            }
            let data: &'static [u8] = unsafe { core::slice::from_raw_parts(ptr as *const u8, len) };
            // On Linux, move the weights into huge pages (file mappings cannot use them).
            if crate::mem::ENABLED {
                if let Some((large, copy)) = crate::mem::LargeAlloc::copy_of(data) {
                    drop(backing);
                    let mut net = Network::from_bytes(copy)?;
                    net._large = Some(large);
                    return Ok(net);
                }
            }
            let mut net = match Network::from_bytes(data) {
                Ok(n) => n,
                Err(e) => {
                    drop(backing);
                    return Err(e);
                }
            };
            net._backing = Some(backing);
            Ok(net)
        }
        #[cfg(not(unix))]
        {
            let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            // Networks live for the lifetime of the process on non-Unix platforms.
            Network::from_bytes(Box::leak(bytes.into_boxed_slice()))
        }
    }

    #[inline(always)]
    fn tp_row(&self, idx: u32) -> &[u8; L1] {
        // Unchecked: indices are filtered below THREAT_DIMS (+ PAIR_DIMS for pawn
        // pairs) where they are made, and the loader insists on the exact file
        // size, so every row lies inside `data`.
        debug_assert!((idx as usize) < THREAT_DIMS + PAIR_DIMS);
        // SAFETY: see above.
        unsafe { &*(self.data.as_ptr().add(self.tp_off + idx as usize * L1) as *const [u8; L1]) }
    }
    #[inline(always)]
    fn psq_row(&self, idx: u32) -> &[u8; L1] {
        debug_assert!((idx as usize) < PSQ_DIMS);
        // SAFETY: as for `tp_row`; piece-square indices are below PSQ_DIMS.
        unsafe { &*(self.data.as_ptr().add(self.psq_off + idx as usize * L1) as *const [u8; L1]) }
    }
}

// ---------------------------------------------------------------------------
// Feature indexing
// ---------------------------------------------------------------------------

/// Horizontal mirror so the king is on files e..h, plus vertical flip for black.
#[inline(always)]
fn orientation(persp: usize, ksq: usize) -> usize {
    (if ksq & 7 < 4 { 7 } else { 0 }) ^ (56 * persp)
}

#[inline(always)]
fn psq_block(pos: &Position, persp: usize) -> usize {
    if pos.pieces(persp ^ 1, QUEEN) != 0 {
        2
    } else {
        ((pos.by_type[ROOK] | pos.by_type[QUEEN]) != 0) as usize
    }
}

#[inline(always)]
fn psq_base(persp: usize, ksq: usize, block: usize) -> usize {
    let k = ksq ^ (56 * persp);
    let file = k & 7;
    let bucket = (7 - (k >> 3)) * 4 + if file < 4 { file } else { 7 - file };
    block * PSQ_BLOCK_DIMS + bucket * 704
}

#[inline(always)]
fn psq_piece_offset(persp: usize, color: usize, pt: usize) -> usize {
    if pt == KING {
        640
    } else {
        (pt - 1) * 128 + (color ^ persp) * 64
    }
}

struct IdxList<const N: usize> {
    v: [std::mem::MaybeUninit<u32>; N],
    n: usize,
}
impl<const N: usize> IdxList<N> {
    #[inline(always)]
    fn new() -> Self {
        IdxList { v: [std::mem::MaybeUninit::uninit(); N], n: 0 }
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.n = 0;
    }
    #[inline(always)]
    fn push(&mut self, x: u32) {
        self.v[self.n].write(x);
        self.n += 1;
    }
    #[inline(always)]
    fn push_if(&mut self, x: u32, cond: bool) {
        self.v[self.n].write(x);
        self.n += cond as usize;
    }
    #[inline(always)]
    fn swap_remove(&mut self, i: usize) {
        debug_assert!(i < self.n);
        let last = self.as_slice()[self.n - 1];
        self.n -= 1;
        self.v[i].write(last);
    }
    #[inline(always)]
    fn as_slice(&self) -> &[u32] {
        // SAFETY: the first `n` elements have been written.
        unsafe { std::slice::from_raw_parts(self.v.as_ptr() as *const u32, self.n) }
    }
}

struct IdxList16<const N: usize> {
    v: [std::mem::MaybeUninit<u16>; N],
    n: usize,
}
impl<const N: usize> IdxList16<N> {
    #[inline(always)]
    fn new() -> Self {
        IdxList16 { v: [std::mem::MaybeUninit::uninit(); N], n: 0 }
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.n = 0;
    }
    #[inline(always)]
    fn push(&mut self, x: u32) {
        debug_assert!(x <= u16::MAX as u32);
        self.v[self.n].write(x as u16);
        self.n += 1;
    }
    #[inline(always)]
    fn push_if(&mut self, x: u32, cond: bool) {
        self.v[self.n].write(x as u16);
        self.n += cond as usize;
    }
    #[inline(always)]
    fn as_slice(&self) -> &[u16] {
        // SAFETY: the first `n` elements have been written.
        unsafe { std::slice::from_raw_parts(self.v.as_ptr() as *const u16, self.n) }
    }
}

trait IdxSink {
    fn push(&mut self, x: u32);
    fn push_if(&mut self, x: u32, cond: bool);
}

impl<const N: usize> IdxSink for IdxList<N> {
    #[inline(always)]
    fn push(&mut self, x: u32) {
        IdxList::push(self, x);
    }
    #[inline(always)]
    fn push_if(&mut self, x: u32, cond: bool) {
        IdxList::push_if(self, x, cond);
    }
}

impl<const N: usize> IdxSink for IdxList16<N> {
    #[inline(always)]
    fn push(&mut self, x: u32) {
        IdxList16::push(self, x);
    }
    #[inline(always)]
    fn push_if(&mut self, x: u32, cond: bool) {
        IdxList16::push_if(self, x, cond);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_tile_ptr(p: *const i8, off: usize) {
    unsafe {
        use std::arch::x86_64::*;
        _mm_prefetch(p.add(off), _MM_HINT_T0);
        _mm_prefetch(p.add(off + 64), _MM_HINT_T0);
    }
}

/// The whole 512-byte tile the AVX-512 kernels read next from a weight row.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn prefetch_tile_neon(p: *const i8, off: usize) {
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.add(off), options(nostack, preserves_flags, readonly));
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.add(off + 64), options(nostack, preserves_flags, readonly));
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_tile512_ptr(p: *const i8, off: usize) {
    unsafe {
        use std::arch::x86_64::*;
        for line in 0..8 {
            _mm_prefetch(p.add(off + line * 64), _MM_HINT_T0);
        }
    }
}

#[inline(always)]
fn prefetch_row(row: &[u8; L1]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let p = row.as_ptr();
        for k in 0..L1 / 128 {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.add(k * 128), options(nostack, preserves_flags, readonly));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let p = row.as_ptr() as *const i8;
        for k in 0..L1 / 64 {
            _mm_prefetch(p.add(k * 64), _MM_HINT_T0);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = row;
}

/// Piece-square features that differ between two piece placements.
#[inline]
fn psq_diff<const N: usize>(
    persp: usize,
    ksq: usize,
    block: usize,
    old: (&[u64; 2], &[u64; 7]),
    new: (&[u64; 2], &[u64; 7]),
    adds: &mut IdxList<N>,
    subs: &mut IdxList<N>,
) {
    let base = psq_base(persp, ksq, block);
    let orient = orientation(persp, ksq);
    for c in 0..2 {
        for pt in PAWN..=KING {
            let o = old.0[c] & old.1[pt];
            let n = new.0[c] & new.1[pt];
            if o == n {
                continue;
            }
            let off = base + psq_piece_offset(persp, c, pt);
            let mut rem = o & !n;
            while rem != 0 {
                subs.push((off + (pop_lsb(&mut rem) ^ orient)) as u32);
            }
            let mut add = n & !o;
            while add != 0 {
                adds.push((off + (pop_lsb(&mut add) ^ orient)) as u32);
            }
        }
    }
}

/// All piece-attacks-piece relations of a position in a perspective-independent
/// encoding (`from << 14 | to << 8 | attacker << 4 | attacked`), sorted ascending.
#[inline(never)]
fn gen_threats_masked(pos: &Position, mask: u64, out: &mut [u32; MAX_THREATS]) -> usize {
    let occ = pos.occ();
    let pawn_targets = pos.by_type[KNIGHT] | pos.by_type[ROOK];
    let slider_targets = pos.by_type[PAWN] | pos.by_type[KNIGHT] | pos.by_type[BISHOP] | pos.by_type[ROOK];
    let queen_targets = slider_targets | pos.by_type[QUEEN];
    let mut pieces = occ & !pos.by_type[KING] & mask;
    let mut n = 0;
    while pieces != 0 {
        let from = pop_lsb(&mut pieces);
        let pc = unsafe { *pos.board.get_unchecked(from) };
        let mut att = match piece_type(pc) {
            PAWN => pawn_attacks(piece_color(pc), from) & pawn_targets,
            KNIGHT => knight_attacks(from) & queen_targets,
            BISHOP => bishop_attacks(from, occ) & slider_targets,
            ROOK => rook_attacks(from, occ) & slider_targets,
            _ => (bishop_attacks(from, occ) | rook_attacks(from, occ)) & queen_targets,
        };
        while att != 0 {
            let to = pop_lsb(&mut att);
            if n < MAX_THREATS {
                out[n] = ((from as u32) << 14)
                    | ((to as u32) << 8)
                    | ((pc as u32) << 4)
                    | unsafe { *pos.board.get_unchecked(to) } as u32;
                n += 1;
            }
        }
    }
    n
}

fn gen_threats(pos: &Position, out: &mut [u32; MAX_THREATS]) -> usize {
    gen_threats_masked(pos, !0u64, out)
}

/// The board as seen by the threat features: what stands where.
#[derive(Clone, Copy)]
struct ThreatBoard {
    board: [u8; 64],
    by_type: [u64; 7],
    by_color: [u64; 2],
    occ: u64,
}

impl ThreatBoard {
    #[inline(always)]
    fn new(pos: &Position) -> ThreatBoard {
        ThreatBoard { board: pos.board, by_type: pos.by_type, by_color: pos.by_color, occ: pos.occ() }
    }

    #[inline(always)]
    fn toggle(&mut self, sq: usize, pc: u8) {
        let b = bit(sq);
        self.by_type[piece_type(pc)] ^= b;
        self.by_color[piece_color(pc)] ^= b;
        self.occ ^= b;
        self.board[sq] = if self.board[sq] == 0 { pc } else { 0 };
    }

}

/// What `piece_threats` reads of a board: a working copy (`ThreatBoard`) or a
/// position used as it is, which saves copying it.
trait ThreatView {
    fn piece(&self, sq: usize) -> u8;
    fn by_type(&self) -> &[u64; 7];
    fn by_color(&self) -> &[u64; 2];
    fn occupied(&self) -> u64;
    /// Piece types a piece of type `pt` can threaten (kings are never involved).
    #[inline(always)]
    fn targets(&self, pt: usize) -> u64 {
        let t = self.by_type();
        match pt {
            PAWN => t[KNIGHT] | t[ROOK],
            BISHOP | ROOK => t[PAWN] | t[KNIGHT] | t[BISHOP] | t[ROOK],
            KNIGHT | QUEEN => t[PAWN] | t[KNIGHT] | t[BISHOP] | t[ROOK] | t[QUEEN],
            _ => 0,
        }
    }
}

impl ThreatView for ThreatBoard {
    #[inline(always)]
    fn piece(&self, sq: usize) -> u8 {
        self.board[sq]
    }
    #[inline(always)]
    fn by_type(&self) -> &[u64; 7] {
        &self.by_type
    }
    #[inline(always)]
    fn by_color(&self) -> &[u64; 2] {
        &self.by_color
    }
    #[inline(always)]
    fn occupied(&self) -> u64 {
        self.occ
    }
}

impl ThreatView for Position {
    #[inline(always)]
    fn piece(&self, sq: usize) -> u8 {
        self.board[sq]
    }
    #[inline(always)]
    fn by_type(&self) -> &[u64; 7] {
        &self.by_type
    }
    #[inline(always)]
    fn by_color(&self) -> &[u64; 2] {
        &self.by_color
    }
    #[inline(always)]
    fn occupied(&self) -> u64 {
        self.by_color[WHITE] | self.by_color[BLACK]
    }
}

#[inline(always)]
fn threat_code(from: usize, to: usize, attacker: u8, attacked: u8) -> u32 {
    ((from as u32) << 14) | ((to as u32) << 8) | ((attacker as u32) << 4) | attacked as u32
}

/// Threats that involve the piece `pc` standing on `sq` in `tb` (as attacker or
/// target) go to `with`; threats that exist only while `sq` is empty (sliders
/// seeing past it) go to `without` (only with `XRAY`). `other` (64 for none) is
/// the second square of a quiet move: x-ray views along a ray holding both
/// squares appear at one square and vanish at the other, so neither is listed
/// (Stockfish's noRaysContaining).
#[inline(always)]
fn piece_threats<V: ThreatView, const N: usize, const XRAY: bool>(
    tb: &V,
    pc: u8,
    sq: usize,
    with: &mut IdxList<N>,
    without: &mut IdxList<N>,
    other: usize,
) {
    let pt = piece_type(pc);
    let occ = tb.occupied();
    let t = tb.by_type();
    // Slider rays from `sq`, shared by every part below.
    let diag = bishop_attacks(sq, occ);
    let ortho = rook_attacks(sq, occ);
    // What the piece attacks.
    let own = match pt {
        PAWN => pawn_attacks(piece_color(pc), sq),
        KNIGHT => knight_attacks(sq),
        BISHOP => diag,
        ROOK => ortho,
        QUEEN => diag | ortho,
        _ => 0,
    };
    let mut att = own & tb.targets(pt);
    while att != 0 {
        let to = pop_lsb(&mut att);
        with.push(threat_code(sq, to, pc, tb.piece(to)));
    }
    let diag_sliders = diag & (t[BISHOP] | t[QUEEN]);
    let ortho_sliders = ortho & (t[ROOK] | t[QUEEN]);
    if pt != KING {
        // What attacks the piece (kings neither attack nor get attacked, but block lines).
        let mut attackers = (knight_attacks(sq) & t[KNIGHT]) | ((diag | ortho) & t[QUEEN]);
        if pt != QUEEN {
            attackers |= (diag & t[BISHOP]) | (ortho & t[ROOK]);
        }
        if pt == KNIGHT || pt == ROOK {
            let c = tb.by_color();
            attackers |= (pawn_attacks(BLACK, sq) & t[PAWN] & c[WHITE]) | (pawn_attacks(WHITE, sq) & t[PAWN] & c[BLACK]);
        }
        while attackers != 0 {
            let from = pop_lsb(&mut attackers);
            with.push(threat_code(from, sq, tb.piece(from), pc));
        }
    }
    if !XRAY {
        return;
    }
    // Sliders whose line runs through `sq` see one piece further when it is
    // empty: the first piece past `sq` on that line, which `sq`'s own ray in the
    // same direction already found.
    let mut sliders = diag_sliders | ortho_sliders;
    while sliders != 0 {
        let from = pop_lsb(&mut sliders);
        if other < 64 && (between(from, sq) & bit(other) != 0 || between(from, other) & bit(sq) != 0) {
            continue;
        }
        let spc = tb.piece(from);
        let ray = if (diag_sliders >> from) & 1 != 0 { diag } else { ortho };
        let mut beyond = ray & line(from, sq) & !between(from, sq) & !bit(from) & tb.targets(piece_type(spc));
        while beyond != 0 {
            let to = pop_lsb(&mut beyond);
            without.push(threat_code(from, to, spc, tb.piece(to)));
        }
    }
}

/// Threat changes from `p0` to `p1`, worked out from the pieces that left and
/// arrived rather than by rescanning both boards. A threat can appear in both
/// lists; the accumulator arithmetic is linear, so that is harmless.
fn threat_changes_inc<const N: usize>(p0: &Position, p1: &Position, added: &mut IdxList<N>, removed: &mut IdxList<N>) {
    let mut changed = (p0.by_color[0] ^ p1.by_color[0]) | (p0.by_color[1] ^ p1.by_color[1]);
    for pt in PAWN..=KING {
        changed |= p0.by_type[pt] ^ p1.by_type[pt];
    }
    if THREAT_FAST_PATHS && changed.count_ones() == 2 {
        let (a, b) = (changed.trailing_zeros() as usize, 63 - changed.leading_zeros() as usize);
        // `from` empties; `to` is whichever square holds a piece afterwards.
        let (from, to) = if p1.board[a] == 0 { (a, b) } else { (b, a) };
        if p0.board[from] != 0 && p1.board[from] == 0 && p1.board[to] != 0 {
            if p0.board[to] == 0 {
                // Quiet move (or promotion): lift on the old board, set down on
                // the new one; neither needs a working copy.
                piece_threats::<_, N, true>(p0, p0.board[from], from, removed, added, to);
                piece_threats::<_, N, true>(p1, p1.board[to], to, added, removed, from);
            } else {
                // Capture: lift the mover, then swap victim for mover on `to`.
                // `to` stays occupied, so no line through it changes (no
                // x-rays), and the board between the two steps differs from
                // `p1` only on `to`, which neither piece's threats read.
                piece_threats::<_, N, true>(p0, p0.board[from], from, removed, added, 64);
                piece_threats::<_, N, false>(p1, p0.board[to], to, removed, added, 64);
                piece_threats::<_, N, false>(p1, p1.board[to], to, added, removed, 64);
            }
            if FAST_PATH_CANCEL {
                cancel_duplicates(added, removed);
            }
            return;
        }
    }
    let mut tb = ThreatBoard::new(p0);
    // Lift the pieces that leave or are replaced...
    let mut c = changed;
    while c != 0 {
        let sq = pop_lsb(&mut c);
        let pc = tb.board[sq];
        if pc != 0 {
            piece_threats::<_, N, true>(&tb, pc, sq, removed, added, 64);
            tb.toggle(sq, pc);
        }
    }
    // ...then set down the arrivals.
    let mut c = changed;
    while c != 0 {
        let sq = pop_lsb(&mut c);
        let pc = p1.board[sq];
        if pc != 0 {
            tb.toggle(sq, pc);
            piece_threats::<_, N, true>(&tb, pc, sq, added, removed, 64);
        }
    }
    cancel_duplicates(added, removed);
}

/// Drop threats that were both lost and regained: each costs two row updates.
/// (Turning this off cost about 2% in cycles: the duplicate rows cost more.)
#[inline(always)]
fn cancel_duplicates<const N: usize>(added: &mut IdxList<N>, removed: &mut IdxList<N>) {
    if !CANCEL_DUPLICATE_THREATS {
        return;
    }
    let mut i = 0;
    while i < added.n {
        let x = added.as_slice()[i];
        if let Some(j) = removed.as_slice().iter().position(|&r| r == x) {
            removed.swap_remove(j);
            added.swap_remove(i);
        } else {
            i += 1;
        }
    }
}

impl ThreatLuts {
    #[inline(always)]
    fn threat_index(&self, t: u32, persp: usize, orient: usize) -> u32 {
        let from = ((t >> 14) & 63) as usize ^ orient;
        let to = ((t >> 8) & 63) as usize ^ orient;
        let attacker = ((t >> 4) & 15) as usize;
        let attacked = (t & 15) as usize;
        let swap = 8 * persp;
        self.lut1[attacker ^ swap][attacked ^ swap][(from < to) as usize] + self.lut2[attacker & 7][from][to] as u32
    }

    /// Pawn-pair features involving at least one pawn of `updated`.
    #[inline]
    fn pair_indices<S: IdxSink>(
        &self,
        persp: usize,
        orient: usize,
        updated: u64,
        white: u64,
        black: u64,
        out: &mut S,
    ) {
        let unchanged = (white | black) & !updated;
        let mut u = updated;
        while u != 0 {
            let a = pop_lsb(&mut u);
            let mut partners = self.pair_bb[a] & (unchanged | u);
            if partners == 0 {
                continue;
            }
            let a_col = ((black >> a) & 1) as usize;
            let a_id = 48 * (a_col ^ persp) + (a ^ orient) - 8;
            while partners != 0 {
                let b = pop_lsb(&mut partners);
                let b_col = ((black >> b) & 1) as usize;
                let b_id = 48 * (b_col ^ persp) + (b ^ orient) - 8;
                let (hi, lo) = if a_id > b_id { (a_id, b_id) } else { (b_id, a_id) };
                out.push((hi * (hi - 1) / 2 + lo + THREAT_DIMS) as u32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Accumulator row arithmetic
// ---------------------------------------------------------------------------

// `dst = src + rows(adds) - rows(subs)`, processed in tiles small enough to keep
// the running sums in registers while every row is applied.

/// `dst = src + rows(adds) - rows(subs)` with the rows given as feature indices.
#[inline]
fn apply_rows<'a, F: Fn(u32) -> &'a [u8; L1]>(dst: &mut Acc, src: &Acc, row: F, adds: &[u32], subs: &[u32]) {
    if adds.is_empty() && subs.is_empty() {
        *dst = *src;
        return;
    }
    // Row addresses are looked up once, not once per row per tile.
    const MAXR: usize = TP_LIST + 16;
    // Left uninitialised: zero-filling these on every call cost more than it saved.
    let mut add_buf = [std::mem::MaybeUninit::<*const i8>::uninit(); MAXR];
    let mut sub_buf = [std::mem::MaybeUninit::<*const i8>::uninit(); MAXR];
    let (na, ns) = (adds.len().min(MAXR), subs.len().min(MAXR));
    debug_assert!(adds.len() <= MAXR && subs.len() <= MAXR);
    for (p, &i) in add_buf.iter_mut().zip(adds) {
        p.write(row(i).as_ptr() as *const i8);
    }
    for (p, &i) in sub_buf.iter_mut().zip(subs) {
        p.write(row(i).as_ptr() as *const i8);
    }
    // SAFETY: the first `na` / `ns` slots were just written.
    let add_ptrs: &[*const i8] = unsafe { std::slice::from_raw_parts(add_buf.as_ptr() as *const *const i8, na) };
    let sub_ptrs: &[*const i8] = unsafe { std::slice::from_raw_parts(sub_buf.as_ptr() as *const *const i8, ns) };
    apply_ptrs(dst, src, add_ptrs, sub_ptrs);
}

// ---------------------------------------------------------------------------
// Per-thread evaluation state
// ---------------------------------------------------------------------------

/// Largest possible full-refresh list: up to 256 threats plus 120 pawn pairs.
const TP_LIST: usize = 512;
/// Largest incremental threat/pawn-pair diff list for a single move.
const TP_DIFF_LIST: usize = 256;
/// Refresh as Stockfish does: cached piece-square sums plus all active threats in
/// one pass (otherwise: separate piece-square and threat caches, summed).
const SF_REFRESH: bool = true;
/// After a refresh, also compute the accumulators back to the refresh boundary
/// (see `fill_back`).
const BACKWARD_FILL: bool = true;
/// Use Stockfish's hybrid update for king moves (see `hybrid`).
const HYBRID: bool = true;
/// Stockfish only takes the hybrid route with this many pieces on the board.
const HYBRID_MIN_PIECES: u32 = 15;

struct Entry {
    /// Piece-square plus threat contributions, one accumulator per perspective
    /// (as Stockfish keeps it): an update is one pass, the output reads one array.
    acc: [Acc; 2],
    acc_ok: [bool; 2],
    /// The step from the previous ply cannot be done incrementally.
    psq_refresh: [bool; 2],
    tp_refresh: [bool; 2],
    psq_diff_ok: [bool; 2],
    tp_diff_ok: bool,
    /// The move that led from `ply - 1` to `ply` (`Move::NONE` for null move / root).
    mv: Move,
    // Feature rows to add / subtract relative to the previous ply.
    // All threat + pawn-pair indices fit in u16 (THREAT_DIMS + PAIR_DIMS = 64,368).
    psq_adds: [IdxList<8>; 2],
    psq_subs: [IdxList<8>; 2],
    tp_adds: [IdxList16<TP_DIFF_LIST>; 2],
    tp_subs: [IdxList16<TP_DIFF_LIST>; 2],
}

struct CacheEntry {
    acc: Acc,
    by_color: [u64; 2],
    by_type: [u64; 7],
}

pub struct NnueState {
    net: Arc<Network>,
    entries: Vec<Entry>,
    /// Piece-square accumulator cache, indexed `[perspective][block][king square]`.
    cache: Vec<CacheEntry>,
    /// Threat accumulator cache, indexed `[perspective][king on files e-h]`: the
    /// last threat accumulator built for that orientation and its position.
    tp_cache: Vec<Option<(Acc, Position)>>,
    /// Work space for rebuilding a threat cache entry without copying it.
    scratch: Box<Acc>,
    /// Shared threat scratch buffer reused when both perspectives refresh at the same ply.
    threats_buf: [u32; MAX_THREATS],
    threats_len: usize,
    threats_ply: usize,
    /// Accumulators for a small network; everything above is then left empty.
    small: Option<crate::small::SmallState>,
}

/// Above this many changed squares a threat refresh starts from scratch rather
/// than from the cached position.
const TP_CACHE_MAX_CHANGED: u32 = 12;

/// Threats present in `p1` but not `p0` (`added`) and the reverse (`removed`).
/// Only threats touching squares whose contents changed, or attacking through them,
/// can differ, so both positions are scanned on those squares alone.
fn threat_changes(p0: &Position, p1: &Position, added: &mut IdxList<MAX_THREATS>, removed: &mut IdxList<MAX_THREATS>) {
    let mut changed = (p0.by_color[0] ^ p1.by_color[0]) | (p0.by_color[1] ^ p1.by_color[1]);
    for pt in PAWN..=KING {
        changed |= p0.by_type[pt] ^ p1.by_type[pt];
    }
    if changed != 0 {
        let occ0 = p0.occ();
        let occ1 = p1.occ();
        let wp = p0.pieces(WHITE, PAWN) | p1.pieces(WHITE, PAWN);
        let bp = p0.pieces(BLACK, PAWN) | p1.pieces(BLACK, PAWN);
        let kn = p0.by_type[KNIGHT] | p1.by_type[KNIGHT];
        let diag0 = p0.by_type[BISHOP] | p0.by_type[QUEEN];
        let diag1 = p1.by_type[BISHOP] | p1.by_type[QUEEN];
        let ortho0 = p0.by_type[ROOK] | p0.by_type[QUEEN];
        let ortho1 = p1.by_type[ROOK] | p1.by_type[QUEEN];
        let mut affected = changed;
        let mut c = changed;
        while c != 0 {
            let sq = pop_lsb(&mut c);
            affected |= (pawn_attacks(BLACK, sq) & wp)
                | (pawn_attacks(WHITE, sq) & bp)
                | (knight_attacks(sq) & kn)
                | (bishop_attacks(sq, occ0) & diag0)
                | (bishop_attacks(sq, occ1) & diag1)
                | (rook_attacks(sq, occ0) & ortho0)
                | (rook_attacks(sq, occ1) & ortho1);
        }
        let mut old_t = [0u32; MAX_THREATS];
        let mut new_t = [0u32; MAX_THREATS];
        let n_old = gen_threats_masked(p0, affected, &mut old_t);
        let n_new = gen_threats_masked(p1, affected, &mut new_t);
        let (a, b) = (&old_t[..n_old], &new_t[..n_new]);
        let (mut i, mut m) = (0, 0);
        while i < a.len() && m < b.len() {
            let (x, y) = (a[i], b[m]);
            if x == y {
                i += 1;
                m += 1;
            } else if x < y {
                removed.push(x);
                i += 1;
            } else {
                added.push(y);
                m += 1;
            }
        }
        for &x in &a[i..] {
            removed.push(x);
        }
        for &y in &b[m..] {
            added.push(y);
        }
    }
}

/// Feature rows to add and subtract for perspective `p` (oriented by `orient`),
/// given the threat changes and the pawns before and after.
#[allow(clippy::too_many_arguments)]
fn tp_rows<const N: usize, S: IdxSink>(
    net: &Network,
    p: usize,
    orient: usize,
    added: &IdxList<N>,
    removed: &IdxList<N>,
    (w0, b0): (u64, u64),
    (w1, b1): (u64, u64),
    adds: &mut S,
    subs: &mut S,
) {
    for &t in added.as_slice() {
        let idx = net.luts.threat_index(t, p, orient);
        adds.push_if(idx, (idx as usize) < THREAT_DIMS);
    }
    for &t in removed.as_slice() {
        let idx = net.luts.threat_index(t, p, orient);
        subs.push_if(idx, (idx as usize) < THREAT_DIMS);
    }
    if w0 != w1 || b0 != b1 {
        net.luts.pair_indices(p, orient, (w1 & !w0) | (b1 & !b0), w1, b1, adds);
        net.luts.pair_indices(p, orient, (w0 & !w1) | (b0 & !b1), w0, b0, subs);
    }
}

/// `dst = src + rows(adds) - rows(subs)` with the rows given as addresses.
#[cfg(target_arch = "aarch64")]
fn apply_ptrs_neon(dst: &mut Acc, src: &Acc, adds: &[*const i8], subs: &[*const i8]) {
    use std::arch::aarch64::*;
    const TILE: usize = 128; // 16 registers of 8 lanes
    // SAFETY: every address points at a row of L1 weights.
    unsafe {
        let mut off = 0;
        while off < L1 {
            let next_off = off + TILE;
            let do_pf = next_off < L1;
            let mut regs = [vdupq_n_s16(0); TILE / 8];
            let s = src.0.as_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = vld1q_s16(s.add(k * 8));
            }
            for &p in adds {
                if do_pf {
                    prefetch_tile_neon(p, next_off);
                }
                let w = p.add(off);
                for k in 0..TILE / 16 {
                    let v = vld1q_s8(w.add(k * 16));
                    regs[2 * k] = vaddw_s8(regs[2 * k], vget_low_s8(v));
                    regs[2 * k + 1] = vaddw_high_s8(regs[2 * k + 1], v);
                }
            }
            for &p in subs {
                if do_pf {
                    prefetch_tile_neon(p, next_off);
                }
                let w = p.add(off);
                for k in 0..TILE / 16 {
                    let v = vld1q_s8(w.add(k * 16));
                    regs[2 * k] = vsubw_s8(regs[2 * k], vget_low_s8(v));
                    regs[2 * k + 1] = vsubw_high_s8(regs[2 * k + 1], v);
                }
            }
            let d = dst.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter().enumerate() {
                vst1q_s16(d.add(k * 8), *r);
            }
            off = next_off;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_ptrs_avx2(dst: &mut Acc, src: &Acc, adds: &[*const i8], subs: &[*const i8]) {
    use std::arch::x86_64::*;
    const REGS: usize = 8; // 128 lanes
    const TILE: usize = REGS * 16;
    unsafe {
        let mut off = 0;
        while off < L1 {
            let next_off = off + TILE;
            let do_pf = next_off < L1;
            let mut regs = [_mm256_setzero_si256(); REGS];
            let s = src.0.as_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = _mm256_loadu_si256(s.add(k * 16) as *const __m256i);
            }
            for &p in adds {
                if do_pf {
                    prefetch_tile_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(k * 16) as *const __m128i));
                    *r = _mm256_add_epi16(*r, v);
                }
            }
            for &p in subs {
                if do_pf {
                    prefetch_tile_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(k * 16) as *const __m128i));
                    *r = _mm256_sub_epi16(*r, v);
                }
            }
            let d = dst.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter().enumerate() {
                _mm256_storeu_si256(d.add(k * 16) as *mut __m256i, *r);
            }
            off = next_off;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
unsafe fn apply_ptrs_avx512(dst: &mut Acc, src: &Acc, adds: &[*const i8], subs: &[*const i8]) {
    use std::arch::x86_64::*;
    const REGS: usize = 16; // 512 lanes
    const TILE: usize = REGS * 32;
    unsafe {
        let mut off = 0;
        while off < L1 {
            let next_off = off + TILE;
            let do_pf = next_off < L1;
            let mut regs = [_mm512_setzero_si512(); REGS];
            let s = src.0.as_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = _mm512_loadu_si512(s.add(k * 32) as *const _);
            }
            for &p in adds {
                if do_pf {
                    prefetch_tile512_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm512_cvtepi8_epi16(_mm256_loadu_si256(w.add(k * 32) as *const __m256i));
                    *r = _mm512_add_epi16(*r, v);
                }
            }
            for &p in subs {
                if do_pf {
                    prefetch_tile512_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm512_cvtepi8_epi16(_mm256_loadu_si256(w.add(k * 32) as *const __m256i));
                    *r = _mm512_sub_epi16(*r, v);
                }
            }
            let d = dst.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter().enumerate() {
                _mm512_storeu_si512(d.add(k * 32) as *mut _, *r);
            }
            off = next_off;
        }
    }
}

/// Portable version: tiles of plain lanes, which the compiler vectorises.
fn apply_ptrs_scalar(dst: &mut Acc, src: &Acc, adds: &[*const i8], subs: &[*const i8]) {
    const TILE: usize = 256;
    let mut off = 0;
    while off < L1 {
        let mut regs = [0i16; TILE];
        regs.copy_from_slice(&src.0[off..off + TILE]);
        for &p in adds {
            // SAFETY: every address points at a row of L1 weights.
            let w = unsafe { std::slice::from_raw_parts(p.add(off), TILE) };
            for k in 0..TILE {
                regs[k] = regs[k].wrapping_add(w[k] as i16);
            }
        }
        for &p in subs {
            let w = unsafe { std::slice::from_raw_parts(p.add(off), TILE) };
            for k in 0..TILE {
                regs[k] = regs[k].wrapping_sub(w[k] as i16);
            }
        }
        dst.0[off..off + TILE].copy_from_slice(&regs);
        off += TILE;
    }
}

/// `dst = src + rows(adds) - rows(subs)` with the rows given as addresses, on
/// the best code path this CPU supports.
#[inline(always)]
fn apply_ptrs(dst: &mut Acc, src: &Acc, adds: &[*const i8], subs: &[*const i8]) {
    let level = simd::level();
    #[cfg(target_arch = "aarch64")]
    if level >= Level::Neon {
        return apply_ptrs_neon(dst, src, adds, subs);
    }
    // SAFETY: `level` only names instruction sets the CPU was found to have.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if level >= Level::Avx512 {
            return apply_ptrs_avx512(dst, src, adds, subs);
        }
        if level >= Level::Avx2 {
            return apply_ptrs_avx2(dst, src, adds, subs);
        }
    }
    let _ = level;
    apply_ptrs_scalar(dst, src, adds, subs)
}

/// Row addresses for an incremental step: threat rows then piece-square rows.
/// Prefetches only tile 0 upfront; `apply_ptrs` prefetches subsequent tiles ahead.
#[inline(always)]
fn row_ptrs<'a>(
    net: &Network,
    tp: &[u16],
    psq: &[u32],
    buf: &'a mut [std::mem::MaybeUninit<*const i8>; TP_DIFF_LIST + 8],
) -> &'a [*const i8] {
    let n = tp.len() + psq.len();
    debug_assert!(n <= TP_DIFF_LIST + 8);
    for (d, &i) in buf.iter_mut().zip(tp) {
        let r = net.tp_row(i as u32);
        prefetch_row(r);
        d.write(r.as_ptr() as *const i8);
    }
    for (d, &i) in buf[tp.len()..].iter_mut().zip(psq) {
        let r = net.psq_row(i);
        prefetch_row(r);
        d.write(r.as_ptr() as *const i8);
    }
    // SAFETY: the first `n` slots were just written.
    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const *const i8, n) }
}

/// One-pass refresh (Stockfish's update_accumulator_refresh_cache): per tile,
/// `cache += psq rows`, stored back to the cache, then `+ threat rows`, stored to
/// `out`. Rows are given as addresses.
#[cfg(target_arch = "aarch64")]
fn refresh_rows_neon(cache: &mut Acc, out: &mut Acc, padds: &[*const i8], psubs: &[*const i8], tadds: &[*const i8]) {
    use std::arch::aarch64::*;
    const TILE: usize = 128;
    // SAFETY: every row holds L1 weights and the accumulators L1 lanes.
    unsafe {
        let mut off = 0;
        while off < L1 {
            let next_off = off + TILE;
            let do_pf = next_off < L1;
            let mut regs = [vdupq_n_s16(0); TILE / 8];
            let c = cache.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = vld1q_s16(c.add(k * 8));
            }
            for &p in padds {
                if do_pf {
                    prefetch_tile_neon(p, next_off);
                }
                let w = p.add(off);
                for k in 0..TILE / 16 {
                    let v = vld1q_s8(w.add(k * 16));
                    regs[2 * k] = vaddw_s8(regs[2 * k], vget_low_s8(v));
                    regs[2 * k + 1] = vaddw_high_s8(regs[2 * k + 1], v);
                }
            }
            for &p in psubs {
                if do_pf {
                    prefetch_tile_neon(p, next_off);
                }
                let w = p.add(off);
                for k in 0..TILE / 16 {
                    let v = vld1q_s8(w.add(k * 16));
                    regs[2 * k] = vsubw_s8(regs[2 * k], vget_low_s8(v));
                    regs[2 * k + 1] = vsubw_high_s8(regs[2 * k + 1], v);
                }
            }
            if !padds.is_empty() || !psubs.is_empty() {
                for (k, r) in regs.iter().enumerate() {
                    vst1q_s16(c.add(k * 8), *r);
                }
            }
            for &p in tadds {
                if do_pf {
                    prefetch_tile_neon(p, next_off);
                }
                let w = p.add(off);
                for k in 0..TILE / 16 {
                    let v = vld1q_s8(w.add(k * 16));
                    regs[2 * k] = vaddw_s8(regs[2 * k], vget_low_s8(v));
                    regs[2 * k + 1] = vaddw_high_s8(regs[2 * k + 1], v);
                }
            }
            let d = out.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter().enumerate() {
                vst1q_s16(d.add(k * 8), *r);
            }
            off = next_off;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn refresh_rows_avx2(cache: &mut Acc, out: &mut Acc, padds: &[*const i8], psubs: &[*const i8], tadds: &[*const i8]) {
    use std::arch::x86_64::*;
    const REGS: usize = 8;
    const TILE: usize = REGS * 16;
    unsafe {
        let mut off = 0;
        while off < L1 {
            let next_off = off + TILE;
            let do_pf = next_off < L1;
            let mut regs = [_mm256_setzero_si256(); REGS];
            let c = cache.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = _mm256_loadu_si256(c.add(k * 16) as *const __m256i);
            }
            for &p in padds {
                if do_pf {
                    prefetch_tile_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(k * 16) as *const __m128i));
                    *r = _mm256_add_epi16(*r, v);
                }
            }
            for &p in psubs {
                if do_pf {
                    prefetch_tile_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(k * 16) as *const __m128i));
                    *r = _mm256_sub_epi16(*r, v);
                }
            }
            if !padds.is_empty() || !psubs.is_empty() {
                for (k, r) in regs.iter().enumerate() {
                    _mm256_storeu_si256(c.add(k * 16) as *mut __m256i, *r);
                }
            }
            for &p in tadds {
                if do_pf {
                    prefetch_tile_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(k * 16) as *const __m128i));
                    *r = _mm256_add_epi16(*r, v);
                }
            }
            let d = out.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter().enumerate() {
                _mm256_storeu_si256(d.add(k * 16) as *mut __m256i, *r);
            }
            off = next_off;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
unsafe fn refresh_rows_avx512(cache: &mut Acc, out: &mut Acc, padds: &[*const i8], psubs: &[*const i8], tadds: &[*const i8]) {
    use std::arch::x86_64::*;
    const REGS: usize = 16;
    const TILE: usize = REGS * 32;
    unsafe {
        let mut off = 0;
        while off < L1 {
            let next_off = off + TILE;
            let do_pf = next_off < L1;
            let mut regs = [_mm512_setzero_si512(); REGS];
            let c = cache.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = _mm512_loadu_si512(c.add(k * 32) as *const _);
            }
            for &p in padds {
                if do_pf {
                    prefetch_tile512_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm512_cvtepi8_epi16(_mm256_loadu_si256(w.add(k * 32) as *const __m256i));
                    *r = _mm512_add_epi16(*r, v);
                }
            }
            for &p in psubs {
                if do_pf {
                    prefetch_tile512_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm512_cvtepi8_epi16(_mm256_loadu_si256(w.add(k * 32) as *const __m256i));
                    *r = _mm512_sub_epi16(*r, v);
                }
            }
            if !padds.is_empty() || !psubs.is_empty() {
                for (k, r) in regs.iter().enumerate() {
                    _mm512_storeu_si512(c.add(k * 32) as *mut _, *r);
                }
            }
            for &p in tadds {
                if do_pf {
                    prefetch_tile512_ptr(p, next_off);
                }
                let w = p.add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm512_cvtepi8_epi16(_mm256_loadu_si256(w.add(k * 32) as *const __m256i));
                    *r = _mm512_add_epi16(*r, v);
                }
            }
            let d = out.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter().enumerate() {
                _mm512_storeu_si512(d.add(k * 32) as *mut _, *r);
            }
            off = next_off;
        }
    }
}

/// Portable version: tiles of plain lanes, which the compiler vectorises.
fn refresh_rows_scalar(cache: &mut Acc, out: &mut Acc, padds: &[*const i8], psubs: &[*const i8], tadds: &[*const i8]) {
    const TILE: usize = 256;
    let mut off = 0;
    while off < L1 {
        let mut regs = [0i16; TILE];
        regs.copy_from_slice(&cache.0[off..off + TILE]);
        for &p in padds {
            // SAFETY: every address points at a row of L1 weights.
            let w = unsafe { std::slice::from_raw_parts(p.add(off), TILE) };
            for k in 0..TILE {
                regs[k] = regs[k].wrapping_add(w[k] as i16);
            }
        }
        for &p in psubs {
            let w = unsafe { std::slice::from_raw_parts(p.add(off), TILE) };
            for k in 0..TILE {
                regs[k] = regs[k].wrapping_sub(w[k] as i16);
            }
        }
        cache.0[off..off + TILE].copy_from_slice(&regs);
        for &p in tadds {
            let w = unsafe { std::slice::from_raw_parts(p.add(off), TILE) };
            for k in 0..TILE {
                regs[k] = regs[k].wrapping_add(w[k] as i16);
            }
        }
        out.0[off..off + TILE].copy_from_slice(&regs);
        off += TILE;
    }
}

/// One-pass refresh on the best code path this CPU supports.
#[inline(always)]
fn refresh_rows(cache: &mut Acc, out: &mut Acc, padds: &[*const i8], psubs: &[*const i8], tadds: &[*const i8]) {
    let level = simd::level();
    #[cfg(target_arch = "aarch64")]
    if level >= Level::Neon {
        return refresh_rows_neon(cache, out, padds, psubs, tadds);
    }
    // SAFETY: `level` only names instruction sets the CPU was found to have.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if level >= Level::Avx512 {
            return refresh_rows_avx512(cache, out, padds, psubs, tadds);
        }
        if level >= Level::Avx2 {
            return refresh_rows_avx2(cache, out, padds, psubs, tadds);
        }
    }
    let _ = level;
    refresh_rows_scalar(cache, out, padds, psubs, tadds)
}

/// Search the added and removed threat lists for pairs that cancel out.
const CANCEL_DUPLICATE_THREATS: bool = true;

/// Quiet moves and captures skip the working copy of the board.
const THREAT_FAST_PATHS: bool = true;
const FAST_PATH_CANCEL: bool = false;

/// Compute a move's feature changes when it is made rather than when the
/// position is evaluated. Off: many made positions are never evaluated, and the
/// wasted work cost 6% in cycles per node despite the earlier prefetch.
const EAGER_DIFF: bool = false;

/// Threat-feature rows that change from `p0` to `p1`, for both perspectives
/// (those not due a refresh).
#[inline]
fn compute_tp_diff(net: &Network, cur: &mut Entry, p0: &Position, p1: &Position) {
    let mv = cur.mv;
    if mv.is_none() {
        for p in 0..2 {
            if !cur.tp_refresh[p] {
                cur.tp_adds[p].clear();
                cur.tp_subs[p].clear();
            }
        }
        cur.tp_diff_ok = true;
        return;
    }
    let mut added = IdxList::<MAX_THREATS>::new();
    let mut removed = IdxList::<MAX_THREATS>::new();
    let flag = mv.flag();
    if THREAT_FAST_PATHS && flag != FLAG_EP && flag != FLAG_KCASTLE && flag != FLAG_QCASTLE {
        let (from, to) = (mv.from(), mv.to());
        let mover = p0.board[from];
        let cap = p0.board[to];
        let placed = p1.board[to];
        if cap == 0 {
            piece_threats::<_, MAX_THREATS, true>(p0, mover, from, &mut removed, &mut added, to);
            piece_threats::<_, MAX_THREATS, true>(p1, placed, to, &mut added, &mut removed, from);
        } else {
            piece_threats::<_, MAX_THREATS, true>(p0, mover, from, &mut removed, &mut added, 64);
            piece_threats::<_, MAX_THREATS, false>(p1, cap, to, &mut removed, &mut added, 64);
            piece_threats::<_, MAX_THREATS, false>(p1, placed, to, &mut added, &mut removed, 64);
        }
        if FAST_PATH_CANCEL {
            cancel_duplicates(&mut added, &mut removed);
        }
    } else {
        threat_changes_inc(p0, p1, &mut added, &mut removed);
    }
    let (w0, b0) = (p0.pieces(WHITE, PAWN), p0.pieces(BLACK, PAWN));
    let (w1, b1) = (p1.pieces(WHITE, PAWN), p1.pieces(BLACK, PAWN));
    for p in 0..2 {
        if !cur.tp_refresh[p] {
            let orient = orientation(p, p1.king_sq(p));
            let (adds, subs) = (&mut cur.tp_adds[p], &mut cur.tp_subs[p]);
            adds.clear();
            subs.clear();
            tp_rows(net, p, orient, &added, &removed, (w0, b0), (w1, b1), adds, subs);
        }
    }
    cur.tp_diff_ok = true;
}

/// Piece-square rows that change from `p0` to `p1` for perspective `persp`.
#[inline]
fn compute_psq_diff(cur: &mut Entry, p0: &Position, p1: &Position, persp: usize) {
    let (adds, subs) = (&mut cur.psq_adds[persp], &mut cur.psq_subs[persp]);
    adds.clear();
    subs.clear();
    let mv = cur.mv;
    if mv.is_none() {
        cur.psq_diff_ok[persp] = true;
        return;
    }
    let k1 = p1.king_sq(persp);
    let block = psq_block(p1, persp);
    let base = psq_base(persp, k1, block);
    let orient = orientation(persp, k1);
    let flag = mv.flag();
    if flag != FLAG_EP && flag != FLAG_KCASTLE && flag != FLAG_QCASTLE {
        let (from, to) = (mv.from(), mv.to());
        let mover = p0.board[from];
        let cap = p0.board[to];
        let placed = p1.board[to];
        subs.push((base + psq_piece_offset(persp, piece_color(mover), piece_type(mover)) + (from ^ orient)) as u32);
        if cap != 0 {
            subs.push((base + psq_piece_offset(persp, piece_color(cap), piece_type(cap)) + (to ^ orient)) as u32);
        }
        adds.push((base + psq_piece_offset(persp, piece_color(placed), piece_type(placed)) + (to ^ orient)) as u32);
        cur.psq_diff_ok[persp] = true;
        return;
    }
    // En passant or castling: scan the changed squares.
    let mut changed = (p0.by_color[0] ^ p1.by_color[0]) | (p0.by_color[1] ^ p1.by_color[1]);
    for pt in PAWN..=KING {
        changed |= p0.by_type[pt] ^ p1.by_type[pt];
    }
    while changed != 0 {
        let sq = pop_lsb(&mut changed);
        let (old, new) = (p0.board[sq], p1.board[sq]);
        if old != 0 {
            subs.push((base + psq_piece_offset(persp, piece_color(old), piece_type(old)) + (sq ^ orient)) as u32);
        }
        if new != 0 {
            adds.push((base + psq_piece_offset(persp, piece_color(new), piece_type(new)) + (sq ^ orient)) as u32);
        }
    }
    cur.psq_diff_ok[persp] = true;
}

impl NnueState {
    pub fn new(net: Arc<Network>, max_ply: usize) -> NnueState {
        let small = net.small.as_ref().map(|n| crate::small::SmallState::new(n, max_ply));
        let (max_ply, caches) = if small.is_some() { (0, 0) } else { (max_ply, 2 * 3 * 64) };
        let mut entries = Vec::with_capacity(max_ply);
        for _ in 0..max_ply {
            entries.push(Entry {
                acc: [Acc([0; L1]); 2],
                acc_ok: [false; 2],
                psq_refresh: [true; 2],
                tp_refresh: [true; 2],
                psq_diff_ok: [false; 2],
                tp_diff_ok: false,
                mv: Move::NONE,
                psq_adds: [IdxList::new(), IdxList::new()],
                psq_subs: [IdxList::new(), IdxList::new()],
                tp_adds: [IdxList16::new(), IdxList16::new()],
                tp_subs: [IdxList16::new(), IdxList16::new()],
            });
        }
        let mut cache = Vec::with_capacity(caches);
        for _ in 0..caches {
            cache.push(CacheEntry { acc: net.biases, by_color: [0; 2], by_type: [0; 7] });
        }
        NnueState {
            net,
            entries,
            cache,
            tp_cache: vec![None; 4],
            scratch: Box::new(Acc([0; L1])),
            threats_buf: [0; MAX_THREATS],
            threats_len: 0,
            threats_ply: usize::MAX,
            small,
        }
    }

    /// The network is a small one (see `small.rs`), whose output needs none of
    /// the Stockfish network's post-processing.
    #[inline(always)]
    pub fn is_small(&self) -> bool {
        self.small.is_some()
    }

    /// Start a new stack: the position at ply 0 is unrelated to anything before.
    pub fn reset(&mut self) {
        if let Some(s) = &mut self.small {
            return s.reset();
        }
        let e = &mut self.entries[0];
        e.acc_ok = [false; 2];
        e.psq_refresh = [true; 2];
        e.tp_refresh = [true; 2];
        e.psq_diff_ok = [false; 2];
        e.tp_diff_ok = false;
        self.threats_ply = usize::MAX;
    }

    /// Register the position at `stack[j]` (reached from `stack[j - 1]` via `mv`):
    /// mark accumulator/threat state as pending so diffs are only computed if evaluated.
    pub fn push(&mut self, stack: &[Position], j: usize, mv: Move) {
        if let Some(s) = &mut self.small {
            return s.push(j);
        }
        if self.threats_ply == j {
            self.threats_ply = usize::MAX;
        }
        let cur = &mut self.entries[j];
        let (p0, p1) = (&stack[j - 1], &stack[j]);
        cur.acc_ok = [false; 2];
        cur.psq_diff_ok = [false; 2];
        cur.tp_diff_ok = false;
        cur.mv = mv;
        if mv.is_none() {
            // Null move: board pieces and king squares are unchanged.
            cur.psq_refresh = [false; 2];
            cur.tp_refresh = [false; 2];
            return;
        }
        for persp in 0..2 {
            let (k0, k1) = (p0.king_sq(persp), p1.king_sq(persp));
            cur.psq_refresh[persp] = k0 != k1 || psq_block(p0, persp) != psq_block(p1, persp);
            cur.tp_refresh[persp] = (k0 ^ k1) & 4 != 0;
        }
        if EAGER_DIFF {
            // Work out the feature changes now, as Stockfish does when making a move,
            // and start loading their weight rows so the memory latency overlaps the
            // move generation that follows instead of stalling the evaluation.
            let net = &*self.net;
            let (p0, p1) = (&stack[j - 1], &stack[j]);
            compute_tp_diff(net, cur, p0, p1);
            for persp in 0..2 {
                if !cur.psq_refresh[persp] {
                    compute_psq_diff(cur, p0, p1, persp);
                    for &i in cur.psq_adds[persp].as_slice().iter().chain(cur.psq_subs[persp].as_slice()) {
                        prefetch_row(net.psq_row(i));
                    }
                }
                if !cur.tp_refresh[persp] {
                    for &i in cur.tp_adds[persp].as_slice().iter().chain(cur.tp_subs[persp].as_slice()) {
                        prefetch_row(net.tp_row(i as u32));
                    }
                }
            }
        }
    }

    /// Rebuild the accumulator of `persp` at `ply` as Stockfish does: the cached
    /// piece-square sums for this king square, brought up to date with `pos`,
    /// plus every active threat and pawn-pair row, all applied in one pass.
    #[inline(never)]
    fn refresh(&mut self, pos: &Position, ply: usize, persp: usize) {
        if !SF_REFRESH {
            return self.refresh_cached(pos, ply, persp);
        }
        let net = &*self.net;
        let ksq = pos.king_sq(persp);
        let block = psq_block(pos, persp);
        let orient = orientation(persp, ksq);
        let ci = (persp * 3 + block) * 64 + ksq;
        let c = &mut self.cache[ci];
        let (mut padds, mut psubs) = (IdxList::<64>::new(), IdxList::<64>::new());
        psq_diff(persp, ksq, block, (&c.by_color, &c.by_type), (&pos.by_color, &pos.by_type), &mut padds, &mut psubs);
        if self.threats_ply != ply {
            self.threats_len = gen_threats(pos, &mut self.threats_buf);
            self.threats_ply = ply;
        }
        let e = &mut self.entries[ply];
        // Row addresses: piece-square changes and every active threat and
        // pawn-pair feature; then one pass updates the cache and builds `out`.
        let mut tidx = IdxList::<TP_LIST>::new();
        for &t in &self.threats_buf[..self.threats_len] {
            let idx = net.luts.threat_index(t, persp, orient);
            tidx.push_if(idx, (idx as usize) < THREAT_DIMS);
        }
        let (w, b) = (pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN));
        net.luts.pair_indices(persp, orient, w | b, w, b, &mut tidx);
        let mut pa = [std::ptr::null::<i8>(); 64];
        let mut ps = [std::ptr::null::<i8>(); 64];
        let mut ta = [std::mem::MaybeUninit::<*const i8>::uninit(); TP_LIST];
        for (d, &i) in pa.iter_mut().zip(padds.as_slice()) {
            let r = net.psq_row(i);
            *d = r.as_ptr() as *const i8;
            prefetch_row(r);
        }
        for (d, &i) in ps.iter_mut().zip(psubs.as_slice()) {
            let r = net.psq_row(i);
            *d = r.as_ptr() as *const i8;
            prefetch_row(r);
        }
        for (d, &i) in ta.iter_mut().zip(tidx.as_slice()) {
            let r = net.tp_row(i);
            d.write(r.as_ptr() as *const i8);
            prefetch_row(r);
        }
        // SAFETY: the first `tidx.n` slots were just written.
        let ta = unsafe { std::slice::from_raw_parts(ta.as_ptr() as *const *const i8, tidx.n) };
        refresh_rows(&mut c.acc, &mut e.acc[persp], &pa[..padds.n], &ps[..psubs.n], ta);
        if padds.n + psubs.n > 0 {
            c.by_color = pos.by_color;
            c.by_type = pos.by_type;
        }
        e.acc_ok[persp] = true;
    }

    /// Rebuild the accumulator of `persp` at `ply` from the caches: the piece-square
    /// part from the per-king-square cache, the threat part from the per-side
    /// threat cache, each brought up to date with `pos` and then added together.
    #[inline(never)]
    fn refresh_cached(&mut self, pos: &Position, ply: usize, persp: usize) {
        let net = &*self.net;
        // Piece-square part, written straight into the entry and copied back to the cache.
        let ksq = pos.king_sq(persp);
        let block = psq_block(pos, persp);
        let c = &mut self.cache[(persp * 3 + block) * 64 + ksq];
        let (mut adds, mut subs) = (IdxList::<64>::new(), IdxList::<64>::new());
        psq_diff(persp, ksq, block, (&c.by_color, &c.by_type), (&pos.by_color, &pos.by_type), &mut adds, &mut subs);
        let e = &mut self.entries[ply];
        if adds.n + subs.n > 0 {
            for &i in adds.as_slice().iter().chain(subs.as_slice()) {
                prefetch_row(net.psq_row(i));
            }
            apply_rows(&mut e.acc[persp], &c.acc, |i| net.psq_row(i), adds.as_slice(), subs.as_slice());
            c.acc = e.acc[persp];
            c.by_color = pos.by_color;
            c.by_type = pos.by_type;
        } else {
            e.acc[persp] = c.acc;
        }

        // Threat part: bring the cache entry for this board side up to date.
        let orient = orientation(persp, ksq);
        let slot = persp * 2 + ((ksq >> 2) & 1);
        let mut updated = false;
        if let Some((cacc, cpos)) = &mut self.tp_cache[slot] {
            let mut changed = (cpos.by_color[0] ^ pos.by_color[0]) | (cpos.by_color[1] ^ pos.by_color[1]);
            for pt in PAWN..=KING {
                changed |= cpos.by_type[pt] ^ pos.by_type[pt];
            }
            if changed == 0 {
                updated = true;
            } else if popcount(changed) <= TP_CACHE_MAX_CHANGED {
                let mut added = IdxList::<MAX_THREATS>::new();
                let mut removed = IdxList::<MAX_THREATS>::new();
                threat_changes(cpos, pos, &mut added, &mut removed);
                let mut adds = IdxList::<TP_LIST>::new();
                let mut subs = IdxList::<TP_LIST>::new();
                let before = (cpos.pieces(WHITE, PAWN), cpos.pieces(BLACK, PAWN));
                let after = (pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN));
                tp_rows(net, persp, orient, &added, &removed, before, after, &mut adds, &mut subs);
                for &i in adds.as_slice().iter().chain(subs.as_slice()) {
                    prefetch_row(net.tp_row(i));
                }
                apply_rows(&mut self.scratch, cacc, |i| net.tp_row(i), adds.as_slice(), subs.as_slice());
                std::mem::swap(cacc, &mut *self.scratch);
                *cpos = *pos;
                updated = true;
            }
        }
        if !updated {
            if self.threats_ply != ply {
                self.threats_len = gen_threats(pos, &mut self.threats_buf);
                self.threats_ply = ply;
            }
            let mut adds = IdxList::<TP_LIST>::new();
            for &t in &self.threats_buf[..self.threats_len] {
                let idx = net.luts.threat_index(t, persp, orient);
                adds.push_if(idx, (idx as usize) < THREAT_DIMS);
            }
            let (w, b) = (pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN));
            net.luts.pair_indices(persp, orient, w | b, w, b, &mut adds);
            for &i in adds.as_slice() {
                prefetch_row(net.tp_row(i));
            }
            let zero = Acc([0; L1]);
            apply_rows(&mut self.scratch, &zero, |i| net.tp_row(i), adds.as_slice(), &[]);
            self.tp_cache[slot] = Some((*self.scratch, *pos));
        }
        let (tacc, _) = self.tp_cache[slot].as_ref().unwrap();
        let e = &mut self.entries[ply];
        for (d, t) in e.acc[persp].0.iter_mut().zip(tacc.0.iter()) {
            *d = d.wrapping_add(*t);
        }
        e.acc_ok[persp] = true;
    }

    /// After a refresh at `ply`, undo the moves back to `boundary` (the ply whose
    /// move forced the refresh), so the accumulators in between are computed too
    /// and later positions below `boundary` update incrementally instead of
    /// refreshing again (Stockfish's backward_update_incremental).
    fn fill_back(&mut self, stack: &[Position], ply: usize, boundary: usize, persp: usize) {
        let net = &*self.net;
        for t in (boundary + 1..=ply).rev() {
            let (lo, hi) = self.entries.split_at_mut(t);
            let (prev, cur) = (&mut lo[t - 1], &mut hi[0]);
            if !cur.psq_diff_ok[persp] {
                compute_psq_diff(cur, &stack[t - 1], &stack[t], persp);
            }
            if !cur.tp_diff_ok {
                compute_tp_diff(net, cur, &stack[t - 1], &stack[t]);
            }
            // The step from t-1 to t, reversed: its additions are subtracted.
            let mut abuf = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
            let mut sbuf = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
            let adds = row_ptrs(net, cur.tp_subs[persp].as_slice(), cur.psq_subs[persp].as_slice(), &mut abuf);
            let subs = row_ptrs(net, cur.tp_adds[persp].as_slice(), cur.psq_adds[persp].as_slice(), &mut sbuf);
            apply_ptrs(&mut prev.acc[persp], &cur.acc[persp], adds, subs);
            prev.acc_ok[persp] = true;
        }
    }

    /// Make the accumulator of `persp` at `ply` current: incrementally from the
    /// nearest computed ply when no refresh lies in between, otherwise rebuilt.
    /// Bring the piece-square cache entry for `pos`'s king square and block up to
    /// date with `pos`; returns its index.
    fn sync_psq_cache(&mut self, pos: &Position, persp: usize) -> usize {
        let net = &*self.net;
        let ksq = pos.king_sq(persp);
        let block = psq_block(pos, persp);
        let idx = (persp * 3 + block) * 64 + ksq;
        let c = &mut self.cache[idx];
        let (mut adds, mut subs) = (IdxList::<64>::new(), IdxList::<64>::new());
        psq_diff(persp, ksq, block, (&c.by_color, &c.by_type), (&pos.by_color, &pos.by_type), &mut adds, &mut subs);
        if adds.n + subs.n > 0 {
            for &i in adds.as_slice().iter().chain(subs.as_slice()) {
                prefetch_row(net.psq_row(i));
            }
            apply_rows(&mut self.scratch, &c.acc, |i| net.psq_row(i), adds.as_slice(), subs.as_slice());
            std::mem::swap(&mut c.acc, &mut *self.scratch);
            c.by_color = pos.by_color;
            c.by_type = pos.by_type;
        }
        idx
    }

    /// Stockfish's hybrid update for a king move that stays on the same side of
    /// the board: the threat part changes incrementally as usual, and the
    /// piece-square part is swapped wholesale (old king square's sum out, new
    /// one's in, both from the cache), avoiding a threat refresh.
    fn hybrid(&mut self, stack: &[Position], ply: usize, persp: usize) {
        let old = self.sync_psq_cache(&stack[ply - 1], persp);
        let new = self.sync_psq_cache(&stack[ply], persp);
        let net = &*self.net;
        let (lo, hi) = self.entries.split_at_mut(ply);
        let (prev, cur) = (&lo[ply - 1], &mut hi[0]);
        if !cur.tp_diff_ok {
            compute_tp_diff(net, cur, &stack[ply - 1], &stack[ply]);
        }
        let mut abuf = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
        let mut sbuf = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
        let adds = row_ptrs(net, cur.tp_adds[persp].as_slice(), &[], &mut abuf);
        let subs = row_ptrs(net, cur.tp_subs[persp].as_slice(), &[], &mut sbuf);
        apply_ptrs(&mut cur.acc[persp], &prev.acc[persp], adds, subs);
        let (a, b) = (&self.cache[old].acc.0, &self.cache[new].acc.0);
        for ((d, &o), &n) in cur.acc[persp].0.iter_mut().zip(a.iter()).zip(b.iter()) {
            *d = d.wrapping_sub(o).wrapping_add(n);
        }
        cur.acc_ok[persp] = true;
    }

    #[inline(never)]
    fn ensure(&mut self, stack: &[Position], ply: usize, persp: usize) {
        // A king move on the same side of the board, straight after a computed ply.
        if HYBRID && ply > 0 {
            let e = &self.entries[ply];
            if !e.acc_ok[persp]
                && e.psq_refresh[persp]
                && !e.tp_refresh[persp]
                && self.entries[ply - 1].acc_ok[persp]
                && stack[ply - 1].king_sq(persp) != stack[ply].king_sq(persp)
                && popcount(stack[ply].occ()) >= HYBRID_MIN_PIECES
            {
                return self.hybrid(stack, ply, persp);
            }
        }
        let mut k = ply;
        while !self.entries[k].acc_ok[persp] {
            let e = &self.entries[k];
            if k == 0 || e.psq_refresh[persp] || e.tp_refresh[persp] {
                self.refresh(&stack[ply], ply, persp);
                if BACKWARD_FILL {
                    self.fill_back(stack, ply, k, persp);
                }
                return;
            }
            k -= 1;
        }
        let net = &*self.net;
        for j in k + 1..=ply {
            let (lo, hi) = self.entries.split_at_mut(j);
            let (prev, cur) = (&lo[j - 1], &mut hi[0]);
            if !cur.psq_diff_ok[persp] {
                compute_psq_diff(cur, &stack[j - 1], &stack[j], persp);
            }
            if !cur.tp_diff_ok {
                compute_tp_diff(net, cur, &stack[j - 1], &stack[j]);
            }
            // Threat and piece-square rows go straight to address lists and are
            // applied in one pass.
            let mut abuf = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
            let mut sbuf = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
            let adds = row_ptrs(net, cur.tp_adds[persp].as_slice(), cur.psq_adds[persp].as_slice(), &mut abuf);
            let subs = row_ptrs(net, cur.tp_subs[persp].as_slice(), cur.psq_subs[persp].as_slice(), &mut sbuf);
            apply_ptrs(&mut cur.acc[persp], &prev.acc[persp], adds, subs);
            cur.acc_ok[persp] = true;
        }
    }

    /// Raw network output for the position at `stack[ply]`, from the side to move's
    /// point of view. `stack[0..=ply]` must be the path from the stack's root, with
    /// every ply after the root registered through `push`.
    pub fn evaluate(&mut self, stack: &[Position], ply: usize) -> i32 {
        if let Some(s) = &mut self.small {
            return s.evaluate(self.net.small.as_ref().unwrap(), stack, ply);
        }
        // Fast path: both perspectives need a 1-ply incremental update from ply - 1.
        // Prefetch both perspectives' rows before applying persp 0 so persp 1's
        // DRAM/L3 latency overlaps persp 0's SIMD compute.
        if ply > 0 {
            let (lo, hi) = self.entries.split_at_mut(ply);
            let (prev, cur) = (&lo[ply - 1], &mut hi[0]);
            if !cur.acc_ok[0]
                && !cur.acc_ok[1]
                && prev.acc_ok[0]
                && prev.acc_ok[1]
                && !cur.psq_refresh[0]
                && !cur.psq_refresh[1]
                && !cur.tp_refresh[0]
                && !cur.tp_refresh[1]
            {
                if cur.mv.is_none() {
                    cur.acc = prev.acc;
                    cur.acc_ok = [true, true];
                } else {
                    let net = &*self.net;
                    let (p0, p1) = (&stack[ply - 1], &stack[ply]);
                    if !cur.psq_diff_ok[0] {
                        compute_psq_diff(cur, p0, p1, 0);
                    }
                    if !cur.psq_diff_ok[1] {
                        compute_psq_diff(cur, p0, p1, 1);
                    }
                    if !cur.tp_diff_ok {
                        compute_tp_diff(net, cur, p0, p1);
                    }
                    let mut abuf0 = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
                    let mut sbuf0 = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
                    let mut abuf1 = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
                    let mut sbuf1 = [std::mem::MaybeUninit::uninit(); TP_DIFF_LIST + 8];
                    let adds0 = row_ptrs(net, cur.tp_adds[0].as_slice(), cur.psq_adds[0].as_slice(), &mut abuf0);
                    let subs0 = row_ptrs(net, cur.tp_subs[0].as_slice(), cur.psq_subs[0].as_slice(), &mut sbuf0);
                    let adds1 = row_ptrs(net, cur.tp_adds[1].as_slice(), cur.psq_adds[1].as_slice(), &mut abuf1);
                    let subs1 = row_ptrs(net, cur.tp_subs[1].as_slice(), cur.psq_subs[1].as_slice(), &mut sbuf1);
                    apply_ptrs(&mut cur.acc[0], &prev.acc[0], adds0, subs0);
                    apply_ptrs(&mut cur.acc[1], &prev.acc[1], adds1, subs1);
                    cur.acc_ok = [true, true];
                }
            } else {
                for persp in 0..2 {
                    self.ensure(stack, ply, persp);
                }
            }
        } else {
            for persp in 0..2 {
                self.ensure(stack, ply, persp);
            }
        }
        let pos = &stack[ply];
        let e = &self.entries[ply];
        let net = &*self.net;

        let s = &net.stacks[net.stack_index[layer_stack(pos)] as usize];
        let out = forward(&e.acc[pos.stm], &e.acc[pos.stm ^ 1], s);
        let scaled = (out as i64 * (600 * 16) / (128 * 64 * 2)) as i32;
        scaled / 16
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
//
// Each step exists once per instruction set; `forward` runs the chain built from
// the best set the CPU has (see `simd::level`). Functions that use instructions
// beyond the build's baseline carry `#[target_feature]`, so the binary itself
// stays runnable on older CPUs.

/// Non-zero 4-byte groups of the transformer output, as a list of group indices.
type Nnz = [u16; L1 / 4 + 8];
/// The same as a bitset, one bit per group.
type Bits = [u8; L1 / 32];

#[repr(C, align(64))]
struct Ft([u8; L1]);

#[repr(C, align(64))]
struct Act([u8; 2 * L2 + 2 * L3]);

/// Append up to 8 non-zero group indices described by `mask` starting at `base`.
#[inline(always)]
#[allow(dead_code)]
fn push_nnz8(mask: usize, base: u16, nnz: &mut Nnz, n: &mut usize) {
    let lut = &NNZ_LUT[mask];
    let dst = unsafe { nnz.as_mut_ptr().add(*n) };
    for k in 0..8 {
        unsafe { *dst.add(k) = base + *lut.get_unchecked(k) };
    }
    *n += mask.count_ones() as usize;
}

/// For every 8-bit mask, the positions of its set bits (padded with zeros).
#[allow(dead_code)]
static NNZ_LUT: [[u16; 8]; 256] = {
    let mut t = [[0u16; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut bit) = (0, 0);
        while bit < 8 {
            if m & (1 << bit) != 0 {
                t[m][k] = bit as u16;
                k += 1;
            }
            bit += 1;
        }
        m += 1;
    }
    t
};

// --- Feature transformer output -------------------------------------------
//
// out[j] = clamp(a[j], 0, 255) * clamp(a[j + 512], 0, 255) / 512, where `a` is
// the accumulator, while recording the non-zero 4-byte groups: as a bitset
// (`bits`) for the NEON dot-product path, as a list (`nnz`) for the others.

/// NEON; `DOT` selects the bitset recording the dot-product layer reads.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transform_neon_impl<const DOT: bool>(
    acc: &Acc,
    out: &mut [u8; L1 / 2],
    base_group: u16,
    nnz: &mut Nnz,
    n_nnz: &mut usize,
    bits: &mut Bits,
) {
    use std::arch::aarch64::*;
    const H: usize = L1 / 2;
    const BITS: [u32; 4] = [1, 2, 4, 8];
    let bit4 = vld1q_u32(BITS.as_ptr());
    let a = acc.0.as_ptr();
    // 16 outputs: two widening multiplies, then the high bytes of all 16
    // products in one de-interleave and a byte shift: (x * y) >> 9, as
    // Stockfish's NEON transform does it.
    let step16 = |j: usize| {
        let x0 = vqmovun_s16(vld1q_s16(a.add(j)));
        let x1 = vqmovun_s16(vld1q_s16(a.add(j + 8)));
        let y0 = vqmovun_s16(vld1q_s16(a.add(j + H)));
        let y1 = vqmovun_s16(vld1q_s16(a.add(j + H + 8)));
        let m0 = vreinterpretq_u8_u16(vmull_u8(x0, y0));
        let m1 = vreinterpretq_u8_u16(vmull_u8(x1, y1));
        vshrq_n_u8::<1>(vuzp2q_u8(m0, m1))
    };
    for j in (0..H).step_by(32) {
        let q0 = step16(j);
        let q1 = step16(j + 16);
        vst1q_u8(out.as_mut_ptr().add(j), q0);
        vst1q_u8(out.as_mut_ptr().add(j + 16), q1);
        let u0 = vreinterpretq_u32_u8(q0);
        let u1 = vreinterpretq_u32_u8(q1);
        if DOT {
            // Stockfish's recording: one byte per 8 groups, bit g for group g,
            // from a single horizontal add.
            const MASK8: [u16; 8] = [1, 16, 2, 32, 4, 64, 8, 128];
            let t0 = vreinterpretq_u16_u32(vtstq_u32(u0, u0));
            let t1 = vreinterpretq_u16_u32(vtstq_u32(u1, u1));
            let packed = vtrn1q_u16(t0, t1);
            let byte = vaddvq_u16(vandq_u16(packed, vld1q_u16(MASK8.as_ptr())));
            *bits.get_unchecked_mut(base_group as usize / 8 + j / 32) = byte as u8;
        } else {
            let lo = vaddvq_u32(vandq_u32(vtstq_u32(u0, u0), bit4));
            let hi = vaddvq_u32(vandq_u32(vtstq_u32(u1, u1), bit4));
            push_nnz8((lo | (hi << 4)) as usize, base_group + (j / 4) as u16, nnz, n_nnz);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transform_neon_bits(
    acc: &Acc,
    out: &mut [u8; L1 / 2],
    base_group: u16,
    nnz: &mut Nnz,
    n_nnz: &mut usize,
    bits: &mut Bits,
) {
    transform_neon_impl::<true>(acc, out, base_group, nnz, n_nnz, bits)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transform_neon_list(
    acc: &Acc,
    out: &mut [u8; L1 / 2],
    base_group: u16,
    nnz: &mut Nnz,
    n_nnz: &mut usize,
    bits: &mut Bits,
) {
    transform_neon_impl::<false>(acc, out, base_group, nnz, n_nnz, bits)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transform_avx2(
    acc: &Acc,
    out: &mut [u8; L1 / 2],
    base_group: u16,
    nnz: &mut Nnz,
    n_nnz: &mut usize,
    _bits: &mut Bits,
) {
    use std::arch::x86_64::*;
    const H: usize = L1 / 2;
    let a = acc.0.as_ptr();
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi16(255);
    let ld = |p: *const i16| _mm256_loadu_si256(p as *const __m256i);
    let step16 = |j: usize| {
        let x = ld(a.add(j));
        let y = ld(a.add(j + H));
        let x = _mm256_min_epi16(_mm256_max_epi16(x, zero), max);
        let y = _mm256_min_epi16(_mm256_max_epi16(y, zero), max);
        let p = _mm256_srli_epi16::<9>(_mm256_mullo_epi16(x, y));
        _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi16(p, p)))
    };
    for j in (0..H).step_by(32) {
        let v = _mm256_set_m128i(step16(j + 16), step16(j));
        _mm256_storeu_si256(out.as_mut_ptr().add(j) as *mut __m256i, v);
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, zero))) as usize;
        push_nnz8(mask, base_group + (j / 4) as u16, nnz, n_nnz);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
#[inline]
unsafe fn transform_avx512(
    acc: &Acc,
    out: &mut [u8; L1 / 2],
    base_group: u16,
    nnz: &mut Nnz,
    n_nnz: &mut usize,
    _bits: &mut Bits,
) {
    use std::arch::x86_64::*;
    const H: usize = L1 / 2;
    let a = acc.0.as_ptr();
    let zero = _mm512_setzero_si512();
    let zero256 = _mm256_setzero_si256();
    let max = _mm512_set1_epi16(255);
    let ld = |p: *const i16| _mm512_loadu_si512(p as *const _);
    for j in (0..H).step_by(32) {
        let x = ld(a.add(j));
        let y = ld(a.add(j + H));
        let x = _mm512_min_epi16(_mm512_max_epi16(x, zero), max);
        let y = _mm512_min_epi16(_mm512_max_epi16(y, zero), max);
        let p = _mm512_srli_epi16::<9>(_mm512_mullo_epi16(x, y));
        let v = _mm512_cvtepi16_epi8(p);
        _mm256_storeu_si256(out.as_mut_ptr().add(j) as *mut __m256i, v);
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, zero256))) as usize;
        push_nnz8(mask, base_group + (j / 4) as u16, nnz, n_nnz);
    }
}

#[inline(always)]
unsafe fn transform_scalar(
    acc: &Acc,
    out: &mut [u8; L1 / 2],
    base_group: u16,
    nnz: &mut Nnz,
    n_nnz: &mut usize,
    _bits: &mut Bits,
) {
    const H: usize = L1 / 2;
    let a = &acc.0;
    for j in (0..H).step_by(4) {
        let mut any = 0u8;
        for k in 0..4 {
            let x = a[j + k].clamp(0, 255) as u32;
            let y = a[j + k + H].clamp(0, 255) as u32;
            let v = ((x * y) >> 9) as u8;
            out[j + k] = v;
            any |= v;
        }
        if any != 0 {
            nnz[*n_nnz] = base_group + (j / 4) as u16;
            *n_nnz += 1;
        }
    }
}

// --- Dot product ------------------------------------------------------------
//
// Unsigned activations (at most 127) times signed weights; the length is a
// multiple of 64.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn dot_neon_dot(a: &[u8], w: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    debug_assert!(a.len() == w.len() && a.len() % 64 == 0);
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    for i in (0..a.len()).step_by(32) {
        acc0 = vdotq_s32(acc0, vld1q_s8(w.as_ptr().add(i)), vreinterpretq_s8_u8(vld1q_u8(a.as_ptr().add(i))));
        acc1 =
            vdotq_s32(acc1, vld1q_s8(w.as_ptr().add(i + 16)), vreinterpretq_s8_u8(vld1q_u8(a.as_ptr().add(i + 16))));
    }
    vaddvq_s32(vaddq_s32(acc0, acc1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_avx2(a: &[u8], w: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert!(a.len() == w.len() && a.len() % 64 == 0);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    for i in (0..a.len()).step_by(32) {
        let x = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let y = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        // 2 * 127 * 128 fits i16, so the pairwise sum never saturates.
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, y), ones));
    }
    let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b0100_1110>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b1011_0001>(s));
    _mm_cvtsi128_si32(s)
}

/// AVX-512 dot product and second layer, with (`$vnni`) or without the VNNI
/// dot-product instruction.
#[cfg(target_arch = "x86_64")]
macro_rules! tail_avx512_fns {
    ($dot:ident, $layer1:ident, $features:literal, $vnni:literal) => {
        #[target_feature(enable = $features)]
        #[inline]
        unsafe fn $dot(a: &[u8], w: &[i8]) -> i32 {
            use std::arch::x86_64::*;
            debug_assert!(a.len() == w.len() && a.len() % 64 == 0);
            let ones = _mm512_set1_epi16(1);
            let mut acc = _mm512_setzero_si512();
            for i in (0..a.len()).step_by(64) {
                let x = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
                let y = _mm512_loadu_si512(w.as_ptr().add(i) as *const _);
                if $vnni {
                    acc = _mm512_dpbusd_epi32(acc, x, y);
                } else {
                    // 2 * 127 * 128 fits i16, so the pairwise sum never saturates.
                    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(x, y), ones));
                }
            }
            _mm512_reduce_add_epi32(acc)
        }

        /// Input-group-major weights (`w1c`): each group of four activations is
        /// broadcast once and feeds all 32 outputs, so no horizontal sums are needed.
        #[target_feature(enable = $features)]
        #[inline]
        unsafe fn $layer1(s: &LayerStack, act: &[u8; 2 * L2]) -> [i32; L3] {
            use std::arch::x86_64::*;
            let ones = _mm512_set1_epi16(1);
            let mut acc = [_mm512_setzero_si512(); L3 / 16];
            for (k, a) in acc.iter_mut().enumerate() {
                *a = _mm512_loadu_si512(s.b1.as_ptr().add(k * 16) as *const _);
            }
            for c in 0..2 * L2 / 4 {
                let input = _mm512_set1_epi32(std::ptr::read_unaligned(act.as_ptr().add(c * 4) as *const i32));
                let w = s.w1c.get_unchecked(c).as_ptr();
                for (k, a) in acc.iter_mut().enumerate() {
                    let wv = _mm512_loadu_si512(w.add(k * 64) as *const _);
                    if $vnni {
                        *a = _mm512_dpbusd_epi32(*a, input, wv);
                    } else {
                        *a = _mm512_add_epi32(*a, _mm512_madd_epi16(_mm512_maddubs_epi16(input, wv), ones));
                    }
                }
            }
            let mut h = [0i32; L3];
            for (k, a) in acc.iter().enumerate() {
                _mm512_storeu_si512(h.as_mut_ptr().add(k * 16) as *mut _, *a);
            }
            h
        }
    };
}
#[cfg(target_arch = "x86_64")]
tail_avx512_fns!(dot_avx512, layer1_avx512, "avx2,avx512f,avx512bw", false);
#[cfg(target_arch = "x86_64")]
tail_avx512_fns!(dot_avx512_vnni, layer1_avx512_vnni, "avx2,avx512f,avx512bw,avx512vnni", true);

#[inline(always)]
unsafe fn dot_scalar(a: &[u8], w: &[i8]) -> i32 {
    debug_assert!(a.len() == w.len() && a.len() % 64 == 0);
    let mut sum = 0i32;
    for i in 0..a.len() {
        sum += a[i] as i32 * w[i] as i32;
    }
    sum
}

// --- Activation -------------------------------------------------------------
//
// Squared and linear clipped ReLU of 32 values:
// `out[i] = min(clamp(h,±16384)^2 >> SQ, 127)` and `out[32 + i] = clamp(h >> LIN, 0, 127)`.

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn activate_neon<const SQ: i32, const LIN: i32>(h: &[i32; L2], out: &mut [u8]) {
    use std::arch::aarch64::*;
    debug_assert!(out.len() >= 2 * L2);
    let (lo, hi, top) = (vdupq_n_s32(-16384), vdupq_n_s32(16384), vdupq_n_s32(127));
    let zero = vdupq_n_s32(0);
    for k in (0..L2).step_by(8) {
        let (a, b) = (vld1q_s32(h.as_ptr().add(k)), vld1q_s32(h.as_ptr().add(k + 4)));
        let (ca, cb) = (vmaxq_s32(vminq_s32(a, hi), lo), vmaxq_s32(vminq_s32(b, hi), lo));
        let sa = vminq_s32(vshrq_n_s32::<SQ>(vmulq_s32(ca, ca)), top);
        let sb = vminq_s32(vshrq_n_s32::<SQ>(vmulq_s32(cb, cb)), top);
        let la = vmaxq_s32(vminq_s32(vshrq_n_s32::<LIN>(a), top), zero);
        let lb = vmaxq_s32(vminq_s32(vshrq_n_s32::<LIN>(b), top), zero);
        let sq8 = vqmovn_u16(vcombine_u16(vqmovun_s32(sa), vqmovun_s32(sb)));
        let li8 = vqmovn_u16(vcombine_u16(vqmovun_s32(la), vqmovun_s32(lb)));
        vst1_u8(out.as_mut_ptr().add(k), sq8);
        vst1_u8(out.as_mut_ptr().add(L2 + k), li8);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
#[inline]
unsafe fn activate_avx512<const SQ: i32, const LIN: i32>(h: &[i32; L2], out: &mut [u8]) {
    use std::arch::x86_64::*;
    debug_assert!(out.len() >= 2 * L2);
    let (lo, hi) = (_mm512_set1_epi32(-16384), _mm512_set1_epi32(16384));
    let (top, zero) = (_mm512_set1_epi32(127), _mm512_setzero_si512());
    let (sq, lin) = (_mm_cvtsi32_si128(SQ), _mm_cvtsi32_si128(LIN));
    for k in (0..L2).step_by(16) {
        let a = _mm512_loadu_si512(h.as_ptr().add(k) as *const _);
        let c = _mm512_max_epi32(_mm512_min_epi32(a, hi), lo);
        let s = _mm512_min_epi32(_mm512_sra_epi32(_mm512_mullo_epi32(c, c), sq), top);
        let l = _mm512_max_epi32(_mm512_min_epi32(_mm512_sra_epi32(a, lin), top), zero);
        // Both are within 0..=127, so the narrowing to bytes is exact.
        _mm_storeu_si128(out.as_mut_ptr().add(k) as *mut __m128i, _mm512_cvtepi32_epi8(s));
        _mm_storeu_si128(out.as_mut_ptr().add(L2 + k) as *mut __m128i, _mm512_cvtepi32_epi8(l));
    }
}

#[inline(always)]
unsafe fn activate_scalar<const SQ: i32, const LIN: i32>(h: &[i32; L2], out: &mut [u8]) {
    debug_assert!(out.len() >= 2 * L2);
    for i in 0..L2 {
        let c = h[i].clamp(-16384, 16384);
        out[i] = ((c * c) >> SQ).min(127) as u8;
        out[L2 + i] = (h[i] >> LIN).clamp(0, 127) as u8;
    }
}

// --- Second layer: 64 activations to 32 outputs -----------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn layer1_neon_dot(s: &LayerStack, act: &[u8; 2 * L2]) -> [i32; L3] {
    use std::arch::aarch64::*;
    // Input-group-major weights: load 16 activations (4 groups of 4) once and
    // feed all 32 outputs via lane-indexed sdot into 16 accumulators.
    let mut acc0 = [vdupq_n_s32(0); L3 / 4];
    let mut acc1 = [vdupq_n_s32(0); L3 / 4];
    for (k, a) in acc0.iter_mut().enumerate() {
        *a = vld1q_s32(s.b1.as_ptr().add(k * 4));
    }
    let act_ptr = act.as_ptr();
    for g in 0..(2 * L2 / 16) {
        let v = vreinterpretq_s8_u8(vld1q_u8(act_ptr.add(g * 16)));
        let c = g * 4;
        let w0 = s.w1c.get_unchecked(c).as_ptr();
        let w1 = s.w1c.get_unchecked(c + 1).as_ptr();
        let w2 = s.w1c.get_unchecked(c + 2).as_ptr();
        let w3 = s.w1c.get_unchecked(c + 3).as_ptr();
        for k in 0..L3 / 4 {
            acc0[k] = vdotq_laneq_s32::<0>(acc0[k], vld1q_s8(w0.add(k * 16)), v);
            acc1[k] = vdotq_laneq_s32::<1>(acc1[k], vld1q_s8(w1.add(k * 16)), v);
            acc0[k] = vdotq_laneq_s32::<2>(acc0[k], vld1q_s8(w2.add(k * 16)), v);
            acc1[k] = vdotq_laneq_s32::<3>(acc1[k], vld1q_s8(w3.add(k * 16)), v);
        }
    }
    let mut h = [0i32; L3];
    for k in 0..L3 / 4 {
        vst1q_s32(h.as_mut_ptr().add(k * 4), vaddq_s32(acc0[k], acc1[k]));
    }
    h
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn layer1_avx2(s: &LayerStack, act: &[u8; 2 * L2]) -> [i32; L3] {
    let mut h = s.b1;
    for o in 0..L3 {
        h[o] += dot_avx2(act, &s.w1[o]);
    }
    h
}

#[inline(always)]
unsafe fn layer1_scalar(s: &LayerStack, act: &[u8; 2 * L2]) -> [i32; L3] {
    let mut h = s.b1;
    for o in 0..L3 {
        h[o] += dot_scalar(act, &s.w1[o]);
    }
    h
}

// --- First (sparse-input) layer ---------------------------------------------
//
// Only groups of four inputs that are not all zero are visited.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn layer0_neon_dot(s: &LayerStack, ft: &[u8; L1], _nnz: &[u16], bits: &Bits) -> [i32; L2] {
    use std::arch::aarch64::*;
    let mut acc0 = [vdupq_n_s32(0); L2 / 4];
    let mut acc1 = [vdupq_n_s32(0); L2 / 4];
    for (k, a) in acc0.iter_mut().enumerate() {
        *a = vld1q_s32(s.b0.as_ptr().add(k * 4));
    }
    let ft_ptr = ft.as_ptr() as *const u32;
    // Walk non-zero groups from the bitset, unrolled 2x into 16 accumulators
    // to hide the 3-cycle sdot latency and saturate NEON pipes.
    for word in 0..L1 / 4 / 64 {
        let mut b = u64::from_le_bytes(*(bits.as_ptr().add(word * 8) as *const [u8; 8]));
        let base = word * 64;
        while b != 0 {
            let c0 = base + b.trailing_zeros() as usize;
            b &= b - 1;
            if b != 0 {
                let c1 = base + b.trailing_zeros() as usize;
                b &= b - 1;
                let in0 = vreinterpretq_s8_u32(vld1q_dup_u32(ft_ptr.add(c0)));
                let in1 = vreinterpretq_s8_u32(vld1q_dup_u32(ft_ptr.add(c1)));
                let w0 = s.w0.get_unchecked(c0).as_ptr();
                let w1 = s.w0.get_unchecked(c1).as_ptr();
                for k in 0..L2 / 4 {
                    acc0[k] = vdotq_s32(acc0[k], vld1q_s8(w0.add(k * 16)), in0);
                    acc1[k] = vdotq_s32(acc1[k], vld1q_s8(w1.add(k * 16)), in1);
                }
            } else {
                let in0 = vreinterpretq_s8_u32(vld1q_dup_u32(ft_ptr.add(c0)));
                let w0 = s.w0.get_unchecked(c0).as_ptr();
                for k in 0..L2 / 4 {
                    acc0[k] = vdotq_s32(acc0[k], vld1q_s8(w0.add(k * 16)), in0);
                }
            }
        }
    }
    let mut h0 = [0i32; L2];
    for k in 0..L2 / 4 {
        vst1q_s32(h0.as_mut_ptr().add(k * 4), vaddq_s32(acc0[k], acc1[k]));
    }
    h0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn layer0_avx2(s: &LayerStack, ft: &[u8; L1], nnz: &[u16], _bits: &Bits) -> [i32; L2] {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); L2 / 8];
    for (k, a) in acc.iter_mut().enumerate() {
        *a = _mm256_loadu_si256(s.b0.as_ptr().add(k * 8) as *const __m256i);
    }
    for &c in nnz {
        let c = c as usize;
        let input = _mm256_set1_epi32(std::ptr::read_unaligned(ft.as_ptr().add(c * 4) as *const i32));
        let w = s.w0.get_unchecked(c).as_ptr();
        for (k, a) in acc.iter_mut().enumerate() {
            let wv = _mm256_loadu_si256(w.add(k * 32) as *const __m256i);
            *a = _mm256_add_epi32(*a, _mm256_madd_epi16(_mm256_maddubs_epi16(input, wv), ones));
        }
    }
    let mut h0 = [0i32; L2];
    for (k, a) in acc.iter().enumerate() {
        _mm256_storeu_si256(h0.as_mut_ptr().add(k * 8) as *mut __m256i, *a);
    }
    h0
}

/// AVX-512, with (`$vnni`) or without the VNNI dot-product instruction.
#[cfg(target_arch = "x86_64")]
macro_rules! layer0_avx512_fn {
    ($name:ident, $features:literal, $vnni:literal) => {
        #[target_feature(enable = $features)]
        #[inline]
        unsafe fn $name(s: &LayerStack, ft: &[u8; L1], nnz: &[u16], _bits: &Bits) -> [i32; L2] {
            use std::arch::x86_64::*;
            let mut acc = [_mm512_setzero_si512(); L2 / 16];
            for (k, a) in acc.iter_mut().enumerate() {
                *a = _mm512_loadu_si512(s.b0.as_ptr().add(k * 16) as *const _);
            }
            let ones = _mm512_set1_epi16(1);
            for &c in nnz {
                let c = c as usize;
                let input = _mm512_set1_epi32(std::ptr::read_unaligned(ft.as_ptr().add(c * 4) as *const i32));
                let w = s.w0.get_unchecked(c).as_ptr();
                for (k, a) in acc.iter_mut().enumerate() {
                    let wv = _mm512_loadu_si512(w.add(k * 64) as *const _);
                    if $vnni {
                        *a = _mm512_dpbusd_epi32(*a, input, wv);
                    } else {
                        *a = _mm512_add_epi32(*a, _mm512_madd_epi16(_mm512_maddubs_epi16(input, wv), ones));
                    }
                }
            }
            let mut h0 = [0i32; L2];
            for (k, a) in acc.iter().enumerate() {
                _mm512_storeu_si512(h0.as_mut_ptr().add(k * 16) as *mut _, *a);
            }
            h0
        }
    };
}
#[cfg(target_arch = "x86_64")]
layer0_avx512_fn!(layer0_avx512, "avx2,avx512f,avx512bw", false);
#[cfg(target_arch = "x86_64")]
layer0_avx512_fn!(layer0_avx512_vnni, "avx2,avx512f,avx512bw,avx512vnni", true);

#[inline(always)]
unsafe fn layer0_scalar(s: &LayerStack, ft: &[u8; L1], nnz: &[u16], _bits: &Bits) -> [i32; L2] {
    let mut h0 = s.b0;
    for &c in nnz {
        let c = c as usize;
        let x = std::ptr::read_unaligned(ft.as_ptr().add(c * 4) as *const u32).to_ne_bytes().map(|b| b as i32);
        let w = &s.w0[c];
        for o in 0..L2 {
            h0[o] +=
                x[0] * w[o * 4] as i32 + x[1] * w[o * 4 + 1] as i32 + x[2] * w[o * 4 + 2] as i32 + x[3] * w[o * 4 + 3] as i32;
        }
    }
    h0
}

// --- The whole pass ---------------------------------------------------------

/// Defines the network output for one instruction set from its steps: `us` and
/// `them` are the accumulators of the side to move and its opponent.
macro_rules! forward_fn {
    ($(#[$attr:meta])* $name:ident: $transform:ident, $layer0:ident, $activate:ident, $layer1:ident, $dot:ident) => {
        $(#[$attr])*
        unsafe fn $name(us: &Acc, them: &Acc, s: &LayerStack) -> i32 {
            // Feature transformer output: pairwise product of clipped halves,
            // collecting the non-zero 4-byte groups on the way.
            let mut ft = std::mem::MaybeUninit::<Ft>::uninit();
            let ft_ref = &mut *ft.as_mut_ptr();
            let mut nnz_uninit = std::mem::MaybeUninit::<Nnz>::uninit();
            let nnz = &mut *nnz_uninit.as_mut_ptr();
            let mut n_nnz = 0usize;
            let mut bits = [0u8; L1 / 32];
            for (p, acc) in [us, them].into_iter().enumerate() {
                $transform(
                    acc,
                    (&mut ft_ref.0[p * (L1 / 2)..(p + 1) * (L1 / 2)]).try_into().unwrap(),
                    (p * (L1 / 8)) as u16,
                    nnz,
                    &mut n_nnz,
                    &mut bits,
                );
            }
            let h0 = $layer0(s, &ft_ref.0, &nnz[..n_nnz], &bits);

            // Squared and linear clipped activations. Clamping first keeps the
            // square inside i32 without changing the (saturated) result.
            let mut act = Act([0; 2 * L2 + 2 * L3]);
            $activate::<21, 7>(&h0, &mut act.0[..2 * L2]);
            let h1 = $layer1(s, (&act.0[..2 * L2]).try_into().unwrap());
            $activate::<19, 6>(&h1, &mut act.0[2 * L2..]);

            // Output layer (sees both hidden layers) plus a skip connection.
            s.b2 + $dot(&act.0, &s.w2) + h0[L2 - 2] - h0[L2 - 1]
        }
    };
}

forward_fn!(
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "dotprod")]
    forward_neon_dot: transform_neon_bits, layer0_neon_dot, activate_neon, layer1_neon_dot, dot_neon_dot
);
forward_fn!(
    #[cfg(target_arch = "aarch64")]
    forward_neon: transform_neon_list, layer0_scalar, activate_neon, layer1_scalar, dot_scalar
);
forward_fn!(
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vnni")]
    forward_avx512_vnni: transform_avx512, layer0_avx512_vnni, activate_avx512, layer1_avx512_vnni, dot_avx512_vnni
);
forward_fn!(
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,avx512f,avx512bw")]
    forward_avx512: transform_avx512, layer0_avx512, activate_avx512, layer1_avx512, dot_avx512
);
forward_fn!(
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    forward_avx2: transform_avx2, layer0_avx2, activate_scalar, layer1_avx2, dot_avx2
);
forward_fn!(forward_scalar: transform_scalar, layer0_scalar, activate_scalar, layer1_scalar, dot_scalar);

/// Raw network output from the two accumulators, on the best code path this CPU
/// supports.
#[inline(always)]
fn forward(us: &Acc, them: &Acc, s: &LayerStack) -> i32 {
    let level = simd::level();
    // SAFETY: `level` only names instruction sets the CPU was found to have.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        match level {
            Level::NeonDot => return forward_neon_dot(us, them, s),
            Level::Neon => return forward_neon(us, them, s),
            _ => {}
        }
        #[cfg(target_arch = "x86_64")]
        match level {
            Level::Avx512Vnni => return forward_avx512_vnni(us, them, s),
            Level::Avx512 => return forward_avx512(us, them, s),
            Level::Avx2 => return forward_avx2(us, them, s),
            _ => {}
        }
        let _ = level;
        forward_scalar(us, them, s)
    }
}

/// Which layer stack evaluates this position.
#[inline]
fn layer_stack(pos: &Position) -> usize {
    let stm = pos.stm;
    let band = ((popcount(pos.occ()) - 1) / 4) as usize;
    let queens = 2 * (pos.pieces(stm, QUEEN) != 0) as usize + (pos.pieces(stm ^ 1, QUEEN) != 0) as usize;
    const QUEEN_SIDE: u64 = 0x0F0F_0F0F_0F0F_0F0F;
    let pawns = pos.by_type[PAWN];
    let layout = if pawns == 0 {
        2
    } else if pawns & QUEEN_SIDE != 0 && pawns & !QUEEN_SIDE != 0 {
        0
    } else {
        1
    };
    let officers = |c: usize| popcount(pos.by_color[c] & !(pos.by_type[PAWN] | pos.by_type[KING]));
    let many = (officers(WHITE).max(officers(BLACK)) >= 2) as usize;
    ((band * 4 + queens) * 3 + layout) * 2 + many
}

#[cfg(all(test, target_arch = "x86_64"))]
mod avx512_tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The AVX-512 dot product, second layer and activation give exactly the
    /// portable code's results. Skipped on CPUs without AVX-512.
    #[test]
    fn avx512_kernels_match_scalar() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx2")) {
            return;
        }
        let vnni = is_x86_feature_detected!("avx512vnni");
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut s: Box<LayerStack> = unsafe { Box::new(std::mem::zeroed()) };
        for round in 0..200 {
            for o in 0..L3 {
                s.b1[o] = (rng(&mut st) % 20001) as i32 - 10000;
                for i in 0..2 * L2 {
                    // Round 0 uses the extreme weights and activations.
                    s.w1[o][i] = if round == 0 { if (o + i) % 2 == 0 { -128 } else { 127 } } else { rng(&mut st) as i8 };
                    s.w1c[i / 4][o * 4 + i % 4] = s.w1[o][i];
                }
            }
            let mut act = [0u8; 2 * L2 + 2 * L3];
            let mut w2 = [0i8; 128];
            for i in 0..act.len() {
                act[i] = if round == 0 { 127 } else { (rng(&mut st) % 128) as u8 };
                w2[i] = if round == 0 { -128 } else { rng(&mut st) as i8 };
            }
            let mut h = [0i32; L2];
            for v in h.iter_mut() {
                *v = match rng(&mut st) % 4 {
                    0 => (rng(&mut st) % 2_000_001) as i32 - 1_000_000,
                    1 => (rng(&mut st) % 40001) as i32 - 20000,
                    2 => rng(&mut st) as i32,
                    _ => (rng(&mut st) % 257) as i32 - 128,
                };
            }
            let first: &[u8; 2 * L2] = (&act[..2 * L2]).try_into().unwrap();
            unsafe {
                let want_l1 = layer1_scalar(&s, first);
                let want_dot = dot_scalar(&act, &w2);
                assert_eq!(layer1_avx512(&s, first), want_l1);
                assert_eq!(dot_avx512(&act, &w2), want_dot);
                if vnni {
                    assert_eq!(layer1_avx512_vnni(&s, first), want_l1);
                    assert_eq!(dot_avx512_vnni(&act, &w2), want_dot);
                }
                let (mut a, mut b) = ([0u8; 2 * L2], [0u8; 2 * L2]);
                activate_scalar::<21, 7>(&h, &mut a);
                activate_avx512::<21, 7>(&h, &mut b);
                assert_eq!(a, b);
                activate_scalar::<19, 6>(&h, &mut a);
                activate_avx512::<19, 6>(&h, &mut b);
                assert_eq!(a, b);
            }
        }
    }
}

#[cfg(test)]
mod threat_tests {
    use super::*;
    use crate::movegen::legal_moves;

    fn cancelled(mut add: Vec<u32>, mut rem: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
        let mut i = 0;
        while i < add.len() {
            if let Some(j) = rem.iter().position(|&r| r == add[i]) {
                rem.swap_remove(j);
                add.swap_remove(i);
            } else {
                i += 1;
            }
        }
        add.sort_unstable();
        rem.sort_unstable();
        (add, rem)
    }

    /// The move-based threat changes must equal the difference of the two boards.
    #[test]
    fn incremental_threats_match_rescan() {
        crate::bitboard::init();
        let fens = [
            crate::position::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ];
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut checked = 0;
        for fen in fens {
            for _ in 0..40 {
                let mut pos = Position::from_fen(fen).unwrap();
                for _ in 0..60 {
                    let moves = legal_moves(&pos);
                    if moves.is_empty() {
                        break;
                    }
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    for &m in &moves {
                        let next = pos.make(m);
                        let (mut a1, mut r1) = (IdxList::<MAX_THREATS>::new(), IdxList::<MAX_THREATS>::new());
                        threat_changes(&pos, &next, &mut a1, &mut r1);
                        let (mut a2, mut r2) = (IdxList::<MAX_THREATS>::new(), IdxList::<MAX_THREATS>::new());
                        threat_changes_inc(&pos, &next, &mut a2, &mut r2);
                        let want = cancelled(a1.as_slice().to_vec(), r1.as_slice().to_vec());
                        let got = cancelled(a2.as_slice().to_vec(), r2.as_slice().to_vec());
                        assert_eq!(want, got, "{} {}", pos.to_fen(), m.to_uci());
                        checked += 1;
                    }
                    pos = pos.make(moves[(seed % moves.len() as u64) as usize]);
                }
            }
        }
        assert!(checked > 10_000);
    }
    /// Move pairs from random games, the kind of steps the search updates.
    fn sample_steps() -> Vec<(Position, Position)> {
        crate::bitboard::init();
        let fens = [
            crate::position::START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 9",
        ];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut steps = Vec::new();
        for fen in fens {
            for _ in 0..30 {
                let mut pos = Position::from_fen(fen).unwrap();
                for _ in 0..40 {
                    let moves = legal_moves(&pos);
                    if moves.is_empty() {
                        break;
                    }
                    for &m in &moves {
                        steps.push((pos, pos.make(m)));
                    }
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    pos = pos.make(moves[(seed % moves.len() as u64) as usize]);
                }
            }
        }
        steps
    }

    /// Time per call of the incremental threat diff (run with
    /// `cargo test --release threat_diff_bench -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn threat_diff_bench() {
        // A cache-sized sample: in the search both positions are hot.
        let mut steps = sample_steps();
        let stride = steps.len() / 4000;
        steps = steps.into_iter().step_by(stride.max(1)).collect();
        let mut total = 0usize;
        let t = std::time::Instant::now();
        let rounds: usize = std::env::var("TD_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(800);
        for _ in 0..rounds {
            for (p0, p1) in &steps {
                let (mut a, mut r) = (IdxList::<MAX_THREATS>::new(), IdxList::<MAX_THREATS>::new());
                threat_changes_inc(std::hint::black_box(p0), std::hint::black_box(p1), &mut a, &mut r);
                total += a.n + r.n;
                std::hint::black_box((&a, &r));
            }
        }
        let calls = (steps.len() * rounds) as f64;
        println!(
            "threat_changes_inc: {} steps, {:.1} ns/call, {:.2} threats/call",
            steps.len(),
            t.elapsed().as_nanos() as f64 / calls,
            total as f64 / calls
        );
    }
}
