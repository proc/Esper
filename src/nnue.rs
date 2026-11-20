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

use crate::bitboard::*;
use crate::position::*;
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
                LayerStack { w0: [[0; 4 * L2]; L1 / 4], b0: [0; L2], w1: [[0; 2 * L2]; L3], b1: [0; L3], w2: [0; 128], b2: 0 };
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
        let o = self.tp_off + idx as usize * L1;
        self.data[o..o + L1].try_into().unwrap()
    }
    #[inline(always)]
    fn psq_row(&self, idx: u32) -> &[u8; L1] {
        let o = self.psq_off + idx as usize * L1;
        self.data[o..o + L1].try_into().unwrap()
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

#[inline(always)]
fn prefetch_row(row: &[u8; L1]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let p = row.as_ptr();
        for k in 0..L1 / 64 {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.add(k * 64), options(nostack, preserves_flags, readonly));
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

    /// Piece types a piece of type `pt` can threaten (kings are never involved).
    #[inline(always)]
    fn targets(&self, pt: usize) -> u64 {
        let t = &self.by_type;
        match pt {
            PAWN => t[KNIGHT] | t[ROOK],
            BISHOP | ROOK => t[PAWN] | t[KNIGHT] | t[BISHOP] | t[ROOK],
            KNIGHT | QUEEN => t[PAWN] | t[KNIGHT] | t[BISHOP] | t[ROOK] | t[QUEEN],
            _ => 0,
        }
    }
}

#[inline(always)]
fn piece_attacks(pc: u8, sq: usize, occ: u64) -> u64 {
    match piece_type(pc) {
        PAWN => pawn_attacks(piece_color(pc), sq),
        KNIGHT => knight_attacks(sq),
        BISHOP => bishop_attacks(sq, occ),
        ROOK => rook_attacks(sq, occ),
        QUEEN => bishop_attacks(sq, occ) | rook_attacks(sq, occ),
        _ => 0,
    }
}

#[inline(always)]
fn threat_code(from: usize, to: usize, attacker: u8, attacked: u8) -> u32 {
    ((from as u32) << 14) | ((to as u32) << 8) | ((attacker as u32) << 4) | attacked as u32
}

/// Threats that involve the piece `pc` standing on `sq` in `tb` (as attacker or
/// target) go to `with`; threats that exist only while `sq` is empty (sliders
/// seeing past it) go to `without`.
#[inline]
fn piece_threats(tb: &ThreatBoard, pc: u8, sq: usize, with: &mut IdxList<MAX_THREATS>, without: &mut IdxList<MAX_THREATS>) {
    let pt = piece_type(pc);
    let occ = tb.occ;
    let t = &tb.by_type;
    // What the piece attacks.
    let mut att = piece_attacks(pc, sq, occ) & tb.targets(pt);
    while att != 0 {
        let to = pop_lsb(&mut att);
        with.push(threat_code(sq, to, pc, tb.board[to]));
    }
    if pt == KING {
        // Kings neither attack nor get attacked, but they do block lines.
    } else {
        // What attacks the piece.
        let diag = bishop_attacks(sq, occ);
        let ortho = rook_attacks(sq, occ);
        let mut attackers = (knight_attacks(sq) & t[KNIGHT]) | ((diag | ortho) & t[QUEEN]);
        if pt != QUEEN {
            attackers |= (diag & t[BISHOP]) | (ortho & t[ROOK]);
        }
        if pt == KNIGHT || pt == ROOK {
            let white = tb.by_type[PAWN] & tb.by_color[WHITE];
            let black = tb.by_type[PAWN] & tb.by_color[BLACK];
            attackers |= (pawn_attacks(BLACK, sq) & white) | (pawn_attacks(WHITE, sq) & black);
        }
        while attackers != 0 {
            let from = pop_lsb(&mut attackers);
            with.push(threat_code(from, sq, tb.board[from], pc));
        }
    }
    // Sliders whose line runs through `sq` see one piece further when it is empty.
    let mut sliders = (bishop_attacks(sq, occ) & (t[BISHOP] | t[QUEEN])) | (rook_attacks(sq, occ) & (t[ROOK] | t[QUEEN]));
    let open = occ ^ bit(sq);
    while sliders != 0 {
        let from = pop_lsb(&mut sliders);
        let spc = tb.board[from];
        let mut beyond = piece_attacks(spc, from, open) & !piece_attacks(spc, from, occ) & open & tb.targets(piece_type(spc));
        while beyond != 0 {
            let to = pop_lsb(&mut beyond);
            without.push(threat_code(from, to, spc, tb.board[to]));
        }
    }
}

/// Threat changes from `p0` to `p1`, worked out from the pieces that left and
/// arrived rather than by rescanning both boards. A threat can appear in both
/// lists; the accumulator arithmetic is linear, so that is harmless.
fn threat_changes_inc(p0: &Position, p1: &Position, added: &mut IdxList<MAX_THREATS>, removed: &mut IdxList<MAX_THREATS>) {
    let mut tb = ThreatBoard::new(p0);
    let mut changed = (p0.by_color[0] ^ p1.by_color[0]) | (p0.by_color[1] ^ p1.by_color[1]);
    for pt in PAWN..=KING {
        changed |= p0.by_type[pt] ^ p1.by_type[pt];
    }
    // Lift the pieces that leave or are replaced...
    let mut c = changed;
    while c != 0 {
        let sq = pop_lsb(&mut c);
        let pc = tb.board[sq];
        if pc != 0 {
            piece_threats(&tb, pc, sq, removed, added);
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
            piece_threats(&tb, pc, sq, added, removed);
        }
    }
    // Drop threats that were both lost and regained: each costs two row updates.
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
    fn pair_indices<const N: usize>(
        &self,
        persp: usize,
        orient: usize,
        updated: u64,
        white: u64,
        black: u64,
        out: &mut IdxList<N>,
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

#[cfg(simd_neon)]
#[inline]
fn apply_rows<'a, F: Fn(u32) -> &'a [u8; L1]>(dst: &mut Acc, src: &Acc, row: F, adds: &[u32], subs: &[u32]) {
    use std::arch::aarch64::*;
    if adds.is_empty() && subs.is_empty() {
        *dst = *src;
        return;
    }
    // Fast fused paths for the common 1-add/1-sub (quiet) and 1-add/2-sub (capture) updates.
    if adds.len() == 1 && subs.len() == 1 {
        let a0 = row(adds[0]).as_ptr() as *const i8;
        let s0 = row(subs[0]).as_ptr() as *const i8;
        let src_p = src.0.as_ptr();
        let dst_p = dst.0.as_mut_ptr();
        unsafe {
            for off in (0..L1).step_by(16) {
                let va = vld1q_s8(a0.add(off));
                let vs = vld1q_s8(s0.add(off));
                let r0 = vsubw_s8(vaddw_s8(vld1q_s16(src_p.add(off)), vget_low_s8(va)), vget_low_s8(vs));
                let r1 = vsubw_high_s8(vaddw_high_s8(vld1q_s16(src_p.add(off + 8)), va), vs);
                vst1q_s16(dst_p.add(off), r0);
                vst1q_s16(dst_p.add(off + 8), r1);
            }
        }
        return;
    }
    if adds.len() == 1 && subs.len() == 2 {
        let a0 = row(adds[0]).as_ptr() as *const i8;
        let s0 = row(subs[0]).as_ptr() as *const i8;
        let s1 = row(subs[1]).as_ptr() as *const i8;
        let src_p = src.0.as_ptr();
        let dst_p = dst.0.as_mut_ptr();
        unsafe {
            for off in (0..L1).step_by(16) {
                let va = vld1q_s8(a0.add(off));
                let vs0 = vld1q_s8(s0.add(off));
                let vs1 = vld1q_s8(s1.add(off));
                let r0 = vsubw_s8(
                    vsubw_s8(vaddw_s8(vld1q_s16(src_p.add(off)), vget_low_s8(va)), vget_low_s8(vs0)),
                    vget_low_s8(vs1),
                );
                let r1 = vsubw_high_s8(
                    vsubw_high_s8(vaddw_high_s8(vld1q_s16(src_p.add(off + 8)), va), vs0),
                    vs1,
                );
                vst1q_s16(dst_p.add(off), r0);
                vst1q_s16(dst_p.add(off + 8), r1);
            }
        }
        return;
    }
    const TILE: usize = 128; // 16 registers of 8 lanes
    // SAFETY: all pointers stay within the 1024-element arrays.
    unsafe {
        let mut off = 0;
        while off < L1 {
            let mut regs = [vdupq_n_s16(0); TILE / 8];
            let s = src.0.as_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = vld1q_s16(s.add(k * 8));
            }
            for &i in adds {
                let w = row(i).as_ptr().add(off) as *const i8;
                for k in 0..TILE / 16 {
                    let v = vld1q_s8(w.add(k * 16));
                    regs[2 * k] = vaddw_s8(regs[2 * k], vget_low_s8(v));
                    regs[2 * k + 1] = vaddw_high_s8(regs[2 * k + 1], v);
                }
            }
            for &i in subs {
                let w = row(i).as_ptr().add(off) as *const i8;
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
            off += TILE;
        }
    }
}

#[cfg(simd_avx2)]
#[inline]
fn apply_rows<'a, F: Fn(u32) -> &'a [u8; L1]>(dst: &mut Acc, src: &Acc, row: F, adds: &[u32], subs: &[u32]) {
    use std::arch::x86_64::*;
    if adds.is_empty() && subs.is_empty() {
        *dst = *src;
        return;
    }
    if adds.len() == 1 && subs.len() == 1 {
        let a0 = row(adds[0]).as_ptr();
        let s0 = row(subs[0]).as_ptr();
        let src_p = src.0.as_ptr();
        let dst_p = dst.0.as_mut_ptr();
        unsafe {
            for off in (0..L1).step_by(16) {
                let r = _mm256_loadu_si256(src_p.add(off) as *const __m256i);
                let va = _mm256_cvtepi8_epi16(_mm_loadu_si128(a0.add(off) as *const __m128i));
                let vs = _mm256_cvtepi8_epi16(_mm_loadu_si128(s0.add(off) as *const __m128i));
                _mm256_storeu_si256(dst_p.add(off) as *mut __m256i, _mm256_sub_epi16(_mm256_add_epi16(r, va), vs));
            }
        }
        return;
    }
    if adds.len() == 1 && subs.len() == 2 {
        let a0 = row(adds[0]).as_ptr();
        let s0 = row(subs[0]).as_ptr();
        let s1 = row(subs[1]).as_ptr();
        let src_p = src.0.as_ptr();
        let dst_p = dst.0.as_mut_ptr();
        unsafe {
            for off in (0..L1).step_by(16) {
                let r = _mm256_loadu_si256(src_p.add(off) as *const __m256i);
                let va = _mm256_cvtepi8_epi16(_mm_loadu_si128(a0.add(off) as *const __m128i));
                let vs0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(s0.add(off) as *const __m128i));
                let vs1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(s1.add(off) as *const __m128i));
                let res = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(r, va), vs0), vs1);
                _mm256_storeu_si256(dst_p.add(off) as *mut __m256i, res);
            }
        }
        return;
    }
    const REGS: usize = 8; // of 16 lanes each
    const TILE: usize = REGS * 16;
    // SAFETY: all pointers stay within the 1024-element arrays.
    unsafe {
        let mut off = 0;
        while off < L1 {
            let mut regs = [_mm256_setzero_si256(); REGS];
            let s = src.0.as_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = _mm256_loadu_si256(s.add(k * 16) as *const __m256i);
            }
            for &i in adds {
                let w = row(i).as_ptr().add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(k * 16) as *const __m128i));
                    *r = _mm256_add_epi16(*r, v);
                }
            }
            for &i in subs {
                let w = row(i).as_ptr().add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(k * 16) as *const __m128i));
                    *r = _mm256_sub_epi16(*r, v);
                }
            }
            let d = dst.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter().enumerate() {
                _mm256_storeu_si256(d.add(k * 16) as *mut __m256i, *r);
            }
            off += TILE;
        }
    }
}

#[cfg(simd_avx512)]
#[inline]
fn apply_rows<'a, F: Fn(u32) -> &'a [u8; L1]>(dst: &mut Acc, src: &Acc, row: F, adds: &[u32], subs: &[u32]) {
    use std::arch::x86_64::*;
    const REGS: usize = 16; // of 32 lanes each
    const TILE: usize = REGS * 32;
    // SAFETY: all pointers stay within the 1024-element arrays.
    unsafe {
        let mut off = 0;
        while off < L1 {
            let mut regs = [_mm512_setzero_si512(); REGS];
            let s = src.0.as_ptr().add(off);
            for (k, r) in regs.iter_mut().enumerate() {
                *r = _mm512_loadu_si512(s.add(k * 32) as *const _);
            }
            for &i in adds {
                let w = row(i).as_ptr().add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm512_cvtepi8_epi16(_mm256_loadu_si256(w.add(k * 32) as *const __m256i));
                    *r = _mm512_add_epi16(*r, v);
                }
            }
            for &i in subs {
                let w = row(i).as_ptr().add(off);
                for (k, r) in regs.iter_mut().enumerate() {
                    let v = _mm512_cvtepi8_epi16(_mm256_loadu_si256(w.add(k * 32) as *const __m256i));
                    *r = _mm512_sub_epi16(*r, v);
                }
            }
            let d = dst.0.as_mut_ptr().add(off);
            for (k, r) in regs.iter().enumerate() {
                _mm512_storeu_si512(d.add(k * 32) as *mut _, *r);
            }
            off += TILE;
        }
    }
}

#[cfg(simd_none)]
#[inline]
fn apply_rows<'a, F: Fn(u32) -> &'a [u8; L1]>(dst: &mut Acc, src: &Acc, row: F, adds: &[u32], subs: &[u32]) {
    const TILE: usize = 256;
    let mut off = 0;
    while off < L1 {
        let mut regs = [0i16; TILE];
        regs.copy_from_slice(&src.0[off..off + TILE]);
        for &i in adds {
            let w = &row(i)[off..off + TILE];
            for k in 0..TILE {
                regs[k] = regs[k].wrapping_add(w[k] as i8 as i16);
            }
        }
        for &i in subs {
            let w = &row(i)[off..off + TILE];
            for k in 0..TILE {
                regs[k] = regs[k].wrapping_sub(w[k] as i8 as i16);
            }
        }
        dst.0[off..off + TILE].copy_from_slice(&regs);
        off += TILE;
    }
}

// ---------------------------------------------------------------------------
// Per-thread evaluation state
// ---------------------------------------------------------------------------

/// Largest possible change lists: up to 256 threats plus 120 pawn pairs.
const TP_LIST: usize = 512;

struct Entry {
    psq: [Acc; 2],
    tp: [Acc; 2],
    psq_ok: [bool; 2],
    tp_ok: [bool; 2],
    /// The step from the previous ply cannot be done incrementally.
    psq_refresh: [bool; 2],
    tp_refresh: [bool; 2],
    threats_ok: bool,
    psq_diff_ok: [bool; 2],
    tp_diff_ok: bool,
    n_threats: usize,
    threats: [u32; MAX_THREATS],
    // Feature rows to add / subtract relative to the previous ply.
    psq_adds: [IdxList<8>; 2],
    psq_subs: [IdxList<8>; 2],
    tp_adds: [IdxList<TP_LIST>; 2],
    tp_subs: [IdxList<TP_LIST>; 2],
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
fn tp_rows(
    net: &Network,
    p: usize,
    orient: usize,
    added: &IdxList<MAX_THREATS>,
    removed: &IdxList<MAX_THREATS>,
    (w0, b0): (u64, u64),
    (w1, b1): (u64, u64),
    adds: &mut IdxList<TP_LIST>,
    subs: &mut IdxList<TP_LIST>,
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

/// Compute a move's feature changes when it is made rather than when the
/// position is evaluated. Off: many made positions are never evaluated, and the
/// wasted work cost 6% in cycles per node despite the earlier prefetch.
const EAGER_DIFF: bool = false;

/// Threat-feature rows that change from `p0` to `p1`, for both perspectives
/// (those not due a refresh).
#[inline]
fn compute_tp_diff(net: &Network, cur: &mut Entry, p0: &Position, p1: &Position) {
    let mut added = IdxList::<MAX_THREATS>::new();
    let mut removed = IdxList::<MAX_THREATS>::new();
    threat_changes_inc(p0, p1, &mut added, &mut removed);
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
    let k1 = p1.king_sq(persp);
    let block = psq_block(p1, persp);
    let (adds, subs) = (&mut cur.psq_adds[persp], &mut cur.psq_subs[persp]);
    adds.clear();
    subs.clear();
    psq_diff(persp, k1, block, (&p0.by_color, &p0.by_type), (&p1.by_color, &p1.by_type), adds, subs);
    cur.psq_diff_ok[persp] = true;
}

impl NnueState {
    pub fn new(net: Arc<Network>, max_ply: usize) -> NnueState {
        let mut entries = Vec::with_capacity(max_ply);
        for _ in 0..max_ply {
            entries.push(Entry {
                psq: [Acc([0; L1]); 2],
                tp: [Acc([0; L1]); 2],
                psq_ok: [false; 2],
                tp_ok: [false; 2],
                psq_refresh: [true; 2],
                tp_refresh: [true; 2],
                threats_ok: false,
                psq_diff_ok: [false; 2],
                tp_diff_ok: false,
                n_threats: 0,
                threats: [0; MAX_THREATS],
                psq_adds: [IdxList::new(), IdxList::new()],
                psq_subs: [IdxList::new(), IdxList::new()],
                tp_adds: [IdxList::new(), IdxList::new()],
                tp_subs: [IdxList::new(), IdxList::new()],
            });
        }
        let mut cache = Vec::with_capacity(2 * 3 * 64);
        for _ in 0..2 * 3 * 64 {
            cache.push(CacheEntry { acc: net.biases, by_color: [0; 2], by_type: [0; 7] });
        }
        NnueState { net, entries, cache, tp_cache: vec![None; 4] }
    }

    /// Start a new stack: the position at ply 0 is unrelated to anything before.
    pub fn reset(&mut self) {
        let e = &mut self.entries[0];
        e.psq_ok = [false; 2];
        e.tp_ok = [false; 2];
        e.psq_refresh = [true; 2];
        e.tp_refresh = [true; 2];
        e.threats_ok = false;
        e.psq_diff_ok = [false; 2];
        e.tp_diff_ok = false;
    }

    /// Register the position at `stack[j]` (reached from `stack[j - 1]`): mark
    /// accumulator/threat state as pending so diffs are only computed if evaluated.
    pub fn push(&mut self, stack: &[Position], j: usize) {
        let cur = &mut self.entries[j];
        let (p0, p1) = (&stack[j - 1], &stack[j]);
        cur.psq_ok = [false; 2];
        cur.tp_ok = [false; 2];
        cur.threats_ok = false;
        cur.psq_diff_ok = [false; 2];
        cur.tp_diff_ok = false;
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
                        prefetch_row(net.tp_row(i));
                    }
                }
            }
        }
    }

    #[inline(never)]
    fn refresh_psq(&mut self, pos: &Position, ply: usize, persp: usize) {
        let net = &*self.net;
        let ksq = pos.king_sq(persp);
        let block = psq_block(pos, persp);
        let c = &mut self.cache[(persp * 3 + block) * 64 + ksq];
        let (mut adds, mut subs) = (IdxList::<64>::new(), IdxList::<64>::new());
        psq_diff(persp, ksq, block, (&c.by_color, &c.by_type), (&pos.by_color, &pos.by_type), &mut adds, &mut subs);
        if adds.n + subs.n > 0 {
            for &i in adds.as_slice().iter().chain(subs.as_slice()) {
                prefetch_row(net.psq_row(i));
            }
            let src = c.acc;
            apply_rows(&mut c.acc, &src, |i| net.psq_row(i), adds.as_slice(), subs.as_slice());
            c.by_color = pos.by_color;
            c.by_type = pos.by_type;
        }
        let e = &mut self.entries[ply];
        e.psq[persp] = c.acc;
        e.psq_ok[persp] = true;
    }

    #[inline(never)]
    fn ensure_psq(&mut self, stack: &[Position], ply: usize, persp: usize) {
        let mut k = ply;
        while !self.entries[k].psq_ok[persp] {
            if k == 0 || self.entries[k].psq_refresh[persp] {
                return self.refresh_psq(&stack[ply], ply, persp);
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
            for &i in cur.psq_adds[persp].as_slice().iter().chain(cur.psq_subs[persp].as_slice()) {
                prefetch_row(net.psq_row(i));
            }
            apply_rows(
                &mut cur.psq[persp],
                &prev.psq[persp],
                |i| net.psq_row(i),
                cur.psq_adds[persp].as_slice(),
                cur.psq_subs[persp].as_slice(),
            );
            cur.psq_ok[persp] = true;
        }
    }

    #[inline(never)]
    fn refresh_tp(&mut self, pos: &Position, ply: usize, persp: usize) {
        let net = &*self.net;
        let ksq = pos.king_sq(persp);
        let orient = orientation(persp, ksq);
        let slot = persp * 2 + ((ksq >> 2) & 1);
        // Start from the cached accumulator of the same orientation when the
        // boards are close: only the threat changes between them need applying.
        if let Some((acc, cpos)) = &self.tp_cache[slot] {
            let mut changed = (cpos.by_color[0] ^ pos.by_color[0]) | (cpos.by_color[1] ^ pos.by_color[1]);
            for pt in PAWN..=KING {
                changed |= cpos.by_type[pt] ^ pos.by_type[pt];
            }
            if popcount(changed) <= TP_CACHE_MAX_CHANGED {
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
                let src = *acc;
                let e = &mut self.entries[ply];
                apply_rows(&mut e.tp[persp], &src, |i| net.tp_row(i), adds.as_slice(), subs.as_slice());
                e.tp_ok[persp] = true;
                self.tp_cache[slot] = Some((e.tp[persp], *pos));
                return;
            }
        }
        let e = &mut self.entries[ply];
        if !e.threats_ok {
            e.n_threats = gen_threats(pos, &mut e.threats);
            e.threats_ok = true;
        }
        let mut adds = IdxList::<TP_LIST>::new();
        for &t in &e.threats[..e.n_threats] {
            let idx = net.luts.threat_index(t, persp, orient);
            adds.push_if(idx, (idx as usize) < THREAT_DIMS);
        }
        let (w, b) = (pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN));
        net.luts.pair_indices(persp, orient, w | b, w, b, &mut adds);
        for &i in adds.as_slice() {
            prefetch_row(net.tp_row(i));
        }
        let zero = Acc([0; L1]);
        apply_rows(&mut e.tp[persp], &zero, |i| net.tp_row(i), adds.as_slice(), &[]);
        e.tp_ok[persp] = true;
        self.tp_cache[slot] = Some((e.tp[persp], *pos));
    }

    #[inline(never)]
    fn ensure_tp(&mut self, stack: &[Position], ply: usize, persp: usize) {
        let mut k = ply;
        while !self.entries[k].tp_ok[persp] {
            if k == 0 || self.entries[k].tp_refresh[persp] {
                return self.refresh_tp(&stack[ply], ply, persp);
            }
            k -= 1;
        }
        let net = &*self.net;
        for j in k + 1..=ply {
            let (lo, hi) = self.entries.split_at_mut(j);
            let (prev, cur) = (&mut lo[j - 1], &mut hi[0]);
            if !cur.tp_diff_ok {
                compute_tp_diff(net, cur, &stack[j - 1], &stack[j]);
            }
            for &i in cur.tp_adds[persp].as_slice().iter().chain(cur.tp_subs[persp].as_slice()) {
                prefetch_row(net.tp_row(i));
            }
            apply_rows(
                &mut cur.tp[persp],
                &prev.tp[persp],
                |i| net.tp_row(i),
                cur.tp_adds[persp].as_slice(),
                cur.tp_subs[persp].as_slice(),
            );
            cur.tp_ok[persp] = true;
        }
    }

    /// Raw network output for the position at `stack[ply]`, from the side to move's
    /// point of view. `stack[0..=ply]` must be the path from the stack's root, with
    /// every ply after the root registered through `push`.
    pub fn evaluate(&mut self, stack: &[Position], ply: usize) -> i32 {
        for persp in 0..2 {
            self.ensure_psq(stack, ply, persp);
            self.ensure_tp(stack, ply, persp);
        }
        let pos = &stack[ply];
        let e = &self.entries[ply];
        let net = &*self.net;

        // Feature transformer output: pairwise product of clipped halves, collecting
        // non-zero 4-byte group indices on the fly while SIMD registers are live.
        #[repr(C, align(64))]
        struct Ft([u8; L1]);
        let mut ft = std::mem::MaybeUninit::<Ft>::uninit();
        let ft_ref = unsafe { &mut *ft.as_mut_ptr() };
        let mut nnz = [0u16; L1 / 4 + 8];
        let mut n_nnz = 0usize;
        for p in 0..2 {
            let c = pos.stm ^ p;
            transform(
                &e.psq[c],
                &e.tp[c],
                (&mut ft_ref.0[p * (L1 / 2)..(p + 1) * (L1 / 2)]).try_into().unwrap(),
                (p * (L1 / 8)) as u16,
                &mut nnz,
                &mut n_nnz,
            );
        }

        let s = &net.stacks[net.stack_index[layer_stack(pos)] as usize];
        let h0 = layer0(s, &ft_ref.0, &nnz[..n_nnz]);

        // Squared and linear clipped activations. Clamping first keeps the square
        // inside i32 without changing the (saturated) result.
        #[repr(C, align(64))]
        struct Act([u8; 2 * L2 + 2 * L3]);
        let mut act = Act([0; 2 * L2 + 2 * L3]);
        for i in 0..L2 {
            let c = h0[i].clamp(-16384, 16384);
            act.0[i] = ((c * c) >> 21).min(127) as u8;
            act.0[L2 + i] = (h0[i] >> 7).clamp(0, 127) as u8;
        }

        // Layer 1
        let mut h1 = s.b1;
        for o in 0..L3 {
            h1[o] += dot_u8_i8(&act.0[..2 * L2], &s.w1[o]);
        }
        for i in 0..L3 {
            let c = h1[i].clamp(-16384, 16384);
            act.0[2 * L2 + i] = ((c * c) >> 19).min(127) as u8;
            act.0[2 * L2 + L3 + i] = (h1[i] >> 6).clamp(0, 127) as u8;
        }

        // Output layer (sees both hidden layers) plus a skip connection.
        let out = s.b2 + dot_u8_i8(&act.0, &s.w2) + h0[L2 - 2] - h0[L2 - 1];
        let scaled = (out as i64 * (600 * 16) / (128 * 64 * 2)) as i32;
        scaled / 16
    }
}

/// Append up to 8 non-zero group indices described by `mask` starting at `base`.
#[inline(always)]
fn push_nnz8(mask: usize, base: u16, nnz: &mut [u16; L1 / 4 + 8], n: &mut usize) {
    let lut = &NNZ_LUT[mask];
    let dst = unsafe { nnz.as_mut_ptr().add(*n) };
    for k in 0..8 {
        unsafe { *dst.add(k) = base + *lut.get_unchecked(k) };
    }
    *n += mask.count_ones() as usize;
}

/// out[j] = clamp(a[j], 0, 255) * clamp(a[j + 512], 0, 255) / 512, where `a` is the
/// sum of the two accumulator components, while recording non-zero 4-byte groups.
#[inline(always)]
fn transform(
    psq: &Acc,
    tp: &Acc,
    out: &mut [u8; L1 / 2],
    base_group: u16,
    nnz: &mut [u16; L1 / 4 + 8],
    n_nnz: &mut usize,
) {
    const H: usize = L1 / 2;
    #[cfg(simd_neon)]
    unsafe {
        use std::arch::aarch64::*;
        const BITS: [u32; 4] = [1, 2, 4, 8];
        let bits = vld1q_u32(BITS.as_ptr());
        let (a, b) = (psq.0.as_ptr(), tp.0.as_ptr());
        let step8 = |j: usize| {
            let x = vaddq_s16(vld1q_s16(a.add(j)), vld1q_s16(b.add(j)));
            let y = vaddq_s16(vld1q_s16(a.add(j + H)), vld1q_s16(b.add(j + H)));
            let prod = vmull_u8(vqmovun_s16(x), vqmovun_s16(y));
            vshrn_n_u16::<8>(vshrq_n_u16::<1>(prod))
        };
        for j in (0..H).step_by(32) {
            let q0 = vcombine_u8(step8(j), step8(j + 8));
            let q1 = vcombine_u8(step8(j + 16), step8(j + 24));
            vst1q_u8(out.as_mut_ptr().add(j), q0);
            vst1q_u8(out.as_mut_ptr().add(j + 16), q1);
            let u0 = vreinterpretq_u32_u8(q0);
            let u1 = vreinterpretq_u32_u8(q1);
            let lo = vaddvq_u32(vandq_u32(vtstq_u32(u0, u0), bits));
            let hi = vaddvq_u32(vandq_u32(vtstq_u32(u1, u1), bits));
            push_nnz8((lo | (hi << 4)) as usize, base_group + (j / 4) as u16, nnz, n_nnz);
        }
    }
    #[cfg(simd_avx2)]
    unsafe {
        use std::arch::x86_64::*;
        let (a, b) = (psq.0.as_ptr(), tp.0.as_ptr());
        let zero = _mm256_setzero_si256();
        let max = _mm256_set1_epi16(255);
        let ld = |p: *const i16| _mm256_loadu_si256(p as *const __m256i);
        let step16 = |j: usize| {
            let x = _mm256_add_epi16(ld(a.add(j)), ld(b.add(j)));
            let y = _mm256_add_epi16(ld(a.add(j + H)), ld(b.add(j + H)));
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
    #[cfg(simd_avx512)]
    unsafe {
        use std::arch::x86_64::*;
        let (a, b) = (psq.0.as_ptr(), tp.0.as_ptr());
        let zero = _mm512_setzero_si512();
        let zero256 = _mm256_setzero_si256();
        let max = _mm512_set1_epi16(255);
        let ld = |p: *const i16| _mm512_loadu_si512(p as *const _);
        for j in (0..H).step_by(32) {
            let x = _mm512_add_epi16(ld(a.add(j)), ld(b.add(j)));
            let y = _mm512_add_epi16(ld(a.add(j + H)), ld(b.add(j + H)));
            let x = _mm512_min_epi16(_mm512_max_epi16(x, zero), max);
            let y = _mm512_min_epi16(_mm512_max_epi16(y, zero), max);
            let p = _mm512_srli_epi16::<9>(_mm512_mullo_epi16(x, y));
            let v = _mm512_cvtepi16_epi8(p);
            _mm256_storeu_si256(out.as_mut_ptr().add(j) as *mut __m256i, v);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, zero256))) as usize;
            push_nnz8(mask, base_group + (j / 4) as u16, nnz, n_nnz);
        }
    }
    #[cfg(simd_none)]
    {
        let (a, b) = (&psq.0, &tp.0);
        for j in (0..H).step_by(4) {
            let mut any = 0u8;
            for k in 0..4 {
                let x = a[j + k].wrapping_add(b[j + k]).clamp(0, 255) as u32;
                let y = a[j + k + H].wrapping_add(b[j + k + H]).clamp(0, 255) as u32;
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
}

/// For every 8-bit mask, the positions of its set bits (padded with zeros).
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

/// Dot product of unsigned activations (at most 127) with signed weights; the
/// length is a multiple of 64.
#[inline(always)]
fn dot_u8_i8(a: &[u8], w: &[i8]) -> i32 {
    debug_assert!(a.len() == w.len() && a.len() % 64 == 0);
    #[cfg(simd_neon_dot)]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = vdupq_n_s32(0);
        for i in (0..a.len()).step_by(16) {
            acc = vdotq_s32(acc, vld1q_s8(w.as_ptr().add(i)), vreinterpretq_s8_u8(vld1q_u8(a.as_ptr().add(i))));
        }
        vaddvq_s32(acc)
    }
    #[cfg(simd_x86)]
    unsafe {
        use std::arch::x86_64::*;
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
    #[cfg(not(any(simd_neon_dot, simd_x86)))]
    {
        let mut sum = 0i32;
        for i in 0..a.len() {
            sum += a[i] as i32 * w[i] as i32;
        }
        sum
    }
}

/// First (sparse-input) affine layer: only groups of four inputs that are not all
/// zero are visited.
#[inline(never)]
fn layer0(s: &LayerStack, ft: &[u8; L1], nnz: &[u16]) -> [i32; L2] {
    let group = |c: usize| unsafe { std::ptr::read_unaligned(ft.as_ptr().add(c * 4) as *const u32) };

    #[cfg(simd_neon_dot)]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = [vdupq_n_s32(0); L2 / 4];
        for (k, a) in acc.iter_mut().enumerate() {
            *a = vld1q_s32(s.b0.as_ptr().add(k * 4));
        }
        for &c in nnz {
            let c = c as usize;
            let input = vreinterpretq_s8_u32(vdupq_n_u32(group(c)));
            let w = s.w0.get_unchecked(c).as_ptr();
            for (k, a) in acc.iter_mut().enumerate() {
                *a = vdotq_s32(*a, vld1q_s8(w.add(k * 16)), input);
            }
        }
        let mut h0 = [0i32; L2];
        for (k, a) in acc.iter().enumerate() {
            vst1q_s32(h0.as_mut_ptr().add(k * 4), *a);
        }
        h0
    }
    #[cfg(simd_avx2)]
    unsafe {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); L2 / 8];
        for (k, a) in acc.iter_mut().enumerate() {
            *a = _mm256_loadu_si256(s.b0.as_ptr().add(k * 8) as *const __m256i);
        }
        for &c in nnz {
            let c = c as usize;
            let input = _mm256_set1_epi32(group(c) as i32);
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
    #[cfg(simd_avx512)]
    unsafe {
        use std::arch::x86_64::*;
        let mut acc = [_mm512_setzero_si512(); L2 / 16];
        for (k, a) in acc.iter_mut().enumerate() {
            *a = _mm512_loadu_si512(s.b0.as_ptr().add(k * 16) as *const _);
        }
        #[cfg(not(simd_vnni))]
        let ones = _mm512_set1_epi16(1);
        for &c in nnz {
            let c = c as usize;
            let input = _mm512_set1_epi32(group(c) as i32);
            let w = s.w0.get_unchecked(c).as_ptr();
            for (k, a) in acc.iter_mut().enumerate() {
                let wv = _mm512_loadu_si512(w.add(k * 64) as *const _);
                #[cfg(simd_vnni)]
                {
                    *a = _mm512_dpbusd_epi32(*a, input, wv);
                }
                #[cfg(not(simd_vnni))]
                {
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
    #[cfg(not(any(simd_neon_dot, simd_x86)))]
    {
        let mut h0 = s.b0;
        for &c in nnz {
            let c = c as usize;
            let x = group(c).to_ne_bytes().map(|b| b as i32);
            let w = &s.w0[c];
            for o in 0..L2 {
                h0[o] += x[0] * w[o * 4] as i32
                    + x[1] * w[o * 4 + 1] as i32
                    + x[2] * w[o * 4 + 2] as i32
                    + x[3] * w[o * 4 + 3] as i32;
            }
        }
        h0
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
}
