//! Syzygy endgame tablebase access through the vendored Fathom library.

use crate::bitboard::popcount;
use crate::movegen::legal_moves;
use crate::position::*;
use std::ffi::CString;
use std::os::raw::{c_char, c_uint};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

extern "C" {
    fn tb_init(path: *const c_char) -> bool;
    #[allow(clippy::too_many_arguments)]
    fn tb_probe_wdl_impl(
        white: u64,
        black: u64,
        kings: u64,
        queens: u64,
        rooks: u64,
        bishops: u64,
        knights: u64,
        pawns: u64,
        ep: c_uint,
        turn: bool,
    ) -> c_uint;
    #[allow(clippy::too_many_arguments)]
    fn tb_probe_root_impl(
        white: u64,
        black: u64,
        kings: u64,
        queens: u64,
        rooks: u64,
        bishops: u64,
        knights: u64,
        pawns: u64,
        rule50: c_uint,
        ep: c_uint,
        turn: bool,
        results: *mut c_uint,
    ) -> c_uint;
    static TB_LARGEST: c_uint;
}

const RESULT_FAILED: u32 = 0xFFFF_FFFF;
/// Largest piece count covered by the loaded tables (0 = none loaded).
static LARGEST: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy)]
pub struct TbConfig {
    /// Minimum remaining depth for probing inside the search.
    pub probe_depth: i32,
    /// Treat wins / losses that the 50-move rule turns into draws as draws.
    pub rule50: bool,
    /// Only probe positions with at most this many pieces.
    pub limit: u32,
}

impl TbConfig {
    pub const DEFAULT: TbConfig = TbConfig { probe_depth: 1, rule50: true, limit: 7 };
    /// Piece count up to which probing is possible (0 = disabled).
    pub fn cardinality(&self) -> u32 {
        LARGEST.load(Relaxed).min(self.limit)
    }
}

/// Load tables from `path` (directories separated by ':' or, on Windows, ';').
/// An empty path unloads them. Must not be called while a search is running.
/// Returns the largest piece count found.
pub fn init(path: &str) -> u32 {
    let path = if path == "<empty>" { "" } else { path };
    let Ok(c) = CString::new(path) else { return LARGEST.load(Relaxed) };
    // SAFETY: Fathom copies the string; no probe is in flight (see above).
    let ok = unsafe { tb_init(c.as_ptr()) };
    let largest = if ok && !path.is_empty() { unsafe { TB_LARGEST } } else { 0 };
    LARGEST.store(largest, Relaxed);
    largest
}

#[inline(always)]
fn ep_square(pos: &Position) -> c_uint {
    if pos.ep == NO_SQ {
        0
    } else {
        pos.ep as c_uint
    }
}

/// Outcome for the side to move: -2 loss, -1 blessed loss (drawn by the 50-move
/// rule), 0 draw, 1 cursed win, 2 win.
pub type Wdl = i32;

/// Collapse the 50-move-rule cases according to the configuration: -1, 0 or 1.
#[inline]
pub fn outcome(wdl: Wdl, cfg: &TbConfig) -> i32 {
    if cfg.rule50 {
        wdl / 2
    } else {
        wdl.signum()
    }
}

/// Win/draw/loss probe for use inside the search (thread safe). The position must
/// have no castling rights and a zero half-move clock.
#[inline]
pub fn probe_wdl(pos: &Position) -> Option<Wdl> {
    debug_assert!(pos.castling == 0 && pos.rule50 == 0);
    // SAFETY: plain value arguments; Fathom's WDL probe is thread safe.
    let r = unsafe {
        tb_probe_wdl_impl(
            pos.by_color[WHITE],
            pos.by_color[BLACK],
            pos.by_type[KING],
            pos.by_type[QUEEN],
            pos.by_type[ROOK],
            pos.by_type[BISHOP],
            pos.by_type[KNIGHT],
            pos.by_type[PAWN],
            ep_square(pos),
            pos.stm == WHITE,
        )
    };
    if r > 4 {
        None
    } else {
        Some(r as i32 - 2)
    }
}

pub struct RootResult {
    pub mv: Move,
    pub wdl: Wdl,
    pub dtz: u32,
    /// All moves that keep the same win/draw/loss value as `mv`.
    pub keeping: Vec<Move>,
}

/// Distance-to-zero probe at the root: the move to play and its exact value.
/// Not thread safe; call it once per search, before any search thread starts.
pub fn probe_root(pos: &Position, cfg: &TbConfig) -> Option<RootResult> {
    if pos.castling != 0 || popcount(pos.occ()) > cfg.cardinality() {
        return None;
    }
    let mut results = [RESULT_FAILED; 193]; // TB_MAX_MOVES
    let r = unsafe {
        tb_probe_root_impl(
            pos.by_color[WHITE],
            pos.by_color[BLACK],
            pos.by_type[KING],
            pos.by_type[QUEEN],
            pos.by_type[ROOK],
            pos.by_type[BISHOP],
            pos.by_type[KNIGHT],
            pos.by_type[PAWN],
            pos.rule50 as c_uint,
            ep_square(pos),
            pos.stm == WHITE,
            results.as_mut_ptr(),
        )
    };
    if r == RESULT_FAILED {
        return None;
    }
    let legal = legal_moves(pos);
    let decode = |r: u32| -> Option<Move> {
        let to = ((r >> 4) & 0x3F) as usize;
        let from = ((r >> 10) & 0x3F) as usize;
        let promo = match (r >> 16) & 7 {
            1 => QUEEN,
            2 => ROOK,
            3 => BISHOP,
            4 => KNIGHT,
            _ => 0,
        };
        legal.iter().copied().find(|m| {
            m.from() == from && m.to() == to && (if m.is_promo() { m.promo_type() } else { 0 }) == promo
        })
    };
    let wdl = (r & 0xF) as i32 - 2;
    // Checkmate / stalemate come back without a move and simply find no match.
    let mv = decode(r)?;
    let keeping = results
        .iter()
        .take_while(|&&x| x != RESULT_FAILED)
        .filter(|&&x| (x & 0xF) as i32 - 2 == wdl)
        .filter_map(|&x| decode(x))
        .collect();
    Some(RootResult { mv, wdl, dtz: r >> 20, keeping })
}
