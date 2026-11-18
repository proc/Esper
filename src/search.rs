//! Alpha-beta search: iterative deepening, aspiration windows, PVS with a shared
//! transposition table (lazy SMP), and the usual selectivity (null move, LMR,
//! singular extensions, futility / SEE / history pruning, ...).

use crate::bitboard::*;
use crate::movegen::*;
use crate::nnue::{Network, NnueState};
use crate::position::*;
use crate::tb::{self, TbConfig};
use crate::tune;
use crate::tt::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

pub const MAX_PLY: usize = 128;
pub const MATE: i32 = 32000;
/// Scores at or beyond this are real mates.
pub const MATE_IN_MAX: i32 = MATE - 2 * MAX_PLY as i32;
/// Tablebase wins sit just below the mate range.
pub const TB_WIN: i32 = 30000;
/// Scores at or beyond this are decided (tablebase win or mate) rather than estimates.
pub const MATE_BOUND: i32 = TB_WIN - 2 * MAX_PLY as i32;
pub const INF: i32 = 32001;
pub const NONE: i32 = 32002;

// History table ranges and starting values.
const MAIN_HIST_LIMIT: i32 = 7183;
const CAPTURE_HIST_LIMIT: i32 = 10692;
const CONT_HIST_LIMIT: i32 = 30000;
const PAWN_HIST_LIMIT: i32 = 8192;
const PAWN_HIST_SIZE: usize = 2048;
const LOW_PLY_SIZE: usize = 5;
/// Quiet moves scoring at or below this are tried after everything else.
const GOOD_QUIET_THRESHOLD: i32 = -14000;
const KILLER_ORDERING: bool = true;
// Stockfish-style forms of the main pruning rules (see `search()` / `qsearch()`).
const SF_RFP: bool = true;
const SF_RAZOR: bool = true;
const SF_NMP: bool = true;
const SF_FUTILITY: bool = true;
const SF_QSEARCH: bool = true;
// Further Stockfish-derived techniques.
const SF_EVAL_SCALE: bool = true;
const SF_CORRECTION: bool = true;
const SF_SINGULAR: bool = true;
const SF_LMR_EXTEND: bool = true;
const SF_TT_CUTOFF: bool = true;
const SF_SMALL_PROBCUT: bool = true;
const SF_CAPTURE_PRUNE: bool = true;
const SF_SEEK_MATE: bool = false;
const SF_SHUFFLING: bool = true;
const SF_TIME: bool = true;
/// Range of the Stockfish-style correction history entries.
const SF_CORR_LIMIT: i32 = 1024;
/// Reverse futility pruning applies up to this remaining depth.
const RFP_MAX_DEPTH: i32 = 8;
const RFP_SKIP_TT_PV: bool = false;
/// Depth limit used instead when the opponent has a pawn within two steps of promoting.
const RFP_RUNNER_DEPTH: i32 = 0;
/// Extra null-move reduction (plies) granted when the evaluation is far above beta.
const NMP_EVAL_PLIES: i32 = 3;
/// Piece values in internal evaluation units, used to weigh moves for ordering.
const PIECE_VALUE: [i32; 8] = [0, 208, 781, 825, 1276, 2538, 0, 0];
const CORR_SIZE: usize = 16384;
const CORR_LIMIT: i32 = 96 * 256;
const NO_CONT: usize = usize::MAX;
// Time management
const DOMINANT_SHARE: f64 = 0.65;
const DISPUTED_SHARE: f64 = 0.35;
const STABLE_ITERATIONS: u32 = 4;
/// Score drop between iterations (40 centipawns) that buys extra time.
const FAIL_LOW_DROP: i32 = 40 * NORMALIZE_PAWN / 100;
/// Multi-cut (off by default: it cost about 80 Elo in testing): prune a cut node when `MC_CUTOFFS` of its first `MC_CANDIDATES`
/// moves fail high at reduced depth.
const MULTI_CUT: bool = false;
const MC_CANDIDATES: usize = 8;
const MC_CUTOFFS: usize = 3;
/// Quiet moves whose history is below `-margin * depth` are pruned at low depth.
const HISTORY_PRUNE_MARGIN: i32 = 8000;
// The history tables sit well below zero on average (negative starting values,
// penalties larger than bonuses), so reductions measure against typical scores.
const QUIET_HIST_CENTRE: i32 = -16000;
const QUIET_HIST_STEP: i32 = 4500;
const CAPTURE_HIST_CENTRE: i32 = -7000;
const CAPTURE_HIST_STEP: i32 = 3500;
/// Internal evaluation units per pawn, for UCI centipawn output.
pub const NORMALIZE_PAWN: i32 = 290;

#[derive(Clone, Default)]
pub struct Limits {
    pub time: [Option<u64>; 2],
    pub inc: [u64; 2],
    pub movestogo: Option<u64>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u64>,
    pub infinite: bool,
    pub ponder: bool,
    pub searchmoves: Vec<Move>,
}

#[repr(align(128))]
pub struct PaddedAtomicU64(pub AtomicU64);

#[repr(align(128))]
pub struct PaddedAtomicBool(pub AtomicBool);

pub struct Shared {
    pub stop: PaddedAtomicBool,
    pub pondering: PaddedAtomicBool,
    pub nodes: Vec<PaddedAtomicU64>,
    pub tb_hits: Vec<PaddedAtomicU64>,
    /// Root best-move changes summed over all threads (time management).
    pub best_move_changes: PaddedAtomicU64,
}

impl Shared {
    pub fn new() -> Shared {
        Shared {
            stop: PaddedAtomicBool(AtomicBool::new(false)),
            pondering: PaddedAtomicBool(AtomicBool::new(false)),
            nodes: (0..1024).map(|_| PaddedAtomicU64(AtomicU64::new(0))).collect(),
            tb_hits: (0..1024).map(|_| PaddedAtomicU64(AtomicU64::new(0))).collect(),
            best_move_changes: PaddedAtomicU64(AtomicU64::new(0)),
        }
    }
    pub fn total_tb_hits(&self, threads: usize) -> u64 {
        self.tb_hits[..threads].iter().map(|n| n.0.load(Relaxed)).sum()
    }
    pub fn total_nodes(&self, threads: usize) -> u64 {
        self.nodes[..threads].iter().map(|n| n.0.load(Relaxed)).sum()
    }
}

#[derive(Clone, Copy)]
struct Stack {
    mv: Move,
    cont: usize,
    static_eval: i32,
    killers: [Move; 2],
    excluded: Move,
    double_ext: i32,
    move_count: i32,
    tt_hit: bool,
    cutoff_cnt: i32,
}

struct LmrTable([[i32; 64]; 64]);

fn lmr_table() -> &'static LmrTable {
    static T: std::sync::OnceLock<LmrTable> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = LmrTable([[0; 64]; 64]);
        for d in 1..64 {
            for m in 1..64 {
                t.0[d][m] = ((0.75 + (d as f64).ln() * (m as f64).ln() / 2.0) * 1024.0) as i32;
            }
        }
        t
    })
}

pub struct Searcher {
    pub id: usize,
    pub threads: usize,
    pub tt: Arc<TT>,
    pub shared: Arc<Shared>,
    nnue: NnueState,
    pos: Vec<Position>,
    ss: Vec<Stack>,
    keys: Vec<u64>,
    pv: Vec<[Move; MAX_PLY + 2]>,
    pv_len: [usize; MAX_PLY + 2],
    // history heuristics
    butterfly: Box<[[i16; 4096]; 2]>,
    capture_hist: Box<[[[i16; 8]; 64]; 16]>,
    cont_hist: Vec<i16>,
    /// Indexed by pawn structure, piece and destination.
    pawn_hist: Vec<i16>,
    /// Butterfly-style history for the first few plies of the current search.
    low_ply_hist: Box<[[i16; 4096]; LOW_PLY_SIZE]>,
    /// Quiet move that last refuted a given (piece, to-square) move.
    counter: Vec<Move>,
    pawn_corr: Box<[[i16; CORR_SIZE]; 2]>,
    /// Correction by knight and bishop placement.
    minor_corr: Box<[[i16; CORR_SIZE]; 2]>,
    /// Correction by move pairs: (move k plies back, previous move), k = 2, 4, 6.
    cont_corr: Vec<i16>,
    /// How often the hash move turned out best at non-PV nodes.
    tt_move_hist: i16,
    /// Optimism of each side, from the root move's running average score.
    optimism: [i32; 2],
    /// Depth of the current iteration.
    root_depth: i32,
    /// Look harder for a mate: the root score already suggests one.
    seek_mate: bool,
    // Kept across moves by the main thread for time management.
    prev_time_reduction: f64,
    prev_avg_score: i32,
    prev_best_score: i32,
    nonpawn_corr: Box<[[[i16; CORR_SIZE]; 2]; 2]>,
    // search state
    pub nodes: u64,
    tb_hits: u64,
    tb: TbConfig,
    tb_cardinality: u32,
    seldepth: usize,
    stopped: bool,
    completed_depth: i32,
    root_excluded: Vec<Move>,
    root_nodes: Box<[u64; 4096]>,
    nmp_min_ply: usize,
    // limits / time management (used by the main thread)
    pub limits: Limits,
    pub multipv: usize,
    start: Instant,
    soft_ms: u64,
    hard_ms: u64,
    pub silent: bool,
    pub show_wdl: bool,
    // results
    pub best_move: Move,
    pub ponder_move: Move,
    pub best_score: i32,
    best_pv: Vec<Move>,
    best_seldepth: usize,
}

/// Linear interpolation of `x` from [x0, x1] to [y0, y1] (not clamped).
#[inline]
fn interpolate(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

/// Bonus for a move that caused a cutoff at the given depth.
#[inline(always)]
fn stat_bonus(depth: i32) -> i32 {
    (133 * depth - 81).min(1487)
}

/// Penalty for the moves tried before the one that caused the cutoff.
#[inline(always)]
fn stat_malus(depth: i32) -> i32 {
    (968 * depth - 235).min(2244)
}

/// Move a history entry towards `bonus`, saturating at +/- `limit`.
#[inline(always)]
fn hist_update(h: &mut i16, bonus: i32, limit: i32) {
    let b = bonus.clamp(-limit, limit);
    let v = *h as i32;
    *h = (v + b - v * b.abs() / limit) as i16;
}

#[inline(always)]
fn score_to_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s + ply as i32
    } else if s <= -MATE_BOUND {
        s - ply as i32
    } else {
        s
    }
}

#[inline(always)]
fn score_from_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s - ply as i32
    } else if s <= -MATE_BOUND {
        s + ply as i32
    } else {
        s
    }
}

/// Centipawn score reported for a tablebase win.
pub const TB_CP: i32 = 20000;

pub fn score_to_uci(s: i32) -> String {
    if s >= MATE_IN_MAX {
        format!("mate {}", (MATE - s + 1) / 2)
    } else if s <= -MATE_IN_MAX {
        format!("mate {}", -(MATE + s) / 2)
    } else if s >= MATE_BOUND {
        format!("cp {}", TB_CP - (TB_WIN - s))
    } else if s <= -MATE_BOUND {
        format!("cp {}", -TB_CP + (TB_WIN + s))
    } else {
        format!("cp {}", s * 100 / NORMALIZE_PAWN)
    }
}

/// Convert an internal search score to `(win, draw, loss)` permille values.
pub fn score_to_wdl(s: i32, pos: &Position) -> (i32, i32, i32) {
    if s >= MATE_BOUND {
        return (1000, 0, 0);
    }
    if s <= -MATE_BOUND {
        return (0, 0, 1000);
    }
    let material = (popcount(pos.by_type[PAWN])
        + 3 * popcount(pos.by_type[KNIGHT] | pos.by_type[BISHOP])
        + 5 * popcount(pos.by_type[ROOK])
        + 9 * popcount(pos.by_type[QUEEN]))
    .clamp(17, 78) as f64;
    let m = material / 58.0;
    let a = NORMALIZE_PAWN as f64 * (0.75 + 0.25 * m);
    let b = NORMALIZE_PAWN as f64 * (0.28 + 0.10 * m);
    let x = (s as f64).clamp(-4000.0, 4000.0);
    let w = (1000.0 / (1.0 + ((a - x) / b).exp())).round() as i32;
    let l = (1000.0 / (1.0 + ((a + x) / b).exp())).round() as i32;
    let d = (1000 - w - l).max(0);
    (w, d, l)
}

// ---------------------------------------------------------------------------
// Staged move picker
// ---------------------------------------------------------------------------

// Moves come out in the order: hash move, captures that do not lose material
// (most valuable victim first, adjusted by capture history), quiet moves with a
// reasonable history score, then losing captures and poorly rated quiets. In
// check, all evasions form one list with captures first; in quiescence only
// captures and promotions are produced.

#[derive(PartialEq, Clone, Copy)]
enum Stage {
    TtMove,
    Init,
    GoodCapture,
    Killer0,
    Killer1,
    Counter,
    GenQuiet,
    GoodQuiet,
    Bad,
    /// One fully sorted list: evasions, or captures in quiescence / ProbCut.
    All,
    Done,
}

#[derive(Clone, Copy)]
struct Ext {
    m: Move,
    v: i32,
    hist: i32,
}

struct Picker {
    stage: Stage,
    tt_move: Move,
    depth: i32,
    /// Captures / promotions only (quiescence, ProbCut); ignored when in check.
    noisy_only: bool,
    last_hist: i32,
    has_last_hist: bool,
    buf: [std::mem::MaybeUninit<Ext>; MAX_MOVES],
    cur: usize,
    end_cur: usize,
    /// Deferred moves live at the front of the buffer: bad captures, then bad quiets.
    end_bad: usize,
    end_bad_caps: usize,
}

/// Sort descending all entries with a value of at least `limit` to the front;
/// the order of the remaining ones is unspecified.
fn partial_insertion_sort(list: &mut [Ext], limit: i32) {
    let mut sorted_end = 0;
    for p in 1..list.len() {
        if list[p].v >= limit {
            let tmp = list[p];
            sorted_end += 1;
            list[p] = list[sorted_end];
            let mut q = sorted_end;
            while q > 0 && list[q - 1].v < tmp.v {
                list[q] = list[q - 1];
                q -= 1;
            }
            list[q] = tmp;
        }
    }
}

impl Picker {
    fn new(tt_move: Move, noisy_only: bool, depth: i32) -> Picker {
        Picker {
            stage: if tt_move.is_none() { Stage::Init } else { Stage::TtMove },
            tt_move,
            depth,
            noisy_only,
            last_hist: 0,
            has_last_hist: false,
            buf: [std::mem::MaybeUninit::uninit(); MAX_MOVES],
            cur: 0,
            end_cur: 0,
            end_bad: 0,
            end_bad_caps: 0,
        }
    }

    /// Is the move just returned one of the captures that passed the exchange test?
    #[inline(always)]
    fn in_good_captures(&self) -> bool {
        self.stage == Stage::GoodCapture
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Ext {
        // SAFETY: callers only read slots below `end_cur` / `end_bad`, all written.
        unsafe { self.buf[i].assume_init() }
    }

    #[inline(always)]
    fn slice_mut(&mut self, from: usize, to: usize) -> &mut [Ext] {
        // SAFETY: the range was filled by one of the `score_*` functions.
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr().add(from) as *mut Ext, to - from) }
    }

    /// Captures and promotions: victim value plus capture history.
    fn score_captures(&mut self, s: &Searcher, pos: &Position, at: usize) -> usize {
        let mut list = MoveList::new();
        generate(pos, &mut list, true, false);
        for (i, &m) in list.as_slice().iter().enumerate() {
            let pc = pos.board[m.from()] as usize;
            let victim = if m.flag() == FLAG_EP { PAWN } else { piece_type(pos.board[m.to()]) };
            let mut v = s.capture_hist[pc][m.to()][victim] as i32 + 7 * PIECE_VALUE[victim];
            if m.is_promo() && m.promo_type() != QUEEN {
                // Under-promotions never pass the good-capture test.
                v -= 1 << 24;
            }
            self.buf[at + i].write(Ext { m, v, hist: 0 });
        }
        at + list.len
    }

    /// Quiet moves: history tables, threats by lesser pieces and checks.
    fn score_quiets(&mut self, s: &Searcher, ply: usize, pos: &Position, at: usize) -> usize {
        let mut list = MoveList::new();
        generate(pos, &mut list, false, true);
        let us = pos.stm;
        let them = us ^ 1;
        let occ = pos.occ();
        let detailed = self.depth > 1;

        // Squares attacked by enemy pieces of lesser value than the mover.
        let mut lesser = [0u64; 8];
        let mut checks = [0u64; 8];
        if detailed {
            let pawns = pos.pieces(them, PAWN);
            let pawn_att = if them == WHITE {
                ((pawns & !FILE_A) << 7) | ((pawns & !FILE_H) << 9)
            } else {
                ((pawns & !FILE_A) >> 9) | ((pawns & !FILE_H) >> 7)
            };
            let (mut minor_att, mut rook_att) = (0u64, 0u64);
            let mut b = pos.pieces(them, KNIGHT);
            while b != 0 {
                minor_att |= knight_attacks(pop_lsb(&mut b));
            }
            let mut b = pos.pieces(them, BISHOP);
            while b != 0 {
                minor_att |= bishop_attacks(pop_lsb(&mut b), occ);
            }
            let mut b = pos.pieces(them, ROOK);
            while b != 0 {
                rook_att |= rook_attacks(pop_lsb(&mut b), occ);
            }
            lesser[KNIGHT] = pawn_att;
            lesser[BISHOP] = pawn_att;
            lesser[ROOK] = pawn_att | minor_att;
            lesser[QUEEN] = lesser[ROOK] | rook_att;

            // Squares from which each piece type would give check.
            let ksq = pos.king_sq(them);
            checks[PAWN] = pawn_attacks(them, ksq);
            checks[KNIGHT] = knight_attacks(ksq);
            checks[BISHOP] = bishop_attacks(ksq, occ);
            checks[ROOK] = rook_attacks(ksq, occ);
            checks[QUEEN] = checks[BISHOP] | checks[ROOK];
        }

        let cont_slice = |back: usize| -> Option<&[i16]> {
            if ply >= back {
                let c = s.ss[ply - back].cont;
                if c != NO_CONT {
                    let base = c * 1024;
                    return Some(&s.cont_hist[base..base + 1024]);
                }
            }
            None
        };
        let c1 = cont_slice(1);
        let c2 = cont_slice(2);
        let c3 = cont_slice(3);
        let c4 = cont_slice(4);
        let c6 = cont_slice(6);

        let pawn_base = (pos.pawn_key as usize % PAWN_HIST_SIZE) * 1024;
        let killers = s.ss[ply].killers;
        let counter = s.counter_move(ply);
        for (i, &m) in list.as_slice().iter().enumerate() {
            // Killers and the countermove get lighter reductions in this search, so
            // they are also tried first among the quiet moves.
            if KILLER_ORDERING && (m == killers[0] || m == killers[1] || m == counter) {
                let v = if m == killers[0] {
                    3 << 24
                } else if m == killers[1] {
                    2 << 24
                } else {
                    1 << 24
                };
                self.buf[at + i].write(Ext { m, v, hist: 0 });
                continue;
            }
            let (from, to) = (m.from(), m.to());
            let pc = pos.board[from] as usize;
            let pt = pc & 7;
            let idx = pc * 64 + to;
            let h1 = c1.map_or(0, |t| t[idx] as i32);
            let h2 = c2.map_or(0, |t| t[idx] as i32);
            let hist = 2 * s.butterfly[us][m.from_to()] as i32 + h1 + h2;
            let mut v = hist
                + 2 * s.pawn_hist[pawn_base + idx] as i32
                + c3.map_or(0, |t| t[idx] as i32)
                + c4.map_or(0, |t| t[idx] as i32)
                + c6.map_or(0, |t| t[idx] as i32);
            if detailed {
                // Bonus for escaping an attack by a lesser piece, penalty for walking into one.
                let threat = (lesser[pt] & bit(from) != 0) as i32 - (lesser[pt] & bit(to) != 0) as i32;
                v += PIECE_VALUE[pt] * 20 * threat;
                if checks[pt] & bit(to) != 0 && pos.see_ge(m, -36) {
                    v += 16384;
                }
            }
            if ply < LOW_PLY_SIZE {
                v += 8 * s.low_ply_hist[ply][m.from_to()] as i32 / (1 + ply as i32);
            }
            self.buf[at + i].write(Ext { m, v, hist });
        }
        at + list.len
    }

    /// Check evasions: captures by victim value, then quiets by history.
    fn score_evasions(&mut self, s: &Searcher, ply: usize, pos: &Position, at: usize) -> usize {
        let mut list = MoveList::new();
        generate(pos, &mut list, true, true);
        for (i, &m) in list.as_slice().iter().enumerate() {
            let v = if m.is_capture() {
                let victim = if m.flag() == FLAG_EP { PAWN } else { piece_type(pos.board[m.to()]) };
                PIECE_VALUE[victim] + (1 << 28)
            } else {
                let idx = pos.board[m.from()] as usize * 64 + m.to();
                s.butterfly[pos.stm][m.from_to()] as i32 + s.cont_score(ply, 1, idx)
            };
            self.buf[at + i].write(Ext { m, v, hist: 0 });
        }
        at + list.len
    }

    fn next(&mut self, s: &Searcher, ply: usize, skip_quiets: bool) -> Move {
        let pos = &s.pos[ply];
        self.has_last_hist = false;
        loop {
            match self.stage {
                Stage::TtMove => {
                    self.stage = Stage::Init;
                    return self.tt_move;
                }
                Stage::Init => {
                    self.cur = 0;
                    self.end_bad = 0;
                    if pos.checkers != 0 {
                        self.end_cur = self.score_evasions(s, ply, pos, 0);
                        self.stage = Stage::All;
                    } else {
                        self.end_cur = self.score_captures(s, pos, 0);
                        self.stage = if self.noisy_only { Stage::All } else { Stage::GoodCapture };
                    }
                    let end = self.end_cur;
                    partial_insertion_sort(self.slice_mut(0, end), i32::MIN);
                }
                Stage::GoodCapture => {
                    while self.cur < self.end_cur {
                        let e = self.get(self.cur);
                        self.cur += 1;
                        if e.m == self.tt_move {
                            continue;
                        }
                        if pos.see_ge(e.m, -e.v / 37) {
                            return e.m;
                        }
                        self.buf[self.end_bad].write(e);
                        self.end_bad += 1;
                    }
                    self.end_bad_caps = self.end_bad;
                    if skip_quiets {
                        self.cur = 0;
                        self.end_cur = self.end_bad;
                        self.stage = Stage::Bad;
                    } else {
                        self.stage = Stage::Killer0;
                    }
                }
                Stage::Killer0 => {
                    self.stage = Stage::Killer1;
                    if !skip_quiets {
                        let k = s.ss[ply].killers[0];
                        if !k.is_none() && k != self.tt_move && !k.is_noisy() && pos.is_pseudo_legal(k) {
                            return k;
                        }
                    }
                }
                Stage::Killer1 => {
                    self.stage = Stage::Counter;
                    if !skip_quiets {
                        let killers = s.ss[ply].killers;
                        let k = killers[1];
                        if !k.is_none() && k != self.tt_move && k != killers[0] && !k.is_noisy() && pos.is_pseudo_legal(k) {
                            return k;
                        }
                    }
                }
                Stage::Counter => {
                    self.stage = Stage::GenQuiet;
                    if !skip_quiets {
                        let killers = s.ss[ply].killers;
                        let c = s.counter_move(ply);
                        if !c.is_none()
                            && c != self.tt_move
                            && c != killers[0]
                            && c != killers[1]
                            && !c.is_noisy()
                            && pos.is_pseudo_legal(c)
                        {
                            return c;
                        }
                    }
                }
                Stage::GenQuiet => {
                    if skip_quiets {
                        self.cur = 0;
                        self.end_cur = self.end_bad;
                        self.stage = Stage::Bad;
                    } else {
                        self.cur = self.end_bad;
                        self.end_cur = self.score_quiets(s, ply, pos, self.cur);
                        let (from, to, limit) = (self.cur, self.end_cur, -3560 * self.depth);
                        partial_insertion_sort(self.slice_mut(from, to), limit);
                        self.stage = Stage::GoodQuiet;
                    }
                }
                Stage::GoodQuiet => {
                    if skip_quiets {
                        // Drop the remaining quiets, including those already deferred.
                        self.end_bad = self.end_bad_caps;
                    } else {
                        let killers = s.ss[ply].killers;
                        let counter = s.counter_move(ply);
                        while self.cur < self.end_cur {
                            let e = self.get(self.cur);
                            self.cur += 1;
                            if e.m == self.tt_move || e.m == killers[0] || e.m == killers[1] || e.m == counter {
                                continue;
                            }
                            if e.v > GOOD_QUIET_THRESHOLD {
                                self.last_hist = e.hist;
                                self.has_last_hist = true;
                                return e.m;
                            }
                            self.buf[self.end_bad].write(e);
                            self.end_bad += 1;
                        }
                    }
                    self.cur = 0;
                    self.end_cur = self.end_bad;
                    self.stage = Stage::Bad;
                }
                Stage::Bad => {
                    if skip_quiets && self.end_cur > self.end_bad_caps {
                        self.end_cur = self.end_bad_caps;
                    }
                    let killers = s.ss[ply].killers;
                    let counter = s.counter_move(ply);
                    while self.cur < self.end_cur {
                        let e = self.get(self.cur);
                        self.cur += 1;
                        if e.m == self.tt_move
                            || (self.cur > self.end_bad_caps
                                && (e.m == killers[0] || e.m == killers[1] || e.m == counter))
                        {
                            continue;
                        }
                        if self.cur > self.end_bad_caps {
                            self.last_hist = e.hist;
                            self.has_last_hist = true;
                        }
                        return e.m;
                    }
                    self.stage = Stage::Done;
                }
                Stage::All => {
                    while self.cur < self.end_cur {
                        let e = self.get(self.cur);
                        self.cur += 1;
                        if e.m != self.tt_move {
                            return e.m;
                        }
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return Move::NONE,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Searcher
// ---------------------------------------------------------------------------

struct Line {
    score: i32,
    seldepth: usize,
    pv: Vec<Move>,
}

impl Searcher {
    pub fn new(id: usize, threads: usize, net: Arc<Network>, tt: Arc<TT>, shared: Arc<Shared>) -> Box<Searcher> {
        Box::new(Searcher {
            id,
            threads,
            tt,
            shared,
            nnue: NnueState::new(net, MAX_PLY + 8),
            pos: vec![Position::empty(); MAX_PLY + 8],
            ss: vec![
                Stack {
                    mv: Move::NONE,
                    cont: NO_CONT,
                    static_eval: NONE,
                    killers: [Move::NONE; 2],
                    excluded: Move::NONE,
                    double_ext: 0,
                    move_count: 0,
                    tt_hit: false,
                    cutoff_cnt: 0
                };
                MAX_PLY + 8
            ],
            keys: Vec::with_capacity(1024),
            pv: vec![[Move::NONE; MAX_PLY + 2]; MAX_PLY + 2],
            pv_len: [0; MAX_PLY + 2],
            butterfly: vec![[-5i16; 4096]; 2].into_boxed_slice().try_into().unwrap(),
            capture_hist: vec![[[-742i16; 8]; 64]; 16].into_boxed_slice().try_into().unwrap(),
            cont_hist: vec![-586; 1024 * 1024],
            pawn_hist: vec![5; PAWN_HIST_SIZE * 1024],
            low_ply_hist: vec![[102i16; 4096]; LOW_PLY_SIZE].into_boxed_slice().try_into().unwrap(),
            counter: vec![Move::NONE; 1024],
            pawn_corr: vec![[0i16; CORR_SIZE]; 2].into_boxed_slice().try_into().unwrap(),
            minor_corr: vec![[0i16; CORR_SIZE]; 2].into_boxed_slice().try_into().unwrap(),
            cont_corr: vec![0; 1024 * 1024],
            tt_move_hist: 0,
            optimism: [0; 2],
            root_depth: 1,
            seek_mate: false,
            prev_time_reduction: 0.85,
            prev_avg_score: NONE,
            prev_best_score: NONE,
            nonpawn_corr: vec![[[0i16; CORR_SIZE]; 2]; 2].into_boxed_slice().try_into().unwrap(),
            nodes: 0,
            tb_hits: 0,
            tb: TbConfig::DEFAULT,
            tb_cardinality: 0,
            seldepth: 0,
            stopped: false,
            completed_depth: 0,
            root_excluded: Vec::new(),
            root_nodes: vec![0u64; 4096].into_boxed_slice().try_into().unwrap(),
            nmp_min_ply: 0,
            limits: Limits::default(),
            multipv: 1,
            start: Instant::now(),
            soft_ms: u64::MAX,
            hard_ms: u64::MAX,
            silent: false,
            show_wdl: false,
            best_move: Move::NONE,
            ponder_move: Move::NONE,
            best_score: 0,
            best_pv: Vec::new(),
            best_seldepth: 0,
        })
    }

    pub fn in_check_root(&self) -> bool {
        self.pos[0].in_check()
    }

    pub fn clear(&mut self) {
        self.butterfly.iter_mut().for_each(|a| a.fill(-5));
        self.capture_hist.iter_mut().flatten().for_each(|a| a.fill(-742));
        self.cont_hist.fill(-586);
        self.pawn_hist.fill(5);
        self.counter.fill(Move::NONE);
        self.pawn_corr.iter_mut().for_each(|a| a.fill(0));
        self.minor_corr.iter_mut().for_each(|a| a.fill(0));
        self.cont_corr.fill(0);
        self.tt_move_hist = 0;
        self.prev_time_reduction = 0.85;
        self.prev_avg_score = NONE;
        self.prev_best_score = NONE;
        self.nonpawn_corr.iter_mut().flatten().for_each(|a| a.fill(0));
    }

    /// Static evaluation of `pos` (not in check), for the `eval` command.
    pub fn eval_position(&mut self, pos: &Position) -> (i32, i32) {
        self.pos[0] = *pos;
        self.nnue.reset();
        let raw = self.nnue.evaluate(&self.pos[..1], 0);
        (raw, self.scale_eval(raw, pos))
    }

    pub fn prepare(
        &mut self,
        root: &Position,
        game_keys: &[u64],
        limits: &Limits,
        multipv: usize,
        overhead: u64,
        tb: TbConfig,
    ) {
        self.tb = tb;
        self.tb_cardinality = tb.cardinality();
        self.tb_hits = 0;
        self.pos[0] = *root;
        self.nnue.reset();
        self.keys.clear();
        self.keys.extend_from_slice(game_keys);
        self.limits = limits.clone();
        self.multipv = multipv;
        self.nodes = 0;
        self.stopped = false;
        self.completed_depth = 0;
        self.nmp_min_ply = 0;
        self.best_move = Move::NONE;
        self.ponder_move = Move::NONE;
        self.best_score = 0;
        self.best_pv.clear();
        self.root_nodes.fill(0);
        self.low_ply_hist.iter_mut().for_each(|a| a.fill(102));
        for s in self.ss.iter_mut() {
            s.mv = Move::NONE;
            s.cont = NO_CONT;
            s.static_eval = NONE;
            s.killers = [Move::NONE; 2];
            s.excluded = Move::NONE;
            s.double_ext = 0;
            s.move_count = 0;
            s.tt_hit = false;
            s.cutoff_cnt = 0;
        }
        self.start = Instant::now();
        self.soft_ms = u64::MAX;
        self.hard_ms = u64::MAX;
        let us = root.stm;
        if let Some(mt) = limits.movetime {
            let t = mt.saturating_sub(overhead).max(1);
            self.soft_ms = t;
            self.hard_ms = t;
        } else if let Some(time) = limits.time[us] {
            let t = time as f64;
            let inc = limits.inc[us] as f64;
            let move_overhead = overhead as f64;
            let ply = game_keys.len().saturating_sub(1).min(50) as f64;
            let centi_mtg = limits.movestogo.map_or(5051.0, |m| (m.max(1) as f64 * 100.0).min(5000.0));
            let time_left =
                (t + inc * (centi_mtg - 100.0) / 100.0 - move_overhead * (2.0 + centi_mtg / 100.0)).max(1.0);
            let (opt_scale, max_scale) = match limits.movestogo {
                Some(_) => (
                    ((0.88 + ply / 116.4) / (centi_mtg / 100.0)).min(0.88 * t / time_left),
                    1.3 + 0.11 * (centi_mtg / 100.0),
                ),
                None => (
                    (0.0084 + (ply + 3.0).sqrt() * 0.0042).min(0.213 * t / time_left),
                    (4.0 + ply / 12.0).min(6.67),
                ),
            };
            let soft = opt_scale * time_left;
            let hard = (0.851 * t - move_overhead).min(max_scale * soft);
            self.hard_ms = (hard.max(1.0)) as u64;
            self.soft_ms = (soft.max(1.0) as u64).min(self.hard_ms);
        }
    }

    // -- helpers ------------------------------------------------------------

    #[inline]
    fn scale_eval(&self, raw: i32, pos: &Position) -> i32 {
        if SF_EVAL_SCALE {
            return self.scale_eval_sf(raw, pos);
        }
        let phase = popcount(pos.by_type[PAWN]) as i32
            + 3 * popcount(pos.by_type[KNIGHT] | pos.by_type[BISHOP]) as i32
            + 5 * popcount(pos.by_type[ROOK]) as i32
            + 9 * popcount(pos.by_type[QUEEN]) as i32;
        let mut v = raw * (800 + 3 * phase.min(78)) / 800;
        v = v * (200 - pos.rule50.min(100) as i32) / 200;
        v.clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    /// Stockfish's post-processing of the network output: agreement with a plain
    /// material count makes the evaluation bolder, optimism leans towards the
    /// side that is doing well at the root, then material and 50-move scaling.
    #[inline]
    fn scale_eval_sf(&self, nnue: i32, pos: &Position) -> i32 {
        const VAL: [i32; 7] = [0, 208, 781, 825, 1276, 2538, 0];
        let (us, them) = (pos.stm, pos.stm ^ 1);
        let mut se = 0;
        let mut npm = 0;
        for pt in PAWN..=QUEEN {
            let (a, b) = (popcount(pos.pieces(us, pt)) as i32, popcount(pos.pieces(them, pt)) as i32);
            se += VAL[pt] * (a - b);
            if pt != PAWN {
                npm += VAL[pt] * (a + b);
            }
        }
        let optimism = self.optimism[us];
        let se_norm = se * 1024 / (se.abs() + 1024);
        let nnue_norm = nnue * 1024 / (nnue.abs() + 1024);
        let correl = se_norm * nnue_norm / 512;
        let mut v = nnue + (nnue + optimism) * correl / 65536;
        let material = 521 * popcount(pos.by_type[PAWN]) as i32 + npm;
        v += (material as i64 * v as i64 / 90649) as i32;
        v -= v * pos.rule50.min(100) as i32 / 189;
        v.clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    #[inline]
    fn evaluate(&mut self, ply: usize) -> i32 {
        let raw = self.nnue.evaluate(&self.pos[..=ply], ply);
        self.scale_eval(raw, &self.pos[ply])
    }

    /// Stockfish-style correction value of the position at `ply`, in its raw
    /// weighted form (divide by 131072 for evaluation units).
    #[inline]
    fn correction_raw(&self, ply: usize) -> i32 {
        let pos = &self.pos[ply];
        let stm = pos.stm;
        let i = |k: u64| (k as usize) % CORR_SIZE;
        let pcv = self.pawn_corr[stm][i(pos.pawn_key)] as i32;
        let micv = self.minor_corr[stm][i(pos.minor_key)] as i32;
        let wnp = self.nonpawn_corr[WHITE][stm][i(pos.nonpawn_key[WHITE])] as i32;
        let bnp = self.nonpawn_corr[BLACK][stm][i(pos.nonpawn_key[BLACK])] as i32;
        let cont = |back: usize| -> i32 {
            if ply >= back && self.ss[ply - back].cont != NO_CONT && self.ss[ply - 1].cont != NO_CONT {
                self.cont_corr[self.ss[ply - back].cont * 1024 + self.ss[ply - 1].cont] as i32
            } else {
                0
            }
        };
        let cntcv = if ply >= 1 && self.ss[ply - 1].cont != NO_CONT {
            7885 * (cont(2) + cont(4)) + 6307 * cont(6)
        } else {
            80695
        };
        13806 * pcv + 9512 * micv + 11615 * (wnp + bnp) + cntcv
    }

    /// Stockfish-style correction update with `bonus` in [-256, 256].
    fn update_correction_sf(&mut self, ply: usize, bonus: i32) {
        let pos = self.pos[ply];
        let stm = pos.stm;
        let i = |k: u64| (k as usize) % CORR_SIZE;
        hist_update(&mut self.pawn_corr[stm][i(pos.pawn_key)], bonus, SF_CORR_LIMIT);
        hist_update(&mut self.minor_corr[stm][i(pos.minor_key)], bonus * 150 / 128, SF_CORR_LIMIT);
        hist_update(&mut self.nonpawn_corr[WHITE][stm][i(pos.nonpawn_key[WHITE])], bonus * 186 / 128, SF_CORR_LIMIT);
        hist_update(&mut self.nonpawn_corr[BLACK][stm][i(pos.nonpawn_key[BLACK])], bonus * 186 / 128, SF_CORR_LIMIT);
        if ply >= 1 && self.ss[ply - 1].cont != NO_CONT {
            let to = self.ss[ply - 1].cont;
            for (back, w) in [(2usize, 130), (4, 70), (6, 35)] {
                if ply >= back && self.ss[ply - back].cont != NO_CONT {
                    let idx = self.ss[ply - back].cont * 1024 + to;
                    hist_update(&mut self.cont_corr[idx], bonus * w / 128, SF_CORR_LIMIT);
                }
            }
        }
    }

    /// Correction to add to the static evaluation of the position at `ply`.
    #[inline]
    fn correction_at(&self, ply: usize) -> i32 {
        if SF_CORRECTION {
            self.correction_raw(ply) / 131072
        } else {
            self.correction(&self.pos[ply])
        }
    }

    #[inline]
    fn correction(&self, pos: &Position) -> i32 {
        let stm = pos.stm;
        let p = self.pawn_corr[stm][(pos.pawn_key as usize) % CORR_SIZE] as i32;
        let w = self.nonpawn_corr[WHITE][stm][(pos.nonpawn_key[WHITE] as usize) % CORR_SIZE] as i32;
        let b = self.nonpawn_corr[BLACK][stm][(pos.nonpawn_key[BLACK] as usize) % CORR_SIZE] as i32;
        (2 * p + w + b) / 512
    }

    fn update_correction(&mut self, pos: &Position, diff: i32, depth: i32) {
        let stm = pos.stm;
        let w = (depth + 1).min(16);
        let target = diff.clamp(-CORR_LIMIT / 256, CORR_LIMIT / 256) * 256;
        let upd = |e: &mut i16| {
            *e = ((*e as i32 * (256 - w) + target * w) / 256).clamp(-CORR_LIMIT, CORR_LIMIT) as i16;
        };
        upd(&mut self.pawn_corr[stm][(pos.pawn_key as usize) % CORR_SIZE]);
        upd(&mut self.nonpawn_corr[WHITE][stm][(pos.nonpawn_key[WHITE] as usize) % CORR_SIZE]);
        upd(&mut self.nonpawn_corr[BLACK][stm][(pos.nonpawn_key[BLACK] as usize) % CORR_SIZE]);
    }

    #[inline(always)]
    fn cont_score(&self, ply: usize, back: usize, idx: usize) -> i32 {
        if ply >= back {
            let c = self.ss[ply - back].cont;
            if c != NO_CONT {
                return self.cont_hist[c * 1024 + idx] as i32;
            }
        }
        0
    }

    /// History score of a quiet move as used by pruning and reductions.
    #[inline]
    fn quiet_score(&self, ply: usize, pos: &Position, m: Move) -> i32 {
        let idx = pos.board[m.from()] as usize * 64 + m.to();
        2 * self.butterfly[pos.stm][m.from_to()] as i32 + self.cont_score(ply, 1, idx) + self.cont_score(ply, 2, idx)
    }

    /// Update the continuation histories of the move `idx` (piece * 64 + to) played
    /// at `ply`, pairing it with the moves made 1, 2, 3, 4 and 6 plies earlier.
    fn update_cont(&mut self, ply: usize, in_check: bool, idx: usize, bonus: i32) {
        const WEIGHTS: [(usize, i32); 5] = [(1, 520), (2, 390), (3, 145), (4, 251), (6, 209)];
        // Entries that already agree with each other reinforce the update.
        const CONSISTENCY: [i32; 6] = [94, 103, 110, 106, 119, 126];
        let mut positive = 0;
        for (back, weight) in WEIGHTS {
            if in_check && back > 2 {
                break;
            }
            if ply >= back {
                let c = self.ss[ply - back].cont;
                if c != NO_CONT {
                    let e = &mut self.cont_hist[c * 1024 + idx];
                    if *e > 0 {
                        positive += 1;
                    }
                    let b = bonus * weight * CONSISTENCY[positive] / 65536 + 73 * (back < 2) as i32;
                    hist_update(e, b, CONT_HIST_LIMIT);
                }
            }
        }
    }

    fn update_quiet(&mut self, ply: usize, pos: &Position, m: Move, bonus: i32) {
        hist_update(&mut self.butterfly[pos.stm][m.from_to()], bonus, MAIN_HIST_LIMIT);
        if ply < LOW_PLY_SIZE {
            hist_update(&mut self.low_ply_hist[ply][m.from_to()], bonus * 712 / 1024, MAIN_HIST_LIMIT);
        }
        let idx = pos.board[m.from()] as usize * 64 + m.to();
        self.update_cont(ply, pos.in_check(), idx, bonus * 750 / 1024);
        let pawn_idx = (pos.pawn_key as usize % PAWN_HIST_SIZE) * 1024 + idx;
        let scale = if bonus > -4 { 1104 } else { 459 };
        hist_update(&mut self.pawn_hist[pawn_idx], bonus * scale / 1024, PAWN_HIST_LIMIT);
    }

    /// The stored refutation of the opponent's last move, if any.
    #[inline(always)]
    fn counter_move(&self, ply: usize) -> Move {
        if ply >= 1 && self.ss[ply - 1].cont != NO_CONT {
            self.counter[self.ss[ply - 1].cont]
        } else {
            Move::NONE
        }
    }

    fn update_capture(&mut self, pos: &Position, m: Move, bonus: i32) {
        let pc = pos.board[m.from()] as usize;
        let victim = if m.flag() == FLAG_EP { PAWN } else { piece_type(pos.board[m.to()]) };
        hist_update(&mut self.capture_hist[pc][m.to()][victim], bonus, CAPTURE_HIST_LIMIT);
    }

    #[inline(always)]
    fn make(&mut self, ply: usize, m: Move) {
        let pc = self.pos[ply].board[m.from()] as usize;
        self.ss[ply].mv = m;
        self.ss[ply].cont = pc * 64 + m.to();
        self.pos[ply + 1] = self.pos[ply].make(m);
        let key = self.pos[ply + 1].key;
        self.tt.prefetch(key);
        self.keys.push(key);
        self.nnue.push(&self.pos, ply + 1);
        self.nodes += 1;
    }

    #[inline(always)]
    fn make_null(&mut self, ply: usize) {
        self.ss[ply].mv = Move::NONE;
        self.ss[ply].cont = NO_CONT;
        self.pos[ply + 1] = self.pos[ply].make_null();
        let key = self.pos[ply + 1].key;
        self.tt.prefetch(key);
        self.keys.push(key);
        self.nnue.push(&self.pos, ply + 1);
        self.nodes += 1;
    }

    /// Before a deep hash cutoff, check that the position after the hash move
    /// agrees: its own entry must not contradict the cutoff. This guards against
    /// scores carried over from transpositions with a different history.
    fn tt_cutoff_confirmed(&self, pos: &Position, tt_move: Move, tt_score: i32, beta: i32, depth: i32, ply: usize) -> bool {
        if depth < 7 || tt_move.is_none() || tt_score.abs() >= MATE_BOUND || !pos.is_legal(tt_move) {
            return true;
        }
        let next_key = pos.key_after(tt_move);
        match self.tt.probe(next_key) {
            None => true,
            Some(t) => {
                let s = score_from_tt(t.score, ply + 1);
                (tt_score >= beta) == (-s >= beta)
            }
        }
    }

    /// Futility and exchange pruning for a capture in the move loop.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn prune_capture(
        &self,
        pos: &Position,
        m: Move,
        depth: i32,
        static_eval: i32,
        alpha: i32,
        in_check: bool,
        gives_check: bool,
        capt_hist: i32,
        lmr: &LmrTable,
    ) -> bool {
        let victim = if m.flag() == FLAG_EP { PAWN } else { piece_type(pos.board[m.to()]) };
        if !gives_check && !in_check {
            let base = (lmr.0[depth.min(63) as usize][1] + 512) >> 10;
            let lmr_depth = (depth - 1 - base).max(0);
            if lmr_depth < 8 {
                let fut = static_eval
                    + tune::T_CapFutBase.get()
                    + tune::T_CapFutMult.get() * lmr_depth
                    + PIECE_VALUE[victim]
                    + 134 * capt_hist / 1024;
                if fut <= alpha {
                    return true;
                }
            }
        }
        // Exchange test, in centipawns (Stockfish's margin is in units of 208 per pawn).
        let margin = (tune::T_CapSeeMult.get() * depth + capt_hist * 34 / 1024) * 100 / 208;
        !self.stalemate_sac(pos, m, alpha) && !pos.see_ge(m, -margin)
    }

    /// Giving away the last piece can be a stalemate trick when we are worse.
    #[inline]
    fn stalemate_sac(&self, pos: &Position, m: Move, alpha: i32) -> bool {
        let us = pos.stm;
        let officers = pos.by_color[us] & !(pos.by_type[PAWN] | pos.by_type[KING]);
        alpha < 0 && officers == bit(m.from())
    }

    /// Moves that only shuffle a piece back and forth late in a quiet stretch.
    #[inline]
    fn is_shuffling(&self, ply: usize, m: Move) -> bool {
        let pos = &self.pos[ply];
        if m.is_noisy() || pos.rule50 < 10 || pos.plies_from_null < 6 || ply < 20 {
            return false;
        }
        let (m2, m4) = (self.ss[ply - 2].mv, self.ss[ply - 4].mv);
        !m2.is_none() && !m4.is_none() && m.from() == m2.to() && m2.from() == m4.to()
    }

    #[inline(always)]
    fn unmake(&mut self) {
        self.keys.pop();
    }

    fn is_draw(&self, ply: usize) -> bool {
        let pos = &self.pos[ply];
        if pos.rule50 >= 100 && (pos.checkers == 0 || has_legal_move(pos)) {
            return true;
        }
        if pos.insufficient_material() {
            return true;
        }
        let end = (pos.rule50 as usize).min(pos.plies_from_null as usize);
        let cur = self.keys.len() - 1;
        let mut reps = 0;
        let mut i = 4;
        while i <= end && i <= cur {
            if self.keys[cur - i] == pos.key {
                // A single repetition inside the search tree is enough.
                if i <= ply {
                    return true;
                }
                reps += 1;
                if reps == 2 {
                    return true;
                }
            }
            i += 2;
        }
        false
    }

    #[inline(always)]
    fn draw_score(&self) -> i32 {
        1 - (self.nodes & 2) as i32
    }

    fn check_time(&mut self) {
        self.shared.nodes[self.id].0.store(self.nodes, Relaxed);
        self.shared.tb_hits[self.id].0.store(self.tb_hits, Relaxed);
        if self.shared.stop.0.load(Relaxed) && (self.id != 0 || self.completed_depth > 0) {
            self.stopped = true;
            return;
        }
        if self.id != 0 || self.completed_depth == 0 || self.shared.pondering.0.load(Relaxed) {
            return;
        }
        let mut stop = self.hard_ms != u64::MAX && self.start.elapsed().as_millis() as u64 >= self.hard_ms;
        if let Some(n) = self.limits.nodes {
            stop |= self.shared.total_nodes(self.threads) >= n;
        }
        if stop {
            self.shared.stop.0.store(true, Relaxed);
            self.stopped = true;
        }
    }

    // -- quiescence search ----------------------------------------------------

    fn qsearch(&mut self, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        let pv_node = beta - alpha > 1;
        self.pv_len[ply] = 0;
        if self.nodes & 511 == 0 {
            self.check_time();
        }
        if self.stopped {
            return 0;
        }
        if pv_node && self.seldepth < ply + 1 {
            self.seldepth = ply + 1;
        }
        let pos = self.pos[ply];
        let in_check = pos.checkers != 0;
        if self.is_draw(ply) {
            return self.draw_score();
        }
        if ply >= MAX_PLY {
            return if in_check { 0 } else { self.evaluate(ply) };
        }

        let tte = self.tt.probe(pos.key);
        let mut tt_move = Move::NONE;
        let mut tt_pv = pv_node;
        if let Some(t) = tte {
            let s = score_from_tt(t.score, ply);
            if !pv_node && t.bound & (if s >= beta { BOUND_LOWER } else { BOUND_UPPER }) != 0 {
                return s;
            }
            tt_move = t.mv;
            tt_pv |= t.pv;
        }

        let raw_eval;
        let mut best;
        let futility_base;
        if in_check {
            raw_eval = NONE;
            best = -INF;
            futility_base = -INF;
        } else {
            raw_eval = match tte {
                Some(t) if t.eval != NONE => t.eval,
                _ => self.evaluate(ply),
            };
            best = (raw_eval + self.correction_at(ply)).clamp(-MATE_BOUND + 1, MATE_BOUND - 1);
            if let Some(t) = tte {
                let s = score_from_tt(t.score, ply);
                if s.abs() < MATE_BOUND && t.bound & (if s > best { BOUND_LOWER } else { BOUND_UPPER }) != 0 {
                    best = s;
                }
            }
            if best >= beta {
                if tte.is_none() {
                    self.tt.store(pos.key, Move::NONE, score_to_tt(best, ply), raw_eval, 0, BOUND_LOWER, tt_pv);
                }
                return best;
            }
            if best > alpha {
                alpha = best;
            }
            futility_base = best + if SF_QSEARCH { tune::T_QsFutBase.get() } else { 180 };
        }

        if !(pos.is_pseudo_legal(tt_move) && (in_check || tt_move.is_noisy())) {
            tt_move = Move::NONE;
        }
        let mut picker = Picker::new(tt_move, !in_check, 0);
        let mut best_move = Move::NONE;
        let mut move_count = 0;
        loop {
            let m = picker.next(self, ply, false);
            if m.is_none() {
                break;
            }
            if !pos.is_legal(m) {
                continue;
            }
            move_count += 1;
            if SF_QSEARCH && best > -MATE_BOUND && !in_check {
                // Late captures, futility and exchange tests, except for checks,
                // recaptures and promotions.
                let recapture = ply > 0 && !self.ss[ply - 1].mv.is_none() && m.to() == self.ss[ply - 1].mv.to();
                if !m.is_promo() && !recapture && !pos.gives_check(m) {
                    if move_count > 2 {
                        continue;
                    }
                    let victim = if m.flag() == FLAG_EP { PAWN } else { piece_type(pos.board[m.to()]) };
                    let fut = futility_base + PIECE_VALUE[victim];
                    if fut <= alpha {
                        best = best.max(fut);
                        continue;
                    }
                    let see_thresh = (alpha - futility_base) * 100 / NORMALIZE_PAWN;
                    if !pos.see_ge(m, see_thresh.max(-36)) {
                        if see_thresh >= -36 || !pos.see_ge(m, see_thresh) {
                            best = best.max(alpha.min(futility_base));
                        }
                        continue;
                    }
                } else if !pos.see_ge(m, -36) {
                    continue;
                }
            } else if best > -MATE_BOUND {
                if in_check {
                    // Once an evasion holds, don't bother with quiet ones.
                    if !m.is_noisy() {
                        break;
                    }
                } else {
                    if !m.is_promo() {
                        let victim = if m.flag() == FLAG_EP { PAWN } else { piece_type(pos.board[m.to()]) };
                        let fut = futility_base + SEE_VALUE[victim] * 2;
                        if fut <= alpha {
                            best = best.max(fut);
                            continue;
                        }
                    }
                }
                if !pos.see_ge(m, -80) {
                    continue;
                }
            }
            self.make(ply, m);
            let score = -self.qsearch(ply + 1, -beta, -alpha);
            self.unmake();
            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    best_move = m;
                    if score >= beta {
                        self.ss[ply].cutoff_cnt += 1;
                        break;
                    }
                    alpha = score;
                }
            }
        }
        if in_check && move_count == 0 {
            return -MATE + ply as i32;
        }
        let bound = if best >= beta { BOUND_LOWER } else { BOUND_UPPER };
        self.tt.store(pos.key, best_move, score_to_tt(best, ply), raw_eval, 0, bound, tt_pv);
        best
    }

    // -- main search ------------------------------------------------------------

    fn search(&mut self, ply: usize, mut alpha: i32, mut beta: i32, mut depth: i32, cut_node: bool) -> i32 {
        if depth <= 0 {
            return self.qsearch(ply, alpha, beta);
        }
        let pv_node = beta - alpha > 1;
        let root = ply == 0;
        self.pv_len[ply] = 0;
        if self.nodes & 511 == 0 {
            self.check_time();
        }
        if self.stopped {
            return 0;
        }
        if pv_node && self.seldepth < ply + 1 {
            self.seldepth = ply + 1;
        }
        let pos = self.pos[ply];
        let in_check = pos.checkers != 0;
        if !root {
            if self.is_draw(ply) {
                return self.draw_score();
            }
            if ply >= MAX_PLY {
                return if in_check { 0 } else { self.evaluate(ply) };
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let excluded = self.ss[ply].excluded;
        // Check extension
        if in_check && !root && excluded.is_none() && ply < 2 * self.completed_depth.max(1) as usize + 4 {
            depth += 1;
        }
        let tte = if excluded.is_none() { self.tt.probe(pos.key) } else { None };
        if excluded.is_none() {
            self.ss[ply].tt_hit = tte.is_some();
        }
        let mut tt_move = Move::NONE;
        let mut tt_score = NONE;
        let mut tt_depth = -1;
        let mut tt_bound = 0;
        let mut tt_pv = pv_node;
        if let Some(t) = tte {
            tt_score = score_from_tt(t.score, ply);
            tt_depth = t.depth;
            tt_bound = t.bound;
            tt_pv |= t.pv;
            if pos.is_pseudo_legal(t.mv) {
                tt_move = t.mv;
            }
            if !pv_node
                && tt_depth >= depth
                && tt_bound & (if tt_score >= beta { BOUND_LOWER } else { BOUND_UPPER }) != 0
                && (!SF_TT_CUTOFF || cut_node == (tt_score >= beta) || depth > 4)
                && pos.rule50 < if SF_TT_CUTOFF { 96 } else { 90 }
                && (!SF_TT_CUTOFF || self.tt_cutoff_confirmed(&pos, tt_move, tt_score, beta, depth, ply))
            {
                // Reward the hash move for causing a cutoff again.
                if tt_score >= beta && !tt_move.is_none() {
                    if !tt_move.is_noisy() {
                        self.update_quiet(ply, &pos, tt_move, 131 * depth);
                    }
                    // The opponent's early quiet move ran into a known refutation.
                    if ply >= 1 {
                        let prev = self.ss[ply - 1];
                        if prev.cont != NO_CONT && !prev.mv.is_capture() && prev.move_count < 5 {
                            let prev_in_check = self.pos[ply - 1].in_check();
                            self.update_cont(ply - 1, prev_in_check, prev.cont, -2210);
                        }
                    }
                }
                return tt_score;
            }
        }

        // Tablebase probe: an exact win/draw/loss verdict replaces searching.
        let mut tb_floor = -INF;
        let mut tb_ceiling = INF;
        if !root
            && self.tb_cardinality > 0
            && excluded.is_none()
            && depth >= self.tb.probe_depth
            && pos.rule50 == 0
            && pos.castling == 0
            && popcount(pos.occ()) <= self.tb_cardinality
        {
            if let Some(wdl) = tb::probe_wdl(&pos) {
                self.tb_hits += 1;
                let (v, bound) = match tb::outcome(wdl, &self.tb) {
                    1 => (TB_WIN - ply as i32, BOUND_LOWER),
                    -1 => (-TB_WIN + ply as i32, BOUND_UPPER),
                    _ => (0, BOUND_EXACT),
                };
                if bound == BOUND_EXACT
                    || (bound == BOUND_LOWER && v >= beta)
                    || (bound == BOUND_UPPER && v <= alpha)
                {
                    let d = (depth + 6).min(MAX_PLY as i32 - 1);
                    self.tt.store(pos.key, Move::NONE, score_to_tt(v, ply), NONE, d, bound, tt_pv);
                    return v;
                }
                // In PV nodes keep searching for the line, bounded by the verdict.
                if bound == BOUND_LOWER {
                    tb_floor = v;
                    alpha = alpha.max(v);
                } else {
                    tb_ceiling = v;
                }
            }
        }

        // Static evaluation
        let raw_eval;
        let static_eval;
        let mut eval;
        if in_check {
            raw_eval = NONE;
            static_eval = NONE;
            eval = NONE;
        } else if !excluded.is_none() {
            raw_eval = NONE;
            static_eval = self.ss[ply].static_eval;
            eval = static_eval;
        } else {
            raw_eval = match tte {
                Some(t) if t.eval != NONE => t.eval,
                _ => self.evaluate(ply),
            };
            static_eval = (raw_eval + self.correction_at(ply)).clamp(-MATE_BOUND + 1, MATE_BOUND - 1);
            eval = static_eval;
            if tte.is_some()
                && tt_score.abs() < MATE_BOUND
                && tt_bound & (if tt_score > eval { BOUND_LOWER } else { BOUND_UPPER }) != 0
            {
                eval = tt_score;
            }
        }
        self.ss[ply].static_eval = static_eval;

        // The change in static evaluation caused by the opponent's last quiet move
        // says something about that move's quality.
        if !in_check && excluded.is_none() && ply >= 1 {
            let prev = self.ss[ply - 1];
            if prev.cont != NO_CONT && !prev.mv.is_capture() && prev.static_eval != NONE {
                let diff = (-(prev.static_eval + static_eval)).clamp(-189, 194) + 60;
                hist_update(&mut self.butterfly[pos.stm ^ 1][prev.mv.from_to()], diff * 11, MAIN_HIST_LIMIT);
                if tte.is_none() && prev.cont / 64 & 7 != PAWN && !prev.mv.is_promo() {
                    let i = (pos.pawn_key as usize % PAWN_HIST_SIZE) * 1024 + prev.cont;
                    hist_update(&mut self.pawn_hist[i], diff * 13, PAWN_HIST_LIMIT);
                }
            }
        }
        let improving = !in_check && {
            if ply >= 2 && self.ss[ply - 2].static_eval != NONE {
                static_eval > self.ss[ply - 2].static_eval
            } else if ply >= 4 && self.ss[ply - 4].static_eval != NONE {
                static_eval > self.ss[ply - 4].static_eval
            } else {
                true
            }
        };
        self.ss[ply + 1].killers = [Move::NONE; 2];
        self.ss[ply + 1].cutoff_cnt = 0;
        self.ss[ply].double_ext = if ply > 0 { self.ss[ply - 1].double_ext } else { 0 };

        if !pv_node && !in_check && excluded.is_none() {
            // Reverse futility pruning (static null move)
            // The static evaluation is least trustworthy when the opponent has a pawn
            // about to promote, or when this position has been on a principal variation.
            let their_runners = pos.pieces(pos.stm ^ 1, PAWN)
                & if pos.stm == WHITE { RANK_2 | RANK_3 } else { RANK_6 | RANK_7 };
            let rfp_depth = if their_runners != 0 { RFP_RUNNER_DEPTH } else { RFP_MAX_DEPTH };
            if SF_RFP {
                if their_runners == 0
                    && !tt_pv
                    && depth < if self.seek_mate { 6 } else { 19 }
                    && eval >= beta
                    && (tt_move.is_none() || tt_move.is_noisy())
                    && eval < MATE_BOUND
                    && beta > -MATE_BOUND
                {
                    let mult = (45 + 4 * depth).min(85) - 20 * tte.is_none() as i32;
                    let margin = mult * depth - tune::T_RfpImproving.get() * improving as i32 * mult / 1024;
                    if eval - margin >= beta {
                        return (661 * beta + 363 * eval) / 1024;
                    }
                }
            } else if depth <= rfp_depth
                && !(RFP_SKIP_TT_PV && tt_pv)
                && eval - (if improving { 80 } else { 60 }) * depth >= beta
                && eval < MATE_BOUND
                && beta > -MATE_BOUND
            {
                return (eval + beta) / 2;
            }

            // Razoring
            if SF_RAZOR {
                if !self.seek_mate && eval < alpha - 485 - 280 * depth * depth {
                    return self.qsearch(ply, alpha, beta);
                }
            } else if depth <= 3 && static_eval + 150 + 125 * depth <= alpha {
                let v = self.qsearch(ply, alpha, alpha + 1);
                if v <= alpha {
                    return v;
                }
            }

            // Null move pruning
            let nmp_ok = if SF_NMP {
                cut_node && static_eval >= beta - 13 * depth - 47 * improving as i32 + 365
            } else {
                depth >= 3 && eval >= beta
            };
            if nmp_ok
                && ply > 0
                && !self.ss[ply - 1].mv.is_none()
                && pos.has_non_pawn_material(pos.stm)
                && ply >= self.nmp_min_ply
                && beta > -MATE_BOUND
            {
                // Total depth taken off the null-move search.
                let r = if SF_NMP {
                    7 + depth / 3 + ((static_eval - beta) / 256).max(0)
                } else {
                    1 + 3 + depth / 4 + !improving as i32 + ((eval - beta) / 200).clamp(0, NMP_EVAL_PLIES)
                };
                // Searching for a mate, null-move cutoffs are less trustworthy.
                let r = if self.seek_mate { r - (r - 1) / 2 } else { r };
                self.make_null(ply);
                let v = -self.search(ply + 1, -beta, -beta + 1, depth - r, !cut_node);
                self.unmake();
                if self.stopped {
                    return 0;
                }
                if v >= beta {
                    let v = if v >= MATE_BOUND { beta } else { v };
                    if self.nmp_min_ply != 0 || depth < if SF_NMP { 16 } else { 14 } {
                        return v;
                    }
                    // Verification search at high depth guards against zugzwang.
                    self.nmp_min_ply = ply + 3 * (depth - r).max(0) as usize / 4;
                    let verify = self.search(ply, beta - 1, beta, depth - r, false);
                    self.nmp_min_ply = 0;
                    if self.stopped {
                        return 0;
                    }
                    if verify >= beta {
                        return v;
                    }
                }
            }

            // ProbCut: a capture that beats beta by a margin at reduced depth
            let pc_beta = beta + 200 - 48 * improving as i32;
            if cut_node
                && depth >= 5
                && beta.abs() < MATE_BOUND
                && (tt_move.is_none() || tt_move.is_noisy())
                && !(tt_depth >= depth - 3 && tt_score != NONE && tt_score < pc_beta)
            {
                let see_margin = (pc_beta - static_eval) * 100 / NORMALIZE_PAWN;
                let tm = if tt_move.is_noisy() { tt_move } else { Move::NONE };
                let mut picker = Picker::new(tm, true, 0);
                loop {
                    let m = picker.next(self, ply, true);
                    if m.is_none() {
                        break;
                    }
                    if !pos.is_legal(m) || !pos.see_ge(m, see_margin) {
                        continue;
                    }
                    self.make(ply, m);
                    let mut v = -self.qsearch(ply + 1, -pc_beta, -pc_beta + 1);
                    if v >= pc_beta {
                        v = -self.search(ply + 1, -pc_beta, -pc_beta + 1, depth - 4, !cut_node);
                    }
                    self.unmake();
                    if self.stopped {
                        return 0;
                    }
                    if v >= pc_beta {
                        self.tt.store(pos.key, m, score_to_tt(v, ply), raw_eval, depth - 3, BOUND_LOWER, tt_pv);
                        return v;
                    }
                }
            }

            // Multi-cut: if several of the first moves each refute the node at
            // reduced depth, trust that one of them would at full depth too.
            if MULTI_CUT && cut_node && depth >= 6 && beta.abs() < MATE_BOUND {
                let mut picker = Picker::new(tt_move, false, 0);
                let (mut tried, mut cuts) = (0, 0);
                while tried < MC_CANDIDATES {
                    let m = picker.next(self, ply, false);
                    if m.is_none() {
                        break;
                    }
                    if !pos.is_legal(m) {
                        continue;
                    }
                    tried += 1;
                    self.make(ply, m);
                    let v = -self.search(ply + 1, -beta, -beta + 1, depth - 4, false);
                    self.unmake();
                    if self.stopped {
                        return 0;
                    }
                    if v >= beta {
                        cuts += 1;
                        if cuts >= MC_CUTOFFS {
                            return beta;
                        }
                    }
                }
            }
        }

        // Internal iterative reduction
        if (depth >= 4 && tt_move.is_none()) || (cut_node && depth >= 6 && tt_depth + 4 <= depth) {
            depth -= 1;
        }

        // Hash-entry ProbCut: the stored lower bound already beats beta by a wide
        // margin at nearly this depth.
        if SF_SMALL_PROBCUT && !pv_node && excluded.is_none() && !self.seek_mate {
            let pc_beta = beta + tune::T_SmallProbCut.get();
            if tt_bound & BOUND_LOWER != 0
                && tt_depth >= depth - 4
                && tt_score != NONE
                && tt_score >= pc_beta
                && beta.abs() < MATE_BOUND
                && tt_score.abs() < MATE_BOUND
            {
                return pc_beta;
            }
        }

        let lmr = lmr_table();
        let mut picker = Picker::new(tt_move, false, depth);
        let mut best = tb_floor;
        let mut best_move = Move::NONE;
        let mut move_count = 0;
        let mut skip_quiets = false;
        let mut quiets = [Move::NONE; 64];
        let mut n_quiets = 0;
        let mut noisies = [Move::NONE; 32];
        let mut n_noisies = 0;
        let orig_alpha = alpha;

        loop {
            let m = picker.next(self, ply, skip_quiets);
            if m.is_none() {
                break;
            }
            if m == excluded {
                continue;
            }
            if root
                && (self.root_excluded.contains(&m)
                    || (!self.limits.searchmoves.is_empty() && !self.limits.searchmoves.contains(&m)))
            {
                continue;
            }
            if !pos.is_legal(m) {
                continue;
            }
            move_count += 1;
            let moves_searched = move_count - 1;
            let quiet = !m.is_noisy();
            let good_capture = !quiet && picker.in_good_captures();

            // Late move pruning (checked before history/check lookups)
            if quiet
                && !root
                && !in_check
                && best > -MATE_BOUND
                && pos.has_non_pawn_material(pos.stm)
                && move_count > (3 + depth * depth) / (2 - improving as i32)
            {
                skip_quiets = true;
                continue;
            }

            let hist = if quiet {
                if picker.has_last_hist {
                    picker.last_hist
                } else {
                    self.quiet_score(ply, &pos, m)
                }
            } else {
                let victim = if m.flag() == FLAG_EP { PAWN } else { piece_type(pos.board[m.to()]) };
                self.capture_hist[pos.board[m.from()] as usize][m.to()][victim] as i32
            };
            let killers = self.ss[ply].killers;
            let is_killer = quiet && (m == killers[0] || m == killers[1]);
            let is_counter = quiet && !is_killer && m == self.counter_move(ply);
            let gives_check = pos.gives_check(m);

            // Move-loop pruning
            if !root && best > -MATE_BOUND && pos.has_non_pawn_material(pos.stm) {
                if quiet {
                    let protected = is_killer || is_counter;
                    let base = (lmr.0[depth.min(63) as usize][(move_count as usize).min(63)] + 512) >> 10;
                    let lmr_depth =
                        (depth - 1 - base + ((hist - QUIET_HIST_CENTRE) / QUIET_HIST_STEP).clamp(-2, 2)).max(0);
                    if SF_FUTILITY && !in_check && !gives_check {
                        // Futility on the depth this move would be searched at after
                        // reductions, adjusted by its history.
                        let fut = static_eval
                            + tune::T_FutMult.get() * lmr_depth
                            + tune::T_FutAboveAlpha.get() * (static_eval > alpha) as i32
                            + tune::T_FutBase.get();
                        if lmr_depth < 12 && fut <= alpha {
                            if best <= fut && fut.abs() < MATE_BOUND {
                                best = fut;
                            }
                            continue;
                        }
                    }
                    // Futility pruning
                    if !SF_FUTILITY
                        && depth <= 6
                        && !in_check
                        && !gives_check
                        && !protected
                        && static_eval + (if improving { 80 } else { 60 }) * depth <= alpha
                    {
                        continue;
                    }
                    // History pruning
                    if lmr_depth <= 6 && !protected && hist < -HISTORY_PRUNE_MARGIN * lmr_depth.max(1) {
                        continue;
                    }
                    // SEE pruning
                    if !pos.see_ge(m, -25 * lmr_depth * lmr_depth) {
                        continue;
                    }
                } else if SF_CAPTURE_PRUNE {
                    if self.prune_capture(&pos, m, depth, static_eval, alpha, in_check, gives_check, hist, lmr) {
                        continue;
                    }
                } else if !good_capture && !pos.see_ge(m, -90 * depth) {
                    continue;
                }
                // Checking quiet moves: the same exchange test as captures.
                if SF_CAPTURE_PRUNE && quiet && gives_check {
                    let margin = tune::T_CapSeeMult.get() * depth * 100 / 208;
                    if !self.stalemate_sac(&pos, m, alpha) && !pos.see_ge(m, -margin) {
                        continue;
                    }
                }
            }

            // Singular extension: is the hash move much better than everything else?
            let mut ext = 0;
            if SF_SINGULAR {
                if !root
                    && depth >= 6 + tt_pv as i32
                    && m == tt_move
                    && excluded.is_none()
                    && tt_depth >= depth - 3
                    && tt_bound & BOUND_LOWER != 0
                    && tt_score.abs() < MATE_BOUND
                    && !(SF_SHUFFLING && self.is_shuffling(ply, m))
                    && !self.seek_mate
                    && ply < 2 * self.completed_depth.max(1) as usize + 4
                {
                    let margin = (tune::T_SingularMargin.get() + 66 * (tt_pv && !pv_node) as i32) * depth / 63;
                    let s_beta = tt_score - margin;
                    self.ss[ply].excluded = m;
                    let v = self.search(ply, s_beta - 1, s_beta, (depth - 1) / 2, cut_node);
                    self.ss[ply].excluded = Move::NONE;
                    if self.stopped {
                        return 0;
                    }
                    let tt_capture = tt_move.is_noisy();
                    if v < s_beta {
                        let corr_adj = if SF_CORRECTION { self.correction_raw(ply).abs() / 198368 } else { 0 };
                        let deep = (ply as i32 > self.root_depth) as i32;
                        let double = tune::T_DoubleExtMargin.get() + 204 * pv_node as i32
                            - 152 * !tt_capture as i32
                            - corr_adj
                            - 1175 * self.tt_move_hist as i32 / 114178
                            - 38 * deep;
                        let triple = tune::T_TripleExtMargin.get() + 279 * pv_node as i32 - 188 * !tt_capture as i32
                            + 81 * tt_pv as i32
                            - corr_adj
                            - 43 * deep;
                        if !pv_node && self.ss[ply].double_ext <= 8 {
                            ext = 1 + (v < s_beta - double) as i32 + (v < s_beta - triple) as i32;
                            if ext >= 2 {
                                self.ss[ply].double_ext += 1;
                            }
                        } else {
                            ext = 1;
                        }
                    } else if v >= beta && v.abs() < MATE_BOUND {
                        // Multi-cut: other moves beat beta too.
                        hist_update(&mut self.tt_move_hist, -421 - 110 * depth, 8192);
                        if SF_CORRECTION && !in_check && v > static_eval {
                            let bonus = ((v - static_eval) * 664 / 1024).clamp(-SF_CORR_LIMIT / 4, SF_CORR_LIMIT / 4);
                            self.update_correction_sf(ply, bonus);
                        }
                        return v;
                    } else if tt_score >= beta || cut_node {
                        ext = -3;
                    }
                }
            } else if !root
                && depth >= 7
                && m == tt_move
                && excluded.is_none()
                && tt_depth >= depth - 3
                && tt_bound & BOUND_LOWER != 0
                && tt_score.abs() < MATE_BOUND
                && ply < 2 * self.completed_depth.max(1) as usize + 4
            {
                let s_beta = tt_score - depth * 2;
                self.ss[ply].excluded = m;
                let v = self.search(ply, s_beta - 1, s_beta, (depth - 1) / 2, cut_node);
                self.ss[ply].excluded = Move::NONE;
                if self.stopped {
                    return 0;
                }
                if v < s_beta {
                    ext = 1;
                    if !pv_node && v < s_beta - 30 && self.ss[ply].double_ext <= 8 {
                        ext = 2;
                        self.ss[ply].double_ext += 1;
                    }
                } else if s_beta >= beta {
                    // Multi-cut: several moves beat beta.
                    return s_beta;
                } else if tt_score >= beta {
                    ext = -2;
                } else if cut_node {
                    ext = -1;
                }
            }

            self.ss[ply].move_count = move_count;
            self.make(ply, m);
            debug_assert_eq!(gives_check, self.pos[ply + 1].checkers != 0);
            let new_depth = depth - 1 + ext;
            let nodes_before = self.nodes;
            let mut score;

            // Late move reductions: quiets from the third move on, captures later.
            let do_lmr =
                if quiet { depth >= 3 && moves_searched >= 2 } else { depth >= 4 && moves_searched >= 4 };
            if do_lmr {
                let base = lmr.0[depth.min(63) as usize][(move_count as usize).min(63)];
                let mut r;
                if quiet {
                    r = base;
                    r += if pv_node { -1024 } else { 1024 };
                    r += if improving { -1024 } else { 1024 };
                    r += if tt_pv { -512 } else { 1024 };
                    if cut_node {
                        r += if tt_move.is_none() { 1024 } else { 1536 };
                    }
                    if !tt_move.is_none() && tt_move.is_noisy() {
                        r += 1024;
                    }
                    if self.ss[ply + 1].cutoff_cnt > 3 {
                        r += 1024;
                    }
                    if is_killer {
                        r -= 2048;
                    } else if is_counter {
                        r -= 1024;
                    }
                    r -= ((hist - QUIET_HIST_CENTRE) * 1024 / QUIET_HIST_STEP).clamp(-2048, 2560);
                } else {
                    r = base / 2;
                    r += if tt_pv { -512 } else { 1024 };
                    if cut_node {
                        r += 1024;
                    }
                    r -= ((hist - CAPTURE_HIST_CENTRE) * 1024 / CAPTURE_HIST_STEP).clamp(-1024, 1536);
                    if !good_capture && !pos.see_ge(m, 0) {
                        r += 1024;
                    }
                }
                if in_check {
                    r -= 1024;
                }
                if gives_check {
                    r -= 1024;
                }
                // When looking for a mate, reduce checks less and other moves more.
                if SF_SEEK_MATE && self.seek_mate && r > 0 {
                    r += (if gives_check { -r } else { r }) / 4;
                }
                let r_plies = (r + 512) >> 10;
                let d = if SF_LMR_EXTEND {
                    // A negative reduction may extend by 1 ply near the root.
                    let cap = if (ply as i32) < 2 * self.root_depth { 1 } else { 0 };
                    (new_depth + (-r_plies).min(cap)).max(1)
                } else {
                    (new_depth - r_plies.max(0)).clamp(1, new_depth.max(1))
                };
                score = -self.search(ply + 1, -alpha - 1, -alpha, d, true);
                // A reduced search that fails high is repeated at adjusted full depth.
                if score > alpha && d < new_depth {
                    let do_deeper = score > best + 50 + 2 * new_depth;
                    let do_shallower = score < best + new_depth;
                    let re_depth = (new_depth + do_deeper as i32 - do_shallower as i32).max(1);
                    score = -self.search(ply + 1, -alpha - 1, -alpha, re_depth, !cut_node);
                    // The move did better than its late position suggested.
                    let idx = pos.board[m.from()] as usize * 64 + m.to();
                    self.update_cont(ply, in_check, idx, 1334);
                }
            } else if !pv_node || move_count > 1 {
                score = -self.search(ply + 1, -alpha - 1, -alpha, new_depth, !cut_node);
            } else {
                score = alpha + 1;
            }
            if pv_node && (move_count == 1 || (score > alpha && (root || score < beta))) {
                score = -self.search(ply + 1, -beta, -alpha, new_depth, false);
            }
            self.unmake();
            if self.stopped {
                return 0;
            }
            if root {
                self.root_nodes[m.from_to()] += self.nodes - nodes_before;
            }

            if score > best {
                best = score;
                if score > alpha {
                    if root && move_count > 1 {
                        self.shared.best_move_changes.0.fetch_add(1, Relaxed);
                    }
                    best_move = m;
                    if pv_node {
                        let n = self.pv_len[ply + 1];
                        let (a, b) = self.pv.split_at_mut(ply + 1);
                        a[ply][0] = m;
                        a[ply][1..=n].copy_from_slice(&b[0][..n]);
                        self.pv_len[ply] = n + 1;
                    }
                    if score >= beta {
                        self.ss[ply].cutoff_cnt += 1;
                        break;
                    }
                    alpha = score;
                }
            }
            if m != best_move {
                if quiet {
                    if n_quiets < quiets.len() {
                        quiets[n_quiets] = m;
                        n_quiets += 1;
                    }
                } else if n_noisies < noisies.len() {
                    noisies[n_noisies] = m;
                    n_noisies += 1;
                }
            }
        }

        if move_count == 0 {
            return if !excluded.is_none() {
                alpha
            } else if in_check {
                -MATE + ply as i32
            } else {
                0
            };
        }

        if best >= beta {
            // The more moves had to be tried first, the more the refutation is worth.
            let mut bonus = stat_bonus(depth) + 364 * (best_move == tt_move) as i32;
            bonus += bonus * (n_quiets + n_noisies) as i32 / 256;
            let malus = stat_malus(depth);
            if !best_move.is_noisy() {
                let k = &mut self.ss[ply].killers;
                if k[0] != best_move {
                    k[1] = k[0];
                    k[0] = best_move;
                }
                if ply >= 1 && self.ss[ply - 1].cont != NO_CONT {
                    let c = self.ss[ply - 1].cont;
                    self.counter[c] = best_move;
                }
                self.update_quiet(ply, &pos, best_move, bonus * 899 / 1024);
                // Earlier quiets are penalised most, later ones progressively less.
                let mut m = malus * 1159 / 1024;
                for &q in &quiets[..n_quiets] {
                    m = m * 921 / 1024;
                    self.update_quiet(ply, &pos, q, -m);
                }
            } else {
                self.update_capture(&pos, best_move, bonus * 1427 / 1024);
            }
            // The opponent's previous quiet move was an early try and got refuted.
            if ply >= 1 {
                let prev = self.ss[ply - 1];
                if prev.cont != NO_CONT && !prev.mv.is_capture() && prev.move_count == 1 + prev.tt_hit as i32 {
                    let prev_in_check = self.pos[ply - 1].in_check();
                    self.update_cont(ply - 1, prev_in_check, prev.cont, -malus * 713 / 1024);
                }
            }
            for &n in &noisies[..n_noisies] {
                self.update_capture(&pos, n, -malus * 1489 / 1024);
            }
        }

        // Nothing here beat alpha, so the opponent's previous move did well.
        if best < beta && best_move.is_none() && excluded.is_none() && ply >= 1 {
            let prev = self.ss[ply - 1];
            if prev.cont != NO_CONT && !prev.mv.is_capture() {
                let prev_in_check = self.pos[ply - 1].in_check();
                let mut scale = -241 + (59 * depth).min(420) + 186 * (prev.move_count > 9) as i32;
                scale += 142 * (!in_check && best <= static_eval - 106) as i32;
                scale += 159 * (!prev_in_check && prev.static_eval != NONE && best <= -prev.static_eval - 68) as i32;
                let scaled = (150 * depth - 85).min(1337) * scale.max(0);
                self.update_cont(ply - 1, prev_in_check, prev.cont, scaled * 263 / 16384);
                hist_update(&mut self.butterfly[pos.stm ^ 1][prev.mv.from_to()], scaled * 215 / 32768, MAIN_HIST_LIMIT);
                if prev.cont / 64 & 7 != PAWN && !prev.mv.is_promo() {
                    let i = (pos.pawn_key as usize % PAWN_HIST_SIZE) * 1024 + prev.cont;
                    hist_update(&mut self.pawn_hist[i], scaled * 324 / 8192, PAWN_HIST_LIMIT);
                }
            } else if prev.cont != NO_CONT {
                // A capture: credit it in the capture history.
                let before = &self.pos[ply - 1];
                let to = prev.mv.to();
                let victim = if prev.mv.flag() == FLAG_EP { PAWN } else { piece_type(before.board[to]) };
                hist_update(&mut self.capture_hist[pos.board[to] as usize][to][victim], 892, CAPTURE_HIST_LIMIT);
            }
        }
        if !pv_node && !best_move.is_none() && excluded.is_none() {
            hist_update(&mut self.tt_move_hist, if best_move == tt_move { 918 } else { -747 }, 8192);
        }
        best = best.min(tb_ceiling);
        if excluded.is_none() {
            let bound = if best >= beta {
                BOUND_LOWER
            } else if pv_node && best > orig_alpha {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            if SF_CORRECTION {
                // Update when the result lies on the side of the static eval that the
                // outcome (a best move or none) says it should.
                if !in_check
                    && !(!best_move.is_none() && best_move.is_capture())
                    && (best > static_eval) == !best_move.is_none()
                    && best.abs() < MATE_BOUND
                {
                    let k = if best_move.is_none() { 18 } else { 12 };
                    let bonus = ((best - static_eval) * depth * k / 128).clamp(-SF_CORR_LIMIT / 4, SF_CORR_LIMIT / 4);
                    self.update_correction_sf(ply, 1061 * bonus / 1024);
                }
            } else if !in_check
                && (best_move.is_none() || !best_move.is_noisy())
                && !(bound == BOUND_LOWER && best <= static_eval)
                && !(bound == BOUND_UPPER && best >= static_eval)
                && best.abs() < MATE_BOUND
            {
                self.update_correction(&pos, best - static_eval, depth);
            }
            self.tt.store(pos.key, best_move, score_to_tt(best, ply), raw_eval, depth, bound, tt_pv);
        }
        best
    }

    // -- iterative deepening ------------------------------------------------------

    fn print_info(&self, depth: i32, idx: usize, line: &Line, bound: &str) {
        if self.silent {
            return;
        }
        let ms = self.start.elapsed().as_millis().max(1) as u64;
        let nodes = self.shared.total_nodes(self.threads).max(self.nodes);
        let pv: Vec<String> = line.pv.iter().map(|m| m.to_uci()).collect();
        let wdl = if self.show_wdl {
            let (w, d, l) = score_to_wdl(line.score, &self.pos[0]);
            format!(" wdl {w} {d} {l}")
        } else {
            String::new()
        };
        println!(
            "info depth {} seldepth {} multipv {} score {}{}{} nodes {} nps {} hashfull {} tbhits {} time {} pv {}",
            depth,
            line.seldepth.max(1),
            idx + 1,
            score_to_uci(line.score),
            bound,
            wdl,
            nodes,
            nodes * 1000 / ms,
            self.tt.hashfull(),
            self.shared.total_tb_hits(self.threads).max(self.tb_hits),
            ms,
            pv.join(" ")
        );
    }

    pub fn iterate(&mut self) {
        let root = self.pos[0];
        let mut root_moves = legal_moves(&root);
        if !self.limits.searchmoves.is_empty() {
            root_moves.retain(|m| self.limits.searchmoves.contains(m));
        }
        if root_moves.is_empty() {
            return;
        }
        self.best_move = root_moves[0];
        let main = self.id == 0;
        let multipv = if main { self.multipv.min(root_moves.len()).max(1) } else { 1 };
        let max_depth = self.limits.depth.unwrap_or(MAX_PLY as i32 - 1).clamp(1, MAX_PLY as i32 - 1);
        let mut prev_scores = vec![0i32; multipv];
        let mut avg_scores = vec![NONE; multipv];
        let mut stability = 0u32;
        let mut last_best = Move::NONE;
        // Time management state (main thread).
        let mut last_best_depth = 0;
        let start_score = if self.prev_best_score == NONE { 0 } else { self.prev_best_score };
        let mut iter_values = [start_score; 4];
        let mut iter_idx = 0;
        let mut time_reduction = 1.0f64;
        let mut tot_changes = 0.0f64;

        let start_depth = if !main && max_depth > 1 && self.id % 2 == 1 { 2 } else { 1 };
        for depth in start_depth..=max_depth {
            // Helpers skip iterations in a staggered pattern, so the threads spread
            // over neighbouring depths and feed the shared table from different angles.
            if !main && depth > 1 && depth < max_depth {
                const SKIP_SIZE: [i32; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
                const SKIP_PHASE: [i32; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];
                let i = (self.id - 1) % 20;
                if ((depth + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 != 0 {
                    continue;
                }
            }
            let mut lines: Vec<Line> = Vec::with_capacity(multipv);
            self.root_excluded.clear();
            for pv_idx in 0..multipv {
                self.seldepth = 0;
                self.root_depth = depth;
                let prev = prev_scores[pv_idx];
                let have_prev = depth > start_depth;
                // Look harder for a mate once the root score is far from equal.
                let mate_threshold = 750 + 220000 / (depth * depth);
                self.seek_mate = SF_SEEK_MATE && have_prev && prev.abs() >= mate_threshold;
                // Optimism follows the running average score of the root move.
                if SF_EVAL_SCALE {
                    let avg = if avg_scores[pv_idx] == NONE { prev } else { avg_scores[pv_idx] };
                    let opt = if have_prev {
                        tune::T_OptimismScale.get() * avg / (avg.abs() + tune::T_OptimismDiv.get())
                    } else {
                        0
                    };
                    self.optimism[root.stm] = opt;
                    self.optimism[root.stm ^ 1] = -opt;
                }
                let mut delta = 14 + (self.id as i32 % 4) * 3 + prev * prev / 24000;
                let (mut alpha, mut beta) = (-INF, INF);
                if depth >= 4 {
                    alpha = (prev - delta).max(-INF);
                    beta = (prev + delta).min(INF);
                }
                let mut fail_highs = 0;
                let score = loop {
                    let s = self.search(0, alpha, beta, (depth - fail_highs).max(1), false);
                    if self.stopped {
                        break s;
                    }
                    let label = if s <= alpha { " upperbound" } else { " lowerbound" };
                    if s <= alpha {
                        beta = (alpha + beta) / 2;
                        alpha = (s - delta).max(-INF);
                        fail_highs = 0;
                    } else if s >= beta {
                        beta = (s + delta).min(INF);
                        if s.abs() < MATE_BOUND {
                            fail_highs = (fail_highs + 1).min(3);
                        }
                    } else {
                        break s;
                    }
                    if main && self.start.elapsed().as_millis() > 2500 && self.pv_len[0] > 0 {
                        let line = Line { score: s, seldepth: self.seldepth, pv: self.pv[0][..self.pv_len[0]].to_vec() };
                        self.print_info(depth, pv_idx, &line, label);
                    }
                    delta += delta / 2 + 4;
                    if delta > 1500 {
                        alpha = -INF;
                        beta = INF;
                    }
                };
                if self.stopped {
                    // A move that was fully searched at this depth and beat the previous
                    // best is still worth playing, even though the iteration was cut short.
                    if pv_idx == 0 && self.completed_depth > 0 && self.pv_len[0] > 0 {
                        self.best_move = self.pv[0][0];
                        self.ponder_move = if self.pv_len[0] > 1 { self.pv[0][1] } else { Move::NONE };
                    }
                    break;
                }
                if self.pv_len[0] == 0 {
                    break;
                }
                let pv = self.pv[0][..self.pv_len[0]].to_vec();
                prev_scores[pv_idx] = score;
                avg_scores[pv_idx] = if avg_scores[pv_idx] == NONE || score.abs() >= MATE_BOUND {
                    score
                } else {
                    (score + avg_scores[pv_idx]) / 2
                };
                self.root_excluded.push(pv[0]);
                lines.push(Line { score, seldepth: self.seldepth, pv });
            }
            self.root_excluded.clear();
            if self.stopped || lines.len() < multipv {
                break;
            }
            lines.sort_by_key(|l| -l.score);
            for (i, l) in lines.iter().enumerate() {
                prev_scores[i] = l.score;
            }
            self.completed_depth = depth;
            let prev_score = self.best_score;
            self.best_move = lines[0].pv[0];
            self.ponder_move = lines[0].pv.get(1).copied().unwrap_or(Move::NONE);
            self.best_score = lines[0].score;
            self.best_pv.clear();
            self.best_pv.extend_from_slice(&lines[0].pv);
            self.best_seldepth = lines[0].seldepth;
            if self.best_move == last_best {
                stability += 1;
            } else {
                stability = 0;
                last_best = self.best_move;
                last_best_depth = depth;
            }
            if !main {
                continue;
            }
            for (i, l) in lines.iter().enumerate() {
                self.print_info(depth, i, l, "");
            }

            // Stockfish's time management: scale the optimum time by how much the
            // score is falling, how long the best move has been stable, how often it
            // changed across all threads, and how much of the effort it took.
            if SF_TIME && self.soft_ms != u64::MAX && !self.shared.pondering.0.load(Relaxed) {
                let best = self.best_score;
                let prev_avg = if self.prev_avg_score == NONE { best } else { self.prev_avg_score };
                let falling = ((11.48 + 2.30 * (prev_avg - best) as f64 + 1.1 * (iter_values[iter_idx] - best) as f64)
                    / 100.0)
                    .clamp(0.576, 1.728);
                time_reduction =
                    interpolate((depth - last_best_depth) as f64, 4.96, 18.79, 0.639, 1.712).clamp(0.629, 1.544);
                let reduction = (1.468 + self.prev_time_reduction) / (2.284 * time_reduction);
                tot_changes = tot_changes / 2.0 + self.shared.best_move_changes.0.swap(0, Relaxed) as f64;
                let instability = 1.077 + 2.229 * tot_changes / self.threads as f64;
                let effort = self.root_nodes[self.best_move.from_to()] as f64 * 100000.0 / self.nodes.max(1) as f64;
                let high_effort = interpolate(effort, 75800.0, 104510.0, 0.969, 0.714).clamp(0.693, 0.838);
                let mut total = self.soft_ms as f64 * falling * reduction * instability * high_effort;
                if root_moves.len() == 1 {
                    total = total.min(500.0);
                }
                iter_values[iter_idx] = best;
                iter_idx = (iter_idx + 1) & 3;
                let elapsed = self.start.elapsed().as_millis() as f64;
                if elapsed > total.min(self.hard_ms as f64) || best >= MATE - 3 || best == -(MATE - 2) {
                    break;
                }
            } else if self.soft_ms != u64::MAX && !self.shared.pondering.0.load(Relaxed) {
                let elapsed = self.start.elapsed().as_millis() as f64;
                let mut scale = 1.0;
                // Root move dominance: share of the nodes spent on the best move.
                let frac = self.root_nodes[self.best_move.from_to()] as f64 / self.nodes.max(1) as f64;
                if frac >= DOMINANT_SHARE {
                    scale *= 0.75;
                } else if frac < DISPUTED_SHARE {
                    scale *= 1.30;
                }
                // Best move stability: the same move for several iterations in a row.
                if stability >= STABLE_ITERATIONS {
                    scale *= 0.80;
                }
                // Fail-low recovery: the score dropped, look for a way out.
                if depth > 4 && self.best_score < prev_score - FAIL_LOW_DROP {
                    scale *= 1.40;
                }
                if elapsed >= self.soft_ms as f64 * scale {
                    break;
                }
            }
            if let Some(n) = self.limits.nodes {
                if self.shared.total_nodes(self.threads) >= n {
                    break;
                }
            }
        }
        if main {
            self.prev_time_reduction = time_reduction;
            if self.completed_depth > 0 {
                self.prev_avg_score = avg_scores[0];
                self.prev_best_score = self.best_score;
            }
        }
        self.shared.nodes[self.id].0.store(self.nodes, Relaxed);
        self.shared.tb_hits[self.id].0.store(self.tb_hits, Relaxed);
    }
}

#[derive(Clone)]
struct ThreadResult {
    completed_depth: i32,
    best_score: i32,
    best_seldepth: usize,
    best_pv: Vec<Move>,
}

impl ThreadResult {
    fn from_searcher(s: &Searcher) -> Self {
        ThreadResult {
            completed_depth: s.completed_depth,
            best_score: s.best_score,
            best_seldepth: s.best_seldepth,
            best_pv: s.best_pv.clone(),
        }
    }
}

/// Pick the thread whose result to play. Every thread votes for its best move
/// with a weight growing with its score and completed depth; the move with the
/// most votes wins, represented by the strongest thread that chose it. Proven
/// mates take precedence, shortest first.
fn best_thread(results: &[ThreadResult], multipv: usize) -> usize {
    let valid = |r: &ThreadResult| r.completed_depth > 0 && !r.best_pv.is_empty();
    if multipv > 1 || !valid(&results[0]) {
        return 0;
    }
    let min_score = results.iter().filter(|r| valid(r)).map(|r| r.best_score).min().unwrap_or(0);
    let weight = |r: &ThreadResult| (r.best_score - min_score + 14) as i64 * r.completed_depth as i64;
    let mut votes = vec![0i64; 4096];
    for r in results.iter().filter(|r| valid(r)) {
        votes[r.best_pv[0].from_to()] += weight(r);
    }
    let mut best = 0;
    for (i, r) in results.iter().enumerate().skip(1) {
        if !valid(r) {
            continue;
        }
        let b = &results[best];
        let (vs, vb) = (votes[r.best_pv[0].from_to()], votes[b.best_pv[0].from_to()]);
        let better = if b.best_score >= MATE_BOUND {
            r.best_score > b.best_score
        } else if r.best_score >= MATE_BOUND {
            true
        } else {
            r.best_score > -MATE_BOUND && (vs > vb || (vs == vb && weight(r) > weight(b)))
        };
        if better {
            best = i;
        }
    }
    best
}

enum WorkerJob {
    Idle,
    Search {
        pos: Position,
        keys: Arc<[u64]>,
        limits: Limits,
        overhead: u64,
        tb: TbConfig,
        show_wdl: bool,
    },
    Clear,
    ClearTT(Arc<TT>),
    Quit,
}

struct WorkerState {
    gen: u64,
    job: WorkerJob,
    done: bool,
    result: Option<ThreadResult>,
}

struct WorkerHandle {
    state: Arc<(Mutex<WorkerState>, Condvar)>,
    join: Option<JoinHandle<()>>,
}

/// Persistent worker pool: helper threads stay parked between searches and
/// allocate their own `Searcher` inside their own thread (first-touch locality).
pub struct ThreadPool {
    pub main: Box<Searcher>,
    helpers: Vec<WorkerHandle>,
    threads: usize,
}

impl ThreadPool {
    pub fn new(threads: usize, net: Arc<Network>, tt: Arc<TT>, shared: Arc<Shared>) -> ThreadPool {
        let main = Searcher::new(0, threads, net.clone(), tt.clone(), shared.clone());
        let mut helpers = Vec::with_capacity(threads.saturating_sub(1));
        for id in 1..threads {
            let state = Arc::new((
                Mutex::new(WorkerState { gen: 0, job: WorkerJob::Idle, done: true, result: None }),
                Condvar::new(),
            ));
            let st = state.clone();
            let net = net.clone();
            let tt = tt.clone();
            let shared = shared.clone();
            let join = std::thread::Builder::new()
                .stack_size(32 * 1024 * 1024)
                .spawn(move || {
                    let mut searcher = Searcher::new(id, threads, net, tt, shared);
                    let mut last_gen = 0;
                    loop {
                        let job = {
                            let (lock, cv) = &*st;
                            let mut guard = lock.lock().unwrap();
                            while guard.gen == last_gen {
                                guard = cv.wait(guard).unwrap();
                            }
                            last_gen = guard.gen;
                            std::mem::replace(&mut guard.job, WorkerJob::Idle)
                        };
                        let result = match job {
                            WorkerJob::Idle => None,
                            WorkerJob::Search { pos, keys, limits, overhead, tb, show_wdl } => {
                                searcher.prepare(&pos, &keys, &limits, 1, overhead, tb);
                                searcher.silent = true;
                                searcher.show_wdl = show_wdl;
                                searcher.iterate();
                                Some(ThreadResult::from_searcher(&searcher))
                            }
                            WorkerJob::Clear => {
                                searcher.clear();
                                None
                            }
                            WorkerJob::ClearTT(tt) => {
                                let len = tt.len();
                                let start = id * len / threads;
                                let end = (id + 1) * len / threads;
                                tt.clear_range(start, end);
                                None
                            }
                            WorkerJob::Quit => break,
                        };
                        let (lock, cv) = &*st;
                        let mut guard = lock.lock().unwrap();
                        guard.result = result;
                        guard.done = true;
                        cv.notify_all();
                    }
                })
                .expect("failed to spawn search worker thread");
            helpers.push(WorkerHandle { state, join: Some(join) });
        }
        ThreadPool { main, helpers, threads }
    }

    fn dispatch(&self, mut make_job: impl FnMut(usize) -> WorkerJob) {
        for (idx, h) in self.helpers.iter().enumerate() {
            let (lock, cv) = &*h.state;
            let mut guard = lock.lock().unwrap();
            guard.gen += 1;
            guard.done = false;
            guard.result = None;
            guard.job = make_job(idx + 1);
            cv.notify_all();
        }
    }

    fn wait_helpers(&self) -> Vec<ThreadResult> {
        let mut results = Vec::with_capacity(self.helpers.len());
        for h in &self.helpers {
            let (lock, cv) = &*h.state;
            let mut guard = lock.lock().unwrap();
            while !guard.done {
                guard = cv.wait(guard).unwrap();
            }
            if let Some(r) = guard.result.take() {
                results.push(r);
            }
        }
        results
    }

    pub fn clear(&mut self) {
        self.dispatch(|_| WorkerJob::Clear);
        self.main.clear();
        let _ = self.wait_helpers();
    }

    pub fn clear_tt(&self, tt: &Arc<TT>) {
        self.dispatch(|_| WorkerJob::ClearTT(tt.clone()));
        let len = tt.len();
        tt.clear_range(0, len / self.threads);
        let _ = self.wait_helpers();
        tt.reset_gen();
    }

    /// Run a search on all threads; returns (best move, ponder move, total nodes).
    pub fn run(
        &mut self,
        pos: &Position,
        keys: &[u64],
        limits: &Limits,
        multipv: usize,
        overhead: u64,
        tb: TbConfig,
        silent: bool,
        show_wdl: bool,
        shared: &Arc<Shared>,
    ) -> (Move, Move, u64) {
        for n in shared.nodes[..self.threads].iter().chain(&shared.tb_hits[..self.threads]) {
            n.0.store(0, Relaxed);
        }
        shared.best_move_changes.0.store(0, Relaxed);
        self.main.tt.new_search();
        self.main.prepare(pos, keys, limits, multipv, overhead, tb);
        self.main.silent = silent;
        self.main.show_wdl = show_wdl;

        let shared_keys: Arc<[u64]> = Arc::from(keys);
        self.dispatch(|_| WorkerJob::Search {
            pos: *pos,
            keys: shared_keys.clone(),
            limits: limits.clone(),
            overhead,
            tb,
            show_wdl,
        });

        self.main.iterate();
        while (shared.pondering.0.load(Relaxed) || self.main.limits.infinite) && !shared.stop.0.load(Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        shared.stop.0.store(true, Relaxed);

        let helper_results = self.wait_helpers();
        let nodes = shared.total_nodes(self.threads);

        let mut results = Vec::with_capacity(1 + helper_results.len());
        results.push(ThreadResult::from_searcher(&self.main));
        results.extend(helper_results);

        let best = best_thread(&results, multipv);
        if best == 0 {
            return (self.main.best_move, self.main.ponder_move, nodes);
        }
        let r = &results[best];
        self.main.best_move = r.best_pv[0];
        self.main.ponder_move = r.best_pv.get(1).copied().unwrap_or(Move::NONE);
        if !silent {
            let line = Line { score: r.best_score, seldepth: r.best_seldepth, pv: r.best_pv.clone() };
            self.main.print_info(r.completed_depth, 0, &line, "");
        }
        (self.main.best_move, self.main.ponder_move, nodes)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.dispatch(|_| WorkerJob::Quit);
        for h in &mut self.helpers {
            if let Some(j) = h.join.take() {
                let _ = j.join();
            }
        }
    }
}
