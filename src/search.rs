//! Alpha-beta search: iterative deepening, aspiration windows, PVS with a shared
//! transposition table (lazy SMP), and the usual selectivity (null move, LMR,
//! singular extensions, futility / SEE / history pruning, ...).

use crate::bitboard::*;
use crate::cuckoo;
use crate::movegen::*;
use crate::nnue::{Network, NnueState};
use crate::position::*;
use crate::tb::{self, TbConfig};
use crate::tune;
use crate::tt::*;
use std::sync::atomic::{AtomicBool, AtomicI16, AtomicU64, Ordering::Relaxed};
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
/// Pawn-history entries per thread (Stockfish's PAWN_HISTORY_BASE_SIZE).
const PAWN_HIST_SIZE: usize = 8192;
const LOW_PLY_SIZE: usize = 5;
/// Quiet moves scoring at or below this are tried after everything else.
const GOOD_QUIET_THRESHOLD: i32 = -14000;
const KILLER_ORDERING: bool = true;
/// Esper's killer and countermove stages (Stockfish has neither).
const KILLERS: bool = false;
// Stockfish-style forms of the main pruning rules (see `search()` / `qsearch()`).
/// Prefetch the child's hash entry when a move is picked (see `prefetch_after`).
const EARLY_TT_PREFETCH: bool = true;
const SF_NMP: bool = true;
// Further Stockfish-derived techniques.
const SF_EVAL_SCALE: bool = true;
const SF_CORRECTION: bool = true;
const SF_TT_CUTOFF: bool = true;
const SF_SMALL_PROBCUT: bool = true;
const SF_SEEK_MATE: bool = true;
const SF_SHUFFLING: bool = true;
/// Range of the Stockfish-style correction history entries.
const SF_CORR_LIMIT: i32 = 1024;
/// Share the continuation, pawn and correction histories between threads.
const SHARED_HISTORY: bool = true;
// Second round of Stockfish ports (each can be switched off on its own).
/// Raise alpha to a draw when the side to move can repeat a position next move.
const SF_UPCOMING_REP: bool = true;
/// Adjust depth by the parent's reduction and how the evaluation moved.
const SF_HINDSIGHT: bool = true;
/// Accept a hash entry one ply shallower when it is an upper bound at or below beta.
const SF_TT_DEPTH: bool = true;
/// Store the static evaluation in the hash table on a miss; inherit it in check.
const SF_EVAL_STORE: bool = true;
/// Third round: line-by-line parity with Stockfish's search (see the audit).
const SF_PARITY: bool = true;
/// Esper's old check extension at node entry (Stockfish has none).
const SF_CHECK_EXT: bool = false;
/// Esper's guard against the deeper LMR re-search chaining down long lines.
const DEEPER_PLY_GUARD: bool = false;
/// Re-search a failed-high root move first, as Stockfish's root-move sort does.
const TM_RESEARCH_FIRST: bool = true;
/// Cut a stable iteration short once twice the planned time has passed.
const TM_SOFT_STOP: bool = true;
/// Searched moves remembered for history maluses (Stockfish's SEARCHEDLIST_CAPACITY).
const SEARCHED_CAP: usize = 32;
/// Extra null-move reduction (plies) granted when the evaluation is far above beta.
const NMP_EVAL_PLIES: i32 = 3;
/// Piece values in internal evaluation units, used to weigh moves for ordering.
const PIECE_VALUE: [i32; 8] = [0, 208, 781, 825, 1276, 2538, 0, 0];
/// Correction-history entries per thread (Stockfish's CORRHIST_BASE_SIZE).
const CORR_SIZE: usize = 65536;
#[allow(dead_code)]
const CORR_LIMIT: i32 = 96 * 256;
const NO_CONT: usize = usize::MAX;
/// Multi-cut (off by default: it cost about 80 Elo in testing): prune a cut node when `MC_CUTOFFS` of its first `MC_CANDIDATES`
/// moves fail high at reduced depth.
const MULTI_CUT: bool = false;
const MC_CANDIDATES: usize = 8;
const MC_CUTOFFS: usize = 3;
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
    /// When the `go` command arrived; the clock runs from here.
    pub received: Option<Instant>,
    /// The `Ponder` option is on (Stockfish then plans a quarter more time).
    pub ponder_option: bool,
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
    /// Stockfish's threads.increaseDepth: false once half the planned time is used.
    pub increase_depth: PaddedAtomicBool,
    /// Time is up but the engine is pondering: stop as soon as the ponder hits.
    pub stop_on_ponderhit: PaddedAtomicBool,
}

impl Shared {
    pub fn new() -> Shared {
        Shared {
            stop: PaddedAtomicBool(AtomicBool::new(false)),
            pondering: PaddedAtomicBool(AtomicBool::new(false)),
            nodes: (0..1024).map(|_| PaddedAtomicU64(AtomicU64::new(0))).collect(),
            tb_hits: (0..1024).map(|_| PaddedAtomicU64(AtomicU64::new(0))).collect(),
            best_move_changes: PaddedAtomicU64(AtomicU64::new(0)),
            increase_depth: PaddedAtomicBool(AtomicBool::new(true)),
            stop_on_ponderhit: PaddedAtomicBool(AtomicBool::new(false)),
        }
    }
    pub fn total_tb_hits(&self, threads: usize) -> u64 {
        self.tb_hits[..threads].iter().map(|n| n.0.load(Relaxed)).sum()
    }
    pub fn total_nodes(&self, threads: usize) -> u64 {
        self.nodes[..threads].iter().map(|n| n.0.load(Relaxed)).sum()
    }
}

#[derive(PartialEq, Clone, Copy)]
enum DrawState {
    None,
    NearRep,
    Draw,
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
    /// Depth taken off this ply's current move by LMR (read by the child).
    reduction: i32,
    /// History score of this ply's current move (Stockfish's statScore).
    stat_score: i32,
    /// This node lies on the previous iteration's principal variation.
    follow_pv: bool,
    /// Stockfish's ss->ttPv: this position is or was on a principal variation.
    tt_pv: bool,
    /// Stockfish's priorNMPFailHigh.
    prior_nmp_fail_high: i32,
    /// Which continuation table this ply's move feeds: in check * 2 + capture.
    cont_block: usize,
}

/// Stockfish's reduction table: `reductions[i] = 2872 / 128 * ln(i)`.
fn sf_reductions() -> &'static [i32; 256] {
    static T: std::sync::OnceLock<[i32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0; 256];
        for (i, v) in t.iter_mut().enumerate().skip(1) {
            *v = (2872.0 / 128.0 * (i as f64).ln()) as i32;
        }
        t
    })
}

/// Base reduction in 1024ths of a ply.
#[inline(always)]
fn sf_reduction(improving: bool, depth: i32, move_count: i32) -> i32 {
    let t = sf_reductions();
    let scale = t[depth.clamp(0, 255) as usize] * t[move_count.clamp(0, 255) as usize];
    scale + !improving as i32 * scale * 197 / 512 + 948
}

/// Divisor turning history into plies of pruning depth.
#[inline(always)]
fn lmr_divisor(depth: i32) -> i32 {
    let d = depth.min(16);
    3000 + 7 * (d - 8) * (d - 8)
}

/// `ESPER_DEBUG_TIME` set in the environment: print each move's time decisions to stderr.
fn debug_time() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ESPER_DEBUG_TIME").is_some())
}

/// Running statistics of one root move (Stockfish's RootMove fields).
#[derive(Clone, Copy)]
struct RootStat {
    mv: Move,
    /// Nodes spent below this move during the whole search.
    effort: u64,
    /// Exponential moving average of its scores; -INF before the first one.
    avg: i32,
    /// Moving average of score * |score|.
    mss: i64,
}

const MSS_NONE: i64 = -(INF as i64) * (INF as i64);

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
    /// Continuation, pawn and correction histories (shared between threads).
    hist: Arc<SharedHist>,
    /// Butterfly-style history for the first few plies of the current search.
    low_ply_hist: Box<[[i16; 4096]; LOW_PLY_SIZE]>,
    /// Quiet move that last refuted a given (piece, to-square) move.
    counter: Vec<Move>,
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
    // search state
    pub nodes: u64,
    tb_hits: u64,
    tb: TbConfig,
    tb_cardinality: u32,
    seldepth: usize,
    stopped: bool,
    pub completed_depth: i32,
    tactical_helper: bool,
    root_excluded: Vec<Move>,
    /// Move to try first at the root (the current line's previous best).
    root_move: Move,
    /// Index of the MultiPV line being searched.
    pv_idx: usize,
    root_stats: Vec<RootStat>,
    /// Iterations that did not get the go-ahead to deepen (Stockfish's searchAgainCounter).
    search_again: i32,
    /// Depth held back after fail highs on the first line (failHighRecovery).
    fail_high_recovery: i32,
    /// Per-game time scale from the first move's clock (originalTimeAdjust); -1 unset.
    orig_time_adjust: f64,
    /// The clock (wtime/btime) drives this search.
    time_managed: bool,
    /// A `go movetime` limit, checked as is.
    movetime: Option<u64>,
    /// Average score of the best root move, for the next move's time management.
    pub avg_score: i32,
    /// Twice the latest planned time; past it a stable iteration is cut short.
    soft_deadline: u64,
    /// The current iteration failed low or changed the best move.
    iter_unstable: bool,
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
    pub contempt: i32,
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

/// `hist_update` on a table shared between threads. Racing updates may be lost,
/// which costs nothing measurable; relaxed atomics compile to plain loads and stores.
#[inline(always)]
fn hist_update_at(h: &AtomicI16, bonus: i32, limit: i32) {
    let mut v = h.load(Relaxed);
    hist_update(&mut v, bonus, limit);
    h.store(v, Relaxed);
}

#[inline(always)]
fn ld(h: &AtomicI16) -> i32 {
    h.load(Relaxed) as i32
}

fn atomic_table(len: usize, fill: i16) -> Box<[AtomicI16]> {
    (0..len).map(|_| AtomicI16::new(fill)).collect()
}

/// The tables Stockfish shares between all search threads: the continuation and
/// pawn histories and the pawn, minor-piece and non-pawn correction histories.
/// With `SHARED_HISTORY` off each thread gets its own copy.
pub struct SharedHist {
    /// Continuation history, `[in check][capture][piece-to of the earlier move][piece-to]`.
    cont_hist: Box<[AtomicI16]>,
    /// Stand-in row for a missing earlier move (Stockfish's sentinel): reads -586
    /// and is never written.
    cont_sentinel: Box<[AtomicI16]>,
    /// Indexed by pawn structure, piece and destination.
    pawn_hist: Box<[AtomicI16]>,
    pawn_mask: usize,
    /// `[stm][key]`
    pawn_corr: Box<[AtomicI16]>,
    /// Correction by knight and bishop placement, `[stm][key]`.
    minor_corr: Box<[AtomicI16]>,
    /// `[color][stm][key]`
    nonpawn_corr: Box<[AtomicI16]>,
    corr_size: usize,
}

/// Stockfish's starting values.
const CONT_HIST_FILL: i16 = -586;
const PAWN_HIST_FILL: i16 = -1338;
const CORR_FILL: i16 = -5;

impl SharedHist {
    /// Tables sized for `threads` threads, as Stockfish sizes its shared ones.
    pub fn new(threads: usize) -> SharedHist {
        let scale = threads.max(1).next_power_of_two();
        let pawn_entries = PAWN_HIST_SIZE * scale;
        let corr_size = CORR_SIZE * scale;
        SharedHist {
            cont_hist: atomic_table(4 * 1024 * 1024, CONT_HIST_FILL),
            cont_sentinel: atomic_table(1024, CONT_HIST_FILL),
            pawn_hist: atomic_table(pawn_entries * 1024, PAWN_HIST_FILL),
            pawn_mask: pawn_entries - 1,
            pawn_corr: atomic_table(2 * corr_size, CORR_FILL),
            minor_corr: atomic_table(2 * corr_size, CORR_FILL),
            nonpawn_corr: atomic_table(4 * corr_size, CORR_FILL),
            corr_size,
        }
    }

    fn clear(&self) {
        let fill = |t: &[AtomicI16], v: i16| t.iter().for_each(|e| e.store(v, Relaxed));
        fill(&self.cont_hist, CONT_HIST_FILL);
        fill(&self.pawn_hist, PAWN_HIST_FILL);
        fill(&self.pawn_corr, CORR_FILL);
        fill(&self.minor_corr, CORR_FILL);
        fill(&self.nonpawn_corr, CORR_FILL);
    }

    #[inline(always)]
    fn pawn_corr(&self, stm: usize, key: u64) -> &AtomicI16 {
        &self.pawn_corr[stm * self.corr_size + (key as usize & (self.corr_size - 1))]
    }
    #[inline(always)]
    fn minor_corr(&self, stm: usize, key: u64) -> &AtomicI16 {
        &self.minor_corr[stm * self.corr_size + (key as usize & (self.corr_size - 1))]
    }
    #[inline(always)]
    fn nonpawn_corr(&self, color: usize, stm: usize, key: u64) -> &AtomicI16 {
        &self.nonpawn_corr[(color * 2 + stm) * self.corr_size + (key as usize & (self.corr_size - 1))]
    }
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

/// Stockfish's value_from_tt: mate and tablebase scores that the 50-move rule
/// could overtake are downgraded to the largest non-decisive bound.
#[inline(always)]
fn score_from_tt(s: i32, ply: usize, rule50: u8) -> i32 {
    if s == NONE {
        return NONE;
    }
    let r50 = rule50 as i32;
    if s >= MATE_BOUND {
        if (s >= MATE_IN_MAX && MATE - s > 100 - r50) || TB_WIN - s > 100 - r50 {
            return MATE_BOUND - 1;
        }
        s - ply as i32
    } else if s <= -MATE_BOUND {
        if (s <= -MATE_IN_MAX && MATE + s > 100 - r50) || TB_WIN + s > 100 - r50 {
            return -MATE_BOUND + 1;
        }
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

/// A scored move. The explicit padding makes it a plain 8 bytes, so moving one
/// while sorting is a single 64-bit copy rather than separate field copies.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct Ext {
    v: i32,
    m: Move,
    _pad: u16,
}

struct Picker {
    stage: Stage,
    tt_move: Move,
    depth: i32,
    /// Captures / promotions only (quiescence, ProbCut); ignored when in check.
    noisy_only: bool,
    buf: [std::mem::MaybeUninit<Ext>; MAX_MOVES],
    cur: usize,
    end_cur: usize,
    /// Deferred moves live at the front of the buffer: bad captures, then bad quiets.
    end_bad: usize,
    end_bad_caps: usize,
    /// The killers and countermove actually returned by their own stages, so
    /// the later stages skip exactly those. (Reading the tables again there
    /// could see a countermove updated meanwhile by a deeper node: the move
    /// tried earlier would come back a second time and the new one be lost.)
    tried: [Move; 3],
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
            buf: [std::mem::MaybeUninit::uninit(); MAX_MOVES],
            cur: 0,
            end_cur: 0,
            end_bad: 0,
            end_bad_caps: 0,
            tried: [Move::NONE; 3],
        }
    }

    /// Is the move just returned one of the captures that passed the exchange test?
    #[allow(dead_code)]
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
            // En passant lands on an empty square: no victim, as in Stockfish.
            let victim = if m.flag() == FLAG_EP { 0 } else { piece_type(pos.board[m.to()]) };
            let v = s.capture_hist[pc][m.to()][victim] as i32 + 7 * PIECE_VALUE[victim];
            self.buf[at + i].write(Ext { v, m, _pad: 0 });
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
        // Enemy sliders see through our king (Stockfish's threat squares).
        let occ_no_king = occ ^ bit(pos.king_sq(us));
        let detailed = true;

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
                minor_att |= bishop_attacks(pop_lsb(&mut b), occ_no_king);
            }
            let mut b = pos.pieces(them, ROOK);
            while b != 0 {
                rook_att |= rook_attacks(pop_lsb(&mut b), occ_no_king);
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

        let cont_slice = |back: usize| -> Option<&[AtomicI16]> { Some(s.cont_row(ply, back)) };
        let c1 = cont_slice(1);
        let c2 = cont_slice(2);
        let c3 = cont_slice(3);
        let c4 = cont_slice(4);
        let c6 = cont_slice(6);

        let pawn_base = s.pawn_hist_base(pos);
        let killers = s.ss[ply].killers;
        let counter = s.counter_move(ply);
        for (i, &m) in list.as_slice().iter().enumerate() {
            // Killers and the countermove get lighter reductions in this search, so
            // they are also tried first among the quiet moves.
            if KILLERS && KILLER_ORDERING && (m == killers[0] || m == killers[1] || m == counter) {
                let v = if m == killers[0] {
                    3 << 24
                } else if m == killers[1] {
                    2 << 24
                } else {
                    1 << 24
                };
                self.buf[at + i].write(Ext { v, m, _pad: 0 });
                continue;
            }
            let (from, to) = (m.from(), m.to());
            let pc = unsafe { *pos.board.get_unchecked(from) } as usize;
            let pt = pc & 7;
            let idx = pc * 64 + to;
            let (h1, h2, h3, h4, h6, bf, ph) = unsafe {
                (
                    c1.map_or(0, |t| ld(t.get_unchecked(idx))),
                    c2.map_or(0, |t| ld(t.get_unchecked(idx))),
                    c3.map_or(0, |t| ld(t.get_unchecked(idx))),
                    c4.map_or(0, |t| ld(t.get_unchecked(idx))),
                    c6.map_or(0, |t| ld(t.get_unchecked(idx))),
                    *s.butterfly.get_unchecked(us).get_unchecked(m.from_to()) as i32,
                    ld(s.hist.pawn_hist.get_unchecked(pawn_base + idx)),
                )
            };
            let hist = 2 * bf + h1 + h2;
            let mut v = hist + 2 * ph + h3 + h4 + h6;
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
            self.buf[at + i].write(Ext { v, m, _pad: 0 });
        }
        at + list.len
    }

    /// Check evasions: captures by victim value, then quiets by history.
    fn score_evasions(&mut self, s: &Searcher, ply: usize, pos: &Position, at: usize) -> usize {
        let mut list = MoveList::new();
        generate(pos, &mut list, true, true);
        for (i, &m) in list.as_slice().iter().enumerate() {
            let v = if m.is_capture_stage() {
                let victim = if m.flag() == FLAG_EP { 0 } else { piece_type(pos.board[m.to()]) };
                PIECE_VALUE[victim] + (1 << 28)
            } else {
                let idx = pos.board[m.from()] as usize * 64 + m.to();
                s.butterfly[pos.stm][m.from_to()] as i32 + s.cont_score(ply, 1, idx)
            };
            self.buf[at + i].write(Ext { v, m, _pad: 0 });
        }
        at + list.len
    }

    fn next(&mut self, s: &Searcher, ply: usize, skip_quiets: bool) -> Move {
        let pos = &s.pos[ply];
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
                        self.stage = if KILLERS { Stage::Killer0 } else { Stage::GenQuiet };
                    }
                }
                Stage::Killer0 => {
                    self.stage = Stage::Killer1;
                    if !skip_quiets {
                        let k = s.ss[ply].killers[0];
                        if !k.is_none() && k != self.tt_move && !k.is_noisy() && pos.is_pseudo_legal(k) {
                            self.tried[0] = k;
                            return k;
                        }
                    }
                }
                Stage::Killer1 => {
                    self.stage = Stage::Counter;
                    if !skip_quiets {
                        let k = s.ss[ply].killers[1];
                        if !k.is_none() && k != self.tt_move && k != self.tried[0] && !k.is_noisy() && pos.is_pseudo_legal(k) {
                            self.tried[1] = k;
                            return k;
                        }
                    }
                }
                Stage::Counter => {
                    self.stage = Stage::GenQuiet;
                    if !skip_quiets {
                        let c = s.counter_move(ply);
                        if !c.is_none()
                            && c != self.tt_move
                            && c != self.tried[0]
                            && c != self.tried[1]
                            && !c.is_noisy()
                            && pos.is_pseudo_legal(c)
                        {
                            self.tried[2] = c;
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
                        let [k0, k1, counter] = self.tried;
                        while self.cur < self.end_cur {
                            let e = self.get(self.cur);
                            self.cur += 1;
                            if e.m == self.tt_move || e.m == k0 || e.m == k1 || e.m == counter {
                                continue;
                            }
                            if e.v > GOOD_QUIET_THRESHOLD {
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
                    let [k0, k1, counter] = self.tried;
                    while self.cur < self.end_cur {
                        let e = self.get(self.cur);
                        self.cur += 1;
                        if e.m == self.tt_move || (self.cur > self.end_bad_caps && (e.m == k0 || e.m == k1 || e.m == counter))
                        {
                            continue;
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
    pub fn new(
        id: usize,
        threads: usize,
        net: Arc<Network>,
        tt: Arc<TT>,
        shared: Arc<Shared>,
        hist: Arc<SharedHist>,
    ) -> Box<Searcher> {
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
                    cutoff_cnt: 0,
                    reduction: 0,
                    stat_score: 0,
                    follow_pv: false,
                    tt_pv: false,
                    prior_nmp_fail_high: 0,
                    cont_block: 0,
                };
                MAX_PLY + 8
            ],
            keys: Vec::with_capacity(1024),
            pv: vec![[Move::NONE; MAX_PLY + 2]; MAX_PLY + 2],
            pv_len: [0; MAX_PLY + 2],
            butterfly: vec![[-5i16; 4096]; 2].into_boxed_slice().try_into().unwrap(),
            capture_hist: vec![[[-742i16; 8]; 64]; 16].into_boxed_slice().try_into().unwrap(),
            hist,
            low_ply_hist: vec![[102i16; 4096]; LOW_PLY_SIZE].into_boxed_slice().try_into().unwrap(),
            counter: vec![Move::NONE; 1024],
            cont_corr: vec![5; 1024 * 1024],
            tt_move_hist: 0,
            optimism: [0; 2],
            root_depth: 1,
            seek_mate: false,
            prev_time_reduction: 0.85,
            prev_avg_score: NONE,
            prev_best_score: NONE,
            nodes: 0,
            tb_hits: 0,
            tb: TbConfig::DEFAULT,
            tb_cardinality: 0,
            seldepth: 0,
            stopped: false,
            completed_depth: 0,
            tactical_helper: threads >= 4 && id >= 2 && id % 4 == 3,
            root_excluded: Vec::new(),
            root_move: Move::NONE,
            pv_idx: 0,
            root_stats: Vec::new(),
            search_again: 0,
            fail_high_recovery: 0,
            orig_time_adjust: -1.0,
            time_managed: false,
            movetime: None,
            avg_score: -INF,
            soft_deadline: u64::MAX,
            iter_unstable: false,
            root_nodes: vec![0u64; 4096].into_boxed_slice().try_into().unwrap(),
            nmp_min_ply: 0,
            limits: Limits::default(),
            multipv: 1,
            start: Instant::now(),
            soft_ms: u64::MAX,
            hard_ms: u64::MAX,
            silent: false,
            show_wdl: false,
            contempt: 0,
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
        // Shared tables are cleared once, by the main thread.
        if self.id == 0 || !SHARED_HISTORY {
            self.hist.clear();
        }
        self.counter.fill(Move::NONE);
        self.cont_corr.fill(5);
        self.tt_move_hist = 0;
        self.prev_time_reduction = 0.85;
        self.prev_avg_score = NONE;
        self.prev_best_score = NONE;
        self.orig_time_adjust = -1.0;
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
        // Stockfish decays the move history at the start of every search.
        self.butterfly.iter_mut().flatten().for_each(|h| *h = (*h as i32 * 729 / 1024) as i16);
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
        self.start = limits.received.unwrap_or_else(Instant::now);
        self.soft_ms = u64::MAX;
        self.hard_ms = u64::MAX;
        self.time_managed = false;
        self.soft_deadline = u64::MAX;
        self.movetime = limits.movetime.map(|mt| mt.saturating_sub(overhead).max(1));
        let us = root.stm;
        // Snappy, flag-proof time allocation.
        if let Some(time) = limits.time[us] {
            self.time_managed = true;
            let reserve = (2 * overhead).max(50);
            if time <= reserve {
                self.hard_ms = 1;
                self.soft_ms = 1;
                return;
            }
            let usable = (time - reserve) as f64;
            let inc = limits.inc[us] as f64;
            let ply = (2 * (root.fullmove as i64 - 1) + (us == BLACK) as i64).max(0) as f64;
            let (mut soft, mut hard) = match limits.movestogo {
                Some(mtg) => {
                    let m = mtg.max(1) as f64;
                    let s = (usable / (m + 1.5) + inc * 0.70).min(usable * 0.50);
                    let h = (s * 2.5).min(usable * 0.70);
                    (s, h)
                }
                None => {
                    // Assume 40 moves remaining early, tapering to 25 later in the game.
                    let mtg = (40.0 - (ply * 0.25).min(15.0)).max(25.0);
                    let s = (usable / mtg + inc * 0.70).min(usable * 0.10);
                    let h = (s * 2.5).min(usable * 0.25);
                    (s, h)
                }
            };
            // Behind on the clock, play faster.
            if limits.movestogo != Some(1) {
                let theirs = limits.time[us ^ 1].unwrap_or(0) as f64;
                if theirs > 0.0 && (time as f64) < theirs {
                    let ratio = ((time as f64) / theirs).clamp(0.5, 1.0);
                    soft *= ratio;
                    hard *= ratio;
                }
            }
            // Extra safety in sudden-death time scrambles.
            if inc == 0.0 && time < 2000 {
                soft = soft.min(usable * 0.04);
                hard = hard.min(usable * 0.12);
            }
            if limits.ponder_option {
                soft *= 1.15;
            }
            self.hard_ms = (hard.min(usable).max(1.0)) as u64;
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
        let h = &*self.hist;
        let pcv = ld(h.pawn_corr(stm, pos.pawn_key));
        let micv = ld(h.minor_corr(stm, pos.minor_key));
        let wnp = ld(h.nonpawn_corr(WHITE, stm, pos.nonpawn_key[WHITE]));
        let bnp = ld(h.nonpawn_corr(BLACK, stm, pos.nonpawn_key[BLACK]));
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
        let pos = self.pos_at(ply);
        let stm = pos.stm;
        let h = &*self.hist;
        hist_update_at(h.pawn_corr(stm, pos.pawn_key), bonus, SF_CORR_LIMIT);
        hist_update_at(h.minor_corr(stm, pos.minor_key), bonus * 150 / 128, SF_CORR_LIMIT);
        hist_update_at(h.nonpawn_corr(WHITE, stm, pos.nonpawn_key[WHITE]), bonus * 186 / 128, SF_CORR_LIMIT);
        hist_update_at(h.nonpawn_corr(BLACK, stm, pos.nonpawn_key[BLACK]), bonus * 186 / 128, SF_CORR_LIMIT);
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
        let h = &*self.hist;
        let p = ld(h.pawn_corr(stm, pos.pawn_key));
        let w = ld(h.nonpawn_corr(WHITE, stm, pos.nonpawn_key[WHITE]));
        let b = ld(h.nonpawn_corr(BLACK, stm, pos.nonpawn_key[BLACK]));
        (2 * p + w + b) / 512
    }

    #[allow(dead_code)]
    fn update_correction(&mut self, pos: &Position, diff: i32, depth: i32) {
        let stm = pos.stm;
        let w = (depth + 1).min(16);
        let target = diff.clamp(-CORR_LIMIT / 256, CORR_LIMIT / 256) * 256;
        let upd = |e: &AtomicI16| {
            let v = ((ld(e) * (256 - w) + target * w) / 256).clamp(-CORR_LIMIT, CORR_LIMIT) as i16;
            e.store(v, Relaxed);
        };
        let h = &*self.hist;
        upd(h.pawn_corr(stm, pos.pawn_key));
        upd(h.nonpawn_corr(WHITE, stm, pos.nonpawn_key[WHITE]));
        upd(h.nonpawn_corr(BLACK, stm, pos.nonpawn_key[BLACK]));
    }

    /// The continuation-history row of the move made `back` plies before `ply`,
    /// or the never-updated sentinel row when there is no such move.
    #[inline(always)]
    fn cont_row(&self, ply: usize, back: usize) -> &[AtomicI16] {
        if ply >= back {
            let e = &self.ss[ply - back];
            if e.cont != NO_CONT {
                let base = (e.cont_block * 1024 + e.cont) * 1024;
                return &self.hist.cont_hist[base..base + 1024];
            }
        }
        &self.hist.cont_sentinel
    }

    #[inline(always)]
    fn cont_score(&self, ply: usize, back: usize, idx: usize) -> i32 {
        ld(&self.cont_row(ply, back)[idx])
    }

    /// Start of the pawn-history row for `pos`'s pawn structure.
    #[inline(always)]
    fn pawn_hist_base(&self, pos: &Position) -> usize {
        (pos.pawn_key as usize & self.hist.pawn_mask) * 1024
    }

    /// History score of a quiet move as used by pruning and reductions.
    #[allow(dead_code)]
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
                    let base = (self.ss[ply - back].cont_block * 1024 + c) * 1024;
                    let e = &self.hist.cont_hist[base + idx];
                    if ld(e) > 0 {
                        positive += 1;
                    }
                    let b = bonus * weight * CONSISTENCY[positive] / 65536 + 73 * (back < 2) as i32;
                    hist_update_at(e, b, CONT_HIST_LIMIT);
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
        let pawn_idx = self.pawn_hist_base(&pos) + idx;
        let scale = if bonus > -4 { 1104 } else { 459 };
        hist_update_at(&self.hist.pawn_hist[pawn_idx], bonus * scale / 1024, PAWN_HIST_LIMIT);
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
        // En passant lands on an empty square: slot 0, as Stockfish's piece_on(to).
        let victim = if m.flag() == FLAG_EP { 0 } else { piece_type(pos.board[m.to()]) };
        hist_update(&mut self.capture_hist[pc][m.to()][victim], bonus, CAPTURE_HIST_LIMIT);
    }

    /// Start loading the hash entry of the position after `m` as soon as the move
    /// is picked, so the load overlaps the pruning decisions and the move itself
    /// rather than stalling the child's probe. Castling, en passant and
    /// promotions are skipped: their key needs a full move.
    #[inline(always)]
    fn prefetch_after(&self, pos: &Position, m: Move) {
        let f = m.flag();
        if EARLY_TT_PREFETCH && !(f == FLAG_EP || f == FLAG_KCASTLE || f == FLAG_QCASTLE || m.is_promo()) {
            self.tt.prefetch(pos.key_after(m));
        }
    }

    #[inline(always)]
    fn make(&mut self, ply: usize, m: Move) {
        let pc = self.pos[ply].board[m.from()] as usize;
        self.ss[ply].mv = m;
        self.ss[ply].cont = pc * 64 + m.to();
        self.ss[ply].cont_block = 2 * self.pos[ply].in_check() as usize + m.is_capture_stage() as usize;
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
    fn tt_cutoff_confirmed(&self, pos: &Position, tt_move: Move, tt_score: i32, beta: i32, depth: i32) -> bool {
        if depth < 7 || tt_move.is_none() || tt_score.abs() >= MATE_BOUND || !pos.is_legal(tt_move) {
            return true;
        }
        match self.tt.probe(pos.key_after(tt_move)) {
            Some(t) if t.bound != 0 && t.score != NONE => (tt_score >= beta) == (-t.score >= beta),
            _ => true,
        }
    }

    /// Futility and exchange pruning for a capture in the move loop.
    #[allow(dead_code, clippy::too_many_arguments)]
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
        lmr_depth: i32,
    ) -> bool {
        let victim = if m.flag() == FLAG_EP { PAWN } else { piece_type(pos.board[m.to()]) };
        if !gives_check && !in_check {
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

    /// The position at `ply`, borrowed independently of `self` so a node can keep
    /// it while it searches. Sound because `pos` is allocated once (MAX_PLY + 8
    /// entries, never resized) and a node never writes its own entry: moves are
    /// made into `ply + 1` and deeper, and re-searches at the same ply (singular,
    /// razoring, null-move verification) only read it.
    #[inline(always)]
    fn pos_at(&self, ply: usize) -> &'static Position {
        debug_assert!(ply < self.pos.len());
        // SAFETY: see above; the buffer outlives every search call.
        unsafe { &*self.pos.as_ptr().add(ply) }
    }

    #[inline(always)]
    fn unmake(&mut self) {
        self.keys.pop();
    }

    fn is_draw(&self, ply: usize) -> DrawState {
        let pos = &self.pos[ply];
        if pos.rule50 >= 100 && (pos.checkers == 0 || has_legal_move(pos)) {
            return DrawState::Draw;
        }
        if pos.insufficient_material() {
            return DrawState::Draw;
        }
        let end = (pos.rule50 as usize).min(pos.plies_from_null as usize);
        let cur = self.keys.len() - 1;
        let mut reps = 0;
        let mut i = 4;
        while i <= end && i <= cur {
            if self.keys[cur - i] == pos.key {
                // A single repetition inside the search tree is enough.
                if i < ply {
                    return DrawState::Draw;
                }
                reps += 1;
                if reps == 2 {
                    return DrawState::Draw;
                }
            }
            i += 2;
        }
        if reps == 1 { DrawState::NearRep } else { DrawState::None }
    }

    #[inline(always)]
    fn draw_score(&self, ply: usize) -> i32 {
        let jitter = -1 + (self.nodes & 2) as i32;
        if self.contempt == 0 {
            jitter
        } else if ply & 1 == 0 {
            -self.contempt + jitter
        } else {
            self.contempt + jitter
        }
    }

    fn check_time(&mut self) {
        self.shared.nodes[self.id].0.store(self.nodes, Relaxed);
        if self.tb_cardinality > 0 {
            self.shared.tb_hits[self.id].0.store(self.tb_hits, Relaxed);
        }
        if self.shared.stop.0.load(Relaxed) {
            self.stopped = true;
            return;
        }
        if self.id != 0 || self.shared.pondering.0.load(Relaxed) {
            return;
        }
        let elapsed = self.start.elapsed().as_millis() as u64;
        let mut stop = (self.time_managed && elapsed >= self.hard_ms)
            || self.movetime.is_some_and(|mt| elapsed >= mt)
            || (self.completed_depth > 0
                && (self.shared.stop_on_ponderhit.0.load(Relaxed)
                    || (TM_SOFT_STOP && self.time_managed && elapsed >= self.soft_deadline && !self.iter_unstable)));
        if let Some(n) = self.limits.nodes {
            stop |= self.shared.total_nodes(self.threads) >= n;
        }
        if stop {
            self.shared.stop.0.store(true, Relaxed);
            self.stopped = true;
        }
    }

    // -- quiescence search ----------------------------------------------------

    /// `pv_node` is the node type the caller chose, as Stockfish's template
    /// argument (not inferred from the window, which a PV node can narrow).
    fn qsearch(&mut self, ply: usize, mut alpha: i32, beta: i32, pv_node: bool) -> i32 {
        self.pv_len[ply] = 0;
        if self.nodes & 255 == 0 {
            self.check_time();
        }
        if self.stopped {
            return 0;
        }
        if pv_node && self.seldepth < ply + 1 {
            self.seldepth = ply + 1;
        }
        let pos = self.pos_at(ply);
        let in_check = pos.checkers != 0;
        if self.is_draw(ply) == DrawState::Draw {
            return 0;
        }
        if ply >= MAX_PLY {
            return if in_check { 0 } else { self.evaluate(ply) };
        }
        if SF_UPCOMING_REP && alpha < 0 && cuckoo::upcoming_repetition(&pos, &self.keys, ply) {
            alpha = self.draw_score(ply);
            if alpha >= beta {
                return alpha;
            }
        }

        let tte = self.tt.probe(pos.key);
        let mut tt_move = Move::NONE;
        let mut tt_score = NONE;
        let tt_pv = tte.is_some_and(|t| t.pv);
        if let Some(t) = tte {
            tt_score = if t.bound == 0 { NONE } else { score_from_tt(t.score, ply, pos.rule50) };
            if !pv_node
                && t.depth >= 0
                && tt_score != NONE
                && t.bound & (if tt_score >= beta { BOUND_LOWER } else { BOUND_UPPER }) != 0
            {
                return tt_score;
            }
            tt_move = t.mv;
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
            let static_eval = (raw_eval + self.correction_at(ply)).clamp(-MATE_BOUND + 1, MATE_BOUND - 1);
            self.ss[ply].static_eval = static_eval;
            best = static_eval;
            if tt_score != NONE
                && tt_score.abs() < MATE_BOUND
                && tte.is_some_and(|t| t.bound & (if tt_score > best { BOUND_LOWER } else { BOUND_UPPER }) != 0)
            {
                best = tt_score;
            }
            if best >= beta {
                // Lean the stand-pat score toward beta.
                best = (441 * best + 583 * beta) / 1024;
                if tte.is_none() {
                    self.tt.store(pos.key, Move::NONE, NONE, raw_eval, DEPTH_UNSEARCHED, BOUND_LOWER, false);
                }
                return best;
            }
            if best > alpha {
                alpha = best;
            }
            futility_base = static_eval + tune::T_QsFutBase.get();
        }

        // As Stockfish, a quiet hash move is kept: it counts toward the move-count
        // pruning below before being skipped as a non-capture.
        if !pos.is_pseudo_legal(tt_move) {
            tt_move = Move::NONE;
        }
        let prev_to = if ply > 0 && !self.ss[ply - 1].mv.is_none() { Some(self.ss[ply - 1].mv.to()) } else { None };
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
            if best > -MATE_BOUND {
                let gives_check = pos.gives_check(m);
                if !gives_check && Some(m.to()) != prev_to && futility_base > -MATE_BOUND && !m.is_promo() {
                    if move_count > 2 {
                        continue;
                    }
                    let victim = if m.flag() == FLAG_EP { 0 } else { piece_type(pos.board[m.to()]) };
                    let fut = futility_base + PIECE_VALUE[victim];
                    if fut <= alpha {
                        best = best.max(fut);
                        continue;
                    }
                    // Stockfish's exchange threshold, in its units of 208 per pawn.
                    if !pos.see_ge(m, (alpha - futility_base) * 100 / 208) {
                        best = best.max(alpha.min(futility_base));
                        continue;
                    }
                }
                if !m.is_capture_stage() {
                    continue;
                }
                if !pos.see_ge(m, -74 * 100 / 208) {
                    continue;
                }
            }
            self.make(ply, m);
            let score = -self.qsearch(ply + 1, -beta, -alpha, pv_node);
            self.unmake();
            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    best_move = m;
                    if score >= beta {
                        break;
                    }
                    alpha = score;
                }
            }
        }
        if move_count == 0 {
            if in_check {
                return -MATE + ply as i32;
            }
            // Stalemate after a capture of a piece, with nothing left that can move.
            let us = pos.stm;
            let pushes = if us == WHITE { pos.pieces(us, PAWN) << 8 } else { pos.pieces(us, PAWN) >> 8 } & !pos.occ();
            let captured_piece = ply >= 1
                && self.ss[ply - 1].mv.is_capture()
                && piece_type(self.pos[ply - 1].board[self.ss[ply - 1].mv.to()]) >= KNIGHT;
            if pushes == 0 && !pos.has_non_pawn_material(us) && captured_piece && !has_legal_move(&pos) {
                best = 0;
            }
        }
        if best.abs() < MATE_BOUND && best > beta {
            best = (462 * best + 562 * beta) / 1024;
        }
        let bound = if best >= beta { BOUND_LOWER } else { BOUND_UPPER };
        self.tt.store(pos.key, best_move, score_to_tt(best, ply), raw_eval, 0, bound, tt_pv);
        best
    }

    // -- main search ------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn search(&mut self, ply: usize, mut alpha: i32, mut beta: i32, mut depth: i32, cut_node: bool, pv_node: bool) -> i32 {
        if depth <= 0 {
            return self.qsearch(ply, alpha, beta, pv_node);
        }
        let root = ply == 0;
        self.pv_len[ply] = 0;
        if self.nodes & 255 == 0 {
            self.check_time();
        }
        if self.stopped {
            return 0;
        }
        if pv_node && self.seldepth < ply + 1 {
            self.seldepth = ply + 1;
        }
        let pos = self.pos_at(ply);
        let in_check = pos.checkers != 0;
        let mut near_rep = false;
        if !root {
            let draw = self.is_draw(ply);
            if draw == DrawState::Draw {
                return self.draw_score(ply);
            }
            near_rep = draw == DrawState::NearRep;
            if ply >= MAX_PLY {
                return if in_check { self.draw_score(ply) } else { self.evaluate(ply) };
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
            // The side to move can repeat a position: a draw is the least it gets.
            if SF_UPCOMING_REP && alpha < 0 && cuckoo::upcoming_repetition(&pos, &self.keys, ply) {
                alpha = self.draw_score(ply);
                if alpha >= beta {
                    return alpha;
                }
            }
        }
        let all_node = !pv_node && !cut_node;
        let prior_reduction = if ply > 0 { std::mem::take(&mut self.ss[ply - 1].reduction) } else { 0 };
        self.ss[ply].stat_score = 0;
        self.ss[ply + 1].prior_nmp_fail_high = 0;
        self.ss[ply].follow_pv = root
            || (self.ss[ply - 1].follow_pv
                && ply - 1 < self.best_pv.len()
                && self.ss[ply - 1].mv == self.best_pv[ply - 1]);

        let excluded = self.ss[ply].excluded;
        // Esper's old check extension; Stockfish has none.
        if SF_CHECK_EXT
            && in_check
            && !root
            && excluded.is_none()
            && (!cut_node || depth <= 10)
            && ply < 2 * self.completed_depth.max(1) as usize + 4
        {
            depth += 1;
        }
        depth = depth.min(MAX_PLY as i32 - 1);
        self.ss[ply + 2].cutoff_cnt = 0;
        // NearRep (Esper's own) blocks hash cutoffs one repetition early; Stockfish
        // relies on its draw detection alone.
        if SF_PARITY {
            near_rep = false;
        }

        // The hash table is probed in singular (excluded-move) searches too; only the
        // cutoff is skipped there, and the PV flag carries over from the outer visit.
        let tte = self.tt.probe(pos.key);
        self.ss[ply].tt_hit = tte.is_some();
        let mut tt_move = Move::NONE;
        let mut tt_score = NONE;
        let mut tt_depth = -1;
        let mut tt_bound = 0;
        let mut tt_pv = if excluded.is_none() { pv_node || tte.is_some_and(|t| t.pv) } else { self.ss[ply].tt_pv };
        if let Some(t) = tte {
            tt_score = if t.bound == 0 { NONE } else { score_from_tt(t.score, ply, pos.rule50) };
            tt_depth = t.depth;
            tt_bound = t.bound;
            if pos.is_pseudo_legal(t.mv) {
                tt_move = t.mv;
            }
        }
        // At the root the move to try first is the previous iteration's best.
        if root && !self.root_move.is_none() && pos.is_pseudo_legal(self.root_move) {
            tt_move = self.root_move;
        }
        self.ss[ply].tt_pv = tt_pv;
        let tt_capture = tt_move.is_capture_stage();

        // Correction value, computed once per node as Stockfish does.
        let corr_raw = if SF_CORRECTION { self.correction_raw(ply) } else { 0 };

        // Static evaluation
        let raw_eval;
        let static_eval;
        let mut eval;
        if in_check {
            raw_eval = NONE;
            // Stockfish carries the evaluation from two plies back through a check.
            static_eval = if SF_EVAL_STORE && ply >= 2 { self.ss[ply - 2].static_eval } else { NONE };
            eval = static_eval;
        } else if !excluded.is_none() {
            static_eval = self.ss[ply].static_eval;
            raw_eval = static_eval;
            eval = static_eval;
        } else {
            raw_eval = match tte {
                Some(t) if t.eval != NONE => t.eval,
                _ => self.evaluate(ply),
            };
            // Save the evaluation for the next visit even if nothing else gets stored.
            if SF_EVAL_STORE && tte.is_none() {
                self.tt.store(pos.key, Move::NONE, NONE, raw_eval, DEPTH_UNSEARCHED, 0, tt_pv);
            }
            let corr = if SF_CORRECTION { corr_raw / 131072 } else { self.correction(&self.pos[ply]) };
            static_eval = (raw_eval + corr).clamp(-MATE_BOUND + 1, MATE_BOUND - 1);
            eval = static_eval;
            if tt_score != NONE && tt_bound & (if tt_score > eval { BOUND_LOWER } else { BOUND_UPPER }) != 0
            {
                eval = tt_score;
            }
        }
        self.ss[ply].static_eval = static_eval;

        let mut improving = if SF_PARITY {
            static_eval != NONE && ply >= 2 && self.ss[ply - 2].static_eval != NONE && static_eval > self.ss[ply - 2].static_eval
        } else {
            !in_check && {
                if ply >= 2 && self.ss[ply - 2].static_eval != NONE {
                    static_eval > self.ss[ply - 2].static_eval
                } else if ply >= 4 && self.ss[ply - 4].static_eval != NONE {
                    static_eval > self.ss[ply - 4].static_eval
                } else {
                    true
                }
            }
        };
        // Hindsight: a big reduction of the previous move that the evaluation
        // does not support is partly given back; a quiet position loses a ply.
        let prev_eval = if ply >= 1 { self.ss[ply - 1].static_eval } else { NONE };
        let opponent_worsening = static_eval != NONE && (prev_eval == NONE || static_eval > -prev_eval);
        if SF_HINDSIGHT && excluded.is_none() {
            if prior_reduction >= 3 && !opponent_worsening {
                depth += 1;
            }
            if prior_reduction >= 2
                && depth >= 2
                && static_eval != NONE
                && prev_eval != NONE
                && static_eval + prev_eval > 166
            {
                depth -= 1;
            }
        }

        // Hash cutoff (Stockfish's step 6): with enough depth, update the
        // histories as for a cutoff, then return unless the 50-move rule is near or
        // the position after the hash move disagrees.
        if !pv_node
            && excluded.is_none()
            && !near_rep
            && tt_score != NONE
            && if SF_TT_DEPTH { tt_depth > depth - (tt_score <= beta) as i32 } else { tt_depth >= depth }
        {
            if tt_bound & (if tt_score >= beta { BOUND_LOWER } else { BOUND_UPPER }) != 0
                && (!SF_TT_CUTOFF || cut_node == (tt_score >= beta) || depth > 4)
            {
                if tt_score >= beta && !tt_move.is_none() {
                    if !tt_capture {
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
                if pos.rule50 < 96 && self.tt_cutoff_confirmed(&pos, tt_move, tt_score, beta, depth) {
                    return tt_score;
                }
            } else if depth > 5 {
                // Deep enough but bounded the wrong way: let the entry be replaced sooner.
                self.tt.penalize(pos.key, 1);
            }
        }

        // Tablebase probe: an exact win/draw/loss verdict replaces searching.
        let mut tb_floor = -INF;
        let mut tb_ceiling = INF;
        let pieces = popcount(pos.occ());
        if !root
            && self.tb_cardinality > 0
            && excluded.is_none()
            && pieces <= self.tb_cardinality
            && (pieces < self.tb_cardinality || depth >= self.tb.probe_depth)
            && pos.rule50 == 0
            && pos.castling == 0
        {
            if let Some(wdl) = tb::probe_wdl(&pos) {
                self.tb_hits += 1;
                let draw_score = self.tb.rule50 as i32;
                let (v, bound) = if wdl < -draw_score {
                    (-TB_WIN + ply as i32, BOUND_UPPER)
                } else if wdl > draw_score {
                    (TB_WIN - ply as i32, BOUND_LOWER)
                } else {
                    // Cursed wins and blessed losses score just off zero.
                    (2 * wdl * draw_score, BOUND_EXACT)
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
                if pv_node {
                    if bound == BOUND_LOWER {
                        tb_floor = v;
                        alpha = alpha.max(v);
                    } else {
                        tb_ceiling = v;
                    }
                }
            }
        }

        self.ss[ply + 1].killers = [Move::NONE; 2];
        self.ss[ply].double_ext = if ply > 0 { self.ss[ply - 1].double_ext } else { 0 };

        // Stockfish skips everything up to the move loop when in check.
        if !in_check {
            // The change in static evaluation caused by the opponent's last quiet move
            // says something about that move's quality.
            if ply >= 1 {
                let prev = self.ss[ply - 1];
                if prev.cont != NO_CONT && !prev.mv.is_capture() && prev.static_eval != NONE && !self.pos[ply - 1].in_check() {
                    let diff = (-(prev.static_eval + static_eval)).clamp(-189, 194) + 60;
                    hist_update(&mut self.butterfly[pos.stm ^ 1][prev.mv.from_to()], diff * 11, MAIN_HIST_LIMIT);
                    if tte.is_none() && prev.cont / 64 & 7 != PAWN && !prev.mv.is_promo() {
                        let i = self.pawn_hist_base(&pos) + prev.cont;
                        hist_update_at(&self.hist.pawn_hist[i], diff * 13, PAWN_HIST_LIMIT);
                    }
                }
            }

            // Razoring
            if all_node && eval < alpha - 342 * depth && !self.seek_mate {
                return self.qsearch(ply, alpha, beta, false);
            }

            // Reverse futility pruning (static null move)
            let their_runners = pos.pieces(pos.stm ^ 1, PAWN)
                & if pos.stm == WHITE { RANK_2 | RANK_3 } else { RANK_6 | RANK_7 };
            if (SF_PARITY || their_runners == 0)
                && !tt_pv
                && depth < if self.seek_mate { 6 } else { 19 }
                && eval >= beta
                && (tt_move.is_none() || tt_capture)
                && eval < MATE_BOUND
                && beta > -MATE_BOUND
            {
                let mult = (45 + 4 * depth).min(85) - 20 * tte.is_none() as i32;
                let margin = mult * depth
                    - (tune::T_RfpImproving.get() * improving as i32 + 335 * opponent_worsening as i32) * mult / 1024
                    + corr_raw.abs() / 198435;
                if eval - margin >= beta {
                    return (661 * beta + 363 * eval) / 1024;
                }
            }

            // Null move pruning
            let nmp_ok = if SF_NMP {
                cut_node
                    && static_eval + 50 * self.ss[ply].prior_nmp_fail_high >= beta - 13 * depth - 47 * improving as i32 + 365
                    && beta >= -2000
            } else {
                !pv_node && depth >= 3 && eval >= beta && beta > -MATE_BOUND
            };
            if nmp_ok
                && excluded.is_none()
                && ply > 0
                && !self.ss[ply - 1].mv.is_none()
                && pos.has_non_pawn_material(pos.stm)
                && ply >= self.nmp_min_ply
            {
                // Total depth taken off the null-move search.
                let r = if SF_NMP {
                    7 + depth / 3 + ((static_eval - beta) / 256).max(0) - (self.id & 1) as i32
                } else {
                    1 + 3 + depth / 4 + !improving as i32 + ((eval - beta) / 200).clamp(0, NMP_EVAL_PLIES)
                };
                // Searching for a mate, null-move cutoffs are less trustworthy.
                let r = if self.seek_mate { r - r / 2 } else { r };
                self.make_null(ply);
                let v = -self.search(ply + 1, -beta, -beta + 1, depth - r, if SF_NMP { false } else { !cut_node }, false);
                self.unmake();
                if self.stopped {
                    return 0;
                }
                // Unproven mates and tablebase wins are not returned.
                if v >= beta && v < MATE_BOUND {
                    if self.nmp_min_ply != 0 || depth < 16 {
                        self.ss[ply].prior_nmp_fail_high += 1;
                        return v;
                    }
                    // Verification search at high depth guards against zugzwang.
                    self.nmp_min_ply = ply + 3 * (depth - r).max(0) as usize / 4;
                    let verify = self.search(ply, beta - 1, beta, depth - r, false, false);
                    self.nmp_min_ply = 0;
                    if self.stopped {
                        return 0;
                    }
                    if verify >= beta {
                        self.ss[ply].prior_nmp_fail_high += 1;
                        return v;
                    }
                }
            }

            improving |= static_eval >= beta;

            // Internal iterative reduction
            if !self.ss[ply].follow_pv && !all_node && depth >= 6 && tt_move.is_none() {
                depth -= 1;
            }

            // ProbCut: a capture that beats beta by a margin at reduced depth
            let pc_beta = beta + 241 - 64 * improving as i32;
            if depth >= 3 && beta.abs() < MATE_BOUND && !(tt_score != NONE && tt_score < pc_beta) {
                // Stockfish's threshold is in its units of 208 per pawn.
                let see_margin = (pc_beta - static_eval) * 100 / 208;
                let tm = if tt_capture { tt_move } else { Move::NONE };
                let pc_depth = depth - if improving { 5 } else { 3 };
                let mut picker = Picker::new(tm, true, 0);
                loop {
                    let m = picker.next(self, ply, true);
                    if m.is_none() {
                        break;
                    }
                    if m == excluded || !m.is_capture_stage() || !pos.is_legal(m) {
                        continue;
                    }
                    // The hash move skips the exchange test, as in Stockfish's picker.
                    if m != tm && !pos.see_ge(m, see_margin) {
                        continue;
                    }
                    self.make(ply, m);
                    let mut v = -self.qsearch(ply + 1, -pc_beta, -pc_beta + 1, false);
                    if v >= pc_beta && pc_depth > 0 {
                        v = -self.search(ply + 1, -pc_beta, -pc_beta + 1, pc_depth, !cut_node, false);
                    }
                    self.unmake();
                    if self.stopped {
                        return 0;
                    }
                    if v >= pc_beta {
                        self.tt.store(pos.key, m, score_to_tt(v, ply), raw_eval, pc_depth + 1, BOUND_LOWER, tt_pv);
                        if v.abs() < MATE_BOUND {
                            return v - (pc_beta - beta);
                        }
                    }
                }
            }

            // Multi-cut: if several of the first moves each refute the node at
            // reduced depth, trust that one of them would at full depth too.
            if MULTI_CUT && !pv_node && excluded.is_none() && cut_node && depth >= 6 && beta.abs() < MATE_BOUND {
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
                    let v = -self.search(ply + 1, -beta, -beta + 1, depth - 4, false, false);
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

        // Hash-entry ProbCut: the stored lower bound already beats beta by a wide
        // margin at nearly this depth.
        if SF_SMALL_PROBCUT && !self.seek_mate {
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

        let mut picker = Picker::new(tt_move, false, depth);
        let has_pieces = pos.has_non_pawn_material(pos.stm);
        // Built when a second move is searched: most nodes cut off on the first.
        let mut ci: Option<CheckInfo> = None;
        let mut best = if pv_node { tb_floor } else { -INF };
        let mut best_move = Move::NONE;
        let mut move_count = 0;
        let mut skip_quiets = false;
        let mut quiets = [Move::NONE; SEARCHED_CAP];
        let mut n_quiets = 0;
        let mut noisies = [Move::NONE; SEARCHED_CAP];
        let mut n_noisies = 0;

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
            self.prefetch_after(&pos, m);
            move_count += 1;
            let capture = m.is_capture_stage();
            let gives_check = if move_count == 1 {
                pos.gives_check(m)
            } else {
                pos.gives_check_ci(m, ci.get_or_insert_with(|| pos.check_info()))
            };
            let pc = pos.board[m.from()] as usize;
            let idx = pc * 64 + m.to();
            // En passant targets an empty square: no victim, as in Stockfish.
            let victim = if m.flag() == FLAG_EP { 0 } else { piece_type(pos.board[m.to()]) };
            let capt_hist = self.capture_hist[pc][m.to()][victim] as i32;
            let new_depth_base = depth - 1;

            // Stockfish's reduction, in 1024ths of a ply.
            let mut r = sf_reduction(improving, depth, move_count);
            if pv_node {
                r -= 512;
            }
            if tt_pv {
                r += 929;
            }

            // Shallow-depth pruning
            if !root && best > -MATE_BOUND && has_pieces {
                // Move-count pruning: later quiet moves are skipped, this one still counts.
                let mut lmp_limit = (3 + depth * depth) * tune::T_LmpScale.get() / (1024 * (2 - improving as i32));
                if self.tactical_helper {
                    lmp_limit = lmp_limit * 3 / 2;
                }
                if move_count >= lmp_limit.max(1) {
                    skip_quiets = true;
                }
                let mut lmr_depth = new_depth_base - r / 1024;
                if capture || gives_check {
                    // Futility for captures
                    if !gives_check && lmr_depth < 8 {
                        let fut = static_eval
                            + tune::T_CapFutBase.get()
                            + tune::T_CapFutMult.get() * lmr_depth
                            + PIECE_VALUE[victim]
                            + 134 * capt_hist / 1024;
                        if static_eval != NONE && fut <= alpha {
                            continue;
                        }
                    }
                    // Exchange test for captures and checks (Stockfish's units of 208 per pawn).
                    let margin = (tune::T_CapSeeMult.get() * depth + capt_hist * 34 / 1024) * 100 / 208;
                    if !self.stalemate_sac(&pos, m, alpha) && !pos.see_ge(m, -margin) {
                        continue;
                    }
                } else if !self.ss[ply].follow_pv || !pv_node {
                    let pawn_idx = self.pawn_hist_base(&pos) + idx;
                    let mut history =
                        self.cont_score(ply, 1, idx) + self.cont_score(ply, 2, idx) + ld(&self.hist.pawn_hist[pawn_idx]);
                    // Continuation-history pruning
                    if history < -4136 * depth {
                        continue;
                    }
                    history += 69 * self.butterfly[pos.stm][m.from_to()] as i32 / 32;
                    lmr_depth += history / lmr_divisor(depth);
                    // Futility pruning: parent node
                    let fut = static_eval
                        + tune::T_FutMult.get() * lmr_depth
                        + tune::T_FutAboveAlpha.get() * (static_eval > alpha) as i32
                        + tune::T_FutBase.get();
                    if !in_check && lmr_depth < 12 && fut <= alpha {
                        if best <= fut && best.abs() < MATE_BOUND && fut < MATE_BOUND {
                            best = fut;
                        }
                        continue;
                    }
                    lmr_depth = lmr_depth.max(0);
                    if !pos.see_ge(m, -23 * lmr_depth * lmr_depth * 100 / 208) {
                        continue;
                    }
                }
            }

            // Singular extension: is the hash move much better than everything else?
            let mut ext = 0;
            if !root
                && m == tt_move
                && excluded.is_none()
                && depth >= 6 + tt_pv as i32
                && tt_score != NONE
                && tt_score.abs() < MATE_BOUND
                && tt_bound & BOUND_LOWER != 0
                && tt_depth >= depth - 3
                && !(SF_SHUFFLING && self.is_shuffling(ply, m))
                && !self.seek_mate
            {
                let margin = (tune::T_SingularMargin.get() + 66 * (tt_pv && !pv_node) as i32) * depth / 63;
                let s_beta = tt_score - margin;
                self.ss[ply].excluded = m;
                let v = self.search(ply, s_beta - 1, s_beta, (depth - 1) / 2, cut_node, false);
                self.ss[ply].excluded = Move::NONE;
                if self.stopped {
                    return 0;
                }
                if v < s_beta {
                    let corr_adj = corr_raw.abs() / 198368;
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
                    ext = 1 + (v < s_beta - double) as i32 + (v < s_beta - triple) as i32;
                    depth += 1;
                } else if v >= beta && v.abs() < MATE_BOUND {
                    // Multi-cut: other moves beat beta too.
                    hist_update(&mut self.tt_move_hist, -421 - 110 * depth, 8192);
                    if !in_check && static_eval != NONE && v > static_eval {
                        let bonus = ((v - static_eval) * 664 / 1024).clamp(-SF_CORR_LIMIT / 4, SF_CORR_LIMIT / 4);
                        self.update_correction_sf(ply, bonus);
                    }
                    return v;
                } else if tt_score >= beta || cut_node {
                    ext = -3;
                }
            }

            // Stockfish's statScore: how well this move has done before.
            let stat_score = if capture {
                // Stockfish reads the captured piece here, a pawn for en passant.
                if m.flag() == FLAG_EP {
                    873 * PIECE_VALUE[PAWN] / 128 + self.capture_hist[pc][m.to()][PAWN] as i32
                } else {
                    873 * PIECE_VALUE[victim] / 128 + capt_hist
                }
            } else {
                (2252 * self.butterfly[pos.stm][m.from_to()] as i32
                    + 1126 * self.cont_score(ply, 1, idx)
                    + 1093 * self.cont_score(ply, 2, idx))
                    / 1024
            };
            self.ss[ply].stat_score = stat_score;
            self.ss[ply].move_count = move_count;
            self.make(ply, m);
            debug_assert_eq!(gives_check, self.pos[ply + 1].checkers != 0);
            let mut new_depth = new_depth_base + ext;
            let nodes_before = self.nodes;
            let mut score;

            // Late move reductions
            if tt_pv {
                r -= 3023
                    + pv_node as i32 * 1004
                    + (tt_score > alpha) as i32 * 885
                    + (tt_depth >= depth) as i32 * (816 + cut_node as i32 * 940);
            }
            r += 697;
            r -= move_count * 65;
            r -= corr_raw.abs() / 26310;
            if cut_node {
                r += tune::T_LmrCutNode.get() + 933 * tt_move.is_none() as i32;
            }
            if self.tactical_helper {
                r -= 1024;
            }
            if tt_capture {
                r += 1079;
            }
            let cutoffs = self.ss[ply + 1].cutoff_cnt;
            if cutoffs > 1 {
                r += 264 + 1095 * (cutoffs > 2) as i32 + 1138 * all_node as i32;
            } else if m == tt_move {
                r -= 2179;
            }
            r -= stat_score * 439 / 4096;
            if !capture && alpha.abs() < MATE_BOUND {
                r += 3 * (alpha - eval).clamp(-64, 96);
            }
            if all_node {
                r += r * 276 / (256 * depth + 268);
            }
            if self.seek_mate && r > 0 {
                r += (if gives_check { -r } else { r }) / 4;
            }
            if depth >= 2 && move_count > 1 {
                // A negative reduction may extend, but not deep in the tree.
                let cap = if (ply as i32) < 2 * self.root_depth { 2 } else { 0 };
                let d = (new_depth + (-r / 1024).min(cap)).max(1) + pv_node as i32;
                self.ss[ply].reduction = new_depth - d;
                score = -self.search(ply + 1, -alpha - 1, -alpha, d, true, false);
                self.ss[ply].reduction = 0;
                if score > alpha {
                    let deeper = d < new_depth && score > best + 53 && (!DEEPER_PLY_GUARD || (ply as i32) < 2 * self.root_depth);
                    let shallower = score < best + 8;
                    new_depth += deeper as i32 - shallower as i32;
                    if new_depth > d {
                        score = -self.search(ply + 1, -alpha - 1, -alpha, new_depth, !cut_node, false);
                    }
                    // The move did better than its late position suggested.
                    self.update_cont(ply, in_check, idx, 1334);
                }
            } else if !pv_node || move_count > 1 {
                // No LMR here, but a very large reduction still costs a ply or two.
                if tt_move.is_none() {
                    r += 1127;
                }
                let d = new_depth - (r > 5234) as i32 - (r > 5487 && new_depth > 2) as i32;
                score = -self.search(ply + 1, -alpha - 1, -alpha, d, !cut_node, false);
            } else {
                score = alpha + 1;
            }
            if pv_node && (move_count == 1 || score > alpha) {
                // Extend the hash move rather than drop into qsearch.
                if m == tt_move && ((tt_score != NONE && tt_score.abs() >= MATE_BOUND && tt_depth > 0) || tt_depth > 1) {
                    new_depth = new_depth.max(1);
                }
                score = -self.search(ply + 1, -beta, -alpha, new_depth, false, true);
            }
            self.unmake();
            if self.stopped {
                return 0;
            }
            if root {
                let n = self.nodes - nodes_before;
                self.root_nodes[m.from_to()] += n;
                // Stockfish's effort-weighted moving averages of the root move's score.
                if let Some(rs) = self.root_stats.iter_mut().find(|r| r.mv == m) {
                    rs.effort += n;
                    let e_prev = (rs.effort - n).max(1);
                    let w = ((32 * n * 2) / (n * 2 + 3 * e_prev)).clamp(12, 24) as i64;
                    let w_mss = w.min(16);
                    let v = score as i64;
                    let v2 = v * v.abs();
                    rs.avg = if rs.avg == -INF { score } else { ((v * w + rs.avg as i64 * (32 - w)) / 32) as i32 };
                    rs.mss = if rs.mss == MSS_NONE { v2 } else { (v2 * w_mss + rs.mss * (32 - w_mss)) / 32 };
                }
            }

            // Equal scores late in the tree occasionally count as better (Stockfish).
            let inc = (score == best
                && ply as i32 + 2 >= self.root_depth
                && (self.nodes as i32 & 14) == 0
                && (score.abs() + 1) < MATE_BOUND) as i32;
            if score + inc > best {
                best = score;
                if score + inc > alpha {
                    if root && move_count > 1 && self.pv_idx == 0 {
                        self.shared.best_move_changes.0.fetch_add(1, Relaxed);
                        self.iter_unstable = true;
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
                        self.ss[ply].cutoff_cnt += (ext < 2 || pv_node) as i32;
                        break;
                    }
                    // Having raised alpha, search the remaining moves shallower.
                    if depth > 3 && depth < 12 && score.abs() < MATE_BOUND {
                        depth -= 3;
                    }
                    alpha = score;
                }
            }
            if m != best_move && move_count as usize <= SEARCHED_CAP {
                if capture {
                    noisies[n_noisies] = m;
                    n_noisies += 1;
                } else {
                    quiets[n_quiets] = m;
                    n_quiets += 1;
                }
            }
        }

        // A fail high is averaged toward beta.
        if best >= beta && best.abs() < MATE_BOUND && alpha.abs() < MATE_BOUND {
            best = (best * depth + beta) / (depth + 1);
        }

        if move_count == 0 {
            best = if !excluded.is_none() {
                alpha
            } else if in_check {
                -MATE + ply as i32
            } else {
                0
            };
        } else if !best_move.is_none() {
            // The more moves had to be tried first, the more the best one is worth.
            let prev_stat = if ply >= 1 { self.ss[ply - 1].stat_score } else { 0 };
            let mut bonus = stat_bonus(depth) + 364 * (best_move == tt_move) as i32 + prev_stat / 28;
            bonus += (bonus * (n_quiets + n_noisies) as i32).div_euclid(256);
            let malus = stat_malus(depth);
            if !best_move.is_capture_stage() {
                // Killers and the countermove are Esper's own, kept on quiet cutoffs.
                if KILLERS && best >= beta {
                    let k = &mut self.ss[ply].killers;
                    if k[0] != best_move {
                        k[1] = k[0];
                        k[0] = best_move;
                    }
                    if ply >= 1 && self.ss[ply - 1].cont != NO_CONT {
                        let c = self.ss[ply - 1].cont;
                        self.counter[c] = best_move;
                    }
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
            if !pv_node {
                hist_update(&mut self.tt_move_hist, if best_move == tt_move { 918 } else { -747 }, 8192);
            }
        } else if ply >= 1 {
            // Nothing here beat alpha, so the opponent's previous move did well.
            let prev = self.ss[ply - 1];
            if prev.cont != NO_CONT && !prev.mv.is_capture() {
                let prev_in_check = self.pos[ply - 1].in_check();
                let mut scale = -241 + (59 * depth).min(420) + 186 * (prev.move_count > 9) as i32;
                scale -= prev.stat_score / 98;
                scale += 142 * (!in_check && static_eval != NONE && best <= static_eval - 106) as i32;
                scale += 159 * (!prev_in_check && prev.static_eval != NONE && best <= -prev.static_eval - 68) as i32;
                let scaled = (150 * depth - 85).min(1337) * scale.max(0);
                self.update_cont(ply - 1, prev_in_check, prev.cont, scaled * 263 / 16384);
                hist_update(&mut self.butterfly[pos.stm ^ 1][prev.mv.from_to()], scaled * 215 / 32768, MAIN_HIST_LIMIT);
                if prev.cont / 64 & 7 != PAWN && !prev.mv.is_promo() {
                    let i = self.pawn_hist_base(&pos) + prev.cont;
                    hist_update_at(&self.hist.pawn_hist[i], scaled * 324 / 8192, PAWN_HIST_LIMIT);
                }
            } else if prev.cont != NO_CONT {
                // A capture: credit it in the capture history.
                let before = &self.pos[ply - 1];
                let to = prev.mv.to();
                let captured = if prev.mv.flag() == FLAG_EP { PAWN } else { piece_type(before.board[to]) };
                hist_update(&mut self.capture_hist[pos.board[to] as usize][to][captured], 892, CAPTURE_HIST_LIMIT);
            }
        }
        if pv_node {
            best = best.min(tb_ceiling);
        }
        // A position that fails low under a PV parent is marked as PV too.
        if best <= alpha && ply >= 1 {
            tt_pv |= self.ss[ply - 1].tt_pv;
            self.ss[ply].tt_pv = tt_pv;
        }
        if excluded.is_none() && !(root && self.pv_idx > 0) {
            let bound = if best >= beta {
                BOUND_LOWER
            } else if pv_node && !best_move.is_none() {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            let d = if move_count != 0 { depth } else { (depth + 6).min(MAX_PLY as i32 - 1) };
            self.tt.store(pos.key, best_move, score_to_tt(best, ply), raw_eval, d, bound, tt_pv);
        }
        // Correction history: update when the result lies on the side of the static
        // eval that the outcome (a best move or none) says it should.
        if !in_check
            && static_eval != NONE
            && !(!best_move.is_none() && best_move.is_capture())
            && (best > static_eval) == !best_move.is_none()
        {
            let k = if best_move.is_none() { 18 } else { 12 };
            let bonus = ((best - static_eval) * depth * k / 128).clamp(-SF_CORR_LIMIT / 4, SF_CORR_LIMIT / 4);
            self.update_correction_sf(ply, 1061 * bonus / 1024);
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

    fn root_stat(&self, m: Move) -> (i32, i64, u64) {
        self.root_stats.iter().find(|r| r.mv == m).map_or((-INF, MSS_NONE, 0), |r| (r.avg, r.mss, r.effort))
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
        self.root_stats = root_moves.iter().map(|&mv| RootStat { mv, effort: 0, avg: -INF, mss: MSS_NONE }).collect();
        self.avg_score = -INF;
        let main = self.id == 0;
        let multipv = if main { self.multipv.min(root_moves.len()).max(1) } else { 1 };
        // Only the main thread stops at a depth limit; helpers run until told to stop.
        let max_depth = if main {
            self.limits.depth.unwrap_or(MAX_PLY as i32 - 1).clamp(1, MAX_PLY as i32 - 1)
        } else {
            MAX_PLY as i32 - 1
        };
        let mut prev_lines: Vec<Move> = Vec::new();
        let mut prev_scores = vec![-INF; multipv];
        let mut last_best = Move::NONE;
        let mut last_best_depth = 0;
        let start_score = if self.prev_best_score == NONE { 0 } else { self.prev_best_score };
        let mut iter_values = [start_score; 4];
        let mut iter_idx = 0;
        let mut time_reduction = 1.0f64;
        let mut tot_changes = 0.0f64;
        let mut tm_trace = String::new();
        self.search_again = 0;
        self.fail_high_recovery = 0;

        for depth in 1..=max_depth {
            if !self.shared.increase_depth.0.load(Relaxed) {
                self.search_again += 1;
            }
            self.iter_unstable = false;
            let mut lines: Vec<Line> = Vec::with_capacity(multipv);
            self.root_excluded.clear();
            for pv_idx in 0..multipv {
                self.pv_idx = pv_idx;
                self.seldepth = 0;
                self.root_depth = depth;
                // This line's move from the previous iteration leads, and its score
                // averages centre the aspiration window.
                let line_move = prev_lines.get(pv_idx).copied().unwrap_or(Move::NONE);
                self.root_move = line_move;
                let (avg, mss, _) = self.root_stat(line_move);
                let prev = prev_scores[pv_idx];
                let mate_threshold = 750 + 220000 / (depth * depth);
                self.seek_mate = SF_SEEK_MATE && prev != -INF && prev.abs() >= mate_threshold;
                if SF_EVAL_SCALE {
                    let opt = tune::T_OptimismScale.get() * avg / (avg.abs() + tune::T_OptimismDiv.get());
                    self.optimism[root.stm] = opt;
                    self.optimism[root.stm ^ 1] = -opt;
                }
                let mut delta = (5 + (self.id % 8) as i64 + mss.abs() / 10193).min(2 * INF as i64) as i32;
                let mut alpha = (avg as i64 - delta as i64).max(-INF as i64) as i32;
                let mut beta = (avg as i64 + delta as i64).min(INF as i64) as i32;
                let mut fail_highs = 0;
                if pv_idx == 0 {
                    self.fail_high_recovery = (self.fail_high_recovery - 2).max(0);
                }
                let score = loop {
                    let adjusted =
                        (depth - fail_highs - self.fail_high_recovery - 3 * (self.search_again + 1) / 4).max(1);
                    let s = self.search(0, alpha, beta, adjusted, false, true);
                    if self.stopped {
                        break s;
                    }
                    let label = if s <= alpha { " upperbound" } else { " lowerbound" };
                    if s <= alpha && alpha > -INF {
                        beta = alpha;
                        alpha = (s - delta).max(-INF);
                        fail_highs = 0;
                        self.iter_unstable = true;
                        if main {
                            self.shared.stop_on_ponderhit.0.store(false, Relaxed);
                        }
                    } else if s >= beta && beta < INF {
                        alpha = (beta - delta).max(alpha);
                        beta = (s + delta).min(INF);
                        fail_highs += 1;
                        // The move that failed high leads the re-search, as Stockfish's
                        // stable sort of the root moves arranges; otherwise it has to beat
                        // the old best again, counting another best-move change each time.
                        if TM_RESEARCH_FIRST && self.pv_len[0] > 0 {
                            self.root_move = self.pv[0][0];
                        }
                    } else {
                        break s;
                    }
                    if main && self.start.elapsed().as_millis() > 2500 && self.pv_len[0] > 0 {
                        let line = Line { score: s, seldepth: self.seldepth, pv: self.pv[0][..self.pv_len[0]].to_vec() };
                        self.print_info(depth, pv_idx, &line, label);
                    }
                    delta += 47 * delta / 128;
                    if delta > 1500 {
                        alpha = -INF;
                        beta = INF;
                    }
                };
                if fail_highs > 0 && pv_idx == 0 {
                    self.fail_high_recovery = (fail_highs + 1) / 2 + 2;
                }
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
                self.root_excluded.push(pv[0]);
                lines.push(Line { score, seldepth: self.seldepth, pv });
            }
            self.root_excluded.clear();
            self.pv_idx = 0;
            if self.stopped || lines.len() < multipv {
                break;
            }
            lines.sort_by_key(|l| -l.score);
            prev_lines = lines.iter().map(|l| l.pv[0]).collect();
            for (i, l) in lines.iter().enumerate() {
                prev_scores[i] = l.score;
            }
            self.completed_depth = depth;
            self.best_move = lines[0].pv[0];
            self.ponder_move = lines[0].pv.get(1).copied().unwrap_or(Move::NONE);
            self.best_score = lines[0].score;
            self.best_pv.clear();
            self.best_pv.extend_from_slice(&lines[0].pv);
            self.best_seldepth = lines[0].seldepth;
            self.avg_score = self.root_stat(self.best_move).0;
            if self.best_move != last_best {
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
            let best = lines[lines.len() - 1].score;
            let pondering = self.shared.pondering.0.load(Relaxed);
            if self.time_managed
                && !self.shared.stop.0.load(Relaxed)
                && !self.shared.stop_on_ponderhit.0.load(Relaxed)
            {
                let prev_avg = if self.prev_avg_score == NONE { best } else { self.prev_avg_score };
                let falling_w = tune::T_TimeFalling.get() as f64 / 100.0;
                let falling = ((11.48 + falling_w * (prev_avg - best) as f64 + 1.1 * (iter_values[iter_idx] - best) as f64)
                    / 100.0)
                    .clamp(0.576, 1.728);
                time_reduction =
                    interpolate((depth - last_best_depth) as f64, 4.96, 18.79, 0.639, 1.712).clamp(0.629, 1.544);
                let reduction = (1.468 + self.prev_time_reduction) / (2.284 * time_reduction);
                tot_changes = tot_changes / 2.0 + self.shared.best_move_changes.0.swap(0, Relaxed) as f64;
                let instab_w = tune::T_TimeInstability.get() as f64 / 1000.0;
                let instability = (1.077 + instab_w * tot_changes / self.threads as f64).min(2.5);
                let effort = (self.root_stat(self.best_move).2 * 100000 / self.nodes.max(1)) as f64;
                let high_effort = interpolate(effort, 75800.0, 104510.0, 0.969, 0.714).clamp(0.693, 0.838);
                let mut total = self.soft_ms as f64 * falling * reduction * instability * high_effort;
                if root_moves.len() == 1 {
                    total = total.min(100.0);
                }
                self.soft_deadline = (1.15 * total).min(self.hard_ms as f64) as u64;
                tm_trace = format!(
                    "depth {depth} falling {falling:.2} reduction {reduction:.2} instability {instability:.2} effort {:.0}k/{high_effort:.2} total {total:.0}",
                    effort / 1000.0
                );
                let elapsed = self.start.elapsed().as_millis() as f64;
                let stop_limit = if self.iter_unstable { total } else { total * 0.65 };
                if elapsed > stop_limit.min(self.hard_ms as f64)
                    || lines[multipv - 1].score >= MATE - 3
                    || lines[0].score == -(MATE - 2)
                {
                    if pondering {
                        self.shared.stop_on_ponderhit.0.store(true, Relaxed);
                    } else {
                        break;
                    }
                } else {
                    self.shared.increase_depth.0.store(pondering || elapsed <= total * 0.5, Relaxed);
                }
            }
            iter_values[iter_idx] = best;
            iter_idx = (iter_idx + 1) & 3;
            if let Some(n) = self.limits.nodes {
                if self.shared.total_nodes(self.threads) >= n {
                    break;
                }
            }
        }
        if main && self.time_managed && debug_time() {
            eprintln!(
                "time: optimum {} maximum {} used {} | {}",
                self.soft_ms,
                self.hard_ms,
                self.start.elapsed().as_millis(),
                tm_trace
            );
        }
        if main {
            self.prev_time_reduction = time_reduction;
        }
        self.shared.nodes[self.id].0.store(self.nodes, Relaxed);
        self.shared.tb_hits[self.id].0.store(self.tb_hits, Relaxed);
    }
}

#[derive(Clone)]
struct ThreadResult {
    completed_depth: i32,
    best_score: i32,
    avg_score: i32,
    best_seldepth: usize,
    best_pv: Vec<Move>,
}

impl ThreadResult {
    fn from_searcher(s: &Searcher) -> Self {
        ThreadResult {
            completed_depth: s.completed_depth,
            best_score: s.best_score,
            avg_score: s.avg_score,
            best_seldepth: s.best_seldepth,
            best_pv: s.best_pv.clone(),
        }
    }
}

/// Pick the thread whose result to play (Stockfish's get_best_thread). Every
/// thread votes for its best move with its score above the lowest one plus 14;
/// the move with the most votes wins (ties to the longer PV). A thread with a
/// proven result is preferred, the more decisive the better.
fn best_thread(results: &[ThreadResult]) -> usize {
    let valid = |r: &ThreadResult| r.completed_depth > 0 && !r.best_pv.is_empty();
    if !valid(&results[0]) {
        return 0;
    }
    let min_score = results.iter().filter(|r| valid(r)).map(|r| r.best_score).min().unwrap_or(0);
    let mut votes = vec![0i64; 65536];
    for r in results.iter().filter(|r| valid(r)) {
        votes[r.best_pv[0].0 as usize] += (r.best_score - min_score + 14) as i64;
    }
    let decisive = |s: i32| s.abs() >= MATE_BOUND;
    let mut best = 0;
    for (i, r) in results.iter().enumerate().skip(1) {
        if !valid(r) {
            continue;
        }
        let b = &results[best];
        let (vr, vb) = (votes[r.best_pv[0].0 as usize], votes[b.best_pv[0].0 as usize]);
        let better = if decisive(b.best_score) {
            decisive(r.best_score) && r.best_score.abs() > b.best_score.abs()
        } else {
            decisive(r.best_score)
                || (r.best_score > -MATE_BOUND && (vr > vb || (vr == vb && r.best_pv.len() > b.best_pv.len())))
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
        contempt: i32,
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
    done_flag: Arc<PaddedAtomicBool>,
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
        let hist = Arc::new(SharedHist::new(if SHARED_HISTORY { threads } else { 1 }));
        let thread_hist = |id: usize| if SHARED_HISTORY || id == 0 { hist.clone() } else { Arc::new(SharedHist::new(1)) };
        let main = Searcher::new(0, threads, net.clone(), tt.clone(), shared.clone(), thread_hist(0));
        let mut helpers = Vec::with_capacity(threads.saturating_sub(1));
        for id in 1..threads {
            let state = Arc::new((
                Mutex::new(WorkerState { gen: 0, job: WorkerJob::Idle, done: true, result: None }),
                Condvar::new(),
            ));
            let done_flag = Arc::new(PaddedAtomicBool(AtomicBool::new(true)));
            let st = state.clone();
            let df = done_flag.clone();
            let net = net.clone();
            let tt = tt.clone();
            let shared = shared.clone();
            let hist = thread_hist(id);
            let join = std::thread::Builder::new()
                .stack_size(16 * 1024 * 1024)
                .spawn(move || {
                    let mut searcher = Searcher::new(id, threads, net, tt, shared, hist);
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
                            WorkerJob::Search { pos, keys, limits, overhead, tb, show_wdl, contempt } => {
                                searcher.prepare(&pos, &keys, &limits, 1, overhead, tb);
                                searcher.silent = true;
                                searcher.show_wdl = show_wdl;
                                searcher.contempt = contempt;
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
                        df.0.store(true, std::sync::atomic::Ordering::Release);
                        cv.notify_all();
                    }
                })
                .expect("failed to spawn search worker thread");
            helpers.push(WorkerHandle { state, done_flag, join: Some(join) });
        }
        ThreadPool { main, helpers, threads }
    }

    fn dispatch(&self, mut make_job: impl FnMut(usize) -> WorkerJob) {
        for (idx, h) in self.helpers.iter().enumerate() {
            h.done_flag.0.store(false, Relaxed);
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
            for _ in 0..64 {
                if h.done_flag.0.load(std::sync::atomic::Ordering::Acquire) {
                    break;
                }
                std::hint::spin_loop();
            }
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
        contempt: i32,
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
        self.main.contempt = contempt;

        shared.increase_depth.0.store(true, Relaxed);
        shared.stop_on_ponderhit.0.store(false, Relaxed);
        let shared_keys: Arc<[u64]> = Arc::from(keys);
        self.dispatch(|_| WorkerJob::Search {
            pos: *pos,
            keys: shared_keys.clone(),
            limits: limits.clone(),
            overhead,
            tb,
            show_wdl,
            contempt,
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

        // Stockfish only votes when no depth limit was given.
        let best = if self.main.limits.depth.is_some() || multipv > 1 { 0 } else { best_thread(&results) };
        let r = results[best].clone();
        // The next move's time management starts from the chosen thread's result.
        if r.completed_depth > 0 {
            self.main.prev_best_score = r.best_score;
            self.main.prev_avg_score = if r.avg_score == -INF { NONE } else { r.avg_score };
        }
        if best != 0 {
            self.main.best_move = r.best_pv[0];
            self.main.ponder_move = r.best_pv.get(1).copied().unwrap_or(Move::NONE);
            self.main.best_score = r.best_score;
            self.main.completed_depth = r.completed_depth;
            if !silent {
                let line = Line { score: r.best_score, seldepth: r.best_seldepth, pv: r.best_pv.clone() };
                self.main.print_info(r.completed_depth, 0, &line, "");
            }
        }
        // Without a reply in the PV, take one from the hash table (Stockfish).
        if self.main.ponder_move.is_none() && !self.main.best_move.is_none() {
            let next = self.main.pos[0].make(self.main.best_move);
            if let Some(t) = self.main.tt.probe(next.key) {
                if !t.mv.is_none() && next.is_pseudo_legal(t.mv) && next.is_legal(t.mv) {
                    self.main.ponder_move = t.mv;
                }
            }
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
