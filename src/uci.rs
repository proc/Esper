//! UCI protocol front end.

use crate::book::{self, Book};
use crate::movegen::{legal_moves, parse_move, parse_san, perft, to_san};
use crate::nnue::{Network, NnueState, DEFAULT_NET_NAME};
use crate::position::*;
use crate::search::{self, Limits, Shared, ThreadPool};
use crate::tb::{self, TbConfig};
use crate::tt::TT;
use std::io::BufRead;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

const NAME: &str = concat!("Esper ", env!("CARGO_PKG_VERSION"));
/// The "Win at Chess" tactical test suite (300 positions).
const WAC_EPD: &str = include_str!("../suites/wac.epd");
const SEARCH_STACK: usize = 16 * 1024 * 1024;
const BENCH_FENS: [&str; 12] = [
    START_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 9",
    "2rq1rk1/pb1n1ppp/1p2pn2/2pp4/2PP4/1PN1PN2/PB2BPPP/2RQ1RK1 w - - 0 12",
    "8/8/1p1k4/1P1p2p1/3P2P1/3K4/8/8 w - - 0 1",
    "3r2k1/pp3ppp/2n1b3/2p1P3/2P2B2/1P3N1P/P4PP1/3R2K1 b - - 0 22",
    "r2q1rk1/1b2bppp/p2p1n2/1pn1p3/4P3/1BP2N1P/PP1N1PP1/R1BQR1K1 w - - 2 14",
    "6k1/5p2/6p1/8/7p/8/6PP/6K1 b - - 0 1",
    "4rrk1/pp1n1pp1/q5p1/P1pP4/2n3P1/7P/1P3PB1/R1BQ1RK1 w - - 3 22",
];

struct SearchTask {
    pool: ThreadPool,
    pos: Position,
    keys: Vec<u64>,
    limits: Limits,
    multipv: usize,
    overhead: u64,
    tb_cfg: TbConfig,
    show_wdl: bool,
    silent: bool,
    shared: Arc<Shared>,
    root_tb: Option<tb::RootResult>,
}

enum CoordCmd {
    Run(Box<SearchTask>),
    Quit,
}

struct CoordState {
    cmd: Option<CoordCmd>,
    result: Option<ThreadPool>,
    busy: bool,
}

pub struct Engine {
    net: Arc<Network>,
    tt: Arc<TT>,
    shared: Arc<Shared>,
    pool: Option<ThreadPool>,
    coord: Arc<(Mutex<CoordState>, Condvar)>,
    coord_thread: Option<JoinHandle<()>>,
    pos: Position,
    keys: Vec<u64>,
    threads: usize,
    /// Thread count given on the command line; advertised as the UCI default.
    default_threads: usize,
    multipv: usize,
    overhead: u64,
    show_wdl: bool,
    tb: TbConfig,
    /// Play from the opening book (`OwnBook`); off by default.
    own_book: bool,
    book_file: String,
    /// Loaded on first use, so an engine without OwnBook never touches it.
    book: Option<Book>,
    /// Always play the most heavily weighted book move instead of a weighted draw.
    book_best: bool,
    /// Last move number played from the book; 0 means no limit.
    book_depth: u32,
}

pub fn load_default_network() -> Result<Network, String> {
    if let Some(net) = Network::embedded() {
        return net;
    }
    let mut candidates = vec![std::path::PathBuf::from(DEFAULT_NET_NAME)];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(DEFAULT_NET_NAME));
        }
    }
    for c in &candidates {
        if c.exists() {
            return Network::from_file(&c.to_string_lossy());
        }
    }
    Err(format!("network file {DEFAULT_NET_NAME} not found; put it next to the executable or in the working directory"))
}

impl Engine {
    pub fn new(net: Network) -> Engine {
        let net = Arc::new(net);
        let tt = Arc::new(TT::new(16));
        let shared = Arc::new(Shared::new());
        let pos = Position::startpos();
        let coord = Arc::new((Mutex::new(CoordState { cmd: None, result: None, busy: false }), Condvar::new()));
        let coord_bg = coord.clone();
        let coord_thread = std::thread::Builder::new()
            .stack_size(SEARCH_STACK)
            .spawn(move || loop {
                let cmd = {
                    let (lock, cv) = &*coord_bg;
                    let mut guard = lock.lock().unwrap();
                    while guard.cmd.is_none() {
                        guard = cv.wait(guard).unwrap();
                    }
                    guard.cmd.take().unwrap()
                };
                match cmd {
                    CoordCmd::Quit => break,
                    CoordCmd::Run(task) => {
                        let SearchTask {
                            mut pool,
                            pos,
                            keys,
                            limits,
                            multipv,
                            overhead,
                            tb_cfg,
                            show_wdl,
                            silent,
                            shared,
                            root_tb,
                        } = *task;
                        let hold = limits.infinite;
                        if let Some(r) = root_tb {
                            let outcome = tb::outcome(r.wdl, &tb_cfg);
                            let score = match outcome {
                                1 => search::TB_CP - r.dtz as i32,
                                -1 => -search::TB_CP + r.dtz as i32,
                                _ => 0,
                            };
                            let wdl_str = if show_wdl {
                                match outcome {
                                    1 => " wdl 1000 0 0",
                                    -1 => " wdl 0 0 1000",
                                    _ => " wdl 0 1000 0",
                                }
                            } else {
                                ""
                            };
                            println!(
                                "info depth 1 seldepth 1 multipv 1 score cp {score}{wdl_str} nodes 0 nps 0 tbhits 1 time 0 pv {}",
                                r.mv.to_uci()
                            );
                            while (shared.pondering.0.load(Relaxed) || hold) && !shared.stop.0.load(Relaxed) {
                                std::thread::sleep(std::time::Duration::from_millis(1));
                            }
                            println!("bestmove {}", r.mv.to_uci());
                        } else {
                            let (best, ponder, _) =
                                pool.run(&pos, &keys, &limits, multipv, overhead, tb_cfg, silent, show_wdl, &shared);
                            if !silent {
                                if best.is_none() {
                                    println!(
                                        "info depth 0 score {}",
                                        if pool.main.in_check_root() { "mate 0" } else { "cp 0" }
                                    );
                                    println!("bestmove 0000");
                                } else if ponder.is_none() {
                                    println!("bestmove {}", best.to_uci());
                                } else {
                                    println!("bestmove {} ponder {}", best.to_uci(), ponder.to_uci());
                                }
                            }
                        }
                        let (lock, cv) = &*coord_bg;
                        let mut guard = lock.lock().unwrap();
                        guard.result = Some(pool);
                        guard.busy = false;
                        cv.notify_all();
                    }
                }
            })
            .expect("failed to spawn search coordinator thread");
        let mut e = Engine {
            net,
            tt,
            shared,
            pool: None,
            coord,
            coord_thread: Some(coord_thread),
            keys: vec![pos.key],
            pos,
            threads: 1,
            default_threads: 1,
            multipv: 1,
            overhead: 25,
            show_wdl: false,
            tb: TbConfig::DEFAULT,
            own_book: false,
            book_file: book::EMBEDDED_NAME.to_string(),
            book: None,
            book_best: false,
            book_depth: 0,
        };
        e.rebuild_searchers();
        e
    }

    /// Set the thread count from the command line. A later `setoption name Threads`
    /// overrides it.
    pub fn set_default_threads(&mut self, n: usize) {
        self.threads = n.clamp(1, 512);
        self.default_threads = self.threads;
        self.rebuild_searchers();
    }

    fn rebuild_searchers(&mut self) {
        self.wait();
        self.pool = None;
        self.pool = Some(ThreadPool::new(self.threads, self.net.clone(), self.tt.clone(), self.shared.clone()));
    }

    /// Block until a running search has finished and handed its state back.
    fn wait(&mut self) {
        let (lock, cv) = &*self.coord;
        let mut guard = lock.lock().unwrap();
        while guard.busy {
            guard = cv.wait(guard).unwrap();
        }
        if let Some(pool) = guard.result.take() {
            self.pool = Some(pool);
        }
    }

    fn stop(&mut self) {
        self.shared.stop.0.store(true, Relaxed);
        self.shared.pondering.0.store(false, Relaxed);
        self.wait();
    }

    fn set_position(&mut self, args: &[&str]) {
        let i;
        let mut pos = match args.first() {
            Some(&"startpos") => {
                i = 1;
                Position::startpos()
            }
            Some(&"fen") => {
                let end = args.iter().position(|&a| a == "moves").unwrap_or(args.len());
                let fen = args[1..end].join(" ");
                i = end;
                match Position::from_fen(&fen) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("info string invalid fen: {e}");
                        return;
                    }
                }
            }
            _ => return,
        };
        let mut keys = vec![pos.key];
        if args.get(i) == Some(&"moves") {
            for mv in &args[i + 1..] {
                match parse_move(&pos, mv) {
                    Some(m) => {
                        pos = pos.make(m);
                        if pos.rule50 == 0 {
                            keys.clear();
                        }
                        keys.push(pos.key);
                    }
                    None => {
                        println!("info string illegal move: {mv}");
                        break;
                    }
                }
            }
        }
        self.pos = pos;
        self.keys = keys;
    }

    fn go(&mut self, args: &[&str]) {
        if args.first() == Some(&"perft") {
            let depth = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
            return self.perft(depth);
        }
        let mut limits = Limits::default();
        let mut i = 0;
        let num = |i: usize| args.get(i + 1).and_then(|s| s.parse::<i64>().ok()).map(|v| v.max(0) as u64);
        while i < args.len() {
            match args[i] {
                "wtime" => limits.time[WHITE] = num(i),
                "btime" => limits.time[BLACK] = num(i),
                "winc" => limits.inc[WHITE] = num(i).unwrap_or(0),
                "binc" => limits.inc[BLACK] = num(i).unwrap_or(0),
                "movestogo" => limits.movestogo = num(i).filter(|&v| v > 0),
                "depth" => limits.depth = num(i).map(|v| v as i32),
                "nodes" => limits.nodes = num(i),
                "movetime" => limits.movetime = num(i),
                "infinite" => limits.infinite = true,
                "ponder" => limits.ponder = true,
                "searchmoves" => {
                    while let Some(m) = args.get(i + 1).and_then(|s| parse_move(&self.pos, s)) {
                        limits.searchmoves.push(m);
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        if self.own_book && !limits.ponder && !limits.infinite && limits.searchmoves.is_empty() {
            if let Some(m) = self.book_move() {
                let next = self.pos.make(m);
                let seen = |key: u64| key == next.key || self.keys.contains(&key);
                let ponder = self.book.as_ref().and_then(|b| b.probe(&next, true, |p| !seen(next.make(p).key)));
                println!("info string book move {}", to_san(&self.pos, m));
                match ponder {
                    Some(p) => println!("bestmove {} ponder {}", m.to_uci(), p.to_uci()),
                    None => println!("bestmove {}", m.to_uci()),
                }
                return;
            }
        }
        self.start_search(limits, false);
    }

    /// Load the book named by `BookFile` if it is not loaded yet.
    fn ensure_book(&mut self) -> bool {
        if self.book.is_none() {
            match Book::load(&self.book_file) {
                Ok(b) => {
                    println!("info string book {} loaded: {} entries", b.name, b.len());
                    self.book = Some(b);
                }
                Err(e) => {
                    println!("info string could not load book: {e}");
                    self.own_book = false;
                }
            }
        }
        self.book.is_some()
    }

    fn book_move(&mut self) -> Option<Move> {
        if self.book_depth != 0 && self.pos.fullmove as u32 > self.book_depth {
            return None;
        }
        if !self.ensure_book() {
            return None;
        }
        // Book lines can loop (engines shuffled pieces in the source games), and
        // Polyglot keys know nothing of the game's history, so never let the book
        // repeat a position this game has already had.
        let (pos, keys) = (&self.pos, &self.keys);
        self.book.as_ref()?.probe(pos, self.book_best, |m| !keys.contains(&pos.make(m).key))
    }

    /// `book`: list the book moves of the current position with their weights.
    fn show_book(&mut self) {
        if !self.ensure_book() {
            return;
        }
        let moves = self.book.as_ref().unwrap().moves(&self.pos);
        if moves.is_empty() {
            println!("no book moves (key {:016x})", book::hash(&self.pos));
            return;
        }
        let total: u64 = moves.iter().map(|m| m.weight as u64).sum::<u64>().max(1);
        for m in moves {
            println!("{:<8} {:>5}  {:5.1}%", to_san(&self.pos, m.mv), m.weight, 100.0 * m.weight as f64 / total as f64);
        }
    }

    fn start_search(&mut self, mut limits: Limits, silent: bool) {
        self.stop();
        // With the root position in the tablebases its exact value is known. A won
        // or lost position is played straight from the distance-to-zero tables; in
        // a drawn one the search still runs, but only over moves that hold the draw.
        let mut root_tb = if !silent && self.multipv == 1 && limits.searchmoves.is_empty() {
            tb::probe_root(&self.pos, &self.tb)
        } else {
            None
        };
        if let Some(r) = &root_tb {
            if tb::outcome(r.wdl, &self.tb) == 0 && r.keeping.len() > 1 {
                limits.searchmoves = r.keeping.clone();
                root_tb = None;
            }
        }
        self.shared.stop.0.store(false, Relaxed);
        self.shared.pondering.0.store(limits.ponder, Relaxed);
        let task = Box::new(SearchTask {
            pool: self.pool.take().unwrap(),
            pos: self.pos,
            keys: self.keys.clone(),
            limits,
            multipv: self.multipv,
            overhead: self.overhead,
            tb_cfg: self.tb,
            show_wdl: self.show_wdl,
            silent,
            shared: self.shared.clone(),
            root_tb,
        });
        let (lock, cv) = &*self.coord;
        let mut guard = lock.lock().unwrap();
        guard.busy = true;
        guard.cmd = Some(CoordCmd::Run(task));
        cv.notify_all();
    }

    fn perft(&mut self, depth: u32) {
        let start = Instant::now();
        let mut total = 0;
        for m in legal_moves(&self.pos) {
            let n = if depth <= 1 { 1 } else { perft(&self.pos.make(m), depth - 1) };
            println!("{}: {}", m.to_uci(), n);
            total += n;
        }
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        println!("\nNodes searched: {total}\nTime: {:.3}s  NPS: {:.0}", secs, total as f64 / secs);
    }

    fn set_option(&mut self, args: &[&str]) {
        let vi = args.iter().position(|&a| a == "value");
        let ni = args.iter().position(|&a| a == "name");
        let Some(ni) = ni else { return };
        let name = args[ni + 1..vi.unwrap_or(args.len())].join(" ").to_ascii_lowercase();
        let value = vi.map(|v| args[v + 1..].join(" ")).unwrap_or_default();
        self.stop();
        match name.as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    // Free the old table before allocating the new one.
                    self.pool = None;
                    self.tt = Arc::new(TT::new(1));
                    self.tt = Arc::new(TT::new(mb.clamp(1, 1 << 20)));
                    self.rebuild_searchers();
                }
            }
            "threads" => {
                if let Ok(n) = value.parse::<usize>() {
                    self.threads = n.clamp(1, 512);
                    self.rebuild_searchers();
                }
            }
            "multipv" => {
                if let Ok(n) = value.parse::<usize>() {
                    self.multipv = n.clamp(1, 256);
                }
            }
            "move overhead" => {
                if let Ok(n) = value.parse::<u64>() {
                    self.overhead = n.min(5000);
                }
            }
            "uci_showwdl" => self.show_wdl = value.trim().eq_ignore_ascii_case("true"),
            "clear hash" => self.new_game(),
            "syzygypath" => {
                let n = tb::init(&value);
                if n > 0 {
                    println!("info string Syzygy tablebases loaded: up to {n} pieces");
                } else if !value.is_empty() && value != "<empty>" {
                    println!("info string no Syzygy tablebases found in {value}");
                }
            }
            "syzygyprobedepth" => {
                if let Ok(n) = value.parse::<i32>() {
                    self.tb.probe_depth = n.clamp(1, 100);
                }
            }
            "syzygy50moverule" => self.tb.rule50 = value.trim().eq_ignore_ascii_case("true"),
            "syzygyprobelimit" => {
                if let Ok(n) = value.parse::<u32>() {
                    self.tb.limit = n.min(7);
                }
            }
            "evalfile" => {
                let net = if value.is_empty() || value == "<embedded>" || value == "<default>" {
                    load_default_network()
                } else {
                    Network::from_file(&value)
                };
                match net {
                    Ok(n) => {
                        println!("info string loaded network: {}", n.description);
                        self.net = Arc::new(n);
                        self.rebuild_searchers();
                    }
                    Err(e) => println!("info string could not load network: {e}"),
                }
            }
            "ownbook" => {
                self.own_book = value.trim().eq_ignore_ascii_case("true");
                if self.own_book {
                    self.ensure_book();
                }
            }
            "bookfile" => {
                let v = value.trim();
                self.book_file = if v.is_empty() || v == "<empty>" { book::EMBEDDED_NAME.to_string() } else { v.to_string() };
                self.book = None;
                if self.own_book {
                    self.ensure_book();
                }
            }
            "bookbestmove" => self.book_best = value.trim().eq_ignore_ascii_case("true"),
            "bookdepth" => {
                if let Ok(n) = value.parse::<u32>() {
                    self.book_depth = n.min(1000);
                }
            }
            "ponder" => {}
            _ if crate::tune::set_by_name(&name, &value) => {}
            _ => println!("info string unknown option: {name}"),
        }
    }

    fn new_game(&mut self) {
        self.stop();
        if let Some(pool) = self.pool.as_mut() {
            pool.clear_tt(&self.tt);
            pool.clear();
        } else {
            self.tt.clear();
        }
    }

    fn eval(&mut self) {
        self.wait();
        if self.pos.in_check() {
            println!("Evaluation: none (in check)");
            return;
        }
        let pos = self.pos;
        let s = &mut self.pool.as_mut().unwrap().main;
        let (raw, scaled) = s.eval_position(&pos);
        let sign = if pos.stm == WHITE { 1 } else { -1 };
        println!("NNUE raw output: {raw} (side to move, internal units)");
        println!(
            "Evaluation: {:+.2} pawns (white side, scaled)",
            (sign * scaled) as f64 / search::NORMALIZE_PAWN as f64
        );
    }

    fn run_bench_fens(&mut self, depth: i32, verbose: bool) -> (u64, u64) {
        let saved = (self.pos, self.keys.clone());
        self.new_game();
        let start = Instant::now();
        let mut nodes = 0;
        for fen in BENCH_FENS {
            self.pos = Position::from_fen(fen).unwrap();
            self.keys = vec![self.pos.key];
            let limits = Limits { depth: Some(depth), ..Limits::default() };
            self.start_search(limits, true);
            self.wait();
            let n = self.shared.total_nodes(self.threads);
            if verbose {
                let bm = self.pool.as_ref().unwrap().main.best_move;
                println!("{:<72} bestmove {:<6} nodes {}", fen, bm.to_uci(), n);
            }
            nodes += n;
        }
        let ms = start.elapsed().as_millis().max(1) as u64;
        self.new_game();
        self.pos = saved.0;
        self.keys = saved.1;
        (nodes, ms)
    }

    fn bench(&mut self, depth: i32) {
        let (nodes, ms) = self.run_bench_fens(depth, true);
        println!("\nTime: {ms} ms\nNodes: {nodes}\nNPS: {}", nodes * 1000 / ms);
    }

    /// Benchmark SMP scaling across thread counts up to `max_threads`.
    fn smp_bench(&mut self, depth: i32, max_threads: usize) {
        let saved_threads = self.threads;
        let max_t = max_threads.clamp(1, 512);
        let mut counts = Vec::new();
        let mut t = 1;
        while t < max_t {
            counts.push(t);
            t *= 2;
        }
        counts.push(max_t);

        println!("{:<8} {:>10} {:>12} {:>12} {:>8}", "Threads", "Time (ms)", "Nodes", "NPS", "Speedup");
        let mut base_nps = 0.0;
        for t in counts {
            self.threads = t;
            self.rebuild_searchers();
            let (nodes, ms) = self.run_bench_fens(depth, false);
            let nps = nodes * 1000 / ms;
            if base_nps == 0.0 {
                base_nps = nps as f64;
            }
            let speedup = nps as f64 / base_nps;
            println!("{:<8} {:>10} {:>12} {:>12} {:>7.2}x", t, ms, nodes, nps, speedup);
        }
        if self.threads != saved_threads {
            self.threads = saved_threads;
            self.rebuild_searchers();
        }
    }

    /// Micro-benchmark NNUE evaluation throughput across `BENCH_FENS`.
    fn eval_bench(&mut self, iters: usize) {
        self.wait();
        let positions: Vec<Position> = BENCH_FENS.iter().map(|f| Position::from_fen(f).unwrap()).collect();
        let s = &mut self.pool.as_mut().unwrap().main;
        let start = Instant::now();
        let mut acc: i64 = 0;
        for _ in 0..iters {
            for pos in &positions {
                acc += s.eval_position(pos).0 as i64;
            }
        }
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        let total = (iters * positions.len()) as u64;
        println!(
            "Evals: {total}  Time: {:.3}s  Rate: {:.0} evals/s  (checksum {acc})",
            secs,
            total as f64 / secs
        );
    }

    /// Run standard perft positions, verifying node counts and measuring movegen speed.
    fn perft_bench(&mut self) {
        const SUITE: [(&str, u32, u64); 6] = [
            (START_FEN, 5, 4_865_609),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4_085_603),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674_624),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422_333),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2_103_487),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3_894_594),
        ];
        let start = Instant::now();
        let mut total = 0u64;
        let mut passed = 0usize;
        for (i, &(fen, depth, expected)) in SUITE.iter().enumerate() {
            let pos = Position::from_fen(fen).unwrap();
            let t0 = Instant::now();
            let n = perft(&pos, depth);
            let ms = t0.elapsed().as_millis().max(1) as u64;
            let ok = n == expected;
            if ok {
                passed += 1;
            }
            total += n;
            println!(
                "pos {} (d{}): {:>10} nodes  {:>4} ms  [{}]",
                i + 1,
                depth,
                n,
                ms,
                if ok { "PASS" } else { "FAIL" }
            );
        }
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        println!(
            "\nPassed: {passed}/{}  Nodes: {total}  Time: {:.3}s  NPS: {:.0}",
            SUITE.len(),
            secs,
            total as f64 / secs
        );
    }

    /// Run a tactical test suite in EPD format: search each position for `ms`
    /// milliseconds and compare the move chosen with the `bm` (best move) and `am`
    /// (avoid move) operations.
    fn run_suite(&mut self, name: &str, epd: &str, ms: u64) {
        let saved = (self.pos, self.keys.clone());
        let start = Instant::now();
        let (mut total, mut solved, mut nodes) = (0u32, 0u32, 0u64);
        let mut failed: Vec<String> = Vec::new();
        for (line_no, line) in epd.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Four FEN fields, then `;`-separated operations.
            let fields: Vec<&str> = line.splitn(5, ' ').collect();
            if fields.len() < 5 {
                println!("info string line {}: not an EPD record", line_no + 1);
                continue;
            }
            let fen = format!("{} 0 1", fields[..4].join(" "));
            let pos = match Position::from_fen(&fen) {
                Ok(p) => p,
                Err(e) => {
                    println!("info string line {}: {e}", line_no + 1);
                    continue;
                }
            };
            let (mut bm, mut am, mut id) = (Vec::new(), Vec::new(), format!("#{}", line_no + 1));
            let mut bad_move = false;
            for op in fields[4].split(';') {
                let op = op.trim();
                let (code, rest) = op.split_once(' ').unwrap_or((op, ""));
                match code {
                    "bm" | "am" => {
                        for san in rest.split_whitespace() {
                            match parse_san(&pos, san) {
                                Some(m) if code == "bm" => bm.push(m),
                                Some(m) => am.push(m),
                                None => {
                                    println!("info string line {}: cannot read move {san}", line_no + 1);
                                    bad_move = true;
                                }
                            }
                        }
                    }
                    "id" => id = rest.trim_matches('"').to_string(),
                    _ => {}
                }
            }
            if bad_move || (bm.is_empty() && am.is_empty()) {
                continue;
            }

            self.new_game();
            self.pos = pos;
            self.keys = vec![pos.key];
            self.start_search(Limits { movetime: Some(ms), ..Limits::default() }, true);
            self.wait();
            nodes += self.shared.total_nodes(self.threads);
            let best = self.pool.as_ref().unwrap().main.best_move;
            let ok = !best.is_none() && (bm.is_empty() || bm.contains(&best)) && !am.contains(&best);
            total += 1;
            solved += ok as u32;
            let played = if best.is_none() { "none".to_string() } else { to_san(&pos, best) };
            let mut expect = bm.iter().map(|&m| to_san(&pos, m)).collect::<Vec<_>>().join(" ");
            if !am.is_empty() {
                let avoid = am.iter().map(|&m| to_san(&pos, m)).collect::<Vec<_>>().join(" ");
                expect = format!("{expect} (avoid {avoid})").trim().to_string();
            }
            println!("{:<10} {:<5} played {:<8} expected {}", id, if ok { "ok" } else { "FAIL" }, played, expect);
            if !ok {
                failed.push(id);
            }
        }
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        println!(
            "\n{name}: solved {solved} of {total} ({:.1}%) at {ms} ms per position, {} thread(s)",
            100.0 * solved as f64 / total.max(1) as f64,
            self.threads
        );
        println!("Time: {:.1} s  Nodes: {nodes}  NPS: {:.0}", secs, nodes as f64 / secs);
        if !failed.is_empty() {
            println!("Failed: {}", failed.join(" "));
        }
        self.new_game();
        self.pos = saved.0;
        self.keys = saved.1;
    }

    /// Print the raw network output for every FEN in a file (for regression checks
    /// against a reference implementation).
    fn eval_file(&mut self, path: &str) {
        self.wait();
        let Ok(text) = std::fs::read_to_string(path) else {
            println!("info string cannot read {path}");
            return;
        };
        let s = &mut self.pool.as_mut().unwrap().main;
        for line in text.lines() {
            match Position::from_fen(line) {
                Ok(pos) if !pos.in_check() => println!("{}", s.eval_position(&pos).0),
                _ => println!("none"),
            }
        }
    }

    /// Check the incrementally updated accumulators against full refreshes.
    fn selftest(&mut self, games: usize) {
        self.wait();
        let mut inc = NnueState::new(self.net.clone(), 520);
        let mut fresh = NnueState::new(self.net.clone(), 2);
        let mut seed = 0x1234_5678_9ABC_DEFu64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (mut checked, mut bad) = (0u64, 0u64);
        for _ in 0..games {
            let mut stack = vec![Position::startpos()];
            inc.reset();
            for ply in 0..500 {
                let pos = stack[ply];
                let moves = legal_moves(&pos);
                if moves.is_empty() || pos.rule50 >= 100 || pos.insufficient_material() {
                    break;
                }
                let caps: Vec<Move> = moves.iter().copied().filter(|m| m.is_capture()).collect();
                let m = if !caps.is_empty() && rnd() % 3 == 0 {
                    caps[(rnd() % caps.len() as u64) as usize]
                } else {
                    moves[(rnd() % moves.len() as u64) as usize]
                };
                stack.push(pos.make(m));
                inc.push(&stack, ply + 1);
                // Evaluate only some plies so multi-ply lazy updates get exercised.
                if !stack[ply + 1].in_check() && rnd() % 3 != 0 {
                    let a = inc.evaluate(&stack, ply + 1);
                    fresh.reset();
                    let b = fresh.evaluate(&stack[ply + 1..], 0);
                    checked += 1;
                    if a != b {
                        bad += 1;
                        if bad <= 5 {
                            println!("mismatch {a} vs {b}: {}", stack[ply + 1].to_fen());
                        }
                    }
                }
            }
        }
        println!("selftest: {checked} positions checked, {bad} mismatches");
    }

    /// Execute one command line. Returns false on `quit`.
    pub fn command(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = tokens.first() else { return true };
        let args = &tokens[1..];
        match cmd {
            "uci" => {
                println!("id name {NAME}");
                println!("id author the Esper developers");
                println!("option name Hash type spin default 16 min 1 max 1048576");
                println!("option name Threads type spin default {} min 1 max 512", self.default_threads);
                println!("option name MultiPV type spin default 1 min 1 max 256");
                println!("option name Move Overhead type spin default 25 min 0 max 5000");
                println!("option name Ponder type check default false");
                println!("option name UCI_ShowWDL type check default false");
                println!("option name EvalFile type string default <embedded>");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name SyzygyProbeDepth type spin default 1 min 1 max 100");
                println!("option name Syzygy50MoveRule type check default true");
                println!("option name SyzygyProbeLimit type spin default 7 min 0 max 7");
                println!("option name OwnBook type check default false");
                println!("option name BookFile type string default {}", book::EMBEDDED_NAME);
                println!("option name BookBestMove type check default false");
                println!("option name BookDepth type spin default 0 min 0 max 1000");
                println!("option name Clear Hash type button");
                crate::tune::print_uci_options();
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => self.new_game(),
            "position" => {
                self.stop();
                self.set_position(args);
            }
            "go" => self.go(args),
            "stop" => self.stop(),
            "ponderhit" => self.shared.pondering.0.store(false, Relaxed),
            "setoption" => self.set_option(args),
            "quit" => {
                self.stop();
                return false;
            }
            "d" => print!("{}", self.pos.pretty()),
            "eval" => self.eval(),
            "perft" => self.perft(args.first().and_then(|s| s.parse().ok()).unwrap_or(1)),
            "wac" => {
                let ms = args.first().and_then(|s| s.parse().ok()).unwrap_or(1000);
                self.run_suite("WAC", WAC_EPD, ms);
            }
            "epd" => match args.first().map(std::fs::read_to_string) {
                Some(Ok(text)) => {
                    let ms = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
                    self.run_suite(args[0], &text, ms);
                }
                _ => println!("info string usage: epd <file> [ms per position]"),
            },
            "bench" => self.bench(args.first().and_then(|s| s.parse().ok()).unwrap_or(13)),
            "smpbench" => {
                let depth = args.first().and_then(|s| s.parse().ok()).unwrap_or(12);
                let default_max = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
                let max_t = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(default_max);
                self.smp_bench(depth, max_t);
            }
            "evalbench" => self.eval_bench(args.first().and_then(|s| s.parse().ok()).unwrap_or(50_000)),
            "perftbench" => self.perft_bench(),
            "book" => self.show_book(),
            "tbprobe" => {
                let wdl = if self.pos.castling == 0 && self.pos.rule50 == 0 { tb::probe_wdl(&self.pos) } else { None };
                let root = tb::probe_root(&self.pos, &self.tb);
                println!(
                    "wdl {} root {}",
                    wdl.map_or("none".to_string(), |w| w.to_string()),
                    root.map_or("none".to_string(), |r| format!("{} wdl {} dtz {}", r.mv.to_uci(), r.wdl, r.dtz))
                );
            }
            "evalfile" => self.eval_file(args.first().copied().unwrap_or("")),
            "selftest" => self.selftest(args.first().and_then(|s| s.parse().ok()).unwrap_or(50)),
            _ => println!("info string unknown command: {cmd}"),
        }
        true
    }

    pub fn run(&mut self) {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if !self.command(&line) {
                return;
            }
        }
        // stdin closed: let a running search finish, then exit.
        self.wait();
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop();
        {
            let (lock, cv) = &*self.coord;
            let mut guard = lock.lock().unwrap();
            guard.cmd = Some(CoordCmd::Quit);
            cv.notify_all();
        }
        if let Some(h) = self.coord_thread.take() {
            let _ = h.join();
        }
    }
}
