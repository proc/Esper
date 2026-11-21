//! Self-play training data for a new evaluation network.
//!
//!     esper datagen out=data.txt games=100000 nodes=5000 threads=8 format=text
//!
//! Each thread plays games against itself from short random openings, searching
//! a fixed number of nodes per move, and appends the positions it met to the
//! output file in one of two formats. Every run plays different games unless
//! `seed=N` is given, so stopping a run and starting another just adds more.
//!
//! `format=text` (the default; read by `tools/train.py` and by the bullet
//! trainer's tools), one quiet position per line:
//!
//!     <FEN> | <score> | <result>
//!
//! `score` is the search score in centipawns and `result` the game's outcome
//! (1.0, 0.5 or 0.0), both from White's point of view. A position is kept when
//! the side to move is not in check, the move played is not a capture or
//! promotion, and the score is not a mate score: a network learns static
//! evaluation from positions where the static value means something.
//!
//! `format=plain` (Stockfish's "plain" training text, which its tools convert to
//! the binpack files the nnue-pytorch trainer reads), every position of every
//! game in order, six lines each:
//!
//!     fen <FEN>
//!     move <move played, UCI>
//!     score <search score, internal units, side to move's view>
//!     ply <game ply>
//!     result <1, 0 or -1, side to move's view>
//!     e
//!
//! Nothing is filtered there: that trainer does its own position skipping.
//!
//! With `pgn=FILE` (repeatable, or comma separated) the games come from PGN
//! files instead of self-play: each game with a result is replayed and its
//! positions written with that result. `score=search` (the default) searches
//! every position for `nodes` nodes, as self-play does; `score=pgn` takes the
//! evaluations recorded in the PGN's comments (`[%eval 0.35]`, or an engine's
//! `{+0.35/12 ...}`) and drops positions without one, which needs no searching.
//! The first `skip` plies of each game (default 8) are left out: opening
//! positions repeat across games and their scores say little.
//!
//! A PGN run records how far it has got in `<out>.progress`. Run again with the
//! same files and settings, it carries on with the next game (or does nothing
//! if it had finished) rather than writing the earlier games a second time.

use crate::movegen::legal_moves;
use crate::nnue::Network;
use crate::position::*;
use crate::search::{Limits, Shared, ThreadPool, MATE_BOUND, NORMALIZE_PAWN};
use crate::tb::TbConfig;
use crate::tt::TT;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct Config {
    pub out: String,
    pub games: u64,
    pub nodes: u64,
    pub threads: usize,
    pub seed: u64,
    /// Write Stockfish's "plain" format instead of one line per quiet position.
    pub plain: bool,
    /// Read games from these PGN files instead of playing them.
    pub pgn: Vec<String>,
    /// Score PGN positions from the evaluations in its comments, not by searching.
    pub pgn_scores: bool,
    /// Plies left out at the start of each PGN game.
    pub skip: usize,
}

impl Config {
    /// Read `key=value` arguments; unknown keys and bad values are errors.
    pub fn parse(args: &[&str]) -> Result<Config, String> {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let mut c = Config {
            out: String::new(),
            games: 1000,
            nodes: 5000,
            threads,
            // A different seed every run unless one is given: a second self-play
            // run with the same seed would play the same games again.
            seed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |d| d.as_nanos() as u64 | 1),
            plain: false,
            pgn: Vec::new(),
            pgn_scores: false,
            skip: 8,
        };
        let mut games_given = false;
        for a in args {
            let (k, v) = a.split_once('=').ok_or_else(|| format!("expected key=value, got {a}"))?;
            let num = || v.parse::<u64>().map_err(|_| format!("{k} needs a number, got {v}"));
            match k {
                "out" => c.out = v.to_string(),
                "games" => {
                    c.games = num()?;
                    games_given = true;
                }
                "pgn" => c.pgn.extend(v.split(',').filter(|p| !p.is_empty()).map(str::to_string)),
                "score" => match v {
                    "search" => c.pgn_scores = false,
                    "pgn" => c.pgn_scores = true,
                    _ => return Err(format!("score is search or pgn, got {v}")),
                },
                "skip" => c.skip = num()? as usize,
                "nodes" => c.nodes = num()?.max(1),
                "threads" => c.threads = (num()? as usize).clamp(1, 512),
                "seed" => c.seed = num()?,
                "format" => match v {
                    "text" => c.plain = false,
                    "plain" => c.plain = true,
                    _ => return Err(format!("format is text or plain, got {v}")),
                },
                _ => return Err(format!("unknown argument {k}")),
            }
        }
        if c.out.is_empty() {
            return Err("usage: datagen out=FILE [games=N] [nodes=N] [threads=N] [seed=N] [format=text|plain] \
                        [pgn=FILE[,FILE...]] [score=search|pgn] [skip=N]"
                .to_string());
        }
        // Every game of the PGN files unless a limit was asked for.
        if !c.pgn.is_empty() && !games_given {
            c.games = u64::MAX / 4;
        }
        Ok(c)
    }
}

/// Random plies played before the engine takes over (one more in half the games,
/// so both colours get the first engine move).
const OPENING_PLIES: u64 = 8;
/// An opening the first search scores beyond this (centipawns) is thrown away.
const MAX_OPENING_CP: i32 = 400;
/// A side this far ahead (centipawns) for `WIN_PLIES` plies running wins.
const WIN_CP: i32 = 1500;
const WIN_PLIES: u32 = 4;
/// From ply `DRAW_FROM`, a score this close to zero for `DRAW_PLIES` plies is a draw.
const DRAW_CP: i32 = 8;
const DRAW_PLIES: u32 = 12;
const DRAW_FROM: usize = 80;
const MAX_PLIES: usize = 500;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// One searched position of a game.
struct Record {
    fen: String,
    white_to_move: bool,
    /// The move played.
    mv: Move,
    /// Search score for the side to move: centipawns (`None` for a mate or
    /// tablebase score) and internal units.
    cp: Option<i32>,
    raw: i32,
    /// Game ply (0 at the starting position).
    ply: usize,
    /// Not in check and the move played is neither a capture nor a promotion.
    quiet: bool,
}

impl Record {
    /// Append this position in the chosen format; `white_result` is the game's
    /// outcome for White (1.0, 0.5, 0.0).
    fn write(&self, plain: bool, white_result: f32, text: &mut String) {
        use std::fmt::Write;
        if plain {
            let stm_result = if self.white_to_move { white_result } else { 1.0 - white_result };
            let result = (stm_result * 2.0) as i32 - 1;
            let score = self.raw.clamp(-32000, 32000);
            let _ = write!(
                text,
                "fen {}\nmove {}\nscore {score}\nply {}\nresult {result}\ne\n",
                self.fen,
                self.mv.to_uci(),
                self.ply
            );
        } else if let (Some(cp), true) = (self.cp, self.quiet) {
            let white_cp = if self.white_to_move { cp } else { -cp };
            let _ = writeln!(text, "{} | {white_cp} | {white_result:.1}", self.fen);
        }
    }
}

struct Worker {
    pool: ThreadPool,
    tt: Arc<TT>,
    shared: Arc<Shared>,
    rng: Rng,
    nodes: u64,
}

impl Worker {
    /// Search `pos`; returns the best move and the score in centipawns for the
    /// side to move (`None` for a mate or tablebase score).
    fn search(&mut self, pos: &Position, keys: &[u64]) -> (Move, Option<i32>, i32) {
        self.shared.stop.0.store(false, Relaxed);
        self.shared.pondering.0.store(false, Relaxed);
        let limits = Limits { nodes: Some(self.nodes), ..Limits::default() };
        let (best, _, _) =
            self.pool.run(pos, keys, &limits, 1, 0, TbConfig::DEFAULT, true, false, 0, &self.shared.clone());
        let s = self.pool.main.best_score;
        let cp = (s.abs() < MATE_BOUND).then_some(s * 100 / NORMALIZE_PAWN);
        (best, cp, s)
    }

    /// The positions of a game read from a PGN file, scored by searching or, with
    /// `from_pgn`, from the evaluations in its comments. In the text format only
    /// positions that will be written are searched.
    fn replay(&mut self, game: &crate::pgn::Game, from_pgn: bool, skip: usize, plain: bool) -> Vec<Record> {
        use crate::pgn::Eval;
        self.tt.clear();
        self.pool.clear();
        let mut pos = game.start;
        let mut keys = vec![pos.key];
        let mut kept = Vec::new();
        for (i, &(mv, eval)) in game.moves.iter().enumerate() {
            let quiet = !pos.in_check() && !mv.is_noisy();
            if i >= skip && (plain || quiet) {
                // The side to move's score in centipawns, and in internal units.
                let score = if from_pgn {
                    let before = match eval {
                        Some(Eval::BeforeMover(cp)) => Some(cp),
                        _ => None,
                    };
                    // A Lichess evaluation belongs to the position after the previous move.
                    let after_previous = match i.checked_sub(1).and_then(|j| game.moves[j].1) {
                        Some(Eval::AfterWhite(cp)) => Some(if pos.stm == WHITE { cp } else { -cp }),
                        _ => None,
                    };
                    before.or(after_previous).map(|cp| (Some(cp), cp * NORMALIZE_PAWN / 100))
                } else {
                    let (_, cp, raw) = self.search(&pos, &keys);
                    Some((cp, raw))
                };
                if let Some((cp, raw)) = score {
                    kept.push(Record {
                        fen: pos.to_fen(),
                        white_to_move: pos.stm == WHITE,
                        mv,
                        cp,
                        raw,
                        ply: i,
                        quiet,
                    });
                }
            }
            pos = pos.make(mv);
            keys.push(pos.key);
        }
        kept
    }

    /// Play one game; its searched positions and White's result, or `None` when
    /// the random opening was unusable.
    fn play(&mut self) -> Option<(Vec<Record>, f32)> {
        self.tt.clear();
        self.pool.clear();
        let mut pos = Position::startpos();
        let mut keys = vec![pos.key];
        for _ in 0..OPENING_PLIES + (self.rng.next() & 1) {
            let moves = legal_moves(&pos);
            if moves.is_empty() {
                return None;
            }
            pos = pos.make(moves[(self.rng.next() % moves.len() as u64) as usize]);
            keys.push(pos.key);
        }
        let mut kept = Vec::new();
        let (mut win_run, mut draw_run) = (0u32, 0u32);
        let mut first = true;
        // White's result once the game is decided.
        let white = |pos: &Position, stm_result: f32| if pos.stm == WHITE { stm_result } else { 1.0 - stm_result };
        loop {
            if legal_moves(&pos).is_empty() {
                let result = if pos.in_check() { white(&pos, 0.0) } else { 0.5 };
                return Some((kept, result));
            }
            let repeats = keys.iter().filter(|&&k| k == pos.key).count();
            if pos.rule50 >= 100 || pos.insufficient_material() || repeats >= 3 || keys.len() > MAX_PLIES {
                return Some((kept, 0.5));
            }
            let (best, cp, raw) = self.search(&pos, &keys);
            if best.is_none() {
                return Some((kept, 0.5));
            }
            if first {
                first = false;
                if cp.map_or(true, |c| c.abs() > MAX_OPENING_CP) {
                    return None;
                }
            }
            // A mate score counts as decisive for whoever it favours.
            let decisive = cp.map_or(true, |c| c.abs() >= WIN_CP);
            win_run = if decisive { win_run + 1 } else { 0 };
            if win_run >= WIN_PLIES {
                return Some((kept, white(&pos, if raw > 0 { 1.0 } else { 0.0 })));
            }
            let level = cp.is_some_and(|c| c.abs() <= DRAW_CP);
            draw_run = if level && keys.len() >= DRAW_FROM { draw_run + 1 } else { 0 };
            if draw_run >= DRAW_PLIES {
                return Some((kept, 0.5));
            }
            kept.push(Record {
                fen: pos.to_fen(),
                white_to_move: pos.stm == WHITE,
                mv: best,
                cp,
                raw,
                ply: keys.len() - 1,
                quiet: !pos.in_check() && !best.is_noisy(),
            });
            pos = pos.make(best);
            keys.push(pos.key);
        }
    }
}

/// Where a PGN run has got to, kept next to the output as `<out>.progress` so
/// that a run stopped part-way resumes with the next game instead of writing
/// the earlier ones again. One line per run: the number of games dealt with, a
/// tab, and the run's key (its PGN files, their sizes and the scoring settings).
struct Progress {
    path: String,
    key: String,
    /// Lines belonging to other runs into the same output file.
    others: Vec<String>,
}

impl Progress {
    /// The progress record for this run and the number of games already done.
    fn open(cfg: &Config) -> (Progress, u64) {
        let sizes: Vec<u64> = cfg.pgn.iter().map(|p| std::fs::metadata(p).map_or(0, |m| m.len())).collect();
        let key = format!(
            "pgn={:?} sizes={:?} score={} nodes={} skip={} format={}",
            cfg.pgn,
            sizes,
            if cfg.pgn_scores { "pgn" } else { "search" },
            if cfg.pgn_scores { 0 } else { cfg.nodes },
            cfg.skip,
            if cfg.plain { "plain" } else { "text" }
        );
        let path = format!("{}.progress", cfg.out);
        let mut done = 0;
        let mut others = Vec::new();
        for line in std::fs::read_to_string(&path).unwrap_or_default().lines() {
            match line.split_once('\t') {
                Some((n, k)) if k == key => done = n.parse().unwrap_or(0),
                _ => others.push(line.to_string()),
            }
        }
        (Progress { path, key, others }, done)
    }

    /// Record that the first `done` games are dealt with. Written to a temporary
    /// file and renamed, so a stop mid-write cannot leave a broken record.
    fn save(&self, done: u64) -> std::io::Result<()> {
        let mut text = String::new();
        for o in &self.others {
            text.push_str(o);
            text.push('\n');
        }
        text.push_str(&format!("{done}\t{}\n", self.key));
        let tmp = format!("{}.tmp", self.path);
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &self.path)
    }
}

/// The output file. Each game is written with one call and nothing is buffered
/// here, so stopping the run loses at most the games still being worked on and
/// never leaves half a game behind in a buffer.
struct Output {
    file: std::fs::File,
    /// PGN runs write games in file order so progress is a single number.
    progress: Option<Progress>,
    next: u64,
    pending: std::collections::BTreeMap<u64, String>,
}

impl Output {
    /// Write a finished game: at once for self-play, in file order for a PGN
    /// game (`index` is its position in the PGN files).
    fn submit(&mut self, index: Option<u64>, text: String) -> std::io::Result<()> {
        let (Some(index), Some(progress)) = (index, &self.progress) else {
            return self.file.write_all(text.as_bytes());
        };
        self.pending.insert(index, text);
        let mut ready = String::new();
        let mut next = self.next;
        while let Some(t) = self.pending.remove(&next) {
            ready.push_str(&t);
            next += 1;
        }
        if next != self.next {
            // Progress first: a stop between the two steps then drops a game
            // rather than writing it twice.
            progress.save(next)?;
            self.file.write_all(ready.as_bytes())?;
            self.next = next;
        }
        Ok(())
    }
}

/// Play `cfg.games` games (or replay the PGN files' games) on `cfg.threads`
/// threads, appending to `cfg.out`.
pub fn run(net: Arc<Network>, cfg: &Config) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};
    let cannot_write = |e: std::io::Error| format!("cannot write {}: {e}", cfg.out);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&cfg.out)
        .map_err(|e| format!("cannot open {}: {e}", cfg.out))?;
    // A run that was killed mid-write can leave a partial last line: end it, so
    // the first new position does not run into it.
    if file.seek(SeekFrom::End(0)).map_err(cannot_write)? > 0 {
        let mut last = [0u8; 1];
        file.seek(SeekFrom::End(-1)).map_err(cannot_write)?;
        file.read_exact(&mut last).map_err(cannot_write)?;
        if last[0] != b'\n' {
            file.write_all(b"\n").map_err(cannot_write)?;
        }
    }
    let (started, games, positions) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    // Games skipped (no result, unreadable) and games cut short by a move the
    // parser could not read.
    let (skipped, truncated) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut progress = None;
    let mut resumed = 0;
    // The PGN reader and the index of the next game it will hand out.
    let reader = if cfg.pgn.is_empty() {
        None
    } else {
        let mut r = crate::pgn::Reader::new(&cfg.pgn)?;
        let (p, done) = Progress::open(cfg);
        while resumed < done && r.next_game().is_some() {
            resumed += 1;
        }
        progress = Some(p);
        Some(Mutex::new((r, resumed)))
    };
    let out = Mutex::new(Output { file, progress, next: resumed, pending: Default::default() });
    let failed = Mutex::new(None::<String>);
    let start = Instant::now();
    if reader.is_some() {
        let how = if cfg.pgn_scores { "scores from the PGN".to_string() } else { format!("{} nodes per position", cfg.nodes) };
        println!("datagen: games from {}, {how}, {} thread(s), appending to {}", cfg.pgn.join(", "), cfg.threads, cfg.out);
        if resumed > 0 {
            println!(
                "datagen: resuming after the {resumed} games already done (delete {}.progress to start again from the first game)",
                cfg.out
            );
        }
    } else {
        println!(
            "datagen: {} games at {} nodes per move on {} thread(s), seed {}, appending to {}",
            cfg.games, cfg.nodes, cfg.threads, cfg.seed, cfg.out
        );
    }
    std::thread::scope(|scope| {
        for id in 0..cfg.threads {
            let (net, out, started, games, positions, failed) = (net.clone(), &out, &started, &games, &positions, &failed);
            let (reader, skipped, truncated) = (&reader, &skipped, &truncated);
            // The search recurses deeply; give it the room the engine's own threads get.
            let spawned = std::thread::Builder::new().stack_size(64 << 20).spawn_scoped(scope, move || {
                let tt = Arc::new(TT::new(16));
                let shared = Arc::new(Shared::new());
                let seed = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (id as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                let mut w = Worker {
                    pool: ThreadPool::new(1, net, tt.clone(), shared.clone()),
                    tt,
                    shared,
                    rng: Rng(seed | 1),
                    nodes: cfg.nodes,
                };
                while started.fetch_add(1, Relaxed) < cfg.games {
                    // The game's index in the PGN files, its positions and White's
                    // result; no positions for a PGN game that cannot be used.
                    let (index, kept, result, counted) = if let Some(reader) = reader {
                        let (index, text) = {
                            let mut r = reader.lock().unwrap();
                            let Some(text) = r.0.next_game() else { return };
                            r.1 += 1;
                            (r.1 - 1, text)
                        };
                        match crate::pgn::parse_game(&text) {
                            Some(game) => {
                                truncated.fetch_add(game.truncated as u64, Relaxed);
                                (Some(index), w.replay(&game, cfg.pgn_scores, cfg.skip, cfg.plain), game.result, true)
                            }
                            None => {
                                skipped.fetch_add(1, Relaxed);
                                (Some(index), Vec::new(), 0.5, false)
                            }
                        }
                    } else {
                        // An unusable opening does not use up one of the requested games.
                        let (kept, result) = loop {
                            if let Some(g) = w.play() {
                                break g;
                            }
                        };
                        (None, kept, result, true)
                    };
                    let mut text = String::with_capacity(kept.len() * 110);
                    let mut written = 0u64;
                    for r in &kept {
                        let at = text.len();
                        r.write(cfg.plain, result, &mut text);
                        written += (text.len() > at) as u64;
                    }
                    if let Err(e) = out.lock().unwrap().submit(index, text) {
                        *failed.lock().unwrap() = Some(format!("cannot write {}: {e}", cfg.out));
                        started.store(u64::MAX / 2, Relaxed);
                        return;
                    }
                    if !counted {
                        continue;
                    }
                    let p = positions.fetch_add(written, Relaxed) + written;
                    let g = games.fetch_add(1, Relaxed) + 1;
                    if g % 100 == 0 || g == cfg.games {
                        let secs = start.elapsed().as_secs_f64().max(1e-9);
                        println!("datagen: {g} games, {p} positions, {:.0} positions/s", p as f64 / secs);
                    }
                }
            });
            if let Err(e) = spawned {
                *failed.lock().unwrap() = Some(format!("cannot start a thread: {e}"));
            }
        }
    });
    if let Some(e) = failed.into_inner().unwrap() {
        return Err(e);
    }
    println!(
        "datagen: done, {} games and {} positions in {:.0} s",
        games.load(Relaxed),
        positions.load(Relaxed),
        start.elapsed().as_secs_f64()
    );
    if reader.is_some() {
        if resumed > 0 && games.load(Relaxed) + skipped.load(Relaxed) == 0 {
            println!("datagen: every game of these PGN files was already done; nothing was added");
        }
        if cfg.pgn_scores && positions.load(Relaxed) == 0 && resumed == 0 {
            println!("datagen: no evaluations found in the PGN comments; use score=search to have Esper score the positions");
        }
        println!(
            "datagen: {} games skipped (no result or unreadable), {} cut short at a move that could not be read",
            skipped.load(Relaxed),
            truncated.load(Relaxed)
        );
    }
    Ok(())
}
