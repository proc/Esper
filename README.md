<p align="center">
  <img src="esper.svg" alt="Esper logo" width="200" />
</p>

<h1 align="center">Esper</h1>

Esper is a UCI chess engine written in Rust. It evaluates positions with
Stockfish's `nn-bddd9d57eb87.nnue` network and has a Stockfish-style search with
Lazy SMP, Syzygy tablebase support, and NEON, AVX2 and AVX-512 code paths.

## Quick start

```bash
cargo build --release
```

```bash
./target/release/esper bench
```

The binary is `target/release/esper`. The network is embedded in it, so that one
file is all a GUI (Ponder) needs. Add it as a
UCI engine; standard chess only, Chess960 is not supported.

## Building

| Command | Result |
|---|---|
| `cargo build --release` | Optimised build, network embedded; picks its SIMD path at run time |
| `cargo build --release --no-default-features` | Network and opening book left out; the network is loaded from `nn-bddd9d57eb87.nnue` in the working directory or next to the binary |
| `cargo build --release --no-default-features --features embed` | Network embedded, opening book left out |
| `cargo build --release --features portable` | Hand-written SIMD left unused (for checking the SIMD paths) |
| `cargo build --release --features tune` | Search parameters exposed as `T_*` UCI options for SPSA |
| `tools/pgo.sh` (`tools\pgo.ps1` on Windows) | Profile-guided build (`target/release/esper-pgo`; with `--install` also copied to `target/release/esper`): instruments, runs the bench, rebuilds with the profile. Needs an `llvm-profdata` matching rustc's LLVM (rustup's `llvm-tools` or Homebrew's `llvm`) |

Building needs Rust (2021 edition) and a C compiler, since `build.rs` compiles the
vendored Fathom tablebase prober. The release profile uses fat LTO, one codegen
unit and `panic = "abort"`.

The build does not target the build machine's CPU: the binary detects what the
CPU it runs on supports and uses the best SIMD path for it (see below), so one
binary can be copied to other machines. On x86-64, `.cargo/config.toml` sets
`-C target-cpu=x86-64-v2` (SSE4.2 and POPCNT, every CPU since about 2009), which
the board code needs for a fast population count; a CPU older than that gets an
error message at startup. Building with `RUSTFLAGS="-C target-cpu=native"` still
works and is a little faster on x86-64 (the board code can then use BMI2), at
the cost of a binary tied to that CPU.

### SIMD

Every path for the architecture is compiled in; `src/simd.rs` picks one when the
engine starts and reports it as `info string SIMD path: …` in reply to `uci`:

| CPU | Path |
|---|---|
| aarch64 with the dot-product instructions (Apple silicon, Graviton 2+…) | `neon-dot` |
| other aarch64 | `neon` |
| x86-64 with AVX-512F, BW and VNNI | `avx512-vnni` (compiles, but has not been run by the author) |
| x86-64 with AVX-512F and BW | `avx512` (compiles, but has not been run by the author) |
| x86-64 with AVX2 | `avx2` |
| anything else | `scalar`: portable loops |

All paths produce identical evaluations. Setting the environment variable
`ESPER_SIMD` to one of those names selects a lower path than the detected one
(for testing); a path the CPU cannot run is never selected. A build whose target
already guarantees the best path (Apple silicon) has the choice fixed at compile
time and ignores the variable.

## Training your own network

`esper datagen` plays self-play games and writes the positions it meets, scored
by the search, with the game result. Two trainers can use them.

Games can also come from PGN files instead of self-play:

```bash
target/release/esper datagen pgn=games.pgn out=data.txt nodes=5000
```

replays every game that has a result and writes its positions with that result
(`pgn=` can be repeated or given a comma-separated list; the first `skip=8`
plies of each game are left out). By default Esper searches each position for
`nodes` nodes to score it. `score=pgn` instead uses the evaluations recorded in
the PGN's comments (Lichess's `[%eval 0.35]`, or an engine's `{+0.35/12 …}`) and
drops positions without one, which needs no searching. Output from PGN files and
from self-play can be appended to the same file.

A run can be stopped at any time. Each finished game is written at once, so
nothing but the games in progress is lost. A PGN run records how far it got in
`<out>.progress` and, run again with the same files and settings, carries on
with the next game instead of writing the earlier ones twice (delete that file
to start over). Self-play uses a new random seed each run, so a second run adds
different games; `seed=N` makes a run repeatable.

### A small network of your own

```bash
target/release/esper datagen out=data.txt games=200000 nodes=5000
python3 tools/train.py data.txt --out my.esprnet --engine target/release/esper
```

then `setoption name EvalFile value my.esprnet`. `tools/train.py` (PyTorch; uses
a CUDA or Apple GPU when present) trains the network `src/small.rs` reads: 768
inputs per perspective, one hidden layer (`--hidden`, default 256) with a squared
clipped ReLU, one output. With `--engine` it loads the exported file into that
Esper binary and checks that the engine's evaluation matches its own integer
arithmetic exactly on a sample of positions.

Training can be continued rather than started afresh. Each epoch the run saves a
checkpoint, `my.esprnet.ckpt`, and

```bash
python3 tools/train.py more.txt --out my.esprnet --resume
```

trains the same network further: for `--epochs` more on whatever data files are
given after a finished run, or from the epoch it was in after a run that was
stopped part-way. `--resume FILE` starts from another checkpoint or from a
network file. `python3 tools/train.py --help` has the details.

Expect far less strength than the Stockfish network: this is a much simpler
network, and it needs a lot of data (a hundred million positions is a sensible
start; `datagen` produces roughly a thousand a second on eight cores at 5000
nodes). Compare networks with
`tools/sprt.py --new-option EvalFile=my.esprnet`.

### The Stockfish architecture, on your data

```bash
target/release/esper datagen out=data.plain games=200000 nodes=5000 format=plain
```

writes Stockfish's "plain" training text (every position, with the move played).
Stockfish's tools convert that to `.binpack`, which the
[nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) trainer reads;
its serialised `.nnue` output loads with `EvalFile` as long as it has the
architecture of the embedded network. This route has not been run by the author:
the plain output is written to the documented layout but has not been through
the converter or the trainer.

## Command line

```bash
esper -t 8
```

`-t N` (or `--threads N`) starts the engine with N search threads, and becomes the
default the `Threads` option reports. A `setoption name Threads` from the GUI
overrides it. Any other arguments are run as a single command, after which the
engine exits:

```bash
esper -t 4 bench
```

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 16 | Transposition table size in MB |
| `Threads` | 1 (or `-t N`) | Search threads (Lazy SMP) |
| `MultiPV` | 1 | Number of principal variations reported |
| `Move Overhead` | 25 | Milliseconds held back per move for GUI and network lag |
| `Auto Move Overhead` | true | Measure what each move really costs on the clock (from the times the GUI or server reports) and use that when it is more than `Move Overhead` |
| `Ponder` | false | Pondering is supported through `go ponder` and `ponderhit` |
| `UCI_ShowWDL` | false | Adds `wdl` win/draw/loss figures to `info` lines |
| `EvalFile` | `<embedded>` | Path to another network with the same architecture |
| `SyzygyPath` | `<empty>` | Syzygy table directories, separated by `:` (`;` on Windows) |
| `SyzygyProbeDepth` | 1 | Minimum remaining depth for probing inside the search |
| `Syzygy50MoveRule` | true | Treat wins and losses that the 50-move rule would draw as draws |
| `SyzygyProbeLimit` | 7 | Only probe positions with at most this many pieces |
| `OwnBook` | false | Play opening moves from the book |
| `BookFile` | `<embedded>` | The built-in book, or the path of any Polyglot `.bin` book |
| `BookBestMove` | false | Always play the most heavily weighted book move instead of choosing at random by weight |
| `BookDepth` | 0 | Last move number taken from the book; 0 means no limit |
| `Clear Hash` | | Clears the hash table and the history tables |

`go` understands `wtime`, `btime`, `winc`, `binc`, `movestogo`, `depth`, `nodes`,
`movetime`, `infinite`, `ponder` and `searchmoves`.

## Extra commands

These are typed at the engine prompt (or given on the command line).

| Command | What it does |
|---|---|
| `d` | Prints the board, FEN and key |
| `eval` | Prints the network's evaluation of the current position |
| `perft N` | Counts leaf nodes to depth N (`go perft N` works too) |
| `perftbench` | Runs the standard perft positions, checks the counts and reports speed |
| `bench [depth]` | Searches a fixed set of positions (default depth 13) and prints nodes and NPS |
| `smpbench [depth] [threads]` | Times the same search at 1, 2, 4… threads and reports the speed-up |
| `evalbench [n]` | Measures NNUE evaluation throughput |
| `wac [ms]` | Runs the 300 "Win at Chess" positions (built in from `suites/wac.epd`) at `ms` per position, default 1000, and reports the number solved |
| `epd <file> [ms]` | The same for any EPD file with `bm` or `am` operations |
| `book` | Lists the book moves for the current position with their weights |
| `tbprobe` | Shows the tablebase WDL and root DTZ result for the current position |
| `datagen out=FILE [games=N] [nodes=N] [threads=N] [seed=N] [format=text\|plain] [pgn=FILE] [score=search\|pgn] [skip=N]` | Plays self-play games and appends training positions to FILE (see Training your own network) |
| `selftest [games]` | Plays random games and checks incremental NNUE updates against full refreshes |
| `evalfile <fens.txt>` | Prints the raw network output for each FEN in a file |

## How it works

### Board and move generation

Bitboards with hard-coded magic numbers for sliding pieces (checked at start-up),
copy-make positions, and pseudo-legal generation that produces only evasions when
in check. Zobrist keys are kept for the whole position, the pawns, each side's
non-pawn pieces and the minor pieces, the last three feeding the correction
histories. Perft matches the published counts for the standard test positions.

### Evaluation

The network (file version `0x6A448AFA`) has a 1024-wide feature transformer with
int8 weights fed by three feature sets:

- king-bucketed piece-squares in three material blocks (K32H3),
- piece-attacks-piece threats (FullThreats, 59,808 features),
- pawn pairs (PP_3Wide).

A 1024 → 32 → 32 → 1 head follows, one of up to 192, chosen by material. Esper
reproduces Stockfish's raw output for this network exactly.

Accumulators are updated lazily and incrementally. Each ply records which feature
rows change, found by comparing the position with its parent, so making a move
does no NNUE work, and prefetches those rows. The arithmetic runs only if the node
is actually evaluated. Piece-square sums are refreshed through a per-king-square
cache; threat sums are refreshed only when a king crosses the centre files.

The raw score is then scaled as Stockfish does: blended with the correction
histories, adjusted for optimism and material, and damped as the 50-move counter
rises.

### Search

Iterative deepening with aspiration windows and principal variation search. The
search was compared with Stockfish's current source line by line and follows it
closely, down to its constants; the remaining differences are deliberate (killer
and countermove ordering, and the switches below).

- **Move ordering**: a staged move picker (hash move, good captures, quiets,
  bad captures) with butterfly, low-ply, capture, continuation (1, 2, 3, 4 and 6
  plies back) and pawn histories, updated where Stockfish updates them.
- **Pruning**: reverse futility pruning, razoring, null-move pruning with
  verification, ProbCut and its hash-entry variant, futility, history and SEE
  pruning of quiet moves and captures, late move pruning, and quiescence search
  with futility and SEE pruning.
- **Reductions and extensions**: Stockfish's late move reduction formula (history,
  PV, cut-node and correction terms, extending by up to two plies near the root),
  hindsight depth adjustment from the parent's reduction and the change in
  evaluation, internal iterative reductions, and singular extensions of up to
  three plies with negative extensions.
- **Repetitions**: cuckoo tables detect when the side to move can repeat a position
  with its next move, so a draw is known to be available before it is played.
- **Correction histories** keyed by pawns, non-pawns, minor pieces and the
  continuation, which adjust the static evaluation from search results.
- **Transposition table**: lock-free, five entries per 64-byte bucket with 32-bit checks, shared by
  all threads, with Stockfish's replacement and aging rules.
- **Root**: each root move keeps effort-weighted averages of its scores, which
  centre the aspiration window and set the evaluation's optimism.

`src/search.rs` begins with a block of constants that switch individual features
on or off (for example `MULTI_CUT`, `SF_NMP`, `SF_PARITY`, `SHARED_HISTORY`), so each one can be
measured by playing a build with it against a build without.

### Lazy SMP

Helper threads search the same tree from a persistent thread pool, sharing the
transposition table and, as Stockfish does, the continuation, pawn and correction
histories. When the search ends, the reported move comes from a
vote among the threads, weighted by depth and score.

### Time management

Following Stockfish, each move gets an optimum and a maximum time from the clock,
increment, `movestogo`, the game ply and the opponent's clock (less time when
behind), plus a quarter more when `Ponder` is on. The search stops early or runs
longer depending on how stable the best move is, whether the score is falling, and
what share of the nodes went to the best move; past half the planned time it stops
deepening. Some of this state carries over from move to move. Esper adds one
safeguard: under a second on the clock it never plans to use more than half of it.

### Opening book

The book is off by default; `setoption name OwnBook value true` turns it on.
While the position is in the book, `go` answers at once with a book move (and a
book reply as the ponder move) instead of searching. `go infinite`, `go ponder`
and `go searchmoves` always search. Moves are drawn at random in proportion to
their weights unless `BookBestMove` is set.

The embedded book (`books/esper.bin`, 1.70 million entries, 27 MB) is built from
the first 40 plies of all 2,483,177 games in the CCRL 40/40 database
(`CCRL-4040-commented.[2483177].pgn`). A move is included if engines played it in
at least 3 games, it scored at least 20% for the side playing it, and it scored
within 30 points per 100 games of the best such move in its position. Only
positions reachable from the start through book moves are kept. A move's weight
is its game count times the square of its score, so popular, successful moves are
chosen most often. The book was made with `pgn2book` from the devilfish project:

```bash
pgn2book -in CCRL-4040.pgn -plies 40 -min-games 3 -min-share 0 -max-loss 0.30 -min-score 0.20 -out books/esper.bin
```

Any Polyglot book can be used instead through `BookFile`. Books are looked up in
place (a binary search over the raw entries), so the embedded one costs no
loading time or extra memory.

### Tablebases

With `SyzygyPath` set, a won or lost root position within the tables is played
straight from the distance-to-zero tables, and a drawn one is searched over only
the moves that keep the draw. Inside the search, positions within the tables (no
castling rights, half-move clock at zero) are resolved by a win/draw/loss probe.
Tablebase wins score just below mates and are reported as `cp 20000` minus the
distance. `info` lines include `tbhits`.

## Source layout

| Path | Contents |
|---|---|
| `src/main.rs` | Start-up and command-line arguments |
| `src/bitboard.rs` | Attack tables and magic bitboards |
| `src/position.rs` | Board state, copy-make, legality, SEE, FEN and Zobrist keys |
| `src/movegen.rs` | Move generation, SAN, perft |
| `src/nnue.rs` | Network loading and inference, with SIMD paths |
| `src/small.rs` | The small trainable network: loading and inference |
| `src/datagen.rs` | Training data generation from self-play or PGN files |
| `src/pgn.rs` | PGN reading for data generation |
| `tools/train.py` | Trainer for the small network |
| `src/tt.rs` | Transposition table |
| `src/search.rs` | Search, move ordering, histories, time management, Lazy SMP |
| `src/tb.rs` | Syzygy probing (bindings to Fathom) |
| `src/book.rs`, `src/polyglot_keys.rs` | Polyglot opening book |
| `books/esper.bin` | The embedded opening book |
| `src/tune.rs` | Tunable search parameters |
| `src/uci.rs` | UCI front end and the extra commands |
| `suites/wac.epd` | Win at Chess test suite |
| `tools/` | SPRT and SPSA scripts |
| `esper.svg`, `esper.ico`, `esper.rc` | Logo, and the Windows icon made from it (built into `esper.exe` by `build.rs`) |
| `third_party/fathom/` | Vendored Fathom tablebase prober (C, MIT) |

## Testing and tuning

`tools/sprt.py` plays one build against another using paired openings and a
pentanomial sequential probability ratio test, and stops once the change is shown
better or worse:

```bash
python3 -I tools/sprt.py --new ./esper-new --base ./esper --tc 8+0.08 --concurrency 4 --elo0 0 --elo1 5
```

`tools/spsa.py` tunes the parameters defined in `src/tune.rs`. They appear as UCI
options (names starting with `T_`) only in builds made with `--features tune`.
Both scripts need python-chess and are adapted from the devilfish project.

## License

GPL-3.0-or-later. The evaluation code implements the network format defined by
[Stockfish](https://github.com/official-stockfish/Stockfish) (GPL-3.0), and the
network file comes from the Stockfish project. `third_party/fathom` is MIT
licensed (see the headers of its files).
