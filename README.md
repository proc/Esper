# Esper

A UCI chess engine written in Rust, evaluating positions with Stockfish's
`nn-bddd9d57eb87.nnue` network.

## Build

```bash
cargo build --release
```

The binary is `target/release/esper`. The network is embedded in the executable
by default, so the single file is all a GUI needs. `.cargo/config.toml` builds for
the host CPU (`target-cpu=native`); remove that for a portable binary.

- `--no-default-features` leaves the network out; the engine then loads
  `nn-bddd9d57eb87.nnue` from the working directory or from next to the executable.
- `--features portable` disables the hand-written SIMD paths (testing aid).

SIMD is chosen at compile time from the enabled CPU features (`build.rs`):

| Target | Path |
|---|---|
| aarch64 | NEON, plus dot-product instructions where available |
| x86-64 with AVX-512F+BW | AVX-512, using VNNI if enabled (compiled but never run by the author) |
| x86-64 with AVX2 | AVX2 |
| anything else | portable loops |

On x86 that means building with `-C target-cpu=native` (the default here) or an
explicit level such as `x86-64-v3` / `x86-64-v4`; a plain baseline x86-64 build
gets the portable path.

## Command line

`esper -t N` (or `--threads N`) starts with N search threads; a `Threads` option
sent over UCI overrides it. Any other arguments are run as one command, e.g.
`esper -t 4 bench`.

## UCI options

| Option | Default | |
|---|---|---|
| `Hash` | 16 | Transposition table size in MB |
| `Threads` | 1 | Search threads (lazy SMP) |
| `MultiPV` | 1 | Number of principal variations reported |
| `Move Overhead` | 25 | Milliseconds reserved per move for GUI / network lag |
| `Ponder` | false | Pondering is supported (`go ponder` / `ponderhit`) |
| `EvalFile` | `<embedded>` | Path of another network with the same architecture |
| `SyzygyPath` | `<empty>` | Directories with Syzygy tables, separated by `:` (`;` on Windows) |
| `SyzygyProbeDepth` | 1 | Minimum remaining depth for probing inside the search |
| `Syzygy50MoveRule` | true | Count wins and losses that the 50-move rule draws as draws |
| `SyzygyProbeLimit` | 7 | Only probe positions with at most this many pieces |
| `Clear Hash` | | Clears the hash table and history tables |

Extra commands: `d` (show board), `eval`, `perft N` / `go perft N`, `bench [depth]`,
`selftest [games]` (checks incremental NNUE updates against full refreshes) and
`evalfile <fens.txt>` (raw network output per FEN). `wac [ms]` runs the 300-position
"Win at Chess" tactical suite (`suites/wac.epd`, built into the binary) at `ms`
milliseconds per position (default 1000) and reports how many were solved;
`epd <file> [ms]` does the same for any EPD file with `bm` / `am` operations.
Standard chess only; Chess960 castling is not supported.

## Layout

| File | |
|---|---|
| `src/bitboard.rs` | Attack tables, magic bitboards |
| `src/position.rs` | Board state, copy-make, legality, SEE, FEN, Zobrist keys |
| `src/movegen.rs` | Staged pseudo-legal generation, perft |
| `src/nnue.rs` | Network loading and inference |
| `src/tt.rs` | Lock-free transposition table |
| `src/search.rs` | Iterative deepening PVS, move ordering, pruning, time management |
| `src/tb.rs` | Syzygy tablebase probing (bindings to Fathom) |
| `src/uci.rs` | Protocol front end |
| `third_party/fathom/` | Vendored Fathom probing library (C, MIT), built by `build.rs` |

### Evaluation

The network (file version `0x6A448AFA`) has a 1024-wide feature transformer with
int8 weights fed by three feature sets: king-bucketed piece-squares in three
material blocks (K32H3), piece-attacks-piece threats (FullThreats) and pawn pairs
(PP_3Wide). It is followed by a 1024 → 32 → 32 → 1 head, one of up to 192 chosen
by material configuration. Esper reproduces Stockfish's raw output for it exactly.

Accumulators are updated lazily and incrementally. Each ply registers which
feature rows change (found by diffing the position with its parent, so make-move
carries no NNUE bookkeeping) and prefetches them; the arithmetic only happens if
that node is actually evaluated. Piece-square sums are refreshed through a
per-king-square cache, threat sums only when a king crosses the centre files.
The row updates, the feature transform, the sparse first layer and the small
dense layers have NEON, AVX2 and AVX-512 implementations; all produce identical
output.

### Search

Aspiration windows, principal variation search, transposition table cutoffs,
null-move pruning with verification, reverse futility pruning, razoring, ProbCut,
internal iterative reductions, late move reductions and pruning, futility, history
and SEE pruning, singular extensions with multi-cut, quiescence search, and
killer / butterfly / capture / continuation histories plus pawn and non-pawn
correction histories.

### Tablebases

With `SyzygyPath` set, a won or lost root position within the tables is played
directly from the distance-to-zero tables, and a drawn one is searched over the
moves that hold the draw. Inside the search, positions within the tables (no
castling rights, half-move clock zero) are resolved by a win/draw/loss probe;
tablebase wins score just below mates and are reported as `cp 20000` minus the
distance. `info` lines carry `tbhits`. Building needs a C compiler for Fathom.

## Testing and tuning

`tools/sprt.py` plays one build against another with a sequential test (paired
openings, pentanomial SPRT) and stops as soon as the change is shown better or
worse:

```bash
python3 -I tools/sprt.py --new ./esper-new --base ./esper --tc 8+0.08 --concurrency 4 --elo0 0 --elo1 5
```

`tools/spsa.py` tunes the search parameters in `src/tune.rs`. They are listed
as UCI options (names starting with `T_`) only in builds with
`cargo build --release --features tune`. Both scripts need python-chess and
are adapted from the devilfish project.

## License

GPL-3.0-or-later. The evaluation code implements the network format defined by
[Stockfish](https://github.com/official-stockfish/Stockfish) (GPL-3.0), and the
network file itself comes from the Stockfish project. `third_party/fathom` is
MIT-licensed (see the headers of its files).
