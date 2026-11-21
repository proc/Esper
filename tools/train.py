#!/usr/bin/env python3
"""Train a small evaluation network for Esper on its own self-play data.

    target/release/esper datagen out=data.txt games=200000 nodes=5000
    python3 tools/train.py data.txt --out my.esprnet --engine target/release/esper

then, in the engine:   setoption name EvalFile value my.esprnet

The network is the one src/small.rs reads: 768 inputs per perspective (colour
relative to the perspective x piece type x square, the board flipped vertically
for Black) -> HIDDEN units with a squared clipped ReLU -> one output from both
perspectives, side to move first.

Data is the text format `esper datagen` writes, one position per line:

    <FEN> | <score in centipawns, White's view> | <result for White: 1.0, 0.5, 0.0>

The network learns a win probability: the target mixes the search score, squashed
by a sigmoid, with the game result (--lambda is the score's share). Its output
times --scale is the evaluation in centipawns.

The parsed positions are cached next to the first data file (<file>.cache.npz),
so later runs on the same data start at once. With --engine, the exported file is
loaded into that Esper binary and its evaluations are compared with this script's
own integer arithmetic on a sample of positions: they must agree exactly.

Training can be continued instead of started afresh. Every epoch the run saves a
checkpoint, <out>.ckpt (weights, optimiser state and how far the run has got):

    python3 tools/train.py more.txt --out my.esprnet --resume

* after a finished run, --resume trains for --epochs more, on whatever data files
  are given (the same, more, or different ones), on a new learning-rate schedule
  that starts at --lr, by default 0.3 times the rate the previous run started
  at, since refining a trained network wants a gentler rate than a first run;
* after a run that was stopped part-way (Ctrl-C, a crash), --resume carries on
  with the epoch it was in and the same learning-rate schedule, and --epochs and
  --lr are ignored until that run is complete;
* --resume FILE starts from another checkpoint, or from a network file
  (.esprnet) when the checkpoint is gone; a network file holds rounded weights
  and no optimiser state, so a checkpoint is the better starting point.

The network's width and --scale come from what is being continued.

Needs PyTorch and NumPy. Uses a CUDA or Apple GPU when there is one.
"""
import argparse
import multiprocessing
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

MAGIC = b"ESPRNET1"
INPUTS = 768
MAX_PIECES = 32
PAD = INPUTS  # index of the "no piece" column
QA, QB = 255, 64
INTERNAL_PER_100CP = 290  # search::NORMALIZE_PAWN
PIECES = {c: i for i, c in enumerate("PNBRQKpnbrqk")}


def parse_lines(lines):
    """Text lines -> (white-view feature indices, side to move, score, result)."""
    n = len(lines)
    feats = np.full((n, MAX_PIECES), PAD, dtype=np.uint16)
    stm = np.zeros(n, dtype=np.uint8)
    score = np.zeros(n, dtype=np.int16)
    result = np.zeros(n, dtype=np.uint8)
    keep = np.zeros(n, dtype=bool)
    for i, line in enumerate(lines):
        parts = line.split("|")
        if len(parts) != 3:
            continue
        fields = parts[0].split()
        if len(fields) < 2:
            continue
        try:
            sc = int(parts[1])
            res = round(float(parts[2]) * 2)
        except ValueError:
            continue
        sq, k, ok = 56, 0, True
        for ch in fields[0]:
            if ch == "/":
                sq -= 16
            elif ch.isdigit():
                sq += int(ch)
            else:
                p = PIECES.get(ch)
                if p is None or k >= MAX_PIECES or not 0 <= sq < 64:
                    ok = False
                    break
                # colour * 384 + piece type * 64 + square, from White's side
                feats[i, k] = (p // 6) * 384 + (p % 6) * 64 + sq
                k += 1
                sq += 1
        if not ok or res not in (0, 1, 2):
            continue
        stm[i] = 0 if fields[1] == "w" else 1
        score[i] = max(-32000, min(32000, sc))
        result[i] = res
        keep[i] = True
    return feats[keep], stm[keep], score[keep], result[keep]


def load(paths, workers):
    cache = paths[0] + ".cache.npz"
    stamp = [(os.path.abspath(p), os.path.getsize(p), int(os.path.getmtime(p))) for p in paths]
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if z["stamp"].tolist() == stamp:
            print(f"using cached positions from {cache}")
            return z["feats"], z["stm"], z["score"], z["result"]
    t0 = time.time()
    chunks, chunk = [], []
    for p in paths:
        with open(p) as f:
            for line in f:
                chunk.append(line)
                if len(chunk) == 200_000:
                    chunks.append(chunk)
                    chunk = []
    if chunk:
        chunks.append(chunk)
    if not chunks:
        sys.exit("no positions in the data files")
    with multiprocessing.Pool(workers) as pool:
        parts = pool.map(parse_lines, chunks)
    feats, stm, score, result = (np.concatenate([p[i] for p in parts]) for i in range(4))
    print(f"parsed {len(stm)} positions in {time.time() - t0:.0f} s")
    np.savez(cache, feats=feats, stm=stm, score=score, result=result, stamp=np.array(stamp, dtype=object))
    return feats, stm, score, result


def black_view(feats):
    """Feature indices from Black's side: colours swapped, board flipped vertically."""
    colour, rest = feats // 384, feats % 384
    flipped = (1 - colour) * 384 + (rest // 64) * 64 + ((rest % 64) ^ 56)
    return torch.where(feats == PAD, feats, flipped)


class Net(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.ft = torch.nn.Linear(INPUTS, hidden)
        self.out = torch.nn.Linear(2 * hidden, 1)

    def forward(self, feats, stm):
        """`feats`: white-view indices [B, 32]; `stm`: 0 white, 1 black."""
        b = feats.shape[0]

        def side(idx):
            x = torch.zeros(b, INPUTS + 1, device=feats.device)
            x.scatter_(1, idx, 1.0)
            return torch.clamp(self.ft(x[:, :INPUTS]), 0.0, 1.0) ** 2

        white, black = side(feats), side(black_view(feats))
        white_to_move = (stm == 0).unsqueeze(1)
        us = torch.where(white_to_move, white, black)
        them = torch.where(white_to_move, black, white)
        return self.out(torch.cat([us, them], dim=1)).squeeze(1)


def quantise(net):
    r = lambda t, q: np.clip(np.round(t.detach().cpu().numpy().astype(np.float64) * q), -32767, 32767)
    ft_w = r(net.ft.weight.t(), QA).astype(np.int16)  # [768, hidden]
    ft_b = r(net.ft.bias, QA).astype(np.int16)
    out_w = r(net.out.weight[0], QB).astype(np.int16)
    out_b = int(round(net.out.bias[0].item() * QA * QB))
    return ft_w, ft_b, out_w, out_b


def save(path, q, scale):
    ft_w, ft_b, out_w, out_b = q
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<4I", ft_b.shape[0], QA, QB, scale))
        f.write(ft_w.astype("<i2").tobytes())
        f.write(ft_b.astype("<i2").tobytes())
        f.write(out_w.astype("<i2").tobytes())
        f.write(struct.pack("<i", out_b))


def load_start(path):
    """What to continue from: a checkpoint, or a network file turned back into weights."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if head[:8] == MAGIC:
                hidden, qa, qb, scale = struct.unpack("<4I", head[8:24])
                i16 = lambda n: np.frombuffer(f.read(2 * n), dtype="<i2").astype(np.float32)
                ft_w, ft_b, out_w = i16(INPUTS * hidden), i16(hidden), i16(2 * hidden)
                (out_b,) = struct.unpack("<i", f.read(4))
                model = {
                    "ft.weight": torch.from_numpy(ft_w.reshape(INPUTS, hidden).T.copy() / qa),
                    "ft.bias": torch.from_numpy(ft_b / qa),
                    "out.weight": torch.from_numpy(out_w.reshape(1, 2 * hidden) / qb),
                    "out.bias": torch.tensor([out_b / (qa * qb)], dtype=torch.float32),
                }
                return {"model": model, "hidden": hidden, "scale": scale}
        return torch.load(path, map_location="cpu")
    except FileNotFoundError:
        sys.exit(f"nothing to resume from: {path} does not exist")
    except Exception as e:  # a truncated or foreign file
        sys.exit(f"cannot resume from {path}: {e}")


def trunc_div(a, b):
    """Integer division rounding towards zero, as Rust's `/` does."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def integer_eval(q, scale, feats, stm):
    """The engine's exact arithmetic (src/small.rs) for one position."""
    ft_w, ft_b, out_w, out_b = q
    h = ft_b.shape[0]
    idx = [int(i) for i in feats if i != PAD]
    flip = lambda i: (1 - i // 384) * 384 + ((i % 384) // 64) * 64 + ((i % 64) ^ 56)
    white = ft_b.astype(np.int64) + ft_w[idx].astype(np.int64).sum(axis=0)
    black = ft_b.astype(np.int64) + ft_w[[flip(i) for i in idx]].astype(np.int64).sum(axis=0)
    us, them = (white, black) if stm == 0 else (black, white)
    total = 0
    for acc, w in ((us, out_w[:h]), (them, out_w[h:])):
        v = np.clip(acc, 0, QA)
        total += int((v * v * w.astype(np.int64)).sum())
    t = trunc_div(total, QA) + out_b
    return max(-30000, min(30000, trunc_div(t * scale * INTERNAL_PER_100CP, QA * QB * 100)))


def fen_of(feats, stm):
    """A FEN for a parsed position (castling and counters do not affect the evaluation)."""
    board = [None] * 64
    for i in feats:
        if i != PAD:
            board[int(i) % 64] = "PNBRQKpnbrqk"[int(i) // 384 * 6 + (int(i) % 384) // 64]
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            c = board[rank * 8 + file]
            if c is None:
                empty += 1
            else:
                row += (str(empty) if empty else "") + c
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows) + (" w" if stm == 0 else " b") + " - - 0 1"


def verify(engine, net_path, q, scale, feats, stm, count):
    """Load the exported file into the engine and compare evaluations exactly."""
    pick = np.linspace(0, len(stm) - 1, num=min(count, len(stm)), dtype=np.int64)
    fens = [fen_of(feats[i], stm[i]) for i in pick]
    with tempfile.NamedTemporaryFile("w", suffix=".fens", delete=False) as f:
        f.write("\n".join(fens) + "\n")
        fens_path = f.name
    try:
        script = f"setoption name EvalFile value {os.path.abspath(net_path)}\nevalfile {fens_path}\nquit\n"
        out = subprocess.run([engine], input=script, capture_output=True, text=True, timeout=120).stdout
    finally:
        os.unlink(fens_path)
    if "could not load network" in out:
        sys.exit("verification FAILED: the engine could not load the exported network")
    got = [l.strip() for l in out.splitlines() if l.strip().lstrip("-").isdigit() or l.strip() == "none"]
    if len(got) != len(fens):
        sys.exit(f"verification FAILED: expected {len(fens)} evaluations from the engine, got {len(got)}")
    checked = bad = 0
    for i, g in zip(pick, got):
        if g == "none":  # side to move in check: the engine does not evaluate it
            continue
        checked += 1
        want = integer_eval(q, scale, feats[i], stm[i])
        if int(g) != want:
            bad += 1
            if bad <= 5:
                print(f"  mismatch: engine {g}, trainer {want}: {fen_of(feats[i], stm[i])}")
    if bad or not checked:
        sys.exit(f"verification FAILED: {bad} of {checked} evaluations differ")
    print(f"verified: the engine's evaluation matches on all {checked} sampled positions")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="text files written by `esper datagen`")
    ap.add_argument("--out", default="esper.esprnet", help="network file to write")
    ap.add_argument("--hidden", type=int, help="hidden units per perspective (default 256)")
    ap.add_argument("--epochs", type=int, help="passes over the data (default 20)")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, help="starting learning rate, decaying to 1%% of it (default 0.001, or 0.3 times the previous run's with --resume)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.7, help="share of the search score in the target")
    ap.add_argument("--scale", type=int, help="centipawns per unit of network output (default 400)")
    ap.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="FILE",
        help="continue from a checkpoint (default <out>.ckpt) or a network file instead of starting afresh",
    )
    ap.add_argument("--validation", type=float, default=0.02, help="fraction of positions held out")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes for parsing")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engine", help="Esper binary to verify the exported network with")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt_path = args.out + ".ckpt"
    start = load_start(args.resume or ckpt_path) if args.resume is not None else None
    for name, default in (("hidden", 256), ("scale", 400)):
        given = getattr(args, name)
        if start and given is not None and given != start[name]:
            sys.exit(f"--{name} {given} does not match what is being continued ({start[name]})")
        setattr(args, name, start[name] if start else given if given is not None else default)
    # A run that was stopped part-way is finished as it was planned.
    unfinished = bool(start) and start.get("epoch", 0) < start.get("epochs", 0)
    if unfinished:
        first, args.epochs, args.lr = start["epoch"] + 1, start["epochs"], start["lr"]
    else:
        first = 1
        args.epochs = args.epochs if args.epochs is not None else 20
        if args.lr is None:
            # Refining what is already trained wants a gentler rate than a first run.
            args.lr = start.get("lr", 1e-3) * 0.3 if start else 1e-3

    feats, stm, score, result = load(args.data, args.workers)
    n = len(stm)
    order = rng.permutation(n)
    held = max(1, min(n // 2, int(n * args.validation)))
    val, train = order[:held], order[held:]
    if len(train) == 0:
        sys.exit("not enough positions to train on")
    print(f"{len(train)} training and {len(val)} validation positions, hidden {args.hidden}, device {device}")

    net = Net(args.hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    if start:
        net.load_state_dict(start["model"])
        if "opt" in start:
            opt.load_state_dict(start["opt"])
        if not unfinished:
            # A new stretch of training: its own learning-rate schedule from --lr.
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] = args.lr
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=args.lr * 0.01)
    if unfinished:
        sched.load_state_dict(start["sched"])
        print(f"continuing the unfinished run at epoch {first} of {args.epochs} (--epochs and --lr wait until it is done)")
    elif start:
        kind = "checkpoint" if "opt" in start else "network file (rounded weights, fresh optimiser)"
        print(f"continuing from a {kind}: {args.epochs} more epoch(s) starting at learning rate {args.lr:g}")

    def checkpoint(epoch):
        state = {
            "model": net.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "scale": args.scale,
        }
        # Written beside the target and renamed, so a stop mid-save keeps the old one.
        torch.save(state, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    def batch_loss(idx):
        f = torch.from_numpy(feats[idx].astype(np.int64)).to(device)
        s = torch.from_numpy(stm[idx].astype(np.int64)).to(device)
        sc = torch.from_numpy(score[idx].astype(np.float32)).to(device)
        res = torch.from_numpy(result[idx].astype(np.float32)).to(device) / 2.0
        # The data is from White's side; the network scores the side to move.
        black = s == 1
        sc = torch.where(black, -sc, sc)
        res = torch.where(black, 1.0 - res, res)
        target = args.lam * torch.sigmoid(sc / args.scale) + (1.0 - args.lam) * res
        return torch.mean((torch.sigmoid(net(f, s)) - target) ** 2)

    def validation_loss():
        net.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(val), args.batch):
                idx = np.sort(val[i : i + args.batch])
                total += batch_loss(idx).item() * len(idx)
        net.train()
        return total / len(val)

    stopped = False
    try:
        for epoch in range(first, args.epochs + 1):
            t0 = time.time()
            perm = rng.permutation(train)
            total = 0.0
            for i in range(0, len(perm), args.batch):
                idx = np.sort(perm[i : i + args.batch])
                loss = batch_loss(idx)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            secs = time.time() - t0
            print(
                f"epoch {epoch:3d}/{args.epochs}  train loss {total / len(perm):.6f}  "
                f"validation loss {validation_loss():.6f}  {len(perm) / secs:,.0f} positions/s",
                flush=True,
            )
            checkpoint(epoch)
    except KeyboardInterrupt:
        stopped = True
        print(f"\nstopped; the checkpoint holds the last finished epoch. Carry on with --resume")

    q = quantise(net)
    save(args.out, q, args.scale)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")
    if not stopped:
        print(f"checkpoint {ckpt_path}: train further with --resume")
    if args.engine:
        verify(args.engine, args.out, q, args.scale, feats, stm, 500)


if __name__ == "__main__":
    main()
