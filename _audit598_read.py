"""598: train READ(frame) on residual. Unique frozen. PMI not in the key.

Q[len_bin] += +1 if unique_hit else -0.08
Pick argmax Q. Tie = random among max, not PMI.

S  random shortest frame (length rule)
A  random frame
O  oracle
B  bag-PMI ceiling only

COPY  agree(L,S) >= 0.8  -> learned length, not a mind
GATE  L-A > 0.05 AND L-S > 0.05 AND not COPY
VOID  n_res < 40 or n_keys < 2

    python _check598_read.py
    python _audit598_read.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit597_seek1 import collect_seek, unique_hit

OUT = Path("results/_stage598_read.json")


def residual(row):
    held, uniq, ranked = row["held"], row["uniq"], row["ranked"]
    u_rank = [tok for tok in ranked if tok in set(uniq)]
    u_hit = bool(u_rank) and u_rank[0] == held
    return held in set(row["bag"]) and (not u_hit)


def len_bin(fr):
    n = len(fr)
    return 0 if n <= 1 else (1 if n == 2 else 2)


def pick_q(frames, table, rng):
    best, sc = [], None
    for fr in frames:
        s = table.get(len_bin(fr), 0.0)
        if sc is None or s > sc:
            best, sc = [fr], s
        elif s == sc:
            best.append(fr)
    return best[rng.randrange(len(best))]


def pick_short(frames, rng):
    m = min(len(x) for x in frames)
    cands = [x for x in frames if len(x) == m]
    return cands[rng.randrange(len(cands))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=80)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=120000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"598 read  {path}  {kind}  windows={len(windows)}", flush=True)

    cut = max(1, int(0.7 * len(windows)))
    train, test = [], []
    for i, lines in enumerate(windows):
        rows = collect_seek(lines, args, rng)
        (train if i < cut else test).extend(rows)

    table = defaultdict(float)
    keys = set()
    tr_n = 0
    for row in train:
        if not residual(row):
            continue
        tr_n += 1
        held, frames = row["held"], row["frames"]
        for fr in frames:
            k = len_bin(fr)
            keys.add(k)
            table[k] += 1.0 if unique_hit(fr, held) else -0.08

    n = h_l = h_s = h_a = h_o = h_b = agr = 0
    rnd = random.Random(args.seed + 11)
    for row in test:
        if not residual(row):
            continue
        n += 1
        held, frames, ranked = row["held"], row["frames"], row["ranked"]
        lp = pick_q(frames, table, rnd)
        sp = pick_short(frames, rnd)
        apick = frames[rnd.randrange(len(frames))]
        h_l += int(unique_hit(lp, held))
        h_s += int(unique_hit(sp, held))
        h_a += int(unique_hit(apick, held))
        h_o += int(any(unique_hit(fr, held) for fr in frames))
        h_b += int(bool(ranked) and ranked[0] == held)
        agr += int(lp == sp)

    fl = h_l / n if n else 0.0
    fs = h_s / n if n else 0.0
    fa = h_a / n if n else 0.0
    fo = h_o / n if n else 0.0
    fb = h_b / n if n else 0.0
    agree = agr / n if n else 0.0
    copy = agree >= 0.8
    void = n < 40 or len(keys) < 2
    gate = (not void) and (not copy) and (fl - fa > 0.05) and (fl - fs > 0.05)
    print(
        f"test residual {n}  train {tr_n}  keys {len(keys)}  "
        f"L {fl:.3f}  short {fs:.3f}  rnd {fa:.3f}  O {fo:.3f}  BAG {fb:.3f}  agree {agree:.3f}"
    )
    print(f"VOID {void}  COPY {copy}  GATE {gate}")
    if void:
        print("VOID: residual/keys thin.")
    elif copy:
        print("COPY LENGTH: same as shortest-frame rule. Not a mind.")
    elif gate:
        print("LEARN READ: beats random and shortest; not the length rule.")
    else:
        print("STOP: no lift over length/random.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), copy=bool(copy), gate=bool(gate),
        n_res=n, train_res=tr_n, n_keys=len(keys),
        fill_L=fl, fill_short=fs, fill_rnd=fa, fill_o=fo, fill_bag=fb,
        agree=agree,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
