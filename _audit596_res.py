"""596: learn on residual only. Unique frozen. PMI not in the key.

Train Q[repeats, bag-width] on leftover extras: +1 held, −0.08 other.
Pick argmax Q on the bag. Tie = mention order, not PMI.

A unique (0 on residual)
B bag PMI  ← ceiling, abort if we copy it
C random
L learned
N null (same mass, random key)

GATE  L−C > 0.05 AND (L < B−0.05 OR agree(L,B) < 0.8)
      beat random, do not copy PMI
VOID  n_res < 40 or n_keys < 3
COPY  agree≥0.8 and L ≥ B−0.05 → 38.3, not a mind

    python _check596_res.py
    python _audit596_res.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit596_res.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit596_res.py --seed 2890 --corpus data/_tinystories_train.txt
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
from _audit593_mix import collect_mix

OUT = Path("results/_stage596_res.json")


def residual(row):
    held, bag, uniq, ranked = row["held"], row["bag"], row["uniq"], row["ranked"]
    u_rank = [tok for tok in ranked if tok in set(uniq)]
    u_hit = bool(u_rank) and u_rank[0] == held
    return held in set(bag) and (not u_hit)


def key_of(tok, bag):
    rep = 1 if bag.count(tok) >= 2 else 0
    w = 0 if len(bag) < 5 else (1 if len(bag) < 9 else 2)
    return (rep, w)


def pick_q(bag, table, rng):
    seen = []
    best = None
    score = None
    for tok in bag:
        if tok in seen:
            continue
        seen.append(tok)
        s = table.get(key_of(tok, bag), 0.0)
        if score is None or s > score:
            score, best = s, tok
    return best if best is not None else bag[rng.randrange(len(bag))]


def run_split(rows, rng, train=True, table=None, null=False):
    table = table if table is not None else defaultdict(float)
    n_res = h_u = h_b = h_r = h_l = agree = 0
    keys = set()
    for row in rows:
        if not residual(row):
            continue
        held, bag, ranked = row["held"], row["bag"], row["ranked"]
        n_res += 1
        u_rank = [tok for tok in ranked if tok in set(row["uniq"])]
        h_u += int(bool(u_rank) and u_rank[0] == held)
        bpick = ranked[0] if ranked else None
        h_b += int(bpick == held)
        h_r += int(bag[rng.randrange(len(bag))] == held)
        if train:
            for tok in set(bag):
                k = key_of(tok, bag)
                if null:
                    k = (rng.randrange(3), rng.randrange(3))
                keys.add(k)
                table[k] += 1.0 if tok == held else -0.08
        else:
            lp = pick_q(bag, table, rng)
            h_l += int(lp == held)
            agree += int(lp == bpick)
            keys.add(key_of(lp, bag) if lp else (0, 0))
    return dict(
        table=table, n_res=n_res, keys=keys,
        h_u=h_u, h_b=h_b, h_r=h_r, h_l=h_l, agree=agree,
    )


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
    print(f"596 res  {path}  {kind}  windows={len(windows)}", flush=True)

    cut = max(1, int(0.7 * len(windows)))
    train_rows, test_rows = [], []
    for i, lines in enumerate(windows):
        rows = collect_mix(lines, args, rng)
        (train_rows if i < cut else test_rows).extend(rows)

    tr = run_split(train_rows, rng, train=True, null=False)
    nul = run_split(train_rows, random.Random(args.seed + 3), train=True, null=True)
    te = run_split(test_rows, random.Random(args.seed + 5), train=False, table=tr["table"])
    tn = run_split(test_rows, random.Random(args.seed + 7), train=False, table=nul["table"])

    n = te["n_res"]
    fu = te["h_u"] / n if n else 0.0
    fb = te["h_b"] / n if n else 0.0
    fr = te["h_r"] / n if n else 0.0
    fl = te["h_l"] / n if n else 0.0
    fn = tn["h_l"] / n if n else 0.0
    agr = te["agree"] / n if n else 0.0
    n_keys = len(tr["keys"])
    copy = agr >= 0.8 and fl >= fb - 0.05
    void = n < 40 or n_keys < 3
    gate = (not void) and (not copy) and (fl - fr > 0.05) and (
        fl < fb - 0.05 or agr < 0.8
    )
    print(
        f"test residual {n}  keys {n_keys}  "
        f"U {fu:.3f}  BAG {fb:.3f}  rnd {fr:.3f}  L {fl:.3f}  null {fn:.3f}  agree {agr:.3f}"
    )
    print(f"VOID {void}  COPY {copy}  GATE {gate}")
    if void:
        print("VOID: residual/keys thin.")
    elif copy:
        print("COPY PMI: learner matches bag ceiling. 38.3, not a mind.")
    elif gate:
        print("LEARN: residual policy beats random and is not the bag ranker.")
    else:
        print("STOP: no lift over random, or still the bag.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), copy=bool(copy), gate=bool(gate),
        n_res=n, n_keys=n_keys,
        fill_u=fu, fill_bag=fb, fill_rnd=fr, fill_L=fl, fill_null=fn,
        agree=agr, train_res=tr["n_res"],
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
