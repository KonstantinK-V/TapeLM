"""612: AGREE pin vs UNION lottery. Frozen SEARCH extract. No Q.

608 paid if any of k extracts == held. 436 pins only when they agree.
k=3 mention order. Unique+PMI frozen.

bins
  THIN   <2 non-null extracts → refuse
  CONST  all equal
  PEAK   unique mode, not all equal
  TIE    no unique mode → refuse

AGREE  pin unique mode (CONST∪PEAK), else refuse (miss)
UNION  any extract == held  (608)
MAJ    bag majority, 0 READs

VOID  n < 40 or n_const < 20
GO    hit_const - maj_const > 0.05
LOTTERY DIAG  union - agree > 0.05  (608 win lived on mix tickets)
k=6 printed, not gated.
608/609/611 not retrained. 557 closed.

    python _check612_agree.py
    python _audit612_agree.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit612_agree.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit612_agree.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import collect

OUT = Path("results/_stage612_agree.json")
K = 3


def taken(places, k):
    return [pl["extract"] for pl in places[:k] if pl["extract"] is not None]


def unique_mode(vals):
    if len(vals) < 2:
        return None
    cnt = Counter(vals)
    (tok, n1), *rest = cnt.most_common()
    n2 = rest[0][1] if rest else 0
    if n1 > n2:
        return tok
    return False


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
    print(f"612 agree  {path}  {kind}  windows={len(windows)}  k={K}", flush=True)

    n = u = a = m = 0
    bin_n = Counter()
    bin_hit = Counter()
    bin_maj = Counter()
    bin_uni = Counter()
    for lines in windows:
        for row in collect(lines, args, rng):
            n += 1
            held, places, bag0 = row["held"], row["places"], row["bag0"]
            maj = Counter(bag0).most_common(1)[0][0]
            vals = taken(places, K)
            mode = unique_mode(vals)
            if mode is None:
                kind_b = "thin"
                pin = None
            elif mode is False:
                kind_b = "tie"
                pin = None
            elif len(set(vals)) == 1:
                kind_b = "const"
                pin = mode
            else:
                kind_b = "peak"
                pin = mode
            uni = int(any(v == held for v in vals))
            ag = int(pin == held)
            mj = int(maj == held)
            u += uni
            a += ag
            m += mj
            bin_n[kind_b] += 1
            bin_hit[kind_b] += ag
            bin_maj[kind_b] += mj
            bin_uni[kind_b] += uni

    def r(x, d):
        return x / d if d else 0.0

    n_const = bin_n["const"]
    void = n < 40 or n_const < 20
    hit_c = r(bin_hit["const"], n_const)
    maj_c = r(bin_maj["const"], n_const)
    d_c = hit_c - maj_c
    union = r(u, n)
    agree = r(a, n)
    maj = r(m, n)
    lottery = union - agree
    gate = (not void) and d_c > 0.05
    print(f"n {n}  CONST {n_const}  PEAK {bin_n['peak']}  TIE {bin_n['tie']}  THIN {bin_n['thin']}")
    print(f"UNION {union:.3f}  AGREE {agree:.3f}  MAJ {maj:.3f}  lottery {lottery:+.3f}")
    print(
        f"CONST hit {hit_c:.3f}  maj {maj_c:.3f}  d {d_c:+.3f}  "
        f"union {r(bin_uni['const'], n_const):.3f}"
    )
    if bin_n["peak"]:
        print(
            f"PEAK  hit {r(bin_hit['peak'], bin_n['peak']):.3f}  "
            f"maj {r(bin_maj['peak'], bin_n['peak']):.3f}  "
            f"union {r(bin_uni['peak'], bin_n['peak']):.3f}"
        )
    if bin_n["tie"]:
        print(
            f"TIE   agree {r(bin_hit['tie'], bin_n['tie']):.3f}  "
            f"union {r(bin_uni['tie'], bin_n['tie']):.3f}  (refuse by construction)"
        )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: thin CONST. Hungry, not STOP of SEARCH.")
    elif gate:
        print("GO PIN: agreed extract beats bag on CONST. Lottery printed, not gated.")
    else:
        print("STOP: CONST pin does not beat bag. 608 win is not a 436 pin.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n, k=K,
        union=union, agree=agree, fill_bag_maj=maj, lottery=lottery,
        n_const=n_const, n_peak=bin_n["peak"], n_tie=bin_n["tie"],
        n_thin=bin_n["thin"],
        hit_const=hit_c, maj_const=maj_c, d_const=d_c,
        union_const=r(bin_uni["const"], n_const),
        hit_peak=r(bin_hit["peak"], bin_n["peak"]),
        maj_peak=r(bin_maj["peak"], bin_n["peak"]),
        union_peak=r(bin_uni["peak"], bin_n["peak"]),
        union_tie=r(bin_uni["tie"], bin_n["tie"]),
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
