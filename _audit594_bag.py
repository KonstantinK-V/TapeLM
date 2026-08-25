"""594: unique-PMI vs bag-PMI on the SAME holes. No Φ.

A  unique extras only; none → REFUSE
B  PMI-top of the whole mid bag
C  random bag extra

GATE  B−A > 0.05 AND B−C > 0.05
VOID  n < 40

    python _check594_bag.py
    python _audit594_bag.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit594_bag.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit594_bag.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit593_mix import collect_mix

OUT = Path("results/_stage594_bag.json")


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
    rnd = random.Random(args.seed + 17)
    t0 = time.time()
    print(f"594 bag  {path}  {kind}  windows={len(windows)}", flush=True)

    n = h_u = h_b = h_r = n_uniq = n_refuse = 0
    n_eps = 0
    for lines in windows:
        rows = collect_mix(lines, args, rng)
        n_eps += len(rows)
        for row in rows:
            held, bag, uniq, ranked = row["held"], row["bag"], row["uniq"], row["ranked"]
            if held not in set(bag):
                continue
            n += 1
            u_rank = [tok for tok in ranked if tok in set(uniq)]
            if u_rank:
                n_uniq += 1
                h_u += int(u_rank[0] == held)
            else:
                n_refuse += 1
            h_b += int(bool(ranked) and ranked[0] == held)
            h_r += int(bag[rnd.randrange(len(bag))] == held)

    fu, fb, fr = (h_u / n if n else 0.0), (h_b / n if n else 0.0), (h_r / n if n else 0.0)
    d_u, d_r = fb - fu, fb - fr
    void = n < 40
    gate = (not void) and d_u > 0.05 and d_r > 0.05
    print(
        f"n {n}  unique-holes {n_uniq}  refuse {n_refuse}  "
        f"U {fu:.3f}  BAG {fb:.3f}  rnd {fr:.3f}"
    )
    print(f"BAG-U {d_u:+.3f}  BAG-rnd {d_r:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: no holes.")
    elif gate:
        print("WIDEN LEGS: bag PMI beats unique-refuse. Still not Phi.")
    else:
        print("KEEP UNIQUE: bag does not beat unique+refuse on the full set.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), n_eps=n_eps,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        n=n, n_uniq=n_uniq, n_refuse=n_refuse,
        fill_u=fu, fill_bag=fb, fill_rnd=fr, d_u=d_u, d_r=d_r,
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
