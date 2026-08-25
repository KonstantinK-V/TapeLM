"""595: number the residual. Unique law FROZEN. Bag PMI = ceiling, not law.

residual  held ∈ bag  AND  (no unique extra OR unique-PMI ≠ held)
A  unique law (0 on this set by construction if no unique-hit)
B  bag PMI     ← how much legs WOULD steal. Do not promote.
C  random bag

GATE  mass > 0.05 AND B−A > 0.05 AND B−C > 0.05
VOID  n_res < 40
Print prize = B−A. Next learner must beat A and lose to B or beat B
without using PMI. If it matches B → 38.3, abort.

    python _check595_gap.py
    python _audit595_gap.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit595_gap.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit595_gap.py --seed 2890 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage595_gap.json")


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
    print(f"595 gap  {path}  {kind}  windows={len(windows)}", flush=True)

    n = n_res = h_u = h_b = h_r = 0
    n_eps = 0
    for lines in windows:
        rows = collect_mix(lines, args, rng)
        n_eps += len(rows)
        for row in rows:
            held, bag, uniq, ranked = row["held"], row["bag"], row["uniq"], row["ranked"]
            n += 1
            u_rank = [tok for tok in ranked if tok in set(uniq)]
            u_hit = bool(u_rank) and u_rank[0] == held
            if held not in set(bag) or u_hit:
                continue
            n_res += 1
            h_u += int(u_hit)
            h_b += int(bool(ranked) and ranked[0] == held)
            h_r += int(bag[rnd.randrange(len(bag))] == held)

    mass = n_res / n if n else 0.0
    fu = h_u / n_res if n_res else 0.0
    fb = h_b / n_res if n_res else 0.0
    fr = h_r / n_res if n_res else 0.0
    prize = fb - fu
    void = n_res < 40
    gate = (not void) and mass > 0.05 and prize > 0.05 and (fb - fr) > 0.05
    print(
        f"n {n}  residual {n_res} mass {mass:.3f}  "
        f"U {fu:.3f}  BAG {fb:.3f}  rnd {fr:.3f}  prize {prize:+.3f}"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: residual too thin.")
    elif gate:
        print("PRIZE OPEN: unique frozen; bag ceiling is the stolen mass. Do not put bag in the law.")
    else:
        print("STOP: no numbered prize (mass or bag-vs-unique too small).")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), n_eps=n_eps,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        n=n, n_res=n_res, mass=mass,
        fill_u=fu, fill_bag=fb, fill_rnd=fr, prize=prize,
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
