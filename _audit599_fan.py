"""599: did a PMI fan of UNIQUE extras already cover the residual?

Not a learner. Unique list frozen; we only count how many of it we would try.
fan1 = unique-PMI top (≈0 on residual)
fan2 / fan3 / fan_all = held in first k / any unique extra
crowd = held in bag, not in unique set  ← SEEK-1 territory
rankmiss = held in unique set, wrong PMI top  ← fan territory

VOID n_res < 40
No GATE for training. Print which half of 595 the residual is.

    python _check599_fan.py
    python _audit599_fan.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit599_fan.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit599_fan.py --seed 2890 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage599_fan.json")


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
    print(f"599 fan  {path}  {kind}  windows={len(windows)}", flush=True)

    n = n_res = rankmiss = crowd = f1 = f2 = f3 = fall = 0
    for lines in windows:
        for row in collect_mix(lines, args, rng):
            held, bag, uniq, ranked = row["held"], row["bag"], row["uniq"], row["ranked"]
            n += 1
            u_ord = [tok for tok in ranked if tok in set(uniq)]
            u_hit = bool(u_ord) and u_ord[0] == held
            if held not in set(bag) or u_hit:
                continue
            n_res += 1
            in_u = held in set(uniq)
            rankmiss += int(in_u)
            crowd += int(not in_u)
            f1 += int(bool(u_ord) and u_ord[0] == held)
            f2 += int(held in set(u_ord[:2]))
            f3 += int(held in set(u_ord[:3]))
            fall += int(in_u)

    def rate(x):
        return x / n_res if n_res else 0.0

    void = n_res < 40
    print(
        f"n {n}  residual {n_res}  rankmiss {rate(rankmiss):.3f}  crowd {rate(crowd):.3f}"
    )
    print(
        f"fan1 {rate(f1):.3f}  fan2 {rate(f2):.3f}  fan3 {rate(f3):.3f}  fan_all {rate(fall):.3f}"
    )
    print(f"VOID {void}")
    if void:
        print("VOID: residual thin.")
    elif rate(rankmiss) > 0.5:
        print("FAN: leftover is mostly wrong unique rank. k-unique extras, not SEEK.")
    elif rate(crowd) > 0.5:
        print("SEEK: leftover is crowded frames. Unique fan does not cover it.")
    else:
        print("SPLIT: both halves live; do not mix fan and SEEK in one gate.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), n=n, n_res=n_res,
        rankmiss=rate(rankmiss), crowd=rate(crowd),
        fan1=rate(f1), fan2=rate(f2), fan3=rate(f3), fan_all=rate(fall),
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
