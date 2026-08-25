"""601: PAIR-INTERSECT ceiling on crowd only.

Crowd = held in bag, not in unique extras.
O2  exists i!=j: extras(i) intersect extras(j) == {held}
A2  random pair
S2  two shortest frames
MAJ majority extra of the bag
BAG bag-PMI — ceiling only, not the action

GATE  O-A > 0.05 AND O-S > 0.05 AND O-MAJ > 0.05
VOID  n_crowd < 40
fan2 / unique k=2 not in this gate.

    python _check601_pair.py
    python _audit601_pair.py --seed 1337 --corpus data/_tinystories_train.txt
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
from _audit597_seek1 import collect_seek

OUT = Path("results/_stage601_pair.json")


def crowd(row):
    held = row["held"]
    return held in set(row["bag"]) and held not in set(row["uniq"])


def pair_hit(a, b, held):
    return (set(a) & set(b)) == {held}


def oracle(frames, held):
    sets = [set(fr) for fr in frames]
    n = len(sets)
    for i in range(n):
        for j in range(i + 1, n):
            if sets[i] & sets[j] == {held}:
                return True
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
    rnd = random.Random(args.seed + 23)
    t0 = time.time()
    print(f"601 pair  {path}  {kind}  windows={len(windows)}", flush=True)

    n = n_c = n_ok = o = a = s = m = b = 0
    for lines in windows:
        for row in collect_seek(lines, args, rng):
            n += 1
            if not crowd(row):
                continue
            n_c += 1
            frames, held, bag, ranked = row["frames"], row["held"], row["bag"], row["ranked"]
            if len(frames) < 2:
                continue
            n_ok += 1
            o += int(oracle(frames, held))
            i, j = rnd.sample(range(len(frames)), 2)
            a += int(pair_hit(frames[i], frames[j], held))
            short = sorted(frames, key=len)[:2]
            s += int(pair_hit(short[0], short[1], held))
            maj = Counter(bag).most_common(1)[0][0]
            m += int(maj == held)
            b += int(bool(ranked) and ranked[0] == held)

    def r(x):
        return x / n_ok if n_ok else 0.0

    fo, fa, fs, fm, fb = r(o), r(a), r(s), r(m), r(b)
    void = n_c < 40 or n_ok < 40
    gate = (not void) and (fo - fa > 0.05) and (fo - fs > 0.05) and (fo - fm > 0.05)
    print(
        f"n {n}  crowd {n_c}  pairable {n_ok}  "
        f"O {fo:.3f}  rnd {fa:.3f}  short {fs:.3f}  MAJ {fm:.3f}  BAG {fb:.3f}"
    )
    print(f"O-rnd {fo - fa:+.3f}  O-short {fo - fs:+.3f}  O-MAJ {fo - fm:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: crowd/pairable thin.")
    elif gate:
        print("PAIR OPEN: two frames isolate held. READ-pair can be taught.")
    else:
        print("STOP: pair oracle not above random/short/majority.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        n=n, n_crowd=n_c, n_ok=n_ok,
        fill_o=fo, fill_rnd=fa, fill_short=fs, fill_maj=fm, fill_bag=fb,
        d_rnd=fo - fa, d_short=fo - fs, d_maj=fo - fm,
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
