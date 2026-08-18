"""CAN A PLACE BE MADE THICK? The two knobs that move mentions-per-place, swept together.

WHY. 346 measured a place at 4.24 MENTIONS. One is hidden, so the mind sees three values and can
form three lens pairs, of which 1.7 are non-empty. 84% of co-occurring pairs are seen EXACTLY
ONCE. Every second-order operation this project has tried - composition (310), enumeration at
scale (335), the constraint (345) - has been computing a statistic over about four samples. No
statistic works on four samples, and that is one sentence for three separate failures.

AND THE CORPUS TEST FOR IT DID NOT HAPPEN. 30 MB gave 4.24 mentions per place; 120 MB gave 3.99.
The tape is built from a --window-lines region, so the window is 400 lines whatever the corpus
is: more bytes only move where the window can land. The knob never moved the quantity - the same
shape of fault as the dead --addresses flag in 335, caught this time by printing the quantity
next to the knob.

THE TWO KNOBS THAT DO MOVE IT, and they are different in kind:

  --window-lines   thickness bought with MORE TEXT. A place accumulates mentions from a wider
                   region, so recurrence rises without changing what a place IS.
  --frame-max      thickness bought with a LOOSER DEFINITION of a place. A frame matching on 3
                   tokens each side recurs rarely by construction; 2 or 1 recurs far more often,
                   but the places mean less. This is the write path (342d) - counting, so the
                   invariant is untouched - and it is the first time it has been swept.

WHAT TO READ, in this order:
  mentions_per_place   did the knob move the quantity at all. If not, stop reading.
  support_2plus        the share of co-occurring pairs seen more than once. A distribution of
                       singletons has no peak, and no rule for taking a maximum will invent one.
  one_present_topm     what a lens could reach at a matched offer - the payoff of thickness.
  one_count_right      what it actually resolves to. The gap between these two is the illness.

A LOOSER FRAME BUYS THICKNESS AT THE COST OF MEANING, so the two knobs must be read apart: if
--window-lines raises support and --frame-max does not, thickness is real; if only --frame-max
does, we have made places bigger and emptier and the numbers will say so through
one_present_topm failing to follow.

    python _sweep347_thick.py
    python _sweep347_thick.py --quick
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OUT = Path("results/_stage347_thick.json")
SRC = Path("results/_stage346_lens.json")
KEYS = ("places", "mentions_per_place", "support_1", "support_2plus", "support_3plus",
        "one_present_topm", "one_count_right", "pair_present_topm", "mean_lens_offer",
        "in_own", "questions")


def one(extra):
    r = subprocess.run([sys.executable, "_audit346_lens.py"] + extra,
                       capture_output=True, text=True)
    if r.returncode != 0 or not SRC.exists():
        print(f"  FAILED {' '.join(extra)}\n{r.stdout[-300:]}{r.stderr[-300:]}")
        return None
    d = json.loads(SRC.read_text(encoding="utf-8"))
    return {k: d.get(k) for k in KEYS}


def show(name, knob, rows):
    print(f"\n{name}")
    print(f"  {knob:>8} {'places':>7} {'ment/pl':>8} {'sup2+':>7} {'sup3+':>7} "
          f"{'offer':>7} {'present@8':>10} {'argmax':>8} {'pair@8':>8}")
    for v, d in rows:
        if d is None:
            continue
        print(f"  {v:>8} {d['places']:>7} {d['mentions_per_place']:8.2f} "
              f"{d['support_2plus']:7.4f} {d['support_3plus']:7.4f} "
              f"{d['mean_lens_offer']:7.1f} {d['one_present_topm']:10.4f} "
              f"{d['one_count_right']:8.4f} {d['pair_present_topm']:8.4f}")
    pts = [(v, d) for v, d in rows if d]
    if len(pts) < 2:
        return
    # THE KNOB MUST MOVE THE QUANTITY BEFORE ANYTHING ELSE IS READ. 346's corpus test failed
    # exactly here and the failure was invisible until the quantity was printed beside the knob.
    m0, m1 = pts[0][1]["mentions_per_place"], pts[-1][1]["mentions_per_place"]
    if abs(m1 - m0) < 0.25:
        print(f"  DEAD KNOB: mentions per place {m0:.2f} -> {m1:.2f}. This knob does not "
              f"thicken a place, so nothing below it is a test of thickness.")
        return
    s0, s1 = pts[0][1]["support_2plus"], pts[-1][1]["support_2plus"]
    p0, p1 = pts[0][1]["one_present_topm"], pts[-1][1]["one_present_topm"]
    a0, a1 = pts[0][1]["one_count_right"], pts[-1][1]["one_count_right"]
    print(f"  thickness {m0:.2f} -> {m1:.2f}   support2+ {s0:.4f} -> {s1:.4f}   "
          f"present@8 {p0:.4f} -> {p1:.4f}   argmax {a0:.4f} -> {a1:.4f}")
    if s1 > s0 + 0.02 and a1 > a0 + 0.01:
        print("  THICKNESS PAYS: the distribution stops being singletons AND the resolved "
              "answer follows. The substrate was the constraint, and the write path is the "
              "lever this project has been looking for.")
    elif s1 > s0 + 0.02:
        print("  THICKER BUT NOT BETTER: support rises and the resolved answer does not. The "
              "places got bigger without getting more meaningful - which is what a looser "
              "frame would do, and it is not thickness in the sense that matters.")
    else:
        print("  NO: the knob moves mentions per place and the distribution stays singletons.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    base = ["--bytes", str(args.bytes), "--addresses", str(args.addresses),
            "--seed", str(args.seed)]

    wins = [400, 1600] if args.quick else [400, 800, 1600, 3200, 6400]
    frames = [3, 2] if args.quick else [3, 2, 1]

    rep = {"bytes": args.bytes, "addresses": args.addresses, "window": {}, "frame_max": {}}
    rows_w = []
    for w in wins:
        print(f"window={w} ...", flush=True)
        d = one(base + ["--window-lines", str(w), "--frame-max", "3"])
        rep["window"][str(w)] = d
        rows_w.append((w, d))
    rows_f = []
    for f in frames:
        print(f"frame_max={f} ...", flush=True)
        d = one(base + ["--window-lines", "400", "--frame-max", str(f)])
        rep["frame_max"][str(f)] = d
        rows_f.append((f, d))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    show("MORE TEXT - a wider region, the place unchanged", "window", rows_w)
    show("A LOOSER PLACE - the write path, same text", "frame_max", rows_f)
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
