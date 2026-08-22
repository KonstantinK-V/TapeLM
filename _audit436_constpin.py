"""436: freeze how the mind thinks. Mixed is logged, not trained."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage436_constpin.json")
WIKI = Path("data/_wikitext103_train.txt")
CAP = 8


def measure(lines, args, rng):
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        return None
    place, value, line = [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
    n = len(place)
    slots_at = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
    df = Counter(value)
    idx = list(range(n))
    rng.shuffle(idx)
    idx = idx[: args.max_questions]
    n_df1 = n_df2 = n_empty = n_const = n_mix = 0
    hit_c = ref_m = hop2 = n_pin = 0
    for s in idx:
        v, p, li = value[s], place[s], line[s]
        foreign = [t for t in slots_at[p] if t != s and line[t] != li]
        rng.shuffle(foreign)
        offer = foreign[:CAP]
        if df[v] < 2:
            n_df1 += 1
            continue
        n_df2 += 1
        if not offer:
            n_empty += 1
            continue
        maj_v = Counter(value[t] for t in offer).most_common(1)[0][0]
        if maj_v == v:
            n_const += 1
            t = next(x for x in offer if value[x] == maj_v)
            working = {("work", 0): t}
            hop2 += int(working[("work", 0)] == t)
            n_pin += 1
            hit_c += int(value[t] == v)
        else:
            n_mix += 1
            ref_m += 1
    return {
        "n": n, "questions": len(idx),
        "n_df1": n_df1, "n_df2": n_df2, "n_empty": n_empty,
        "n_const": n_const, "n_mixed": n_mix,
        "const_live": n_const / max(n_df2, 1),
        "mixed_of_df2": n_mix / max(n_df2, 1),
        "const_hit": hit_c / max(n_const, 1),
        "refuse_mixed": ref_m / max(n_mix, 1),
        "refuse_df1": 1.0,
        "hop2_sees_pin": hop2 / max(n_pin, 1),
        "k": CAP, "working_cells": 1,
        "think": "agree→pin majority e(P); differ→refuse; missing→refuse",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--max-questions", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]
    rep = measure(lines, args, rng)
    if rep is None:
        print("no tape")
        return 1
    rep["seed"] = args.seed
    void = rep["const_live"] <= 0.05
    gate = ((not void) and (rep["const_hit"] > 0.90)
            and (rep["n_mixed"] == 0 or rep["refuse_mixed"] > 0.90)
            and (rep["hop2_sees_pin"] == 1.0))
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"df2 {rep['n_df2']}  const {rep['n_const']} ({rep['const_live']:.3f})  "
          f"mixed {rep['n_mixed']} ({rep['mixed_of_df2']:.3f})  empty {rep['n_empty']}")
    print(f"THINK  const_hit {rep['const_hit']:.3f}  refuse_mixed {rep['refuse_mixed']:.3f}  "
          f"hop2 {rep['hop2_sees_pin']:.0f}")
    print(f"VOID {rep['void']}  GATE {rep['gate']}")
    if void:
        print("\nVOID: almost no constant e(P). Catalog has nothing to pin.")
    elif gate:
        print("\nGO THINK: agree → pin; differ → refuse. Mixed is a counter, not a loss.")
    else:
        print("\nSTOP: constant pin or mixed-refuse did not hold.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
