"""434: offer = foreign slots of place P. 433 window VOID; pin was not the hole."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage434_placepin.json")
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
    n_df2 = n_live = n_const = n_mix = 0
    ora_m, rnd_m, maj_m = [], [], []
    ora_c, rnd_c = [], []
    n_pin = hop2_ok = 0
    for s in idx:
        v, p, li = value[s], place[s], line[s]
        if df[v] < 2:
            continue
        n_df2 += 1
        foreign = [t for t in slots_at[p] if t != s and line[t] != li]
        rng.shuffle(foreign)
        offer = foreign[:CAP]
        teach = next((t for t in offer if value[t] == v), None)
        if teach is None:
            continue
        n_live += 1
        working = {("work", 0): teach}
        hop2 = working[("work", 0)]
        n_pin += 1
        hop2_ok += int(hop2 == teach)
        rnd = rng.choice(offer)
        maj_v = Counter(value[t] for t in offer).most_common(1)[0][0]
        maj = next(t for t in offer if value[t] == maj_v)
        hit_o, hit_r, hit_j = 1, int(value[rnd] == v), int(value[maj] == v)
        if maj_v == v:
            n_const += 1
            ora_c.append(hit_o)
            rnd_c.append(hit_r)
        else:
            n_mix += 1
            ora_m.append(hit_o)
            rnd_m.append(hit_r)
            maj_m.append(hit_j)

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    live = n_live / max(n_df2, 1)
    return {
        "n": n, "n_df2": n_df2, "n_live": n_live, "live": live,
        "n_const": n_const, "n_mixed": n_mix,
        "const_of_live": n_const / max(n_live, 1),
        "mixed_of_live": n_mix / max(n_live, 1),
        "ora_mixed": mean(ora_m), "rnd_mixed": mean(rnd_m), "maj_mixed": mean(maj_m),
        "d_mixed": mean(ora_m) - mean(rnd_m),
        "ora_const": mean(ora_c), "rnd_const": mean(rnd_c),
        "d_const": mean(ora_c) - mean(rnd_c),
        "hop2_sees_pin": hop2_ok / max(n_pin, 1),
        "k": CAP, "working_cells": 1,
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
    void = rep["live"] <= 0.05
    gate = (not void) and (rep["n_mixed"] >= 30) and (rep["d_mixed"] > 0.05)
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"df≥2 {rep['n_df2']}  live {rep['live']:.3f}  "
          f"const {rep['n_const']} mixed {rep['n_mixed']}  hop2 {rep['hop2_sees_pin']:.0f}")
    print(f"const  ora {rep['ora_const']:.3f} rnd {rep['rnd_const']:.3f}  Δ {rep['d_const']:+.3f}")
    print(f"mixed  ora {rep['ora_mixed']:.3f} rnd {rep['rnd_mixed']:.3f}  "
          f"maj {rep['maj_mixed']:.3f}  Δ {rep['d_mixed']:+.3f}")
    print(f"VOID {rep['void']}  GATE {rep['gate']}")
    if void:
        print("\nVOID: even the place's other cells don't hold v. Nothing to pin.")
    elif rep["n_mixed"] < 30:
        print("\nCatalog already pins (constants). No pick.")
    elif gate:
        print("\nGO OFFER: e(P) has a minority v; random misses it. Then train pick.")
    else:
        print("\nSTOP: live, mixed, oracle not above random.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
