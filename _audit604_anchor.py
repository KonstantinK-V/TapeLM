"""604: ANCHOR-IMPORT ceiling on crowd. No learner, no PMI in resolver.

Crowd = held in pin-neighborhood, but held is not a frozen unique extra.
Action IMPORT(q) chooses a visible qkey place, never a bag word.
Tape resolves:
  P       = mid fillers from other frames of the pin
  N(q)    = mid fillers from other frames mentioning visible anchor q
  C(q)    = P intersect N(q)
  C(q1,q2)= C(q1) intersect C(q2)
RESOLVE only when C == {one place}; otherwise REFUSE.

O1/O2  exists one anchor / anchor pair resolving exactly to held
A1/A2  random anchor / pair
MIN1/2 smallest nonempty constraint set (strong hand rival)
MAJ    majority filler of P
BAG    bag-PMI ceiling, report only

GATE at a depth requires O beat random, MIN, and MAJ by > 0.05.
VOID n_crowd < 40. Unique k=2 is not in this gate.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import adjust_frame_stats, co_table, env_mid, prefix_windows
from _audit593_mix import bag_of, pmi_rank

OUT = Path("results/_stage604_anchor.json")


def import_set(g, by, anchor, s_q, qkeys, pin, mid_set):
    """Exact tape import for one visible anchor; current query frame is hidden."""
    out = set()
    for t in by.get(anchor, ()):
        if t == s_q:
            continue
        for tok in comps(g, t, anchor):
            if tok in mid_set and tok not in qkeys and tok != pin:
                out.add(tok)
    return out


def collect(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return []
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    co, frames, n_fr = co_table(g, by)
    df = g.get("df") or {tok: len(slots) for tok, slots in by.items()}
    out = []
    pins = list(mid)
    rng.shuffle(pins)
    for pin in pins:
        slots = list(by.get(pin, ()))
        if len(slots) < 2:
            continue
        rng.shuffle(slots)
        for s_q in slots[1: args.cap_probe + 1]:
            frame = list(comps(g, s_q, pin))
            if len(frame) < 2:
                continue
            rng.shuffle(frame)
            held, env = frame[0], set(frame[1:])
            if held not in mid_set or held == pin:
                continue
            env_m = env_mid(env, mid_set, high_set)
            if not env_m:
                continue
            qtoks = frames.get(s_q)
            if qtoks:
                adjust_frame_stats(co, df, qtoks, -1)
                n_use = max(n_fr - 1, 1)
            else:
                n_use = n_fr
            try:
                bag, uniq = bag_of(g, by, pin, s_q, env_m, mid_set)
                qkeys = set(env)
                pin_set = set(bag)
                anchors = []
                for q in qkeys:
                    imported = import_set(g, by, q, s_q, qkeys, pin, mid_set)
                    constrained = pin_set & imported
                    if constrained:
                        anchors.append((q, constrained))
            finally:
                if qtoks:
                    adjust_frame_stats(co, df, qtoks, +1)
            if not bag:
                continue
            ranked = pmi_rank(bag, env_m, co, df, n_use)
            out.append(dict(
                held=held, bag=bag, uniq=uniq, ranked=ranked,
                anchors=anchors,
            ))
    return out


def singleton_hit(cands, held):
    return cands == {held}


def choose_min(sets, rng):
    if not sets:
        return set()
    size = min(len(cands) for cands in sets)
    best = [cands for cands in sets if len(cands) == size]
    return best[rng.randrange(len(best))]


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
    rnd = random.Random(args.seed + 37)
    t0 = time.time()
    print(f"604 anchor  {path}  {kind}  windows={len(windows)}", flush=True)

    n = n_c = has1 = has2 = 0
    o1 = a1 = mn1 = o2 = a2 = mn2 = maj = bag_hit = 0
    for lines in windows:
        for row in collect(lines, args, rng):
            n += 1
            held, bag, uniq = row["held"], row["bag"], row["uniq"]
            if held not in set(bag) or held in set(uniq):
                continue
            n_c += 1
            sets1 = [cands for _q, cands in row["anchors"]]
            has1 += int(bool(sets1))
            o1 += int(any(singleton_hit(cands, held) for cands in sets1))
            if sets1:
                a1 += int(singleton_hit(sets1[rnd.randrange(len(sets1))], held))
                mn1 += int(singleton_hit(choose_min(sets1, rnd), held))

            sets2 = [a & b for a, b in combinations(sets1, 2) if a & b]
            has2 += int(bool(sets2))
            o2 += int(any(singleton_hit(cands, held) for cands in sets2))
            if sets2:
                a2 += int(singleton_hit(sets2[rnd.randrange(len(sets2))], held))
                mn2 += int(singleton_hit(choose_min(sets2, rnd), held))

            maj_pick = Counter(bag).most_common(1)[0][0]
            maj += int(maj_pick == held)
            ranked = row["ranked"]
            bag_hit += int(bool(ranked) and ranked[0] == held)

    def rate(x):
        return x / n_c if n_c else 0.0

    f_o1, f_a1, f_mn1 = rate(o1), rate(a1), rate(mn1)
    f_o2, f_a2, f_mn2 = rate(o2), rate(a2), rate(mn2)
    f_maj, f_bag = rate(maj), rate(bag_hit)
    void = n_c < 40
    gate1 = (
        (not void)
        and (f_o1 - f_a1 > 0.05)
        and (f_o1 - f_mn1 > 0.05)
        and (f_o1 - f_maj > 0.05)
    )
    gate2 = (
        (not void)
        and (f_o2 - f_a2 > 0.05)
        and (f_o2 - f_mn2 > 0.05)
        and (f_o2 - f_maj > 0.05)
    )
    gate = gate1 or gate2
    print(
        f"n {n}  crowd {n_c}  anchors {rate(has1):.3f}  pairs {rate(has2):.3f}  "
        f"MAJ {f_maj:.3f}  BAG {f_bag:.3f}"
    )
    print(
        f"A1 O {f_o1:.3f}  rnd {f_a1:.3f}  min {f_mn1:.3f}  "
        f"A2 O {f_o2:.3f}  rnd {f_a2:.3f}  min {f_mn2:.3f}"
    )
    print(f"VOID {void}  GATE1 {gate1}  GATE2 {gate2}  GATE {gate}")
    if void:
        print("VOID: crowd thin.")
    elif gate1:
        print("ANCHOR-1 OPEN: one imported query place resolves held.")
    elif gate2:
        print("ANCHOR-2 OPEN: two imported query places resolve held.")
    else:
        print("STOP: exact anchor constraints do not beat random/min/majority.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate1=bool(gate1), gate2=bool(gate2), gate=bool(gate),
        n=n, n_crowd=n_c, has_anchor=rate(has1), has_pair=rate(has2),
        fill_o1=f_o1, fill_rnd1=f_a1, fill_min1=f_mn1,
        fill_o2=f_o2, fill_rnd2=f_a2, fill_min2=f_mn2,
        fill_maj=f_maj, fill_bag=f_bag,
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
