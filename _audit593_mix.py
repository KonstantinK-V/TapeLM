"""593: leftover mass where unique extra is silent. Ceiling, no Φ.

cover  held ∈ bag of all mid extras (other frames)
r      held ∈ unique extras
mixed  held ∈ bag and not unique   ← PMI-top law does not apply

On mixed only:
  PMI_bag   rank every bag extra by PMI (no unique filter)
  maj       mode of bag
  rnd       uniform bag

VOID  n_mixed < 40
Print mass = n_mixed/n. Not a close of v1.

    python _check593_mix.py
    python _audit593_mix.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit593_mix.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit593_mix.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import (
    adjust_frame_stats, co_table, env_mid, mean_pmi, prefix_windows,
)

OUT = Path("results/_stage593_mix.json")


def bag_of(g, by, node, s_q, env_m, mid_set):
    bag = []
    unique = []
    seen_u = set()
    for t in by.get(node, ()):
        if t == s_q:
            continue
        fr = set(comps(g, t, node))
        extra = [
            tok for tok in fr
            if tok not in env_m and tok != node and tok in mid_set
        ]
        for tok in fr:
            if tok in mid_set and tok != node:
                bag.append(tok)
        if len(extra) != 1:
            continue
        tok = extra[0]
        if tok not in seen_u:
            seen_u.add(tok)
            unique.append(tok)
    return bag, unique


def pmi_rank(cands, env_m, co, df, n_fr):
    scored = [(mean_pmi(tok, env_m, co, df, n_fr), tok) for tok in set(cands)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [tok for _p, tok in scored]


def collect_mix(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return []
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    co, frames, n_fr = co_table(g, by)
    df = g.get("df") or {tok: len(slots) for tok, slots in by.items()}
    out = []
    keys = list(mid)
    rng.shuffle(keys)
    for v in keys:
        slots = list(by.get(v, ()))
        if len(slots) < 2:
            continue
        rng.shuffle(slots)
        for s_q in slots[1: args.cap_probe + 1]:
            frame = list(comps(g, s_q, v))
            if len(frame) < 2:
                continue
            rng.shuffle(frame)
            held, env = frame[0], set(frame[1:])
            if held not in mid_set or held == v:
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
                bag, uniq = bag_of(g, by, v, s_q, env_m, mid_set)
            finally:
                if qtoks:
                    adjust_frame_stats(co, df, qtoks, +1)
            if not bag:
                continue
            ranked = pmi_rank(bag, env_m, co, df, n_use)
            out.append(dict(bag=bag, uniq=uniq, ranked=ranked, held=held))
    return out


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
    print(f"593 mix  {path}  {kind}  windows={len(windows)}", flush=True)

    n = n_u = n_m = 0
    h_u = h_m_pmi = h_m_maj = h_m_rnd = 0
    n_eps = 0
    for lines in windows:
        rows = collect_mix(lines, args, rng)
        n_eps += len(rows)
        for row in rows:
            held, bag, uniq, ranked = row["held"], row["bag"], row["uniq"], row["ranked"]
            if held not in set(bag):
                continue
            n += 1
            if held in set(uniq):
                n_u += 1
                u_rank = [tok for tok in ranked if tok in set(uniq)]
                h_u += int(bool(u_rank) and u_rank[0] == held)
                continue
            n_m += 1
            h_m_pmi += int(bool(ranked) and ranked[0] == held)
            maj = Counter(bag).most_common(1)[0][0]
            h_m_maj += int(maj == held)
            h_m_rnd += int(bag[rnd.randrange(len(bag))] == held)

    mass = n_m / n if n else 0.0
    fill_u = h_u / n_u if n_u else 0.0
    fp = h_m_pmi / n_m if n_m else 0.0
    fm = h_m_maj / n_m if n_m else 0.0
    fr = h_m_rnd / n_m if n_m else 0.0
    void = n_m < 40
    # gap for a mind: neither PMI_bag nor maj beat rnd by 0.05
    pmi_live = fp - fr > 0.05
    maj_live = fm - fr > 0.05
    gap = (not void) and (not pmi_live) and (not maj_live)
    print(
        f"n {n}  unique {n_u} fill {fill_u:.3f}  mixed {n_m} mass {mass:.3f}"
    )
    print(f"mixed  PMI_bag {fp:.3f}  maj {fm:.3f}  rnd {fr:.3f}")
    print(f"VOID {void}  PMI_live {pmi_live}  maj_live {maj_live}  GAP {gap}")
    if void:
        print("VOID: mixed remainder too thin.")
    elif gap:
        print("GAP: mixed mass exists; PMI and maj silent vs rnd. Road for learning.")
    elif pmi_live:
        print("LEGS: PMI on the bag still ranks mixed. Not a mind exam.")
    else:
        print("KNOW: majority of mixed bag. Knowledge, not mind.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), n_eps=n_eps,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gap=bool(gap),
        pmi_live=bool(pmi_live), maj_live=bool(maj_live),
        n=n, n_unique=n_u, n_mixed=n_m, mass=mass,
        fill_unique=fill_u, fill_pmi=fp, fill_maj=fm, fill_rnd=fr,
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
