"""602: mini-composition ceiling on crowd. Fill is still one word.

qkeys = neighbors of the hole on the question line (not held).
score(frame) = |raw frame intersect qkeys|
take frames with score >= 2 (else empty)
E = extras of those frames
O  E == {held}

A    random bag extra
MAJ  majority bag extra
BAG  PMI ceiling only
U    unique extras (0 on crowd by construction)

GATE  O-A > 0.05 AND O-MAJ > 0.05
VOID  n_crowd < 40
Not unique-k=2. Not pair-intersect 601.

    python _check602_joint.py
    python _audit602_joint.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import adjust_frame_stats, co_table, env_mid, prefix_windows
from _audit593_mix import bag_of, pmi_rank

OUT = Path("results/_stage602_joint.json")


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
                qkeys = set(env)
                frs = []
                for t in by.get(v, ()):
                    if t == s_q:
                        continue
                    raw = tuple(comps(g, t, v))
                    extra = tuple(
                        tok for tok in raw
                        if tok not in qkeys and tok != v and tok in mid_set
                    )
                    frs.append((raw, extra))
            finally:
                if qtoks:
                    adjust_frame_stats(co, df, qtoks, +1)
            if not bag or not frs:
                continue
            ranked = pmi_rank(bag, env_m, co, df, n_use)
            out.append(dict(
                held=held, bag=bag, uniq=uniq, ranked=ranked,
                qkeys=qkeys, frames=frs,
            ))
    return out


def joint_E(frames, qkeys):
    acc = []
    for raw, extra in frames:
        if len(set(raw) & qkeys) >= 2:
            acc.extend(extra)
    return set(acc)


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
    rnd = random.Random(args.seed + 29)
    t0 = time.time()
    print(f"602 joint  {path}  {kind}  windows={len(windows)}", flush=True)

    n = n_c = o = a = m = b = u = nonempty = 0
    for lines in windows:
        for row in collect(lines, args, rng):
            n += 1
            held, bag, uniq = row["held"], row["bag"], row["uniq"]
            if held not in set(bag) or held in set(uniq):
                continue
            n_c += 1
            E = joint_E(row["frames"], row["qkeys"])
            nonempty += int(bool(E))
            o += int(E == {held})
            a += int(bag[rnd.randrange(len(bag))] == held)
            maj = Counter(bag).most_common(1)[0][0]
            m += int(maj == held)
            ranked = row["ranked"]
            b += int(bool(ranked) and ranked[0] == held)
            u += int(held in set(uniq))

    def r(x):
        return x / n_c if n_c else 0.0

    fo, fa, fm, fb, fu, fn = r(o), r(a), r(m), r(b), r(u), r(nonempty)
    void = n_c < 40
    gate = (not void) and (fo - fa > 0.05) and (fo - fm > 0.05)
    print(
        f"n {n}  crowd {n_c}  joint_nonempty {fn:.3f}  "
        f"O {fo:.3f}  rnd {fa:.3f}  MAJ {fm:.3f}  BAG {fb:.3f}  U {fu:.3f}"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: crowd thin.")
    elif gate:
        print("JOINT OPEN: query mini-composition isolates held on crowd.")
    else:
        print("STOP: joint keys do not isolate held above random/majority.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        n=n, n_crowd=n_c, joint_nonempty=fn,
        fill_o=fo, fill_rnd=fa, fill_maj=fm, fill_bag=fb, fill_u=fu,
        d_rnd=fo - fa, d_maj=fo - fm,
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
