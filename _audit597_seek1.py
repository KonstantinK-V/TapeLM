"""597: SEEK-1 ceiling. Action = extra FRAME of the pin, not a bag word.

Frozen unique on the chosen frame: |extra|==1 -> pin, else refuse.
Oracle: exists a frame whose only extra is held.
A random frame. F shortest frame (not PMI). B bag-PMI = ceiling only.

GATE  O-A > 0.05 AND O-F > 0.05  on residual
VOID  n_res < 40
If O~A: no place-action to learn. Do not train.

    python _check597_seek1.py
    python _audit597_seek1.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import adjust_frame_stats, co_table, env_mid, prefix_windows
from _audit593_mix import bag_of, pmi_rank

OUT = Path("results/_stage597_seek1.json")


def frame_extras(g, by, node, s_q, env_m, mid_set):
    """Other frames of pin: each as the mid-extra list vs query env."""
    out = []
    for t in by.get(node, ()):
        if t == s_q:
            continue
        fr = set(comps(g, t, node))
        extra = [
            tok for tok in fr
            if tok not in env_m and tok != node and tok in mid_set
        ]
        if extra:
            out.append(tuple(extra))
    return out


def collect_seek(lines, args, rng):
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
                fr = frame_extras(g, by, v, s_q, env_m, mid_set)
            finally:
                if qtoks:
                    adjust_frame_stats(co, df, qtoks, +1)
            if not bag or not fr:
                continue
            ranked = pmi_rank(bag, env_m, co, df, n_use)
            out.append(dict(
                held=held, frames=fr, bag=bag, uniq=uniq, ranked=ranked,
            ))
    return out


def unique_hit(extra, held):
    return len(extra) == 1 and extra[0] == held


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
    print(f"597 seek1  {path}  {kind}  windows={len(windows)}", flush=True)

    n = n_res = o = a = fhit = b = 0
    n_eps = 0
    for lines in windows:
        rows = collect_seek(lines, args, rng)
        n_eps += len(rows)
        for row in rows:
            held, fr, uniq, ranked = row["held"], row["frames"], row["uniq"], row["ranked"]
            n += 1
            u_rank = [tok for tok in ranked if tok in set(uniq)]
            u_hit = bool(u_rank) and u_rank[0] == held
            if held not in set(row["bag"]) or u_hit:
                continue
            n_res += 1
            o += int(any(unique_hit(ex, held) for ex in fr))
            pick_a = fr[rnd.randrange(len(fr))]
            a += int(unique_hit(pick_a, held))
            pick_f = min(fr, key=len)
            fhit += int(unique_hit(pick_f, held))
            b += int(bool(ranked) and ranked[0] == held)

    mass = n_res / n if n else 0.0
    fo = o / n_res if n_res else 0.0
    fa = a / n_res if n_res else 0.0
    ff = fhit / n_res if n_res else 0.0
    fb = b / n_res if n_res else 0.0
    void = n_res < 40
    gate = (not void) and (fo - fa > 0.05) and (fo - ff > 0.05)
    print(
        f"n {n}  residual {n_res} mass {mass:.3f}  "
        f"O {fo:.3f}  rnd {fa:.3f}  short {ff:.3f}  BAG {fb:.3f}"
    )
    print(f"O-rnd {fo - fa:+.3f}  O-short {fo - ff:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: residual thin.")
    elif gate:
        print("SEEK-1 OPEN: a frame exists that re-opens unique. Policy can learn READ(place).")
    else:
        print("STOP: oracle not above random/short. No place-action on this residual.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), n_eps=n_eps,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        n=n, n_res=n_res, mass=mass,
        fill_o=fo, fill_rnd=fa, fill_short=ff, fill_bag=fb,
        d_rnd=fo - fa, d_short=fo - ff,
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
