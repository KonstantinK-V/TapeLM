"""603: JOINT-SELECT ceiling on crowd.

One frame, not the union (602). extra = raw - qkeys - pin.
O     exists p: |raw intersect qkeys|>=2 AND extra(p)=={held}
A     random eligible frame
S     shortest eligible
MX    max |match| — hand rule, must beat
MAJ   majority bag extra
BAG   PMI report only

GATE  O-A>0.05 AND O-S>0.05 AND O-MX>0.05 AND O-MAJ>0.05
VOID  n_crowd < 40
unique k=2 not in this gate.

    python _check603_jsel.py
    python _audit603_jsel.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage603_jsel.json")


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
                # qkeys = visible question keys; held is the hole, not a key
                qkeys = set(env)
                frs = [
                    tuple(comps(g, t, v))
                    for t in by.get(v, ())
                    if t != s_q
                ]
            finally:
                if qtoks:
                    adjust_frame_stats(co, df, qtoks, +1)
            if not bag or not frs:
                continue
            ranked = pmi_rank(bag, env_m, co, df, n_use)
            out.append(dict(
                held=held, bag=bag, uniq=uniq, ranked=ranked,
                qkeys=qkeys, frames=frs, pin=v, mid_set=mid_set,
            ))
    return out


def extra_of(raw, qkeys, pin, mid_set):
    return (set(raw) & mid_set) - qkeys - {pin}


def eligible(frames, qkeys, pin, mid_set):
    out = []
    for raw in frames:
        if len(set(raw) & qkeys) >= 2:
            out.append((
                len(set(raw) & qkeys), raw,
                extra_of(raw, qkeys, pin, mid_set),
            ))
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
    rnd = random.Random(args.seed + 31)
    t0 = time.time()
    print(f"603 jsel  {path}  {kind}  windows={len(windows)}", flush=True)

    n = n_c = o = a = s = mx = m = b = n_el = 0
    for lines in windows:
        for row in collect(lines, args, rng):
            n += 1
            held, bag, uniq = row["held"], row["bag"], row["uniq"]
            if held not in set(bag) or held in set(uniq):
                continue
            n_c += 1
            el = eligible(
                row["frames"], row["qkeys"], row["pin"], row["mid_set"],
            )
            n_el += int(bool(el))
            o += int(any(ex == {held} for _, _, ex in el))
            if el:
                a += int(el[rnd.randrange(len(el))][2] == {held})
                short = min(el, key=lambda t: (len(t[1]), t[0]))
                s += int(short[2] == {held})
                best = max(el, key=lambda t: (t[0], -len(t[1])))
                mx += int(best[2] == {held})
            maj = Counter(bag).most_common(1)[0][0]
            m += int(maj == held)
            ranked = row["ranked"]
            b += int(bool(ranked) and ranked[0] == held)

    def r(x):
        return x / n_c if n_c else 0.0

    fo, fa, fs, fmx, fm, fb = r(o), r(a), r(s), r(mx), r(m), r(b)
    void = n_c < 40
    gate = (
        (not void)
        and (fo - fa > 0.05)
        and (fo - fs > 0.05)
        and (fo - fmx > 0.05)
        and (fo - fm > 0.05)
    )
    print(
        f"n {n}  crowd {n_c}  elig {n_el / n_c if n_c else 0:.3f}  "
        f"O {fo:.3f}  rnd {fa:.3f}  short {fs:.3f}  max {fmx:.3f}  "
        f"MAJ {fm:.3f}  BAG {fb:.3f}"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: crowd thin.")
    elif gate:
        print("SELECT OPEN: one joint frame isolates held; max-overlap does not close it.")
    else:
        print("STOP: select-one-frame does not beat random/short/max/majority.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        n=n, n_crowd=n_c, elig=n_el / n_c if n_c else 0.0,
        fill_o=fo, fill_rnd=fa, fill_short=fs, fill_max=fmx,
        fill_maj=fm, fill_bag=fb,
        d_rnd=fo - fa, d_short=fo - fs, d_max=fo - fmx, d_maj=fo - fm,
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
