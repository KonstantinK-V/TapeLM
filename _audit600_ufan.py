"""600: unique fan k=2 on ALL holes + 511 as chooser on crowd residual.

Not bag in the law. Unique extras only, budget 2.
Crowd (held not in unique set): walk-order 511 vs bag PMI vs random.

A  unique-PMI top1
U2 unique-PMI first 2
Uall unique set
B  bag PMI
R  random bag extra
W  511: first extra that sits in cheap_rec of the pin (crowd only)

GATE  U2-A > 0.05 on all holes AND bag still above U2 (fan is not the bag)
COPY  abs(U2-B) <= 0.02
VOID  n < 200
Crowd is diagnostic, not in the gate.

    python _check600_ufan.py
    python _audit600_ufan.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import cheap_rec, comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import adjust_frame_stats, co_table, env_mid, prefix_windows
from _audit593_mix import bag_of, pmi_rank

OUT = Path("results/_stage600_ufan.json")


def collect(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return []
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    co, frames, n_fr = co_table(g, by)
    df = g.get("df") or {tok: len(slots) for tok, slots in by.items()}
    cache = {}
    out = []
    keys = list(mid)
    rng.shuffle(keys)
    for v in keys:
        slots = list(by.get(v, ()))
        if len(slots) < 2:
            continue
        rng.shuffle(slots)
        rec = cheap_rec(g, by, v, cache)
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
            out.append(dict(
                held=held, bag=bag, uniq=uniq, ranked=ranked, rec=list(rec),
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
    rnd = random.Random(args.seed + 19)
    t0 = time.time()
    print(f"600 ufan  {path}  {kind}  windows={len(windows)}", flush=True)

    n = h1 = h2 = hall = hb = hr = 0
    nc = hcw = hcb = hcr = 0
    for lines in windows:
        for row in collect(lines, args, rng):
            held, bag, uniq, ranked, rec = (
                row["held"], row["bag"], row["uniq"], row["ranked"], row["rec"]
            )
            n += 1
            u_ord = [tok for tok in ranked if tok in set(uniq)]
            h1 += int(bool(u_ord) and u_ord[0] == held)
            h2 += int(held in set(u_ord[:2]))
            hall += int(held in set(uniq))
            bpick = ranked[0] if ranked else None
            hb += int(bpick == held)
            hr += int(bag[rnd.randrange(len(bag))] == held)
            if held in set(bag) and held not in set(uniq):
                nc += 1
                wpick = next((c for c in rec if c in set(bag)), None)
                hcw += int(wpick == held)
                hcb += int(bpick == held)
                hcr += int(bag[rnd.randrange(len(bag))] == held)

    def r(x, d):
        return x / d if d else 0.0

    a, u2, ua, b, rr = r(h1, n), r(h2, n), r(hall, n), r(hb, n), r(hr, n)
    void = n < 200
    copy = (not void) and (abs(u2 - b) <= 0.02)
    gate = (not void) and (u2 - a > 0.05) and (b - u2 > 0.02) and (not copy)
    print(
        f"ALL n {n}  U1 {a:.3f}  U2 {u2:.3f}  Uall {ua:.3f}  BAG {b:.3f}  rnd {rr:.3f}  "
        f"U2-U1 {u2 - a:+.3f}"
    )
    print(
        f"CROWD n {nc}  511 {r(hcw, nc):.3f}  BAG {r(hcb, nc):.3f}  rnd {r(hcr, nc):.3f}"
    )
    print(f"VOID {void}  COPY {copy}  GATE {gate}")
    if void:
        print("VOID: thin.")
    elif copy:
        print("COPY: unique fan2 ~ bag. Do not put fan in the law.")
    elif gate:
        print("FAN: two unique extras beat top-1 and are not the bag.")
    else:
        print("STOP: fan2 does not clear +0.05 over unique-1, or equals bag.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), copy=bool(copy), gate=bool(gate),
        n=n, fill_u1=a, fill_u2=u2, fill_uall=ua, fill_bag=b, fill_rnd=rr,
        d_u2=u2 - a,
        n_crowd=nc, crowd_511=r(hcw, nc), crowd_bag=r(hcb, nc), crowd_rnd=r(hcr, nc),
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
