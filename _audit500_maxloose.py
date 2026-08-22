"""500 NIGHT MAX-LOOSE: pin=always (random slot), hops=8, unique only as bonus.

The last freestyle: almost no gate. Q only ranks where to step next.
If even this cannot beat random walk on chain length / unique-stumble,
loose pin is empty. If it beats random → something to chase later.

    python _audit500_maxloose.py --seed 1337 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit485_hunt import (
    build_window, load_lines, pick_by_q, pick_corpus, pre, touch, unique_next,
)

OUT = Path("results/_stage500_maxloose.json")


def grab(P, g, rng):
    sl = g["slots_at"][P]
    if not sl:
        return None
    return g["value"][rng.choice(sl)]


def episode(g, rng, q, tot, win, eps, budget, max_hops, learn):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 1]
    if len(places) < 4:
        return dict(len=0, uniq=0)
    # pick start by Q
    P0 = pick_by_q(places, q, lambda x: pre(x, g) if x in g["slots_at"] and len(g["slots_at"][x]) >= 2
                   else (1, 1, 0, 0), rng, eps if learn else 0.2)
    if P0 is None:
        P0 = rng.choice(places)
    v = grab(P0, g, rng)
    if v is None:
        return dict(len=0, uniq=0)
    length = 0
    uniq = int(unique_next(P0, g) is not None) if len(g["slots_at"].get(P0, [])) >= 2 else 0
    visited = {P0}
    if learn:
        touch(q, tot, win, pre(P0, g) if len(g["slots_at"].get(P0, [])) >= 2 else (0, 0, 0, 0), 0.1)
    for _ in range(max_hops):
        nbrs = [T for T in g["by_key"].get(v, set()) if T not in visited]
        if not nbrs:
            break
        T = pick_by_q(nbrs, q, lambda x: pre(x, g) if len(g["slots_at"].get(x, [])) >= 2
                      else (1, 1, 0, 0), rng, eps if learn else 0.25)
        if T is None:
            T = rng.choice(nbrs)
        v2 = grab(T, g, rng)
        if v2 is None:
            if learn:
                touch(q, tot, win, pre(T, g) if len(g["slots_at"].get(T, [])) >= 2 else (0, 0, 0, 0), -0.1)
            break
        length += 1
        visited.add(T)
        if len(g["slots_at"].get(T, [])) >= 2 and unique_next(T, g) is not None:
            uniq = 1
            if learn:
                touch(q, tot, win, pre(T, g), 0.6)
        elif learn:
            touch(q, tot, win, pre(T, g) if len(g["slots_at"].get(T, [])) >= 2 else (0, 0, 0, 0), 0.2)
        v = v2
    return dict(len=length, uniq=uniq)


def rand_ep(g, rng, max_hops):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 1]
    if len(places) < 4:
        return dict(len=0, uniq=0)
    P0 = rng.choice(places)
    v = grab(P0, g, rng)
    if v is None:
        return dict(len=0, uniq=0)
    length = 0
    uniq = int(len(g["slots_at"].get(P0, [])) >= 2 and unique_next(P0, g) is not None)
    visited = {P0}
    for _ in range(max_hops):
        nbrs = [T for T in g["by_key"].get(v, set()) if T not in visited]
        if not nbrs:
            break
        T = rng.choice(nbrs)
        v2 = grab(T, g, rng)
        if v2 is None:
            break
        length += 1
        visited.add(T)
        if len(g["slots_at"].get(T, [])) >= 2 and unique_next(T, g) is not None:
            uniq = 1
        v = v2
    return dict(len=length, uniq=uniq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--max-hops", type=int, default=8)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    path = pick_corpus(args.corpus)
    lines = load_lines(path, args.bytes, 20, random.Random(args.seed))
    print(f"500 max-loose  corpus={path}  hops={args.max_hops}", flush=True)

    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng = random.Random(args.seed)
    n_ok = 0
    s = defaultdict(float)
    t0 = time.time()
    for i in range(args.steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, args.window, args.frame_max)
        if g is None:
            continue
        n_ok += 1
        eps = max(0.05, 0.5 * (1 - i / max(args.steps, 1)))
        h = episode(g, rng, q, tot, win, eps, 1, args.max_hops, True)
        b = rand_ep(g, rng, args.max_hops)
        for k, v in h.items():
            s["h_" + k] += v
        for k, v in b.items():
            s["r_" + k] += v
        if (i + 1) % args.log_every == 0:
            print(
                f"  step {i+1}/{args.steps}  len {s['h_len']/n_ok:.3f}/{s['r_len']/n_ok:.3f}  "
                f"uniq {s['h_uniq']/n_ok:.3f}/{s['r_uniq']/n_ok:.3f}  "
                f"lift_len {(s['h_len']-s['r_len'])/n_ok:.3f}  "
                f"lift_uniq {(s['h_uniq']-s['r_uniq'])/n_ok:.3f}  "
                f"q {len(q)}  {time.time()-t0:.0f}s",
                flush=True,
            )

    rec = dict(
        seed=args.seed,
        corpus=str(path),
        max_hops=args.max_hops,
        n_windows=n_ok,
        len_h=s["h_len"] / max(n_ok, 1),
        len_r=s["r_len"] / max(n_ok, 1),
        uniq_h=s["h_uniq"] / max(n_ok, 1),
        uniq_r=s["r_uniq"] / max(n_ok, 1),
        lift_len=(s["h_len"] - s["r_len"]) / max(n_ok, 1),
        lift_uniq=(s["h_uniq"] - s["r_uniq"]) / max(n_ok, 1),
        n_pre=len(q),
        elapsed_s=round(time.time() - t0, 1),
        note="max loose: any-slot pin, 8 hops, unique only bonus",
    )
    print("---- done ----", flush=True)
    print(f"lift_len {rec['lift_len']:.4f}  lift_uniq {rec['lift_uniq']:.4f}", flush=True)
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
