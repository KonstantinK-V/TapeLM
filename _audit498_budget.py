"""498 NIGHT: budget sweep 4/8/16/32 — diminishing returns of hunt?

    python _audit498_budget.py --seed 1337 --steps 2000
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

OUT = Path("results/_stage498_budget.json")


def hunt(g, rng, q, tot, win, eps, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q, lambda x: pre(x, g), rng, eps)
        if P is None:
            break
        seen.add(P)
        k = pre(P, g)
        hit = unique_next(P, g) is not None
        touch(q, tot, win, k, 1.0 if hit else -0.08)
        if hit:
            return 1
    return 0


def rand_find(g, rng, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if not places:
        return 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = rng.choice(pool)
        seen.add(P)
        if unique_next(P, g) is not None:
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    path = pick_corpus(args.corpus)
    lines = load_lines(path, args.bytes, 20, random.Random(args.seed))
    print(f"498 budget sweep  corpus={path}", flush=True)
    results = {}
    for B in (4, 8, 16, 32):
        q, tot, win = {}, defaultdict(int), defaultdict(float)
        rng = random.Random(args.seed + B)
        n = h = r = 0
        t0 = time.time()
        for i in range(args.steps):
            if (i + 1) % 100 == 0:
                tframes._KEEP_MEMO.clear()
            g = build_window(lines, rng, args.window, args.frame_max)
            if g is None:
                continue
            n += 1
            eps = max(0.05, 0.5 * (1 - i / max(args.steps, 1)))
            h += hunt(g, rng, q, tot, win, eps, B)
            r += rand_find(g, rng, B)
            if (i + 1) % args.log_every == 0:
                print(f"  B={B} {i+1}/{args.steps} hunt {h/n:.3f} rand {r/n:.3f} "
                      f"lift {(h-r)/n:.3f}", flush=True)
        results[str(B)] = dict(
            hunt=h / max(n, 1), rand=r / max(n, 1), lift=(h - r) / max(n, 1),
            n_windows=n, elapsed_s=round(time.time() - t0, 1),
        )
        print(f"B={B} lift {results[str(B)]['lift']:.4f}", flush=True)

    rec = dict(seed=args.seed, corpus=str(path), by_budget=results,
               note="lift vs budget size")
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
