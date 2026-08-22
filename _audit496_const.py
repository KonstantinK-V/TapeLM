"""496 NIGHT: soft majority (0.6) vs true const (0.999) unique hunt.

Same Q machine; only unique_next min_frac differs. If soft carries all lift
and const is dead → signal is soft-concession, not place constancy.

    python _audit496_const.py --seed 1337 --steps 2500
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

OUT = Path("results/_stage496_const.json")


def hunt(g, rng, q, tot, win, eps, budget, frac):
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
        hit = unique_next(P, g, min_frac=frac) is not None
        touch(q, tot, win, k, 1.0 if hit else -0.08)
        if hit:
            return 1
    return 0


def rand_find(g, rng, budget, frac):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if not places:
        return 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = rng.choice(pool)
        seen.add(P)
        if unique_next(P, g, min_frac=frac) is not None:
            return 1
    return 0


def run_arm(lines, seed, steps, window, fm, budget, log_every, frac):
    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng = random.Random(seed)
    n = h = r = 0
    t0 = time.time()
    tag = f"frac{frac}"
    for i in range(steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, window, fm)
        if g is None:
            continue
        n += 1
        eps = max(0.05, 0.5 * (1 - i / max(steps, 1)))
        h += hunt(g, rng, q, tot, win, eps, budget, frac)
        r += rand_find(g, rng, budget, frac)
        if (i + 1) % log_every == 0:
            print(f"  {tag} {i+1}/{steps} hunt {h/n:.3f} rand {r/n:.3f} "
                  f"lift {(h-r)/n:.3f}  {time.time()-t0:.0f}s", flush=True)
    return dict(frac=frac, hunt=h / max(n, 1), rand=r / max(n, 1),
                lift=(h - r) / max(n, 1), n_windows=n,
                elapsed_s=round(time.time() - t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    path = pick_corpus(args.corpus)
    lines = load_lines(path, args.bytes, 20, random.Random(args.seed))
    print(f"496 const ablation  corpus={path}", flush=True)
    soft = run_arm(lines, args.seed, args.steps, args.window, args.frame_max,
                   args.budget, args.log_every, 0.6)
    hard = run_arm(lines, args.seed + 3, args.steps, args.window, args.frame_max,
                   args.budget, args.log_every, 0.999)
    rec = dict(seed=args.seed, corpus=str(path), soft=soft, const=hard,
               note="soft0.6 vs const0.999 unique hunt")
    print(f"soft lift {soft['lift']:.4f}  const lift {hard['lift']:.4f}", flush=True)
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
