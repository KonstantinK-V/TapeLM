"""491 NIGHT CONTROL: shuffle values inside the window — does lift die?

If 486-style unique-hunt still beats random after shuffling fillers across
slots, the signal is fake (gaming pre counts). If lift collapses → structure
on the real tape mattered.

    python _audit491_shuffle.py --seed 1337 --steps 2500
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

OUT = Path("results/_stage491_shuffle.json")


def shuffle_values(g, rng):
    g2 = dict(g)
    vals = list(g["value"])
    rng.shuffle(vals)
    g2["value"] = vals
    # rebuild nothing else — by_key stays on frame keys; unique_next uses values
    return g2


def hunt(g, rng, q, tot, win, eps, budget, learn):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q, lambda x: pre(x, g), rng, eps if learn else max(eps, 0.05))
        if P is None:
            break
        seen.add(P)
        k = pre(P, g)
        hit = unique_next(P, g) is not None
        if learn:
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


def run_arm(lines, seed, steps, window, frame_max, budget, log_every, shuffled):
    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng = random.Random(seed)
    n_ok = h = r = 0
    t0 = time.time()
    tag = "SHUF" if shuffled else "REAL"
    for i in range(steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, window, frame_max)
        if g is None:
            continue
        if shuffled:
            g = shuffle_values(g, rng)
        n_ok += 1
        eps = max(0.05, 0.5 * (1 - i / max(steps, 1)))
        h += hunt(g, rng, q, tot, win, eps, budget, True)
        r += rand_find(g, rng, budget)
        if (i + 1) % log_every == 0:
            print(
                f"  {tag} {i+1}/{steps}  hunt {h/n_ok:.3f}  rand {r/n_ok:.3f}  "
                f"lift {(h-r)/n_ok:.3f}  q {len(q)}  {time.time()-t0:.0f}s",
                flush=True,
            )
    return dict(
        shuffled=shuffled,
        hunt=h / max(n_ok, 1),
        rand=r / max(n_ok, 1),
        lift=(h - r) / max(n_ok, 1),
        n_windows=n_ok,
        n_pre=len(q),
        elapsed_s=round(time.time() - t0, 1),
    )


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
    print(f"491 shuffle control  corpus={path}  steps={args.steps}", flush=True)
    real = run_arm(lines, args.seed, args.steps, args.window, args.frame_max,
                   args.budget, args.log_every, False)
    shuf = run_arm(lines, args.seed + 999, args.steps, args.window, args.frame_max,
                   args.budget, args.log_every, True)
    rec = dict(
        seed=args.seed,
        corpus=str(path),
        real=real,
        shuffled=shuf,
        lift_drop=real["lift"] - shuf["lift"],
        note="if shuffled lift ~= real lift → fake; drop>0 → tape structure",
    )
    print("---- done ----", flush=True)
    print(f"REAL lift {real['lift']:.4f}  SHUF lift {shuf['lift']:.4f}  "
          f"drop {rec['lift_drop']:.4f}", flush=True)
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
