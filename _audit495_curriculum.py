"""495 NIGHT: curriculum narrow(k<=2) train → unique-only eval.

Train rewarding fan-in in {1,2}; freeze Q; eval unique (==1) vs random.
If lift on unique after narrow training > from-scratch unique → curriculum helps.

    python _audit495_curriculum.py --seed 1337
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
    build_window, load_lines, narrow_next, pick_by_q, pick_corpus, pre, touch, unique_next,
)

OUT = Path("results/_stage495_curriculum.json")


def hunt(g, rng, q, tot, win, eps, budget, mode: str, learn: bool):
    """mode: narrow2 | unique"""
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q, lambda x: pre(x, g), rng, eps if learn else 0.0)
        if P is None:
            break
        seen.add(P)
        k = pre(P, g)
        if mode == "unique":
            hit = unique_next(P, g) is not None
        else:
            hit = narrow_next(P, g, kmax=2) is not None
        if learn:
            touch(q, tot, win, k, 1.0 if hit else -0.08)
        if hit:
            return 1
    return 0


def rand_unique(g, rng, budget):
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
    ap.add_argument("--train-steps", type=int, default=2500)
    ap.add_argument("--eval-steps", type=int, default=1500)
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
    print(f"495 curriculum  corpus={path}", flush=True)

    # arm A: curriculum
    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng = random.Random(args.seed)
    t0 = time.time()
    n = hits = 0
    for i in range(args.train_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, args.window, args.frame_max)
        if g is None:
            continue
        n += 1
        eps = max(0.05, 0.5 * (1 - i / max(args.train_steps, 1)))
        hits += hunt(g, rng, q, tot, win, eps, args.budget, "narrow2", True)
        if (i + 1) % args.log_every == 0:
            print(f"  curric train {i+1} narrow_hit {hits/n:.3f} q {len(q)}", flush=True)

    rng_e = random.Random(args.seed + 77)
    n_e = hq = hr = 0
    for i in range(args.eval_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng_e, args.window, args.frame_max)
        if g is None:
            continue
        n_e += 1
        hq += hunt(g, rng_e, q, tot, win, 0.0, args.budget, "unique", False)
        hr += rand_unique(g, rng_e, args.budget)

    # arm B: from-scratch unique train then same eval stream
    q2, tot2, win2 = {}, defaultdict(int), defaultdict(float)
    rng2 = random.Random(args.seed)
    n2 = h2 = 0
    for i in range(args.train_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng2, args.window, args.frame_max)
        if g is None:
            continue
        n2 += 1
        eps = max(0.05, 0.5 * (1 - i / max(args.train_steps, 1)))
        h2 += hunt(g, rng2, q2, tot2, win2, eps, args.budget, "unique", True)

    rng_e2 = random.Random(args.seed + 77)
    n_e2 = hq2 = hr2 = 0
    for i in range(args.eval_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng_e2, args.window, args.frame_max)
        if g is None:
            continue
        n_e2 += 1
        hq2 += hunt(g, rng_e2, q2, tot2, win2, 0.0, args.budget, "unique", False)
        hr2 += rand_unique(g, rng_e2, args.budget)

    rec = dict(
        seed=args.seed,
        corpus=str(path),
        curriculum=dict(
            train_narrow_hit=hits / max(n, 1),
            eval_unique=hq / max(n_e, 1),
            eval_rand=hr / max(n_e, 1),
            lift=(hq - hr) / max(n_e, 1),
        ),
        scratch_unique=dict(
            train_hit=h2 / max(n2, 1),
            eval_unique=hq2 / max(n_e2, 1),
            eval_rand=hr2 / max(n_e2, 1),
            lift=(hq2 - hr2) / max(n_e2, 1),
        ),
        curric_minus_scratch=(hq - hr) / max(n_e, 1) - (hq2 - hr2) / max(n_e2, 1),
        elapsed_s=round(time.time() - t0, 1),
        note="narrow2 curriculum vs scratch unique; same eval RNG",
    )
    print("---- done ----", flush=True)
    print(f"curric lift {rec['curriculum']['lift']:.4f}  "
          f"scratch lift {rec['scratch_unique']['lift']:.4f}  "
          f"delta {rec['curric_minus_scratch']:.4f}", flush=True)
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
