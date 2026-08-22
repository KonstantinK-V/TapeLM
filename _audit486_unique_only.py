"""486 NIGHT freestyle: unique-ONLY hunt + budget-matched random.

After 485 showed unique exists under soft majority (~4–5% of soft places),
ask: can Q[pre] find unique when narrow is NOT rewarded?

Fair lift = hunter unique-hit rate vs random scan with SAME budget.

    python _audit486_unique_only.py --seed 1337 --steps 3000
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
    build_window,
    load_lines,
    pick_corpus,
    pick_by_q,
    pre,
    touch,
    unique_next,
)
from _audit440_compose import think_place as _think

OUT = Path("results/_stage486_unique_only.json")


def hunt_unique(g, rng, q_pre, tot, win, eps, budget: int):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return dict(tried=0, found=0, hopped=0)
    tried = found = hopped = 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q_pre, lambda x: pre(x, g), rng, eps)
        if P is None:
            break
        seen.add(P)
        tried += 1
        k = pre(P, g)
        nxt = unique_next(P, g)
        if nxt is None:
            touch(q_pre, tot, win, k, -0.08)
            continue
        found += 1
        touch(q_pre, tot, win, k, 1.0)
        R = nxt[0]
        if _think(list(g["slots_at"][R]), g["value"], rng) is not None:
            hopped += 1
            touch(q_pre, tot, win, k, 0.5)
        break
    return dict(tried=tried, found=found, hopped=hopped)


def random_unique_budget(g, rng, budget: int) -> int:
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
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--min-line", type=int, default=20)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=400)
    args = ap.parse_args()

    path = pick_corpus(args.corpus)
    rng0 = random.Random(args.seed)
    print(f"486 unique-ONLY  corpus={path}  steps={args.steps}", flush=True)
    lines = load_lines(path, args.bytes, args.min_line, rng0)
    print(f"line pool {len(lines)}", flush=True)

    q_pre, tot, win = {}, defaultdict(int), defaultdict(float)
    sum_h = defaultdict(float)
    n_ok = 0
    t0 = time.time()
    rng = random.Random(args.seed)
    out = Path(args.out)

    for i in range(args.steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, args.window, args.frame_max)
        if g is None:
            continue
        n_ok += 1
        eps = max(0.05, 0.5 * (1.0 - i / max(args.steps, 1)))
        h = hunt_unique(g, rng, q_pre, tot, win, eps, args.budget)
        for k, v in h.items():
            sum_h[k] += v
        sum_h["rand"] += random_unique_budget(g, rng, args.budget)
        if (i + 1) % args.log_every == 0:
            el = time.time() - t0
            print(
                f"  step {i+1}/{args.steps}  "
                f"hunt_u {sum_h['found']/n_ok:.3f}  "
                f"rand_u {sum_h['rand']/n_ok:.3f}  "
                f"lift {(sum_h['found']-sum_h['rand'])/n_ok:.3f}  "
                f"hop {sum_h['hopped']/n_ok:.3f}  "
                f"q_pre {len(q_pre)}  {el:.0f}s",
                flush=True,
            )
            _save(out, args, path, n_ok, el, sum_h, q_pre, partial=True)

    el = time.time() - t0
    rec = _save(out, args, path, n_ok, el, sum_h, q_pre, partial=False)
    print("---- done ----", flush=True)
    print(
        f"hunt_u {rec['hunt_u']:.4f}  rand_u {rec['rand_u']:.4f}  "
        f"lift {rec['lift']:.4f}  hopped {rec['hopped_per_win']:.4f}",
        flush=True,
    )
    return 0


def _save(out, args, path, n_ok, el, sum_h, q_pre, partial):
    rec = dict(
        seed=args.seed,
        corpus=str(path),
        steps=args.steps,
        n_windows=n_ok,
        elapsed_s=round(el, 1),
        partial=partial,
        hunt_u=sum_h["found"] / max(n_ok, 1),
        rand_u=sum_h["rand"] / max(n_ok, 1),
        lift=(sum_h["found"] - sum_h["rand"]) / max(n_ok, 1),
        hopped_per_win=sum_h["hopped"] / max(n_ok, 1),
        unique_found=sum_h["found"],
        n_pre_keys=len(q_pre),
        top_pre=[[list(k), v] for k, v in
                 sorted(q_pre.items(), key=lambda x: -x[1])[:10]],
        note="unique-only; fair budget-matched random; night freestyle",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if partial:
        prev[f"{args.seed}_partial"] = rec
    else:
        prev.pop(f"{args.seed}_partial", None)
        prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return rec


if __name__ == "__main__":
    raise SystemExit(main())
