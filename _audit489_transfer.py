"""489 NIGHT: does unique-hunt Q transfer to a foreign RNG stream?

Train Q[pre] on train_seed windows, FREEZE, eval find-rate on eval_seed
windows (no updates) vs budget-matched random. If lift dies → memorized
window quirks. If lift holds → cheap transferable hunt habit.

    python _audit489_transfer.py --train-seed 1337 --eval-seed 2024
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

OUT = Path("results/_stage489_transfer.json")


def hunt(g, rng, q, tot, win, eps, budget, learn: bool):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seed", type=int, default=1337)
    ap.add_argument("--eval-seed", type=int, default=2024)
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
    lines = load_lines(path, args.bytes, 20, random.Random(args.train_seed))
    print(f"489 transfer train={args.train_seed} eval={args.eval_seed} "
          f"pool={len(lines)}", flush=True)

    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng = random.Random(args.train_seed)
    t0 = time.time()
    n_tr = hits = 0
    for i in range(args.train_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, args.window, args.frame_max)
        if g is None:
            continue
        n_tr += 1
        eps = max(0.05, 0.5 * (1 - i / max(args.train_steps, 1)))
        hits += hunt(g, rng, q, tot, win, eps, args.budget, True)
        if (i + 1) % args.log_every == 0:
            print(f"  train {i+1}  hit {hits/max(n_tr,1):.3f}  q {len(q)}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    # freeze eval
    rng_e = random.Random(args.eval_seed)
    n_ev = h_q = h_r = 0
    for i in range(args.eval_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng_e, args.window, args.frame_max)
        if g is None:
            continue
        n_ev += 1
        h_q += hunt(g, rng_e, q, tot, win, 0.0, args.budget, False)
        h_r += rand_find(g, rng_e, args.budget)
        if (i + 1) % args.log_every == 0:
            print(
                f"  eval {i+1}  q {h_q/n_ev:.3f}  rand {h_r/n_ev:.3f}  "
                f"lift {(h_q-h_r)/n_ev:.3f}",
                flush=True,
            )

    rec = dict(
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        corpus=str(path),
        train_hit=hits / max(n_tr, 1),
        eval_q=h_q / max(n_ev, 1),
        eval_rand=h_r / max(n_ev, 1),
        lift=(h_q - h_r) / max(n_ev, 1),
        n_pre=len(q),
        top_pre=[[list(k), v] for k, v in sorted(q.items(), key=lambda x: -x[1])[:10]],
        elapsed_s=round(time.time() - t0, 1),
        note="frozen Q transfer across RNG streams; same line pool",
    )
    print("---- done ----", flush=True)
    print(f"eval q {rec['eval_q']:.4f} rand {rec['eval_rand']:.4f} "
          f"lift {rec['lift']:.4f}", flush=True)
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.train_seed}->{args.eval_seed}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
