"""493 NIGHT: unique-hunt on wiki (or given corpus) — same machine as 486.

Does soft-majority unique + Q lift show up off TinyStories?

    python _audit493_wiki.py --seed 1337 --steps 2500
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place as _think
from _audit485_hunt import (
    build_window, load_lines, pick_by_q, pre, touch, unique_next,
)

OUT = Path("results/_stage493_wiki.json")
WIKI = Path("data/_wikitext103_train.txt")


def hunt(g, rng, q, tot, win, eps, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return dict(found=0, hopped=0)
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q, lambda x: pre(x, g), rng, eps)
        if P is None:
            break
        seen.add(P)
        k = pre(P, g)
        nxt = unique_next(P, g)
        if nxt is None:
            touch(q, tot, win, k, -0.08)
            continue
        touch(q, tot, win, k, 1.0)
        hopped = int(_think(list(g["slots_at"][nxt[0]]), g["value"], rng) is not None)
        return dict(found=1, hopped=hopped)
    return dict(found=0, hopped=0)


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
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--min-line", type=int, default=80)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    path = Path(args.corpus) if args.corpus else WIKI
    if not path.exists():
        raise SystemExit(f"no corpus {path}")
    lines = load_lines(path, args.bytes, args.min_line, random.Random(args.seed))
    print(f"493 wiki-hunt  corpus={path}  lines={len(lines)}  steps={args.steps}",
          flush=True)

    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng = random.Random(args.seed)
    n_ok = found = hopped = rand = 0
    t0 = time.time()
    for i in range(args.steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, args.window, args.frame_max)
        if g is None:
            continue
        n_ok += 1
        eps = max(0.05, 0.5 * (1 - i / max(args.steps, 1)))
        h = hunt(g, rng, q, tot, win, eps, args.budget)
        found += h["found"]
        hopped += h["hopped"]
        rand += rand_find(g, rng, args.budget)
        if (i + 1) % args.log_every == 0:
            print(
                f"  step {i+1}/{args.steps}  hunt {found/n_ok:.3f}  "
                f"rand {rand/n_ok:.3f}  lift {(found-rand)/n_ok:.3f}  "
                f"hop {hopped/n_ok:.3f}  q {len(q)}  {time.time()-t0:.0f}s",
                flush=True,
            )

    rec = dict(
        seed=args.seed,
        corpus=str(path),
        n_windows=n_ok,
        hunt=found / max(n_ok, 1),
        rand=rand / max(n_ok, 1),
        lift=(found - rand) / max(n_ok, 1),
        hop=hopped / max(n_ok, 1),
        n_pre=len(q),
        elapsed_s=round(time.time() - t0, 1),
        note="486 machine on wiki; cross-corpus unique hunt",
    )
    print("---- done ----", flush=True)
    print(f"hunt {rec['hunt']:.4f} rand {rec['rand']:.4f} lift {rec['lift']:.4f}",
          flush=True)
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
