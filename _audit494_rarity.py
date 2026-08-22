"""494 NIGHT: add rarity of majority value into pre — does lift grow?

pre = (n_slots, n_keys, const, softmaj, rarity_bin)
rarity_bin = digitize log1p(df) of majority filler over the window.

    python _audit494_rarity.py --seed 1337 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit485_hunt import (
    build_window, load_lines, pick_by_q, pick_corpus, place_value, touch, unique_next,
)

OUT = Path("results/_stage494_rarity.json")


def pre_rarity(P, g, df):
    sl = g["slots_at"][P]
    vs = [g["value"][i] for i in sl]
    n_s = len(sl)
    n_k = len({k for i in sl for k in g["keys"][i]})
    v, c = Counter(vs).most_common(1)[0]
    frac = c / max(len(vs), 1)
    rb = min(int(math.log1p(df.get(v, 1))), 8)
    return (min(n_s, 12), min(n_k, 20), int(frac >= 0.999), int(frac >= 0.6), rb)


def df_window(g):
    return Counter(g["value"])


def hunt(g, rng, q, tot, win, eps, budget, use_rarity):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return 0
    df = df_window(g) if use_rarity else None

    def keyfn(P):
        if use_rarity:
            return pre_rarity(P, g, df)
        # baseline 485-style without rarity
        sl = g["slots_at"][P]
        vs = [g["value"][i] for i in sl]
        n_s, n_k = len(sl), len({k for i in sl for k in g["keys"][i]})
        maj = Counter(vs).most_common(1)[0][1] / max(len(vs), 1)
        return (min(n_s, 12), min(n_k, 20), int(maj >= 0.999), int(maj >= 0.6))

    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q, keyfn, rng, eps)
        if P is None:
            break
        seen.add(P)
        k = keyfn(P)
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


def run_arm(lines, seed, steps, window, fm, budget, log_every, use_rarity):
    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng = random.Random(seed)
    n = h = r = 0
    t0 = time.time()
    tag = "RARITY" if use_rarity else "BASE"
    for i in range(steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, window, fm)
        if g is None:
            continue
        n += 1
        eps = max(0.05, 0.5 * (1 - i / max(steps, 1)))
        h += hunt(g, rng, q, tot, win, eps, budget, use_rarity)
        r += rand_find(g, rng, budget)
        if (i + 1) % log_every == 0:
            print(f"  {tag} {i+1}/{steps}  hunt {h/n:.3f} rand {r/n:.3f} "
                  f"lift {(h-r)/n:.3f} q {len(q)}  {time.time()-t0:.0f}s", flush=True)
    return dict(hunt=h / max(n, 1), rand=r / max(n, 1), lift=(h - r) / max(n, 1),
                n_windows=n, n_pre=len(q), elapsed_s=round(time.time() - t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=3000)
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
    print(f"494 rarity  corpus={path}", flush=True)
    base = run_arm(lines, args.seed, args.steps, args.window, args.frame_max,
                   args.budget, args.log_every, False)
    rar = run_arm(lines, args.seed + 1, args.steps, args.window, args.frame_max,
                  args.budget, args.log_every, True)
    rec = dict(seed=args.seed, corpus=str(path), base=base, rarity=rar,
               delta_lift=rar["lift"] - base["lift"],
               note="rarity bin in pre vs base pre")
    print(f"BASE lift {base['lift']:.4f}  RARITY {rar['lift']:.4f}  "
          f"delta {rec['delta_lift']:.4f}", flush=True)
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
