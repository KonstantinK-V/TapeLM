"""487 NIGHT: unique hunt + soft arrive (the hop that 486 barely got).

486 found unique (~2x random) but hard think_place hop ~5%.
Here arrive is softened: majority>=0.6 on R counts as SOFT hop;
hard think_place still tracked separately.

Rewards Q[pre] for find / soft / hard. Fair random: same budget find,
then soft-arrive on that R.

    python _audit487_arrive.py --seed 1337 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place
from _audit485_hunt import (
    build_window,
    load_lines,
    pick_by_q,
    pick_corpus,
    place_value,
    pre,
    touch,
    unique_next,
)

OUT = Path("results/_stage487_arrive.json")


def soft_ok(P, g, frac=0.6) -> bool:
    v, f = place_value(P, g, frac)
    return v is not None


def hunt(g, rng, q, tot, win, eps, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return dict(tried=0, found=0, soft=0, hard=0)
    tried = found = soft = hard = 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q, lambda x: pre(x, g), rng, eps)
        if P is None:
            break
        seen.add(P)
        tried += 1
        k = pre(P, g)
        nxt = unique_next(P, g)
        if nxt is None:
            touch(q, tot, win, k, -0.08)
            continue
        found += 1
        touch(q, tot, win, k, 0.6)
        R = nxt[0]
        if think_place(list(g["slots_at"][R]), g["value"], rng) is not None:
            hard += 1
            soft += 1
            touch(q, tot, win, k, 0.8)
        elif soft_ok(R, g):
            soft += 1
            touch(q, tot, win, k, 0.4)
        else:
            touch(q, tot, win, k, 0.05)
        break
    return dict(tried=tried, found=found, soft=soft, hard=hard)


def random_base(g, rng, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if not places:
        return dict(found=0, soft=0, hard=0)
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = rng.choice(pool)
        seen.add(P)
        nxt = unique_next(P, g)
        if nxt is None:
            continue
        R = nxt[0]
        hard = int(think_place(list(g["slots_at"][R]), g["value"], rng) is not None)
        soft = int(hard or soft_ok(R, g))
        return dict(found=1, soft=soft, hard=hard)
    return dict(found=0, soft=0, hard=0)


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
    print(f"487 arrive  corpus={path}  steps={args.steps}", flush=True)
    lines = load_lines(path, args.bytes, args.min_line, rng0)
    print(f"line pool {len(lines)}", flush=True)

    q, tot, win = {}, defaultdict(int), defaultdict(float)
    s = defaultdict(float)
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
        h = hunt(g, rng, q, tot, win, eps, args.budget)
        for k, v in h.items():
            s["h_" + k] += v
        b = random_base(g, rng, args.budget)
        for k, v in b.items():
            s["r_" + k] += v
        if (i + 1) % args.log_every == 0:
            el = time.time() - t0
            print(
                f"  step {i+1}/{args.steps}  "
                f"find {s['h_found']/n_ok:.3f}/{s['r_found']/n_ok:.3f}  "
                f"soft {s['h_soft']/n_ok:.3f}/{s['r_soft']/n_ok:.3f}  "
                f"hard {s['h_hard']/n_ok:.3f}/{s['r_hard']/n_ok:.3f}  "
                f"lift_find {(s['h_found']-s['r_found'])/n_ok:.3f}  "
                f"lift_soft {(s['h_soft']-s['r_soft'])/n_ok:.3f}  "
                f"q {len(q)}  {el:.0f}s",
                flush=True,
            )
            _save(out, args, path, n_ok, el, s, q, True)

    el = time.time() - t0
    rec = _save(out, args, path, n_ok, el, s, q, False)
    print("---- done ----", flush=True)
    print(
        f"find {rec['find_h']:.4f}/{rec['find_r']:.4f}  "
        f"soft {rec['soft_h']:.4f}/{rec['soft_r']:.4f}  "
        f"hard {rec['hard_h']:.4f}/{rec['hard_r']:.4f}  "
        f"lift_soft {rec['lift_soft']:.4f}",
        flush=True,
    )
    return 0


def _save(out, args, path, n_ok, el, s, q, partial):
    rec = dict(
        seed=args.seed,
        corpus=str(path),
        steps=args.steps,
        n_windows=n_ok,
        elapsed_s=round(el, 1),
        partial=partial,
        find_h=s["h_found"] / max(n_ok, 1),
        find_r=s["r_found"] / max(n_ok, 1),
        soft_h=s["h_soft"] / max(n_ok, 1),
        soft_r=s["r_soft"] / max(n_ok, 1),
        hard_h=s["h_hard"] / max(n_ok, 1),
        hard_r=s["r_hard"] / max(n_ok, 1),
        lift_find=(s["h_found"] - s["r_found"]) / max(n_ok, 1),
        lift_soft=(s["h_soft"] - s["r_soft"]) / max(n_ok, 1),
        lift_hard=(s["h_hard"] - s["r_hard"]) / max(n_ok, 1),
        n_pre_keys=len(q),
        top_pre=[[list(k), v] for k, v in sorted(q.items(), key=lambda x: -x[1])[:10]],
        note="soft arrive after unique; hard pin tracked; not a ladder gate",
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
