"""488 NIGHT: after unique+soft-arrive, try a second hop (chain).

Builds on 486/487: find unique, soft-land on R, then fan-in step to T
(any width, or prefer narrow<=3). Q_mark LIVE/SOFT for continue.

Crazy question: does unique land unlock a wander chain better than random
unique land?

    python _audit488_chain.py --seed 1337 --steps 3000
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
    narrow_next,
    pick_by_q,
    pick_corpus,
    place_value,
    pre,
    touch,
    unique_next,
)

OUT = Path("results/_stage488_chain.json")


def soft_ok(P, g) -> bool:
    return place_value(P, g, 0.6)[0] is not None


def episode(g, rng, q_pre, tot_p, win_p, q_m, tot_m, win_m, eps, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    empty = dict(found=0, soft=0, chain=0, chain_soft=0, tried=0)
    if len(places) < 4:
        return empty
    seen = set()
    found = soft = chain = chain_soft = tried = 0
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
            touch(q_pre, tot_p, win_p, k, -0.08)
            continue
        found = 1
        touch(q_pre, tot_p, win_p, k, 0.5)
        R, v, _f = nxt
        # land
        hard = think_place(list(g["slots_at"][R]), g["value"], rng) is not None
        if hard or soft_ok(R, g):
            soft = 1
            touch(q_pre, tot_p, win_p, k, 0.4)
            touch(q_m, tot_m, win_m, "LIVE" if hard else "SOFT", 0.3)
        else:
            touch(q_pre, tot_p, win_p, k, 0.05)
            touch(q_m, tot_m, win_m, "DEAD", -0.1)
            break
        # second hop: prefer narrow from R, else any nbr of majority value
        sense = max(q_m.get("LIVE", 0.0), q_m.get("SOFT", 0.0))
        if not ((rng.random() < max(eps, 0.2)) or sense > 0):
            break
        nn = narrow_next(R, g, kmax=3)
        if nn is not None:
            T = rng.choice(list(nn[0]))
        else:
            vR, _ = place_value(R, g, 0.5)
            if vR is None:
                break
            nbrs = list(g["by_key"].get(vR, set()) - {P, R})
            if not nbrs:
                break
            T = rng.choice(nbrs)
        chain = 1
        th = think_place(list(g["slots_at"][T]), g["value"], rng) is not None
        if th or soft_ok(T, g):
            chain_soft = 1
            touch(q_m, tot_m, win_m, "LIVE" if th else "SOFT", 0.5)
            touch(q_pre, tot_p, win_p, k, 0.3)
        else:
            touch(q_m, tot_m, win_m, "DEAD", 0.05)
            touch(q_m, tot_m, win_m, "LIVE", -0.2)
        break
    return dict(found=found, soft=soft, chain=chain, chain_soft=chain_soft, tried=tried)


def random_ep(g, rng, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    z = dict(found=0, soft=0, chain=0, chain_soft=0)
    if not places:
        return z
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = rng.choice(pool)
        seen.add(P)
        nxt = unique_next(P, g)
        if nxt is None:
            continue
        z["found"] = 1
        R = nxt[0]
        hard = think_place(list(g["slots_at"][R]), g["value"], rng) is not None
        if not (hard or soft_ok(R, g)):
            return z
        z["soft"] = 1
        vR, _ = place_value(R, g, 0.5)
        if vR is None:
            return z
        nbrs = list(g["by_key"].get(vR, set()) - {P, R})
        if not nbrs:
            return z
        z["chain"] = 1
        T = rng.choice(nbrs)
        th = think_place(list(g["slots_at"][T]), g["value"], rng) is not None
        if th or soft_ok(T, g):
            z["chain_soft"] = 1
        return z
    return z


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
    print(f"488 chain  corpus={path}  steps={args.steps}", flush=True)
    lines = load_lines(path, args.bytes, args.min_line, rng0)
    print(f"line pool {len(lines)}", flush=True)

    q_pre, tot_p, win_p = {}, defaultdict(int), defaultdict(float)
    q_m, tot_m, win_m = {}, defaultdict(int), defaultdict(float)
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
        h = episode(g, rng, q_pre, tot_p, win_p, q_m, tot_m, win_m, eps, args.budget)
        for k, v in h.items():
            s["h_" + k] += v
        b = random_ep(g, rng, args.budget)
        for k, v in b.items():
            s["r_" + k] += v
        if (i + 1) % args.log_every == 0:
            el = time.time() - t0
            print(
                f"  step {i+1}/{args.steps}  "
                f"find {s['h_found']/n_ok:.3f}/{s['r_found']/n_ok:.3f}  "
                f"soft {s['h_soft']/n_ok:.3f}/{s['r_soft']/n_ok:.3f}  "
                f"chain {s['h_chain_soft']/n_ok:.3f}/{s['r_chain_soft']/n_ok:.3f}  "
                f"lift_chain {(s['h_chain_soft']-s['r_chain_soft'])/n_ok:.3f}  "
                f"mark { {k: round(v,2) for k,v in q_m.items()} }  {el:.0f}s",
                flush=True,
            )
            _save(out, args, path, n_ok, el, s, q_pre, q_m, True)

    el = time.time() - t0
    rec = _save(out, args, path, n_ok, el, s, q_pre, q_m, False)
    print("---- done ----", flush=True)
    print(
        f"lift_find {rec['lift_find']:.4f}  lift_soft {rec['lift_soft']:.4f}  "
        f"lift_chain {rec['lift_chain']:.4f}  q_mark {q_m}",
        flush=True,
    )
    return 0


def _save(out, args, path, n_ok, el, s, q_pre, q_m, partial):
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
        chain_h=s["h_chain_soft"] / max(n_ok, 1),
        chain_r=s["r_chain_soft"] / max(n_ok, 1),
        lift_find=(s["h_found"] - s["r_found"]) / max(n_ok, 1),
        lift_soft=(s["h_soft"] - s["r_soft"]) / max(n_ok, 1),
        lift_chain=(s["h_chain_soft"] - s["r_chain_soft"]) / max(n_ok, 1),
        q_mark={k: round(v, 4) for k, v in q_m.items()},
        n_pre_keys=len(q_pre),
        note="unique->soft land->2nd hop; night freestyle",
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
