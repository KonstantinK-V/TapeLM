"""499 NIGHT FREE-SWIM: max-soft pin + unrestricted hops.

Pin: land if place has any slots AND majority frac>=0.5 (or pick random
slot value). No unique required to move.
Hop: any by_key neighbor of landed value.
Chain up to L hops; reward chain length. Fair random walk baseline.

Also track how often a free path STUMBLES on unique_next (bonus signal).

    python _audit499_free_swim.py --seed 1337 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place
from _audit485_hunt import (
    build_window, load_lines, pick_by_q, pick_corpus, place_value, pre, touch,
    unique_next,
)

OUT = Path("results/_stage499_free_swim.json")


def land_value(P, g, rng, mode: str):
    """mode: hard | soft05 | any_slot"""
    sl = list(g["slots_at"][P])
    if not sl:
        return None, "MISS"
    if mode == "hard":
        pin = think_place(sl, g["value"], rng)
        if pin is None:
            return None, "MISS"
        return g["value"][pin], "HARD"
    if mode == "soft05":
        v, f = place_value(P, g, 0.5)
        if v is None:
            return None, "MISS"
        return v, "SOFT"
    # any_slot: random filler at the place
    i = rng.choice(sl)
    return g["value"][i], "ANY"


def walk(g, rng, q_pre, tot_p, win_p, q_m, tot_m, win_m, eps, budget, max_hops, mode, learn):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return dict(start=0, hops=0, unique_stumble=0, len=0)
    seen_try = set()
    start = hops = uniq = length = 0
    for _ in range(budget):
        pool = [P for P in places if P not in seen_try] or places
        P = pick_by_q(pool, q_pre, lambda x: pre(x, g), rng, eps if learn else 0.05)
        if P is None:
            break
        seen_try.add(P)
        k = pre(P, g)
        v, kind = land_value(P, g, rng, mode)
        if v is None:
            if learn:
                touch(q_pre, tot_p, win_p, k, -0.05)
                touch(q_m, tot_m, win_m, "MISS", -0.1)
            continue
        start = 1
        if learn:
            touch(q_pre, tot_p, win_p, k, 0.2)
            touch(q_m, tot_m, win_m, kind, 0.15)
        if unique_next(P, g) is not None:
            uniq = 1
        cur = P
        visited = {P}
        for _h in range(max_hops):
            sense = max((q_m.get(x, 0.0) for x in ("HARD", "SOFT", "ANY")), default=0.0)
            if learn and not ((rng.random() < max(eps, 0.15)) or sense > -0.05):
                break
            nbrs = list(g["by_key"].get(v, set()) - visited)
            if not nbrs:
                break
            # prefer Q on pre of candidates if learning
            T = pick_by_q(nbrs, q_pre, lambda x: pre(x, g), rng, eps if learn else 0.2)
            if T is None:
                T = rng.choice(nbrs)
            v2, kind2 = land_value(T, g, rng, mode)
            hops += 1
            if v2 is None:
                if learn:
                    touch(q_m, tot_m, win_m, "MISS", -0.15)
                break
            length += 1
            if learn:
                touch(q_m, tot_m, win_m, kind2, 0.35)
                touch(q_pre, tot_p, win_p, pre(T, g), 0.15)
            if unique_next(T, g) is not None:
                uniq = 1
                if learn:
                    touch(q_pre, tot_p, win_p, pre(T, g), 0.4)
            visited.add(T)
            cur, v = T, v2
        break
    return dict(start=start, hops=hops, unique_stumble=uniq, len=length)


def rand_walk(g, rng, budget, max_hops, mode):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    z = dict(start=0, hops=0, unique_stumble=0, len=0)
    if len(places) < 4:
        return z
    for _ in range(budget):
        P = rng.choice(places)
        v, _k = land_value(P, g, rng, mode)
        if v is None:
            continue
        z["start"] = 1
        if unique_next(P, g) is not None:
            z["unique_stumble"] = 1
        visited = {P}
        for _h in range(max_hops):
            nbrs = list(g["by_key"].get(v, set()) - visited)
            if not nbrs:
                break
            T = rng.choice(nbrs)
            v2, _ = land_value(T, g, rng, mode)
            z["hops"] += 1
            if v2 is None:
                break
            z["len"] += 1
            if unique_next(T, g) is not None:
                z["unique_stumble"] = 1
            visited.add(T)
            v = v2
        break
    return z


def run_mode(lines, seed, steps, window, fm, budget, max_hops, mode, log_every):
    q_pre, tot_p, win_p = {}, defaultdict(int), defaultdict(float)
    q_m, tot_m, win_m = {}, defaultdict(int), defaultdict(float)
    s = defaultdict(float)
    rng = random.Random(seed)
    n_ok = 0
    t0 = time.time()
    for i in range(steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, window, fm)
        if g is None:
            continue
        n_ok += 1
        eps = max(0.05, 0.45 * (1 - i / max(steps, 1)))
        h = walk(g, rng, q_pre, tot_p, win_p, q_m, tot_m, win_m, eps, budget,
                 max_hops, mode, True)
        for k, v in h.items():
            s["h_" + k] += v
        b = rand_walk(g, rng, budget, max_hops, mode)
        for k, v in b.items():
            s["r_" + k] += v
        if (i + 1) % log_every == 0:
            print(
                f"  {mode} {i+1}/{steps}  start {s['h_start']/n_ok:.3f}/{s['r_start']/n_ok:.3f}  "
                f"len {s['h_len']/n_ok:.3f}/{s['r_len']/n_ok:.3f}  "
                f"uniq {s['h_unique_stumble']/n_ok:.3f}/{s['r_unique_stumble']/n_ok:.3f}  "
                f"lift_len {(s['h_len']-s['r_len'])/n_ok:.3f}  "
                f"mark {{{', '.join(f'{a}:{b:.2f}' for a,b in q_m.items())}}}  "
                f"{time.time()-t0:.0f}s",
                flush=True,
            )
    return dict(
        mode=mode,
        start_h=s["h_start"] / max(n_ok, 1),
        start_r=s["r_start"] / max(n_ok, 1),
        len_h=s["h_len"] / max(n_ok, 1),
        len_r=s["r_len"] / max(n_ok, 1),
        uniq_h=s["h_unique_stumble"] / max(n_ok, 1),
        uniq_r=s["r_unique_stumble"] / max(n_ok, 1),
        lift_len=(s["h_len"] - s["r_len"]) / max(n_ok, 1),
        lift_uniq=(s["h_unique_stumble"] - s["r_unique_stumble"]) / max(n_ok, 1),
        q_mark={k: round(v, 4) for k, v in q_m.items()},
        n_windows=n_ok,
        elapsed_s=round(time.time() - t0, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--max-hops", type=int, default=5)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--modes", default="hard,soft05,any_slot")
    args = ap.parse_args()

    path = pick_corpus(args.corpus)
    lines = load_lines(path, args.bytes, 20, random.Random(args.seed))
    print(f"499 free-swim  corpus={path}  hops={args.max_hops}", flush=True)
    arms = {}
    for i, mode in enumerate(args.modes.split(",")):
        mode = mode.strip()
        if not mode:
            continue
        print(f"-- mode {mode} --", flush=True)
        arms[mode] = run_mode(
            lines, args.seed + 17 * i, args.steps, args.window, args.frame_max,
            args.budget, args.max_hops, mode, args.log_every,
        )
        print(f"  lift_len {arms[mode]['lift_len']:.4f}  "
              f"lift_uniq {arms[mode]['lift_uniq']:.4f}", flush=True)

    rec = dict(seed=args.seed, corpus=str(path), max_hops=args.max_hops, arms=arms,
               note="free swim: hard vs soft05 vs any_slot pin; no unique gate")
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
