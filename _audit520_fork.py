"""520: W on a fork, not v's frame (517 STOP)."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _tape_frames as tframes
from _audit511_ring import cheap_rec, mentions
from _audit517_window import comps, pick_corpus, score_w
from _audit518_reldf import pct_band

OUT = Path("results/_stage520_fork.json")


def graph(lines, frame_max, min_fillers):
    keep, toks, owner = tframes.frame_keep(lines, frame_max, min_fillers)
    if not keep:
        return None
    value, keys = [], []
    for (w, left, right), ps in keep:
        ks = tuple(x for x in list(left) + list(right) if x)
        for i in ps:
            value.append(toks[i])
            keys.append(ks)
    n = len(value)
    df = Counter()
    for s in range(n):
        df[value[s]] += 1
        for k in keys[s]:
            df[k] += 1
    return dict(n=n, value=value, keys=keys, df=df)


def eval_fork(g, by, vs, rng, cap=20):
    n = n_w = n_5 = n_r = 0
    cache = {}
    for v in vs:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        sl = sl[:cap]
        for i, s in enumerate(sl[:3]):
            rest_slots = sl[:i] + sl[i + 1:]
            held = set(comps(g, s, v))
            if not held:
                continue
            saved = by[v]
            by[v] = rest_slots
            cache.pop(v, None)
            rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
            by[v] = saved
            if len(rec) < 3:
                continue
            h1 = rec[0]
            rest = rec[1:]
            W = [h1]
            sc = [score_w(g, by, c, W, cache) for c in rest]
            if max(sc) <= min(sc):
                continue
            pick_w = sorted(rest, key=lambda c: (-score_w(g, by, c, W, cache), g["df"][c]))[0]
            pick_5 = rest[0]
            pick_r = rng.choice(rest)
            n += 1
            n_w += int(pick_w in held)
            n_5 += int(pick_5 in held)
            n_r += int(pick_r in held)
    d = max(n, 1)
    return dict(n=n, hit_w=n_w / d, hit_511=n_5 / d, hit_rnd=n_r / d,
                d511=(n_w - n_5) / d, drnd=(n_w - n_r) / d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    L = args.window_lines
    if L < len(pool):
        s0 = rng.randrange(len(pool) - L + 1)
        lines = pool[s0:s0 + L]
    else:
        lines = pool
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        print("no tape")
        return 1
    by = mentions(g)
    mid, high, p25, p75 = pct_band(g, by)
    mid_rep = eval_fork(g, by, mid, random.Random(args.seed + 1))
    high_rep = eval_fork(g, by, high, random.Random(args.seed + 2))
    void = mid_rep["n"] < 40
    gate = (not void) and (mid_rep["d511"] > 0.05) and (mid_rep["drnd"] > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               p25=p25, p75=p75, mid=mid_rep, high=high_rep,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  p25–75 {p25}-{p75}")
    print(f"MID  forks {mid_rep['n']}  W {mid_rep['hit_w']:.3f}  "
          f"511 {mid_rep['hit_511']:.3f}  rnd {mid_rep['hit_rnd']:.3f}  "
          f"Δ511 {mid_rep['d511']:+.3f}  Δrnd {mid_rep['drnd']:+.3f}")
    print(f"HIGH forks {high_rep['n']}  W {high_rep['hit_w']:.3f}  "
          f"511 {high_rep['hit_511']:.3f}  Δ511 {high_rep['d511']:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: W almost never splits rest. No fork exam.")
    elif gate:
        print("\nGO FORK: on real forks, W beats 511 and random. Per-hop + has a teacher.")
    else:
        print("\nSTOP: even on forks W is not a better cutter. Do not pay + for W-hop yet.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
