"""567: two candidates — try one, DEAD → try the other.

0 / 1 / 3+ as 557. Only two cand: first miss → second.
VOID  n_two < 20
GATE  first miss → second tried; hit on two-cand rows > 0.05
saved reported, not sole gate.

    python _check567_try2.py
    python _audit567_try2.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band

OUT = Path("results/_stage567_try2.json")


def stand(g, by, v, env_m, cap=8):
    sl = list(by.get(v, []))
    if len(sl) < 2:
        return None, set()
    scored = []
    for t in sl[: max(cap, 1)]:
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), t, fr))
    if not scored:
        return None, set()
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], scored[0][2]


def hop_hit(g, by, addr, env_m, held):
    t2, fr2 = stand(g, by, addr, env_m)
    return bool(t2 is not None and held in fr2)


def rank_cand(g, by, cand, env_m):
    scored = []
    for a in cand:
        _t, fr = stand(g, by, a, env_m)
        ov = len((fr or set()) & env_m)
        scored.append((-ov, a))
    scored.sort()
    return [a for _ov, a in scored]


def one_s(g, by, v, s, mid_set, high_set):
    frame = list(comps(g, s, v))
    if len(frame) < 3:
        return None, "frame"
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set or (env - high_set)
    if not env_m:
        return None, "env"
    rest = [x for x in by[v] if x != s]
    if len(rest) < 2:
        return None, "rest"
    t, place = stand(g, by, v, env_m)
    if t is None:
        return None, "stand"
    if held in place:
        return None, "read_hit"
    cand = [c for c in place if c in mid_set and c != v]
    n = len(cand)
    if n == 0 or n >= 3:
        return dict(kind="refuse", n=n, hit=0, second=0, miss1=0), None
    if n == 1:
        hit = int(hop_hit(g, by, cand[0], env_m, held))
        return dict(kind="one", n=1, hit=hit, second=0, miss1=0), None
    order = rank_cand(g, by, cand, env_m)
    h0 = hop_hit(g, by, order[0], env_m, held)
    if h0:
        return dict(kind="two", n=2, hit=1, second=0, miss1=0), None
    h1 = hop_hit(g, by, order[1], env_m, held)
    return dict(kind="two", n=2, hit=int(h1), second=1, miss1=1), None


def windows(pool, n_win, L, rng):
    out = []
    for _ in range(n_win):
        if len(pool) <= L:
            out.append(pool)
        else:
            out.append(pool[rng.randrange(len(pool) - L + 1):][:L])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=8)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--cap-probe", type=int, default=6)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"567 try2 LIVE/DEAD  corpus={path}  {kind}", flush=True)
    rows, sk = [], Counter()
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        g = graph(lines, args.frame_max, args.min_fillers)
        if g is None:
            sk["nograph"] += 1
            continue
        by = mentions(g)
        mid, high, _a, _b = pct_band(g, by)
        mid_set, high_set = set(mid), set(high)
        keys = list(mid)
        rng.shuffle(keys)
        for v in keys:
            sl = list(by[v])
            if len(sl) < 8:
                sk["slots"] += 1
                continue
            rng.shuffle(sl)
            for s in sl[: args.cap_probe]:
                row, why = one_s(g, by, v, s, mid_set, high_set)
                if row is None:
                    sk[why] += 1
                    continue
                sk["keep"] += 1
                rows.append(row)
    one = [r for r in rows if r["kind"] == "one"]
    two = [r for r in rows if r["kind"] == "two"]
    miss1 = [r for r in two if r["miss1"]]
    n1, n2, nm = len(one), len(two), len(miss1)
    h1 = (sum(r["hit"] for r in one) / n1) if n1 else 0.0
    h2 = (sum(r["hit"] for r in two) / n2) if n2 else 0.0
    sec = (sum(r["second"] for r in miss1) / nm) if nm else 0.0
    saved = (sum(r["hit"] for r in miss1) / nm) if nm else 0.0
    void = n2 < 20
    gate = (not void) and (nm == 0 or sec >= 0.99) and h2 > 0.05
    print(f"one {n1} hit {h1:.3f}  two {n2} hit {h2:.3f}  "
          f"miss1 {nm} second {sec:.3f} saved {saved:.3f}")
    print(f"skip {dict(sk)}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: few 2-cand rows. Stories, not a dead try.")
    elif not gate:
        print("\nSTOP: second not tried, or two-cand hit <= 0.05. OPEN, 557 stays.")
    else:
        print("\nGO TRY2. First miss -> second; some 2-cand recovered.")
    rec = dict(seed=args.seed, corpus=kind, n_one=n1, hit_one=h1,
               n_two=n2, hit_two=h2, n_miss1=nm, second=sec, saved=saved,
               skip=dict(sk), elapsed_s=round(time.time() - t0, 1),
               void=bool(void), gate=bool(gate))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
