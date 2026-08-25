"""552: 551 population + SPEC slice. Not a new ranker.

ALL must land near 551 (~765 on stories 8x400). If ALL stays ~150
this file is not the one running. Skip histogram prints why.

SPEC = cnt_mates[held] > cnt_outsiders[held]
VOID n_spec < 40
GATE (RN-GL) on SPEC > 0.05

    python _check552_spec.py
    python _audit552_spec.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import cheap_rec, comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import allow_of

OUT = Path("results/_stage552_spec.json")


def rec_from(g, by, v, slots, cache):
    saved = by.get(v, [])
    by[v] = list(slots)
    cache.pop(v, None)
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    by[v] = saved
    return rec


def one_s(g, by, v, s, rest, cache, k, high_set, mid_set, rng):
    frame = list(comps(g, s, v))
    if len(frame) < 2:
        return None, "frame"
    rng.shuffle(frame)
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set
    if not env_m:
        env_m = env - high_set
    if not env_m:
        return None, "env"
    mates, out = [], []
    for t in rest:
        e = set(comps(g, t, v))
        (mates if e & env_m else out).append(t)
    if not mates:
        return None, "mates"
    rec_gl = rec_from(g, by, v, rest, cache)
    if not rec_gl:
        return None, "rec"
    rec_fl = rec_from(g, by, v, mates, cache)
    allow = allow_of(g, v, k, high_set)
    if v in high_set:
        allow = 1
    cnt_m, cnt_o, cnt_g = Counter(), Counter(), Counter()
    for t in mates:
        cnt_m.update(set(comps(g, t, v)))
    for t in out:
        cnt_o.update(set(comps(g, t, v)))
    for t in rest:
        cnt_g.update(set(comps(g, t, v)))
    rec_rn = sorted(rec_gl, key=lambda c: (-cnt_m[c], -cnt_g[c]))
    take_rn, take_gl = rec_rn[:allow], rec_gl[:allow]
    take_fl = rec_fl[:allow] if rec_fl else []
    spec = cnt_m[held] > cnt_o[held]
    return dict(spec=spec, hit_rn=held in take_rn, hit_gl=held in take_gl,
                hit_fl=held in take_fl, cm=cnt_m[held], co=cnt_o[held]), None


def one_v(g, by, v, cache, k, high_set, mid_set, rng, cap):
    sl = list(by[v])
    if len(sl) < 8:
        return [], Counter(slots=1)
    rng.shuffle(sl)
    rows, sk = [], Counter()
    n_try = max(1, min(cap, len(sl)))
    for s in sl[:n_try]:
        rest = [x for x in sl if x != s]
        row, why = one_s(g, by, v, s, rest, cache, k, high_set, mid_set, rng)
        if row is None:
            sk[why] += 1
        else:
            sk["keep"] += 1
            rows.append(row)
    return rows, sk


def windows(pool, n_win, L, rng):
    out = []
    for _ in range(n_win):
        if len(pool) <= L:
            out.append(pool)
        else:
            s0 = rng.randrange(len(pool) - L + 1)
            out.append(pool[s0:s0 + L])
    return out


def rows_of(lines, args, rng, k_hold=None):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], None, 0, Counter(nograph=1)
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    k = 200.0 / max(g["n"], 1) if k_hold is None else k_hold
    cache, rows, sk = {}, [], Counter()
    for v in mid:
        rs, sv = one_v(g, by, v, cache, k, high_set, mid_set, rng, args.cap_probe)
        rows.extend(rs)
        sk.update(sv)
    return rows, k, len(mid), sk


def pack(rows):
    n = len(rows)
    if n == 0:
        return dict(n=0, hit_rn=0.0, hit_gl=0.0, hit_fl=0.0, d=0.0)
    rn = sum(r["hit_rn"] for r in rows) / n
    gl = sum(r["hit_gl"] for r in rows) / n
    fl = sum(r["hit_fl"] for r in rows) / n
    return dict(n=n, hit_rn=rn, hit_gl=gl, hit_fl=fl, d=rn - gl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=8)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--cap-probe", type=int, default=1)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"552 spec-slice  corpus={path}  {kind}  cap={args.cap_probe}", flush=True)

    rows, k0, n_mid, sk = [], None, 0, Counter()
    n_win_ok = 0
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        rs, k, nm, sv = rows_of(lines, args, rng, k0)
        if k0 is None:
            k0 = k
        rows.extend(rs)
        n_mid += nm
        sk.update(sv)
        n_win_ok += 1 if nm else 0

    spec = [r for r in rows if r["spec"]]
    gen = [r for r in rows if not r["spec"]]
    a, s, g = pack(rows), pack(spec), pack(gen)
    void = s["n"] < 40
    gate = (not void) and s["d"] > 0.05

    print(f"mid_sum {n_mid}  wins {n_win_ok}  skip {dict(sk)}")
    print(f"ALL  n {a['n']}  RN {a['hit_rn']:.4f}  GL {a['hit_gl']:.4f}  "
          f"FL {a['hit_fl']:.4f}  d {a['d']:+.4f}")
    print(f"SPEC n {s['n']}  RN {s['hit_rn']:.4f}  GL {s['hit_gl']:.4f}  "
          f"FL {s['hit_fl']:.4f}  d {s['d']:+.4f}  share {s['n']/max(a['n'],1):.3f}")
    print(f"GEN  n {g['n']}  RN {g['hit_rn']:.4f}  GL {g['hit_gl']:.4f}  "
          f"d {g['d']:+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if a["n"] < 400 and args.cap_probe == 1:
        print("\nPOPULATION != 551. Read skip= above, not SPEC.")
    elif void:
        print("\nVOID: SPEC n < 40.")
    elif s["d"] <= 0.05:
        print("\nSPEC STILL <=0.05. Env offer closed even on poison-vs-tasty.")
    else:
        print("\nGO ON SPEC. Generic held was drowning the exam.")

    rec = dict(seed=args.seed, corpus=kind, k=k0, cap=args.cap_probe,
               n_mid=n_mid, skip=dict(sk), all=a, spec=s, gen=g,
               elapsed_s=round(time.time() - t0, 1),
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
