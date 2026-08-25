"""551: env-bundle RERANKS 511 rec, does not filter it.

550: split alive (mates 5 vs rest 21, peaked 0.61) but CL-filter
lost cover to mixed star (-0.05). Same 537: mark as filter starves
the offer; 536 paid on reorder. 551 keeps rec_gl as the candidate
universe and sorts by count-in-mates, then count-gl.

    GL  rec of all mentions [:allow]          511 order
    RN  same rec, sorted by cnt_mates desc    env first
    FL  rec of mates only [:allow]            550 filter, control
    ONE RN[0]

GATE  hit_RN - hit_GL > 0.05
VOID  n < 40 OR split < 0.20

    python _check551_rerank.py
    python _audit551_rerank.py --seed 1337 --corpus data/_tinystories_train.txt
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
from _audit550_bundle import rec_bundle

OUT = Path("results/_stage551_rerank.json")


def rec_from(g, by, v, slots, cache):
    saved = by.get(v, [])
    by[v] = list(slots)
    cache.pop(v, None)
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    by[v] = saved
    return rec


def peaked(rec, cnt):
    if not rec:
        return False
    n0 = cnt.get(rec[0], 0)
    n1 = cnt.get(rec[1], 0) if len(rec) > 1 else 0
    return len(rec) == 1 or (n0 > 0 and n1 < 0.5 * n0)


def one_v(g, by, v, probe, cache, k, high_set, mid_set, rng):
    sl = list(by[v])
    if len(sl) < 8:
        return None
    rest = [t for t in sl if t != probe]
    if len(rest) < 7:
        return None
    frame = list(comps(g, probe, v))
    if len(frame) < 2:
        return None
    rng.shuffle(frame)
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set
    if not env_m:
        env_m = env - high_set
    if not env_m:
        return None
    mates = [t for t in rest if set(comps(g, t, v)) & env_m]
    if len(mates) < 2:
        return None
    rec_gl = rec_from(g, by, v, rest, cache)
    rec_fl = rec_bundle(g, v, mates)
    if not rec_gl:
        return None
    allow = allow_of(g, v, k, high_set)
    if v in high_set:
        allow = 1
    cnt_m = Counter()
    cnt_g = Counter()
    for t in mates:
        cnt_m.update(set(comps(g, t, v)))
    for t in rest:
        cnt_g.update(set(comps(g, t, v)))
    rec_rn = sorted(rec_gl, key=lambda c: (-cnt_m[c], -cnt_g[c]))
    take_gl = rec_gl[:allow]
    take_rn = rec_rn[:allow]
    take_fl = rec_fl[:allow] if rec_fl else []
    extra = take_rn[1:]
    return dict(
        held=held,
        n_rest=len(rest),
        n_mates=len(mates),
        split=len(mates) < len(rest),
        take_gl=take_gl,
        take_rn=take_rn,
        take_fl=take_fl,
        hit_gl=held in take_gl,
        hit_rn=held in take_rn,
        hit_fl=held in take_fl,
        hit_one=bool(take_rn) and take_rn[0] == held,
        hit_gl0=bool(take_gl) and take_gl[0] == held,
        peaked=peaked(rec_rn, cnt_m),
        distinctive=bool(take_gl) and take_gl[0] != held,
        n_extra=len(extra),
        n_extra_hit=sum(1 for c in extra if c == held),
        allow=allow,
        set_diff=set(take_rn) != set(take_gl),
    )


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
        return [], None
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    k = 200.0 / max(g["n"], 1) if k_hold is None else k_hold
    cache = {}
    rows = []
    for v in mid:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        for probe in sl[: args.cap_probe]:
            r = one_v(g, by, v, probe, cache, k, high_set, mid_set, rng)
            if r:
                rows.append(r)
    return rows, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=12)
    ap.add_argument("--cap-probe", type=int, default=6)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    wins = windows(pool, args.n_win, args.window_lines, rng)
    t0 = time.time()
    print(f"551 rerank-by-env  corpus={path}  {kind}", flush=True)

    rows, k0 = [], None
    for lines in wins:
        rs, k = rows_of(lines, args, rng, k0)
        if k0 is None:
            k0 = k
        rows.extend(rs)
    n = len(rows)
    print(f"trials {n}  k {k0}", flush=True)
    if n == 0:
        rec = dict(seed=args.seed, corpus=kind, n=0, void=True, gate=False)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        prev[str(args.seed)] = rec
        out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
        print("VOID: no trials")
        return 0

    def mean(key):
        return sum(r[key] for r in rows) / n

    hit_rn, hit_gl, hit_fl = mean("hit_rn"), mean("hit_gl"), mean("hit_fl")
    hit_one, hit_gl0 = mean("hit_one"), mean("hit_gl0")
    split, peak = mean("split"), mean("peaked")
    mates, rest = mean("n_mates"), mean("n_rest")
    set_d = mean("set_diff")
    n_ex = sum(r["n_extra"] for r in rows)
    n_exh = sum(r["n_extra_hit"] for r in rows)
    p_extra = n_exh / max(n_ex, 1)
    dist = [r for r in rows if r["distinctive"]]
    nd = len(dist)
    d_hit = ((sum(r["hit_rn"] for r in dist) - sum(r["hit_gl"] for r in dist)) / nd
             if nd else 0.0)
    d_gl = hit_rn - hit_gl
    d1 = hit_one - hit_gl0
    void = n < 40 or split < 0.20
    gate = (not void) and d_gl > 0.05

    print(f"split {split:.3f}  mates/rest {mates:.1f}/{rest:.1f}  "
          f"set_diff {set_d:.3f}  peaked {peak:.3f}")
    print(f"RN {hit_rn:.4f}  GL {hit_gl:.4f}  FL {hit_fl:.4f}  "
          f"ONE {hit_one:.4f}  GL0 {hit_gl0:.4f}")
    print(f"RN-GL {d_gl:+.4f}  ONE-GL0 {d1:+.4f}  p_extra {p_extra:.3f}")
    print(f"distinctive n {nd}  RN-GL | dist {d_hit:+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: too few trials or no split.")
    elif d_gl <= 0.05:
        print("\nRERANK DOES NOT BEAT 511 ORDER. Env weight != better prefix.")
    else:
        print("\nGO: env counts pull the right companion into allow. Filter stays 550.")

    rec = dict(seed=args.seed, corpus=kind, k=k0, n=n, split=split,
               mates=mates, rest=rest, peaked=peak, set_diff=set_d,
               hit_rn=hit_rn, hit_gl=hit_gl, hit_fl=hit_fl,
               hit_one=hit_one, hit_gl0=hit_gl0, d_gl=d_gl, d1=d1,
               p_extra=p_extra, n_dist=nd, d_dist=d_hit,
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
