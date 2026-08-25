"""550: apples are not one apple. Cluster mentions by environment.

Same token, different frames → different roots.
Held-out is ONE companion of a frame; the rest of the frame is env.
Cluster = other mentions of v that share a mid-df env token.
rec_cl from that bundle only. rec_gl = 511 over all mentions. Same allow.

    CL   rec of the env-bundle [:allow]
    GL   rec of all mentions of v [:allow]
    UNI  one random slot from CL offer
    ONE  rec_cl[0] only

GATE  hit_CL - hit_GL > 0.05
VOID  n < void_n  OR  split_rate < 0.20  (bundle ~ all mentions)
      void_n = max(25, min(80, int(0.35 * avg_mid_per_win)))

Reward is NOT the gate. Printed as oracle, held visible:
    +0.1 each hop in held
    stop after last hit: +0.05
    miss hop: 0
    go next if P(hit) > 0.5  (0.1P > 0.05)

p_extra = share of rec[1:allow] that sit in held.

    python _check550_bundle.py
    python _audit550_bundle.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import HIGH_DF, cheap_rec, comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import allow_of

OUT = Path("results/_stage550_bundle.json")


def rec_bundle(g, v, slots):
    """Co-fire rec on env-bundle mentions only (no global len>=8 gate)."""
    if len(slots) < 2:
        return []
    cnt = Counter()
    for s in slots:
        cnt.update(set(comps(g, s, v)))
    rec = [(c, n) for c, n in cnt.items() if n >= 2 and c != v]
    rec.sort(key=lambda cn: g["df"][cn[0]])
    return [c for c, _ in rec if g["df"][c] <= HIGH_DF]


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
    mates = []
    for t in rest:
        e = set(comps(g, t, v))
        if e & env_m:
            mates.append(t)
    if len(mates) < 2:
        return None
    rec_cl = rec_bundle(g, v, mates)
    rec_gl = rec_from(g, by, v, rest, cache)
    allow = allow_of(g, v, k, high_set)
    if v in high_set:
        allow = 1
    if not rec_cl or not rec_gl:
        return None
    take_cl = rec_cl[:allow]
    take_gl = rec_gl[:allow]
    cnt_cl = Counter()
    for t in mates:
        cnt_cl.update(set(comps(g, t, v)))
    extra = take_cl[1:]
    n_extra_hit = sum(1 for c in extra if c == held)
    return dict(
        held=held,
        n_rest=len(rest),
        n_mates=len(mates),
        split=len(mates) < len(rest),
        take_cl=take_cl,
        take_gl=take_gl,
        hit_cl=held in take_cl,
        hit_gl=held in take_gl,
        hit_one=bool(take_cl) and take_cl[0] == held,
        peaked=peaked(rec_cl, cnt_cl),
        n_extra=len(extra),
        n_extra_hit=n_extra_hit,
        allow=allow,
    )


def oracle_reward(take, held):
    """+0.1 per hit hop; +0.05 if we stop on a hit (no trailing miss)."""
    hits = [i for i, c in enumerate(take) if c == held]
    if not hits:
        return 0.0, 0.0
    n_hit = len(hits)
    allgo = 0.1 * n_hit
    stoph = 0.1 * n_hit + 0.05
    return stoph, allgo


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
        return [], None, 0
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
        probes = sl[: args.cap_probe]
        for probe in probes:
            r = one_v(g, by, v, probe, cache, k, high_set, mid_set, rng)
            if r:
                rows.append(r)
    return rows, k, len(mid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=12)
    ap.add_argument("--cap-probe", type=int, default=6,
                    help="held-out probe slots per v per window (528 trials cap)")
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
    print(f"550 bundle-by-env  corpus={path}  {kind}", flush=True)

    rows, k0, mid_sum = [], None, 0
    n_win_used = 0
    for lines in wins:
        rs, k, n_mid = rows_of(lines, args, rng, k0)
        if k0 is None:
            k0 = k
        mid_sum += n_mid
        n_win_used += 1
        rows.extend(rs)
    n = len(rows)
    mid_avg = mid_sum / max(n_win_used, 1)
    void_n = max(25, min(80, int(0.35 * mid_avg)))
    print(f"trials {n}  k {k0}  mid_avg {mid_avg:.0f}  void_n {void_n}", flush=True)
    if n == 0:
        print("VOID: no trials")
        rec = dict(seed=args.seed, corpus=kind, n=0, void=True, gate=False)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        prev[str(args.seed)] = rec
        out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
        return 0

    hit_cl = sum(r["hit_cl"] for r in rows) / n
    hit_gl = sum(r["hit_gl"] for r in rows) / n
    hit_one = sum(r["hit_one"] for r in rows) / n
    split = sum(r["split"] for r in rows) / n
    mates = sum(r["n_mates"] for r in rows) / n
    rest = sum(r["n_rest"] for r in rows) / n
    peak = sum(r["peaked"] for r in rows) / n
    n_ex = sum(r["n_extra"] for r in rows)
    n_exh = sum(r["n_extra_hit"] for r in rows)
    p_extra = n_exh / max(n_ex, 1)
    hit_uni = 0
    for r in rows:
        take = r["take_cl"]
        if take:
            hit_uni += int(r["held"] == rng.choice(take))
    hit_uni /= n
    stoph = allgo = 0.0
    for r in rows:
        s, a = oracle_reward(r["take_cl"], r["held"])
        stoph += s
        allgo += a
    stoph /= n
    allgo /= n
    d_gl = hit_cl - hit_gl
    void = n < void_n or split < 0.20
    gate = (not void) and d_gl > 0.05

    print(f"split {split:.3f}  mates/rest {mates:.1f}/{rest:.1f}  peaked_cl {peak:.3f}")
    print(f"CL {hit_cl:.4f}  GL {hit_gl:.4f}  ONE {hit_one:.4f}  UNI {hit_uni:.4f}")
    print(f"CL-GL {d_gl:+.4f}  p_extra {p_extra:.3f}  (go if >0.5)")
    print(f"oracle stophalf {stoph:.4f}  allgo {allgo:.4f}  d {stoph - allgo:+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print(f"\nVOID: n={n} < {void_n} (corpus) or bundle does not split mentions.")
    elif d_gl <= 0.05:
        print("\nBUNDLE DOES NOT BEAT MIXED STAR. Env cluster != different apples.")
    else:
        print("\nGO: env-bundle rec hits held more than 511 mixed rec. Same allow.")

    rec = dict(seed=args.seed, corpus=kind, k=k0, n=n, void_n=void_n, mid_avg=mid_avg,
               split=split,
               mates=mates, rest=rest, peaked_cl=peak,
               hit_cl=hit_cl, hit_gl=hit_gl, hit_one=hit_one, hit_uni=hit_uni,
               d_gl=d_gl, p_extra=p_extra, oracle_stophalf=stoph,
               oracle_allgo=allgo, elapsed_s=round(time.time() - t0, 1),
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
