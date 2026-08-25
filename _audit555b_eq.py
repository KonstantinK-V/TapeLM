"""555b: 555 with equal budget. Drop trials where rec is shorter than PLACE.

STAR = rec_gl[:len(PLACE)] only if len(rec_gl) >= len(PLACE).
Otherwise skip short_star. gap must be 0.

GATE  hit_PLACE - hit_STAR > 0.05
VOID  n < 40 OR gap > 0.02

    python _check555b_eq.py
    python _audit555b_eq.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage555b_eq.json")


def rec_from(g, by, v, slots, cache):
    saved = by.get(v, [])
    by[v] = list(slots)
    cache.pop(v, None)
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    by[v] = saved
    return rec


def one_s(g, by, v, s, rest, cache, high_set, mid_set, rng):
    frame = list(comps(g, s, v))
    if len(frame) < 2:
        return None, "frame"
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set
    if not env_m:
        env_m = env - high_set
    if not env_m:
        return None, "env"
    rec_gl = rec_from(g, by, v, rest, cache)
    if not rec_gl:
        return None, "rec"
    jac, ora = [], []
    for t in rest:
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        jac.append((ov / max(len(fr), 1), t, fr))
        if held in fr:
            ora.append(t)
    if not ora:
        return None, "no_ora"
    jac.sort(key=lambda x: -x[0])
    fr_p = jac[0][2]
    if len(rec_gl) < len(fr_p):
        return None, "short_star"
    t_rnd = rng.choice(rest)
    fr_r = set(comps(g, t_rnd, v))
    take_s = rec_gl[: len(fr_p)]
    return dict(hit_p=held in fr_p, hit_s=held in set(take_s),
                hit_r=held in fr_r, n_p=len(fr_p), n_s=len(take_s),
                n_r=len(fr_r)), None


def one_v(g, by, v, cache, high_set, mid_set, rng, cap):
    sl = list(by[v])
    if len(sl) < 8:
        return [], Counter(slots=1)
    rng.shuffle(sl)
    rows, sk = [], Counter()
    for s in sl[: max(1, min(cap, len(sl)))]:
        rest = [x for x in sl if x != s]
        row, why = one_s(g, by, v, s, rest, cache, high_set, mid_set, rng)
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
            out.append(pool[rng.randrange(len(pool) - L + 1):][:L])
    return out


def rows_of(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], 0, Counter(nograph=1)
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    cache, rows, sk = {}, [], Counter()
    for v in mid:
        rs, sv = one_v(g, by, v, cache, set(high), set(mid), rng, args.cap_probe)
        rows.extend(rs)
        sk.update(sv)
    return rows, len(mid), sk


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
    print(f"555b equal-budget  corpus={path}  {kind}", flush=True)
    rows, n_mid, sk = [], 0, Counter()
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        rs, nm, sv = rows_of(lines, args, rng)
        rows.extend(rs)
        n_mid += nm
        sk.update(sv)
    n = len(rows)
    print(f"mid_sum {n_mid}  skip {dict(sk)}  n {n}", flush=True)
    if n == 0:
        rec = dict(seed=args.seed, n=0, void=True, gate=False)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        prev[str(args.seed)] = rec
        out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
        print("VOID: no trials")
        return 0

    def mean(key):
        return sum(r[key] for r in rows) / n

    h_p, h_s, h_r = mean("hit_p"), mean("hit_s"), mean("hit_r")
    np_, ns, nr = mean("n_p"), mean("n_s"), mean("n_r")
    d_star, d_rnd = h_p - h_s, h_p - h_r
    gap = abs(np_ - ns) / max(np_, 1e-9)
    void = n < 40 or gap > 0.02
    gate = (not void) and d_star > 0.05
    print(f"PLACE {h_p:.4f}  STAR {h_s:.4f}  RND {h_r:.4f}")
    print(f"|P| {np_:.2f}  |S| {ns:.2f}  |R| {nr:.2f}  gap {gap:.4f}")
    print(f"PLACE-STAR {d_star:+.4f}  PLACE-RND {d_rnd:+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: budget still leaks or too few.")
    elif d_star <= 0.05:
        print("\nPLACE <= STAR at equal n. Standing does not beat mixed rec.")
    else:
        print("\nGO READ. One apple's tape beats the mixed star at equal n.")
    rec = dict(seed=args.seed, corpus=kind, n=n, n_mid=n_mid, skip=dict(sk),
               hit_p=h_p, hit_s=h_s, hit_r=h_r, n_p=np_, n_s=ns, n_r=nr,
               d_star=d_star, d_rnd=d_rnd, gap=gap,
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
