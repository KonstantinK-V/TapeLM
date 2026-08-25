"""554: 553 size confound. Jaccard vs raw |∩|.

553 MARK = argmax |frame ∩ env|. Longer mentions win and more
often contain held. 554 same trials, four picks:

    CNT   553 count overlap
    JACC  |∩| / |frame|
    RND   coin
    MAJ   hub mention (rec_gl[0] in frame)

GATE  hit_JACC - hit_RND > 0.05
VOID  n < 40 OR (ORA - RND) <= 0.05

    python _check554_jacc.py
    python _audit554_jacc.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage554_jacc.json")


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
    hub = rec_gl[0] if rec_gl else None
    cnt, jac, ora, maj = [], [], [], []
    for t in rest:
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        cnt.append((ov, t))
        jac.append((ov / max(len(fr), 1), t))
        if held in fr:
            ora.append(t)
        if hub is not None and hub in fr:
            maj.append(t)
    if not ora:
        return None, "no_ora"
    cnt.sort(key=lambda x: -x[0])
    jac.sort(key=lambda x: -x[0])
    t_cnt, t_jacc = cnt[0][1], jac[0][1]
    t_rnd = rng.choice(rest)
    t_maj = rng.choice(maj) if maj else t_rnd

    def hit(t):
        return held in set(comps(g, t, v))

    return dict(hit_cnt=hit(t_cnt), hit_jacc=hit(t_jacc),
                hit_rnd=hit(t_rnd), hit_maj=hit(t_maj),
                hit_ora=True, len_cnt=len(set(comps(g, t_cnt, v))),
                len_rnd=len(set(comps(g, t_rnd, v))),
                len_jacc=len(set(comps(g, t_jacc, v)))), None


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
    print(f"554 jacc-vs-count  corpus={path}  {kind}", flush=True)
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

    h_c, h_j, h_r, h_m = (mean("hit_cnt"), mean("hit_jacc"),
                          mean("hit_rnd"), mean("hit_maj"))
    d_c, d_j = h_c - h_r, h_j - h_r
    room = 1.0 - h_r
    void = n < 40 or room <= 0.05
    gate = (not void) and d_j > 0.05
    print(f"CNT {h_c:.4f}  JACC {h_j:.4f}  RND {h_r:.4f}  MAJ {h_m:.4f}")
    print(f"CNT-RND {d_c:+.4f}  JACC-RND {d_j:+.4f}  ROOM {room:+.4f}")
    print(f"len CNT {mean('len_cnt'):.2f}  JACC {mean('len_jacc'):.2f}  "
          f"RND {mean('len_rnd'):.2f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: no room or too few.")
    elif d_j <= 0.05 <= d_c:
        print("\nCOUNT PAYS, JACC DOES NOT. 553 was mention length.")
    elif d_j <= 0.05:
        print("\nJACC IS COIN. Place pick does not survive size-norm.")
    else:
        print("\nGO JACC. Env overlap is not just longer frames.")
    rec = dict(seed=args.seed, corpus=kind, n=n, n_mid=n_mid, skip=dict(sk),
               hit_cnt=h_c, hit_jacc=h_j, hit_rnd=h_r, hit_maj=h_m,
               d_cnt=d_c, d_jacc=d_j, room=room,
               len_cnt=mean("len_cnt"), len_jacc=mean("len_jacc"),
               len_rnd=mean("len_rnd"),
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
