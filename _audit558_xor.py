"""558: extra key resolves hop2 refuse. 445 XOR.

557: |cand|>=2 → refuse. 558: one leftover env token as extra.
    HIT   extra co-occurs with exactly one cand → pin that
    MISS  0 or 2+ cands with extra → refuse (hit_p=0)
    STAR  rec[0] still hops

GATE  on HIT trials: hit_P - hit_S > 0.05
VOID  n_hit < 40

    python _check558_xor.py
    python _audit558_xor.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage558_xor.json")


def rec_from(g, by, v, slots, cache):
    saved = by.get(v, [])
    by[v] = list(slots)
    cache.pop(v, None)
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    by[v] = saved
    return rec


def stand_read(g, by, addr, env_m, held, rng, cap=8):
    sl = [x for x in by.get(addr, [])]
    if len(sl) < 2:
        return None
    rng.shuffle(sl)
    scored = []
    for t in sl[: max(cap, 1)]:
        fr = set(comps(g, t, addr))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), fr))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return held in scored[0][1]


def has_extra(g, by, addr, extra, cap=8):
    n = 0
    for t in list(by.get(addr, []))[:cap]:
        if extra in set(comps(g, t, addr)):
            n += 1
    return n > 0


def one_s(g, by, v, s, rest, cache, high_set, mid_set, rng):
    frame = list(comps(g, s, v))
    if len(frame) < 3:
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
    jac = []
    for t in rest:
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        jac.append((ov / max(len(fr), 1), fr))
    if not jac:
        return None, "jac"
    jac.sort(key=lambda x: -x[0])
    fr_p = jac[0][1]
    if held in fr_p:
        return None, "read_hit"
    cand_p = [c for c in fr_p if c in mid_set and c != v]
    if len(cand_p) < 2:
        return None, "not_multi"
    extra_pool = [x for x in env_m if x not in cand_p]
    if not extra_pool:
        return None, "no_extra"
    extra = extra_pool[0]
    addr_s = rec_gl[0]
    if addr_s == v or addr_s in high_set:
        return None, "no_addr_s"
    hit_s = True if addr_s == held else stand_read(
        g, by, addr_s, env_m, held, rng,
    )
    if hit_s is None:
        return None, "read_s"
    hits = [c for c in cand_p if has_extra(g, by, c, extra)]
    if len(hits) != 1:
        return dict(hit_p=0, hit_s=int(hit_s), xor=0, n_hits=len(hits)), None
    hit_p = stand_read(g, by, hits[0], env_m, held, rng)
    if hit_p is None:
        return None, "read_p"
    same = int(hits[0] == addr_s)
    return dict(hit_p=int(hit_p), hit_s=int(hit_s), xor=1, n_hits=1,
                same=same), None


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
    print(f"558 XOR extra->unique hop2  corpus={path}  {kind}", flush=True)
    rows, n_mid, sk = [], 0, Counter()
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        rs, nm, sv = rows_of(lines, args, rng)
        rows.extend(rs)
        n_mid += nm
        sk.update(sv)
    n = len(rows)
    hit_rows = [r for r in rows if r["xor"]]
    n_hit = len(hit_rows)
    n_miss = n - n_hit
    print(f"mid_sum {n_mid}  skip {dict(sk)}  n {n}  xor {n_hit}  miss {n_miss}",
          flush=True)
    if n_hit == 0:
        rec = dict(seed=args.seed, n=n, n_hit=0, void=True, gate=False)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        prev[str(args.seed)] = rec
        out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
        print("VOID: no unique-extra trials")
        return 0

    def mean(rs, key):
        return sum(r[key] for r in rs) / len(rs)

    h_p, h_s = mean(hit_rows, "hit_p"), mean(hit_rows, "hit_s")
    same = mean(hit_rows, "same") if hit_rows else 0.0
    d = h_p - h_s
    void = n_hit < 40
    gate = (not void) and d > 0.05
    print(f"XOR P {h_p:.4f}  S {h_s:.4f}  same {same:.3f}  P-S {d:+.4f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: extra rarely unique. 557 refuse stands.")
    elif d <= 0.05:
        print("\nXOR unique does not beat STAR. Extra does not pick a better address.")
    else:
        print("\nGO XOR. Unique extra resolves refuse; 0/2+ stay silent.")
    rec = dict(seed=args.seed, corpus=kind, n=n, n_hit=n_hit, n_miss=n_miss,
               n_mid=n_mid, skip=dict(sk),
               hit_p=h_p, hit_s=h_s, same=same, d=d,
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
