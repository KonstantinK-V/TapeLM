"""559: chooser arena. Does mind have anything to pick?

PLACE-WALK is licensed. Mind = choose PLACE / STAR / REFUSE.
If always-PLACE is optimal, there is no chooser — 38.3.

    PLACE   unique neighbor hop2 (557); many → miss
    STAR    rec[0] hop2
    REFUSE  miss on purpose

Correct action = first that retrieves held: PLACE, else STAR, else REFUSE.
    u_P  PLACE unique (hits, STAR misses)
    u_S  STAR unique  (hits, PLACE misses)
    u_R  neither hits

GATE  u_S > 0.05 and u_P > 0.05   both ops needed
VOID  n < 40

    python _check559_arena.py
    python _audit559_arena.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage559_arena.json")


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
        return False
    rng.shuffle(sl)
    scored = []
    for t in sl[: max(cap, 1)]:
        fr = set(comps(g, t, addr))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), fr))
    if not scored:
        return False
    scored.sort(key=lambda x: -x[0])
    return held in scored[0][1]


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
    addr_s = rec_gl[0]
    if addr_s == v or addr_s in high_set:
        return None, "no_addr_s"
    hit_s = True if addr_s == held else stand_read(
        g, by, addr_s, env_m, held, rng,
    )
    hit_p = False
    if len(cand_p) == 1:
        hit_p = stand_read(g, by, cand_p[0], env_m, held, rng)
    if hit_p and not hit_s:
        gold = "P"
    elif hit_s and not hit_p:
        gold = "S"
    elif hit_p and hit_s:
        gold = "both"
    else:
        gold = "R"
    return dict(hit_p=int(hit_p), hit_s=int(hit_s), gold=gold,
                n_cand=len(cand_p)), None


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
    print(f"559 chooser arena  corpus={path}  {kind}", flush=True)
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
    gold = Counter(r["gold"] for r in rows)
    u_p = gold["P"] / n
    u_s = gold["S"] / n
    u_r = gold["R"] / n
    both = gold["both"] / n
    always_p = (gold["P"] + gold["both"]) / n
    always_s = (gold["S"] + gold["both"]) / n
    void = n < 40
    gate = (not void) and u_s > 0.05 and u_p > 0.05
    print(f"u_P {u_p:.3f}  u_S {u_s:.3f}  u_R {u_r:.3f}  both {both:.3f}")
    print(f"always_PLACE {always_p:.3f}  always_STAR {always_s:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: too few.")
    elif not gate:
        print("\nNO CHOOSER. One op dominates — mind has nothing to pick. 38.3")
    else:
        print("\nGO ARENA. PLACE and STAR each unique sometimes — chooser licensed.")
    rec = dict(seed=args.seed, corpus=kind, n=n, n_mid=n_mid, skip=dict(sk),
               u_p=u_p, u_s=u_s, u_r=u_r, both=both,
               always_p=always_p, always_s=always_s, gold=dict(gold),
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
