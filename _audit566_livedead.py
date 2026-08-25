"""566: LIVE/DEAD on W[env]. 565 reuse, with meaning.

    PIN → hop2
      hit  → W[env] = (addr, LIVE)  → next same env walks the mark
      miss → W[env] = (addr, DEAD)  → next same env does not walk that addr

VOID  n_live < 40 or n_dead < 10
GATE  live agree ≥ 0.80
      dead follow never hops from the DEAD addr

same_addr not gated: on DEAD unique cand often same — we do not walk.
    python _check566_livedead.py
    python _audit566_livedead.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage566_livedead.json")


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
    if len(cand) != 1:
        return None, "refuse"
    addr = cand[0]
    _t2, fr2 = stand(g, by, addr, env_m)
    hit = bool(_t2 is not None and held in fr2)
    return dict(v=v, addr=addr, env=frozenset(env_m), env_m=env_m,
                held=held, hit=hit), None


def windows(pool, n_win, L, rng):
    out = []
    for _ in range(n_win):
        if len(pool) <= L:
            out.append(pool)
        else:
            out.append(pool[rng.randrange(len(pool) - L + 1):][:L])
    return out


def run_win(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], Counter(nograph=1)
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    W = {}
    rows, sk = [], Counter()
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
            ek, addr, hit = row["env"], row["addr"], row["hit"]
            rec = dict(reuse_l=0, agree_l=0, reuse_d=0, walked_d=0,
                       used_dead=0, hit=int(hit), first_dead=0)
            if ek in W:
                old, mark = W[ek]
                if mark == "LIVE":
                    rec["reuse_l"] = 1
                    rec["agree_l"] = int(old == addr)
                    rows.append(rec)
                    continue
                rec["reuse_d"] = 1
                rec["used_dead"] = int(old == addr)
                rec["walked_d"] = 0
                rows.append(rec)
                continue
            W[ek] = (addr, "LIVE" if hit else "DEAD")
            rec["first_dead"] = int(not hit)
            rows.append(rec)
    return rows, sk


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
    print(f"566 LIVE/DEAD on W[env]  corpus={path}  {kind}", flush=True)
    rows, sk = [], Counter()
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        rs, sv = run_win(lines, args, rng)
        rows.extend(rs)
        sk.update(sv)
    live_r = [r for r in rows if r["reuse_l"]]
    dead_r = [r for r in rows if r["reuse_d"]]
    n_l, n_d = len(live_r), len(dead_r)
    n_dead_w = sum(r["first_dead"] for r in rows)
    al = (sum(r["agree_l"] for r in live_r) / n_l) if n_l else 0.0
    wd = (sum(r["walked_d"] for r in dead_r) / n_d) if n_d else 0.0
    ud = (sum(r["used_dead"] for r in dead_r) / n_d) if n_d else 0.0
    ht = (sum(r["hit"] for r in rows) / len(rows)) if rows else 0.0
    void = n_l < 40 or n_d < 10
    gate = (not void) and al >= 0.80 and wd == 0.0
    print(f"pin {len(rows)}  live_reuse {n_l} agree_l {al:.3f}  "
          f"dead_follow {n_d} walked_d {wd:.3f}  same_addr {ud:.3f}  "
          f"first_dead {n_dead_w}  hit {ht:.3f}")
    print(f"skip {dict(sk)}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: few LIVE reuse or few DEAD. Stories leftover, not a dead mark.")
    elif not gate:
        print("\nSTOP: LIVE unstable or DEAD was walked. OPEN, don't close 481.")
    else:
        print("\nGO MARK. LIVE reused; DEAD not walked again.")
    rec = dict(seed=args.seed, corpus=kind, n_pin=len(rows),
               n_live=n_l, agree_l=al, n_dead_follow=n_d, walked_d=wd,
               same_addr=ud, n_first_dead=n_dead_w, hit=ht, skip=dict(sk),
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
