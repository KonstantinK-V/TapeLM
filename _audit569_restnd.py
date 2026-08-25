"""569: after unique PIN is DEAD, stand on another mention of the same v.

Does not invent a new word. Same 557: unique cand or refuse.
STOP on stories is law: gate retry>=0.99 tests corpus completeness, not a fix target.
Do not lower retry, add windows, third stand, 3-cand, or uncertain substitution.
Runner: retry only if second unique stand already present; else refuse (pears rule).

VOID  n_dead1 < 20
GATE  retry >= 0.99 on unique-DEAD  and  saved > 0.05
saved/same printed; hit1 not gated. 567 two-cand not mixed in.

    python _check569_restnd.py
    python _audit569_restnd.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage569_restnd.json")


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


def pin_unique(g, by, v, s, mid_set, high_set):
    frame = list(comps(g, s, v))
    if len(frame) < 3:
        return None
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set or (env - high_set)
    if not env_m:
        return None
    rest = [x for x in by[v] if x != s]
    if len(rest) < 2:
        return None
    t, place = stand(g, by, v, env_m)
    if t is None:
        return None
    if held in place:
        return None
    cand = [c for c in place if c in mid_set and c != v]
    if len(cand) != 1:
        return None
    addr = cand[0]
    hit = hop_hit(g, by, addr, env_m, held)
    return dict(s=s, held=held, env_m=env_m, addr=addr, hit=hit, t=t)


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
    print(f"569 retry-stand after unique DEAD  corpus={path}  {kind}", flush=True)
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
                p1 = pin_unique(g, by, v, s, mid_set, high_set)
                if p1 is None:
                    sk["nouniq"] += 1
                    continue
                sk["uniq"] += 1
                rec = dict(hit1=int(p1["hit"]), dead1=int(not p1["hit"]),
                           retry=0, same=0, saved=0)
                if p1["hit"]:
                    rows.append(rec)
                    continue
                tried = False
                for s2 in sl:
                    if s2 == s:
                        continue
                    p2 = pin_unique(g, by, v, s2, mid_set, high_set)
                    if p2 is None:
                        continue
                    tried = True
                    rec["retry"] = 1
                    rec["same"] = int(p2["addr"] == p1["addr"])
                    if p2["addr"] != p1["addr"] and p2["hit"]:
                        rec["saved"] = 1
                    break
                if not tried:
                    sk["noretry"] += 1
                rows.append(rec)
    dead = [r for r in rows if r["dead1"]]
    nd, n = len(dead), len(rows)
    h1 = (sum(r["hit1"] for r in rows) / n) if n else 0.0
    rt = (sum(r["retry"] for r in dead) / nd) if nd else 0.0
    sm = (sum(r["same"] for r in dead if r["retry"]) / max(sum(r["retry"] for r in dead), 1))
    sv = (sum(r["saved"] for r in dead) / nd) if nd else 0.0
    void = nd < 20
    gate = (not void) and rt >= 0.99 and sv > 0.05
    print(f"uniq {n} hit1 {h1:.3f}  dead1 {nd} retry {rt:.3f}  "
          f"same {sm:.3f} saved {sv:.3f}")
    print(f"skip {dict(sk)}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: few unique-DEAD. Leftover thin, silence may be right.")
    elif not gate:
        print("\nSTOP (law on stories): retry incomplete or saved thin. "
              "Unique DEAD -> refuse unless 2nd unique already there.")
    else:
        print("\nGO RETRY-STAND. Other mention of v recovered after unique DEAD.")
    rec = dict(seed=args.seed, corpus=kind, n_uniq=n, hit1=h1, n_dead1=nd,
               retry=rt, same=sm, saved=sv, skip=dict(sk),
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
