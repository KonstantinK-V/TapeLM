"""563: PIN writes W[stood]=addr; hop2 reads the mark. 406 on PLACE.

Skeleton (not 562 W[v] agree):
    STAND JACC -> READ
      unique mid neighbor -> W[stood]=addr -> hop2 from the mark
      two+                -> REFUSE, no write

VOID  n_pin < 40
GATE  from_w == 1 on PIN; refuse_wrote == 0
Hit reported, not gated. Meat later: place/env key, hop3, LIVE/DEAD.

    python _check563_pinw.py
    python _audit563_pinw.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage563_pinw.json")


def stand_read(g, by, addr, env_m, held, cap=8):
    sl = list(by.get(addr, []))
    if len(sl) < 2:
        return False, None
    scored = []
    for t in sl[: max(cap, 1)]:
        fr = set(comps(g, t, addr))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), t, fr))
    if not scored:
        return False, None
    scored.sort(key=lambda x: -x[0])
    _ov, t, fr = scored[0]
    return held in fr, t


def stand_jacc(g, by, v, env_m, exclude, cap=32):
    rest = [t for t in by.get(v, []) if t != exclude]
    if len(rest) < 2:
        return None, None, "rest"
    scored = []
    for t in rest:
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), ov, t, fr))
    if not scored:
        return None, None, "jac"
    scored.sort(key=lambda x: (-x[0], -x[1]))
    _j, _ov, stood, fr = scored[0]
    return stood, fr, None


def one_s(g, by, v, s, mid_set, high_set, W):
    frame = list(comps(g, s, v))
    if len(frame) < 3:
        return None, "frame"
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set or (env - high_set)
    if not env_m:
        return None, "env"
    stood, fr_p, why = stand_jacc(g, by, v, env_m, exclude=s)
    if why:
        return None, why
    if held in fr_p:
        return None, "read_hit"
    cand = [c for c in fr_p if c in mid_set and c != v]
    if len(cand) != 1:
        # REFUSE: no write on this stood
        return dict(pin=0, refuse=1, wrote=0, from_w=0, hit=-1,
                    stood=int(stood), n_cand=len(cand)), None
    addr = cand[0]
    W[stood] = addr
    addr_w = W[stood]
    from_w = int(addr_w == addr)
    hit, _ = stand_read(g, by, addr_w, env_m, held)
    return dict(pin=1, refuse=0, wrote=1, from_w=from_w, hit=int(hit),
                stood=int(stood), addr=addr, n_cand=1), None


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
            row, why = one_s(g, by, v, s, mid_set, high_set, W)
            if row is None:
                sk[why] += 1
                continue
            sk["keep"] += 1
            if row["pin"]:
                sk["pin"] += 1
            else:
                sk["refuse"] += 1
            rows.append(row)
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
    print(f"563 PIN -> W[stood]=addr  corpus={path}  {kind}", flush=True)
    wins = windows(pool, args.n_win, args.window_lines, rng)
    rows, sk = [], Counter()
    for lines in wins:
        rs, sv = run_win(lines, args, rng)
        rows.extend(rs)
        sk.update(sv)
    pins = [r for r in rows if r["pin"]]
    refs = [r for r in rows if r["refuse"]]
    n_pin = len(pins)
    n_ref = len(refs)
    from_w = (sum(r["from_w"] for r in pins) / n_pin) if n_pin else 0.0
    refuse_wrote = (sum(r["wrote"] for r in refs) / n_ref) if n_ref else 0.0
    hit = (sum(r["hit"] for r in pins) / n_pin) if n_pin else 0.0
    void = n_pin < 40
    gate = (not void) and from_w >= 1.0 and refuse_wrote <= 0.0
    print(f"pin {n_pin}  refuse {n_ref}  skip {dict(sk)}", flush=True)
    print(f"from_w {from_w:.3f}  refuse_wrote {refuse_wrote:.3f}  hit {hit:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: too few PIN writes.")
    elif not gate:
        print("\nSTOP: wiring broken (hop2 not from W, or refuse wrote).")
    else:
        print("\nGO WIRE. PIN marks stood; hop2 reads W[stood]; refuse silent.")
    rec = dict(seed=args.seed, corpus=kind, n_pin=n_pin, n_ref=n_ref,
               from_w=from_w, refuse_wrote=refuse_wrote, hit=hit,
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
