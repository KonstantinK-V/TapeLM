"""565: write key is env (place), not v (word).

    W_v[v]      = addr     # 562, control
    W_e[env_m]  = addr     # 565

VOID  n_reuse_e < 40
GATE  agree_e ≥ 0.80

agree_v reported beside; hit not gated. Do not rewrite 562.

    python _check565_envkey.py
    python _audit565_envkey.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage565_envkey.json")


def stand_read(g, by, addr, env_m, held, cap=8):
    sl = list(by.get(addr, []))
    if len(sl) < 2:
        return False
    scored = []
    for t in sl[: max(cap, 1)]:
        fr = set(comps(g, t, addr))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), t, fr))
    if not scored:
        return False
    scored.sort(key=lambda x: -x[0])
    return held in scored[0][2]


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
    scored = []
    for t in rest:
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), t, fr))
    if not scored:
        return None, "jac"
    scored.sort(key=lambda x: -x[0])
    fr_p = scored[0][2]
    if held in fr_p:
        return None, "read_hit"
    cand = [c for c in fr_p if c in mid_set and c != v]
    if len(cand) != 1:
        return None, "refuse"
    return dict(v=v, addr=cand[0], env=frozenset(env_m),
                env_m=env_m, held=held), None


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
    W_v, W_e = {}, {}
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
            addr, ek = row["addr"], row["env"]
            rec = dict(reuse_v=0, agree_v=0, reuse_e=0, agree_e=0,
                       hit_e=-1, hit_f=int(stand_read(
                           g, by, addr, row["env_m"], row["held"])))
            if v in W_v:
                rec["reuse_v"] = 1
                rec["agree_v"] = int(W_v[v] == addr)
            if ek in W_e:
                rec["reuse_e"] = 1
                rec["agree_e"] = int(W_e[ek] == addr)
                rec["hit_e"] = int(stand_read(
                    g, by, W_e[ek], row["env_m"], row["held"]))
            W_v[v] = addr
            W_e[ek] = addr
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
    print(f"565 W[env] vs W[v]  corpus={path}  {kind}", flush=True)
    rows, sk = [], Counter()
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        rs, sv = run_win(lines, args, rng)
        rows.extend(rs)
        sk.update(sv)
    rv = [r for r in rows if r["reuse_v"]]
    re = [r for r in rows if r["reuse_e"]]
    n_v, n_e = len(rv), len(re)
    av = (sum(r["agree_v"] for r in rv) / n_v) if n_v else 0.0
    ae = (sum(r["agree_e"] for r in re) / n_e) if n_e else 0.0
    he = (sum(r["hit_e"] for r in re if r["hit_e"] >= 0) / n_e) if n_e else 0.0
    hf = (sum(r["hit_f"] for r in re) / n_e) if n_e else 0.0
    void = n_e < 40
    gate = (not void) and ae >= 0.80
    print(f"pin {len(rows)}  reuse_v {n_v} agree_v {av:.3f}  "
          f"reuse_e {n_e} agree_e {ae:.3f}  hit_e {he:.3f} hit_f {hf:.3f}")
    print(f"skip {dict(sk)}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: env keys rarely collide. Stories, not a dead write.")
    elif not gate:
        print("\nSTOP: env key not the same PIN. OPEN like 562, don't close 481.")
    else:
        print("\nGO ENV. Place/env is a stable write key on this corpus.")
    rec = dict(seed=args.seed, corpus=kind, n_pin=len(rows),
               n_reuse_v=n_v, agree_v=av, n_reuse_e=n_e, agree_e=ae,
               hit_e=he, hit_f=hf, skip=dict(sk),
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
