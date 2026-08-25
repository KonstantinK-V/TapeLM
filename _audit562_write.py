"""562: write PIN on working tape. No Q.

After unique PIN, W[v] = addr. Next trial of the same v hops from the
mark, does not recompute the neighbor.

VOID  n_reuse < 40
GATE  agree ≥ 0.80

    python _check562_write.py
    python _audit562_write.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band

OUT = Path("results/_stage562_write.json")


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
        return dict(pin=0, addr=None, env_m=env_m, held=held), None
    return dict(pin=1, addr=cand[0], env_m=env_m, held=held), None


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
    addrs = defaultdict(set)
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
            if not row["pin"]:
                sk["refuse"] += 1
                continue
            addr_f = row["addr"]
            addrs[v].add(addr_f)
            hit_f, _ = stand_read(g, by, addr_f, row["env_m"], row["held"])
            reuse = 0
            agree = 0
            hit_m = None
            if v in W:
                reuse = 1
                addr_m = W[v]
                agree = int(addr_m == addr_f)
                hit_m, _ = stand_read(g, by, addr_m, row["env_m"], row["held"])
            W[v] = addr_f
            rows.append(dict(reuse=reuse, agree=agree, hit_f=int(hit_f),
                             hit_m=-1 if hit_m is None else int(hit_m),
                             n_addr=len(addrs[v])))
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
    print(f"562 write PIN -> W[v]=addr  corpus={path}  {kind}", flush=True)
    wins = windows(pool, args.n_win, args.window_lines, rng)
    rows, sk = [], Counter()
    for lines in wins:
        rs, sv = run_win(lines, args, rng)
        rows.extend(rs)
        sk.update(sv)
    reuse = [r for r in rows if r["reuse"]]
    n_r = len(reuse)
    n_pin = len(rows)
    agree = (sum(r["agree"] for r in reuse) / n_r) if n_r else 0.0
    hit_m = (sum(r["hit_m"] for r in reuse) / n_r) if n_r else 0.0
    hit_f = (sum(r["hit_f"] for r in reuse) / n_r) if n_r else 0.0
    n_addr = (sum(r["n_addr"] for r in reuse) / n_r) if n_r else 0.0
    void = n_r < 40
    gate = (not void) and agree >= 0.80
    print(f"pin {n_pin}  reuse {n_r}  skip {dict(sk)}", flush=True)
    print(f"agree {agree:.3f}  hit_m {hit_m:.3f}  hit_f {hit_f:.3f}  "
          f"n_addr {n_addr:.2f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: too few second PINs of the same v.")
    elif not gate:
        print("\nSTOP: W[v] is not the same addr. Key is the word, not the place.")
    else:
        print("\nGO WRITE. Mark is the PIN. hop2 can skip the neighbor search.")
    rec = dict(seed=args.seed, corpus=kind, n_pin=n_pin, n_reuse=n_r,
               agree=agree, hit_m=hit_m, hit_f=hit_f, n_addr=n_addr,
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
