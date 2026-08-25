"""553: pick a mention (place), read held from tape. Not rec rerank.

Probe gives env; choose which other mention of v to stand at.
Read companions at that slot; hit if held is in the frame there.

    ORA  probe mention (held lives there)     ceiling
    MARK mention with max |frame cap env_m|
    RND  random mention in rest
    MAJ  mention whose frame contains global majority co-fire

VOID  n < 40  OR  oracle barely beats random (room)
GATE  MARK - RND > 0.05

    python _check553_place.py
    python _audit553_place.py --seed 1337 --corpus data/_tinystories_train.txt
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
from _audit527_learn import majority

OUT = Path("results/_stage553_place.json")


def pick_mark(rest, g, v, env_m):
    scored = []
    for t in rest:
        fr = set(comps(g, t, v))
        scored.append((len(fr & env_m), t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def pick_maj(rest, g, v):
    gm = majority(g, rest, v)
    if gm is None:
        return rest[0]
    best = []
    for t in rest:
        if gm in set(comps(g, t, v)):
            best.append(t)
    return best[0] if best else rest[0]


def one_s(g, by, v, s, rest, mid_set, high_set, rng):
    if len(rest) < 1:
        return None, "rest"
    frame = list(comps(g, s, v))
    if len(frame) < 2:
        return None, "frame"
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set
    if not env_m:
        env_m = env - high_set
    if not env_m:
        return None, "env"
    fr_probe = set(comps(g, s, v))
    hit_ora = held in fr_probe
    t_mark = pick_mark(rest, g, v, env_m)
    t_rnd = rng.choice(rest)
    t_maj = pick_maj(rest, g, v)
    return dict(
        hit_ora=hit_ora,
        hit_mark=held in set(comps(g, t_mark, v)),
        hit_rnd=held in set(comps(g, t_rnd, v)),
        hit_maj=held in set(comps(g, t_maj, v)),
        n_rest=len(rest),
    ), None


def one_v(g, by, v, mid_set, high_set, rng, cap):
    sl = list(by[v])
    if len(sl) < 8:
        return [], Counter(slots=1)
    rng.shuffle(sl)
    rows, sk = [], Counter()
    n_try = max(1, min(cap, len(sl)))
    for s in sl[:n_try]:
        rest = [x for x in sl if x != s]
        if len(rest) < 7:
            sk["rest"] += 1
            continue
        row, why = one_s(g, by, v, s, rest, mid_set, high_set, rng)
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
            s0 = rng.randrange(len(pool) - L + 1)
            out.append(pool[s0:s0 + L])
    return out


def rows_of(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], None, 0, Counter(nograph=1)
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    rows, sk = [], Counter()
    for v in mid:
        rs, sv = one_v(g, by, v, mid_set, high_set, rng, args.cap_probe)
        rows.extend(rs)
        sk.update(sv)
    return rows, len(mid), sk


def pack(rows, key):
    n = len(rows)
    if n == 0:
        return dict(n=0, hit=0.0)
    return dict(n=n, hit=sum(r[key] for r in rows) / n)


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
    print(f"553 place-pick  corpus={path}  {kind}  cap={args.cap_probe}", flush=True)

    rows, n_mid, sk = [], 0, Counter()
    n_win_ok = 0
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        rs, nm, sv = rows_of(lines, args, rng)
        rows.extend(rs)
        n_mid += nm
        sk.update(sv)
        n_win_ok += 1 if nm else 0

    n = len(rows)
    if n == 0:
        print("VOID: no trials")
        rec = dict(seed=args.seed, corpus=kind, n=0, void=True, gate=False)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        prev[str(args.seed)] = rec
        out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
        return 0

    ora = pack(rows, "hit_ora")
    mark = pack(rows, "hit_mark")
    rnd = pack(rows, "hit_rnd")
    maj = pack(rows, "hit_maj")
    room = ora["hit"] - rnd["hit"]
    d_rnd = mark["hit"] - rnd["hit"]
    d_maj = mark["hit"] - maj["hit"]
    void = n < 40 or room <= 0.05
    gate = (not void) and d_rnd > 0.05

    print(f"mid_sum {n_mid}  wins {n_win_ok}  skip {dict(sk)}")
    print(f"n {n}  ORA {ora['hit']:.4f}  MARK {mark['hit']:.4f}  "
          f"RND {rnd['hit']:.4f}  MAJ {maj['hit']:.4f}")
    print(f"room {room:+.4f}  MARK-RND {d_rnd:+.4f}  MARK-MAJ {d_maj:+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: no ceiling or too few trials. Oracle does not beat coin.")
    elif d_rnd <= 0.05:
        print("\nMARK does not beat random place. Env overlap is not where to stand.")
    else:
        print("\nGO: env mark picks the right mention better than chance.")

    rec = dict(seed=args.seed, corpus=kind, cap=args.cap_probe,
               n_mid=n_mid, skip=dict(sk), n=n,
               hit_ora=ora["hit"], hit_mark=mark["hit"],
               hit_rnd=rnd["hit"], hit_maj=maj["hit"],
               room=room, d_rnd=d_rnd, d_maj=d_maj,
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
