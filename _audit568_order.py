"""568: learn which of 2 cand to try first.

Features: (band, n_mentions, ov>0) — no word, no held.
Null = same rewards into a shuffled key.
567 try2 stays. Rank beside; beating it not required.
If Q ≈ rank → STOP, 567 remains order teacher.

VOID  n_two_te < 20
GATE  Q − random > 0.05  and  Q − null > 0.05

    python _check568_order.py
    python _audit568_order.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band

OUT = Path("results/_stage568_order.json")


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


def ov_of(g, by, a, env_m):
    _t, fr = stand(g, by, a, env_m)
    return len((fr or set()) & env_m)


def feat(a, by, mid_set, high_set, ov):
    n = min(len(by.get(a, [])), 8) // 2
    band = 2
    if a in high_set:
        band = 1
    elif a in mid_set:
        band = 0
    return (band, n, int(ov > 0))


def collect_two(g, by, mid_set, high_set, rng, cap):
    rows = []
    keys = list(mid_set)
    rng.shuffle(keys)
    for v in keys:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        for s in sl[:cap]:
            frame = list(comps(g, s, v))
            if len(frame) < 3:
                continue
            held, env = frame[0], set(frame[1:])
            env_m = (env & mid_set) - high_set or (env - high_set)
            if not env_m:
                continue
            rest = [x for x in by[v] if x != s]
            if len(rest) < 2:
                continue
            t, place = stand(g, by, v, env_m)
            if t is None:
                continue
            if held in place:
                continue
            cand = [c for c in place if c in mid_set and c != v]
            if len(cand) != 2:
                continue
            a, b = cand[0], cand[1]
            oa, ob = ov_of(g, by, a, env_m), ov_of(g, by, b, env_m)
            ha, hb = hop_hit(g, by, a, env_m, held), hop_hit(g, by, b, env_m, held)
            rows.append(dict(a=a, b=b, oa=oa, ob=ob, ha=ha, hb=hb,
                             fa=feat(a, by, mid_set, high_set, oa),
                             fb=feat(b, by, mid_set, high_set, ob)))
    return rows


def pick(row, Q, rng, mode):
    fa, fb = row["fa"], row["fb"]
    if mode == "rand":
        return "a" if rng.random() < 0.5 else "b"
    if mode == "rank":
        if row["oa"] != row["ob"]:
            return "a" if row["oa"] > row["ob"] else "b"
        return "a" if rng.random() < 0.5 else "b"
    qa, qb = Q.get(fa, 0.0), Q.get(fb, 0.0)
    if qa > qb:
        return "a"
    if qb > qa:
        return "b"
    if row["oa"] != row["ob"]:
        return "a" if row["oa"] > row["ob"] else "b"
    return "a" if rng.random() < 0.5 else "b"


def first_hit(row, who):
    return bool(row["ha"] if who == "a" else row["hb"])


def train(rows, rng, null=False):
    Q = {}
    for r in rows:
        who = pick(r, Q, rng, "q")
        hit = first_hit(r, who)
        key = r["fa"] if who == "a" else r["fb"]
        if null:
            key = (key[0], rng.randrange(4), key[2])
        Q[key] = Q.get(key, 0.0) + (1.0 if hit else -0.08)
    return Q


def eval_hit(rows, Q, rng, mode):
    if not rows:
        return 0.0
    return sum(first_hit(r, pick(r, Q, rng, mode)) for r in rows) / len(rows)


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
    ap.add_argument("--n-win", type=int, default=12)
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
    print(f"568 order-Q  corpus={path}  {kind}", flush=True)
    wins = windows(pool, args.n_win, args.window_lines, rng)
    cut = max(1, int(0.7 * len(wins)))
    tr, te = [], []
    for i, lines in enumerate(wins):
        g = graph(lines, args.frame_max, args.min_fillers)
        if g is None:
            continue
        by = mentions(g)
        mid, high, _a, _b = pct_band(g, by)
        rows = collect_two(g, by, set(mid), set(high), rng, args.cap_probe)
        (tr if i < cut else te).extend(rows)
    ntr, nte = len(tr), len(te)
    Q = train(tr, random.Random(args.seed + 1), null=False)
    Qn = train(tr, random.Random(args.seed + 2), null=True)
    rng_e = random.Random(args.seed + 3)
    h_q = eval_hit(te, Q, rng_e, "q")
    rng_e = random.Random(args.seed + 3)
    h_n = eval_hit(te, Qn, rng_e, "q")
    rng_e = random.Random(args.seed + 3)
    h_r = eval_hit(te, {}, rng_e, "rank")
    rng_e = random.Random(args.seed + 3)
    h_u = eval_hit(te, {}, rng_e, "rand")
    void = nte < 20
    gate = (not void) and (h_q - h_u > 0.05) and (h_q - h_n > 0.05)
    print(f"train {ntr}  test {nte}  Q {h_q:.3f}  rank {h_r:.3f}  "
          f"null {h_n:.3f}  rand {h_u:.3f}  Q-U {h_q-h_u:+.3f}  Q-N {h_q-h_n:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: few 2-cand on test. Stories, not a dead learner.")
    elif not gate:
        print("\nSTOP: order-Q not above random+null. 567 rank stays.")
    else:
        print("\nGO ORDER. Q picks first better than coin and shuffled keys.")
    rec = dict(seed=args.seed, corpus=kind, n_train=ntr, n_test=nte,
               hit_q=h_q, hit_rank=h_r, hit_null=h_n, hit_rand=h_u,
               d_rand=h_q - h_u, d_null=h_q - h_n, d_rank=h_q - h_r,
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
