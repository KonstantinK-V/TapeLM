"""548: hop2 is the PATH, not a new star.

547 took cheap_rec(красные) as hop2 — мак/закат, forbidden.
Here hop1 = rec[0] of v; hop2 = rec(v) ∩ rec(hop1) minus {v, hop1}.
Only companions that still sit with BOTH. Wave on whether that cut pays.

    A  hop1 only
    B  +hop2 if Φ > 0
    C  shuffled pairs
    D  +hop2 if peaked
    ALLGO  hop2 always (when intersect nonempty)

Teacher residual: held in hop2 and not in hop1.
GATE  B-A > 0.05 AND B-D > 0.05 AND B-C > 0.05
VOID  n_pos < 40 or n_test < 50 or B hit ≥ 0.99
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from math import log1p
from pathlib import Path

import torch

from _audit511_ring import cheap_rec, comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import allow_of

OUT = Path("results/_stage548_path.json")
FDIM = 7


def feat(n_sl, rec, cnt, allow, n_cut):
    n0 = cnt.get(rec[0], 0) if rec else 0
    n1 = cnt.get(rec[1], 0) if len(rec) > 1 else 0
    tot = max(sum(cnt.values()), 1)
    return [log1p(n_sl) / 5.0, log1p(len(rec)) / 5.0, n0 / tot, n1 / tot,
            min(n0 / max(n1, 1), 8.0) / 8.0, min(allow, 40) / 40.0,
            min(n_cut, 20) / 20.0]


def peaked_of(rec, cnt):
    if not rec:
        return False
    n0 = cnt.get(rec[0], 0)
    n1 = cnt.get(rec[1], 0) if len(rec) > 1 else 0
    return len(rec) == 1 or (n0 > 0 and n1 < 0.5 * n0)


def one_v(g, by, v, cache, k, high_set, rng):
    sl = list(by[v])
    if len(sl) < 8:
        return None
    rng.shuffle(sl)
    held_s, rest = sl[0], sl[1:]
    held = set(comps(g, held_s, v))
    if not held or not rest:
        return None
    saved = by[v]
    by[v] = list(rest)
    cache.pop(v, None)
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    allow = allow_of(g, v, k, high_set)
    if not rec:
        by[v] = saved
        return None
    h1 = rec[0]
    rec_h1 = [c for c in cheap_rec(g, by, h1, cache) if c != h1]
    by[v] = saved
    rec_set = set(rec)
    cut = [c for c in rec_h1 if c in rec_set and c != v]
    if v in high_set:
        cut = []
        allow = 1
    remain = max(0, allow - 1)
    hop2 = cut[:remain]
    cnt = Counter()
    for s in rest:
        cnt.update(set(comps(g, s, v)))
    hit1 = h1 in held
    hit2 = any(c in held for c in hop2)
    return dict(x=feat(len(rest), rec, cnt, allow, len(cut)),
                r1=[h1], r2=hop2, held=held,
                peaked=peaked_of(rec, cnt),
                residual=bool(hit2 and not hit1),
                h1=hit1, h12=hit1 or hit2)


class RankPhi:
    def __init__(self, seed, shuffle=False):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FDIM, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.opt = torch.optim.SGD(self.net.parameters(), lr=0.05)
        self.shuffle = shuffle

    def step(self, pos, neg, rng):
        if not pos or not neg:
            return
        k = min(len(pos), len(neg))
        pos, neg = rng.sample(pos, k), rng.sample(neg, k)
        if self.shuffle:
            both = pos + neg
            rng.shuffle(both)
            pos, neg = both[:k], both[k:k + k]
        loss = torch.relu(
            1.0 - (self.net(torch.tensor(pos, dtype=torch.float32)).squeeze(1)
                   - self.net(torch.tensor(neg, dtype=torch.float32)).squeeze(1))
        ).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def score(self, x):
        with torch.no_grad():
            return float(self.net(torch.tensor([x], dtype=torch.float32)).item())


def windows(pool, n_win, L, rng):
    out = []
    for _ in range(n_win):
        if len(pool) <= L:
            out.append(pool)
        else:
            s0 = rng.randrange(len(pool) - L + 1)
            out.append(pool[s0:s0 + L])
    return out


def rows_of(lines, args, rng, k_hold=None):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], None
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    high_set = set(high)
    k = 200.0 / max(g["n"], 1) if k_hold is None else k_hold
    cache = {}
    rows = []
    for v in mid:
        r = one_v(g, by, v, cache, k, high_set, rng)
        if r:
            rows.append(r)
    return rows, k


def hit_of(rows, go):
    n = max(len(rows), 1)
    hits = hops = 0
    for r in rows:
        take = list(r["r1"]) + (list(r["r2"]) if go(r) else [])
        hops += len(take)
        hits += int(any(c in r["held"] for c in take))
    return hits / n, hops / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng_win = random.Random(args.seed)
    rng_pol = random.Random(args.seed + 5)
    tr_w = windows(pool, args.n_train, args.window_lines, rng_win)
    te_w = windows(pool, args.n_test, args.window_lines, rng_win)
    t0 = time.time()
    print(f"548 path-hop2  corpus={path}  {kind}", flush=True)

    net_b = RankPhi(args.seed)
    net_c = RankPhi(args.seed, shuffle=True)
    n_pos = n_neg = n_tr = n_cut = 0
    k0 = None
    for lines in tr_w:
        rows, k = rows_of(lines, args, rng_pol, k0)
        if k0 is None:
            k0 = k
        if not rows:
            continue
        n_tr += 1
        n_cut += sum(1 for r in rows if r["r2"])
        pos = [r["x"] for r in rows if r["residual"]]
        neg = [r["x"] for r in rows if (not r["residual"]) and r["r2"]]
        n_pos += len(pos)
        n_neg += len(neg)
        net_b.step(pos, neg, rng_pol)
        net_c.step(pos, neg, rng_pol)
    print(f"train wins {n_tr}  pos {n_pos}  neg {n_neg}  rows_with_cut {n_cut}  k {k0}",
          flush=True)

    test = []
    for lines in te_w:
        rows, _k = rows_of(lines, args, random.Random(args.seed + 77), k0)
        test.extend(rows)
    print(f"test rows {len(test)}  with hop2 {sum(1 for r in test if r['r2'])}", flush=True)

    def go_b(r):
        return bool(r["r2"]) and net_b.score(r["x"]) > 0

    def go_c(r):
        return bool(r["r2"]) and net_c.score(r["x"]) > 0

    def go_d(r):
        return bool(r["r2"]) and r["peaked"]

    def go_a(_r):
        return False

    def go_all(r):
        return bool(r["r2"])

    arms = {}
    for name, fn in (("A_hop1", go_a), ("B_wave", go_b), ("C_null", go_c),
                     ("D_peak", go_d), ("ALLGO", go_all)):
        hit, hops = hit_of(test, fn)
        arms[name] = dict(n=len(test), hit=hit, hops=hops)
        print(f"{name:8} hit {hit:.4f}  hops {hops:.2f}", flush=True)

    la, lb = arms["A_hop1"]["hit"], arms["B_wave"]["hit"]
    lc, ld = arms["C_null"]["hit"], arms["D_peak"]["hit"]
    void = n_pos < 40 or len(test) < 50 or lb >= 0.99
    gate = (not void) and lb - la > 0.05 and lb - ld > 0.05 and lb - lc > 0.05
    rec = dict(seed=args.seed, corpus=kind, k=k0, n_pos=n_pos, n_neg=n_neg,
               n_train_win=n_tr, arms=arms, d_hop1=lb - la, d_peak=lb - ld,
               d_null=lb - lc, d_allgo=lb - arms["ALLGO"]["hit"],
               elapsed_s=round(time.time() - t0, 1),
               void=bool(void), gate=bool(gate))
    print(f"B-A {lb - la:+.4f}  B-D {lb - ld:+.4f}  B-C {lb - lc:+.4f}  "
          f"B-ALLGO {lb - arms['ALLGO']['hit']:+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: path-cut residual still rare, or perfect hit.")
    elif lb - la <= 0.05:
        print("\nPATH HOP2 DOES NOT PAY. Intersect does not add held.")
    elif lb - ld <= 0.05:
        print("\nPEAK CARRIES IT.")
    elif lb - lc <= 0.05:
        print("\nRANK IS JITTER.")
    else:
        print("\nGO: hop2 from rec(v)∩rec(hop1) adds held, beats hop1/peak/null.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
