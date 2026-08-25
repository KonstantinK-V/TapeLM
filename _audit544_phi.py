"""544: Phi walks instead of the Q-table.

The 5xx mind is a lookup over HAND-DESIGNED keys (pre = capped 4-tuple). This arm
swaps the lookup for a tiny Phi over RAW counts - no buckets, no caps, no words -
trained on the exact same reward stream (+1 / -0.08), same curriculum (narrow2 ->
eval unique, the 542 winner), same paired windows. One mechanism changes: the scorer.

Separation contract: features are counts only (sizes, distinct-key counts,
majority fractions). No token identity, no hash of a word, ever.

The read the table cannot survive: `probe_unseen` - eval places whose designed key
was NEVER seen in training. There the table is random BY CONSTRUCTION; if Phi beats
random on that subpool, it generalizes where the table has nothing.

    A_table    q[pre(P,g)]      the incumbent (542's winner)
    B_phi      Phi(counts(P,g)) same reward, same curriculum
    C_phinull  Phi trained on a shuffled reward stream (536rnd control)

GATE: paired AND lift_B - lift_C > 0.05 AND lift_B > lift_A - 0.05
      AND probe_phi - probe_rand > 0.05.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from math import log1p
from pathlib import Path

import torch

import _tape_frames as tframes
from _audit485_hunt import (
    build_window, load_lines, narrow_next, pick_by_q, pick_corpus, pre, touch,
    unique_next,
)
from _audit542_curric import rand_unique

OUT = Path("results/_stage544_phi.json")
FDIM = 7


def features(P, g):
    """Counts only. No word may enter this vector."""
    sl = g["slots_at"][P]
    vs = [g["value"][i] for i in sl]
    n_s = len(sl)
    n_k = len({k for i in sl for k in g["keys"][i]})
    cnt = {}
    for v in vs:
        cnt[v] = cnt.get(v, 0) + 1
    top = sorted(cnt.values(), reverse=True)
    m1 = top[0] / max(n_s, 1)
    m2 = (top[1] / max(n_s, 1)) if len(top) > 1 else 0.0
    feats = [log1p(n_s) / 5.0, log1p(n_k) / 5.0, m1, m2,
             n_k / max(n_s, 1) / 4.0, min(n_s, 12) / 12.0, min(n_k, 20) / 20.0]
    return feats


class TableScorer:
    def __init__(self, seed):
        self.q, self.tot, self.win = {}, defaultdict(int), defaultdict(float)

    def pick(self, pool, g, rng, eps):
        return pick_by_q(pool, self.q, lambda x: pre(x, g), rng, eps)

    def update(self, P, g, r, rng):
        touch(self.q, self.tot, self.win, pre(P, g), r)

    def n_keys(self):
        return len(self.q)


class RuleScorer:
    """409 over_single: pick the place with highest majority fraction m1. No learning."""

    def __init__(self, seed):
        self.seed = seed

    def pick(self, pool, g, rng, eps):
        if not pool:
            return None
        if eps and rng.random() < eps:
            return rng.choice(pool)
        best, br = None, -1e18
        for P in pool:
            sl = g["slots_at"][P]
            vs = [g["value"][i] for i in sl]
            if not vs:
                continue
            cnt = {}
            for v in vs:
                cnt[v] = cnt.get(v, 0) + 1
            m1 = max(cnt.values()) / len(vs)
            if m1 > br:
                br, best = m1, P
        return best if best is not None else rng.choice(pool)

    def n_keys(self):
        return 0


class PhiScorer:
    def __init__(self, seed, lr=0.05):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FDIM, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.n_upd = 0

    def score(self, P, g):
        with torch.no_grad():
            return float(self.net(torch.tensor(features(P, g))).item())

    def pick(self, pool, g, rng, eps):
        if not pool:
            return None
        if rng.random() < eps:
            return rng.choice(pool)
        best, br = None, -1e18
        for x in pool:
            s = self.score(x, g)
            if s > br:
                br, best = s, x
        return best

    def _step(self, P, g, r):
        pred = self.net(torch.tensor(features(P, g)))
        loss = (pred - r) ** 2
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.n_upd += 1

    def update(self, P, g, r, rng):
        self._step(P, g, r)

    def n_keys(self):
        return sum(p.numel() for p in self.net.parameters())


class PhiNullScorer(PhiScorer):
    """Same mass, association destroyed: trains on a reward drawn from the past."""

    def __init__(self, seed, lr=0.05):
        super().__init__(seed, lr)
        self.buf = []

    def update(self, P, g, r, rng):
        self.buf.append(r)
        r2 = self.buf[rng.randrange(len(self.buf))]
        self._step(P, g, r2)


def hunt(g, rng, scorer, eps, budget, mode, learn):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return 0, 0
    seen = set()
    n_touch = 0
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = scorer.pick(pool, g, rng, eps if learn else 0.0)
        if P is None:
            break
        seen.add(P)
        if mode == "unique":
            hit = unique_next(P, g) is not None
        else:
            hit = narrow_next(P, g, kmax=2) is not None
        if learn:
            scorer.update(P, g, 1.0 if hit else -0.08, rng)
            n_touch += 1
        if hit:
            return 1, n_touch
    return 0, n_touch


def train(lines, args, scorer):
    rng_win = random.Random(args.seed)
    rng_pol = random.Random(args.seed + 5)
    n = hits = touches = 0
    for i in range(args.train_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng_win, args.window, args.frame_max)
        if g is None:
            continue
        n += 1
        eps = max(0.05, 0.5 * (1 - i / max(args.train_steps, 1)))
        h, t = hunt(g, rng_pol, scorer, eps, args.budget, "narrow2", True)
        hits += h
        touches += t
    return dict(train_hit=hits / max(n, 1), n_train=n, n_touch=touches,
                n_keys=scorer.n_keys())


def evaluate(lines, args, scorer):
    rng_win = random.Random(args.seed + 77)
    rng_pol = random.Random(args.seed + 991)
    rng_rnd = random.Random(args.seed + 2024)
    n = hq = hr = 0
    for i in range(args.eval_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng_win, args.window, args.frame_max)
        if g is None:
            continue
        n += 1
        h, _ = hunt(g, rng_pol, scorer, 0.0, args.budget, "unique", False)
        hq += h
        hr += rand_unique(g, rng_rnd, args.budget)
    return dict(n_eval=n, eval_unique=hq / max(n, 1), eval_rand=hr / max(n, 1),
                lift=(hq - hr) / max(n, 1))


def probe_unseen(lines, args, phi, table):
    """Places whose designed key was never trained: table = random by construction."""
    K = set(table.q)
    rng_win = random.Random(args.seed + 77)
    rng_probe = random.Random(args.seed + 4242)
    n = ph = th = rh = 0
    for i in range(args.eval_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng_win, args.window, args.frame_max)
        if g is None:
            continue
        places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
        sub = [P for P in places if pre(P, g) not in K]
        if len(sub) < 4:
            continue
        n += 1
        Pp = phi.pick(sub, g, rng_probe, 0.0)
        Pt = table.pick(sub, g, rng_probe, 0.0)
        Pr = rng_probe.choice(sub)
        ph += int(unique_next(Pp, g) is not None)
        th += int(unique_next(Pt, g) is not None)
        rh += int(unique_next(Pr, g) is not None)
    return dict(probe_n=n, probe_phi=ph / max(n, 1), probe_table=th / max(n, 1),
                probe_rand=rh / max(n, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-steps", type=int, default=2500)
    ap.add_argument("--eval-steps", type=int, default=1500)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path = pick_corpus(args.corpus)
    lines = load_lines(path, args.bytes, 20, random.Random(args.seed))
    t0 = time.time()
    print(f"544 phi-walk  corpus={path}", flush=True)

    scorers = dict(A_table=TableScorer(args.seed),
                   B_phi=PhiScorer(args.seed),
                   C_phinull=PhiNullScorer(args.seed))
    arms = {}
    for name, sc in scorers.items():
        tr = train(lines, args, sc)
        ev = evaluate(lines, args, sc)
        arms[name] = dict(**tr, **ev)
        print(f"{name:10} train_hit {tr['train_hit']:.3f} touches {tr['n_touch']} "
              f"size {tr['n_keys']}  eval {ev['eval_unique']:.4f} "
              f"rand {ev['eval_rand']:.4f} lift {ev['lift']:+.4f}", flush=True)

    pr = probe_unseen(lines, args, scorers["B_phi"], scorers["A_table"])
    print(f"PROBE unseen-key  n {pr['probe_n']}  phi {pr['probe_phi']:.4f}  "
          f"table {pr['probe_table']:.4f}  rand {pr['probe_rand']:.4f}", flush=True)

    rands = {a["eval_rand"] for a in arms.values()}
    paired = len(rands) == 1
    lift_A = arms["A_table"]["lift"]
    lift_B = arms["B_phi"]["lift"]
    lift_C = arms["C_phinull"]["lift"]
    d_probe = pr["probe_phi"] - pr["probe_rand"]
    void = (not paired) or arms["B_phi"]["n_eval"] < 200 or pr["probe_n"] < 100
    gate = (not void) and lift_B - lift_C > 0.05 and lift_B > lift_A - 0.05 and d_probe > 0.05
    rec = dict(seed=args.seed, corpus=str(path), arms=arms, probe=pr,
               paired_rand=bool(paired), lift_A=lift_A, lift_B=lift_B,
               lift_C=lift_C, d_probe=d_probe,
               elapsed_s=round(time.time() - t0, 1),
               void=bool(void), gate=bool(gate))
    print(f"B-A {lift_B - lift_A:+.4f}   B-C {lift_B - lift_C:+.4f}   "
          f"probe phi-rand {d_probe:+.4f}   paired {paired}")
    print(f"VOID {void}   GATE {gate}")
    if not paired:
        print("\nVOID: arms saw different windows.")
    elif void:
        print("\nVOID: few eval windows or a starved probe.")
    elif lift_B - lift_C <= 0.05:
        print("\nPHI IS JITTER. The association does not carry it. Table stays.")
    elif lift_B <= lift_A - 0.05:
        print("\nPHI LOSES TO THE TABLE on seen keys. Buckets suffice here.")
    elif d_probe <= 0.05:
        print("\nNO GENERALIZATION: Phi is random where the table is random.")
    else:
        print("\nGO: Phi matches the table on seen keys AND lives where the "
              "table is random by construction.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
