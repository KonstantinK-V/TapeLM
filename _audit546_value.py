"""546: V(P) = r(P) + gamma * max V(Q) - reach without layers, without learning.

545a answered the layer question: `chain - hop1` was +0.020 for the trained net and
+0.019 for a dumb rule, so depth as a separate target carries nothing. And D2 kept
reading zero because it asked the tape to CHANGE. This asks nothing to change: the
value of a place is what it pays now plus the best thing it opens, propagated over
the graph by a few sweeps. Long links cost O(edges) per sweep, not O(paths).

    THE FIRST VERSION LEAKED, and its GATE is void. It used r = 1/|cands|, which
    is exactly 1.0 when |cands| == 1 - the very condition `unique_next` tests. The
    reward WAS the label, so argmax r meant "pick the place where the answer already
    is": A_now read 0.752 and B_val read 1.0000 on 1500 windows. Not a discovery,
    a lookup.

    r is now the 545a scorer over the place's own counts - size, distinct keys,
    majority shares - and nothing from the target: no |cands|, no narrow_next.

    THE SECOND VERSION LEAKED THROUGH THE EDGES, and its GATE is void too. Its
    neighbours came from `hop1_of` = narrow_next(P, kmax=2), so a place HAD a
    neighbour only when it was already nearly answered - and unique is a subset of
    narrow. `max(..., default=0.0)` then handed a bonus to exactly the places that
    were the target, and B_val read 0.994. The reward was clean; the graph was not.

    Neighbours are now plain adjacency - places that share a key or a value with P,
    read off the window's own index. Every place has them, so the bonus stops being
    a label and propagation has to be earned. C_flat is the control that proves it:
    the same sweep over the same edges with r held CONSTANT. If C_flat alone scores
    high, all the signal was in the edge set and nothing was learned.

    r(P) = Phi(counts(P))                    trained by 545a's rank loss
    V(P) = r(P) + g * max_Q [ V(Q) - c ]     Q in hop1_of(P), c the price of a hop

    The cost c is subtracted per step, so stopping falls out of the arithmetic: a
    hop that does not pay is not taken. Nothing in V can see the answer.

    A_now    pick by Phi alone (gamma = 0) - no propagation at all
    B_val    pick by V (gamma = 0.7, `sweeps` passes) - propagation
    C_flat   the same sweep with r = 1 everywhere - pure graph shape
    D_rule   argmax m1, the standing bar (+0.302 in 545a)

A_now is the ablation that matters: if B does not beat it, the chain buys nothing
and only the immediate payoff was ever working. Eval is the 542/544/545 walk
(hunt, budget 16, unique) so every number in the ladder compares.

GATE: paired AND B - A > 0.05 AND B - D > 0.05 AND B - C > 0.05.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import _tape_frames as tframes
from _audit485_hunt import build_window, load_lines, pick_corpus, unique_next
from _audit542_curric import rand_unique
from _audit544_phi import RuleScorer
from _audit545a_layer import Layered, places_of, train

OUT = Path("results/_stage546_value.json")


def neighbours(g, ps):
    """Plain adjacency: places that share a key or a value. Nothing from the target."""
    at = {}
    keep = set(ps)
    for P in ps:
        for i in g["slots_at"][P]:
            for t in list(g["keys"][i]) + [g["value"][i]]:
                at.setdefault(t, set()).add(P)
    return {P: [Q for t in {x for i in g["slots_at"][P]
                            for x in list(g["keys"][i]) + [g["value"][i]]}
                for Q in at.get(t, ()) if Q != P and Q in keep] for P in ps}


class Value:
    """Phi scores a place from its own counts; the sweep carries reach. No target
    quantity may enter r - that was the leak the first version shipped."""

    def __init__(self, net, gamma, sweeps, cost, flat=False):
        self.net = net
        self.flat = flat
        self.gamma = gamma
        self.sweeps = sweeps
        self.cost = cost
        self.cache = {}

    def table(self, g):
        key = id(g)
        if key in self.cache:
            return self.cache[key]
        ps = places_of(g)
        if self.flat:
            r = {P: 1.0 for P in ps}
        else:
            r = dict(zip(ps, self.net.score([(P, 1) for P in ps], g).tolist()))
        nbr = neighbours(g, ps)
        V = dict(r)
        for _ in range(self.sweeps if self.gamma else 0):
            V = {P: r[P] + self.gamma * max((V[Q] - self.cost for Q in nbr[P]),
                                            default=0.0) for P in ps}
        self.cache[key] = V
        return V

    def pick(self, pool, g, rng, eps):
        if not pool:
            return None
        V = self.table(g)
        return max(pool, key=lambda P: V.get(P, 0.0))


def evaluate(lines, args, scorer):
    tframes._KEEP_MEMO.clear()
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
        ps = places_of(g)
        if len(ps) >= 4:
            seen = set()
            for _ in range(args.budget):
                pool = [P for P in ps if P not in seen] or ps
                P = scorer.pick(pool, g, rng_pol, 0.0)
                if P is None:
                    break
                seen.add(P)
                if unique_next(P, g) is not None:
                    hq += 1
                    break
        hr += rand_unique(g, rng_rnd, args.budget)
    return dict(n_eval=n, eval_unique=hq / max(n, 1), eval_rand=hr / max(n, 1),
                lift=(hq - hr) / max(n, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-steps", type=int, default=1500)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--train-steps", type=int, default=3000)
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
    print(f"546 value-iteration  corpus={path}  gamma={args.gamma} "
          f"sweeps={args.sweeps} cost={args.cost}", flush=True)
    net = Layered(args.seed)
    tr = train(lines, args, net)
    print(f"phi trained  pairs d1 {tr['pairs_d1']} d2 {tr['pairs_d2']}", flush=True)

    arms = {}
    for name, sc in (("A_now", Value(net, 0.0, 0, args.cost)),
                     ("B_val", Value(net, args.gamma, args.sweeps, args.cost)),
                     ("C_flat", Value(net, args.gamma, args.sweeps, args.cost, True)),
                     ("D_rule", RuleScorer(args.seed))):
        ev = evaluate(lines, args, sc)
        arms[name] = ev
        print(f"{name:8} eval {ev['eval_unique']:.4f} rand {ev['eval_rand']:.4f} "
              f"lift {ev['lift']:+.4f}", flush=True)

    paired = len({a["eval_rand"] for a in arms.values()}) == 1
    la, lb = arms["A_now"]["lift"], arms["B_val"]["lift"]
    lc, ld = arms["C_flat"]["lift"], arms["D_rule"]["lift"]
    void = ((not paired) or arms["B_val"]["n_eval"] < 200
            or tr["pairs_d1"] < 1000
            or arms["A_now"]["eval_unique"] >= 0.999
            or arms["B_val"]["eval_unique"] >= 0.999)
    gate = (not void) and lb - la > 0.05 and lb - ld > 0.05 and lb - lc > 0.05
    rec = dict(seed=args.seed, corpus=str(path), gamma=args.gamma,
               sweeps=args.sweeps, cost=args.cost, phi=tr, arms=arms, paired_rand=bool(paired),
               lift_A=la, lift_B=lb, lift_C=lc, lift_D=ld,
               d_now=lb - la, d_rule=lb - ld, d_flat=lb - lc,
               elapsed_s=round(time.time() - t0, 1), void=bool(void),
               gate=bool(gate))
    print(f"B-A {lb - la:+.4f}   B-C {lb - lc:+.4f}   B-D {lb - ld:+.4f}   "
          f"paired {paired}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: unpaired windows, a starved Phi, or a perfect score - "
              "1.0000 means the target leaked into r again.")
    elif lb - la <= 0.05:
        print("\nPROPAGATION BUYS NOTHING. The immediate payoff was the whole "
              "signal; long links do not pay on this tape.")
    elif lb - lc <= 0.05:
        print("\nTHE GRAPH SHAPE CARRIES IT. A constant r sweeps just as well - "
              "the edges are doing the work, not Phi.")
    elif lb - ld <= 0.05:
        print("\nONE COUNT STILL CARRIES IT. argmax m1 matches the value sweep.")
    else:
        print("\nGO: what a place OPENS beats what it pays now, and beats the "
              "single count - reach without layers and without learning.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
