"""545a: layered waves. Depth is part of the target, and hop1 must open hop2.

545 collapsed: with 1.2% positives and an MSE target the minimum is the mean, so
both arms learned the SAME constant (-0.0681 vs -0.0679) and argmax over a flat
surface picked degenerately - below random. Two faults, both mine: the reading was
ordering while the training was absolute, and the batch was 99% negative. And the
global corpus counts were a frequency lexicon, which breaks the separation
contract and lumps every sense of a word together. All three are gone here.

    rank loss     softplus(-(s_pos - s_neg)) on pairs. Trains exactly what is read.
    balanced      every positive is paired with one sampled negative. A count.
    no gc         no corpus frequency anywhere. Reach must come from the graph.

AND THE AVALANCHE. In 545 a place carried one target no matter which hop reached
it, so depth was averaged away. Here:

    depth in the input   the target is a (place, hop) pair, so "good at depth 2"
                         never teaches "good at depth 1"
    chain condition      a hop1 counts only if it OPENS a hop2 that hits - the
                         449/450 n_follow rule, so becoming hop1 obliges leading
                         to hop2 of that same word
    layers in turn       pass 1 trains depth 1; pass 2 trains depth 2 on places
                         reached THROUGH the depth-1 places the net already ranks
                         high. The flood becomes a schedule.

TWO MECHANISMS, TWO NUMBERS. Rank-teaching and layering are separate claims, so
each arm is evaluated twice: `lift_hop1` is the one-step walk (the rank teacher
alone, against D's +0.302) and `lift_chain` is the two-step walk that actually
scores the second hop AT DEPTH 2. Without the second eval the layer would only
regularize shared weights and the gate would read "rank beats m1 at hop1", not
"the avalanche was structured". The gate is on the chain; the hop1 column is
printed so the rank teacher can be read apart from the layers.

`both_rate` = the share of the depth-2 pool whose target at depth 1 DISAGREES with
its target at depth 2. That is the smear itself. (Two earlier denominators were
unreadable: against all places of the window it is ~1 by construction, and against
pos1 it is bounded by |pos1|, which the chain condition holds near 1.) High means a
place is being taught two different things and the layers are load-bearing; low
means the hop index in the input would have sufficed.

    B_layer  the layered wave        C_null  the same net, never trained
    D_rule   argmax m1, no learning  (409 over_single; it scored +0.302 in 545)

Eval is identical to 542/544/545. GATE: paired AND B-D > 0.05 AND B-C > 0.05.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import _tape_frames as tframes
from _audit485_hunt import (
    build_window, load_lines, narrow_next, pick_corpus, unique_next,
)
from _audit542_curric import rand_unique
from _audit544_phi import RuleScorer, features

OUT = Path("results/_stage545a_layer.json")
FDIM = 8


def feat_d(P, g, d):
    """544's local counts plus the hop index. No corpus frequency, no identity."""
    return features(P, g) + [d / 2.0]


def places_of(g):
    return [P for P, sl in g["slots_at"].items() if len(sl) >= 2]


def hop1_of(P, g):
    r = narrow_next(P, g, kmax=2)
    return list(r[0]) if r else []


class Layered:
    def __init__(self, seed, lr=0.05, shuffle=False, frozen=False):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FDIM, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.shuffle = shuffle
        self.frozen = frozen
        self.n_pair = self.n_upd = 0

    def score(self, items, g):
        X = torch.tensor([feat_d(P, g, d) for P, d in items], dtype=torch.float32)
        with torch.no_grad():
            return self.net(X).squeeze(1)

    def learn(self, pos, neg, g, rng):
        """One rank step. pos/neg are lists of (place, depth)."""
        if self.frozen or not pos or not neg:
            return 0
        pairs = [(p, neg[rng.randrange(len(neg))]) for p in pos]
        if self.shuffle:
            pairs = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]
        Xp = torch.tensor([feat_d(P, g, d) for (P, d), _ in pairs], dtype=torch.float32)
        Xn = torch.tensor([feat_d(P, g, d) for _, (P, d) in pairs], dtype=torch.float32)
        loss = F.softplus(-(self.net(Xp) - self.net(Xn))).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.n_pair += len(pairs)
        self.n_upd += 1
        return len(pairs)

    def pick(self, pool, g, rng, eps, d=1):
        if not pool:
            return None
        if eps and rng.random() < eps:
            return rng.choice(pool)
        s = self.score([(P, d) for P in pool], g)
        return pool[int(torch.argmax(s).item())]

    def n_keys(self):
        return sum(p.numel() for p in self.net.parameters())


def train(lines, args, net):
    tframes._KEEP_MEMO.clear()
    rng_win = random.Random(args.seed)
    rng_pol = random.Random(args.seed + 5)
    n = n1 = n2 = 0
    both_num = both_den = 0
    for i in range(args.train_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng_win, args.window, args.frame_max)
        if g is None:
            continue
        ps = places_of(g)
        if len(ps) < 16:
            continue
        n += 1
        opens = {}
        for P in ps:
            cs = hop1_of(P, g)
            opens[P] = [Q for Q in cs if Q in g["slots_at"] and hop1_of(Q, g)]
        pos1 = [(P, 1) for P in ps if opens[P]]
        neg1 = [(P, 1) for P in ps if not opens[P]]
        n1 += net.learn(pos1, neg1, g, rng_pol)

        k = max(8, len(ps) // 4)
        top = [P for _, P in sorted(
            zip(net.score([(P, 1) for P in ps], g).tolist(), range(len(ps))),
            key=lambda t: -t[0])[:k]]
        top = [ps[j] for j in top]
        pool2 = {Q for P in top for Q in hop1_of(P, g) if Q in g["slots_at"]}
        if pool2:
            d1pos = {P for P, _ in pos1}
            pos2 = [(Q, 2) for Q in pool2 if hop1_of(Q, g)]
            d2pos = {Q for Q, _ in pos2}
            both_num += sum(1 for Q in pool2
                            if (Q in d1pos) != (Q in d2pos))
            both_den += len(pool2)
            neg2 = [(Q, 2) for Q in pool2 if not hop1_of(Q, g)]
            n2 += net.learn(pos2, neg2, g, rng_pol)
    return dict(n_train=n, pairs_d1=n1, pairs_d2=n2, n_upd=net.n_upd,
                both_rate=both_num / max(both_den, 1), n_keys=net.n_keys())


def pick_at(scorer, pool, g, rng, d):
    """Depth is honoured where the scorer has one; the rule arm has none."""
    if isinstance(scorer, Layered):
        return scorer.pick(pool, g, rng, 0.0, d)
    return scorer.pick(pool, g, rng, 0.0)


def evaluate(lines, args, scorer, chain):
    tframes._KEEP_MEMO.clear()
    rng_win = random.Random(args.seed + 77)
    rng_pol = random.Random(args.seed + 991)
    rng_rnd = random.Random(args.seed + 2024)
    n = hq = hr = n_second = 0
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
                P = pick_at(scorer, pool, g, rng_pol, 1)
                if P is None:
                    break
                seen.add(P)
                if unique_next(P, g) is not None:
                    hq += 1
                    break
                if chain:
                    nxt = [Q for Q in hop1_of(P, g) if Q in g["slots_at"]]
                    if nxt:
                        n_second += 1
                        Q = pick_at(scorer, nxt, g, rng_pol, 2)
                        if Q is not None and unique_next(Q, g) is not None:
                            hq += 1
                            break
        hr += rand_unique(g, rng_rnd, args.budget)
    return dict(n_eval=n, n_second=n_second, eval_unique=hq / max(n, 1),
                eval_rand=hr / max(n, 1), lift=(hq - hr) / max(n, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-steps", type=int, default=3000)
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
    print(f"545a layered  corpus={path}", flush=True)

    arms = {}
    scorers = dict(B_layer=Layered(args.seed),
                   C_null=Layered(args.seed, frozen=True),
                   D_rule=RuleScorer(args.seed))
    for name, sc in scorers.items():
        tr = (dict(n_train=0, pairs_d1=0, pairs_d2=0, n_upd=0, both_rate=0.0,
                   n_keys=0) if name == "D_rule" else train(lines, args, sc))
        e1 = evaluate(lines, args, sc, chain=False)
        ec = evaluate(lines, args, sc, chain=True)
        arms[name] = dict(**tr, lift_hop1=e1["lift"], lift_chain=ec["lift"],
                          eval_hop1=e1["eval_unique"], eval_chain=ec["eval_unique"],
                          eval_rand=ec["eval_rand"], n_eval=ec["n_eval"],
                          n_second=ec["n_second"])
        print(f"{name:8} pairs d1 {tr['pairs_d1']} d2 {tr['pairs_d2']} "
              f"both_rate {tr['both_rate']:.3f}  hop1 {e1['lift']:+.4f}  "
              f"chain {ec['lift']:+.4f} (2nd {ec['n_second']})", flush=True)

    rands = {a["eval_rand"] for a in arms.values()}
    paired = len(rands) == 1
    lb, lc, ld = (arms["B_layer"]["lift_chain"], arms["C_null"]["lift_chain"],
                  arms["D_rule"]["lift_chain"])
    h_b, h_d = arms["B_layer"]["lift_hop1"], arms["D_rule"]["lift_hop1"]
    void = ((not paired) or arms["B_layer"]["n_eval"] < 200
            or arms["B_layer"]["n_second"] < 200
            or arms["B_layer"]["pairs_d1"] < 1000 or arms["B_layer"]["pairs_d2"] < 1000)
    gate = (not void) and lb - ld > 0.05 and lb - lc > 0.05
    rec = dict(seed=args.seed, corpus=str(path), arms=arms, paired_rand=bool(paired),
               lift_B=lb, lift_C=lc, lift_D=ld, d_null=lb - lc, d_rule=lb - ld,
               hop1_B=h_b, hop1_D=h_d, d_rank=h_b - h_d,
               both_rate=arms["B_layer"]["both_rate"],
               elapsed_s=round(time.time() - t0, 1), void=bool(void), gate=bool(gate))
    print(f"CHAIN B-C {lb - lc:+.4f}  B-D {lb - ld:+.4f}   "
          f"HOP1 B-D {h_b - h_d:+.4f}   both_rate {rec['both_rate']:.3f}   "
          f"paired {paired}")
    print(f"VOID {void}   GATE {gate}")
    if not paired:
        print("\nVOID: arms saw different windows.")
    elif void:
        print("\nVOID: few eval windows, a starved second hop, or a layer under "
              "1000 pairs.")
    elif lb - lc <= 0.05:
        print("\nRANKING IS JITTER. An untrained net of the same shape does as "
              "well.")
    elif lb - ld <= 0.05:
        print("\nONE COUNT CARRIES IT. argmax m1 matches the layered wave.")
    else:
        print("\nGO on the chain. Read HOP1 B-D beside it: if that is also > 0.05 "
              "the rank teacher carries on its own; if it is ~0 the layers do.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
