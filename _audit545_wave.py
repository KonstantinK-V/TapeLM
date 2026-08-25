"""545: the whole window learns at every step, and reach comes from global counts.

Until now `hunt` picked ONE place per window and stopped at the first hit - 542
spent ~8 touches on a window holding thousands of places. We read the whole tape
and learned from a single point. Here every place of the window gets a gradient on
every step, in one batch: the signal per pass rises by orders of magnitude at the
same reading cost. This is the only part of the layered plan that is free.

FAR INTERSECTIONS. Widening the window would only move the same locality. Instead
three GLOBAL counts enter the feature vector - how often the place's majority value
and its keys occur across the WHOLE line pool, not this window. A local place is
then scored by how the tape as a whole uses those tokens, so two distant places can
be judged by the same global structure. It is still counts only: no token identity
ever reaches Phi. And it is what makes the wave a wave - add lines to the tape and
the same local place scores differently.

WHAT TO EXPECT, WRITTEN BEFORE THE RUN. The base rate is ~0.007, so with +1/-0.08
the expected target per place is 0.007 - 0.993*0.08 = -0.072: EVERY value goes
negative. That is arithmetic, not a failure to learn. Only the ORDER carries
meaning here, and `mean_value` / `frac_pos` are printed so nobody reads the sign.

    B_wave   Phi over local+global counts, all places, batched
    C_null   the same, targets shuffled inside the batch (association destroyed)
    D_rule   argmax m1, no learning (409's over_single control)

Eval is identical to 542/544 (hunt, budget 16, unique) so the numbers compare.
GATE: paired AND lift_B - lift_D > 0.05 AND lift_B - lift_C > 0.05.
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

import _tape_frames as tframes
from _audit485_hunt import (
    build_window, load_lines, narrow_next, pick_corpus, unique_next,
)
from _audit542_curric import rand_unique
from _audit544_phi import RuleScorer, features

OUT = Path("results/_stage545_wave.json")
FDIM = 10


def global_counts(lines):
    """One pass over the whole pool. Counts only - no identity leaves this dict."""
    c = Counter()
    for ln in lines:
        c.update(ln.split())
    return c


def wide_features(P, g, gc, gtot):
    """The 544 local vector plus three counts that reach outside the window."""
    loc = features(P, g)
    sl = g["slots_at"][P]
    vs = [g["value"][i] for i in sl]
    top = Counter(vs).most_common(1)[0][0] if vs else ""
    ks = {k for i in sl for k in g["keys"][i]}
    gv = gc.get(top, 0)
    gk = sum(gc.get(k, 0) for k in ks) / max(len(ks), 1)
    local = len(sl)
    return loc + [log1p(gv) / 12.0, log1p(gk) / 12.0,
                  log1p(local) / max(log1p(gv), 1e-6) if gv else 1.0]


class Wave:
    def __init__(self, seed, lr=0.05, shuffle=False):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FDIM, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.shuffle = shuffle
        self.n_upd = self.n_row = 0
        self.sum_v = self.n_pos = 0.0

    def step(self, X, y, rng):
        if self.shuffle:
            idx = list(range(len(y)))
            rng.shuffle(idx)
            y = [y[i] for i in idx]
        xb = torch.tensor(X, dtype=torch.float32)
        yb = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        pred = self.net(xb)
        loss = ((pred - yb) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.n_upd += 1
        self.n_row += len(y)
        self.sum_v += float(pred.mean().item()) * len(y)
        self.n_pos += sum(1 for v in y if v > 0)

    def pick(self, pool, g, rng, eps, gc=None):
        if not pool:
            return None
        if eps and rng.random() < eps:
            return rng.choice(pool)
        X = torch.tensor([wide_features(P, g, gc, 0) for P in pool],
                         dtype=torch.float32)
        with torch.no_grad():
            s = self.net(X).squeeze(1)
        return pool[int(torch.argmax(s).item())]

    def n_keys(self):
        return sum(p.numel() for p in self.net.parameters())


def places_of(g):
    return [P for P, sl in g["slots_at"].items() if len(sl) >= 2]


def train_wave(lines, args, w, gc):
    tframes._KEEP_MEMO.clear()
    rng_win = random.Random(args.seed)
    rng_pol = random.Random(args.seed + 5)
    n = 0
    for i in range(args.train_steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng_win, args.window, args.frame_max)
        if g is None:
            continue
        ps = places_of(g)
        if len(ps) < 8:
            continue
        n += 1
        X = [wide_features(P, g, gc, 0) for P in ps]
        y = [1.0 if narrow_next(P, g, kmax=2) is not None else -0.08 for P in ps]
        w.step(X, y, rng_pol)
    return dict(n_train=n, n_upd=w.n_upd, n_row=w.n_row,
                batch=w.n_row / max(w.n_upd, 1),
                mean_value=w.sum_v / max(w.n_row, 1),
                frac_pos=w.n_pos / max(w.n_row, 1), n_keys=w.n_keys())


def evaluate(lines, args, scorer, gc):
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
                P = (scorer.pick(pool, g, rng_pol, 0.0, gc)
                     if isinstance(scorer, Wave)
                     else scorer.pick(pool, g, rng_pol, 0.0))
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
    ap.add_argument("--train-steps", type=int, default=800)
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
    gc = global_counts(lines)
    t0 = time.time()
    print(f"545 wave  corpus={path}  vocab {len(gc)}", flush=True)

    arms = {}
    scorers = dict(B_wave=Wave(args.seed), C_null=Wave(args.seed, shuffle=True),
                   D_rule=RuleScorer(args.seed))
    for name, sc in scorers.items():
        tr = (dict(n_train=0, n_upd=0, n_row=0, batch=0.0, mean_value=0.0,
                   frac_pos=0.0, n_keys=0) if name == "D_rule"
              else train_wave(lines, args, sc, gc))
        ev = evaluate(lines, args, sc, gc)
        arms[name] = dict(**tr, **ev)
        print(f"{name:8} batch {tr['batch']:.0f} rows {tr['n_row']} "
              f"mean_v {tr['mean_value']:+.4f} frac_pos {tr['frac_pos']:.4f}  "
              f"eval {ev['eval_unique']:.4f} rand {ev['eval_rand']:.4f} "
              f"lift {ev['lift']:+.4f}", flush=True)

    rands = {a["eval_rand"] for a in arms.values()}
    paired = len(rands) == 1
    lb, lc, ld = (arms["B_wave"]["lift"], arms["C_null"]["lift"],
                  arms["D_rule"]["lift"])
    void = (not paired) or arms["B_wave"]["n_eval"] < 200 or arms["B_wave"]["batch"] < 50
    gate = (not void) and lb - ld > 0.05 and lb - lc > 0.05
    rec = dict(seed=args.seed, corpus=str(path), arms=arms, paired_rand=bool(paired),
               lift_B=lb, lift_C=lc, lift_D=ld, d_null=lb - lc, d_rule=lb - ld,
               elapsed_s=round(time.time() - t0, 1), void=bool(void), gate=bool(gate))
    print(f"B-C {lb - lc:+.4f}   B-D {lb - ld:+.4f}   paired {paired}")
    print(f"VOID {void}   GATE {gate}")
    if not paired:
        print("\nVOID: arms saw different windows.")
    elif void:
        print("\nVOID: few eval windows, or the batch is not the whole window.")
    elif lb - lc <= 0.05:
        print("\nTHE WAVE IS JITTER. Shuffled targets do as well; association "
              "does not carry at this width.")
    elif lb - ld <= 0.05:
        print("\nONE COUNT CARRIES IT. argmax m1 matches the wave - 409's "
              "over_single, not learning.")
    else:
        print("\nGO: learning every place of the window beats both the single "
              "count and the shuffled association.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
