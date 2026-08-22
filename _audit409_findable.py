"""CAN THE TAIL BE FOUND WITHOUT THE ANSWERS? The last thing between 407's ceiling and the loop.

407 says WHERE the mind stands matters: speaking at the B best places beats speaking anywhere by
+0.55..+0.64 (46.1). But the median place has value ZERO - the gain is a selection over a heavy
tail - and `value` is DEFINED by the answers, which a policy does not have. 408 then removed the
obvious observable: SIZE does not take that ceiling (-0.015 / +0.002 / -0.012), so it is not
"speak at the big hubs".

So one question decides whether 351's loop can be written at all:

    IS THERE ANY ANSWER-FREE OBSERVABLE THAT PREDICTS value(place)?

    target    value(place) - 407's, the share of its holes the walk from it already reaches.
              Used ONLY as a label; no feature may read it.
    features  ten counts of the place and its neighbourhood. No identifier as identity, no hole,
              no truth. Size is among them, as the control 408 already priced.
    fit       linear, on the places of ONE window; read on the places of a DISJOINT window - so
              the split is a different stretch of corpus, not a reshuffle of the same places.
    metric    the mean value of the top-B places BY PREDICTION, against the mean of all - exactly
              407's oracle/random, so the two numbers are directly comparable.

  VOID   the test window's oracle gain is itself under 0.05 - then there is no tail to find there
         and nothing below is about findability.
  GATE   learned - random > 0.05, AND learned above the best FITTED-SINGLE feature by > 0.05, or
         it is one count and 404's lesson applies. `recovered` = learned gain / oracle gain is
         reported beside them: it says how much of 407's ceiling an answer-free rule reaches.

If it fails, the loop has a ceiling it cannot climb: the good places exist and cannot be told from
the bad ones without asking, which is what a policy would have to do.

    python _audit409_findable.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import _audit390_address as A

FEATS = ("n_slots", "n_fill", "dominance", "n_nbr", "mean_ov", "n_lines", "width",
         "left_class", "right_class", "filler_freq", "shared_fill")


def place_feats(T, pid):
    """Ten counts. Not one of them reads the hole being asked about, or any identifier AS an
    identifier - `filler_freq` is how often this place's values occur on the tape, a number."""
    slots = T["places"][pid]
    prof = T["prof"][pid]
    n = len(slots)
    nb = Counter()
    for v in prof:
        for j in T["at_value"].get(v, ()):
            if j != pid:
                nb[j] += 1
    w, left, right = T["addrs"][pid]
    freq = [len(T["at_value"].get(v, ())) for v in prof]
    return [float(n), float(len(prof)),
            (max(prof.values()) / n) if n else 0.0,
            float(len(nb)),
            (sum(nb.values()) / len(nb)) if nb else 0.0,
            float(len({T["owner"][s] for s in slots})),
            float(w),
            float(len(T["by_left"].get(left, ()))),
            float(len(T["by_right"].get(right, ()))),
            (sum(freq) / len(freq)) if freq else 0.0,
            (sum(1 for f in freq if f >= 2) / len(freq)) if freq else 0.0]


def value_of(T, pid, args):
    slots = T["places"][pid]
    if len(slots) < 2:
        return None
    toks, owner = T["toks"], T["owner"]
    ok = 0
    for s in slots:
        truth = toks[s]
        own = {toks[x] for x in slots if x != s}
        qprof = Counter(toks[x] for x in slots if x != s)
        drop = set(T["on_line"][owner[s]])
        drop.discard(pid)
        walked = A.walk_order(T, pid, qprof, args.places, drop)
        ok += int(truth in A.fillers_of(T, walked, own)[:args.topm])
    return ok / len(slots)


def tape_of(lines, args):
    return A.build_tape(lines, args.frame_max, args.min_fillers)


def sample(T, args, rng):
    ids = list(range(len(T["places"])))
    rng.shuffle(ids)
    X, y = [], []
    for pid in ids[:args.max_places]:
        v = value_of(T, pid, args)
        if v is None:
            continue
        X.append(place_feats(T, pid))
        y.append(v)
    return X, y


def scale(X):
    if not X:
        return X, []
    m = [max(1e-9, max(abs(r[j]) for r in X)) for j in range(len(X[0]))]
    return [[r[j] / m[j] for j in range(len(r))] for r in X], m


def fit(X, y, epochs, lr, seed):
    if not X:
        return []
    w = [0.0] * len(X[0])
    rng, idx = random.Random(seed), list(range(len(X)))
    for _e in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            p = sum(wj * xj for wj, xj in zip(w, X[i]))
            g = p - y[i]
            for j, xj in enumerate(X[i]):
                if xj:
                    w[j] -= lr * g * xj
    return w


def topB(X, y, w, B):
    order = sorted(range(len(y)), key=lambda i: -sum(wj * xj for wj, xj in zip(w, X[i])))
    b = min(B, len(y))
    return sum(y[i] for i in order[:b]) / b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--places", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--max-places", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default="results/_stage409_findable.json")
    args = ap.parse_args()
    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    W = args.window_lines
    if len(lines) < 3 * W:
        print("corpus too short for two disjoint windows")
        return 1
    # TWO DISJOINT WINDOWS - a different stretch of corpus, not a reshuffle of the same places
    a0 = rng.randrange(len(lines) - 2 * W)
    b0 = a0 + W + rng.randrange(max(1, len(lines) - a0 - 2 * W))
    TA, TB = tape_of(lines[a0:a0 + W], args), tape_of(lines[b0:b0 + W], args)
    if not TA["places"] or not TB["places"]:
        print("no tape")
        return 1
    Xa, ya = sample(TA, args, random.Random(args.seed))
    Xb, yb = sample(TB, args, random.Random(args.seed + 1))
    Xa, _m = scale(Xa)
    Xb, _m2 = scale(Xb)
    rnd = sum(yb) / len(yb)
    oracle = topB(Xb, yb, [0.0] * len(Xb[0]), args.budget) if False else (
        sum(sorted(yb, reverse=True)[:min(args.budget, len(yb))]) / min(args.budget, len(yb)))
    w = fit(Xa, ya, args.epochs, args.lr, args.seed)
    learned = topB(Xb, yb, w, args.budget)
    rep = {"seed": args.seed, "n_train": len(ya), "n_test": len(yb), "random": rnd,
           "oracle": oracle, "learned": learned, "oracle_gain": oracle - rnd,
           "learned_gain": learned - rnd}
    rep["recovered"] = (rep["learned_gain"] / rep["oracle_gain"]) if rep["oracle_gain"] else 0.0
    best, bj = -9.9, -1
    for j in range(len(Xa[0])):
        wj = fit([[r[j]] for r in Xa], ya, args.epochs, args.lr, args.seed)
        full = [wj[0] if c == j else 0.0 for c in range(len(w))]
        s = max(topB(Xb, yb, full, args.budget),
                topB(Xb, yb, [-x for x in full], args.budget))
        if s > best:
            best, bj = s, j
    rep["best_single"], rep["best_single_feat"] = best, FEATS[bj] if bj >= 0 else ""
    rep["over_single"] = learned - best
    print(f"places  train {len(ya)}  test {len(yb)}   budget {min(args.budget, len(yb))}")
    print(f"VOID CHECK  oracle gain on the test window {rep['oracle_gain']:+.4f}  <- read first")
    print(f"FINDABLE    random {rnd:.4f}   learned {learned:.4f} ({rep['learned_gain']:+.4f})   "
          f"oracle {oracle:.4f}   recovered {rep['recovered']:.3f}")
    print(f"ABLATION    best fitted-single {best:.4f} ({rep['best_single_feat']})   "
          f"family over it {rep['over_single']:+.4f}")
    void = rep["oracle_gain"] <= 0.05
    gate = rep["learned_gain"] > 0.05 and rep["over_single"] > 0.05
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print("\n" + ("VOID: no tail on the test window." if void else
                  ("THE TAIL IS FINDABLE WITHOUT THE ANSWERS: an answer-free rule fitted on one "
                   "window picks the good places of another, above the best single count. The "
                   "loop has something to steer by." if gate else
                   "THE TAIL IS NOT FINDABLE" + (" by this family" if rep["learned_gain"] > 0.05
                   else "") + ": the good places exist and cannot be told from the bad ones "
                   "without asking, which is exactly what a policy would have to do.")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
