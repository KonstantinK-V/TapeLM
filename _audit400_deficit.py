"""THE DUAL OF SHAPE: DOES THE LINE CLOSE WHAT THE REST OF THE SCOPE LACKS? Torch-free.

399 asked "is the line LIKE the rest of the def" and the answer was a coin on three seeds, below
the SIZE confound every time (section 39). This is the dual question, and it is the second and
last object drawn from that failure: not similarity but DEFICIT - the line brings node types the
remainder does NOT have, because a body missing exactly this line is missing exactly what it does.

THE PHRASE, DECLARED BEFORE THE RUN

    deficit(S)   = how many of the line's AST node TYPES are ABSENT from S's remaining body
    twin(S)      = the scope of this file whose remaining body has the NEAREST line count to S's,
                   S itself excluded; ties by tape order, so the twin is deterministic and no
                   number moves with a draw
    score(S)     = deficit(S) - deficit(twin(S))

The twin is what makes this not the size prior wearing a new hat: a small scope lacks more types
for the same reason it holds fewer of everything, and subtracting a size-matched scope's deficit
removes precisely that. How well it matches is REPORTED (`twin_gap`), so "size was removed" is a
number rather than a claim.

THE DIRECTION IS DECLARED AND WILL NOT BE FLIPPED. The true scope is the one that MISSES what the
line brings, so the argmax of `score` is the attachment. If the measurement comes back symmetric
below the coin - which would mean the argmin wins - that is a FINDING and not a licence to turn
the sign around after the fact. Reversing a declared direction on seeing the number is the
after-the-fact rescue this project has refused four times.

NO IDENTIFIER IS READ. The feature is `type(node).__name__` and nothing else, exactly as in 399,
and `_check400_deficit.py` re-asserts the equivariance: rename every symbol, every number
unchanged.

ONE BIT, REPORTED SEPARATELY because a diffuse deficit can be noisy while a single closure is not:
a `Return` line offered to a scope whose REMAINING body has no `Return` at all. Its own accuracy
and its own population mass, beside the general score and never blended into it.

    POPULATION   only `amb_live` - the lines where the name count ties at a non-zero score, which
                 is 23% of the corpus and the only place a decision exists (38.3). The full
                 population is the PAIRED CONTROL, not the target.
    RIVALS       the coin (1/|tied|), the UNDIVIDED name overlap (a coin there by construction),
                 SIZE (the incumbent carrier, +0.03..+0.08 at 39), and the RAW deficit without
                 the twin - the control that says whether the twin did the work.

  VOID CHECK, READ FIRST
      `twin_gap` - the mean difference in remaining lines between a candidate and its twin. If the
      twin cannot match size, the subtraction did not remove the size prior and nothing below is
      about deficit.

  GATE, as Kostya set it
      score - coin > 0.05 on 3 of 3 seeds, AND score >= the name count.
      DECLARED READING, not a gate clause: if the score beats the coin but sits BELOW SIZE, that
      is section 39's verdict repeating in another currency and it is NOT a pass.

  If it fails: the pair (two pooled lines into one def) is a SEPARATE step with its own phrase, or
  the honest end - on code there is no evidence above a counter of scopes and a size prior. No 1b,
  no third form.

    python _audit400_deficit.py
    python _audit400_deficit.py --seed 8642
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _audit398_scope as S
import _audit399_shape as H

OUT = Path("results/_stage400_deficit.json")


def twin_of(rem, i):
    """The scope with the nearest remaining-line count, itself excluded. Deterministic: nearest
    first, then the lowest index, so the twin is a property of the file and not of a draw."""
    best, bj = None, None
    for j in range(len(rem)):
        if j == i:
            continue
        d = abs(rem[j] - rem[i])
        if best is None or d < best:
            best, bj = d, j
    return bj, (0 if bj is None else abs(rem[bj] - rem[i]))


def run(files, args, rng):
    c = Counter()
    for p in files:
        try:
            src = Path(p).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        sc = S.scopes_of(src)
        if not sc or len(sc["funcs"]) < args.min_scopes:
            continue
        line_t, body_t = H.types_of(src, sc["owner"])
        if line_t is None:
            continue
        nf = len(sc["funcs"])
        n_body = Counter(sc["owner"].values())
        lines = [ln for ln in sorted(sc["owner"]) if ln not in sc["heads"]]
        rng.shuffle(lines)
        taken = 0
        for ln in lines:
            if taken >= args.per_file:
                break
            names = sc["used"].get(ln, set())
            if not names:
                continue
            taken += 1
            true_i = sc["owner"][ln]
            counts = S.bind_counts(sc, ln)
            s_name = [S.score(names, S.bound_wo(sc, i, ln), counts, True) for i in range(nf)]
            s_raw = [S.score(names, S.bound_wo(sc, i, ln), counts, False) for i in range(nf)]
            top = max(s_name)
            tied = [i for i in range(nf) if s_name[i] == top]
            c["full_n"] += 1
            if len(tied) < 2 or top <= 0.0 or true_i not in tied:
                continue
            c["n"] += 1
            # THE REMAINING BODY: the pooled line is out of its own scope, in the line count and
            # in the type set alike - 398's discipline in both currencies.
            rem = [n_body[i] - (1 if i == true_i else 0) for i in range(nf)]
            trem = {i: H.types_wo(body_t, i, ln) for i in range(nf)}
            lt = line_t.get(ln, set())
            deficit = [len([t for t in lt if t not in trem[i]]) for i in range(nf)]
            s_def = []
            for i in range(nf):
                tj, gap = twin_of(rem, i)
                c["twin_gap"] += gap if i in tied else 0
                c["twin_n"] += 1 if i in tied else 0
                s_def.append(float(deficit[i] - (deficit[tj] if tj is not None else 0)))
            s_size = [float(rem[i]) for i in range(nf)]
            c["tied_size"] += len(tied)
            c["coin"] += 1.0 / len(tied)
            c["deficit"] += H.expected_acc(s_def, true_i, tied)
            c["deficit_raw"] += H.expected_acc([float(x) for x in deficit], true_i, tied)
            c["rawname"] += H.expected_acc(s_raw, true_i, tied)
            c["size"] += H.expected_acc(s_size, true_i, tied)
            # THE ONE BIT, on its own population: a Return line against a remainder with no Return
            if "Return" in lt:
                c["bit_n"] += 1
                bit = [1.0 if "Return" not in trem[i] else 0.0 for i in range(nf)]
                c["bit"] += H.expected_acc(bit, true_i, tied)
                c["bit_coin"] += 1.0 / len(tied)
                # IS THE BIT ABOUT THE LINE, OR ABOUT THE CUT? Only ONE line is removed here, so
                # exactly one scope acquires a hole. If every rival candidate has a Return in its
                # FULL body, then "no Return in the remainder" is true of the true scope and of
                # nothing else BY THE REMOVAL - the bit would be detecting our own cut. Measured,
                # because a +0.34 that means that is worth nothing.
                full = {i: H.types_wo(body_t, i, -1) for i in tied}
                c["bit_rivals"] += len(tied) - 1
                c["bit_rivals_noret"] += sum(1 for i in tied
                                             if i != true_i and "Return" not in full[i])
                c["bit_true_noret_full"] += int("Return" not in full[true_i])
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="_*.py")
    ap.add_argument("--per-file", type=int, default=40)
    ap.add_argument("--min-scopes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    files = sorted(Path(".").glob(args.files))
    c = run(files, args, rng)
    n, bn = max(1, c["n"]), max(1, c["bit_n"])
    rep = {"seed": args.seed, "files": len(files), "scored": c["full_n"], "decided_on": c["n"],
           "amb_live_with_truth": c["n"] / max(1, c["full_n"]),
           "tied_size": c["tied_size"] / n, "twin_gap": c["twin_gap"] / max(1, c["twin_n"]),
           "coin": c["coin"] / n, "deficit": c["deficit"] / n,
           "deficit_raw": c["deficit_raw"] / n, "rawname": c["rawname"] / n,
           "size": c["size"] / n, "bit_n": c["bit_n"], "bit": c["bit"] / bn,
           "bit_coin": c["bit_coin"] / bn,
           "bit_rivals_noret": c["bit_rivals_noret"] / max(1, c["bit_rivals"]),
           "bit_true_noret_full": c["bit_true_noret_full"] / bn}
    rep["deficit_minus_coin"] = rep["deficit"] - rep["coin"]
    rep["deficit_minus_name"] = rep["deficit"] - rep["rawname"]
    rep["deficit_minus_size"] = rep["deficit"] - rep["size"]
    rep["bit_minus_coin"] = rep["bit"] - rep["bit_coin"]

    print(f"{rep['files']} files, {rep['scored']} lines scored, decided on {rep['decided_on']} "
          f"({rep['amb_live_with_truth']:.4f} of them), {rep['tied_size']:.2f} tied scopes each")
    print(f"VOID CHECK  twin_gap {rep['twin_gap']:.3f} lines   <- read first: how far the "
          f"size-twin misses. If it cannot match size, nothing below is about deficit")
    print(f"TIE-BREAK   deficit {rep['deficit']:.4f}   coin {rep['coin']:.4f}   "
          f"raw name {rep['rawname']:.4f}   size {rep['size']:.4f}   "
          f"deficit w/o twin {rep['deficit_raw']:.4f}")
    print(f"            deficit-coin {rep['deficit_minus_coin']:+.4f}   "
          f"deficit-name {rep['deficit_minus_name']:+.4f}   "
          f"deficit-size {rep['deficit_minus_size']:+.4f}")
    print(f"ONE BIT     Return into a remainder with no Return: {rep['bit']:.4f} against "
          f"{rep['bit_coin']:.4f} on {rep['bit_n']} lines  ({rep['bit_minus_coin']:+.4f})")
    print(f"            IS IT THE CUT? rivals whose FULL body has no Return "
          f"{rep['bit_rivals_noret']:.4f}; the true scope's full body has none "
          f"{rep['bit_true_noret_full']:.4f}  <- if the first is ~0 the bit is detecting the "
          f"removal, not the line")

    gate = rep["deficit_minus_coin"] > 0.05 and rep["deficit"] >= rep["rawname"]
    under_size = rep["deficit"] < rep["size"]
    rep["gate"], rep["under_size"] = bool(gate), bool(under_size)
    if gate and not under_size:
        print("\nDEFICIT CARRIES THE TIE. The line closes what its own scope lacks, the "
              "size-twin removed the prior that carried 399, and the name count is a coin there. "
              "This is evidence about the LINE, and it is the first thing on this object that a "
              "counter does not already hold.")
    elif gate and under_size:
        print("\nNOT A PASS, BY THE READING DECLARED BEFORE THE RUN: deficit beats the coin but "
              "sits BELOW SIZE, which is section 39's verdict in another currency. The twin did "
              "not remove the prior.")
    else:
        print("\nDEFICIT DOES NOT CARRY THE TIE: " +
              ("it does not beat the coin. " if rep["deficit_minus_coin"] <= 0.05 else "") +
              ("the undivided name overlap does as well. " if rep["deficit"] < rep["rawname"]
               else "") +
              "The declared direction is not reversed on seeing the number. Next is the PAIR as "
              "its own step with its own phrase, or the honest end: on code there is no evidence "
              "above a counter of scopes and a size prior.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
