"""D2: does coverage grow with the ROUND? The project's central bet, as one count.

Everything the walk does lives on a thin slice - uniq_base ~0.007, const_live ~0.13.
The bet written on day one is that the mind gets smarter as the tape grows, and
481/482 built the mechanism (a mark written on T, read by a later hop), but the
count was never taken. This is it.

    round 0   measure the SHARPNESS of the ORIGINAL places
    pin       for every answerable Q, unique_next(Q) = (P, v) derives v. The mark
              is written at a HOLE - a place whose own majority is too weak to
              answer (frac < weak_frac) and which does NOT already hold v - so the
              write creates an edge that did not exist. It goes in its own
              structure, never into the text (60's guard).

              THE FIRST VERSION WROTE v AT P ITSELF, and P already holds v by
              construction, so the index gained nothing and growth was impossible.
              THE SECOND VERSION WROTE INTO A HOLE and ADDED an edge - and write
              matched shuffle to five digits (-0.00083 both), which says coverage
              was reacting to the fact of an added edge and not at all to what the
              mark said. That is the metric, not the tape: `unique_next` demands
              EXACTLY one candidate, so any appended edge can only destroy
              uniqueness, at any threshold. Growth was forbidden by the measure -
              and contract v1 says it itself: UNIQUE is not the definition of walk.

              AND THE VALUE MUST BE DERIVED *FOR THAT HOLE*. An earlier draft
              paired a list of holes with a list of derived values by position -
              the two had nothing to do with each other, which is the second reason
              write kept tying shuffle. Now a hole H is pinned to v only when some
              answerable Q has unique_next(Q) = (H, v): the walk itself names H.

              AND THE MEASURE ITSELF WAS THE LAST FAULT. `unique_next` answers
              only when there is EXACTLY one candidate - a knife edge, where one
              edge flips a place entirely and nothing in between exists. Every D2
              variant read ~0 because a step function was being measured with a
              scalpel; there was no gradient to find, whatever was written.
              Contract v1 said it from the start: UNIQUE is not the definition of
              walk. Coverage is now graded:

                  sharp(P) = 1 / |cands(P)|    cov = mean sharp over places that
                                               have a value at all

              One candidate reads 1.0, two 0.5, five 0.2, so cutting an edge moves
              the number smoothly and a write can show a gradient. Still a count,
              no threshold, and consistent with STAR.

              SO THE MARK NOW RESOLVES INSTEAD OF APPENDING. "v stands at H" means
              H stops being a candidate for every OTHER value: the index loses
              edges rather than gaining them, and uniqueness can appear. It stays
              two-sided - a wrong pin removes the right candidate and breaks a
              neighbour - so a win is still earned against its own losses.
    round k   rebuild the value->places index INCLUDING marks, and count again -
              but only ever on the ORIGINAL places (60's second guard). Otherwise
              coverage "grows" by eating its own tail

Growth is two-sided and cannot be rigged: `unique_next` needs EXACTLY one candidate,
so a mark can make a place answerable (0 candidates -> 1) and can just as easily
destroy one (1 -> 2). A win has to be earned against its own losses.

    A_write     the derived values, resolving holes
    B_shuffle   the same marks at the same places, but each value is drawn at
                random from that round's pinned values. The mass is identical at
                round 1 and may diverge afterwards - a wrong mark changes what is
                answerable next round - so `pins` is printed per arm and the
                divergence is read, not assumed away
    C_none      no writing at all; its growth MUST be exactly 0, which is the
                invariant that proves the counting itself is honest

GATE: A grows by more than 0.01 AND A - B > 0.01, with C at exactly 0.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import _tape_frames as tframes
from _audit485_hunt import build_window, load_lines, pick_corpus

OUT = Path("results/_stageD2_round.json")


def pv_aug(P, g, marks, min_frac=0.6):
    """place_value of 485. A resolved place reads as its pinned value outright."""
    if P in marks:
        return marks[P].most_common(1)[0][0], 1.0
    c = Counter(g["value"][i] for i in g["slots_at"].get(P, ()))
    if not c:
        return None, 0.0
    v, n = c.most_common(1)[0]
    frac = n / sum(c.values())
    return (v if frac >= min_frac else None), frac


def bykey_aug(g, marks):
    """A mark RESOLVES a place: it keeps the pinned value there and drops the rest."""
    bk = {k: set(s) for k, s in g["by_key"].items()}
    for P, cnt in marks.items():
        keep = cnt.most_common(1)[0][0]
        for u, s in bk.items():
            if u != keep:
                s.discard(P)
        bk.setdefault(keep, set()).add(P)
    return bk


def uniq_aug(P, g, marks, bk):
    if len(g["slots_at"].get(P, ())) < 2:
        return None
    v, frac = pv_aug(P, g, marks)
    if v is None:
        return None
    cands = bk.get(v, set()) - {P}
    if len(cands) != 1:
        return None
    return next(iter(cands)), v, frac


def sharp(P, g, marks, bk):
    """1/|cands|: graded, so one cut edge moves it instead of flipping it."""
    v, _f = pv_aug(P, g, marks, 0.0)
    if v is None:
        return None
    k = len(bk.get(v, set()) - {P})
    return 1.0 / k if k else 0.0


def coverage(orig, g, marks, bk):
    vals = [x for x in (sharp(P, g, marks, bk) for P in orig) if x is not None]
    return sum(vals) / max(len(vals), 1)


def holes_of(g, orig, marks, weak_frac):
    """Places whose own majority is too weak to answer - what a derived value is for."""
    out = []
    for P in orig:
        _v, frac = pv_aug(P, g, marks, weak_frac)
        if _v is None:
            out.append((P, frac))
    out.sort(key=lambda t: -t[1])
    return [P for P, _ in out]


def rounds(g, arm, R, pin_frac, weak_frac, rng):
    """One window, R rounds. Returns the coverage curve over ORIGINAL places."""
    orig = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(orig) < 8:
        return None, 0, 0
    marks = {}
    bk = bykey_aug(g, marks)
    curve = [coverage(orig, g, marks, bk)]
    n_pin = n_edge = 0
    for _ in range(R):
        holes = set(holes_of(g, orig, marks, weak_frac))
        pins = []
        for Q in orig:
            u = uniq_aug(Q, g, marks, bk)
            if u is None:
                continue
            H, v, frac = u
            if frac >= pin_frac and H in holes:
                pins.append((H, v))
        if arm == "none":
            pins = []
        elif arm == "shuffle" and pins:
            vals = [v for _, v in pins]
            pins = [(H, vals[rng.randrange(len(vals))]) for H, _ in pins]
        for H, v in pins:
            n_edge += sum(1 for u, sset in bk.items() if u != v and H in sset)
            marks.setdefault(H, Counter())[v] += 1
        n_pin += len(pins)
        bk = bykey_aug(g, marks)
        curve.append(coverage(orig, g, marks, bk))
    return curve, n_pin, n_edge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--pin-frac", type=float, default=0.8)
    ap.add_argument("--weak-frac", type=float, default=0.6)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path = pick_corpus(args.corpus)
    lines = load_lines(path, args.bytes, 20, random.Random(args.seed))
    t0 = time.time()
    print(f"D2 coverage-by-round  corpus={path}  rounds={args.rounds}", flush=True)

    arms = {}
    for arm in ("write", "shuffle", "none"):
        tframes._KEEP_MEMO.clear()
        rng_win = random.Random(args.seed + 77)
        rng_arm = random.Random(args.seed + 5)
        acc = [0.0] * (args.rounds + 1)
        n = pins = edges = 0
        for i in range(args.windows):
            if (i + 1) % 50 == 0:
                tframes._KEEP_MEMO.clear()
            g = build_window(lines, rng_win, args.window, args.frame_max)
            if g is None:
                continue
            curve, np_, ne = rounds(g, arm, args.rounds, args.pin_frac,
                                    args.weak_frac, rng_arm)
            if curve is None:
                continue
            n += 1
            for j, c in enumerate(curve):
                acc[j] += c
            pins += np_
            edges += ne
        curve = [a / max(n, 1) for a in acc]
        arms[arm] = dict(n_win=n, curve=curve, cov0=curve[0], covR=curve[-1],
                         growth=curve[-1] - curve[0], pins=pins, edges_cut=edges)
        print(f"{arm:8} n {n}  cov {' '.join(f'{c:.5f}' for c in curve)}  "
              f"growth {curve[-1] - curve[0]:+.5f}  pins {pins}  "
              f"edges_cut {edges}", flush=True)

    gw, gs, gn = (arms["write"]["growth"], arms["shuffle"]["growth"],
                  arms["none"]["growth"])
    same_cov0 = len({round(arms[a]["cov0"], 9) for a in arms}) == 1
    void = (arms["write"]["n_win"] < 100 or arms["write"]["pins"] < 50
            or arms["write"]["edges_cut"] < 50
            or abs(gn) > 1e-9 or not same_cov0)
    gate = (not void) and gw > 0.01 and gw - gs > 0.01
    rec = dict(seed=args.seed, corpus=str(path), rounds=args.rounds,
               pin_frac=args.pin_frac, weak_frac=args.weak_frac,
               arms=arms, growth_write=gw,
               growth_shuffle=gs, growth_none=gn, d_shuffle=gw - gs,
               same_cov0=bool(same_cov0), elapsed_s=round(time.time() - t0, 1),
               void=bool(void), gate=bool(gate))
    print(f"write {gw:+.4f}   shuffle {gs:+.4f}   none {gn:+.4f}   "
          f"W-S {gw - gs:+.4f}   same_cov0 {same_cov0}")
    print(f"VOID {void}   GATE {gate}")
    if not same_cov0 or abs(gn) > 1e-9:
        print("\nVOID: the counting is not honest - arms start apart, or the "
              "no-write arm moved.")
    elif void:
        print("\nVOID: too few windows, too few pins, or the write cut no edges "
              "- then the arm cannot change anything by construction.")
    elif gw <= 0.01:
        print("\nWRITING DOES NOT GROW COVERAGE. What a mark opens, it also "
              "closes. The bet is not paid on this tape.")
    elif gw - gs <= 0.01:
        print("\nMASS, NOT TRUTH. Random values grow it as well - the tape is "
              "reacting to added marks, not to what they say.")
    else:
        print("\nGO: coverage of the ORIGINAL places grows round over round, and "
              "only when the written value is the derived one.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
