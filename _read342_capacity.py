"""342a read: capability against the mind's size, with the invariant beside it at every point.

THE THREE COLUMNS ARE READ TOGETHER OR NOT AT ALL. A capability that rises while the transplant
gap opens is not a better mind, it is a mind that has started holding facts - and the size where
that happens is the number this sweep exists to find. A capability that rises while the shuffled
tape still carries signal is not a capability at all.

  CAPABILITY  walk-only pooled, PICK vs COUNT pooled, GATE-WO at 10% and 25%, route enrichment
  INVARIANT   336's paired native-vs-transplanted block, pooled. A SMALL z IS THE GOOD RESULT,
              so the discordant total is printed with it - 144 of 8000 is a powered null, 1 of
              402 is no measurement.
  NULL        the --shuffle-tape run at that size, read on hit_rate over ALL questions. Never
              on a conditional rate: shuffling collapses the walk-only subset itself, so
              hit_of_walk_only stays flat while the signal underneath it dies, and the first
              run of this sweep printed "105% of the real signal" because of exactly that.

AND CONVERGENCE, because a flat capability curve has two explanations and only one of them is
about capacity. A bigger mind at the same --train-steps may simply be undertrained; if the
probe's best step is the LAST step, the point is still improving and its capability is a lower
bound, not a measurement. Printed as NOT CONVERGED rather than left to be inferred.

    python _read342_capacity.py
    python _read342_capacity.py --tag 342news        # capability on news (the default)
    python _read342_capacity.py --tag 342wiki        # the same sizes read on wiki
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# BOTH CONVENTIONS. The stage writes results/stage289_decision_<tag>.json; the reports read in
# this project arrive as out/_stage289_decision_<tag>.json. A reader that knows only one of them
# would print "one size only" over a finished sweep, which is worse than an error.
RES = (Path("results"), Path("out"))


def z_of(b, c):
    return (b - c) / math.sqrt(b + c) if b + c else float("nan")


def load(pattern):
    by_d = defaultdict(list)
    files = sorted({f for d in RES if d.is_dir() for f in d.glob(pattern)})
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        r = (d.get("reach") or {}).get("held_out")
        if r:
            by_d[d.get("dim")].append((d, r))
    return by_d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="342news")
    args = ap.parse_args()

    cap = load(f"*stage289_decision_{args.tag}_d*_s*.json")
    null = load("*stage289_decision_342null_d*_s*.json")
    if not cap:
        print(__doc__)
        return 1

    rows = []
    print(f"{'d':>5} {'params':>8} {'n':>3} {'walk-only':>14} {'PICKvC':>10} "
          f"{'GATE-WO 10%':>12} {'25%':>7} {'route':>7} {'INVARIANT':>18} {'null hit':>9}")
    for d in sorted(x for x in cap if x):
        runs = cap[d]
        wb = wc = pb = pc = 0
        g10 = [0, 0]
        g25 = [0, 0]
        route, unconv = [], 0
        ob = oc = on = 0
        for rep, r in runs:
            wp = r.get("walk_only_paired") or {}
            wb += wp.get("mind_only", 0)
            wc += wp.get("rival_only", 0)
            wk = r.get("walk_only_pick") or {}
            pb += wk.get("vs_count_mind_only", 0)
            pc += wk.get("vs_count_rival_only", 0)
            gw = r.get("gate_walk_only") or {}
            for fr, acc in ((0.10, g10), (0.25, g25)):
                e = gw.get(f"{fr:.2f}")
                if e:
                    acc[0] += e["k"]
                    acc[1] += e["mind"]["yield"]
            rt = r.get("router") or {}
            if rt.get("mind_enrichment") == rt.get("mind_enrichment"):
                route.append(rt["mind_enrichment"])
            om = r.get("other_mind") or {}
            if om:
                ob += om["all"]["this_only"]
                oc += om["all"]["other_only"]
                on += om["all"]["n"]
            es = rep.get("early_stop") or {}
            # STILL IMPROVING AT THE LAST STEP: the point is a lower bound, not a measurement
            unconv += int(es.get("best_step") == es.get("total_steps"))
        nl = null.get(d) or []
        # THE NULL IS READ ON AN ABSOLUTE QUANTITY, and the first run of this sweep is why.
        # It read `hit_of_walk_only` - a rate CONDITIONAL on the walk reaching the truth - and
        # printed "the shuffled tape carries 105% of the real signal" at both sizes. It does
        # not: shuffling collapses the walk-only subset itself, from ~100 questions per seed to
        # ~28, and 5 hits of 28 is 0.179 while 17 of 94 is 0.181. THE DENOMINATOR MOVED, so the
        # ratio said nothing - which is the exact fault this project has caught in the aperture
        # (300), the window (305), min_fillers (305) and the dead --addresses flag (335), now in
        # a control rather than in a claim.
        #
        # `hit_rate` is over ALL questions, so it cannot be rescued by a shrinking subset, and
        # `walk_only_rate` is printed beside it because its collapse IS the null working.
        def m(rows_, k):
            return (sum(x[1].get(k, float("nan")) for x in rows_) / len(rows_)) if rows_ \
                else float("nan")
        nh, nwo = m(nl, "hit_rate"), m(nl, "walk_only_rate")
        real = m(runs, "hit_rate")
        nshare = (nh / real) if (real and nh == nh) else float("nan")
        inv = f"{ob}/{oc} z {z_of(ob, oc):+.2f}" if on else "no rival mind"
        print(f"{d:>5} {runs[0][0].get('params', 0):>8} {len(runs):>3} "
              f"{wb:>4}/{wc:<3} z{z_of(wb, wc):+5.2f} {pb:>4}/{pc:<3} "
              f"{(g10[1] / g10[0] if g10[0] else float('nan')):>12.4f} "
              f"{(g25[1] / g25[0] if g25[0] else float('nan')):>7.4f} "
              f"{(sum(route) / len(route) if route else float('nan')):>6.2f}x "
              f"{inv:>18} {nh:>7.4f}"
              + (f" ({nshare:.0%} of real hit, wo {nwo:.4f})" if nshare == nshare
                 else "")
              + (f"   NOT CONVERGED ({unconv}/{len(runs)})" if unconv else ""))
        rows.append({"d": d, "gate25": (g25[1] / g25[0]) if g25[0] else float("nan"),
                     "wz": z_of(wb, wc), "inv_z": z_of(ob, oc), "inv_n": ob + oc,
                     "null": nh, "null_share": nshare, "unconv": unconv})

    # ---- THE READING, printed rather than left to the eye -----------------------------------
    print()
    if len(rows) < 2:
        print("one size only: nothing to compare. Run at least two.")
        return 0
    base = rows[0]
    for e in rows[1:]:
        up = e["gate25"] > base["gate25"] + 1e-9
        # THE INVARIANT'S VERDICT, and a small z only counts when there were pairs to split
        gap = (abs(e["inv_z"]) >= 1.645) if e["inv_n"] >= 10 else None
        # DEAD MEANS DEAD: the shuffled tape may keep a fraction of the real signal, but half
        # of it is not a null, it is a second channel nobody declared.
        dead = not (e["null_share"] == e["null_share"]) or e["null_share"] < 0.5
        verdict = ("capability FLAT - capacity was not the wall at this size"
                   if not up else
                   "MEMORY, not mind - the transplant gap opened, and this size is the limit"
                   if gap else
                   "capacity was binding - grow, and keep testing" if gap is False else
                   "capability rose but the invariant is UNREADABLE here (too few discordant "
                   "pairs) - the point does not count until it is powered")
        print(f"d {base['d']:>4} -> {e['d']:<4} gate25 {base['gate25']:.4f} -> {e['gate25']:.4f}"
              f"   {verdict}")
        if not dead:
            print(f"          VOID at d={e['d']}: the shuffled tape carries "
                  f"{e['null_share']:.0%} of the real hit rate - nothing at this size may be "
                  f"read until that is explained")
        if e["unconv"]:
            print(f"          and {e['unconv']} of its runs were still improving at the last "
                  f"step, so its capability is a LOWER BOUND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
