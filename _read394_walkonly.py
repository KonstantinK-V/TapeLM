"""THE CEILING OF 34.4, READ OFF DUMPS ALREADY IN HAND. No run, no torch, no corpus.

33-VOID cost this project a step because a ceiling that was already measured went unread. So the
walk_only lever starts here: every quantity that can void it is ALREADY REPORTED by the stage, and
this file is the reading, not a new measurement.

WHAT walk_only IS. `answerable and not truth_in_own` - the truth is NOT among the values already
standing at this hole, and it IS among the ones the walk reached. On that population staying is
arithmetically wrong, so it is the one population where the merge cannot be the answer and the
route has to decide. 34.4 is the proposal to train stay/go THERE and nowhere else.

WHAT THE STAGE ALREADY PRINTS, and where each number lands:

    walk_only_arrive     of the walk_only questions, how many got a step at all   THE ROUTER
    walk_only_pick.n     how many of them were stepped on                         THE SIZE
    walk_only_pick       mind / rival / count_rival on those                      THE PICK
    step_rate            how often the arm steps at all
    deep_only_rate       the truth reachable ONLY by the second read              W1'S CEILING
    hit_of_deep_only     and what the arm does there
    hit_of_own, ceiling  the other two halves of the population

THE FOUR VOID CHECKS, DECLARED HERE BEFORE ANY DUMP IS OPENED. Each one, if it fires, closes the
lever WITHOUT a training run - which is the entire point of reading first.

  V1  walk_only_arrive >= 0.95  ->  VOID. The router ALREADY steps where staying is wrong, so a
      term that teaches it to step there has nothing to teach. Whatever is lost on that
      population is lost in the PICK, and the route is not the lever.
  V2  the walk_only share of the exam < 0.02  ->  VOID. The masked gradient would be under 2% of
      the training signal: the arm is a slower control, and a null on it would say nothing about
      routing. (34.4 cites ~0.036, so this is expected to pass - it is here because expecting is
      not measuring.)
  V3  deep_only_rate <= 0.05  ->  W1 IS VOID, which is Kostya's own condition: depth as a
      decision needs a population where the second read UNIQUELY holds the truth.
  V4  count_rival_rate >= the mind's rate on walk_only  ->  the pick there is a counting problem,
      not a routing one, and the route lever is aimed at the wrong half.

POOLED AS COUNTS, NEVER AS MEANS OF MEANS. `walk_only_pick` carries integers, so the seeds pool
exactly; `walk_only_arrive` is a rate, so its population is recovered as pick.n / arrive and the
pooled rate is the sum of stepped over the sum of the population. A mean of four rates is a
different number and this project has been bitten by that shape before.

    python _read394_walkonly.py results/stage289_decision*391ctl*.json
    python _read394_walkonly.py results/stage289_decision*.json --held
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ARMS = ("held_out", "train_control")


def num(x):
    """a reported nan or a missing key must READ as missing, not as a zero that pools"""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def pull(path, arm):
    d = json.load(open(path, encoding="utf-8"))
    r = (d.get("reach") or {}).get(arm)
    if not r:
        return None
    pick = r.get("walk_only_pick") or {}
    arrive = num(r.get("walk_only_arrive"))
    stepped = pick.get("n")
    # THE POPULATION, RECOVERED EXACTLY: pick.n is the STEPPED walk_only questions and arrive is
    # their share of all walk_only questions, so the total is the one divided by the other. Kept
    # as a float and rounded only for printing - rounding before pooling would drift.
    pop = (stepped / arrive) if (arrive and stepped is not None and arrive > 0) else None
    return {
        "file": Path(path).name, "seed": d.get("seed"), "arm": arm,
        "arrive": arrive, "stepped": stepped, "pop": pop,
        "mind": pick.get("mind"), "rival": pick.get("rival"),
        "count_rival": pick.get("count_rival"),
        "step_rate": num(r.get("step_rate")),
        "deep_only": num(r.get("deep_only_rate")),
        "hit_of_deep_only": num(r.get("hit_of_deep_only")),
        "hit_of_own": num(r.get("hit_of_own")),
        "ceiling": num(r.get("ceiling")),
        "n_rows": r.get("n") or r.get("n_rows"),
        "moves": (d.get("reach") or {}).get("moves"),
        "move_teach": (d.get("reach") or {}).get("move_teach"),
    }


def main(argv) -> int:
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 1
    # out/ and results/ hold the same report under the same name; a duplicate does not merely
    # repeat, it halves the honest denominator. Deduped by NAME, as _read299 does.
    seen, uniq = set(), []
    for f in files:
        nm = Path(f).name.lstrip("_")
        if nm not in seen:
            seen.add(nm)
            uniq.append(f)
    if len(uniq) != len(files):
        print(f"note: dropped {len(files) - len(uniq)} duplicate report name(s)")
    arms = ("held_out",) if "--held" in argv else ARMS

    for arm in arms:
        rows = [g for g in (pull(f, arm) for f in uniq) if g]
        if not rows:
            continue
        print(f"\n=== {arm}  ({len(rows)} run(s)) ===")
        head = ("seed", "walk_only", "arrive", "stepped", "mind", "count",
                "step_rate", "deep_only")
        print("".join(h.rjust(11) for h in head))

        def cell(v, digits=4):
            """digits=None prints the value as it is; 0 is ZERO DECIMALS, not "no format" -
            the falsy-zero version of this line printed a recovered population as
            392.857142857."""
            if v is None:
                return "-".rjust(11)
            return (str(v) if digits is None else f"{v:.{digits}f}").rjust(11)

        for g in rows:
            print(cell(g["seed"], None) + cell(g["pop"], 0)
                  + cell(g["arrive"]) + cell(g["stepped"], None) + cell(g["mind"], None)
                  + cell(g["count_rival"], None) + cell(g["step_rate"]) + cell(g["deep_only"]))
        # POOLED AS COUNTS
        pop = sum(g["pop"] for g in rows if g["pop"])
        stepped = sum(g["stepped"] for g in rows if g["stepped"] is not None)
        mind = sum(g["mind"] for g in rows if g["mind"] is not None)
        cnt = sum(g["count_rival"] for g in rows if g["count_rival"] is not None)
        nrows = sum(g["n_rows"] for g in rows if g["n_rows"])
        arrive = (stepped / pop) if pop else float("nan")
        share = (pop / nrows) if nrows else float("nan")
        deep = [g["deep_only"] for g in rows if g["deep_only"] is not None]
        deep_r = (sum(deep) / len(deep)) if deep else float("nan")
        m_rate = (mind / stepped) if stepped else float("nan")
        c_rate = (cnt / stepped) if stepped else float("nan")
        print(f"\nPOOLED  walk_only {pop:.0f} of {nrows} rows ({share:.4f})  "
              f"arrive {arrive:.4f}  on the stepped: mind {m_rate:.4f}  count {c_rate:.4f}  "
              f"deep_only {deep_r:.4f}")

        v = []
        if not math.isnan(arrive) and arrive >= 0.95:
            v.append(f"V1 FIRED: arrive {arrive:.4f} >= 0.95 - the router already steps where "
                     f"staying is wrong. 34.4's lever has nothing to teach; the loss is in the "
                     f"pick.")
        if not math.isnan(share) and share < 0.02:
            v.append(f"V2 FIRED: walk_only is {share:.4f} of the exam - the masked gradient is "
                     f"under 2% of the signal and the arm is a slower control.")
        if not math.isnan(deep_r) and deep_r <= 0.05:
            v.append(f"V3 FIRED: deep_only {deep_r:.4f} <= 0.05 - depth as a decision (W1) is "
                     f"void on this tape, by Kostya's own condition.")
        if not (math.isnan(m_rate) or math.isnan(c_rate)) and c_rate >= m_rate:
            v.append(f"V4 FIRED: on walk_only the counting rival is {c_rate:.4f} against the "
                     f"mind's {m_rate:.4f} - the pick there is a counting problem, and the "
                     f"route lever is aimed at the wrong half.")
        for line in v:
            print("  " + line)
        if not v:
            print("  no void check fired: 34.4's lever may be run, under the gate in section 35.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
