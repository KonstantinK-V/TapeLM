"""338: WHAT SHOULD THE TAPE KEEP - four rules at one budget, read side by side.

WHY RETENTION AND NOT MORE CORPUS. 335 measured the tape holding 0.807 of the hidden truths
while the offer showed the mind 0.217 of them, and swept both of the tape's sizes. Across WIDTH
the gap opened monotonically, 0.369 -> 0.590, with the share shown FALLING as the tape grew;
across depth at fixed width it was non-monotone and saturated. Growing the corpus makes the
enumeration worse, not better. So the remaining lever is which places are kept at all - and
that is a decision, which belongs to the mind's half of the split.

WHAT IS COMPARED. Every run draws its questions from the WHOLE tape and differs only in which
places the walk may visit and offer from, capped at the same N. Four rules:

    --retain-by random   what the tape does today, and the control the others must beat
    --retain-by own      the places with the most mentions
    --retain-by share    the places whose single most frequent filler dominates them
    --retain-by mind     the places where Phi answers with the widest margin

THE NUMBER THAT DECIDES is `reachable_rate` at matched N: of the same questions, how many have
their truth somewhere in the walk's offer. It is a property of the tape and the walk, with no
sampling noise once the seed is fixed - the only draw is the tape itself. So the statistic is
SIGN CONSISTENCY ACROSS SEEDS, not a z: a rule wins if it is ahead of `random` on at least 3
of 4 seeds. `hit_rate` is printed beside it because reaching more is worthless if the mind
cannot then pick, and a retention rule that raises reach while lowering the pick has moved the
problem rather than solved it.

THE GUARD, written before the first run and not negotiable afterwards. `mind` requires a frozen
transplanted Phi (the stage refuses otherwise). The test is the transplant: choose what a NEWS
tape keeps using a mind fitted only to wiki. If the foreign tape comes out no worse than the
counting rules build it, the retention policy is not corpus-specific and the invariant holds.
If it comes out worse, the idea is dropped whole. There is no rescuing patch for this one.

    python _read338_retention.py out/_stage289_decision_338*_s1337.json
    python _read338_retention.py out/_stage289_decision_338*.json --held
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

# What the retained tape is worth, in the order it has to be read: how much truth is in front
# of the mind, then how much of it the mind takes, then whether it went anywhere at all.
COLS = ("reachable_rate", "ceiling", "own_hit_rate", "walk_only_rate",
        "hit_rate", "hit_of_walk_only", "step_rate")


def main(argv) -> int:
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 1
    arms = ["held_out"] if "--held" in argv else ["held_out", "train_control"]
    # (arm, rule) -> {seed: row}. Keyed by seed so the sign test below compares LIKE TAPES:
    # two rules read on different seeds are two tapes, and their difference is the draw.
    got = defaultdict(dict)
    budgets, addrs = set(), set()
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        rc = d.get("reach") or {}
        rule = d.get("retain_by") if d.get("retain") else "whole tape"
        if d.get("retain"):
            budgets.add(d["retain"])
        # THE TAPE'S SIZE, not the walk's. The first version of this line read `reach.places`,
        # which is REACH_K - the width of the WALK - and printed "[8] places built, 750
        # retained", a sentence that cannot be true. The retained fraction is the whole context
        # for every number below, so getting it from the wrong field made the table unreadable.
        addrs.add(((d.get("tape_shape") or {}).get("held_out") or {}).get("addresses"))
        for a in arms:
            r = rc.get(a)
            if r:
                got[(a, rule)][d["seed"]] = r
    if len(budgets) > 1:
        # MATCHED BUDGET OR NO COMPARISON. A rule keeping more places reaches more truth for a
        # reason that has nothing to do with the rule - the two-factor mistake this project has
        # made three times, and the one retention is most exposed to.
        print(f"NOT COMPARED: {sorted(budgets)} different --retain budgets among these files. "
              f"Retention rules are compared at ONE N; anything else is comparing tape sizes.")
        return 1
    n = (sorted(budgets) or ["all"])[0]
    built = sorted(x for x in addrs if x)
    print(f"tape   {built} addresses built, {n} retained "
          f"({n / built[0]:.2f} of the tape)" if built and isinstance(n, int) else
          f"tape   {n} retained")
    print(f"       {len(files)} files")
    for a in arms:
        rules = sorted({r for (arm, r) in got if arm == a})
        if not rules:
            continue
        print(f"\n[{a}]  {'rule':<12}" + "".join(f"{c.replace('_rate',''):>15}" for c in COLS))
        for rule in rules:
            per = got[(a, rule)]
            m = {c: sum(v.get(c, float('nan')) for v in per.values()) / len(per) for c in COLS}
            print(f"       {rule:<12}" + "".join(f"{m[c]:15.4f}" for c in COLS)
                  + f"   ({len(per)} seeds)")
        # THE SIGN TEST, per seed, against the control. Averages across seeds hide a rule that
        # wins hugely on one tape and loses on three - which is a tape, not a rule.
        base = got[(a, "random")] or got[(a, "whole tape")]
        if not base:
            print("       no `random` control among these files: nothing to compare against")
            continue
        for rule in rules:
            if rule in ("random", "whole tape"):
                continue
            per = got[(a, rule)]
            shared = sorted(set(per) & set(base))
            if not shared:
                continue
            up = sum(1 for s in shared
                     if per[s]["reachable_rate"] > base[s]["reachable_rate"])
            hup = sum(1 for s in shared if per[s]["hit_rate"] > base[s]["hit_rate"])
            print(f"       {rule:<12} vs random: reach ahead on {up}/{len(shared)} seeds, "
                  f"hit ahead on {hup}/{len(shared)}"
                  f"{'   PASSES (>=3 of 4)' if up >= 3 and len(shared) >= 4 else ''}")
    print("\nreachable = truth somewhere in the offer; hit = the mind said it. "
          "A rule that raises the first and lowers the second has moved the problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
