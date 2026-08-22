"""Check of 407, written after its number became load-bearing. No torch, no corpus.

  1. THE LEAK. The asking hole is out of its own place's profile (390's rule), or a place's value
     is measured through the answers it is being asked about.
  2. SAME-LINE PLACES ARE DROPPED - frames overlap, and a neighbour on the hidden slot's line is
     the same words twice.
  3. VALUE IS A SHARE OF THE PLACE'S OWN HOLES, so a fat place is not worth more for being fat.
  4. ORACLE AND RANDOM ARE READ ON THE SAME VECTOR AND THE SAME BUDGET - B best against the mean
     of all, which is the exact expectation of B random draws.
  5. THE SPREAD IS TOP DECILE MINUS MEDIAN, on the sorted values.

    python _check407_where.py
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("_audit407_where.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "qprof = Counter(toks[x] for x in slots if x != s)" not in src:
        f.append("1. the hidden slot is not taken out of the asking place's profile")
    if 'own = {toks[x] for x in slots if x != s}' not in src:
        f.append("1. the offer is not built with the hidden slot excluded")
    if 'drop = set(T["on_line"][owner[s]])' not in src or "drop.discard(pid)" not in src:
        f.append("2. same-line places are not dropped, or the asking place drops itself")
    if "ok / len(slots)" not in src:
        f.append("3. value is not a share of the place's own holes")
    if "oracle = sum(vals[:B]) / B" not in src or "rnd = sum(vals) / n" not in src:
        f.append("4. oracle and random are not the B best against the mean of all")
    if "vals.sort(reverse=True)" not in src:
        f.append("4/5. the values are not sorted, so the B best are not the B best")
    if '"spread": top10 - med' not in src:
        f.append("5. the spread is not top decile minus median")
    return f


MUTANTS = (
    ("the hidden slot stays in the profile",
     "            qprof = Counter(toks[x] for x in slots if x != s)",
     "            qprof = Counter(toks[x] for x in slots)", "1."),
    ("same-line places are kept",
     '            drop = set(T["on_line"][owner[s]])', "            drop = set()", "2."),
    ("value counts holes instead of their share",
     "        vals.append(ok / len(slots))", "        vals.append(float(ok))", "3."),
    ("random is not the expectation of B draws",
     "    rnd = sum(vals) / n", "    rnd = sum(vals[B:]) / max(1, n - B)", "4."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        if not any(g.startswith(tag) for g in props(src.replace(old, new, 1))):
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
