"""Static and behavioural check of 376's copy lane. No torch, no corpus.

Six properties, each of which would be a silent wrong answer rather than a crash:

  1. THE QUESTION'S OWN LINE IS NEVER READ. The hidden value stands on it.
  2. Values already at this hole are the recall channel's and are not re-offered.
  3. A token that is on no place is not offered - it has no rows and cannot be a world.
  4. A value with no OUTSIDE mentions is not offered - it would zero the shared import budget
     for every world in the question and turn "no rows" into a tell for "this is the answer".
  5. Rank is a count: occurrences first, nearest line on a tie.
  6. The lane is INTERLEAVED at the unchanged cap, never appended (347).

    python _check376_copy.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("_stage289_derivation.py")


class FakeTape:
    def __init__(self, values):
        self.values = values


def load_lane():
    """pull reach_copy, copy_index, by_value and outside_mentions out of the stage without
    importing it - the stage needs torch and the point is to check the lane, not the run."""
    src = SRC.read_text(encoding="utf-8")
    ns = {"Counter": Counter, "defaultdict": defaultdict}
    for name in ("copy_index", "reach_copy", "by_value", "outside_mentions"):
        m = re.search(rf"^def {name}\(.*?(?=\n(?:def |@|\w))", src, re.S | re.M)
        if not m:
            print(f"FAIL: {name} not found in {SRC}")
            sys.exit(1)
        exec(compile(m.group(0), name, "exec"), ns)
    return ns


def main() -> int:
    ns = load_lane()
    reach_copy = ns["reach_copy"]

    # a hand-made pack. Lines 10..14; the question's hole is on line 12.
    texts = {10: "alpha alpha beta zzz", 11: "gamma zzz", 12: "delta secret ghost",
             13: "beta ghost", 14: "alpha eps"}
    # slots: one per tape value occurrence, with the line it sits on
    slots = [("alpha", 10), ("alpha", 10), ("beta", 10), ("gamma", 11), ("delta", 12),
             ("secret", 12), ("beta", 13), ("eps", 14), ("alpha", 14)]
    values = [v for v, _li in slots]
    line = [li for _v, li in slots]
    p = {"tape": FakeTape(values), "line": line,
         "texts": [texts[li] for li in line]}
    # the question: its place holds `delta` as evidence and hides slot 5 (`secret`) on line 12
    q = {"slots": [4, 5], "query_row": 1}

    ns["COPY"], ns["COPY_D"] = True, 2
    got = reach_copy(p, q, 8)
    names = [v for v, _r, _n in got]
    fails = []

    if "secret" in names:
        fails.append("1. the question's OWN LINE leaked - `secret` was offered")
    if "delta" in names:
        fails.append("2. `delta` is already at this hole and was offered again")
    if "zzz" in names:
        fails.append("3. `zzz` stands on no place and was offered")
    if "ghost" in names:
        fails.append("3. `ghost` stands on no place and was offered")

    # 4: a value whose ONLY mention is inside this question's evidence must not be offered.
    p2 = {"tape": FakeTape(values + ["lone"]), "line": line + [13],
          "texts": [texts[li] for li in line] + [texts[13] + " lone"]}
    q2 = {"slots": [4, 5, 9], "query_row": 1}
    ns["COPY_D"] = 2
    if "lone" in [v for v, _r, _n in reach_copy(p2, q2, 8)]:
        fails.append("4. `lone` has no outside mention and would zero the import budget")

    # 5: alpha stands twice (line 10) and once (line 14, outside D=2 of line 12 -> not counted);
    #    beta stands on 10 and 13. Within +-2 of line 12: lines 10,11,13,14.
    #    alpha 10,10 + 14 = 3 ; beta 10 + 13 = 2 ; gamma 11 = 1 ; eps 14 = 1
    want = ["alpha", "beta"]
    if names[:2] != want:
        fails.append(f"5. rank is not count-then-distance: got {names[:2]}, want {want}")
    counts = [n for _v, _r, n in got]
    if counts != sorted(counts, reverse=True):
        fails.append(f"5. counts are not descending: {counts}")
    # gamma (line 11, distance 1) must precede eps (line 14, distance 2) at equal count 1
    if "gamma" in names and "eps" in names and names.index("gamma") > names.index("eps"):
        fails.append("5. nearest line does not break a tie")

    # 6: the cap is honoured and the lane is interleaved, not appended, in reach_candidates
    if len(reach_copy(p, q, 2)) > 2:
        fails.append("6. the lane ignores its cap")
    src = SRC.read_text(encoding="utf-8")
    body = re.search(r"^def reach_candidates\(.*?(?=\n(?:def |@|\w))", src, re.S | re.M).group(0)
    if "lanes.append([v for v, _r, _n in cop])" not in body:
        fails.append("6. the copy lane is not entered as a lane")
    if "cands = cands[:REACH_CANDS]" not in body:
        fails.append("6. the unchanged cap after mixing is gone")
    if not re.search(r"cands\s*\+=|cands\.extend", body):
        pass    # appending would be the failure; its absence is what we want
    else:
        fails.append("6. candidates are appended somewhere - the offer grows")

    for v, rows, _n in got:
        if not rows:
            fails.append(f"4. `{v}` was offered with no rows")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print(f"PASS  lane offered {names} with counts {counts}")
    print("  own line dropped, own values excluded, off-tape and rowless values excluded,")
    print("  rank is count then nearest line, cap honoured, interleaved at the unchanged cap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
