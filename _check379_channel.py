"""Static and behavioural check of 379's channel feature. No torch, no corpus.

Seven properties. Every one of them would be a silent wrong number rather than a crash, and
three of them would be a LEAK - a feature that tells Phi which candidate is the answer for a
bookkeeping reason rather than an evidential one.

  1. OFF IS OFF. With REACH_CHANNEL false the tail is empty, so an arm without the lever has a
     bit-for-bit identical node vector to the arm before it.
  2. The walk is the all-zero baseline; connect, home and copy are one-hot and distinct.
  3. Only the ANSWERED row carries the indicators; every other row carries zeros of the same
     width, or the two builders disagree and the graph crashes mid-run.
  4. Both builders append the tail, in the same place, after `confirm`.
  5. The declared width grows by exactly three when the lever is on, in BOTH places that
     construct a Deriver.
  6. A value the offer never proposed reads as the walk baseline, not as a crash.
  7. reach_candidates exports from_place, or the feature reads nothing at all.

    python _check379_channel.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path("_stage289_derivation.py")


def load(names):
    src = SRC.read_text(encoding="utf-8")
    ns = {}
    for name in names:
        m = re.search(rf"^def {name}\(.*?(?=\n(?:def |@|\w))", src, re.S | re.M)
        if not m:
            print(f"FAIL: {name} not found in {SRC}")
            sys.exit(1)
        exec(compile(m.group(0), name, "exec"), ns)
    return ns, src


def main() -> int:
    ns, src = load(("reach_channel", "channel_feat"))
    rc_, cf = ns["reach_channel"], ns["channel_feat"]
    fails = []

    q = {"_reach_c": {"from_place": {"w": 7, "c": -1, "h": -2, "k": -3}}}

    # 1. off is off
    ns["REACH_CHANNEL"] = False
    if rc_(None, q, "k") != (0.0, 0.0, 0.0):
        fails.append("1. the feature is computed while the lever is off")
    if cf({"channel": (0.0, 0.0, 1.0)}, 0, 0) != []:
        fails.append("1. the node tail is non-empty while the lever is off")

    # 2. one-hot and distinct, walk at zero
    ns["REACH_CHANNEL"] = True
    got = {n: rc_(None, q, n) for n in ("w", "c", "h", "k")}
    if got["w"] != (0.0, 0.0, 0.0):
        fails.append(f"2. the walk is not the zero baseline: {got['w']}")
    for n in ("c", "h", "k"):
        if sum(got[n]) != 1.0:
            fails.append(f"2. `{n}` is not one-hot: {got[n]}")
    if len({got[n] for n in ("c", "h", "k")}) != 3:
        fails.append(f"2. two channels collided onto the same indicator: {got}")

    # 3. the answered row alone carries it, at a constant width
    w = {"channel": got["k"]}
    on_row, off_row = cf(w, 2, 2), cf(w, 1, 2)
    if on_row != [0.0, 0.0, 1.0]:
        fails.append(f"3. the answered row does not carry the channel: {on_row}")
    if any(off_row):
        fails.append(f"3. a non-answered row carries the channel: {off_row}")
    if len(on_row) != len(off_row):
        fails.append(f"3. width differs by row: {len(on_row)} vs {len(off_row)}")

    # 6. a value nobody offered, and a question with no walk at all
    if rc_(None, q, "absent") != (0.0, 0.0, 0.0):
        fails.append("6. an unoffered value does not read as the walk baseline")
    if rc_(None, {}, "k") != (0.0, 0.0, 0.0):
        fails.append("6. a question with no walk raises or answers non-zero")
    if cf({}, 0, 0) != [0.0, 0.0, 0.0]:
        fails.append("6. a world with no channel key does not fall back to zeros")

    # 4. both builders, same position
    tails = re.findall(r"REACH_CONFIRM else \[\]\)\n\s*\+ channel_feat\(q, i, qrow\)", src)
    if len(tails) != 2:
        fails.append(f"4. the tail is appended in {len(tails)} builders after confirm, want 2")

    # 5. the declared width, in both constructors
    if len(re.findall(r"\(3 if REACH_CHANNEL else 0\)", src)) != 2:
        fails.append("5. the node width does not grow by three in both Deriver constructions")

    # 7. the provenance is exported
    body = re.search(r"^def reach_candidates\(.*?(?=\n(?:def |@|\w))", src, re.S | re.M).group(0)
    if '"from_place": {c: from_place[c] for c in cands}' not in body:
        fails.append("7. reach_candidates does not export from_place")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print("PASS  walk (0,0,0)  connect (1,0,0)  home (0,1,0)  copy (0,0,1)")
    print("  off is off, one-hot and distinct, answered row only, constant width,")
    print("  both builders and both widths agree, unoffered values fall back to the walk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
