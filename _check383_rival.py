"""Is the counting rival blind anywhere the mind can see? Torch-free.

THE TRAP, walked into three times now and named in reach_count_rival's own docstring: a rival
that cannot NAME a candidate is not losing, it is mute, and its zero is a definition rather than
a defeat. It was closed for depth (`_dp` places are appended) and left open for the channels.

A value contributed by connect or copy stands at NO WALKED PLACE - if it did, the walk lane
would have offered it first and the interleave would have deduped it. The old rival iterated
PLACES and filtered by membership, so those candidates could never be scored. `--connect` has
been in the standing arm since 365.

Seven properties (5-7 are 383, the tie-break):

  1. A candidate standing at a walked place is scored by its share there.
  2. A CHANNEL candidate - at no walked place - is scored too, at the place 381 resolves for it.
  3. The rival can WIN with a channel candidate. If it cannot, the fix is cosmetic.
  4. A candidate with no resolvable place is skipped, not crashed on and not scored as zero-
     denominator.
  5. The walked pass still runs first, and the walk's own order remains the LAST tie-break.
  6. (383) EQUAL SHARE IS BROKEN BY THE RAW COUNT. `top_share` reads 0.999-1.000 on every seed
     of every arm ever run, so the share rule saturates on values that OWN their place - and
     with --min-fillers 1 that is a single-filler frame, where the share is 1.0 whether the
     value stands there nine times or twice. The walk's order was the tie-break most favourable
     to counting in ORDERING and the least favourable in STRENGTH.
  7. (383) The number of candidates tying at the winning share is COUNTED, so "the rule
     saturates" is a measurement rather than an inference from top_share.

    python _check382_rival.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

SRC = Path("_stage289_derivation.py")


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    m = re.search(r"^def reach_count_rival\(.*?(?=\n(?:def |@|\w))", src, re.S | re.M)
    if not m:
        print(f"FAIL: reach_count_rival not found in {SRC}")
        return 1

    # place 0 is walked and holds `w` 1 of 4; place 1 is NOT walked and holds `k` 3 of 4.
    fills = {0: [("w", [], 1), ("x", [], 3)],
             1: [("k", [], 3), ("y", [], 1)],
             2: [("z", [], 5)]}
    state = {}

    def fake_rc(p, q):
        return state["rc"]

    ns = {"reach_candidates": fake_rc, "Counter": Counter,
          "reach_index": lambda p: {"fills": fills},
          "REACH_DEPTH": 1}
    exec(compile(m.group(0), "reach_count_rival", "exec"), ns)
    rival = ns["reach_count_rival"]
    fails = []

    # 1 + 2 + 3: a walk candidate at .25 against a copy candidate at .75
    state["rc"] = {"cands": ["w", "k"], "places": [(0, None, 1.0)],
                   "real_place": {"w": 0, "k": 1}}
    best, sh = rival(None, {})
    if best != "k":
        fails.append(f"3. the rival cannot win with a channel candidate: got {best!r} {sh:.3f}")
    elif abs(sh - 0.75) > 1e-9:
        fails.append(f"2. the channel candidate scored {sh:.3f}, want 0.750 at its own place")

    # 1 alone: with no channel candidate the answer is unchanged from before the fix
    state["rc"] = {"cands": ["w", "x"], "places": [(0, None, 1.0)],
                   "real_place": {"w": 0, "x": 0}}
    if rival(None, {})[0] != "x":
        fails.append("1. a walked candidate is no longer scored by its share at its place")

    # 4: unresolvable place, and a place that is empty
    state["rc"] = {"cands": ["w", "ghost"], "places": [(0, None, 1.0)],
                   "real_place": {"w": 0, "ghost": None}}
    try:
        b4, _ = rival(None, {})
    except Exception as e:                                   # noqa: BLE001 - that is the test
        fails.append(f"4. a placeless candidate raised {type(e).__name__}: {e}")
        b4 = None
    if b4 not in (None, "w", "x"):
        fails.append(f"4. a placeless candidate was scored anyway: {b4!r}")

    # 5: equal share AND equal count -> the walk's own order still decides.
    # `w` at place 0 is 1 of 4; `t` at place 3 is 1 of 4 as well.
    fills[3] = [("t", [], 1), ("u", [], 3)]
    q5 = {}
    state["rc"] = {"cands": ["w", "t"], "places": [(0, None, 1.0)],
                   "real_place": {"w": 0, "t": 3}}
    b5, s5 = rival(None, q5)
    if abs(s5 - 0.25) > 1e-9 or b5 != "w":
        fails.append(f"5. an exact tie did not go to the walk's own order: {b5!r} {s5:.3f}")

    # 6: equal share, HIGHER COUNT -> the count decides, against the walk's order.
    # place 4 gives `h` 9 of 36, the same .25 as `w`, on nine times the evidence.
    fills[4] = [("h", [], 9), ("g", [], 27)]
    state["rc"] = {"cands": ["w", "h"], "places": [(0, None, 1.0)],
                   "real_place": {"w": 0, "h": 4}}
    b6, s6 = rival(None, {})
    if b6 != "h":
        fails.append(f"6. equal share is still decided by walk order, not by the count: {b6!r}")
    elif abs(s6 - 0.25) > 1e-9:
        fails.append(f"6. the winning share changed: {s6:.3f}")
    # and the saturated case the whole fix is about: two values each OWNING their place
    fills[5], fills[6] = [("p", [], 2)], [("r", [], 9)]
    state["rc"] = {"cands": ["p", "r"], "places": [(5, None, 1.0)],
                   "real_place": {"p": 5, "r": 6}}
    if rival(None, {})[0] != "r":
        fails.append("6. with both candidates at share 1.0 the rule still ignores the count - "
                     "this is exactly the saturation the fix exists for")

    # 7: the ties are counted
    q7 = {}
    state["rc"] = {"cands": ["p", "r"], "places": [(5, None, 1.0)],
                   "real_place": {"p": 5, "r": 6}}
    rival(None, q7)
    if q7.get("_cr_ties") != 2:
        fails.append(f"7. two candidates tied at share 1.0 and ties reads {q7.get('_cr_ties')}")
    state["rc"] = {"cands": ["w", "x"], "places": [(0, None, 1.0)],
                   "real_place": {"w": 0, "x": 0}}
    q7b = {}
    rival(None, q7b)
    if q7b.get("_cr_ties") != 1:
        fails.append(f"7. a determinate winner reads ties {q7b.get('_cr_ties')}, want 1")

    body = m.group(0)
    if "for v in cands:" not in body:
        fails.append("2. the channel pass is gone - candidates at no walked place are mute")
    elif body.index("for j, _it, _sim in places:") > body.index("for v in cands:"):
        fails.append("5. the channel pass runs before the walked pass")
    if 'rc.get("real_place"' not in body:
        fails.append("2. the rival does not use 381's resolved place")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print("PASS  walked .25 vs channel .75 - the rival names the channel one; placeless")
    print("  candidates skipped; equal share now breaks by the raw count and only then by the")
    print("  walk's order; ties at the winning share are counted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
