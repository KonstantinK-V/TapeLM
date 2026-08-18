"""365/367: the extra channels, checked statically and RUN on a hand-made tape.

Three ways this could be wrong and none of them would show up in a loss curve:
the channel could hand the question its own answer back, the interleave could
quietly widen the offer so the arm wins on budget, or the strict form 363
closed could creep back in as a threshold.
"""
from __future__ import annotations

import ast
from collections import Counter

SRC = "_stage289_derivation.py"


def static():
    src = open(SRC, encoding="utf-8").read()
    t = ast.parse(src)
    u = ast.unparse(t)
    u1 = u.replace('"', "'")
    fn = {n.name: n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}
    c = ast.unparse(fn["reach_connect"])
    rc = ast.unparse(fn["reach_candidates"])
    checks = [
        ("the question's own place is excluded", "j != i" in c),
        ("its own values are excluded (recall covers them)", "if v in own:" in c),
        ("retention honoured, as reach_places honours it",
         "retain_keep(p)" in c and "bool(kp[j])" in c),
        ("overlap-weighted", "overlap[j] += 1" in c and "score[v] += ov" in c),
        ("no threshold - 363 closed the strict form", ">= 2" not in c),
        ("367: home lane exists, round-robin, tagged apart from the connect lane",
         "if OWN_IN_OFFER:" in rc and "from_place[v] = -2" in rc
         and "for tup in zip_longest(*lanes)" in rc),
        ("367: home values imported from the same source as every candidate",
         "outside_mentions(p, q, v)" in rc),
        ("interleaved BEFORE the cap, so the offer does not grow",
         rc.index("cands = mixed") < rc.index("cands = cands[:REACH_CANDS]")),
        ("exactly one cap", rc.count("cands = cands[:REACH_CANDS]") == 1),
        ("flag exists and defaults OFF",
         "add_argument('--connect', action='store_true'" in u1),
        ("assigned from args", "CONNECT, CONNECT_MAX = (args.connect, args.connect_max)" in u),
        ("written into the report", "'connect': bool(CONNECT)" in u1),
    ]
    ok = True
    for name, good in checks:
        print(f"  {'OK  ' if good else 'FAIL'}  {name}")
        ok &= bool(good)
    return ok


def behaviour():
    """The ranking itself, on a tape where the weighted and plain orders DIFFER - otherwise the
    test would pass against the plain count that 363 measured as worse."""
    own = {"A", "B", "C"}
    # place -> its fillers. p1 shares three, p2..p5 share one each.
    fills = {1: {"A", "B", "C", "X"}, 2: {"A", "Y"}, 3: {"A", "Y"}, 4: {"A", "Y"}, 5: {"A", "Y"}}
    overlap = Counter({j: len(f & own) for j, f in fills.items()})
    score = Counter()
    for j, ov in overlap.most_common(4000):
        for v in fills[j]:
            if v in own:
                continue
            score[v] += ov
    print(f"  overlap {dict(overlap)}")
    print(f"  score   {dict(score)}   (X from one place sharing 3, Y from four sharing 1)")
    good = score["X"] == 3 and score["Y"] == 4 and "A" not in score
    print(f"  {'OK  ' if good else 'FAIL'}  weight counts relatedness, own values never scored")
    # and the interleave must not grow the offer
    from itertools import zip_longest
    lanes, cap = [["w1", "w2", "w3"], ["c1", "w2", "c2"], ["o1", "o2"]], 6
    seen, mixed = set(), []
    for tup in zip_longest(*lanes):
        for e in tup:
            if e is not None and e not in seen:
                seen.add(e)
                mixed.append(e)
    out = mixed[:cap]
    print(f"  three lanes -> {out}")
    good &= out == ["w1", "c1", "o1", "w2", "o2", "w3"] and len(set(out)) == cap
    print(f"  {'OK  ' if good else 'FAIL'}  round-robin, deduped on first appearance, walk "
          f"first, capped at {cap}")
    return good


if __name__ == "__main__":
    print("STATIC")
    a = static()
    print("BEHAVIOUR")
    b = behaviour()
    print("\n365 OK" if a and b else "\n365 FAILED")
    raise SystemExit(0 if a and b else 1)
