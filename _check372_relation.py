"""372b: relation as evidence on the candidate. Static, and it RUNS.

Seven ways this could be silently the wrong 372 again: the offer could change; the walk
could leave cos; an unwitnessed candidate could become an empty world (291's size
marker); the own place could witness itself; retention could be skipped; a threshold
could sneak back in; argparse could not accept `relation` (335's dead knob).
"""
from __future__ import annotations

import ast

SRC = "_stage289_derivation.py"


def static():
    t = ast.parse(open(SRC, encoding="utf-8").read())
    u = ast.unparse(t).replace('"', "'")
    fn = {n.name: n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}
    rr = ast.unparse(fn["reach_relation_rows"])
    rf = ast.unparse(fn["reach_rows_for"])
    rc = ast.unparse(fn["reach_candidates"])
    rp = ast.unparse(fn["reach_places"])
    checks = [
        ("walk still returns on cos before any share scoring",
         "if REACH_COMPASS == 'cos':" in rp and rp.index("if REACH_COMPASS == 'cos':")
         < rp.index("ownc = Counter")),
        ("reach_candidates does not branch on import=relation - the offer is identical",
         "REACH_IMPORT == 'relation'" not in rc and "if REACH_IMPORT" not in rc),
        ("unwitnessed candidate falls back to walk rows, not empty",
         "return rel if rel else rows" in rf),
        ("the question's own place is excluded",
         "if j != i and (kp is None" in rr or "j != i" in rr),
        ("retention honoured, as reach_places honours it",
         "retain_keep(p)" in rr and "bool(kp[j])" in rr),
        ("overlap is a count, no threshold",
         "overlap[j] += 1" in rr and ">= 2" not in rr),
        ("argparse accepts relation",
         "'relation'" in u and "add_argument('--reach-import'" in u),
    ]
    ok = True
    for name, good in checks:
        print(f"  {'OK  ' if good else 'FAIL'}  {name}")
        ok &= bool(good)
    print(f"\n{'372b OK  7/7' if ok else '372b FAILED'}  ({sum(1 for _, g in checks if g)}/{len(checks)})")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if static() else 1)
