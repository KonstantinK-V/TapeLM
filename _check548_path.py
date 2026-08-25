"""Check of 548: hop2 is rec(v)∩rec(hop1), not a new star.

    python _check548_path.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit548_path.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "unique_next" in src or "narrow_next" in src:
        f.append("1. unique exam")
    if "cheap_rec(g, by, a, cache)" in src:
        f.append("1. hop2 walks a new star (547)")
    if "c in rec_set" not in src:
        f.append("1. hop2 must stay inside rec(v)")
    if "residual=bool(hit2 and not hit1)" not in src:
        f.append("1. residual teacher")
    if "ALLGO" not in src:
        f.append("1. always-go diagnostic missing")
    if "lb - la > 0.05" not in src or "lb - lc > 0.05" not in src:
        f.append("2. GATE")
    if "by[v] = list(rest)" not in src:
        f.append("1. held out of rec")
    return f


MUTANTS = (
    ("new star hop2",
     "    cut = [c for c in rec_h1 if c in rec_set and c != v]",
     "    cut = [c for c in rec_h1 if c != v]  # cheap_rec(g, by, a, cache)",
     "1."),
    ("residual dropped",
     "residual=bool(hit2 and not hit1)",
     "residual=bool(hit2)",
     "1."),
    ("gate no hop1",
     "    gate = (not void) and lb - la > 0.05 and lb - ld > 0.05 and lb - lc > 0.05",
     "    gate = (not void) and lb - ld > 0.05 and lb - lc > 0.05",
     "2."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {n}")
            continue
        got = props(src=src.replace(old, new, 1))
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): not caught")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
