"""Check of 570: smoke gate pin+refuse; imports one_query from runner.
    python _check570_news.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit570_news.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _place_walk import one_query" not in src:
        f.append("1. runner import missing")
    if "pin_r > 0 and ref_r > 0" not in src:
        f.append("2. smoke GATE pin+refuse")
    if "n < 40" not in src:
        f.append("2. VOID n")
    if "c1" not in src or "c2" not in src:
        f.append("1. n_cand buckets missing")
    return f


MUTANTS = (
    ("gate always",
     "    gate = (not void) and pin_r > 0 and ref_r > 0",
     "    gate = True",
     "2."),
    ("no import",
     "from _place_walk import one_query, slot_lines",
     "one_query = slot_lines = None",
     "1."),
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
