"""Check of 569: retry-stand after unique DEAD; no invent; 567 not mixed.
    python _check569_restnd.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit569_restnd.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Phi")
    if "len(cand) != 1" not in src:
        f.append("1. not 557 unique")
    if "for s2 in sl:" not in src:
        f.append("1. retry stand missing")
    if "rec[\"saved\"] = 1" not in src:
        f.append("1. saved missing")
    if 'sk["noretry"]' not in src:
        f.append("1. noretry tracking missing")
    if "if held in place:" not in src:
        f.append("1. direct READ not removed before retry")
    if "c != held" in src:
        f.append("1. teacher subtracted from candidate list")
    if "nd < 20" not in src:
        f.append("2. VOID on n_dead1")
    if "rt >= 0.99" not in src or "sv > 0.05" not in src:
        f.append("2. GATE retry/saved")
    for line in src.splitlines():
        if "gate =" in line and "h1" in line:
            f.append("2. hit1 in GATE")
            break
    return f


MUTANTS = (
    ("void soft",
     "    void = nd < 20",
     "    void = nd < 0",
     "2."),
    ("gate on hit1",
     "    gate = (not void) and rt >= 0.99 and sv > 0.05",
     "    gate = (not void) and h1 > 0.05",
     "2."),
    ("not unique",
     "    if len(cand) != 1:",
     "    if len(cand) >= 2:",
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
