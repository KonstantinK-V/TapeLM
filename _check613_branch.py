"""Check 613: branch bags; SEARCH miss extra; random mid; pin out of frame."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit613_branch.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n < 40 or n_miss < 40" not in src:
        fails.append("1. VOID miss missing")
    if "p_e > 0.05" not in src or "d_rand > 0.05" not in src:
        fails.append("1. GATE extra vs rand missing")
    if "pin in fr" not in src:
        fails.append("1. pin not forbidden in door frame")
    if "held not in set(bag0)" in src:
        fails.append("1. 606 bag0 filter leaked — SEARCH miss impossible")
    if "QTab" in src or "touch(" in src:
        fails.append("1. learner leaked")
    if "random mid" not in src and "rnd_doors" not in src:
        fails.append("1. RAND doors missing")
    return fails


MUTANTS = (
    (
        "no pin forbid",
        "        if pin in fr or pin in place[\"vals\"]:",
        "        if False:",
        "1.",
    ),
    (
        "gate search",
        "    gate = (not void) and p_e > 0.05 and d_rand > 0.05",
        "    gate = (not void) and p_s > 0.05",
        "1.",
    ),
    (
        "no miss void",
        "    void = n < 40 or n_miss < 40",
        "    void = n < 40",
        "1.",
    ),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        count = src.count(old)
        if count != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {count}")
            continue
        got = props(src.replace(old, new, 1))
        if not any(item.startswith(tag) for item in got):
            fails.append(f"MUTATION {tag} ({name}): not caught")
    for item in fails:
        print("FAIL " + item)
    print(
        f"{len(fails)} failures" if fails else
        f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
