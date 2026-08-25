"""Check 594: bag PMI vs unique PMI on the same holes; no Phi."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit594_bag.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src or "QTab" in src:
        fails.append("1. Q/torch leaked")
    if "from _audit593_mix import collect_mix" not in src:
        fails.append("1. not 593 holes")
    if "void = n < 40" not in src:
        fails.append("1. VOID missing")
    if "d_u > 0.05" not in src or "d_r > 0.05" not in src:
        fails.append("1. GATE not bag-unique and bag-rnd")
    if "tok in set(uniq)" not in src:
        fails.append("1. unique hand missing")
    if "if held not in set(bag):" not in src:
        fails.append("1. uncovered holes in denominator")
    if "DEAD" in src.split('"""', 2)[-1]:
        fails.append("1. token-DEAD returned")
    return fails


MUTANTS = (
    (
        "no unique hand",
        "            u_rank = [tok for tok in ranked if tok in set(uniq)]",
        "            u_rank = ranked",
        "1.",
    ),
    (
        "no void",
        "    void = n < 40",
        "    void = False",
        "1.",
    ),
    (
        "uncovered holes",
        "            if held not in set(bag):\n                continue",
        "            if False:\n                continue",
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
