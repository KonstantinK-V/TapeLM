"""Check 599: residual split rankmiss vs crowd; fan of unique extras only."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit599_fan.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "crowd += int(not in_u)" not in src:
        fails.append("1. crowd split missing")
    if "void = n_res < 40" not in src:
        fails.append("1. VOID missing")
    if "u_ord[:2]" not in src or "u_ord[:3]" not in src:
        fails.append("1. fan k missing")
    if "held in set(uniq)" not in src:
        fails.append("1. unique-set missing")
    return fails


MUTANTS = (
    (
        "no void",
        "    void = n_res < 40",
        "    void = False",
        "1.",
    ),
    (
        "no crowd split",
        "            crowd += int(not in_u)",
        "            crowd += 0",
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
