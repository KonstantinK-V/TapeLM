"""Check 595: residual numbered; bag is ceiling not law; unique frozen."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit595_gap.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src or "QTab" in src:
        fails.append("1. Q/torch leaked")
    if "held not in set(bag) or u_hit" not in src:
        fails.append("1. residual filter missing")
    if "void = n_res < 40" not in src:
        fails.append("1. VOID missing")
    if "mass > 0.05" not in src or "prize > 0.05" not in src:
        fails.append("1. GATE mass/prize missing")
    if "Do not put bag in the law" not in src and "not law" not in src:
        fails.append("1. bag-not-law missing")
    return fails


MUTANTS = (
    (
        "no residual",
        "            if held not in set(bag) or u_hit:",
        "            if False:",
        "1.",
    ),
    (
        "no void",
        "    void = n_res < 40",
        "    void = False",
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
