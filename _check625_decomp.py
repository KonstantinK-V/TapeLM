"""Check 625: DIRECT vs HOPONLY; two-row hide; held not in forbid."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit625_decomp.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_live < 40" not in src:
        fails.append("1. VOID missing")
    if "(Ho - h_strong) > 0.05" not in src:
        fails.append("1. HOPONLY live gate missing")
    if "n_fr - 2" not in src:
        fails.append("1. two-row n_use missing")
    if "hide_two" not in src:
        fails.append("1. two-row hide missing")
    if "if tok == held_ask:\n        return 1, 0" not in src:
        fails.append("1. DIRECT missing")
    if "return 0, int(cc[1])" not in src:
        fails.append("1. HOPONLY missing")
    if "QTab" in src or "qsum" in src:
        fails.append("1. learner leaked")
    return fails


MUTANTS = (
    (
        "no hop",
        "    return 0, int(cc[1])",
        "    return 0, 0",
        "1.",
    ),
    (
        "n-1",
        "                n_use = max(n_fr - 2, 1)",
        "                n_use = max(n_fr - 1, 1)",
        "1.",
    ),
    (
        "no direct",
        "    if tok == held_ask:\n        return 1, 0",
        "    if False:\n        return 1, 0",
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
