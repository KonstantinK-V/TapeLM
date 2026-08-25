"""Check 600: unique fan2 vs U1 vs bag; 511 crowd chooser; copy abort."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit600_ufan.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "u_ord[:2]" not in src:
        fails.append("1. fan2 missing")
    if "n < 200" not in src:
        fails.append("1. VOID missing")
    if "abs(u2 - b) <= 0.02" not in src:
        fails.append("1. COPY abort missing")
    if "if held in set(bag) and held not in set(uniq):" not in src:
        fails.append("1. crowd split missing")
    if "next((c for c in rec if c in set(bag))" not in src:
        fails.append("1. 511 chooser missing")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("1. query frame not removed from co+df")
    return fails


MUTANTS = (
    (
        "no copy",
        "    copy = (not void) and (abs(u2 - b) <= 0.02)",
        "    copy = False",
        "1.",
    ),
    (
        "no crowd",
        "            if held in set(bag) and held not in set(uniq):",
        "            if False:",
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
