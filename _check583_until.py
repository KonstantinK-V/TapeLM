"""Check 583: oracle unique-path until held; not pmi-walk; query out of co."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit583_until.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if "n < 40" not in src or "cover < 0.15" not in src:
        fails.append("1. VOID missing")
    if "d2 > 0.05" not in src:
        fails.append("1. GATE not hop2-adds-fills")
    if "gate = (not void) and d2 > 0.05" not in src:
        fails.append("1. GATE line not on d2")
    if "held in n1" not in src:
        fails.append("2. r1 not shortest unique extra of v")
    if "range(2, cap + 1)" not in src:
        fails.append("2. no hop2+")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("2. query frame not subtracted from co+df")
    if "len(extra) != 1" not in src:
        fails.append("2. unique-extra filter missing")
    pmi = src.split("def mean_pmi")[1].split("def unique_extras")[0]
    if "held" in pmi:
        fails.append("2. pmi saw held")
    return fails


MUTANTS = (
    (
        "no hop2",
        "    for depth in range(2, cap + 1):",
        "    for depth in range(2, 2):",
        "2.",
    ),
    (
        "query in co",
        "        adjust_frame_stats(co, df, qtoks, -1)",
        "        adjust_frame_stats(co, df, qtoks, 0)",
        "2.",
    ),
    (
        "gate on r1",
        "    gate = (not void) and d2 > 0.05",
        "    gate = (not void) and r1 > 0.05",
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
