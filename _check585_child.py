"""Check 585: on-path child vs PMI-walk vs random; query out of co."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit585_child.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if "n < 40" not in src or "cover < 0.15" not in src:
        fails.append("1. VOID missing")
    if "d_a > 0.05" not in src or "d_c > 0.05" not in src:
        fails.append("1. GATE not B−A and B−C")
    if "onpath" not in src or "g, by, onpath[0]" not in src:
        fails.append("1. missing on-path / one-child walk")
    if "gate = (not void) and d_a > 0.05 and d_c > 0.05" not in src:
        fails.append("1. GATE line not B-A and B-C")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("2. query frame not subtracted from co+df")
    if "len(extra) != 1" not in src:
        fails.append("2. unique-extra filter missing")
    pmi = src.split("def mean_pmi")[1].split("def unique_extras")[0]
    if "held" in pmi:
        fails.append("2. pmi saw held")
    if "randrange" not in src:
        fails.append("1. no random child C")
    return fails


MUTANTS = (
    (
        "no onpath filter",
        "    if onpath:\n        b = walk_from(\n            g, by, onpath[0],",
        "    if onpath:\n        b = walk_from(\n            g, by, pmi_top,",
        "1.",
    ),
    (
        "query in co",
        "        adjust_frame_stats(co, df, qtoks, -1)",
        "        adjust_frame_stats(co, df, qtoks, 0)",
        "2.",
    ),
    (
        "gate A only",
        "    gate = (not void) and d_a > 0.05 and d_c > 0.05",
        "    gate = (not void) and d_a > 0.05",
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
