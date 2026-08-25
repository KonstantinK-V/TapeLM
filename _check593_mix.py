"""Check 593: mixed remainder ceiling; v1 unique law not closed."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit593_mix.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src or "QTab" in src:
        fails.append("1. Q/torch leaked")
    if "n_m < 40" not in src:
        fails.append("1. VOID n_mixed missing")
    if "held in set(uniq)" not in src:
        fails.append("1. unique/mixed split missing")
    if "pmi_rank" not in src:
        fails.append("1. PMI_bag missing")
    if "len(extra) != 1" not in src:
        fails.append("1. unique filter missing in bag_of")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("1. query frame not removed from co+df")
    if "DEAD" in src.split('"""', 2)[-1]:
        fails.append("1. token-DEAD returned")
    return fails


MUTANTS = (
    (
        "no mixed split",
        "            if held in set(uniq):",
        "            if False:",
        "1.",
    ),
    (
        "no void",
        "    void = n_m < 40",
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
