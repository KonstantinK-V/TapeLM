"""Check 603: one-frame select; extra = raw-qkeys-pin; max-overlap rival."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit603_jsel.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "(set(raw) & mid_set) - qkeys - {pin}" not in src:
        fails.append("1. extra definition missing")
    if "any(ex == {held} for" not in src:
        fails.append("1. exists-one-frame oracle missing")
    if "held in set(uniq)" not in src:
        fails.append("1. crowd filter missing")
    if "fo - fmx > 0.05" not in src:
        fails.append("1. max-overlap rival missing")
    if "void = n_c < 40" not in src:
        fails.append("1. VOID missing")
    if "qkeys = set(env)" not in src:
        fails.append("1. qkeys must exclude held")
    body = src.split("def extra_of", 1)[-1].split("def main", 1)[0]
    if "pmi" in body.lower():
        fails.append("1. PMI in oracle")
    return fails


MUTANTS = (
    (
        "union not select",
        "            o += int(any(ex == {held} for _, _, ex in el))",
        "            o += int({held} <= set().union(*(ex for _, _, ex in el)))",
        "1.",
    ),
    (
        "no max rival",
        "        and (fo - fmx > 0.05)",
        "        and True",
        "1.",
    ),
    (
        "non-mid extras",
        "    return (set(raw) & mid_set) - qkeys - {pin}",
        "    return set(raw) - qkeys - {pin}",
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
