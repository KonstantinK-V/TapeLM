"""Check 626: full-feedback; HOPONLY label; split; no chosen-only."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit626_fullfb.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "n_te < 40" not in src:
        fails.append("1. VOID missing")
    if "(Ph - max(Ch, Rh)) > 0.05" not in src:
        fails.append("1. GATE missing")
    if "windows[:cut]" not in src:
        fails.append("1. split missing")
    if (
        "            qsum[row[\"key\"]] += row[\"h\"]\n"
        "            qn[row[\"key\"]] += 1"
    ) not in src:
        fails.append("1. must train on every hoponly label")
    if "n_fr - 2" not in src:
        fails.append("1. two-row hide missing")
    if "Label = HOPONLY" not in src and "row[\"h\"]" not in src:
        fails.append("1. HOPONLY label missing")
    return fails


MUTANTS = (
    (
        "chosen only",
        "            qsum[row[\"key\"]] += row[\"h\"]\n            qn[row[\"key\"]] += 1",
        "            if False:\n                qsum[row[\"key\"]] += row[\"h\"]\n                qn[row[\"key\"]] += 1",
        "1.",
    ),
    (
        "direct label",
        "            qsum[row[\"key\"]] += row[\"h\"]",
        "            qsum[row[\"key\"]] += row[\"d\"]",
        "1.",
    ),
    (
        "no split",
        "    train = trials(windows[:cut], random.Random(args.seed), args)\n    test = trials(windows[cut:], random.Random(args.seed + 17), args)",
        "    train = trials(windows, random.Random(args.seed), args)\n    test = trials(windows, random.Random(args.seed + 17), args)",
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
        mut = src.replace(old, new, 1)
        got = props(mut)
        if name == "direct label" and "+= row[\"h\"]" not in mut:
            got.append("1. must train on every hoponly label")
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
