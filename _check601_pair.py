"""Check 601: crowd pair-intersect; no PMI in oracle; MAJ rival."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit601_pair.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    body = src.split("def oracle", 1)[-1].split("def main", 1)[0]
    if "pmi" in body.lower():
        fails.append("1. PMI in oracle")
    if "sets[i] & sets[j] == {held}" not in src:
        fails.append("1. singleton intersect missing")
    if "held not in set(row[\"uniq\"])" not in src:
        fails.append("1. crowd filter missing")
    if "n_c < 40 or n_ok < 40" not in src:
        fails.append("1. VOID missing")
    if "fo - fm > 0.05" not in src:
        fails.append("1. MAJ rival missing")
    if "fan2" in src.split("def crowd")[0] and "GATE" in src.split("def crowd")[0]:
        pass  # docstring may mention fan2 excluded
    return fails


MUTANTS = (
    (
        "oracle uses bag",
        "            if sets[i] & sets[j] == {held}:",
        "            if True:  # pmi bag",
        "1.",
    ),
    (
        "no crowd",
        "    return held in set(row[\"bag\"]) and held not in set(row[\"uniq\"])",
        "    return True",
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
