"""Check 602: joint keys on crowd; fill is one word; no PMI in oracle."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit602_joint.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    body = src.split("def joint_E", 1)[-1].split("def main", 1)[0]
    if "pmi" in body.lower():
        fails.append("1. PMI in joint oracle")
    if "len(set(raw) & qkeys) >= 2" not in src:
        fails.append("1. mini-composition missing")
    if "tok in mid_set" not in src:
        fails.append("1. extras not restricted to bag alphabet")
    if "o += int(E == {held})" not in src:
        fails.append("1. unique extra among joint missing")
    if "held in set(uniq)" not in src:
        fails.append("1. crowd filter missing")
    if "void = n_c < 40" not in src:
        fails.append("1. VOID missing")
    if "fo - fm > 0.05" not in src:
        fails.append("1. MAJ rival missing")
    return fails


MUTANTS = (
    (
        "no joint",
        "        if len(set(raw) & qkeys) >= 2:",
        "        if True:",
        "1.",
    ),
    (
        "pmi in oracle",
        "            o += int(E == {held})",
        "            o += int(bool(row[\"ranked\"]) and row[\"ranked\"][0] == held)",
        "1.",
    ),
    (
        "non-mid extras",
        "                        if tok not in qkeys and tok != v and tok in mid_set",
        "                        if tok not in qkeys and tok != v",
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
