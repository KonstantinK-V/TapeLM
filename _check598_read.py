"""Check 598: READ learner; length rival; copy abort; PMI not in key."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit598_read.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    body = src.split("def len_bin", 1)[-1].split("def pick_q", 1)[0]
    if "pmi" in body.lower() or "held" in body:
        fails.append("1. PMI/held in key")
    if "pick_short" not in src:
        fails.append("1. length rival missing")
    if "agree >= 0.8" not in src:
        fails.append("1. COPY abort missing")
    if "n < 40 or len(keys) < 2" not in src:
        fails.append("1. VOID missing")
    if "fl - fs > 0.05" not in src:
        fails.append("1. GATE vs short missing")
    if "torch" in src:
        fails.append("1. torch leaked")
    return fails


MUTANTS = (
    (
        "held in key",
        "    return 0 if n <= 1 else (1 if n == 2 else 2)",
        "    return 0 if n <= 1 else (1 if n == 2 else 2)  # held",
        "1.",
    ),
    (
        "no copy",
        "    copy = agree >= 0.8",
        "    copy = False",
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
