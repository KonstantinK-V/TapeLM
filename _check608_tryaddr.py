"""Check 608: sequential address READ; held does not order; no Q."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit608_tryaddr.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "acc[name][k - 1] += hit_prefix(order, held, k)" not in src:
        fails.append("1. sequential k missing")
    if "rnd.shuffle" not in src:
        fails.append("1. shuffled order missing")
    if "extract\"] == held), reverse" in src or "int(pl[\"extract\"] == held)" in src:
        fails.append("1. held orders the list")
    if "d_search > 0.05" not in src:
        fails.append("1. GATE missing")
    if "n < 40" not in src:
        fails.append("1. VOID missing")
    if "QTab" in src or "touch(" in src:
        fails.append("1. learner leaked")
    return fails


MUTANTS = (
    (
        "k=1 only",
        "                    acc[name][k - 1] += hit_prefix(order, held, k)",
        "                    acc[name][k - 1] += hit_prefix(order, held, 1)",
        "1.",
    ),
    (
        "held sorts",
        "            mention = list(places)",
        "            mention = sorted(places, key=lambda pl: int(pl[\"extract\"] == held), reverse=True)",
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
