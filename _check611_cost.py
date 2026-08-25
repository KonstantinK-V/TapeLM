"""Check 611: priced SEARCH; pay actual READs; held does not order."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit611_cost.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "first_hit(order, held)" not in src and "first_hit(places, held)" not in src:
        fails.append("1. first-hit missing")
    if "reads[k] += t" not in src:
        fails.append("1. actual READ cost missing")
    if "reads[k] += k" in src and "reads[k] += t" not in src:
        fails.append("1. charged full k")
    if "k_star in INTERIOR" not in src:
        fails.append("1. GATE missing")
    if "n < 40" not in src:
        fails.append("1. VOID missing")
    if "QTab" in src or "touch(" in src:
        fails.append("1. learner leaked")
    if "extract\"] == held), reverse" in src:
        fails.append("1. held orders the list")
    return fails


MUTANTS = (
    (
        "charge full k",
        "                    reads[k] += t",
        "                    reads[k] += k",
        "1.",
    ),
    (
        "no interior",
        "    gate = (not void) and k_star in INTERIOR",
        "    gate = (not void) and k_star == 8",
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
