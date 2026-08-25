"""Check 609: k-curve; held does not order; three outcomes."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit609_kcurve.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "KS = (1, 2, 3, 4, 6, 8)" not in src:
        fails.append("1. KS missing")
    if "hit_prefix(order, held, k)" not in src:
        fails.append("1. sequential k missing")
    if "rnd.shuffle" not in src:
        fails.append("1. shuffled order missing")
    if "int(pl[\"extract\"] == held)" in src or "extract\"] == held), reverse" in src:
        fails.append("1. held orders the list")
    if "d_still > 0.05" not in src or "gap8 <= 0.05" not in src:
        fails.append("1. CLOSE/STILL missing")
    if "n < 40" not in src:
        fails.append("1. VOID missing")
    if "QTab" in src or "touch(" in src:
        fails.append("1. learner leaked")
    return fails


MUTANTS = (
    (
        "k=3 only",
        "KS = (1, 2, 3, 4, 6, 8)",
        "KS = (1, 2, 3)",
        "1.",
    ),
    (
        "held sorts",
        "            mention = list(places)",
        "            mention = sorted(places, key=lambda pl: int(pl[\"extract\"] == held), reverse=True)",
        "1.",
    ),
    (
        "no still",
        "    still = (not void) and d_still > 0.05",
        "    still = False",
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
