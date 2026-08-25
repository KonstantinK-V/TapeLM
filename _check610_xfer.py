"""Check 610: 609 curve on foreign tape; held does not order."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit610_xfer.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "from _audit609_kcurve import KS, hit_prefix" not in src:
        fails.append("1. 609 curve missing")
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
    if 'default="data/_stage254_news.txt"' not in src:
        fails.append("1. news default missing")
    return fails


MUTANTS = (
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
    (
        "stories default",
        '    ap.add_argument("--corpus", default="data/_stage254_news.txt")',
        '    ap.add_argument("--corpus", default="data/_tinystories_train.txt")',
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
