"""Check 618: peaked vote pin; WIDE=616; not CONST-all-agree."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit618_peakpin.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_peak < 40" not in src:
        fails.append("1. VOID missing")
    if "(cd1 - rd1) > 0.05" not in src:
        fails.append("1. GATE missing")
    if "pin_rate > 0.80" not in src:
        fails.append("1. WIDE 616 guard missing")
    if "n1 <= n2" not in src:
        fails.append("1. peak vs tie missing")
    if "len(uniq) == 1" not in src:
        fails.append("1. bags must not vote")
    if "hat = first_pin(" in src:
        fails.append("1. 616 pin leaked")
    if "pg, held_ctx, held_ask" in src:
        fails.append("1. oracle leaked")
    if "QTab" in src:
        fails.append("1. learner leaked")
    return fails


MUTANTS = (
    (
        "no wide",
        "    wide = pin_rate > 0.80",
        "    wide = False",
        "1.",
    ),
    (
        "tie ok",
        "    if n1 < 2 or n1 <= n2:",
        "    if n1 < 2 or False:",
        "1.",
    ),
    (
        "oracle",
        "                        pg, hat, held_ask, qi, env_m, mid_set, high_set,",
        "                        pg, held_ctx, held_ask, qi, env_m, mid_set, high_set,",
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
