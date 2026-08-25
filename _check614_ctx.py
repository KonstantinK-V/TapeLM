"""Check 614: oracle pin; residual vs Petya SEARCH; RAND mid; no bag0 gate."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit614_ctx.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_res < 40" not in src:
        fails.append("1. VOID residual missing")
    if "d > 0.05" not in src:
        fails.append("1. GATE CTX-RAND missing")
    if "held_ctx, held_ask" not in src:
        fails.append("1. two extras missing")
    if "hit_extract(rows_p, held_ask)" not in src:
        fails.append("1. residual not dropping Petya hits")
    if "bag-hit" not in src and "hit_bag" not in src:
        fails.append("1. bag diag missing")
    if "QTab" in src or "touch(" in src:
        fails.append("1. learner leaked")
    if "len(extras) < 2" not in src:
        fails.append("1. pair filter missing")
    return fails


MUTANTS = (
    (
        "no residual",
        "                if hit_extract(rows_p, held_ask):\n                    petya += 1\n                    continue",
        "                if False:\n                    petya += 1\n                    continue",
        "1.",
    ),
    (
        "gate bag",
        "    gate = (not void) and d > 0.05",
        "    gate = (not void) and r(bag_c, n_res) - r(bag_r, n_res) > 0.05",
        "1.",
    ),
    (
        "one extra",
        "                if len(extras) < 2:",
        "                if len(extras) < 1:",
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
