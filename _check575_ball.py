"""Check 575: weighted vote uses jac, maj ignores env, query out, no torch."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit575_ball.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked into ceiling")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if "n < 40" not in src or "cover < 0.15" not in src:
        fails.append("1. VOID missing")
    if "d_maj > 0.05 and d_rnd > 0.05" not in src:
        fails.append("1. GATE not wvote-maj and wvote-rand")
    if "weighted[tok] += jac" not in src:
        fails.append("2. vote not jac-weighted")
    if "if t == s_q" not in src:
        fails.append("2. query slot not excluded")
    star = src.split("def star_scores")[1].split("def pick_peaked")[0] if "def star_scores" in src else ""
    if "held" in star:
        fails.append("2. star_scores sees held")
    if "Counter(bag)" not in src:
        fails.append("2. unweighted maj missing")
    if "MARGIN" not in src:
        fails.append("2. peaked margin missing")
    return fails


MUTANTS = (
    (
        "unweighted vote",
        "                weighted[tok] += jac",
        "                weighted[tok] += 1",
        "2.",
    ),
    (
        "query slot in",
        "        if t == s_q:\n            continue",
        "        if False and t == s_q:\n            continue",
        "2.",
    ),
    (
        "drop maj from gate",
        "    gate = (not void) and d_maj > 0.05 and d_rnd > 0.05",
        "    gate = (not void) and d_rnd > 0.05",
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
