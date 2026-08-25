"""Check 573: prefix windows, fair coin, held banned from tape features."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit573_real.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled (not prefix/suffix)")
    if "prefix_windows" not in src:
        fails.append("1. file-order windows missing")
    if "coin_fair = 0.5" not in src:
        fails.append("1. analytic coin missing")
    if "fair=learned - coin_fair" not in src:
        fails.append("1. GATE not on fair coin")
    if "n_test < 40" not in src:
        fails.append("1. VOID 40 missing")
    if "all(m > 0.05 for m in margins.values())" not in src:
        fails.append("1. GATE bar missing")
    tape_fn = src.split("def tape_features")[1].split("def leaky_features")[0] if "def tape_features" in src else ""
    if "mid_set" in tape_fn or "held" in tape_fn:
        fails.append("2. tape_features sees held or mid-scan")
    if "fr & env_m" not in src:
        fails.append("2. tape evidence not dest∩env")
    if "leaky_features" not in src:
        fails.append("2. leaky LOOK not isolated")
    if "row[\"ha\"] != row[\"hb\"]" not in src:
        fails.append("2. XOR filter missing")
    if "if held in place:" not in src or "not in {held, v}" in src:
        fails.append("2. direct READ/teacher subtraction broken")
    return fails


MUTANTS = (
    (
        "shuffle windows",
        "    return blocks[: min(n_win, len(blocks))]",
        "    random.Random(0).shuffle(blocks); return blocks[: min(n_win, len(blocks))]",
        "1.",
    ),
    (
        "gate on rng coin",
        "        fair=learned - coin_fair,",
        "        fair=learned - coin_rng,",
        "1.",
    ),
    (
        "held into tape",
        "        inter = fr & env_m",
        "        inter = fr",
        "2.",
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
