"""Check 573: held-out-tail online place learner, chosen reward only."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit573_streamlearn.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "train_pool, test_pool = all_lines[:cut], all_lines[cut:]" not in src:
        fails.append("1. corpus not split before windows")
    if "range(0, len(pool) - length + 1, length)" not in src:
        fails.append("1. tape windows overlap")
    if "from _audit572_xorlearn import PlaceScorer, pre_features" not in src:
        fails.append("1. pre-hop shared scorer missing")
    if "keys = sorted(mid)" not in src or "cand = sorted([" not in src:
        fails.append("1. hash-order nondeterminism")
    if 'hit = ha if choose_a else hb' not in src:
        fails.append("1. update does not use chosen reward only")
    if "if held in place:" not in src or "not in {held, v}" in src:
        fails.append("1. direct READ/teacher subtraction broken")
    if "random.Random(seed + 991).shuffle(outcomes)" not in src:
        fails.append("2. reward-null missing")
    if "bootstrap_delta" not in src or "ci_low > 0.0" not in src:
        fails.append("2. paired confidence gate missing")
    if "rival_name = max(" not in src:
        fails.append("2. strongest rival missing")
    if "delta > 0.02" not in src:
        fails.append("2. practical margin missing")
    if "n_test < 40" not in src:
        fails.append("2. held-out XOR VOID missing")
    if "from _place_walk" in src:
        fails.append("3. runtime law touched")
    if "CrossEntropyLoss" in src or "vocab" in src.lower():
        fails.append("3. vocab objective leaked")
    return fails


MUTANTS = (
    (
        "same pool",
        "    train_pool, test_pool = all_lines[:cut], all_lines[cut:]",
        "    train_pool, test_pool = all_lines, all_lines",
        "1.",
    ),
    (
        "oracle both outcomes",
        '                hit = ha if choose_a else hb',
        '                hit = max(ha, hb)',
        "1.",
    ),
    (
        "no confidence",
        "    gate = (not void) and delta > 0.02 and ci_low > 0.0",
        "    gate = (not void) and delta > 0.02",
        "2.",
    ),
    (
        "void soft",
        "    void = n_test < 40",
        "    void = n_test < 0",
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
