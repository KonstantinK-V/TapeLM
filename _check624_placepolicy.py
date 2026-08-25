"""Check 624: chosen-reward exact-address policy with honest candidates."""
from __future__ import annotations

from pathlib import Path


SRC = Path("_audit624_placepolicy.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "train_pool, test_pool = all_lines[:cut], all_lines[cut:]" not in src:
        fails.append("1. corpus not split before tapes")
    if "range(0, len(pool) - length + 1, length)" not in src:
        fails.append("1. tape windows overlap")
    if "assert \"addr\" in candidates[li]" not in src:
        fails.append("1. action is not an exact place address")
    if "hit = ys[action] if is_read else 0" not in src:
        fails.append("1. update is not chosen-action tape reward")
    if "null_rng.shuffle(ys)" not in src:
        fails.append("1. shuffled-reward null missing")
    if "refuse_feat" not in src or "if is_read else 0.0" not in src:
        fails.append("1. REFUSE action missing")

    candidate_code = src.split("def legal_unique", 1)[-1].split(
        "def action_hit", 1,
    )[0]
    if "held" in candidate_code:
        fails.append("2. hidden held filters candidates/features")
    if "peak_pin(\n                    pg, pin, qi, env_m, mid_set, high_set, {pin}," not in src:
        fails.append("2. honest peak candidate call missing")
    if "adjust_frame_stats(co, df, ctx_row, -1)" not in src:
        fails.append("2. context record not held out")
    if "adjust_frame_stats(co, df, ask_row, -1)" not in src:
        fails.append("2. ask record not held out")
    if "n_use = max(n_fr - 2, 1)" not in src:
        fails.append("2. two-record denominator missing")
    if "set(query[\"keys\"]) | {held_ctx, held_ask}" in src:
        fails.append("2. invented joint ctx-ask row leaked")

    if "\"majority_same\", \"majority_route\"" not in src:
        fails.append("3. biting majority rivals missing")
    if "delta > 0.05" not in src or "ci_low > 0.0" not in src:
        fails.append("3. rank margin/confidence gate missing")
    if "net_delta > 0.0" not in src:
        fails.append("3. priced policy gate missing")
    if "n_test < 100 or room <= 0.05" not in src:
        fails.append("3. held-out ceiling VOID missing")
    if "CrossEntropyLoss" in src or "token_id" in src.lower():
        fails.append("4. vocabulary objective leaked")
    return fails


MUTANTS = (
    (
        "same corpus",
        "    train_pool, test_pool = all_lines[:cut], all_lines[cut:]",
        "    train_pool, test_pool = all_lines, all_lines",
        "1.",
    ),
    (
        "oracle reward",
        "                hit = ys[action] if is_read else 0",
        "                hit = max(ys) if is_read else 0",
        "1.",
    ),
    (
        "token action",
        "            assert \"addr\" in candidates[li]",
        "            assert \"door\" in candidates[li]",
        "1.",
    ),
    (
        "one fake row",
        "            adjust_frame_stats(co, df, ctx_row, -1)\n"
        "            adjust_frame_stats(co, df, ask_row, -1)",
        "            adjust_frame_stats(co, df, ctx_row | ask_row, -1)",
        "2.",
    ),
    (
        "weak gate",
        "        and ci_low > 0.0\n        and net_delta > 0.0",
        "        and ci_low > 0.0",
        "3.",
    ),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props(src)
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
