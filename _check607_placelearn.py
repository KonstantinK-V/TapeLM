"""Check 607: held-out actual-place policy, chosen reward, strongest rival."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit607_placelearn.py")
BRIDGE = Path("_audit606_bridge.py")


def props(src=None, bridge=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    bridge = BRIDGE.read_text(encoding="utf-8") if bridge is None else bridge
    fails = []
    if "train_pool, test_pool = all_lines[:cut], all_lines[cut:]" not in src:
        fails.append("1. corpus not split before tapes")
    if "range(0, len(pool) - length + 1, length)" not in src:
        fails.append("1. tape windows overlap")
    if 'return [place["feat"] for place in row["places"]]' not in src:
        fails.append("1. chooser is not count-feature to place-index")
    if 'assert "addr" in places[pi]' not in src:
        fails.append("1. action is not an actual address")
    if 'hit = ys[action]' not in src:
        fails.append("1. update does not use chosen action reward")
    if "null_rng.shuffle(ys)" not in src:
        fails.append("1. shuffled-label null missing")
    if "fo - strongest" in src or "uniq[0]" in src:
        fails.append("1. oracle or extra leaked into chooser")
    feat = bridge.split("def chooser_features", 1)[-1].split("def collect", 1)[0]
    if "pmi" in feat.lower() or "held" in feat.lower() or "extra" in feat.lower():
        fails.append("1. PMI/held/extra in chooser features")
    if '"majority_same", "majority_route"' not in src:
        fails.append("2. strongest rival set missing")
    if '"bag_majority"' not in src or '"bag_majority"' in src.split("rivals =", 1)[-1].split(")", 1)[0]:
        fails.append("2. pooled bag majority must be report-only")
    if "delta > 0.05" not in src or "ci_low > 0.0" not in src:
        fails.append("2. margin/confidence gate missing")
    if "n_test < 100 or room <= 0.05 or collision > 0.02" not in src:
        fails.append("2. ceiling/test VOID missing")
    if "CrossEntropyLoss" in src or "vocab" in src.lower():
        fails.append("3. vocab objective leaked")
    if "_audit542_curric" in src:
        fails.append("3. 542 retrained/touched")
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
        "                hit = ys[action]",
        "                hit = max(ys)",
        "1.",
    ),
    (
        "weak gate",
        "    gate = (not void) and delta > 0.05 and ci_low > 0.0",
        "    gate = (not void) and delta > 0.05",
        "2.",
    ),
    (
        "token action",
        '            assert "addr" in places[pi]',
        '            assert "extract" in places[pi]',
        "1.",
    ),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    bridge = BRIDGE.read_text(encoding="utf-8")
    fails = props(src, bridge)
    for name, old, new, tag in MUTANTS:
        count = src.count(old)
        if count != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {count}")
            continue
        got = props(src.replace(old, new, 1), bridge)
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
