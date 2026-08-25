"""Check 572: natural one-shot XOR place learner, frozen runtime law."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit572_xorlearn.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "if len(cand) != 2:" not in src:
        fails.append("1. exam not exactly two-place")
    if "if held in place:" not in src or "not in {held, v}" in src:
        fails.append("1. direct READ/teacher subtraction broken")
    signature = "def pre_features(g, by, addr, env_m):"
    if signature not in src:
        fails.append("1. shared pre-hop place features missing")
    pre_block = src.split(signature, 1)[-1].split(
        "\ndef lookahead_features", 1
    )[0]
    if "held" in pre_block or "comps(" in pre_block:
        fails.append("1. post-hop/held leaked into gated scorer features")
    if "lookahead_diagnostic=lookahead" not in src:
        fails.append("1. lookahead diagnostic missing")
    if "disjoint_windows" not in src or "range(0, len(pool) - length + 1, length)" not in src:
        fails.append("1. train/test windows may overlap")
    if "xor_rows(train_rows)" not in src or "xor_rows(test_rows)" not in src:
        fails.append("1. XOR teacher/exam missing")
    if "random.Random(seed + 991).shuffle(labels)" not in src:
        fails.append("2. shuffled-label null missing")
    gate_line = next(
        (line for line in src.splitlines() if line.strip().startswith("gate =")),
        "",
    )
    if "all(margin > 0.05 for margin in margins.values())" not in gate_line:
        fails.append("2. GATE not against all rivals")
    if "n_test < 40" not in src:
        fails.append("2. VOID test XOR")
    if "from _place_walk" in src:
        fails.append("3. runtime law imported/mutated")
    if "CrossEntropyLoss" in src or "vocab" in src.lower():
        fails.append("3. vocab objective leaked")
    return fails


MUTANTS = (
    (
        "held feature",
        "def pre_features(g, by, addr, env_m):",
        "def pre_features(g, by, addr, env_m, held=None):",
        "1.",
    ),
    (
        "overlapping windows",
        "range(0, len(pool) - length + 1, length)",
        "range(0, len(pool) - length + 1, 1)",
        "1.",
    ),
    (
        "gate only coin",
        "    gate = (not void) and all(margin > 0.05 for margin in margins.values())",
        '    gate = (not void) and margins["coin"] > 0.05',
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
