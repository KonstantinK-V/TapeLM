"""Check 571: off-policy hop2/hop3 XOR census; no learner or law change."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit571_xorcensus.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src or "def train(" in src or "Q[" in src:
        fails.append("1. learner leaked into census")
    if "if len(cand) != 2:" not in src:
        fails.append("1. census not restricted to exactly two")
    if "a=branch(" not in src or "b=branch(" not in src:
        fails.append("1. both branches not executed")
    if 'return None, "read_hit"' not in src or "not in {held, v}" in src:
        fails.append("1. direct READ/teacher subtraction broken")
    if "if h2:" not in src:
        fails.append("1. route continues after hop2 answered")
    if "if len(cand3) != 1:" not in src:
        fails.append("1. hop3 not restricted to unique continuation")
    if 'if c2 != "11":' not in src:
        fails.append("2. delayed hop3 arena not conditioned on both hop2 hits")
    if "h3_xor" not in src or "h2_xor" not in src:
        fails.append("2. XOR quadrants missing")
    if "gate =" in src.lower():
        fails.append("2. learner GATE leaked into census")
    return fails


MUTANTS = (
    (
        "allow non-two",
        "    if len(cand) != 2:",
        "    if len(cand) < 1:",
        "1.",
    ),
    (
        "drop branch b",
        "        b=branch(g, by, b, v, held, env_m, mid_set),",
        "        b=dict(h2=0, reach2=0, ov2=0, reach3=0, h3=-1, n3=-1),",
        "1.",
    ),
    (
        "mix hop3 before both hit",
        '        if c2 != "11":',
        '        if c2 == "xx":',
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
