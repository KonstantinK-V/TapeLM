"""Check 627: peak-chain; stop on tie; not leftover; held not gated."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit627_chain.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "n < 40" not in src:
        fails.append("1. VOID missing")
    if "(p_h2 - p_r2) > 0.05" not in src:
        fails.append("1. GATE missing")
    if "held_ask" in src and "gate" in src.lower():
        if "compose ctx→ask" not in src and "not gated" not in src:
            fails.append("1. held must not be the gate")
    if "sum(1 for t in vot if vot[t] == top) != 1" not in src:
        fails.append("1. refuse-on-tie missing")
    if "leftover" in src and "Not leftover" not in src:
        fails.append("1. leftover rank leaked")
    if "n_fr - 2" not in src:
        fails.append("1. two-row hide missing")
    return fails


MUTANTS = (
    (
        "guess on tie",
        "    if sum(1 for t in vot if vot[t] == top) != 1:\n        return None, tuple(ranked)",
        "    if False:\n        return None, tuple(ranked)",
        "1.",
    ),
    (
        "held gate",
        "    gate = (not void) and (p_h2 - p_r2) > 0.05",
        "    gate = (not void) and (r(n_compose, n) > 0.05)",
        "1.",
    ),
    (
        "n-1",
        "                n_use = max(n_fr - 2, 1)",
        "                n_use = max(n_fr - 1, 1)",
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
        mut = src.replace(old, new, 1)
        got = props(mut)
        if name == "held gate" and "(p_h2 - p_r2) > 0.05" not in mut:
            got.append("1. GATE missing")
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
