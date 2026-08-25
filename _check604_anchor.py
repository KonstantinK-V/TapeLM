"""Check 604: crowd ANCHOR-IMPORT; exact constraints; no PMI in resolver."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit604_anchor.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    body = src.split("def import_set", 1)[-1].split("def collect", 1)[0]
    if "pmi" in body.lower() or "held" in body:
        fails.append("1. PMI/held in import resolver")
    if "if t == s_q:" not in body:
        fails.append("1. query frame leaks into import")
    if "constrained = pin_set & imported" not in src:
        fails.append("1. pin/anchor intersection missing")
    if "held in set(uniq)" not in src:
        fails.append("1. crowd filter missing")
    if "any(singleton_hit(cands, held)" not in src:
        fails.append("1. singleton oracle missing")
    if "combinations(sets1, 2)" not in src:
        fails.append("1. anchor pair missing")
    if "f_o1 - f_mn1 > 0.05" not in src or "f_o2 - f_mn2 > 0.05" not in src:
        fails.append("1. MIN rival missing")
    if "f_o1 - f_maj > 0.05" not in src or "f_o2 - f_maj > 0.05" not in src:
        fails.append("1. MAJ rival missing")
    if "void = n_c < 40" not in src:
        fails.append("1. VOID missing")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("1. query frame not removed from co+df")
    return fails


MUTANTS = (
    (
        "query frame leak",
        "        if t == s_q:\n            continue",
        "        if False:\n            continue",
        "1.",
    ),
    (
        "no pin constraint",
        "                    constrained = pin_set & imported",
        "                    constrained = imported",
        "1.",
    ),
    (
        "no min rival",
        "        and (f_o1 - f_mn1 > 0.05)",
        "        and True",
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
