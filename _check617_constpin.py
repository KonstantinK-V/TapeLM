"""Check 617: CONST agree pin; not 616 first extract; VOID hungry."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit617_constpin.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_const < 40" not in src:
        fails.append("1. VOID missing")
    if "(cd1 - rd1) > 0.05" not in src:
        fails.append("1. GATE missing")
    if "len(set(named)) != 1" not in src:
        fails.append("1. agree missing")
    if "len(uniq) == 1" not in src:
        fails.append("1. unique extra missing")
    if "pg, held_ctx, held_ask" in src:
        fails.append("1. oracle start leaked")
    if "QTab" in src:
        fails.append("1. learner leaked")
    if "hat = first_pin(" in src:
        fails.append("1. 616 pin leaked")
    return fails


MUTANTS = (
    (
        "any extract",
        "                    hat = const_pin(\n                        pg, pin, qi, env_m, mid_set, high_set, {pin, held_ask},\n                    )",
        "                    hat = first_pin(rows_p, mid_set, high_set, {pin, held_ask})",
        "1.",
    ),
    (
        "no agree",
        "    if len(set(named)) != 1:\n        return None",
        "    if False:\n        return None",
        "1.",
    ),
    (
        "oracle",
        "                        pg, hat, held_ask, qi, env_m, mid_set, high_set,",
        "                        pg, held_ctx, held_ask, qi, env_m, mid_set, high_set,",
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
