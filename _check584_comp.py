"""Check 584: until-refuse (not cap 3); static skeleton; query out of co."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit584_comp.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if "n < 40" not in src or "cover < 0.15" not in src:
        fails.append("1. VOID missing")
    if "depth < SAFETY" not in src:
        fails.append("1. still capped at 3")
    if "SAFETY = 16" not in src:
        fails.append("1. no until-refuse safety cap")
    if "static_skeleton" not in src or "giant" not in src:
        fails.append("1. no corpus skeleton")
    if "r_inf" not in src:
        fails.append("1. no component reach")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("2. query frame not subtracted from co+df")
    ue = src.split("def unique_extras")[1].split("def static_skeleton")[0]
    if "if len(extra) != 1" not in ue or "if False and len(extra)" in ue:
        fails.append("2. unique-extra filter missing")
    pmi = src.split("def mean_pmi")[1].split("def unique_extras")[0]
    if "held" in pmi:
        fails.append("2. pmi saw held")
    if "no hole" not in src.split("STATIC", 1)[1][:400]:
        fails.append("1. static still uses the hole")
    return fails


MUTANTS = (
    (
        "cap 3 back",
        "    while frontier and depth < SAFETY:",
        "    while frontier and depth < 3:",
        "1.",
    ),
    (
        "query in co",
        "        adjust_frame_stats(co, df, qtoks, -1)",
        "        adjust_frame_stats(co, df, qtoks, 0)",
        "2.",
    ),
    (
        "no unique",
        "        if len(extra) != 1:\n            continue\n        tok = extra[0]",
        "        if False and len(extra) != 1:\n            continue\n        tok = extra[0]",
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
