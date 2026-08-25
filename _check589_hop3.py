"""Check 589: hop3 after hop2 hit; no Q; PMI vs random."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit589_hop3.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src or "QTab" in src:
        fails.append("1. Q/torch leaked")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if 'ev["n3_pmi"] < 40' not in src:
        fails.append("1. VOID n3_pmi missing")
    if "d_rnd > 0.05" not in src:
        fails.append("1. GATE not PMI-rnd")
    if "if a1 and a2:" not in src:
        fails.append("1. hop3 not gated on hop1 and hop2 hits")
    if "len(extra) != 1" not in src:
        fails.append("2. unique-extra filter missing")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("2. query frame not subtracted from co+df")
    pmi = src.split("def mean_pmi")[1].split("def unique_extras")[0]
    if "held" in pmi:
        fails.append("2. pmi saw held")
    return fails


MUTANTS = (
    (
        "hop3 without hits",
        "        if a1 and a2:",
        "        if True:",
        "1.",
    ),
    (
        "query in stats",
        "        adjust_frame_stats(co, df, qtoks, -1)",
        "        adjust_frame_stats(co, df, qtoks, 0)",
        "2.",
    ),
    (
        "not unique",
        "        if len(extra) != 1:",
        "        if False:",
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
