"""Check 579: pmi on unique-extra, query frame out of co, gate vs jac not only maj."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit579_lift.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "shuffle(blocks)" in src:
        fails.append("1. windows shuffled")
    if "n < 40" not in src or "cover < 0.15" not in src:
        fails.append("1. VOID missing")
    if "d_jac > 0.05 and d_maj > 0.05" not in src:
        fails.append("1. GATE not lift-jac and lift-maj")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("2. query frame not subtracted from co+df")
    if "if len(extra) == 1:" not in src:
        fails.append("2. unique-extra filter missing")
    if "if t == s_q" not in src:
        fails.append("2. query slot not excluded")
    pmi = src.split("def mean_pmi")[1].split("def walk")[0]
    if "held" in pmi:
        fails.append("2. pmi saw held")
    if "math.log" not in src:
        fails.append("2. pmi not log-lift")
    return fails


MUTANTS = (
    (
        "query frame in stats",
        "        adjust_frame_stats(co, df, qtoks, -1)",
        "        adjust_frame_stats(co, df, qtoks, 0)",
        "2.",
    ),
    (
        "all extras",
        "            if len(extra) == 1:",
        "            if False and len(extra) == 1:",
        "2.",
    ),
    (
        "gate vs maj only",
        "    gate = (not void) and d_jac > 0.05 and d_maj > 0.05",
        "    gate = (not void) and d_maj > 0.05",
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
