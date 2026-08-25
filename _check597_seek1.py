"""Check 597: SEEK-1 ceiling; action is a frame; bag is ceiling only."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit597_seek1.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "any(unique_hit(ex, held) for ex in fr)" not in src:
        fails.append("1. oracle not unique-on-frame")
    if "void = n_res < 40" not in src:
        fails.append("1. VOID missing")
    if "fo - fa > 0.05" not in src:
        fails.append("1. GATE missing")
    if "fill_bag" not in src:
        fails.append("1. bag ceiling missing")
    if "adjust_frame_stats(co, df, qtoks, -1)" not in src:
        fails.append("1. query frame not removed from co+df")
    body = src.split("def unique_hit", 1)[-1].split("def main", 1)[0]
    if "pmi" in body.lower():
        fails.append("1. PMI in unique_hit")
    return fails


MUTANTS = (
    (
        "oracle uses bag",
        "            o += int(any(unique_hit(ex, held) for ex in fr))",
        "            o += int(bool(ranked) and ranked[0] == held)",
        "1.",
    ),
    (
        "no void",
        "    void = n_res < 40",
        "    void = False",
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
