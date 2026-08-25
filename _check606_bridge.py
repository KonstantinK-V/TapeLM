"""Check 606: exact-address chooser; extras hidden; REACH vs strongest route."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit606_bridge.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "any(pl[\"extract\"] == held" not in src:
        fails.append("1. REACH exists-place missing")
    if "tframes.frame_keep" not in src or "addr=addr" not in src:
        fails.append("1. exact place-id missing")
    if "cand = [c for c in fr" in src or "from _place_walk" in src:
        fails.append("1. value shortlist mislabeled as places")
    if "max(places, key=lambda pl: pl[\"count_key\"])" not in src:
        fails.append("1. count-only place rival missing")
    if 'm += int(pc["majority"] == held)' not in src:
        fails.append("1. same-place majority rival missing")
    if 'tok for tok in dict.fromkeys(query["keys"])' not in src:
        fails.append("1. held is not a row companion")
    if "qval not in env" not in src:
        fails.append("1. row-specific visible cue missing")
    if "adjust_frame_stats(co, df, row_tokens, -1)" not in src:
        fails.append("1. query row not removed from co+df")
    if "pins = sorted(mid_set)" not in src:
        fails.append("1. hash-order nondeterminism")
    if "fo - strongest > 0.05" not in src:
        fails.append("1. strongest-route gate missing")
    if "n < 40 or collision > 0.02" not in src:
        fails.append("1. VOID missing")
    chooser = src.split("def main", 1)[-1]
    if "uniq[0]" in chooser.split("GATE", 1)[0] and "extract" not in chooser:
        fails.append("1. extra leaked into chooser")
    return fails


MUTANTS = (
    (
        "no exists",
        "            o += int(any(pl[\"extract\"] == held for pl in places))",
        "            o += int(places[0][\"extract\"] == held)",
        "1.",
    ),
    (
        "no strongest",
        "    gate = (not void) and (fo - strongest > 0.05)",
        "    gate = (not void) and (fo - fa > 0.05)",
        "1.",
    ),
    (
        "tokens as places",
        "    keep, toks, _owner = tframes.frame_keep(lines, frame_max, min_fillers)",
        "    keep, toks, _owner = frame_keep(lines, frame_max, min_fillers)",
        "1.",
    ),
    (
        "ambiguous filler target",
        '                tok for tok in dict.fromkeys(query["keys"])',
        '                tok for tok in dict.fromkeys(query["vals"])',
        "1.",
    ),
    (
        "no collision void",
        "    void = n < 40 or collision > 0.02",
        "    void = n < 40",
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
