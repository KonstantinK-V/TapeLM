"""Check 590: same 589 walk on foreign tape; gate hop1+hop2 vs rnd."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit590_xfer.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src or "QTab" in src:
        fails.append("1. Q/torch leaked")
    if "from _audit589_hop3 import collect" not in src:
        fails.append("1. not same walk as 589")
    if "d1 > 0.05" not in src or "d2 > 0.05" not in src:
        fails.append("1. GATE not hop1 and hop2 vs rnd")
    if 'ev["n2_pmi"] < 40' not in src:
        fails.append("1. VOID n2_pmi missing")
    if "d3 > 0.05" in src.split("gate =")[1].split("\n")[0]:
        fails.append("1. hop3 wrongly in GATE")
    if "score_eps" not in src:
        fails.append("1. score_eps missing")
    return fails


MUTANTS = (
    (
        "gate hop1 only",
        "    gate = (not void) and d1 > 0.05 and d2 > 0.05",
        "    gate = (not void) and d1 > 0.05",
        "1.",
    ),
    (
        "no 589 import",
        "from _audit589_hop3 import collect, prefix_windows, score_eps",
        "from _audit589_hop3 import prefix_windows, score_eps",
        "1.",
    ),
    (
        "void without n2",
        '    void = ev["n"] < 40 or ev["n2_pmi"] < 40',
        '    void = ev["n"] < 40',
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
