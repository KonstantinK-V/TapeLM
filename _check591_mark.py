"""Check 591: skip DEAD on 589 walk; VOID if mark never bites."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit591_mark.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src or "QTab" in src:
        fails.append("1. Q/torch leaked")
    if "from _audit589_hop3 import collect" not in src:
        fails.append("1. not the 589 walk")
    if "or n_changed < 40" not in src:
        fails.append("1. VOID n_changed missing")
    if "d_pmi > 0.05" not in src:
        fails.append("1. GATE not skip-PMI")
    if "marks.get(tok) != DEAD" not in src:
        fails.append("1. skip DEAD missing")
    if "unique_extras" in src:
        fails.append("1. reimplemented walk")
    return fails


MUTANTS = (
    (
        "no skip",
        "        if marks.get(tok) != DEAD:",
        "        if True:",
        "1.",
    ),
    (
        "no void changed",
        "    void = acc[\"pmi\"][\"n\"] < 40 or n_changed < 40",
        "    void = acc[\"pmi\"][\"n\"] < 40",
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
