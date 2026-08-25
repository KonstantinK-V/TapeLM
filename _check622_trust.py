"""Check 622: TRUST unique argmax; recip from extracts; refuse-only; not Φ."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit622_trust.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_tr < 40" not in src:
        fails.append("1. VOID missing")
    if "(td1 - rd1) > 0.05" not in src:
        fails.append("1. GATE missing")
    if "if hat is not None:" not in src:
        fails.append("1. must skip peaked pin")
    if "def recip(" not in src:
        fails.append("1. reciprocal missing")
    if "len(tops) == 1" not in src:
        fails.append("1. unique argmax missing")
    if "QTab" in src:
        fails.append("1. learner leaked")
    return fails


MUTANTS = (
    (
        "skip pin off",
        "                    if hat is not None:\n                        n_pin += 1\n                        continue",
        "                    if False:\n                        n_pin += 1\n                        continue",
        "1.",
    ),
    (
        "always trust first",
        "                    if len(tops) == 1:\n                        pick = tops[0]",
        "                    if True:\n                        pick = tops[0]",
        "1.",
    ),
    (
        "no recip",
        "def recip(pg, door, skip, env_m, mid_set, co, df, n_use):",
        "def recip_OFF(pg, door, skip, env_m, mid_set, co, df, n_use):",
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
        if name == "always trust first" and "len(tops) == 1" not in mut:
            got.append("1. unique argmax missing")
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
