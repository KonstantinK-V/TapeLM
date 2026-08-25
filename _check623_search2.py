"""Check 623: leftover search, no apples-first, stop on hit, refuse-only."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit623_search2.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_live < 40" not in src:
        fails.append("1. VOID missing")
    if "(sd1 - rd1) > 0.05" not in src:
        fails.append("1. GATE missing")
    if "if hat is not None:" not in src:
        fails.append("1. must skip peaked pin")
    if "use = doors[:CAP]" not in src:
        fails.append("1. name-free order missing")
    if "held_ctx] + use" in src or "[held_ctx] +" in src:
        fails.append("1. apples-first leaked")
    if "QTab" in src:
        fails.append("1. learner leaked")
    if "hit_s = 1\n                            break" not in src:
        fails.append("1. stop-on-hit missing")
    return fails


MUTANTS = (
    (
        "apples first",
        "                    use = doors[:CAP]",
        "                    use = ([held_ctx] + [d for d in doors if d != held_ctx])[:CAP]",
        "1.",
    ),
    (
        "skip pin off",
        "                    if hat is not None:\n                        n_pin += 1\n                        continue",
        "                    if False:\n                        n_pin += 1\n                        continue",
        "1.",
    ),
    (
        "no stop",
        "                        if cc[1]:\n                            hit_s = 1\n                            break",
        "                        if cc[1]:\n                            hit_s = 1",
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
