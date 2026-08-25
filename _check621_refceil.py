"""Check 621: refuse-only ceiling; leftover unique extras; not peaked pin walk."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit621_refceil.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_ref < 40 or n_live < 40" not in src:
        fails.append("1. VOID missing")
    if "(od1 - rd1) > 0.05" not in src:
        fails.append("1. GATE missing")
    if "def leftover_doors(" not in src:
        fails.append("1. leftover missing")
    if "if hat is not None:" not in src:
        fails.append("1. must skip peaked pin")
    if "apples_in" not in src:
        fails.append("1. apples_in missing")
    if "QTab" in src:
        fails.append("1. learner leaked")
    return fails


MUTANTS = (
    (
        "walk pins too",
        "                    if hat is not None:\n                        n_pin += 1\n                        continue",
        "                    if False:\n                        n_pin += 1\n                        continue",
        "1.",
    ),
    (
        "no leftover",
        "def leftover_doors(pg, pin, skip, env_m, mid_set, high_set, forbid):",
        "def leftover_doors_OFF(pg, pin, skip, env_m, mid_set, high_set, forbid):",
        "1.",
    ),
    (
        "oracle apples only always",
        "                    use = use[:CAP]",
        "                    use = [held_ctx]\n                    live = True\n                    ain = True",
        "1.",
    ),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    if "use = [held_ctx]" in src and "use = use[:CAP]" not in src:
        fails.append("1. forced apples oracle")
    for name, old, new, tag in MUTANTS:
        count = src.count(old)
        if count != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {count}")
            continue
        got = props(src.replace(old, new, 1))
        extra = src.replace(old, new, 1)
        if name == "oracle apples only always":
            if "use = [held_ctx]" in extra:
                got.append("1. forced apples oracle")
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
