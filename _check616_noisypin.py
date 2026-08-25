"""Check 616: noisy pin = Petya extract; match-slice; no oracle start."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit616_noisypin.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_pin < 40" not in src:
        fails.append("1. VOID missing")
    if "(nd1 - rd1) > 0.05" not in src:
        fails.append("1. GATE noise vs rand missing")
    if "first_pin" not in src:
        fails.append("1. noisy pin missing")
    if "match = hat == held_ctx" not in src:
        fails.append("1. match-slice missing")
    if "pg, held_ctx, held_ask" in src:
        fails.append("1. oracle start leaked")
    if "QTab" in src:
        fails.append("1. learner leaked")
    return fails


MUTANTS = (
    (
        "oracle start",
        "                        pg, hat, held_ask, qi, env_m, mid_set, high_set,",
        "                        pg, held_ctx, held_ask, qi, env_m, mid_set, high_set,",
        "1.",
    ),
    (
        "no match",
        "                match = hat == held_ctx",
        "                match = False",
        "1.",
    ),
    (
        "gate extra2",
        "    gate = (not void) and (nd1 - rd1) > 0.05",
        "    gate = (not void) and (e2 - e2r) > 0.05",
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
