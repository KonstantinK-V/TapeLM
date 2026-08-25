"""Check 619: hop2 gate on extra2; pin from 618; not oracle."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit619_peakhop2.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if "torch" in src:
        fails.append("1. torch leaked")
    if "void = n_peak < 40" not in src:
        fails.append("1. VOID missing")
    if "e2, e2r = cd2 - cd1, rd2 - rd1" not in src:
        fails.append("1. extra2 formula missing")
    if "(e2 - e2r) > 0.05" not in src:
        fails.append("1. GATE extra2 missing")
    if "peak_pin(" not in src:
        fails.append("1. 618 pin missing")
    if "gate = (not void) and (cd1 - rd1) > 0.05" in src:
        fails.append("1. gated d1 (already 618)")
    if "pg, held_ctx, held_ask" in src:
        fails.append("1. oracle leaked")
    if "QTab" in src:
        fails.append("1. learner leaked")
    return fails


MUTANTS = (
    (
        "gate d1",
        "    gate = (not void) and (e2 - e2r) > 0.05",
        "    gate = (not void) and (cd1 - rd1) > 0.05",
        "1.",
    ),
    (
        "oracle",
        "                        pg, hat, held_ask, qi, env_m, mid_set, high_set,",
        "                        pg, held_ctx, held_ask, qi, env_m, mid_set, high_set,",
        "1.",
    ),
    (
        "no extra2",
        "    e2, e2r = cd2 - cd1, rd2 - rd1",
        "    e2, e2r = 0.0, 0.0",
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
