"""Runtime place-walk contract: READ first, scored reuse, no fake LIVE."""
from __future__ import annotations

from pathlib import Path

SRC = Path("_place_walk.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    fails = []
    if 'out["action"] = "READ"' not in src or 'out["why"] = "read_hit"' not in src:
        fails.append("1. direct READ missing")
    if 'out["why"] = "live_reuse_miss"' not in src:
        fails.append("1. LIVE reuse not checked against current record")
    if 'out["action"] = "STEP"' not in src or '"unscored_read"' not in src:
        fails.append("1. held-free ask path fabricates a scored PIN")
    if "if held is None or held in fr2" in src:
        fails.append("1. held=None is treated as a hit")
    if 'W_env[ek] = (addr, "LIVE")' not in src:
        fails.append("1. scored LIVE write missing")
    return fails


MUTANTS = (
    (
        "fake ask hit",
        "        if held in fr2:",
        "        if held is None or held in fr2:",
        "1.",
    ),
    (
        "reuse never checked",
        '            out["why"] = "live_reuse_miss"',
        '            out["why"] = None',
        "1.",
    ),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props(src)
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {src.count(old)}")
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
