"""Check of 555: READ place vs star, equal n.

    python _check555_read.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit555_read.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "rng.shuffle(frame)" in src:
        f.append("1. one tape row changes teacher across runs")
    if "rec_gl[: len(take_p)]" not in src:
        f.append("1. STAR budget ≠ |PLACE|")
    if "d_star > 0.05" not in src:
        f.append("2. GATE PLACE-STAR")
    if "gap > 0.05" not in src:
        f.append("2. VOID budget leak")
    if "ov / max(len(fr), 1)" not in src:
        f.append("1. stand is JACC, not count")
    return f


MUTANTS = (
    ("star full rec",
     "    take_s = rec_gl[: len(take_p)]",
     "    take_s = rec_gl",
     "1."),
    ("gate vs rnd",
     "    gate = (not void) and d_star > 0.05",
     "    gate = (not void) and d_rnd > 0.05",
     "2."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {n}")
            continue
        got = props(src=src.replace(old, new, 1))
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): not caught")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
