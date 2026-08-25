"""Check of 553: pick a mention, not a rec.

    python _check553_place.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit553_place.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "rng.shuffle(frame)" in src:
        f.append("1. one tape row changes teacher across runs")
    if "take_rn" in src or "rec_rn" in src:
        f.append("1. rec rerank leaked")
    if "len(fr & env_m)" not in src:
        f.append("1. mark = env overlap of a mention")
    if "d_rnd > 0.05" not in src:
        f.append("2. GATE MARK-RND")
    if "room <= 0.05" not in src:
        f.append("2. VOID no ceiling")
    if "hit_ora" not in src:
        f.append("1. oracle place missing")
    return f


MUTANTS = (
    ("rerank rec",
     "    scored.append((len(fr & env_m), t))",
     "    scored.append((0, t))",
     "1."),
    ("gate vs maj",
     "    gate = (not void) and d_rnd > 0.05",
     "    gate = (not void) and d_maj > 0.05",
     "2."),
    ("no ora void",
     "    void = n < 40 or room <= 0.05",
     "    void = n < 40",
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
