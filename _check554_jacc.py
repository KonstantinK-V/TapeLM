"""Check of 554: Jaccard vs count.

    python _check554_jacc.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit554_jacc.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "rng.shuffle(frame)" in src:
        f.append("1. one tape row changes teacher across runs")
    if "ov / max(len(fr), 1)" not in src:
        f.append("1. Jaccard missing")
    if "d_j > 0.05" not in src:
        f.append("2. GATE on JACC not CNT")
    if "len_cnt" not in src:
        f.append("1. frame-length print")
    if "hit_cnt" not in src:
        f.append("1. 553 count arm missing")
    return f


MUTANTS = (
    ("gate on count",
     "    gate = (not void) and d_j > 0.05",
     "    gate = (not void) and d_c > 0.05",
     "2."),
    ("jacc is count",
     "        jac.append((ov / max(len(fr), 1), t))",
     "        jac.append((ov, t))",
     "1."),
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
