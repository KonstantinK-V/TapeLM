"""Check of 552: slice only, no new ranker.

    python _check552_spec.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit552_spec.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "modes" in src or "rn_idf" in src:
        f.append("1. sweep leaked")
    if "cnt_m[held] > cnt_o[held]" not in src:
        f.append("1. SPEC = held prefers mates")
    if 's["d"] > 0.05' not in src:
        f.append("2. GATE on SPEC not ALL")
    if "n_spec < 40" not in src and 's["n"] < 40' not in src:
        f.append("2. VOID n_spec")
    if "sorted(rec_gl" not in src:
        f.append("1. same RN as 551")
    return f


MUTANTS = (
    ("gate on all",
     '    gate = (not void) and s["d"] > 0.05',
     '    gate = (not void) and a["d"] > 0.05',
     "2."),
    ("spec always true",
     "    spec = cnt_m[held] > cnt_o[held]",
     "    spec = True",
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
