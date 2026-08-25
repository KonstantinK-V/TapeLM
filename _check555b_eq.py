"""Check of 555b: drop short rec.
    python _check555b_eq.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit555b_eq.py")
def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "rng.shuffle(frame)" in src:
        f.append("1. one tape row changes teacher across runs")
    if 'return None, "short_star"' not in src:
        f.append("1. no short_star skip")
    if "rec_gl[: len(fr_p)]" not in src:
        f.append("1. STAR not sliced to |PLACE|")
    if "d_star > 0.05" not in src:
        f.append("2. GATE PLACE−STAR")
    if "gap > 0.02" not in src:
        f.append("2. VOID gap")
    return f
MUTANTS = (
    ("keep short",
     '        return None, "short_star"',
     "        pass",
     "1."),
    ("gate rnd",
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
