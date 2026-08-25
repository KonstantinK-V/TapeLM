"""Check of 559: chooser arena — both PLACE and STAR unique needed.
    python _check559_arena.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit559_arena.py")
def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "rng.shuffle(frame)" in src:
        f.append("1. one tape row changes teacher across runs")
    if 'gold = "P"' not in src or 'gold = "S"' not in src:
        f.append("1. no unique PLACE/STAR labels")
    if "u_s > 0.05 and u_p > 0.05" not in src:
        f.append("2. GATE needs both uniques")
    if "len(cand_p) == 1" not in src:
        f.append("1. PLACE not 557 pin")
    if "always_p" not in src:
        f.append("1. no always-PLACE ceiling")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src or "c != held" in src:
        f.append("1. direct READ/teacher subtraction broken")
    if "True if addr_s == held else" not in src or "addr_s == held or" in src:
        f.append("1. STAR direct answer dropped")
    return f
MUTANTS = (
    ("gate only place",
     "    gate = (not void) and u_s > 0.05 and u_p > 0.05",
     "    gate = (not void) and u_p > 0.05",
     "2."),
    ("no star unique",
     '    elif hit_s and not hit_p:\n        gold = "S"',
     '    elif hit_s and not hit_p:\n        gold = "R"',
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
