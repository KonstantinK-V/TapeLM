"""Check of 556: hop2 PLACE neighbor vs STAR hub.
    python _check556_hop2.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit556_hop2.py")
def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "rng.shuffle(frame)" in src:
        f.append("1. one tape row changes teacher across runs")
    if "def stand_read" not in src:
        f.append("1. no second stand")
    if "addr_s = rec_gl[0]" not in src:
        f.append("1. STAR hub missing")
    if "d > 0.05" not in src:
        f.append("2. GATE P−S")
    if "len(frame) < 3" not in src:
        f.append("1. hop2 needs 3 fillers")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src:
        f.append("1. direct READ not removed before hop2")
    if "c != held" in src:
        f.append("1. teacher subtracted from candidate list")
    if "True if addr_s == held else" not in src or "addr_s == held or" in src:
        f.append("1. STAR direct answer dropped")
    return f
MUTANTS = (
    ("no hub",
     "    addr_s = rec_gl[0]",
     "    addr_s = addr_p",
     "1."),
    ("gate flip",
     "    gate = (not void) and d > 0.05",
     "    gate = (not void) and d < 0.05",
     "2."),
    ("teacher-subtracted candidates",
     "    cand_p = [c for c in fr_p if c in mid_set and c != v]",
     "    cand_p = [c for c in fr_p if c in mid_set and c != held and c != v]",
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
