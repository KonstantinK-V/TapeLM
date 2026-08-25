"""Check of 557: pin unique hop2, refuse many.
    python _check557_pin.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit557_pin.py")
def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "rng.shuffle(frame)" in src:
        f.append("1. one tape row changes teacher across runs")
    if "len(cand_p) >= 2" not in src:
        f.append("1. no refuse branch")
    if "n_pin < 40" not in src:
        f.append("2. VOID on pin count")
    if 'r["pin"]' not in src:
        f.append("2. GATE not on PIN only")
    if "addr_p = rng.choice(cand_p)" in src:
        f.append("1. still random among many")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src:
        f.append("1. direct READ not removed before PIN")
    if "c != held" in src:
        f.append("1. teacher subtracted from candidate list")
    if "True if addr_s == held else" not in src or "addr_s == held or" in src:
        f.append("1. STAR direct answer dropped")
    return f
MUTANTS = (
    ("always pin",
     "    if len(cand_p) >= 2:",
     "    if len(cand_p) >= 99:",
     "1."),
    ("gate all rows",
     '    pin_rows = [r for r in rows if r["pin"]]',
     "    pin_rows = rows",
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
