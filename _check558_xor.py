"""Check of 558: extra XOR on hop2 refuse.
    python _check558_xor.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit558_xor.py")
def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if "rng.shuffle(frame)" in src:
        f.append("1. one tape row changes teacher across runs")
    if "len(hits) != 1" not in src:
        f.append("1. no XOR unique")
    if "n_hit < 40" not in src:
        f.append("2. VOID n_hit")
    if 'r["xor"]' not in src:
        f.append("2. GATE not on XOR hits")
    if "len(cand_p) < 2" not in src:
        f.append("1. not the refuse slice")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src or "c != held" in src:
        f.append("1. direct READ/teacher subtraction broken")
    if "True if addr_s == held else" not in src or "addr_s == held or" in src:
        f.append("1. STAR direct answer dropped")
    return f
MUTANTS = (
    ("any extra",
     "    if len(hits) != 1:",
     "    if len(hits) != 99:",
     "1."),
    ("gate all",
     '    hit_rows = [r for r in rows if r["xor"]]',
     "    hit_rows = rows",
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
