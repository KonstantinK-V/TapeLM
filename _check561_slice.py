"""Check of 561: freeze PIN; learn only on refuse.
    python _check561_slice.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit561_slice.py")
def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if 'if r["n_cand"] == 1:\n            return "P"' not in src:
        f.append("1. PIN not frozen to P")
    if 'r["n_cand"] != 1' not in src:
        f.append("1. refuse slice missing")
    if "u_s2 <= 0.05" not in src:
        f.append("2. VOID on refuse STAR mass")
    if "d557 > 0.05 and ds > 0.05" not in src:
        f.append("2. GATE overall and slice")
    if "key2" not in src:
        f.append("1. key without n_cand missing")
    return f
MUTANTS = (
    ("learn on pin",
     '        if r["n_cand"] == 1:\n            return "P"',
     '        if r["n_cand"] == 1:\n            return pick(q, key2(r), rng_te)',
     "1."),
    ("void without u_S2",
     "    void = n2 < 40 or u_s2 <= 0.05",
     "    void = n2 < 40",
     "2."),
    ("gate only overall",
     "    gate = (not void) and d557 > 0.05 and ds > 0.05",
     "    gate = (not void) and d557 > 0.05",
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
