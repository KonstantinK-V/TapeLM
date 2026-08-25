"""Check of 566: LIVE reuse, DEAD not walked.
    python _check566_livedead.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit566_livedead.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if '"LIVE" if hit else "DEAD"' not in src:
        f.append("1. no LIVE/DEAD write")
    if 'mark == "LIVE"' not in src:
        f.append("1. LIVE not reused")
    if "walked_d" not in src:
        f.append("1. DEAD walk not tracked")
    if "Q[" in src or "train_q" in src:
        f.append("3. Q")
    if 'if held in place:' not in src or '"read_hit"' not in src:
        f.append("1. direct READ not removed before PIN")
    if "c != held" in src:
        f.append("1. teacher subtracted from candidate list")
    if "n_d < 10" not in src:
        f.append("2. VOID dead")
    if "al >= 0.80" not in src:
        f.append("2. GATE live agree")
    if 'rec["walked_d"] = 0' not in src:
        f.append("2. GATE dead not walked")
    return f


MUTANTS = (
    ("DEAD still walked",
     '                rec["walked_d"] = 0',
     '                rec["walked_d"] = 1',
     "2."),
    ("no void dead",
     "    void = n_l < 40 or n_d < 10",
     "    void = n_l < 40",
     "2."),
    ("no mark",
     '            W[ek] = (addr, "LIVE" if hit else "DEAD")',
     '            W[ek] = (addr, "LIVE")',
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
