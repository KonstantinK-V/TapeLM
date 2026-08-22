"""Check of 428 hang: rare ≔ 2 ≤ df ≤ 5. Last calibration. Four mutants.

  1. DF_LO=2 (hapax DF_LO=1 forbidden); DF_HI=5.
  2. VOID checks n_bridge_types == 0 first.
  3. Reuses 425 evidence/ctx; GATE d_hang > 0.05.
  4. K=8; no torch / Deriver.
"""
from __future__ import annotations

from pathlib import Path

import _audit428_hang as M

SRC = Path("_audit428_hang.py")


def _pad(k):
    return " " + " ".join(f"p{k}x{j}" for j in range(24))


DESIGNED = [
    "the capital of FRANCE was recorded as the city PARIS in one" + _pad(0),
    "the capital of FRANCE was recorded as the city PARIS in two" + _pad(1),
    "the capital of ITALY was recorded as the city ROME in one" + _pad(2),
    "the capital of ITALY was recorded as the city ROME in two" + _pad(3),
]


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    class A:
        frame_max = 3
        min_fillers = 2
        addresses = 1500
        pairs_per_line = 4
        max_questions = 200
        k = 8
    import random
    rng = random.Random(0)
    rep = M.measure(DESIGNED, A, rng)
    if rep is None:
        return ["0. designed tape built nothing"]

    if "DF_LO, DF_HI = 2, 5" not in src:
        f.append("1. DF band is not 2..5")
    if "DF_LO, DF_HI = 1," in src or "DF_LO = 1" in src:
        f.append("1. hapax DF_LO=1 leaked in")
    if "lo <= df.get(w, 0) <= hi" not in src:
        f.append("1. rare_share is not 2..5 band")

    void_line = (
        'void = (rep["n_bridge_types"] == 0) or (rep["n_comp_only"] < 30) '
        'or (rep["move_rate"] <= 0.05)'
    )
    if void_line not in src:
        f.append("2. VOID without n_bridge_types == 0 first")
    if 'rep["n_bridge_types"] == 0' not in src:
        f.append("2. bridge empty VOID missing")

    if "H425.evidence" not in src or "H425.ctx_of" not in src:
        f.append("3. does not reuse 425 evidence/ctx")
    if 'rep["d_hang"] > 0.05' not in src:
        f.append("3. GATE d_hang > 0.05 missing")

    if "default=K" not in src and "default=8" not in src:
        f.append("4. K default is not 8")
    if "import torch" in src or "Deriver" in src:
        f.append("4. Phi is in this file")
    return f


MUTANTS = (
    ("hapax DF_LO=1",
     "DF_LO, DF_HI = 2, 5",
     "DF_LO, DF_HI = 1, 5",
     "1."),
    ("VOID skips bridge",
     '    void = (rep["n_bridge_types"] == 0) or (rep["n_comp_only"] < 30) or (rep["move_rate"] <= 0.05)',
     '    void = (rep["n_comp_only"] < 30) or (rep["move_rate"] <= 0.05)',
     "2."),
    ("gate bar dropped",
     '    gate = (not void) and (rep["d_hang"] > 0.05)',
     '    gate = (not void) and (rep["d_hang"] > -1.0)',
     "3."),
    ("Phi leaked",
     'OUT = Path("results/_stage428_hang.json")',
     'OUT = Path("results/_stage428_hang.json")\nimport torch',
     "4."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        saved = dict(M.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), M.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
        finally:
            M.__dict__.clear()
            M.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
