"""Check of 427 hang: rare ≔ df <= median. K=8. No Phi. Four mutants.

  1. rare uses df <= med (not strict <).
  2. VOID checks n_rare_types == 0 first (do not read hang if empty).
  3. Reuses 425 evidence/ctx; hang_of(majority) on pair.
  4. K default 8; no torch / Deriver.
"""
from __future__ import annotations

from pathlib import Path

import _audit427_hang as M

SRC = Path("_audit427_hang.py")


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

    if "df.get(w, 0) <= med" not in src:
        f.append("1. rare is not df <= median")
    if "df.get(w, 0) < med" in src:
        f.append("1. strict df < med leaked back")
    if 'v <= med' not in src:
        f.append("1. n_rare_types is not df <= median")

    if 'rep["n_rare_types"] == 0' not in src:
        f.append("2. VOID does not check n_rare_types == 0")
    void_line = 'void = (rep["n_rare_types"] == 0) or (rep["n_comp_only"] < 30) or (rep["move_rate"] <= 0.05)'
    if void_line not in src:
        f.append("2. VOID without n_rare_types first")

    if "H425.evidence" not in src or "H425.ctx_of" not in src:
        f.append("3. does not reuse 425 evidence/ctx")
    if "hang_of(marg[0], marg[1]," not in src:
        f.append("3. hang(majority) not on the pair")

    if "default=K" not in src and "default=8" not in src:
        f.append("4. K default is not 8")
    if "import torch" in src or "Deriver" in src:
        f.append("4. Phi is in this file")
    return f


MUTANTS = (
    ("strict < median",
     "    rare = sum(1 for w in (s1 & s2) if df.get(w, 0) <= med)",
     "    rare = sum(1 for w in (s1 & s2) if df.get(w, 0) < med)",
     "1."),
    ("VOID skips rare-types",
     '    void = (rep["n_rare_types"] == 0) or (rep["n_comp_only"] < 30) or (rep["move_rate"] <= 0.05)',
     '    void = (rep["n_comp_only"] < 30) or (rep["move_rate"] <= 0.05)',
     "2."),
    ("drop 425 evidence",
     "    eA = H425.evidence(va, by_val, line, line[a], {a, b}, k)\n"
     "    eB = H425.evidence(vb, by_val, line, line[a], {a, b}, k)",
     "    eA = by_val.get(va, [])[:k]\n"
     "    eB = by_val.get(vb, [])[:k]",
     "3."),
    ("Phi leaked",
     'OUT = Path("results/_stage427_hang.json")',
     'OUT = Path("results/_stage427_hang.json")\nimport torch',
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
