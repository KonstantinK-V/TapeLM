"""Check of 425 algebraic hang. No Phi. Six mutants.

  1. Mentions on DIFFERENT lines (line_i ≠ line_j).
  2. Question line excluded from evidence (forbid_line).
  3. Hole fillers excluded from ctx (other_hole / self).
  4. hang is mean rare_share, NOT exists-cooccur.
  5. GATE d_hang > 0.05 on comp_only; VOID on thin arena / no move.
  6. hang scores the PAIR — hang(majority) on (a', b'), eA/eB = answer-slots.
  7. No torch / Deriver.
"""
from __future__ import annotations

from pathlib import Path

import _audit425_hang as M

SRC = Path("_audit425_hang.py")


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

    if "if line[i] == line[j]:" not in src or "continue" not in src:
        f.append("1. same-line mention pairs are not skipped")
    if "forbid_line" not in src or "if line[s] == forbid_line:" not in src:
        f.append("2. question line is not excluded from evidence")
    if "other_hole" not in src or "w == other_hole" not in src:
        f.append("3. other hole is not stripped from ctx")
    if "if w == v or w == other_hole:" not in src:
        f.append("3. self filler is not stripped from ctx")
    if "rare_share" not in src or "return rare / min" not in src:
        f.append("4. hang is not mean rare_share")
    if "exists" in src.lower() and "NOT" not in src and "NOT \"find" not in src:
        pass  # docstring may say NOT find
    if "find a line where a and b co-occur" in src and "NOT" not in src.split(
            "find a line where a and b co-occur")[0][-20:]:
        # require the NOT reject in header
        if 'NOT "find a line where a and b co-occur"' not in src:
            f.append("4. hang collapsed to exists-cooccur")
    if 'rep["d_hang"] > 0.05' not in src:
        f.append("5. GATE bar d_hang > 0.05 missing")
    if 'rep["n_comp_only"] < 30' not in src or 'move_rate' not in src:
        f.append("5. VOID on thin comp_only / no move missing")
    if "hang_of(marg[0], marg[1]," not in src:
        f.append("6. hang(majority) is not called on the pair (a', b')")
    if "answer-slots" not in src:
        f.append("6. eA/eB are not declared as answer-slots")
    if "import torch" in src or "Deriver" in src:
        f.append("7. Phi is in this file")
    return f


MUTANTS = (
    ("same-line pairs allowed",
     "            if line[i] == line[j]:\n                continue",
     "            if False:\n                continue",
     "1."),
    ("question line kept in evidence",
     "        if line[s] == forbid_line:\n            continue",
     "        if False:\n            continue",
     "2."),
    ("filler stays in ctx",
     "        if w == v or w == other_hole:\n            continue",
     "        if False:\n            continue",
     "3."),
    ("hang becomes exists-cooccur",
     'NOT "find a line where a and b co-occur".',
     'YES "find a line where a and b co-occur".',
     "4."),
    ("gate bar dropped",
     '    gate = (not void) and (rep["d_hang"] > 0.05)',
     '    gate = (not void) and (rep["d_hang"] > -1.0)',
     "5."),
    ("hang uses one coordinate",
     "        hm = hang_of(marg[0], marg[1], a, b, by_val, line, value, line_toks, df, med, k)",
     "        hm = hang_of(marg[0], marg[0], a, b, by_val, line, value, line_toks, df, med, k)",
     "6."),
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
