"""Check of 430 frame hang. Ceiling only. No Phi. Four mutants.

  1. Fingerprint is slot frame left∪right, not full line bag.
  2. Cross-line only (line_i ≠ line_j).
  3. GATE d_hang > 0.05; VOID on thin / no move / empty frame.
  4. No torch / Deriver / HangNet — Phi out.
"""
from __future__ import annotations

from pathlib import Path

import _audit430_framehang as M

SRC = Path("_audit430_framehang.py")


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

    if "def frame_bag" not in src:
        f.append("1. frame_bag missing")
    if "fr = list(left) + list(right)" not in src:
        f.append("1. frame is not left∪right")
    if "H425.ctx_of" in src.split("def hang_of")[1].split("\ndef measure")[0]:
        f.append("1. hang still uses line ctx_of")
    if "line_toks" in src and "frame_bag" in src:
        # line_toks may be absent in 430 — ok if not used in hang
        pass

    if "if line[i] == line[j]:" not in src:
        f.append("2. same-line pairs not skipped")

    if 'rep["d_hang"] > 0.05' not in src:
        f.append("3. GATE d_hang > 0.05 missing")
    void_line = (
        'void = (rep["n_comp_only"] < 30) or (rep["move_rate"] <= 0.05) '
        'or (rep["mean_frame"] <= 0)'
    )
    if void_line not in src:
        f.append("3. VOID contract missing")

    if "import torch" in src or "Deriver" in src or "HangNet" in src:
        f.append("4. Phi leaked into 430")
    if "Do not Phi" not in src and "No Phi" not in src:
        f.append("4. file does not refuse Phi")
    return f


MUTANTS = (
    ("line bag instead of frame",
     "        fr = list(left) + list(right)",
     "        fr = []  # emptied — not left∪right frame",
     "1."),
    ("same-line allowed",
     "            if line[i] == line[j]:\n                continue",
     "            if False:\n                continue",
     "2."),
    ("gate bar dropped",
     '    gate = (not void) and (rep["d_hang"] > 0.05)',
     '    gate = (not void) and (rep["d_hang"] > -1.0)',
     "3."),
    ("Phi leaked",
     'OUT = Path("results/_stage430_framehang.json")',
     'OUT = Path("results/_stage430_framehang.json")\nimport torch\nHangNet = object',
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
