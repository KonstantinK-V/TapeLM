"""Check of 426 feed autopsy. No hang gate. K=8. No Phi. Four mutants.

  1. Evidence / hang / rare_share come from 425 (answer-slots), not reinvented.
  2. TEXT occ is printed as not evidence; text_occ exists.
  3. DIE cascade: both_nonempty → share_cross → rare → else (hang slice).
  4. No hang gate; default k=8 unchanged; no torch.
"""
from __future__ import annotations

from pathlib import Path

import _audit426_feed as M

SRC = Path("_audit426_feed.py")


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
    if "comp_only" not in rep:
        f.append("0. no comp_only pack")

    if "H425.evidence" not in src or "H425.hang_of" not in src:
        f.append("1. feed does not reuse 425 evidence/hang")
    if "H425.rare_share" not in src or "H425.ctx_of" not in src:
        f.append("1. feed does not reuse 425 rare_share/ctx")

    if "def text_occ" not in src:
        f.append("2. text_occ diagnostic missing")
    if "(not evidence)" not in src:
        f.append("2. TEXT occ is not labeled not evidence")

    if 'if co["n"] and co["both_nonempty"] <= 0.05:' not in src:
        f.append("3. DIE both_nonempty branch missing")
    if 'elif co["n"] and co["share_cross_gt0"] <= 0.05:' not in src:
        f.append("3. DIE share_cross branch missing")
    if 'elif co["n"] and co["share_rare_gt0"] <= 0.05:' not in src:
        f.append("3. DIE rare branch missing")
    if "do not raise K" not in src:
        f.append("3. hang-slice branch / do not raise K missing")

    if 'd_hang' in src and 'gate' in src.lower() and 'rep["d_hang"]' in src:
        f.append("4. hang gate leaked into 426")
    if 'default=8' not in src:
        f.append("4. K default is not 8")
    if "import torch" in src or "Deriver" in src:
        f.append("4. Phi is in this file")
    return f


MUTANTS = (
    ("reinvent evidence",
     "        eA = H425.evidence(va, by_val, line, line[a], {a, b}, k)\n"
     "        eB = H425.evidence(vb, by_val, line, line[a], {a, b}, k)",
     "        eA = by_val.get(va, [])[:k]\n"
     "        eB = by_val.get(vb, [])[:k]",
     "1."),
    ("TEXT sold as evidence",
     "              f\"both {p['share_text_both']:.3f}  (not evidence)\"",
     "              f\"both {p['share_text_both']:.3f}  (evidence)\"",
     "2."),
    ("DIE cascade collapsed",
     '    if co["n"] and co["both_nonempty"] <= 0.05:',
     '    if False and co["n"] and co["both_nonempty"] <= 0.05:',
     "3."),
    ("K raised",
     '    ap.add_argument("--k", type=int, default=8)',
     '    ap.add_argument("--k", type=int, default=32)',
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
