"""Check of 436 const-pin think. Mixed logged, not trained. Three mutants.

  1. Agree → pin majority of e(P); differ → refuse; df1 → skip/refuse.
  2. GATE: const_hit>0.90, refuse_mixed>0.90 (or no mixed), hop2.
  3. Mixed not in loss/gradient; no Phi.
"""
from __future__ import annotations

from pathlib import Path

import _audit436_constpin as M

SRC = Path("_audit436_constpin.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "maj_v == v" not in src:
        f.append("1. agree/const branch missing")
    if "ref_m += 1" not in src:
        f.append("1. mixed refuse missing")
    if 'working[("work", 0)]' not in src and "working[('work', 0)]" not in src:
        f.append("1. pin missing")
    if 'rep["const_hit"] > 0.90' not in src:
        f.append("2. GATE const_hit > 0.90 missing")
    if 'refuse_mixed"] > 0.90' not in src:
        f.append("2. GATE refuse_mixed > 0.90 missing")
    if "hop2_sees_pin" not in src:
        f.append("2. hop2 gate missing")
    if "import torch" in src or "PickNet" in src or "backward" in src:
        f.append("3. Phi/loss leaked — mixed must not train")
    if "not a loss" not in src and "not trained" not in src:
        f.append("3. mixed-as-counter note missing")
    return f


MUTANTS = (
    ("mixed also pins",
     "            ref_m += 1",
     "            hit_c += 1  # mixed wrongly counted as hit",
     "1."),
    ("GATE drops refuse_mixed",
     '    gate = ((not void) and (rep["const_hit"] > 0.90)\n'
     '            and (rep["n_mixed"] == 0 or rep["refuse_mixed"] > 0.90)\n'
     '            and (rep["hop2_sees_pin"] == 1.0))',
     '    gate = ((not void) and (rep["const_hit"] > 0.90)\n'
     '            and (rep["hop2_sees_pin"] == 1.0))',
     "2."),
    ("Phi leaked",
     'OUT = Path("results/_stage436_constpin.json")',
     'OUT = Path("results/_stage436_constpin.json")\nimport torch\nPickNet = object',
     "3."),
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
