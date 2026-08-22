"""Check of 434 place-pin offer. Ceiling only. Three mutants.

  1. Offer = foreign slots of same place P (slots_at[p], other line).
  2. GATE only on mixed (d_mixed > 0.05); constants are not GO.
  3. Pin via working[("work", 0)]; no Phi / pair.
"""
from __future__ import annotations

from pathlib import Path

import _audit434_placepin as M

SRC = Path("_audit434_placepin.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "slots_at[p]" not in src:
        f.append("1. offer is not slots_at[p]")
    if "line[t] != li" not in src:
        f.append("1. question line not excluded from foreign")
    if "keys[s]" in src or "frame keys" in src.lower():
        f.append("1. 433 window key-bag leaked back")
    if 'rep["n_mixed"] >= 30' not in src:
        f.append("2. GATE does not require n_mixed")
    if 'rep["d_mixed"] > 0.05' not in src:
        f.append("2. GATE is not d_mixed > 0.05")
    if "d_const" in src and "gate" in src.lower():
        if 'd_const"] > 0.05' in src or "d_const > 0.05" in src:
            f.append("2. constants counted as GO")
    if 'working[("work", 0)]' not in src and "working[('work', 0)]" not in src:
        f.append("3. pin working[(work, 0)] missing")
    if "import torch" in src or "PickNet" in src:
        f.append("3. Phi leaked")
    if "comp_only" in src or "pair_seen" in src:
        f.append("3. pair object leaked")
    return f


MUTANTS = (
    ("offer is key-window again",
     "        foreign = [t for t in slots_at[p] if t != s and line[t] != li]",
     "        foreign = [t for t in range(len(place)) if t != s and line[t] != li]",
     "1."),
    ("GATE uses const",
     '    gate = (not void) and (rep["n_mixed"] >= 30) and (rep["d_mixed"] > 0.05)',
     '    gate = (not void) and (rep["d_const"] > 0.05)',
     "2."),
    ("no working pin",
     '        working = {("work", 0): teach}\n        hop2 = working[("work", 0)]',
     "        hop2 = teach",
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
