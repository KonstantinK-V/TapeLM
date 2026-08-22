"""Check of 435 placepick: CE on e(P), exam mixed. Three mutants.

  1. Offer = foreign slots of place P (434).
  2. Train/exam on mixed; GATE d>0.05 on mixed (not const).
  3. Pin working[("work", 0)]; 432 idea not closed by this STOP.
"""
from __future__ import annotations

from pathlib import Path

import _train435_placepick as M

SRC = Path("_train435_placepick.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "slots_at[p]" not in src or "line[t] != li" not in src:
        f.append("1. offer is not e(P) foreign slots")
    if "exam_m" not in src or "train_m" not in src:
        f.append("2. mixed train/exam missing")
    if "gate = d > 0.05 and hop == 1.0" not in src:
        f.append("2. GATE is not d>0.05 and hop2")
    if "const" in src and "gate" in src and 'd_const' in src:
        f.append("2. constants in GATE")
    if 'working[("work", 0)]' not in src and "working[('work', 0)]" not in src:
        f.append("3. pin missing")
    if "432-style" not in src and "432 idea" not in src:
        if "STOP PICK" not in src:
            f.append("3. STOP does not preserve 432 idea")
    if "comp_only" in src or "pair_seen" in src:
        f.append("3. pair leaked")
    return f


MUTANTS = (
    ("offer not e(P)",
     "        foreign = [t for t in slots_at[p] if t != s and line[t] != li]",
     "        foreign = [t for t in range(len(place)) if t != s and line[t] != li]",
     "1."),
    ("GATE ignores mixed",
     "    gate = d > 0.05 and hop == 1.0",
     "    gate = d > -1.0 and hop == 1.0",
     "2."),
    ("no pin",
     '            working = {("work", 0): t}\n            hop2 += int(working[("work", 0)] == t)',
     "            hop2 += 1",
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
