"""Check of 433 pinwalk: pin cell that holds v. Ceiling first. Three mutants.

  1. Question line excluded from the offer bag.
  2. CEILING / live before train; STOP if ora−rnd ≤ 0.05.
  3. Pin via working[("work", 0)]; hop2 reads that cell.
"""
from __future__ import annotations

from pathlib import Path

import _train433_pinwalk as M

SRC = Path("_train433_pinwalk.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "if line[t] == li:" not in src:
        f.append("1. question line is not excluded from the bag")
    if "continue" not in src.split("if line[t] == li:", 1)[-1][:80]:
        f.append("1. question line skip is broken")
    if "CEILING" not in src:
        f.append("2. CEILING is not printed")
    if "if d_ceil <= 0.05:" not in src:
        f.append("2. STOP CEILING does not skip train")
    if "live" not in src:
        f.append("2. live is missing")
    if 'working[("work", 0)]' not in src:
        f.append("3. working[(work, 0)] pin missing")
    if "hop2_sees_pin" not in src:
        f.append("3. hop2_sees_pin missing")
    if "comp_only" in src or "pair_seen" in src:
        f.append("3. pair object leaked")
    return f


MUTANTS = (
    ("question line in bag",
     "            if line[t] == li:\n                continue",
     "            if False:\n                continue",
     "1."),
    ("train without ceiling",
     "    if d_ceil <= 0.05:",
     "    if False and d_ceil <= 0.05:",
     "2."),
    ("no working pin",
     '            working[("work", 0)] = t\n            hop2 = working[("work", 0)]',
     "            hop2 = t",
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
