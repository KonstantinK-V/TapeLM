"""Check of 442: branch honesty. FRESH+DRY, refuse when len(c2)!=1.

  1. Designed has FRESH and DRY; ops from 440.
  2. Branch refuse via len(c2) != 1.
  3. GATE hop3_rate == 0 and refuse_h3 == 1.
  4. No wiki, no Phi.
"""
from __future__ import annotations

from pathlib import Path

import _audit442_branch as M

SRC = Path("_audit442_branch.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "FRESH SWEET" not in src or "DRY SWEET" not in src:
        f.append("1. FRESH+DRY branch world missing")
    if "from _audit440_compose import think_place, think_slot" not in src:
        f.append("1. 440 ops not imported")
    if "if len(c2) != 1:" not in src:
        f.append("2. branch refuse (len(c2) != 1) missing")
    if 'rep["hop3_rate"] == 0.0' not in src:
        f.append("3. GATE does not require hop3_rate == 0")
    if 'rep["refuse_h3"] == 1.0' not in src:
        f.append("3. GATE missing refuse_h3")
    if "wikitext" in src:
        f.append("4. wiki leaked")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no DRY",
     '    u = ["fields keep DRY SWEET grain here" + _pad(40 + i) for i in range(3)]',
     '    u = ["fields keep ONLY SWEET grain here" + _pad(40 + i) for i in range(3)]',
     "1."),
    ("always unique",
     "        if len(c2) != 1:",
     "        if False:  # len(c2) != 1",
     "2."),
    ("allow hop3",
     '            and (rep["hop3_rate"] == 0.0)',
     "            and True",
     "3."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {n} times")
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
