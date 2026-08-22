"""Check of 441: hop3 chain, ops from 440, designed only.

  1. Uses 440 think_slot / think_place; designed has FRESH SWEET.
  2. hop3 address is T = next_place(v2, ...).
  3. GATE includes h3_fresh==1 and hop_changes_12/23==1.
  4. No wiki, no Phi.
"""
from __future__ import annotations

from pathlib import Path

import _audit441_hop3 as M

SRC = Path("_audit441_hop3.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit440_compose import think_place, think_slot" not in src:
        f.append("1. 440 ops not imported")
    if "FRESH SWEET" not in src or "def designed" not in src:
        f.append("1. hop3 world missing")
    if "T = next_place(v2" not in src:
        f.append("2. hop3 address is not next_place(v2)")
    if 'rep["h3_fresh"] == 1.0' not in src:
        f.append("3. GATE missing h3")
    if 'rep["hop_changes_12"] == 1.0' not in src or 'rep["hop_changes_23"] == 1.0' not in src:
        f.append("3. GATE missing hop-change controls")
    if "wikitext" in src:
        f.append("4. wiki leaked")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no 440 import",
     "from _audit440_compose import think_place, think_slot",
     "think_place = think_slot = None",
     "1."),
    ("no v2 address",
     "        T = next_place(v2, s, by_key, R)",
     "        T = next_place(v1, s, by_key, R)",
     "2."),
    ("gate coin h3",
     '            and (rep["h3_fresh"] == 1.0)',
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
