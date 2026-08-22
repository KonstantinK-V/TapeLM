"""Check of 443: late branch after FRESH. hop3 ok, refuse hop4.

  1. Designed has CRISP FRESH + RIPE FRESH (tails split).
  2. Branch refuse via len(c3) != 1.
  3. GATE hop4_rate == 0 and refuse_h4 == 1 and h3_fresh == 1.
  4. Ops from 440; no wiki, no Phi.
"""
from __future__ import annotations

from pathlib import Path

import _audit443_late as M

SRC = Path("_audit443_late.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "CRISP FRESH" not in src or "RIPE FRESH" not in src:
        f.append("1. late CRISP/RIPE world missing")
    if 'w = ["shops sell RIPE FRESH witems gone"' not in src:
        f.append("1. w branch (witems gone) missing")
    if "vstock here" not in src:
        f.append("1. v branch (vstock here) missing")
    if "from _audit440_compose import think_place, think_slot" not in src:
        f.append("1. 440 ops not imported")
    if "if len(c3) != 1:" not in src:
        f.append("2. late refuse (len(c3) != 1) missing")
    if 'rep["hop4_rate"] == 0.0' not in src:
        f.append("3. GATE does not require hop4_rate == 0")
    if 'rep["refuse_h4"] == 1.0' not in src or 'rep["h3_fresh"] == 1.0' not in src:
        f.append("3. GATE missing refuse_h4 / h3")
    if "wikitext" in src:
        f.append("4. wiki leaked")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("w line",
     '    w = ["shops sell RIPE FRESH witems gone" + _pad(60 + i) for i in range(3)]',
     '    w = ["shops sell RIPE FRESH vstock here" + _pad(60 + i) for i in range(3)]',
     "1."),
    ("always unique c3",
     "        if len(c3) != 1:",
     "        if False:  # len(c3) != 1",
     "2."),
    ("allow hop4",
     '            and (rep["hop4_rate"] == 0.0)',
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
