"""Check of 444: 2-cand + query-frame extra. CRISP not RIPE.

  1. Designed has CRISP FRESH mat; ops from 440.
  2. Resolve via place_keys[p] & extra.
  3. GATE resolved_crisp==1 and picked_ripe==0.
  4. No Phi / wiki. Without extra this is 442-refuse, not idea failure.
"""
from __future__ import annotations

from pathlib import Path

import _audit444_resolve as M

SRC = Path("_audit444_resolve.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "CRISP FRESH mat" not in src:
        f.append("1. CRISP FRESH mat world missing")
    if "from _audit440_compose import think_place, think_slot" not in src:
        f.append("1. 440 ops not imported")
    if "place_keys[p] & extra" not in src:
        f.append("2. query-frame intersect missing")
    if 'rep["resolved_crisp"] == 1.0' not in src:
        f.append("3. GATE missing resolved_crisp")
    if 'rep["picked_ripe"] == 0.0' not in src:
        f.append("3. GATE allows RIPE")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    if "wikitext" in src:
        f.append("4. wiki leaked")
    return f


MUTANTS = (
    ("no mat on CRISP",
     '    v = ["crates mark CRISP FRESH mat here" + _pad(50 + i) for i in range(3)]',
     '    v = ["crates mark CRISP FRESH vstock here" + _pad(50 + i) for i in range(3)]',
     "1."),
    ("no intersect",
     "        hit = {p for p in c3 if place_keys[p] & extra}",
     "        hit = set(c3)",
     "2."),
    ("allow RIPE",
     '            and (rep["picked_ripe"] == 0.0)',
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
