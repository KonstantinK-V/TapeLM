"""Check of 432: place-teacher, not a pair."""
from __future__ import annotations

from pathlib import Path

import _audit432_place as M

SRC = Path("_audit432_place.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "line[t] != li" not in src:
        f.append("1. question line is in evidence")
    if "slots_at[p]" not in src:
        f.append("1. evidence is not by place")
    if "by_val" in src:
        f.append("1. value is used as address")
    if 'void = (rep["foreign_nonempty"] <= 0.05) or (rep["mixed"] <= 0.05)' not in src:
        f.append("2. VOID is not foreign/mixed")
    if "best > 0.05" not in src:
        f.append("3. GATE is not best d > 0.05")
    if 'mix_hits["maj"]' not in src:
        f.append("3. majority is not the mixed rival")
    if "pair_seen" in src or "comp_only" in src:
        f.append("4. pair object leaked")
    if "import torch" in src:
        f.append("4. Phi is in this file")
    return f


MUTANTS = (
    ("question line in evidence",
     "        foreign = [t for t in slots_at[p] if t != s and line[t] != li][:k]",
     "        foreign = [t for t in slots_at[p] if t != s][:k]",
     "1."),
    ("VOID skips catalog",
     '    void = (rep["foreign_nonempty"] <= 0.05) or (rep["mixed"] <= 0.05)',
     '    void = (rep["foreign_nonempty"] <= 0.05)',
     "2."),
    ("gate is coin",
     "    gate = (not void) and (best > 0.05)",
     "    gate = (not void) and (best > -1.0)",
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
