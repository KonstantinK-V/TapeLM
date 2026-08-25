"""Check of 540: cumulative rank cover, maj included, ranks from 0."""
from __future__ import annotations

from pathlib import Path

import _audit540_budget as M

SRC = Path("_audit540_budget.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def cumul_profile(" not in src:
        f.append("1. cumul_profile missing")
    if "any_so_far = any_so_far or hit[m]" not in src:
        f.append("1. cumulative any-of-ranks update missing")
    if "skip_maj" in src or "c != maj" in src or "held2" in src:
        f.append("1. maj must not be excluded")
    if "sub.append((rec, held))" not in src:
        f.append("1. conditional any-hit slice missing")
    if "allow_of" not in src:
        f.append("1. allow_of mid histogram missing")
    if "gain_a = cum_a[2] - cum_a[0]" not in src:
        f.append("1. Δ02 budget diagnostic missing")
    if "gain_mc >= 0.20 or gain_ac >= 0.20" not in src:
        f.append("2. GATE budget via conditional cumul missing")
    if "import torch" in src:
        f.append("4. CE leaked")
    return f


MUTANTS = (
    ("exclude maj",
     "            if rec:\n                all_rows.append((rec, held))",
     "            if rec:\n                held2 = {c for c in held if c != maj}\n                all_rows.append((rec, held2))",
     "1."),
    ("no conditional slice",
     "        if any(i < len(rec) and rec[i] in held for i in range(R_TOP)):\n            sub.append((rec, held))",
     "        pass",
     "1."),
    ("no cumulative",
     "            any_so_far = any_so_far or hit[m]",
     "            pass",
     "1."),
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
