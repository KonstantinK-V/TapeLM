"""Check of 453: D1/D2/D4/STOP, while-loop, no hops==2 gate."""
from __future__ import annotations

from pathlib import Path

import _audit453_depth as M

SRC = Path("_audit453_depth.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if '"D4": world_d4' not in src or "world_d1" not in src or "world_stop" not in src:
        f.append("1. depth families missing")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "while len(cands) > 1:\n" not in src:
        f.append("2. no depth loop")
    if 'reps["D4"]["mean_hops"] == 4.0' not in src or 'reps["STOP"]["refuse"] == 1.0' not in src:
        f.append("3. GATE missing D4 hops/STOP")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no D4",
     'FAM = {"D1": world_d1, "D2": world_d2, "D4": world_d4, "STOP": world_stop}',
     'FAM = {"D1": world_d1, "D2": world_d2, "STOP": world_stop}',
     "1."),
    ("fixed 2 hops",
     "    while len(cands) > 1:",
     "    while len(cands) > 1 and hops < 2:",
     "2."),
    ("gate drops STOP",
     '            and (reps["STOP"]["refuse"] == 1.0) and (reps["STOP"]["pin"] == 0.0)',
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
