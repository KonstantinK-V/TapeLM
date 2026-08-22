"""Check of 454: BOTH prefers 2-hop unique over 4-hop."""
from __future__ import annotations

from pathlib import Path

import _audit454_cost as M

SRC = Path("_audit454_cost.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if '"BOTH": world_both' not in src or "def soon" not in src:
        f.append("1. BOTH/soon missing")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "(x[0], x[1], x[2])" not in src:
        f.append("2. cost pick missing soon-key")
    if 'reps["BOTH"]["mean_hops"] == 2.0' not in src:
        f.append("3. GATE is not BOTH hops==2")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no BOTH",
     'FAM = {"D1": world_d1, "BOTH": world_both, "D4": world_d4, "STOP": world_stop}',
     'FAM = {"D1": world_d1, "D4": world_d4, "STOP": world_stop}',
     "1."),
    ("no soon sort",
     "    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)",
     "    scored.sort(key=lambda x: (x[0], x[2]), reverse=True)",
     "2."),
    ("gate drops BOTH hops",
     '            and (reps["BOTH"]["pin"] == 1.0) and (reps["BOTH"]["mean_hops"] == 2.0)',
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
