"""Check of 455: five families, shared eval_fam, no if family:."""
from __future__ import annotations

from pathlib import Path

import _audit455_dash as M

SRC = Path("_audit455_dash.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if '"D2": world_d2' not in src or '"BOTH": world_both' not in src or '"STOP": world_stop' not in src:
        f.append("1. mixed FAM missing")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "if family:" in src or "if kind ==" in src:
        f.append("2. per-family branch")
    if "eval_fam(" not in src:
        f.append("2. not using shared eval_fam")
    if 'reps["D2"]["mean_hops"] == 2.0' not in src or 'extra["BOTH"] > 0' not in src:
        f.append("3. GATE missing D2/extra")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("drop D2",
     '    "D2": world_d2,\n',
     "    # D2 dropped\n",
     "1."),
    ("family if",
     "    reps = {k: eval_fam(rng, fn, args.test, rate) for k, fn in FAM.items()}",
     "    reps = {k: eval_fam(rng, fn, args.test, rate) for k, fn in FAM.items()}\n    if family:\n        pass",
     "2."),
    ("gate drops extra",
     '            and (extra["BOTH"] > 0) and (extra["D4"] > 0))',
     "            and True)",
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
