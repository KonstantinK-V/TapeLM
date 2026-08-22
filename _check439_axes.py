"""Check of 439: both axes, 436 think only."""
from __future__ import annotations

from pathlib import Path

import _audit439_axes as M

SRC = Path("_audit439_axes.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "M438.designed" not in src:
        f.append("1. designed axis missing")
    if "wikitext103" not in src and "args.corpus" not in src:
        f.append("1. wiki axis missing")
    if src.count("M436.measure") < 2:
        f.append("2. 436.measure not on both axes")
    if 'rep["const_hit"] == 1.0' not in src:
        f.append("3. GATE-A is not 438")
    if "const_live" not in src or "refuse_mixed" not in src:
        f.append("3. GATE-B missing")
    if "PickNet" in src or "import torch" in src:
        f.append("4. scorer leaked")
    if "composition" in src.lower() and "def composed" in src:
        f.append("4. composition world leaked")
    return f


MUTANTS = (
    ("no designed",
     "    rep = M436.measure(M438.designed(), _ns(args), rng)",
     "    rep = M436.measure([], _ns(args), rng)",
     "1."),
    ("own think on A",
     "    rep = M436.measure(M438.designed(), _ns(args), rng)",
     "    rep = None",
     "2."),
    ("gate-A coin",
     '    gate = ((not void) and (rep["const_hit"] == 1.0)',
     "    gate = ((not void) and True",
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
