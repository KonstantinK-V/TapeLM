"""Check of 437 length sweep: 436 think, no new intellect. Three mutants.

  1. Uses M436.measure; 436 file not rewritten.
  2. Lengths include 100,400,1600,4000; reports const_live / mixed spans.
  3. No Phi; GATE same as 436 (wiring, not smarter-than-directory).
"""
from __future__ import annotations

from pathlib import Path

import _audit437_len as M

SRC = Path("_audit437_len.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "M436.measure" not in src:
        f.append("1. does not call 436 measure")
    if "import _audit436_constpin" not in src:
        f.append("1. 436 not imported")
    if "LENGTHS = (100, 400, 1600, 4000)" not in src:
        f.append("2. lengths are not 100/400/1600/4000")
    if "const_live_span" not in src or "mixed_span" not in src:
        f.append("2. arena spans not reported")
    if "contract_holds" not in src:
        f.append("2. contract_holds missing")
    if "import torch" in src or "PickNet" in src:
        f.append("3. Phi leaked")
    if "No new intelligence" not in src and "no new" not in src.lower():
        f.append("3. file claims new intelligence")
    if "436 unchanged" not in src:
        f.append("3. 436 unchanged note missing")
    return f


MUTANTS = (
    ("reinvent measure",
     "        rep = M436.measure(lines, args, random.Random(args.seed + L))",
     "        rep = None  # skip 436",
     "1."),
    ("drop length 1600",
     "LENGTHS = (100, 400, 1600, 4000)",
     "LENGTHS = (100, 400, 4000)",
     "2."),
    ("Phi leaked",
     'OUT = Path("results/_stage437_len.json")',
     'OUT = Path("results/_stage437_len.json")\nimport torch',
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
