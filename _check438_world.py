"""Check of 438: 436 think, designed P const / Q mixed.

  1. tape is designed(), not wiki.
  2. think is 436.measure.
  3. GATE const_hit==1 and refuse_mixed==1 and hop2==1.
  4. no torch, no pick.

    python _check438_world.py
"""
from __future__ import annotations

from pathlib import Path

import _audit438_world as M

SRC = Path("_audit438_world.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "lines = designed()" not in src:
        f.append("1. designed world missing")
    if "wikitext" in src:
        f.append("1. wiki leaked into 438")
    if "rep = M436.measure" not in src:
        f.append("2. 436.measure is not the think")
    if 'rep["const_hit"] == 1.0' not in src or 'rep["refuse_mixed"] == 1.0' not in src:
        f.append("3. GATE is not exact pin/refuse")
    if "PickNet" in src or "import torch" in src:
        f.append("4. scorer leaked")
    return f


MUTANTS = (
    ("wiki instead of designed",
     "    lines = designed()",
     '    lines = ["wiki line that should not be here"]',
     "1."),
    ("own think",
     "    rep = M436.measure(lines, ns, rng)",
     "    rep = None",
     "2."),
    ("gate coin",
     '    gate = ((not void) and (rep["const_hit"] == 1.0)',
     "    gate = ((not void) and True",
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
