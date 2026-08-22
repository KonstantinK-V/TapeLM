"""Check of 440: compose two 436 ops. Designed only."""
from __future__ import annotations

from pathlib import Path

import _audit440_compose as M

SRC = Path("_audit440_compose.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def designed" not in src or "SWEET APPLES" not in src:
        f.append("1. composition world missing")
    if "measure(designed(), rng)" not in src:
        f.append("1. designed tape not used")
    if "by_key.get(got, set())" not in src:
        f.append("2. v is not the next address")
    if 'rep["composed_sweet"] == 1.0' not in src or 'rep["ctrl_apples"] == 1.0' not in src:
        f.append("3. GATE is not SWEET vs APPLES")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("wiki",
     "    rep = measure(designed(), rng)",
     '    rep = measure(["wiki should not be here"], rng)',
     "1."),
    ("no v-address",
     "        cands = by_key.get(got, set()) - {place[s]}",
     "        cands = {place[s]}",
     "2."),
    ("gate coin",
     '            and (rep["composed_sweet"] == 1.0)',
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
