"""Check of 480: three kinds L/G/C; combo skips unmarked; gate on B."""
from __future__ import annotations

from pathlib import Path

import _audit480_planes as M

SRC = Path("_audit480_planes.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if '"DEAD"' not in src or "sorted(unmarked)" not in src:
        f.append("1. DEAD/unmarked missing")
    if 'kind == "L"' not in src or 'kind == "G"' not in src or '"C"' not in src:
        f.append("1. three kinds missing")
    if 'comB["p_h2"] >= 0.85' not in src or 'locB["p_h2"]' not in src:
        f.append("2. GATE missing B combo vs singles")
    if "import torch" in src or "wikitext" in src:
        f.append("3. leak")
    return f


MUTANTS = (
    ("no unmarked skip",
     "            H = rng.choice(sorted(unmarked))",
     "            H = rng.choice(pool)",
     "1."),
    ("gate drop B",
     '            and (comB["p_h2"] >= 0.85)',
     "            and True",
     "2."),
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
