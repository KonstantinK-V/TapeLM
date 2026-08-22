"""Check of 462 TRACK R: no n_hit in sig."""
from __future__ import annotations

from pathlib import Path

import _audit462_noval as M

SRC = Path("_audit462_noval.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "return (d1, s)" not in src:
        f.append("1. pair sig missing")
    if "n_hit = sum" in src or "n_hit = 0" in src:
        f.append("1. n_hit leaked")
    if "wikitext" in src or "MIX =" in src:
        f.append("1. D-track leaked")
    if 'no["BOTH"]["mean_hops"] > 2.0' not in src:
        f.append("3. GATE missing ablation")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("n_hit back",
     "    return (d1, s)",
     "    n_hit = 0\n    return (n_hit, d1, s)",
     "1."),
    ("gate drops ablation",
     '    gate = (not void) and match_455(yes) and (no["BOTH"]["mean_hops"] > 2.0)',
     "    gate = (not void) and match_455(yes)",
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
