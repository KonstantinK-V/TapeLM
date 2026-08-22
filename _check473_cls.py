"""Check of 473: cls=(shrunk, opened); no n_after; QH_B==0."""
from __future__ import annotations

from pathlib import Path

import _audit473_cls as M

SRC = Path("_audit473_cls.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "return (shrunk, op)" not in src:
        f.append("1. cls tuple missing")
    if "def n_after" in src:
        f.append("1. n_after leaked")
    if 'hB["pin"] == 0.0' not in src:
        f.append("2. GATE missing QH_B")
    if "from _audit471" in src or "from _audit470" in src:
        f.append("1. 470/471 leaked")
    if "import torch" in src or "wikitext" in src:
        f.append("3. leak")
    return f


MUTANTS = (
    ("drop cls",
     "    return (shrunk, op)",
     "    return shrunk",
     "1."),
    ("drop QH_B",
     '            and (hB["pin"] == 0.0)',
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
