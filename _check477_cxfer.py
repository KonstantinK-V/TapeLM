"""Check of 477 C: unique next_place; no table.get; gate A and B."""
from __future__ import annotations

from pathlib import Path

import _audit477_cxfer as M

SRC = Path("_audit477_cxfer.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def next_place(" not in src or "len(cands) != 1" not in src:
        f.append("1. unique next_place missing")
    if "table.get" in src or "def train(" in src:
        f.append("1. Q leaked")
    if "ok_tape(a) and ok_tape(b)" not in src:
        f.append("2. GATE missing both tapes")
    if "import torch" in src or "wikitext" in src:
        f.append("3. leak")
    return f


MUTANTS = (
    ("multi next",
     "    if len(cands) != 1:\n        return None\n    return next(iter(cands))",
     "    if len(cands) < 1:\n        return None\n    return next(iter(cands))",
     "1."),
    ("gate A only",
     "    gate = ((not void) and ok_tape(a) and ok_tape(b) and names_diff)",
     "    gate = ((not void) and ok_tape(a) and names_diff)",
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
