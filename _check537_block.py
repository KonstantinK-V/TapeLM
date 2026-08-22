"""Check of 537b: HAVE vs REC-miss block diagnosis."""
from __future__ import annotations

from pathlib import Path

import _audit537_block as M

SRC = Path("_audit537_block.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "HAVE" not in src or "recmiss" not in src:
        f.append("1. HAVE/REC-miss counters missing")
    if "if c in have_set:" not in src:
        f.append("1. HAVE = mark on 511 path")
    if "elif c not in rec_set:" not in src:
        f.append("1. REC-miss = mark not in cheap_rec")
    if "else:\n                    would += 1" not in src:
        f.append("1. ADD = mark in rec but not on 511")
    if "n_seen < 40" not in src:
        f.append("2. VOID on n_seen < 40")
    if 'rec["p_have"] > rec["p_rec"]' not in src:
        f.append("2. HAVE blocks branch missing")
    if 'rec["p_rec"] > rec["p_have"]' not in src:
        f.append("2. REC blocks branch missing")
    return f


MUTANTS = (
    ("swap have rec",
     "                if c in have_set:\n                    have += 1\n"
     "                elif c not in rec_set:\n                    recmiss += 1",
     "                if c not in rec_set:\n                    recmiss += 1\n"
     "                elif c in have_set:\n                    have += 1",
     "1."),
    ("no void",
     "    void = n_seen < 40",
     "    void = False",
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
