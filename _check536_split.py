"""Check of 536: newnode/reorder/silent baskets, THIN, asserts."""
from __future__ import annotations

from pathlib import Path

import _audit536_split as M

SRC = Path("_audit536_split.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if 'STORIES = "data/_tinystories_train.txt"' not in src:
        f.append("1. STORIES default path missing")
    if "default=STORIES" not in src:
        f.append("1. --corpus default stories missing")
    if "from _audit534_mark import offer" not in src:
        f.append("1. 534 offer reuse missing")
    if "def empty_box(" not in src:
        f.append("1. empty_box helper missing")
    if 'boxes = dict(newnode=empty_box(), reorder=empty_box(), silent=empty_box())' not in src:
        f.append("1. three baskets missing")
    if "newnode = bool(sm - s511)" not in src:
        f.append("1. newnode vs reorder split missing")
    if 'assert d1_self == 0, "silent offer changed hop1"' not in src:
        f.append("1. silent basket assert missing")
    if "d1_self = (cover(nm[:1], held)" not in src:
        f.append("1. SELF hop1 delta via cover (534 match) missing")
    if "extra_n < 40" not in src:
        f.append("2. VOID on extra_n < 40 missing")
    if "n_new < 40 or n_re < 40" not in src:
        f.append("2. THIN basket < 40 missing")
    if 'assert not (n_mark > 0 and n_new == 0), "marks not reaching offer"' not in src:
        f.append("2. marks-not-reaching-offer assert missing")
    if "s_new > 0 and s_re < 0" not in src:
        f.append("2. NEWNODE+/REORDER- verdict missing")
    if "gate =" in src:
        f.append("2. 536 must not have a gate")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    return f


MUTANTS = (
    ("no baskets",
     'boxes = dict(newnode=empty_box(), reorder=empty_box(), silent=empty_box())',
     'boxes = dict(all=empty_box())',
     "1."),
    ("no thin",
     "        if n_new < 40 or n_re < 40:",
     "        if False:",
     "2."),
    ("no marks assert",
     '        assert not (n_mark > 0 and n_new == 0), "marks not reaching offer"',
     '        pass  # marks assert',
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
