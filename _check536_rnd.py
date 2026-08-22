"""Check of 536rnd: jitter shuffle cheap_rec[:k] vs teacher marks."""
from __future__ import annotations

from pathlib import Path

import _audit536_rnd as M

SRC = Path("_audit536_rnd.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def rnd_marks(" not in src:
        f.append("1. rnd_marks helper missing")
    if "k = len(teacher_marks)" not in src:
        f.append("1. k from teacher mark count missing")
    if "rng.shuffle(rec)" not in src:
        f.append("1. shuffle cheap_rec missing")
    if "teacher_mean_d1" not in src:
        f.append("1. teacher vs rnd comparison missing")
    if "from _audit534_mark import offer" not in src:
        f.append("1. same offer() as 534/536")
    if "RND ≈ TEACHER" not in src:
        f.append("2. jitter verdict missing")
    if "RND ≪ TEACHER" not in src:
        f.append("2. real teacher verdict missing")
    return f


MUTANTS = (
    ("no shuffle",
     "    rng.shuffle(rec)\n    return rec[:k]",
     "    return rec[:k]",
     "1."),
    ("fixed k",
     "    k = len(teacher_marks)",
     "    k = 3",
     "1."),
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
