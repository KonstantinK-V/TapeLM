"""Check of 466 TRACK D: no teacher, control gate, BOTH not in fit."""
from __future__ import annotations

from pathlib import Path

import _audit466_d463 as M

SRC = Path("_audit466_d463.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "teacher_" in src or "teacher_trap" in src:
        f.append("1. teacher leaked")
    if "soon_keys" in src or "from _audit464" in src or "from _audit465" in src:
        f.append("1. soon_keys/465 leaked")
    if "def fit(" not in src:
        f.append("2. fit missing")
    fit_src = src[src.find("def fit("):src.find("def main(")]
    if '"BOTH"' in fit_src or "'BOTH'" in fit_src:
        f.append("2. BOTH in fit")
    if 't1c["D2"]["pin"] == 0.0' not in src:
        f.append("3. GATE missing control")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("teacher leak",
     "from _audit463_trap import trap_of",
     "from _audit463_trap import trap_of, teacher_trap",
     "1."),
    ("BOTH in fit",
     '            and (yes["STOP"]["refuse"] == 1.0) and (yes["STOP"]["pin"] == 0.0))',
     '            and (yes["BOTH"]["pin"] == 1.0)\n'
     '            and (yes["STOP"]["refuse"] == 1.0) and (yes["STOP"]["pin"] == 0.0))',
     "2."),
    ("gate drops control",
     '            and (t1c["D2"]["pin"] == 0.0))',
     "            )",
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
