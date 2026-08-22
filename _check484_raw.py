"""Check of 484: raw corpus pick; gate on 436 pin; unique_next not gated."""
from __future__ import annotations

from pathlib import Path

import _audit484_raw as M

SRC = Path("_audit484_raw.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def pick_corpus(" not in src or "tinystories-fallback" not in src:
        f.append("1. corpus pick/fallback missing")
    if "M436.measure" not in src:
        f.append("1. 436 measure missing")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "unique_next" in gate:
        f.append("2. unique_next gated")
    if 'pin["const_hit"] > 0.90' not in src or 'hop2_sees_pin' not in src:
        f.append("2. GATE missing 436 hold")
    if "import torch" in src:
        f.append("3. Phi leaked")
    return f


MUTANTS = (
    ("gate unique",
     '            and (pin["hop2_sees_pin"] == 1.0))',
     '            and (pin["hop2_sees_pin"] == 1.0)\n'
     '            and (comp["unique_next"] > 0.1))',
     "2."),
    ("no fallback",
     '        return FALLBACK, "tinystories-fallback", 20',
     '        raise SystemExit("no tinystories")',
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
