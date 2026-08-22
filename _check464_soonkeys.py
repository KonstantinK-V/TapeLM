"""Check of 464: soon_keys excludes hole."""
from __future__ import annotations

from pathlib import Path

import _audit464_soonkeys as M

SRC = Path("_audit464_soonkeys.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def soon_keys(" not in src or "if k == hole:" not in src:
        f.append("1. soon_keys missing")
    if "from _audit454_cost import soon" in src:
        f.append("1. 454 soon leaked")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "no[\"BOTH\"][\"mean_hops\"] > 2.0" not in src:
        f.append("3. GATE missing ablation")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("include hole",
     "        if k == hole:\n            continue",
     "        if False:\n            continue",
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
