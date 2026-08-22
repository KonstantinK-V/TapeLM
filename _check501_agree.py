"""Check of 501: three thresholds, same window. Gate only PIN436 unique_next > 0.05."""
from __future__ import annotations

from pathlib import Path

import _audit501_agree as M

SRC = Path("_audit501_agree.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def pick_corpus(" not in src or 'return WIKI, "wiki", 80' not in src:
        f.append("1. wiki auto-pick missing")
    if "def arm_436(" not in src or "def arm_soft(" not in src:
        f.append("1. three arms missing")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'pin["rate"] > 0.05' not in gate:
        f.append("2. GATE must be PIN436 unique_next > 0.05")
    if "soft[" in gate or "strict[" in gate:
        f.append("2. soft/strict must not gate")
    if "import torch" in src:
        f.append("3. Phi leaked")
    if "SOFT/STRICT are counters" not in src:
        f.append("3. counter diag missing")
    return f


MUTANTS = (
    ("gate strict",
     '    gate = (not void) and (pin["rate"] > 0.05)',
     '    gate = (not void) and (strict["rate"] > 0.05)',
     "2."),
    ("gate soft",
     '    gate = (not void) and (pin["rate"] > 0.05)',
     '    gate = (not void) and (soft["rate"] > 0.05)',
     "2."),
    ("no wiki",
     '        return WIKI, "wiki", 80',
     '        return FALLBACK, "tinystories-fallback", 20',
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
