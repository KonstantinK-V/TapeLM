"""Check of 448: pick the shrinking seek among ≥2."""
from __future__ import annotations

from pathlib import Path

import _audit448_pick as M

SRC = Path("_audit448_pick.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit447_seek import" not in src:
        f.append("1. 447 worlds missing")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "def all_seeks" not in src or "score < len(cands)" not in src:
        f.append("2. shrink-score pick missing")
    if 'pos["mean_seek"] == 1.0' not in src or 'pos["mean_choice"] < 2.0' not in src:
        f.append("3. GATE is not 1-seek among ≥2")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no worlds",
     "from _audit447_seek import (",
     "from _audit445_xor import (",
     "1."),
    ("take first seek",
     "        if score < len(cands):\n            best.append((score, k, H, pinH, newc))",
     "        best.append((score, k, H, pinH, newc))",
     "2."),
    ("gate allows 447 order",
     '            and (pos["mean_seek"] == 1.0) and (pos["crisp"] == 1.0) and (pos["ripe"] == 0.0)',
     "            and True",
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
