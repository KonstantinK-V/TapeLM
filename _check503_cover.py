"""Check of 503: coverage vs length. Frontier F0/Famb/Fland; GROW on R2."""
from __future__ import annotations

from pathlib import Path

import _audit503_cover as M

SRC = Path("_audit503_cover.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit440_compose import think_place" not in src:
        f.append("1. frozen 440 think_place missing")
    if "F0" not in src or "Famb" not in src or "Fland" not in src:
        f.append("1. frontier buckets missing")
    if "R2" not in src or "R3" not in src:
        f.append("1. reachability R2/R3 missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    if "grow = bool(r2s) and r2s[-1][1] > r2s[0][1] + 0.02" not in src:
        f.append("2. GROW gate missing R2 +0.02 vs shortest L")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "grow" not in gate or "void" not in gate:
        f.append("2. GATE must use void and grow")
    if "Fland" in gate or "Famb" in gate:
        f.append("2. frontier must not gate")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    if "Frontier is not counted as fail" not in src:
        f.append("3. frontier diag missing")
    return f


MUTANTS = (
    ("gate Fland",
     "    gate = (not void) and grow",
     "    gate = (not void) and (by.get('400', {}).get('Fland', 1) < 0.5)",
     "2."),
    ("no grow",
     "    grow = bool(r2s) and r2s[-1][1] > r2s[0][1] + 0.02",
     "    grow = True",
     "2."),
    ("Q leaked",
     "from _audit440_compose import think_place",
     "from _audit440_compose import think_place\nfrom _audit485_hunt import pick_by_q",
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
