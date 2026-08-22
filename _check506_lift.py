"""Check of 506: recurrent companions with lift/pmi. GATE mid pmi > high + 0.10."""
from __future__ import annotations

from pathlib import Path

import _audit506_lift as M

SRC = Path("_audit506_lift.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def star_pmi(" not in src or "n >= 2" not in src:
        f.append("1. repeat>=2 star_pmi missing")
    if "8 <= d <= 30" not in src or "d > 80" not in src:
        f.append("1. mid/high df bins missing")
    if "maj_of" in src or "pick_by_q" in src or "def train(" in src:
        f.append("1. majority/Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "delta > 0.10" not in gate:
        f.append("2. GATE missing delta pmi > 0.10")
    if "mid_rep[\"lift\"]" in gate or "high_rep[\"lift\"]" in gate:
        f.append("2. lift must not gate alone")
    if 'mid_rep["n"] < 15' not in src or 'high_rep["n"] < 15' not in src:
        f.append("2. VOID mid/high n < 15 missing")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate lift",
     "    gate = (not void) and (delta > 0.10)",
     "    gate = (not void) and (mid_rep[\"lift\"] > high_rep[\"lift\"] + 0.10)",
     "2."),
    ("no delta floor",
     "    gate = (not void) and (delta > 0.10)",
     "    gate = (not void) and (delta > 0.0)",
     "2."),
    ("majority hop",
     "    rec = [c for c, n in cnt.items() if n >= 2 and c != v]",
     "    rec = [Counter(bags[0]).most_common(1)[0][0]] if bags else []",
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
