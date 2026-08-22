"""Check of 507: top-lift rec walk vs random rec. GATE mid delta only."""
from __future__ import annotations

from pathlib import Path

import _audit507_walk as M

SRC = Path("_audit507_walk.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def rec_lifts(" not in src or "rec[0][0]" not in src:
        f.append("1. top-lift step missing")
    if "rng.choice(pool)" not in src:
        f.append("1. random-from-rec control missing")
    if "len(cands) != 1" in src or "unique" in src.lower() and "not required" not in src:
        if "unique_next" in src or "len(cands) == 1" in src:
            f.append("1. unique hop leaked")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'mid_rep["delta"] > 0.05' not in gate:
        f.append("2. GATE missing mid delta > 0.05")
    if "high_rep" in gate:
        f.append("2. high must not gate")
    if 'mid_rep["n"] < 50' not in src:
        f.append("2. VOID mid n < 50 missing")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate high",
     "    gate = (not void) and (mid_rep[\"delta\"] > 0.05)",
     "    gate = (not void) and (high_rep[\"delta\"] > 0.05)",
     "2."),
    ("no random pool",
     "            pool = [c for c, _ in rec]\n            n_r += int(rng.choice(pool) in held)",
     "            n_r += int(rec[-1][0] in held)",
     "1."),
    ("no delta floor",
     "    gate = (not void) and (mid_rep[\"delta\"] > 0.05)",
     "    gate = (not void) and (mid_rep[\"delta\"] > 0.0)",
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
