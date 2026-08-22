"""Check of 508: enumerate all rec hops vs df-matched bag. GATE mid delta only."""
from __future__ import annotations

from pathlib import Path

import _audit508_enum as M

SRC = Path("_audit508_enum.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def rec_of(" not in src or "any(c in held for c in rec)" not in src:
        f.append("1. enumerate-all-rec missing")
    if "rng.sample(vocab" not in src:
        f.append("1. df-matched bag control missing")
    if "rec[0]" in src or "rec_lifts" in src:
        f.append("1. max-lift argmax leaked")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'mid_rep["delta"] > 0.05' not in gate:
        f.append("2. GATE missing mid delta > 0.05")
    if "mid_only" in gate or "stop_and" in gate:
        f.append("2. mid-only/stop must not gate")
    if 'mid_rep["n"] < 50' not in src:
        f.append("2. VOID mid n < 50 missing")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate mid_only",
     "    gate = (not void) and (mid_rep[\"delta\"] > 0.05)",
     "    gate = (not void) and (mid_rep[\"mid_only\"] > 0.05)",
     "2."),
    ("max-lift step",
     "            n_e += int(any(c in held for c in rec))",
     "            n_e += int(rec[0] in held) if rec else 0",
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
