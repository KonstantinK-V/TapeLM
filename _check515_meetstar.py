"""Check of 515: MEET as new star center vs random ring2 node."""
from __future__ import annotations

from pathlib import Path

import _audit515_meetstar as M

SRC = Path("_audit515_meetstar.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit511_ring import cheap_rec, mentions, walk" not in src:
        f.append("1. 511 walk reuse missing")
    if "def rings(" not in src or "meet_nodes(" not in src:
        f.append("1. rings + meet_nodes missing")
    if ("        m = rng.choice(sorted(marks))\n"
        "        c = rng.choice(r2)\n"
        "        meets.append(m)") not in src:
        f.append("1. MEET center must come from marks, control from r2")
    if "if not marks or not r2:" not in src:
        f.append("1. require marks and ring2 leftover")
    if 'tape[m] = "MEET"' in src:
        f.append("1. 514 tape write must not gate 515")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'meet_rep["d2"] > r2_rep["d2"]' not in gate:
        f.append("2. GATE missing meet d2 > r2 d2")
    if 'meet_rep["m2"] - r2_rep["m2"] > 0.05' not in gate:
        f.append("2. GATE missing meet m2 delta > 0.05")
    if 'meet_rep["n"] < 20' not in src:
        f.append("2. VOID meet n < 20 missing")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate d2 only",
     "    gate = (not void) and (meet_rep[\"d2\"] > r2_rep[\"d2\"]) and (\n"
     "        meet_rep[\"m2\"] - r2_rep[\"m2\"] > 0.05)",
     "    gate = (not void) and (meet_rep[\"d2\"] > r2_rep[\"d2\"])",
     "2."),
    ("swap control",
     "        m = rng.choice(sorted(marks))\n        c = rng.choice(r2)",
     "        c = rng.choice(sorted(marks))\n        m = rng.choice(r2)",
     "1."),
    ("no ring2 req",
     "        if not marks or not r2:",
     "        if not marks:",
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
