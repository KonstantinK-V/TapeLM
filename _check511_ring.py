"""Check of 511: ring2 from budget remainder. GATE mid d2 and new meets > high."""
from __future__ import annotations

from pathlib import Path

import _audit511_ring as M

SRC = Path("_audit511_ring.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "200 / max(g[\"df\"][v]" not in src:
        f.append("1. allow=200/df(v) missing")
    if "remain = allow - len(r1)" not in src or "r2" not in src:
        f.append("1. second ring from remainder missing")
    if "m12 - m1" not in src:
        f.append("1. new meets m2 missing")
    if "g[\"df\"][c] <= HIGH_DF" not in src:
        f.append("1. skip df>80 missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "dd > 0" not in gate:
        f.append("2. GATE missing d2 mid > high")
    if "dm > 0.05" not in gate:
        f.append("2. GATE missing new meets delta > 0.05")
    if "d1" in gate or "m1" in gate:
        f.append("2. ring1 must not gate alone")
    if 'mid_rep["n"] < 20' not in src or 'high_rep["n"] < 5' not in src:
        f.append("2. VOID missing")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate d2 only",
     "    gate = (not void) and (dd > 0) and (dm > 0.05)",
     "    gate = (not void) and (dd > 0)",
     "2."),
    ("no ring2",
     "    remain = allow - len(r1)\n    r2 = []",
     "    remain = 0\n    r2 = []",
     "1."),
    ("flat allow",
     "    allow = max(1, int(200 / max(g[\"df\"][v], 1)))",
     "    allow = 50",
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
