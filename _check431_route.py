"""Check of 431: structural route, target not in the graph."""
from __future__ import annotations

from pathlib import Path

import _audit431_route as M

SRC = Path("_audit431_route.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "T = {place[s] for s in eA + eB} - {pa, pb}" not in src:
        f.append("1. start places are left in the target")
    if "    def link(u, v):" in src:
        linkb = src.split("    def link(u, v):", 1)[1].split("    qs = []", 1)[0]
        if "by_val" in linkb or "value[" in linkb:
            f.append("2. link() sees values")
    if "len(fu & frames[v]) >= 2" not in src:
        f.append("2. FRAME edge is not |∩|≥2")
    if "order = sorted(ss, key=lambda s: pos[s])" not in src:
        f.append("2. NEIGH mixes lines")
    if 'void = (rep["n_foreign"] < 30) or (rep["reach_any_target"] <= 0.05)' not in src:
        f.append("3. VOID is not n_foreign / reach")
    if 'gate = (not void) and (rep["delta_vs_random"] < -DELTA)' not in src:
        f.append("3. GATE is not Δ vs random < −0.5")
    if "H425.evidence" not in src:
        f.append("4. evidence is not 425's")
    if "import torch" in src:
        f.append("5. Phi/policy is in this file")
    if "class Policy" in src:
        f.append("5. a stepper leaked into the probe")
    return f


MUTANTS = (
    ("start stays in target",
     "        T = {place[s] for s in eA + eB} - {pa, pb}",
     "        T = {place[s] for s in eA + eB}",
     "1."),
    ("FRAME is any overlap",
     "            if len(fu & frames[v]) >= 2:",
     "            if len(fu & frames[v]) >= 0:",
     "2."),
    ("VOID skips empty foreign",
     '    void = (rep["n_foreign"] < 30) or (rep["reach_any_target"] <= 0.05)',
     '    void = False',
     "3."),
    ("GATE is coin",
     '    gate = (not void) and (rep["delta_vs_random"] < -DELTA)',
     '    gate = (not void) and (rep["delta_vs_random"] < 9.0)',
     "3."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
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
