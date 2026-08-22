"""Check of 527: frozen v1 legs + go/stop learner. Teacher beats majority, not 521 held-rec."""
from __future__ import annotations

from pathlib import Path

import _audit527_learn as M

SRC = Path("_audit527_learn.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit511_ring import cheap_rec, graph, mentions, pick_corpus" not in src:
        f.append("1. 511 reuse missing")
    if "from _audit517_window import comps" not in src:
        f.append("1. 517 comps missing")
    if "from _audit518_reldf import pct_band" not in src:
        f.append("1. 518 pct_band missing")
    if "def v1_nodes(" not in src or "def allow_of(" not in src:
        f.append("1. frozen v1 walk missing")
    if "if v in high_set:\n        return 1" not in src:
        f.append("1. high allow=1 missing")
    if "def majority(" not in src:
        f.append("1. majority rival missing")
    if "nodes[0] in held and nodes[0] != maj" not in src:
        f.append("1. teacher must require held and not majority")
    if "c in held and c != maj" not in src:
        f.append("1. per-hop non-majority teacher missing")
    if "unique_next" in src:
        f.append("1. unique reward leaked")
    if "Q[H]" in src or "pick_by_q" in src:
        f.append("1. Q[H] / place-pick leaked")
    if "import torch" in src or "PickNet" in src or "CrossEntropy" in src:
        f.append("4. CE/Phi leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'lm["hops"] > lh["hops"] + 0.5' not in gate:
        f.append("2. GATE missing mid hops > high + 0.5")
    if 'lh["hops"] < 1.5' not in gate:
        f.append("2. GATE missing high hops < 1.5")
    if 'lm["spec"] > a1["spec"] + 0.05' not in gate:
        f.append("2. GATE missing spec > hop1-only + 0.05")
    if 'lm["n"] < 20' not in src:
        f.append("2. VOID missing")
    if "pick_corpus" not in src:
        f.append("3. corpus pick missing")
    return f


MUTANTS = (
    ("gate hops only",
     '    gate = (not void) and (lm["hops"] > lh["hops"] + 0.5) and (lh["hops"] < 1.5) and (\n'
     '        lm["spec"] > a1["spec"] + 0.05)',
     '    gate = (not void) and (lm["hops"] > lh["hops"] + 0.5) and (lh["hops"] < 1.5)',
     "2."),
    ("held-rec only",
     "            r1 = (1.0 if nodes[0] in held and nodes[0] != maj else 0.0) - 0.05  # not majority",
     "            r1 = (1.0 if nodes[0] in held else 0.0) - 0.05  # not majority",
     "1."),
    ("no high cap",
     "    if v in high_set:\n        return 1",
     "    if False:\n        return 1",
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
