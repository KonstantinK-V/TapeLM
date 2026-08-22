"""Check of 531: independent hop layers. Hop k does not train hop k-1. Default W=250."""
from __future__ import annotations

from pathlib import Path

import _audit531_layer as M

SRC = Path("_audit531_layer.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit527_learn import majority, v1_nodes" not in src:
        f.append("1. 527 v1_nodes reuse missing")
    if "from _audit528_step import cover, mean_ep, run_ep, trials" not in src:
        f.append("1. 528 step helpers missing")
    if "from _audit518_reldf import pct_band" not in src:
        f.append("1. 518 pct_band missing")
    if "default=250" not in src:
        f.append("1. default window 250 missing")
    if "def layer_of(" not in src:
        f.append("1. layer_of missing")
    if "# hop1 never updates Q2/Q3\n            for i, c in enumerate(nodes[1:], start=1):" not in src:
        f.append("1. hop1 must not train Q2/Q3; training starts at hop2")
    if "residual = c in held and c not in seen and c != maj" not in src:
        f.append("1. residual = novel held beyond prior hops missing")
    if "Q[(band, Lyr)]" not in src:
        f.append("1. per-layer Q[(band, Lyr)] missing")
    if "end-of-cycle memory" not in src:
        f.append("1. end-of-cycle memory comment missing")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'lm["cover"] > a1["cover"] + 0.05' not in gate:
        f.append("2. GATE missing cover > hop1 + 0.05")
    if "mid_between" not in src or 'a1["hops"] + 0.3 < lm["hops"] < ag["hops"] - 0.3' not in src:
        f.append("2. GATE missing hops between hop1 and allgo")
    if 'lh["hops"] < 1.5' not in gate:
        f.append("2. GATE missing high hops < 1.5")
    return f


MUTANTS = (
    ("gate cover only",
     '    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05) and mid_between and (\n'
     '        lh["hops"] < 1.5)',
     '    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05)',
     "2."),
    ("train from hop1",
     "# hop1 never updates Q2/Q3\n            for i, c in enumerate(nodes[1:], start=1):",
     "# hop1 never updates Q2/Q3\n            for i, c in enumerate(nodes, start=0):",
     "1."),
    ("no residual",
     "                residual = c in held and c not in seen and c != maj",
     "                residual = c in held",
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
