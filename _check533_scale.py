"""Check of 533: hop1 local W=250, hop2+ on pooled train tape. Gate = 531."""
from __future__ import annotations

from pathlib import Path

import _audit533_scale as M

SRC = Path("_audit533_scale.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit531_layer import layer_of" not in src:
        f.append("1. 531 layer_of reuse missing")
    if "from _audit528_step import cover, trials" not in src:
        f.append("1. 528 trials/cover missing")
    if "default=250" not in src:
        f.append("1. default window-lines 250 missing")
    if "default=16" not in src:
        f.append("1. default windows 16 missing")
    if "big_lines = [ln for s in tr_s for ln in s]" not in src:
        f.append("1. pooled train tape (big) from train slices only")
    if "local_nodes = path_of(loc, v, cache_l)" not in src:
        f.append("1. hop1 from local window missing")
    if "pool_nodes = [c for c in path_of(big, v, cache_b) if c != hop1]" not in src:
        f.append("1. hop2+ from pool tape missing")
    if "te_s = slices[n_tr:]" not in src and "tr_s, te_s = slices[:n_tr], slices[n_tr:]" not in src:
        f.append("1. 70/30 train/test slice split missing")
    if "collect(locals_te" not in src:
        f.append("1. test on locals_te only (pool excludes test windows)")
    if "# hop1 never updates Q2/Q3\n                for i, c in enumerate(nodes[1:], start=1):" not in src:
        f.append("1. hop1 must not train Q2/Q3; training starts at hop2")
    if "residual = c in held and c not in seen and c != maj" not in src:
        f.append("1. residual = local held novelty missing")
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
    ("hop2 from local not pool",
     "pool_nodes = [c for c in path_of(big, v, cache_b) if c != hop1]",
     "pool_nodes = [c for c in local_nodes[1:] if c != hop1]",
     "1."),
    ("train from hop1",
     "# hop1 never updates Q2/Q3\n                for i, c in enumerate(nodes[1:], start=1):",
     "# hop1 never updates Q2/Q3\n                for i, c in enumerate(nodes, start=0):",
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
