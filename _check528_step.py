"""Check of 528: per-step novelty on frozen v1. Default 2400. GATE cover>hop1, hops<allgo, high glue."""
from __future__ import annotations

from pathlib import Path

import _audit528_step as M

SRC = Path("_audit528_step.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit527_learn import majority, v1_nodes" not in src:
        f.append("1. 527 v1_nodes reuse missing")
    if "from _audit518_reldf import pct_band" not in src:
        f.append("1. 518 pct_band missing")
    if "default=2400" not in src:
        f.append("1. default window 2400 missing")
    if "C_STEP = 0.02" not in src and "C_STEP = 0.05" not in src:
        f.append("1. step cost missing")
    if "c in held and c not in seen and c != maj" not in src:
        f.append("1. novelty = new held and not maj missing")
    if "def always(" not in src or "allgo_mid" not in src:
        f.append("1. always-go control missing")
    if "unique_next" in src:
        f.append("1. unique reward leaked")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'lm["cover"] > a1["cover"] + 0.05' not in gate:
        f.append("2. GATE missing cover > hop1 + 0.05")
    if 'lm["hops"] < ag["hops"] - 0.5' not in gate:
        f.append("2. GATE missing hops < allgo - 0.5")
    if 'lh["hops"] < 1.5' not in gate:
        f.append("2. GATE missing high hops < 1.5")
    if "ceiling = ag[\"cover\"] - a1[\"cover\"]" not in src:
        f.append("2. ceiling ALLGO-HOP1 missing")
    if "thin_ceil = ceiling <= 0.05" not in src:
        f.append("2. VOID on thin ceiling missing")
    if 'f"{args.seed}_{tag}_{len(lines)}"' not in src:
        f.append("3. JSON key must split seed/corpus/window")
    return f


MUTANTS = (
    ("gate cover only",
     '    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05) and (\n'
     '        lm["hops"] < ag["hops"] - 0.5) and (lh["hops"] < 1.5)',
     '    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05)',
     "2."),
    ("held-rec no novelty",
     "                new = c in held and c not in seen and c != maj",
     "                new = c in held",
     "1."),
    ("no thin ceil void",
     "    thin_ceil = ceiling <= 0.05",
     "    thin_ceil = False",
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
