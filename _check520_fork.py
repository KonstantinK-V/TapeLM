"""Check of 520: W on forks where score splits rest. GATE mid d511 and drnd > 0.05."""
from __future__ import annotations

from pathlib import Path

import _audit520_fork as M

SRC = Path("_audit520_fork.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit517_window import comps, pick_corpus, score_w" not in src:
        f.append("1. 517 W scoring reuse missing")
    if "from _audit518_reldf import pct_band" not in src:
        f.append("1. 518 relative mid bands missing")
    if "if max(sc) <= min(sc):" not in src:
        f.append("1. fork filter (W must split rest) missing")
    if "W = [h1]" not in src:
        f.append("1. working window W from hop1 missing")
    if "pick_w = sorted(rest, key=lambda c: (-score_w" not in src:
        f.append("1. pick_w must rank by score_w")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'mid_rep["d511"] > 0.05' not in gate:
        f.append("2. GATE missing d511 > 0.05")
    if 'mid_rep["drnd"] > 0.05' not in gate:
        f.append("2. GATE missing drnd > 0.05")
    if "hit_w" in gate or "hit_511" in gate:
        f.append("2. raw hits must not gate alone")
    if 'mid_rep["n"] < 40' not in src:
        f.append("2. VOID forks n < 40 missing")
    if "pick_corpus" not in src:
        f.append("3. corpus pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate d511 only",
     '    gate = (not void) and (mid_rep["d511"] > 0.05) and (mid_rep["drnd"] > 0.05)',
     '    gate = (not void) and (mid_rep["d511"] > 0.05)',
     "2."),
    ("no fork filter",
     "            if max(sc) <= min(sc):\n                continue",
     "            if False:\n                continue",
     "1."),
    ("no W rank",
     "            pick_w = sorted(rest, key=lambda c: (-score_w(g, by, c, W, cache), g[\"df\"][c]))[0]",
     "            pick_w = rest[0]",
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
