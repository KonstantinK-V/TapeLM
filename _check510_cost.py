"""Check of 510: 1/df walk budget + meet bonus. GATE mid deeper & more meets than high."""
from __future__ import annotations

from pathlib import Path

import _audit510_cost as M

SRC = Path("_audit510_cost.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "200 / max(g[\"df\"][v]" not in src:
        f.append("1. allow=200/df(v) missing")
    if "src / dfc" not in src or "src / max(g[\"df\"][m]" not in src:
        f.append("1. 1/df step pay and meet bonus missing")
    if "dfc > HIGH_DF" not in src:
        f.append("1. skip df>80 companion missing")
    if "rec.sort(key=lambda cn: g[\"df\"]" not in src:
        f.append("1. rare-first expansion missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "d_meet > 0.05" not in gate:
        f.append("2. GATE missing d_meet > 0.05")
    if "d_depth > 0" not in gate:
        f.append("2. GATE missing depth_mid > depth_high")
    if "score" in gate or "pay" in gate:
        f.append("2. score/pay must not gate")
    if 'mid_rep["n"] < 20' not in src or 'high_rep["n"] < 5' not in src:
        f.append("2. VOID missing")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate depth only",
     "    gate = (not void) and (d_meet > 0.05) and (d_depth > 0)",
     "    gate = (not void) and (d_depth > 0)",
     "2."),
    ("no df skip",
     "        if dfc > HIGH_DF:\n            continue",
     "        if False:\n            continue",
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
