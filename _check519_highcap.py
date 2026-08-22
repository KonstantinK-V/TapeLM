"""Check of 519: 518 relative mid + high allow=1. GATE mid d2 grows; high d2 < 1 by construction."""
from __future__ import annotations

from pathlib import Path

import _audit519_highcap as M

SRC = Path("_audit519_highcap.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit518_reldf import pct_band, walk_rel" not in src:
        f.append("1. 518 relative bands/budget reuse missing")
    if "from _audit511_ring import cheap_rec, graph, mentions, pick_corpus" not in src:
        f.append("1. 511 reuse missing")
    if "if v in high_set:" not in src or "allow = 1  # glue, 510" not in src:
        f.append("1. high allow=1 cap missing")
    if "return dict(d1=len(r1), d2=0, allow=allow)" not in src:
        f.append("1. high walk must not produce ring2")
    if "k = 200.0 / max(g400[\"n\"]" not in src:
        f.append("1. k calibrated from N=400 missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'm2400.get("d2", 0) > m100.get("d2", 0) + 1' not in gate:
        f.append("2. GATE missing mid d2 growth +1")
    if 'h2400.get("d2", 99) < 1.0' not in gate:
        f.append("2. GATE missing high d2 < 1")
    if 'm100.get("n", 0) < 10' not in src or 'm2400.get("n", 0) < 20' not in src:
        f.append("2. VOID missing")
    if "pick_corpus" not in src:
        f.append("3. corpus pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate growth only",
     "    gate = (not void) and (m2400.get(\"d2\", 0) > m100.get(\"d2\", 0) + 1) and (\n"
     "        h2400.get(\"d2\", 99) < 1.0)",
     "    gate = (not void) and (m2400.get(\"d2\", 0) > m100.get(\"d2\", 0) + 1)",
     "2."),
    ("no high cap",
     "    if v in high_set:\n        allow = 1  # glue, 510",
     "    if False:\n        allow = 1  # glue, 510",
     "1."),
    ("high ring2",
     "        return dict(d1=len(r1), d2=0, allow=allow)",
     "        return dict(d1=len(r1), d2=len(r1), allow=allow)",
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
