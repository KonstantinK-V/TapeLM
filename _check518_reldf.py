"""Check of 518: relative df bands + k*n/df budget. GATE mid d2 grows; high d2 stays short."""
from __future__ import annotations

from pathlib import Path

import _audit518_reldf as M

SRC = Path("_audit518_reldf.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit511_ring import cheap_rec, graph, mentions, pick_corpus" not in src:
        f.append("1. 511 reuse missing")
    if "p25 <= d <= p75" not in src or "d > p75" not in src:
        f.append("1. mid/high must come from prefix quantiles")
    if "k * g[\"n\"] / max(g[\"df\"][v]" not in src:
        f.append("1. allow = k*n/df missing")
    if "k = 200.0 / max(g400[\"n\"]" not in src:
        f.append("1. k calibrated from N=400 missing")
    if "SIZES = (100, 400, 1200, 2400)" not in src:
        f.append("1. nested length curve missing")
    if "base[: min(n, len(base))]" not in src:
        f.append("1. nested prefix slices missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'm2400.get("d2", 0) > m100.get("d2", 0) + 1' not in gate:
        f.append("2. GATE missing mid d2 growth +1")
    if 'h2400.get("d2", 99) < 1.0' not in gate:
        f.append("2. GATE missing high d2 < 1")
    if '["allow"]' in gate:
        f.append("2. allow must not gate alone")
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
    ("flat allow",
     "    allow = max(1, int(k * g[\"n\"] / max(g[\"df\"][v], 1)))",
     "    allow = 50",
     "1."),
    ("absolute mid",
     "    mid = [v for v, d in dfn.items() if p25 <= d <= p75]",
     "    mid = [v for v, d in dfn.items() if 8 <= d <= 30]",
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
