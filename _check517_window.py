"""Check of 517: working window W ranks hop2 vs rare-first 511. GATE mid delta > 0.05."""
from __future__ import annotations

from pathlib import Path

import _audit517_window as M

SRC = Path("_audit517_window.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit511_ring import cheap_rec, mentions" not in src:
        f.append("1. 511 cheap_rec reuse missing")
    if "CAP_W = 8" not in src or "W = [h1][:CAP_W]" not in src:
        f.append("1. working window W from hop1 missing")
    if "if use_w:\n        rest = sorted(rest, key=lambda c: (-score_w" not in src:
        f.append("1. W-ranked hop2 must gate on use_w")
    if "by[v] = rest" not in src or "held = set(comps" not in src:
        f.append("1. held-out frame exam missing")
    if 'tape[m] = "MEET"' in src or "PickNet" in src:
        f.append("1. Phi/MEET write leaked")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'mid_rep["delta"] > 0.05' not in gate:
        f.append("2. GATE missing mid delta > 0.05")
    if "hit_w" in gate or "hit_511" in gate:
        f.append("2. raw hits must not gate alone")
    if 'mid_rep["n"] < 50' not in src:
        f.append("2. VOID mid n < 50 missing")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src:
        f.append("4. torch leaked")
    return f


MUTANTS = (
    ("gate delta only",
     '    gate = (not void) and (mid_rep["delta"] > 0.05)',
     '    gate = (not void) and (mid_rep["delta"] > -1)',
     "2."),
    ("no W rank",
     "    if use_w:\n        rest = sorted(rest, key=lambda c: (-score_w(g, by, c, W, cache), g[\"df\"][c]))",
     "    if False:\n        rest = sorted(rest, key=lambda c: (-score_w(g, by, c, W, cache), g[\"df\"][c]))",
     "1."),
    ("no held-out",
     "            by[v] = rest\n            cache.pop(v, None)",
     "            cache.pop(v, None)",
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
