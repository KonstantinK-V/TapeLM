"""Check of 525: peaked + other-branch confirm. GATE peak_conf > peak_only and > tie."""
from __future__ import annotations

from pathlib import Path

import _audit525_confirm as M

SRC = Path("_audit525_confirm.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit511_ring import cheap_rec, graph, mentions, pick_corpus" not in src:
        f.append("1. 511 reuse missing")
    if "from _audit517_window import comps" not in src:
        f.append("1. 517 comps missing")
    if "from _audit518_reldf import pct_band" not in src:
        f.append("1. 518 pct_band missing")
    if "def confirmed(" not in src or "pin in cheap_rec" not in src:
        f.append("1. other-branch confirm missing")
    if "n1 < 0.5 * n0" not in src:
        f.append("1. 524 peaked rule missing")
    if "            both.append(hit)\n        else:\n            peak.append(hit)" not in src:
        f.append("1. peak_only vs peak_conf split missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'b["hit"] - p["hit"] > 0.05' not in gate:
        f.append("2. GATE missing peak_conf > peak_only + 0.05")
    if 'b["hit"] - t["hit"] > 0.05' not in gate:
        f.append("2. GATE missing peak_conf > tie + 0.05")
    if 't["n"] < 15' not in src or 'p["n"] < 15' not in src or 'b["n"] < 15' not in src:
        f.append("2. VOID bins n < 15 missing")
    if "pick_corpus" not in src:
        f.append("3. corpus pick missing")
    if "import torch" in src or "PickNet" in src or "score_w" in src:
        f.append("4. Phi/W leaked")
    return f


MUTANTS = (
    ("gate peak delta only",
     '    gate = (not void) and (b["hit"] - p["hit"] > 0.05) and (b["hit"] - t["hit"] > 0.05)',
     '    gate = (not void) and (b["hit"] - p["hit"] > 0.05)',
     "2."),
    ("no confirm",
     "def confirmed(g, by, pin, others, cache):\n    for c in others:\n        if pin in cheap_rec(g, by, c, cache):\n            return True\n    return False",
     "def confirmed(g, by, pin, others, cache):\n    return False",
     "1."),
    ("no peaked split",
     "        elif conf:\n            both.append(hit)\n        else:\n            peak.append(hit)",
     "        elif conf:\n            peak.append(hit)\n        else:\n            peak.append(hit)",
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
