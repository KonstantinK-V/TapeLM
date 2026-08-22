"""Check of 524: peaked vs tied top-2 within one hop. GATE cert hit > unc + 0.05."""
from __future__ import annotations

from pathlib import Path

import _audit524_onehop as M

SRC = Path("_audit524_onehop.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit511_ring import cheap_rec, graph, mentions, pick_corpus" not in src:
        f.append("1. 511 reuse missing")
    if "from _audit517_window import comps" not in src:
        f.append("1. 517 comps missing")
    if "from _audit518_reldf import pct_band" not in src:
        f.append("1. 518 pct_band missing")
    if "def rec_counts(" not in src or "cnt.update" not in src:
        f.append("1. companion counts for top-2 missing")
    if "n1 < 0.5 * n0" not in src:
        f.append("1. top-2 tie rule n1 < 0.5*n0 missing")
    if "jacc" in src or "A, B" in src:
        f.append("1. 523 A∩B must not gate 524")
    if "            unc.append(hit)\n        else:\n            cert.append(hit)" in src:
        f.append("1. tied hits must go to unc not cert")
    if "            cert.append(hit)\n        else:\n            unc.append(hit)" not in src:
        f.append("1. peaked/tied split missing")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'c["hit"] - u["hit"] > 0.05' not in gate:
        f.append("2. GATE missing cert-unc delta > 0.05")
    if 'c["hit"]' in gate and 'u["hit"]' in gate and "-" not in gate:
        f.append("2. must gate on delta not raw hits")
    if 'c["n"] < 20' not in src or 'u["n"] < 20' not in src:
        f.append("2. VOID missing")
    if "pick_corpus" not in src:
        f.append("3. corpus pick missing")
    if "import torch" in src or "PickNet" in src or "score_w" in src:
        f.append("4. Phi/W leaked")
    return f


MUTANTS = (
    ("gate delta only",
     '    gate = (not void) and (c["hit"] - u["hit"] > 0.05)',
     '    gate = (not void) and (c["hit"] - u["hit"] > -1)',
     "2."),
    ("no tie rule",
     "        peaked = (len(rec) == 1) or (n0 > 0 and n1 < 0.5 * n0)  # top-2 gap",
     "        peaked = True  # top-2 gap",
     "1."),
    ("pin tie",
     "            unc.append(hit)",
     "            cert.append(hit)",
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
