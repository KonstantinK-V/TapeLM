"""Check of 534: end-cycle mark moves offer on same W=250 as 532."""
from __future__ import annotations

from pathlib import Path

import _audit534_mark as M

SRC = Path("_audit534_mark.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit532_pool import slice_graph" not in src:
        f.append("1. 532 slice_graph reuse missing")
    if "from _audit527_learn import allow_of, majority, v1_nodes" not in src:
        f.append("1. 527 allow_of/v1_nodes missing")
    if "def offer(" not in src:
        f.append("1. offer() with marked-first reorder missing")
    if "first = [c for c in marked if c in rec_set]" not in src:
        f.append("1. marked nodes first in cheap_rec missing")
    if "allow = allow_of(g, v, k, high_set)" not in src:
        f.append("1. same allow as 511/v1 missing")
    if "marks[v].append(c)" not in src:
        f.append("1. end-cycle mark write missing")
    if "c in held and c not in seen and c != maj" not in src:
        f.append("1. mark only on residual held (not maj) missing")
    if "train_g, test_g = graphs[:n_tr], graphs[n_tr:]" not in src:
        f.append("1. 70/30 window split missing")
    if "extra=int(any(c not in n511 for c in nm))" not in src:
        f.append("1. extra = nodes in mark-offer not in 511 missing")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "(d1 > 0.05) or (dA > 0.05)" not in gate:
        f.append("2. GATE missing hop1 OR allgo cover > 511 + 0.05")
    if "n_mark < 20" not in src:
        f.append("2. VOID on few marks missing")
    return f


MUTANTS = (
    ("gate hop1 only",
     "    gate = (not void) and ((d1 > 0.05) or (dA > 0.05))",
     "    gate = (not void) and (d1 > 0.05)",
     "2."),
    ("no marked first",
     "    first = [c for c in marked if c in rec_set]\n    rec = first + [c for c in rec if c not in first]",
     "    first = []\n    rec = first + [c for c in rec if c not in first]",
     "1."),
    ("mark any held",
     "                if c in held and c not in seen and c != maj:",
     "                if c in held:",
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
