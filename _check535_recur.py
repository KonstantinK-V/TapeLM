"""Check of 535: recurrent marks, hop1 frozen 511. Gate: extra>0.05 and allgo Δ>0.05."""
from __future__ import annotations

from pathlib import Path

import _audit535_recur as M

SRC = Path("_audit535_recur.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit532_pool import slice_graph" not in src:
        f.append("1. 532 slice_graph reuse missing")
    if "from _audit527_learn import allow_of, majority, v1_nodes" not in src:
        f.append("1. 527 allow_of/v1_nodes missing")
    if "def offer_h1frozen(" not in src:
        f.append("1. offer_h1frozen missing")
    if "h1 = rec[0] if rec else None" not in src:
        f.append("1. hop1 frozen from 511 cheap_rec order missing")
    if "extra = [c for c in marked if c in rec_set and c != h1]" not in src:
        f.append("1. recurrent marks after hop1, not replacing hop1")
    if "pair[vc] += 1" not in src or "n >= args.min_recur" not in src:
        f.append("1. recurrent pair count (506) missing")
    if "default=2" not in src or "--min-recur" not in src:
        f.append("1. min-recur default 2 missing")
    if "h1_same" not in src:
        f.append("1. hop1_same metric missing")
    if "c in held and c not in seen and c != maj" not in src:
        f.append("1. mark only on residual held (not maj) missing")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'te["extra"] > 0.05' not in gate:
        f.append("2. GATE missing extra > 0.05")
    if "dA > 0.05" not in gate:
        f.append("2. GATE missing allgo Δ > 0.05")
    if "n_r < 10" not in src:
        f.append("2. VOID on few recurrent marks missing")
    return f


MUTANTS = (
    ("gate extra only",
     '    gate = (not void) and (te["extra"] > 0.05) and (dA > 0.05)',
     '    gate = (not void) and (te["extra"] > 0.05)',
     "2."),
    ("hop1 not frozen",
     "    h1 = rec[0] if rec else None\n    extra = [c for c in marked if c in rec_set and c != h1]",
     "    h1 = None\n    extra = [c for c in marked if c in rec_set]",
     "1."),
    ("no recur filter",
     "        if n >= args.min_recur:",
     "        if n >= 1:",
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
