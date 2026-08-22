"""Check of 537: additive allow, hop1 frozen, gate on allgo lift."""
from __future__ import annotations

from pathlib import Path

import _audit537_add as M

SRC = Path("_audit537_add.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def offer_add(" not in src:
        f.append("1. offer_add missing")
    if "return list(n511) + extra" not in src:
        f.append("1. extras appended at tail missing")
    if "if v in high_set:\n        return list(n511)" not in src:
        f.append("1. high words: no additive tail")
    if "return list(n511) + list(marked)" in src:
        f.append("1. high path must not append marks")
    if "hop1_same += int(h511 == hm)" not in src:
        f.append("1. hop1 frozen check missing")
    if "add = nm[len(n511):]" not in src:
        f.append("1. additive tail slice missing")
    if 'prev[f"{args.seed}_w{args.windows}"]' not in src:
        f.append("3. JSON key seed_w16/w32 missing")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "same >= 0.99" not in gate:
        f.append("2. GATE missing hop1 frozen >= 0.99")
    if "mean_dg > 0.05" not in gate:
        f.append("2. GATE missing allgo Δ > 0.05")
    if "extra_n < 40" not in src:
        f.append("2. VOID on extra_n < 40 missing")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    return f


MUTANTS = (
    ("gate allgo only",
     "    gate = (not void) and (same >= 0.99) and (mean_dg > 0.05)",
     "    gate = (not void) and (mean_dg > 0.05)",
     "2."),
    ("prepend not append",
     "    return list(n511) + extra",
     "    return extra + list(n511)",
     "1."),
    ("no hop1 freeze high",
     "def offer_add(g, by, v, cache, k, high_set, marked, n511):\n"
     "    if v in high_set:\n        return list(n511)",
     "def offer_add(g, by, v, cache, k, high_set, marked, n511):\n"
     "    if v in high_set:\n        return list(n511) + list(marked)",
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
