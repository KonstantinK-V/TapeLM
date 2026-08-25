"""Check of 547: star wave, no unique leak.

    python _check547_starwave.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit547_starwave.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "unique_next" in src or "narrow_next" in src or "uniq_aug" in src:
        f.append("1. unique exam leaked back")
    if "1/|cands|" in src or "len(cands)" in src:
        f.append("1. cand-count reward")
    if "global_counts" in src or "gc.get" in src:
        f.append("1. lexicon gc")
    if "hash(" in src:
        f.append("1. token hash")
    if "CrossEntropy" in src:
        f.append("4. CE")
    if "residual=bool(h2 and not h1)" not in src:
        f.append("1. hop2 pays only residual held")
    if "hop1 only" not in src and "A_hop1" not in src:
        f.append("1. hop1 ablation missing")
    if "D_peak" not in src:
        f.append("1. peaked control missing")
    if "lb - la > 0.05" not in src or "lb - ld > 0.05" not in src:
        f.append("2. GATE vs hop1/peak missing")
    if "lb - lc > 0.05" not in src:
        f.append("2. GATE vs null missing")
    if "cheap_rec" not in src or "pct_band" not in src:
        f.append("1. 511/519 star reuse missing")
    if "by[v] = list(rest)" not in src:
        f.append("1. held slot must be out of rec")
    return f


MUTANTS = (
    ("unique back",
     "residual=bool(h2 and not h1)",
     "residual=bool(h2 and not h1)  # unique_next",
     "1."),
    ("hop2 pays even without residual",
     "residual=bool(h2 and not h1)",
     "residual=bool(h12)",
     "1."),
    ("gate drops peak",
     "    gate = (not void) and lb - la > 0.05 and lb - ld > 0.05 and lb - lc > 0.05",
     "    gate = (not void) and lb - la > 0.05 and lb - lc > 0.05",
     "2."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {n}")
            continue
        got = props(src=src.replace(old, new, 1))
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): not caught")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
