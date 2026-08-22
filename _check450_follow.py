"""Check of 450: hop1 by n_follow (new addresses). No 449 deep peek.

  1. Uses 449 designed/build/filter/greedy; has def n_follow.
  2. No def oracle_d2 / def deep (peek of hop2 value).
  3. GATE follow_crisp==1, greedy_refuse==1, picked_unlock.
  4. No Phi / wiki.
"""
from __future__ import annotations

from pathlib import Path

import _audit450_follow as M

SRC = Path("_audit450_follow.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit449_adv import" not in src:
        f.append("1. 449 world/helpers missing")
    if "def n_follow(" not in src:
        f.append("1. n_follow missing")
    if "def oracle_d2" in src or "def deep(" in src:
        f.append("2. hop2-value peek leaked (oracle_d2/deep)")
    if "nf = n_follow(" not in src:
        f.append("2. pick_follow does not score by n_follow")
    if 'rep["follow_crisp"] == 1.0' not in src or 'rep["greedy_refuse"] == 1.0' not in src:
        f.append("3. GATE follow/greedy missing")
    if 'rep["picked_unlock"] < 1.0' not in src:
        f.append("3. VOID picked_unlock missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    if "wikitext" in src:
        f.append("4. wiki leaked")
    return f


MUTANTS = (
    ("no n_follow",
     "def n_follow(vH, order, by_key, visited, cands, used_k):",
     "def count_follow(vH, order, by_key, visited, cands, used_k):",
     "1."),
    ("deep peek",
     'OUT = Path("results/_stage450_follow.json")',
     'OUT = Path("results/_stage450_follow.json")\ndef deep(cands): return cands',
     "2."),
    ("gate soft",
     '            and (rep["follow_crisp"] == 1.0) and (rep["follow_ripe"] == 0.0)',
     "            and True",
     "3."),
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
