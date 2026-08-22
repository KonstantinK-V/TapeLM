"""Check of 449: greedy 4->3 refuse vs 2-step deep CRISP.

  1. Uses 447 hop/next_cands and 448 all_seeks; designed 4-way + UNLOCK/TAG.
  2. filter hit==0 continues (address); UNLOCK via order+[value], not used_k.
  3. GATE greedy_refuse==1 and deep_crisp==1.
  4. No Phi / wiki.
"""
from __future__ import annotations

from pathlib import Path

import _audit449_adv as M

SRC = Path("_audit449_adv.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit447_seek import" not in src or "from _audit448_pick import" not in src:
        f.append("1. 447/448 imports missing")
    if "TAG UNLOCK" not in src or "KEYA TAG" not in src:
        f.append("1. UNLOCK/TAG world missing")
    if "if len(hit) == 0:\n            continue" not in src:
        f.append("2. hit==0 must continue (not empty)")
    if "order + [value[pinH]]" not in src:
        f.append("2. UNLOCK not offered as address (order+value)")
    if 'rep["greedy_refuse"] == 1.0' not in src or 'rep["deep_crisp"] == 1.0' not in src:
        f.append("3. GATE greedy_refuse / deep_crisp missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    if "wikitext" in src:
        f.append("4. wiki leaked")
    return f


MUTANTS = (
    ("no deep world",
     '    hT = ["vault keeps TAG UNLOCK safe here" + _pad(110 + i) for i in range(3)]',
     '    hT = ["vault keeps FOG ONLY safe here" + _pad(110 + i) for i in range(3)]',
     "1."),
    ("hit0 empties",
     "        if len(hit) == 0:\n            continue",
     "        if len(hit) == 0:\n            return set()",
     "2."),
    ("gate ignores greedy",
     '    gate = ((not void) and (rep["greedy_refuse"] == 1.0)',
     "    gate = ((not void) and True",
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
