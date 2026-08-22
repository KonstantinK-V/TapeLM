"""Check of 504: Famb resolve via line extra (444). GATE resolve vs foreign-line."""
from __future__ import annotations

from pathlib import Path

import _audit504_famb as M

SRC = Path("_audit504_famb.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit440_compose import think_place" not in src:
        f.append("1. frozen 440 think_place missing")
    if "def unique_hit(" not in src or "len(hit) == 1" not in src:
        f.append("1. 444 unique resolve missing")
    if "len(c) < 2" not in src:
        f.append("1. Famb population filter missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "resolve > 0.05" not in gate:
        f.append("2. GATE missing resolve floor")
    if "resolve > shuffle + 0.05" not in gate:
        f.append("2. GATE missing resolve vs foreign-line")
    if "hop_any" in gate or "hop_rnd" in gate:
        f.append("2. hop_any/hop_rnd must not gate")
    if "n < 30" not in src:
        f.append("2. VOID n_famb < 30 missing")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate hop_any",
     "    gate = (not void) and (resolve > 0.05) and (resolve > shuffle + 0.05)",
     "    gate = (not void) and (hop_any > hop_rnd + 0.05)",
     "2."),
    ("no foreign beat",
     "    gate = (not void) and (resolve > 0.05) and (resolve > shuffle + 0.05)",
     "    gate = (not void) and (resolve > 0.05)",
     "2."),
    ("Q leaked",
     "from _audit440_compose import think_place",
     "from _audit440_compose import think_place\nfrom _audit485_hunt import pick_by_q",
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
