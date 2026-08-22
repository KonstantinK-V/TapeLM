"""Check of 447: refuse->seek; first seek may not cut.

  1. POS has TAG in CRISP frame + red/cat seek rows.
  2. next_seek / filter_cands iterate; not 445 dump.
  3. GATE mean_seek==2; VOID if first_seek_unresolved < 1.
  4. No Phi / wiki.
"""
from __future__ import annotations

from pathlib import Path

import _audit447_seek as M

SRC = Path("_audit447_seek.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "CRISP FRESH TAG NOISE" not in src:
        f.append("1. POS TAG-in-frame world missing")
    if "NOISE red board" not in src or "TAG cat label" not in src:
        f.append("1. seek rows (red/cat) missing")
    if "def designed_neg" not in src:
        f.append("1. NEG world missing")
    if "def next_seek" not in src or "def filter_cands" not in src:
        f.append("2. seek/filter helpers missing")
    if 'pos["mean_seek"] == 2.0' not in src:
        f.append("3. GATE mean_seek==2 missing")
    if 'pos["first_seek_unresolved"] < 1.0' not in src:
        f.append("3. VOID first_seek_unresolved missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    if "wikitext" in src:
        f.append("4. wiki leaked")
    return f


MUTANTS = (
    ("TAG not in frame",
     '    v = ["crates mark xx CRISP FRESH TAG NOISE" + _pad(50 + i) for i in range(3)]',
     '    v = ["crates mark xx CRISP FRESH NOISE TAG" + _pad(50 + i) for i in range(3)]',
     "1."),
    ("no next_seek",
     "def next_seek(order, by_key, visited, cands, used_k):",
     "def seek_next(order, by_key, visited, cands, used_k):",
     "2."),
    ("seek budget 1 ok",
     '            and (pos["mean_seek"] == 2.0) and (pos["crisp"] == 1.0) and (pos["ripe"] == 0.0)',
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
