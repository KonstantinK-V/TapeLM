"""Check of 445: useful extra pin; useless extra refuse.

  1. POS has mat on CRISP; NEG has red on both.
  2. resolve uses len(hit)==1 per key.
  3. GATE: POS crisp==1; NEG refuse==1 and crisp==ripe==0.
  4. No Phi / wiki.
"""
from __future__ import annotations

from pathlib import Path

import _audit445_xor as M

SRC = Path("_audit445_xor.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "CRISP FRESH mat" not in src:
        f.append("1. POS mat world missing")
    if "CRISP FRESH red" not in src or "RIPE FRESH red" not in src:
        f.append("1. NEG red-on-both missing")
    if "if len(hit) == 1:" not in src:
        f.append("2. per-key uniqueness (len(hit)==1) missing")
    if 'pos["crisp"] == 1.0' not in src:
        f.append("3. GATE POS crisp missing")
    if 'neg["refuse"] == 1.0' not in src:
        f.append("3. GATE NEG refuse missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    if "wikitext" in src:
        f.append("4. wiki leaked")
    return f


MUTANTS = (
    ("NEG no shared red",
     '    w = ["shops sell RIPE FRESH red gone" + _pad(60 + i) for i in range(3)]',
     '    w = ["shops sell RIPE FRESH witems gone" + _pad(60 + i) for i in range(3)]',
     "1."),
    ("any intersect",
     "        if len(hit) == 1:",
     "        if len(hit) >= 1:",
     "2."),
    ("NEG may pick",
     '            and (neg["refuse"] == 1.0) and (neg["crisp"] == 0.0) and (neg["ripe"] == 0.0)',
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
