"""Check of 452: shuffle, NEG opener, frozen eval, GATE."""
from __future__ import annotations

from pathlib import Path

import _audit452_xfer as M

SRC = Path("_audit452_xfer.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "rng.shuffle(ctx)" not in src:
        f.append("1. key order not shuffled")
    if 'opener = n["KEYA"] if neg else n["UNLOCK"]' not in src:
        f.append("1. NEG opener not flipped")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "eval_table(rng, args.test, pos_rate, neg=True)" not in src:
        f.append("2. frozen POS-on-NEG missing")
    if "pos_l == 1.0" not in src or "neg_frozen == 0.0" not in src or "neg_l == 1.0" not in src:
        f.append("3. GATE is not pos/frozen/retrain")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no shuffle",
     "    rng.shuffle(ctx)",
     "    ctx = ctx",
     "1."),
    ("no frozen eval",
     "    neg_frozen, _, _ = eval_table(rng, args.test, pos_rate, neg=True)",
     "    neg_frozen, _, _ = 1.0, 0, 1",
     "2."),
    ("gate drops frozen",
     "            and (neg_frozen == 0.0)",
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
