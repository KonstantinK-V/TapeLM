"""Check of 469: two tapes (+99), qb==0; apples not gated."""
from __future__ import annotations

from pathlib import Path

import _audit469_scan as M

SRC = Path("_audit469_scan.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "seed + 99" not in src and "args.seed + 99" not in src:
        f.append("1. foreign tape seed missing")
    if "qb == 0.0" not in src:
        f.append("1. GATE missing qb==0")
    if "APPLES" in src or "wikitext" in src:
        f.append("1. apples/wiki leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "qa" in gate or "qh_on_a" in gate:
        f.append("3. Q on A gated")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("same tape",
     "    lines_b, nb = world_d2(random.Random(args.seed + 99))",
     "    lines_b, nb = world_d2(random.Random(args.seed))",
     "1."),
    ("drop qb",
     "            and (qb == 0.0))",
     "            )",
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
