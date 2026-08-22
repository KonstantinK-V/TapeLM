"""Check of 429: ceiling first, then Phi on 6 letter-free graph feats."""
from __future__ import annotations

from pathlib import Path

import _train429_hang as M

SRC = Path("_train429_hang.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "CEILING" not in src or "hang428" not in src:
        f.append("1. ceiling is not computed")
    if "if best > 0.05:" not in src:
        f.append("2. GO algebra does not skip the net")
    body = src.split("def pair_feats")[1].split("class HangNet")[0]
    if "pair_seen" in body:
        f.append("3. pair_seen leaked into features")
    if "fillers_place" in src or "token_id" in src:
        f.append("3. letter/catalog id in features")
    if "acc > 0.55" not in src:
        f.append("4. GATE is not P(true>maj) > 0.55")
    if "exam_co" not in src:
        f.append("4. exam is not held-out comp_only")
    if "HangNet" not in src:
        f.append("5. net is missing")
    return f


MUTANTS = (
    ("train even if algebra GO",
     "    if best > 0.05:",
     "    if best > 9.0:",
     "2."),
    ("pair_seen feature",
     "        math.log1p(n_cross) / 4.0,\n    ]",
     "        math.log1p(n_cross) / 4.0,\n        1.0,  # pair_seen\n    ]",
     "3."),
    ("gate is coin",
     "    gate = acc > 0.55",
     "    gate = acc > 0.0",
     "4."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
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
