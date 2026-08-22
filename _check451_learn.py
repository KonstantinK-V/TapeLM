"""Check of 451: iso train/test, win-rate, no n_follow chooser."""
from __future__ import annotations

from pathlib import Path

import _audit451_learn as M

SRC = Path("_audit451_learn.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def iso_world(" not in src or "n_train" not in src:
        f.append("1. iso train/test missing")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "max(n_follow)" in src or "argmax n_follow" in src:
        f.append("2. n_follow hardcoded as chooser")
    if "rate.get((n_hit, d1)" not in src:
        f.append("2. win-rate policy missing")
    if 'rep["learned"] == 1.0' not in src or 'rep["greedy"] == 0.0' not in src:
        f.append("3. GATE is not learned vs greedy")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no iso",
     "def iso_world(rng):",
     "def iso_world_off(rng):",
     "1."),
    ("hardcode n_follow",
     "            r = rate.get((n_hit, d1), 0.0)",
     "            r = float(n_follow(value[pinH], order, by_key, visited, cands, used_k))",
     "2."),
    ("gate ignores greedy",
     '            and (rep["greedy"] == 0.0)',
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
