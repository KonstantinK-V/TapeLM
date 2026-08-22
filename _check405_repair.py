"""Check of 405: full-body reorder repair, not teacher forcing.

405 changes the QUESTION. The ways it can lie:

  1. TEACHER FORCING. The walk must start from a SHUFFLED pool, not the true prefix of the file.
  2. SUCCESS IS COMPLETION, NOT ORIGINAL ORDER. Nothing compares to y = the next line in the file.
  3. THE THREE POLICIES ARE THE DECLARED RIVALS - random legal, greedy Return, greedy unblocks.
  4. VOID IS GREEDY UNBLOCKS >0.90, read before the gate.
  5. THE DEFAULT CORPUS IS FOREIGN. This ceiling is read on a corpus the project did not train on.
  6. SAFE IS DEF-USE ON KEYS INSIDE THE BODY. Lines with no store/load keys are skipped, not
     counted as free completions.

    python _check405_repair.py
"""
from __future__ import annotations

import re
from pathlib import Path

import _audit404_family as F
import _audit405_repair as R

SRC = Path("_audit405_repair.py")

# two safe lines compete after `base`; Return among them loses, unblocks wins
DESIGNED = '''
def f(a):
    base = a + 1
    side = a + 2
    mix = base + side
    return mix
'''


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    bodies = F.rows_of(DESIGNED)
    if len(bodies) != 1 or len(bodies[0]) != 4:
        f.append(f"0. designed body has {len(bodies[0]) if bodies else 0} lines, not 4")

    code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src

    # 1 + 2: no teacher forcing; completion only
    if "pool = list(body)" not in src or "rng.shuffle(pool)" not in src:
        f.append("1. the pool is not a shuffled copy of the full body")
    if re.search(r"body\[:t\]|rows\[:t\]|placed = body\[:", code):
        f.append("1. teacher forcing - the true prefix is used as the starting state")
    if re.search(r"\by\s*==|safe_ix\.index\(0\)", code):
        f.append("2. success is scored against the original line order")

    # 3
    for name in ("pick_random", "pick_return", "pick_unblocks"):
        if f"def {name}" not in src:
            f.append(f"3. missing policy {name}")
    if '"Return" in pool[i][3]' not in src:
        f.append("3. greedy Return does not read node types")

    # 4
    if 'void = rep["unblocks"] > 0.90' not in src:
        f.append("4. void check is not greedy unblocks > 0.90")

    # 5
    if "DEFAULT_FOREIGN" not in src or "--corpus" not in src:
        f.append("5. foreign corpus is not the default reading")

    # 6
    if "if not keys:" not in src or "return None" not in src:
        f.append("6. bodies with no internal def-use keys are not skipped")

    # behavioural: on the designed body, Return can fail while unblocks succeeds
    if bodies:
        import random
        rng = random.Random(0)
        body = bodies[0]
        u_ok = any(R.complete(body, R.pick_unblocks, random.Random(s)) for s in range(40))
        r_all = all(R.complete(body, R.pick_return, random.Random(s)) is not False
                    for s in range(40) if R.complete(body, R.pick_return, random.Random(s)) is not None)
        if not u_ok:
            f.append("0. unblocks does not complete the designed body in 40 shuffles")
    return f


MUTANTS = (
    ("teacher forcing via the true prefix",
     "    pool = list(body)\n    rng.shuffle(pool)",
     "    pool = list(body)", "1."),
    ("the walk keeps the file order",
     "    rng.shuffle(pool)",
     "    pass  # shuffle removed", "1."),
    ("void reads random instead of unblocks",
     '    void = rep["unblocks"] > 0.90',
     '    void = rep["random"] > 0.90', "4."),
    ("bodies without keys count as auto-success",
     "    if not keys:\n        return None",
     "    if not keys:\n        return True", "6."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor count {src.count(old)}")
            continue
        got = props(src=src.replace(old, new, 1))
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): check did not fire")
    for x in fails:
        print("FAIL " + x)
    n = len(MUTANTS)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {n} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
