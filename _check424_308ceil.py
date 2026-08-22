"""Check of 424: 308 two-hole ceiling. No torch, no Phi. Five mutants.

  1. Holes further than frame_max (308 leak closed).
  2. VOID = both_offered.
  3. ROOM / ARENA bars.
  4. GO = conjunction; no 390 / Phi; no --joint-lines cap.
  5. line_map lists (no overwrite); source line skipped for joint_seen.
"""
from __future__ import annotations

from pathlib import Path

import _audit424_308ceil as M

SRC = Path("_audit424_308ceil.py")


def _pad(k):
    return " " + " ".join(f"p{k}x{j}" for j in range(24))


DESIGNED = [
    "the capital of FRANCE was recorded as the city PARIS in one" + _pad(0),
    "the capital of FRANCE was recorded as the city PARIS in two" + _pad(1),
    "the capital of ITALY was recorded as the city ROME in one" + _pad(2),
    "the capital of ITALY was recorded as the city ROME in two" + _pad(3),
]


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    class A:
        frame_max = 3
        min_fillers = 2
        addresses = 1500
        pairs_per_line = 4
        max_questions = 200
    import random
    rng = random.Random(0)
    rep = M.measure(DESIGNED, A, rng)
    if rep is None:
        return ["0. designed tape built no questions"]
    if "abs(pos[a] - pos[b]) > args.frame_max" not in src:
        f.append("1. holes are not required to be further than frame_max")
    if 'void = rep["questions"] < 50 or rep["both_offered"] <= 0.05' not in src:
        f.append("2. VOID is not both_offered")
    if 'room = rep["joint_seen"] < 0.90' not in src:
        f.append("3. ROOM is not joint_seen < 0.90")
    if 'arena = rep["comp_only_of_offered"] > 0.05' not in src:
        f.append("3. ARENA is not comp_only_of_offered > 0.05")
    if "go = (not void) and room and arena" not in src:
        f.append("4. GO is not the conjunction")
    if "half_nbrs" in src or "compose_nbrs" in src:
        f.append("4. 390 halves leaked into 424")
    if "import torch" in src or "Deriver" in src:
        f.append("4. Phi is in this file")
    if "--joint-lines" in src or "joint_lines" in src:
        f.append("4. --joint-lines cap still present")
    if "defaultdict(list)" not in src or "slot.append(value[s])" not in src:
        f.append("5. line_map is not a list on (line, place)")
    if "if lj == line[a]:" not in src or "continue" not in src:
        f.append("5. source line is not skipped for joint_seen")
    if "n_line_place_dups" not in src:
        f.append("5. n_line_place_dups diagnostic missing")
    return f


MUTANTS = (
    ("holes can sit inside one frame",
     "                if place[a] != place[b] and abs(pos[a] - pos[b]) > args.frame_max]",
     "                if place[a] != place[b]]",
     "1."),
    ("VOID reads joint_seen",
     '    void = rep["questions"] < 50 or rep["both_offered"] <= 0.05',
     '    void = rep["joint_seen"] <= 0.05',
     "2."),
    ("GO ignores arena",
     "    go = (not void) and room and arena",
     "    go = (not void) and room",
     "4."),
    ("source line counts as seen",
     "            if lj == line[a]:\n                continue",
     "            if False:\n                continue",
     "5."),
    ("overwrite (line, place)",
     "        slot = line_map[line[s]][place[s]]\n        if slot:\n            n_lp_dup += 1\n        slot.append(value[s])",
     "        line_map[line[s]][place[s]] = [value[s]]\n        if False:\n            n_lp_dup += 1",
     "5."),
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
