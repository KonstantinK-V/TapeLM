"""Check of D2: marks stay off the tape, the count stays on original places."""
from __future__ import annotations

from pathlib import Path

import _auditD2_round as M

SRC = Path("_auditD2_round.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit485_hunt import build_window, load_lines, pick_corpus" not in src:
        f.append("1. 485 window builder reuse missing")
    if 'orig = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]' not in src:
        f.append("1. the original place set must be fixed before any writing")
    if "vals = [x for x in (sharp(P, g, marks, bk) for P in orig) if x is not None]" not in src:
        f.append("1. coverage must be measured over the ORIGINAL places only")
    if "return 1.0 / k if k else 0.0" not in src:
        f.append("1. coverage must be graded (1/|cands|), not the unique knife edge")
    if "marks.setdefault(H, Counter())[v] += 1" not in src:
        f.append("1. marks must live in their own structure, not in the text")
    if 'g["value"][i]' in src and 'g["value"].append' in src:
        f.append("1. a mark was written into the tape itself")
    if "bk = bykey_aug(g, marks)" not in src:
        f.append("1. the index must be rebuilt from marks each round")
    if 'pins = [(H, vals[rng.randrange(len(vals))]) for H, _ in pins]' not in src:
        f.append("1. the shuffled-value arm missing")
    if 'if arm == "none":\n            pins = []' not in src:
        f.append("1. the no-write arm missing")
    if "def holes_of(g, orig, marks, weak_frac):" not in src:
        f.append("1. holes_of missing")
    if "if frac >= pin_frac and H in holes:" not in src:
        f.append("1. the pinned value must be derived FOR that hole by the walk")
    if "            if u != keep:\n                s.discard(P)" not in src:
        f.append("1. a mark must RESOLVE (drop the other values), not append an edge")
    if "n_edge += sum(1 for u, sset in bk.items() if u != v and H in sset)" not in src:
        f.append("3. edges_cut must be counted - a write that cuts none cannot change anything")
    if 'arms["write"]["edges_cut"] < 50' not in src:
        f.append("2. VOID when the write cut no edges missing")
    if "import torch" in src:
        f.append("4. a scorer leaked into a counting arm")
    gate = src[src.find("    gate = "):src.find("    rec = dict")]
    if "gw > 0.01 and gw - gs > 0.01" not in gate:
        f.append("2. GATE must require growth AND beating the shuffled values")
    if "or abs(gn) > 1e-9 or not same_cov0)" not in src:
        f.append("2. VOID unless the no-write arm is exactly flat")
    if "same_cov0" not in src:
        f.append("2. VOID unless all arms start from the same coverage")
    return f


MUTANTS = (
    ("coverage measured over marked places too",
     "    vals = [x for x in (sharp(P, g, marks, bk) for P in orig) if x is not None]",
     "    vals = [x for x in (sharp(P, g, marks, bk) for P in list(orig) + list(marks)) if x is not None]",
     "1."),
    ("back to the unique knife edge",
     "    return 1.0 / k if k else 0.0",
     "    return 1.0 if k == 1 else 0.0",
     "1."),
    ("the shuffled-value arm dropped",
     "            pins = [(H, vals[rng.randrange(len(vals))]) for H, _ in pins]",
     "            pins = list(pins)",
     "1."),
    ("the flat-arm invariant removed",
     "            or abs(gn) > 1e-9 or not same_cov0)",
     "            )",
     "2."),
    ("the hole is no longer the one the walk named",
     "            if frac >= pin_frac and H in holes:",
     "            if frac >= pin_frac:",
     "1."),
    ("the mark appends instead of resolving",
     "            if u != keep:\n                s.discard(P)",
     "            if u != keep:\n                pass",
     "1."),
    ("edges-cut VOID removed",
     '            or arms["write"]["edges_cut"] < 50\n',
     "",
     "2."),
    ("gate no longer needs to beat shuffled values",
     "    gate = (not void) and gw > 0.01 and gw - gs > 0.01",
     "    gate = (not void) and gw > 0.01",
     "2."),
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
