"""Check of 514: MEET on working tape vs random walked node. Contract file."""
from __future__ import annotations

from pathlib import Path

import _audit514_meetmark as M

SRC = Path("_audit514_meetmark.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if 'tape[m] = "MEET"' not in src:
        f.append("1. MEET write on working tape missing")
    if "by[v] = rest_slots" not in src or "by[v] = saved" not in src:
        f.append("1. leave-one-out by[v] missing")
    if "200 / max(g[\"df\"][v]" not in src:
        f.append("1. 1/df budget walk missing")
    if "rng.sample(walked" not in src:
        f.append("1. random walked-node control missing")
    if "const_pins" in src or "from _audit512" in src:
        f.append("1. PIN pipeline leaked")
    if "pick_by_q" in src or "Q = defaultdict" in src or 'Q[(b, "go")]' in src:
        f.append("1. Q leaked")
    if "unique_next" in src or "len(cands) == 1" in src:
        f.append("1. unique hop leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'mid_rep["delta"] > 0.05' not in gate:
        f.append("2. GATE missing mid delta > 0.05")
    if "high_rep" in gate:
        f.append("2. high must not gate")
    if "_CONTRACT_STAR.txt" not in src:
        f.append("3. contract file missing")
    if "PIN  ≠ STAR" not in src:
        f.append("3. contract LAW missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no MEET write",
     '        tape[m] = "MEET"',
     '        tape[m] = "WALK"',
     "1."),
    ("no leave-one-out",
     "            by[v] = rest_slots",
     "            pass  # no leave-one-out",
     "1."),
    ("gate high",
     "    gate = (not void) and (mid_rep[\"delta\"] > 0.05)",
     "    gate = (not void) and (high_rep[\"delta\"] > 0.05)",
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
