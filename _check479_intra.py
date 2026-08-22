"""Check of 479: mid touch before hop3 pick; gate1 and gate2 separate."""
from __future__ import annotations

from pathlib import Path

import _audit479_intra as M

SRC = Path("_audit479_intra.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "touch(table, tot, win, k2, 0.15)" not in src:
        f.append("1. mid +0.15 missing")
    ep = src[src.find("def episode("):src.find("def run_n(")]
    mid = ep.find("0.15")
    h3p = ep.find("H3 = pick")
    if mid < 0 or h3p < 0 or mid > h3p:
        f.append("1. mid credit not before hop3 pick")
    if "gate1" not in src or "gate2" not in src:
        f.append("2. dual gate missing")
    if 'pB["p_h3"] - t0["p_h3"] > DELTA' not in src:
        f.append("2. GATE2 missing")
    if "import torch" in src or "wikitext" in src:
        f.append("3. leak")
    return f


MUTANTS = (
    ("no mid",
     "        touch(table, tot, win, k2, 0.15)",
     "        pass",
     "1."),
    ("no gate2",
     '             and (pB["p_h3"] - t0["p_h3"] > DELTA))',
     "             )",
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
