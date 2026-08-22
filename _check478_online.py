"""Check of 478: mid-episode +0.2 before hop3; 2 cands; t1-t0 > DELTA."""
from __future__ import annotations

from pathlib import Path

import _audit478_online as M

SRC = Path("_audit478_online.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "touch(table, tot, win, H, cost + 0.2)" not in src:
        f.append("1. mid +0.2 missing")
    ep = src[src.find("def episode("):src.find("def run_n(")]
    mid = ep.find("cost + 0.2")
    hop3 = ep.find("ok3")
    if mid < 0 or hop3 < 0 or mid > hop3:
        f.append("1. +0.2 not before hop3")
    if "len(cands) != 2" not in src:
        f.append("1. 2-cand missing")
    if 't1["p_h2"] - t0["p_h2"] > DELTA' not in src:
        f.append("2. GATE missing lift")
    if "import torch" in src or "wikitext" in src:
        f.append("3. leak")
    return f


MUTANTS = (
    ("no mid credit",
     "        touch(table, tot, win, H, cost + 0.2)",
     "        pass  # no mid",
     "1."),
    ("no lift",
     '            and (t1["p_h2"] - t0["p_h2"] > DELTA)',
     "            and True",
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
