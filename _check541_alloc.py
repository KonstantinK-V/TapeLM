"""Check of 541: one frozen budget, three distributions, cover at equal hops."""
from __future__ import annotations

from pathlib import Path

import _audit541_alloc as M

SRC = Path("_audit541_alloc.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit532_pool import slice_graph" not in src:
        f.append("1. 532 slice_graph reuse missing")
    if "from _audit527_learn import allow_of" not in src:
        f.append("1. 511/527 law budget allow_of missing")
    if "total = sum(r[\"law\"] for r in rows)" not in src:
        f.append("1. total frozen at what the law already spends missing")
    if "assert sum(a_alloc) == total and sum(a_uni) == total" not in src:
        f.append("1. equal-budget assertion missing")
    if "a_uni = share_out([1.0] * n, total)" not in src:
        f.append("1. even-spending null arm missing")
    if "a_alloc = share_out([float(r[\"n_rec\"]) for r in rows], total)" not in src:
        f.append("1. the arm spends by len(rec) missing")
    if "for name, alloc in ((\"law\", a_law), (\"alloc\", a_alloc), (\"uni\", a_uni)):" not in src:
        f.append("1. the three arms do not run over the same rows")
    if "hops += len(nodes)" not in src:
        f.append("1. hops counted as nodes actually offered missing")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    if "for a in (4, 8, 16, 32)" in src or "--allow-sweep" in src:
        f.append("3. this is a sweep of allow, which cover answers by arithmetic")
    gate = src[src.find("    gate = "):src.find("    rec = dict")]
    if "d_law > 0.01 and d_uni > 0.01" not in gate:
        f.append("2. GATE must beat BOTH the law and the even-spending null")
    if "hop_gap <= 0.02" not in gate:
        f.append("2. GATE on matched hops missing")
    if "void = n < 40 or hop_gap > 0.05" not in src:
        f.append("2. VOID on few trials or unspendable budget missing")
    return f


MUTANTS = (
    ("null arm dropped from the gate",
     "    gate = (not void) and hop_gap <= 0.02 and d_law > 0.01 and d_uni > 0.01",
     "    gate = (not void) and hop_gap <= 0.02 and d_law > 0.01",
     "2."),
    ("budgets no longer equal",
     "    a_uni = share_out([1.0] * n, total)",
     "    a_uni = share_out([1.0] * n, total * 2)",
     "1."),
    ("cover read at unmatched hops",
     "    gate = (not void) and hop_gap <= 0.02 and d_law > 0.01 and d_uni > 0.01",
     "    gate = (not void) and d_law > 0.01 and d_uni > 0.01",
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
