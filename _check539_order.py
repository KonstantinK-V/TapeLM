"""Check of 539: rank profile maj on/off, order-id value, paired vs fixed df_up."""
from __future__ import annotations

from pathlib import Path

import _audit539_order as M

SRC = Path("_audit539_order.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit538_role import RMAX, role_key" not in src:
        f.append("1. 538 role_key reuse missing")
    if "ORDER_NAMES" not in src or "df_up" not in src:
        f.append("1. four order ids missing")
    if "def rec_ordered(" not in src:
        f.append("1. rec_ordered missing")
    if "prof_res" not in src or "prof_maj" not in src:
        f.append("1. rank profile maj on/off missing")
    if "skip_maj=True" not in src or "skip_maj=False" not in src:
        f.append("1. residual vs maj taught_rank missing")
    if "paired_d += d_t - d_f" not in src:
        f.append("1. paired vs fixed order missing")
    if "prof_maj[rm] += 1" not in src:
        f.append("1. prof_maj increment missing")
    if "FIXED_ORDER = 0" not in src:
        f.append("1. fixed baseline FIXED_ORDER=0 missing")
    if "rec_fix = rec_ordered(g, by, v, cx, FIXED_ORDER)" not in src:
        f.append("1. fixed rec_fix from FIXED_ORDER missing")
    if "paired_n >= 40 and paired_d > 0" not in src:
        f.append("2. GATE paired_d > 0 missing")
    if "import torch" in src:
        f.append("4. CE leaked")
    return f


MUTANTS = (
    ("no maj profile",
     "            if rm is not None:\n                prof_maj[rm] += 1",
     "            if rm is not None:\n                pass",
     "1."),
    ("paired uses same rec",
     "            rec_fix = rec_ordered(g, by, v, cx, FIXED_ORDER)",
     "            rec_fix = rec_best",
     "1."),
    ("no fixed order 0",
     "FIXED_ORDER = 0\n",
     "",
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
