"""Check of 513: Q[(df_bin, go/stop)]. No 200/df budget. Separate from 512."""
from __future__ import annotations

from pathlib import Path

import _audit513_qbin as M

SRC = Path("_audit513_qbin.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def df_bin(" not in src or src.count('b = df_bin(g["df"][cur])') != 2:
        f.append("1. df_bin on current token df missing")
    if "Q[(b, \"go\")]" not in src or "Q[(b, \"stop\")]" not in src:
        f.append("1. Q on (df_bin, go/stop) missing")
    if "0.3 * dm - 0.05" not in src:
        f.append("1. meet reward - hop cost missing")
    if "int(0.6 * len(mid))" not in src:
        f.append("1. 60/40 train-test split missing")
    if "200 / max(g[\"df\"]" in src or "allow = max" in src:
        f.append("1. 510 budget must not be in 513 policy")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'lh["d2"] < 1.5' not in gate:
        f.append("2. GATE missing high d2 < 1.5")
    if 'lm["d2"] > lh["d2"] + 1' not in gate:
        f.append("2. GATE missing mid d2 > high + 1")
    if 'lh["d2"] < ah["d2"] - 0.5' not in gate:
        f.append("2. GATE missing learned high < always high")
    if "_stage512" in src:
        f.append("3. must not mix 512 JSON")
    if 'lm["n"] < 15' not in src or 'lh["n"] < 5' not in src:
        f.append("2. VOID test split missing")
    return f


MUTANTS = (
    ("gate always high",
     "    gate = (not void) and (lh[\"d2\"] < 1.5) and (lm[\"d2\"] > lh[\"d2\"] + 1) and (\n"
     "        lh[\"d2\"] < ah[\"d2\"] - 0.5)",
     "    gate = (not void) and (ah[\"d2\"] > 0)",
     "2."),
    ("budget leaked",
     "    while hops < max_h:",
     "    allow = max(1, int(200 / max(g[\"df\"][cur], 1)))\n    while hops < max_h:",
     "1."),
    ("Q on address",
     "            for _h in range(12):\n"
     "                b = df_bin(g[\"df\"][cur])",
     "            for _h in range(12):\n"
     "                b = str(cur)",
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
