"""Check of 468: Q on place only (rec[1]); shuffle gate; no trap/n_open."""
from __future__ import annotations

from pathlib import Path

import _audit468_boot as M

SRC = Path("_audit468_boot.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    pick = src[src.find("def eps_pick("):src.find("def rollout(")]
    if "table.get(rec[1]" not in pick:
        f.append("1. Q not place-only")
    if "trap_of" in src or "n_open" in src or "sig_rec" in src:
        f.append("1. trap/n_open leaked")
    if "t1s[\"p_live\"] < t1[\"p_live\"] - 0.3" not in src:
        f.append("3. GATE missing shuffle")
    if "held" in src[src.find("gate ="):src.find("rec = dict")] and "held[" in src[src.find("gate ="):src.find("rec = dict")]:
        f.append("3. held gated")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("Q uses hole",
     "        r = table.get(rec[1], 0.0)",
     "        r = table.get(rec[2], 0.0)",
     "1."),
    ("gate drops shuffle",
     '            and (t1s["p_live"] < t1["p_live"] - 0.3))',
     "            )",
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
