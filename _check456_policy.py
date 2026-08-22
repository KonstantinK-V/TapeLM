"""Check of 456: delayed R, soon ablation, GATE no-soon BOTH>2."""
from __future__ import annotations

from pathlib import Path

import _audit456_policy as M

SRC = Path("_audit456_policy.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def train_return" not in src or "0.05 * (hops + 1)" not in src:
        f.append("1. delayed return missing")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "use_soon else 0" not in src:
        f.append("2. soon ablation missing")
    if 'no["BOTH"]["mean_hops"] > 2.0' not in src:
        f.append("3. GATE missing no-soon BOTH")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no delay",
     "                    R = (1.0 if ok else 0.0) - 0.05 * (hops + 1)",
     "                    R = 1.0 if ok else 0.0",
     "1."),
    ("no ablation flag",
     "                    s = soon(g[\"value\"][pinH], order, g[\"by_key\"], vis1, cands, used1,\n"
     "                             g[\"place_keys\"]) if use_soon else 0",
     "                    s = soon(g[\"value\"][pinH], order, g[\"by_key\"], vis1, cands, used1,\n"
     "                             g[\"place_keys\"])",
     "2."),
    ("gate drops ablation",
     '            and (no["BOTH"]["mean_hops"] > 2.0))',
     "            and True)",
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
