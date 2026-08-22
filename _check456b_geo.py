"""Check of 456-B: train 455 table, test flat start (no fruit spine)."""
from __future__ import annotations

from pathlib import Path

import _audit456b_geo as M

SRC = Path("_audit456b_geo.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def starts_flat" not in src or "FAM_B" not in src:
        f.append("1. flat geo / FAM_B missing")
    if "train_return" not in src:
        f.append("1. does not train on 456 return-table")
    if "wikitext" in src:
        f.append("1. wiki leaked")
    if "names[\"FRESH\"]" not in src and "names['FRESH']" not in src:
        f.append("2. flat start is not FRESH")
    if "APPLES" in src and "starts_flat" in src:
        # APPLES may appear only in comments; require no hop chain in starts_flat
        pass
    if "hop(" in src.split("def starts_flat")[1].split("def eval_flat")[0]:
        f.append("2. starts_flat still hops the fruit spine")
    if 'no["BOTH"]["mean_hops"] > 2.0' not in src:
        f.append("3. GATE missing no-soon BOTH")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no flat",
     "def starts_flat(g, rng, tok):",
     "def starts_plain(g, rng, tok):",
     "1."),
    ("spine hop",
     "        pin = think_slot(s, g[\"slots_at\"], place, value, line, rng)",
     "        pin = think_slot(s, g[\"slots_at\"], place, value, line, rng)\n        hop(value[pin], place[s], g[\"by_key\"], g[\"slots_at\"], value, rng)",
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
