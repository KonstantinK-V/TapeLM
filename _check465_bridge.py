"""Check of 465: order += keys(H) minus hole."""
from __future__ import annotations

from pathlib import Path

import _audit465_bridge as M

SRC = Path("_audit465_bridge.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def loop_bridge(" not in src:
        f.append("1. loop_bridge missing")
    if "order = list(order) + [x for x in g[\"place_keys\"][H] if x != hole]" not in src:
        f.append("1. hole still appended")
    if "from _audit458_keycut import FAM" in src:
        f.append("1. old FAM leaked")
    if "wikitext" in src or "MIX =" in src:
        f.append("1. D-track leaked")
    if "no[\"BOTH\"][\"mean_hops\"] > 2.0" not in src:
        f.append("3. GATE missing ablation")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("append hole",
     "        order = list(order) + [x for x in g[\"place_keys\"][H] if x != hole]",
     "        order = list(order) + [hole]",
     "1."),
    ("gate drops ablation",
     '    gate = (not void) and match_455(yes) and (no["BOTH"]["mean_hops"] > 2.0)',
     "    gate = (not void) and match_455(yes)",
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
