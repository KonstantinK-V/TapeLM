"""Check of 463 TRACK R: trap_of, teacher without by_key[value]."""
from __future__ import annotations

from pathlib import Path

import _audit463_trap as M

SRC = Path("_audit463_trap.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def trap_of" not in src or "return (d1, tr, s)" not in src:
        f.append("1. trap sig missing")
    if "def teacher_trap" not in src:
        f.append("1. teacher_trap missing")
    teach = src[src.find("def teacher_trap"):src.find("def train")]
    if "by_key" in teach:
        f.append("2. teacher still uses by_key[value]")
    if "wikitext" in src or "MIX =" in src:
        f.append("1. D-track leaked")
    if 'no["BOTH"]["mean_hops"] > 2.0' not in src:
        f.append("3. GATE missing ablation BOTH")
    if "import torch" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no trap",
     "def trap_of(cands, hole, place_keys):",
     "def trap_count(cands, hole, place_keys):",
     "1."),
    ("teacher by_key",
     "def teacher_trap(o, cands, g):\n"
     "    for rec in o:\n"
     "        H, pinH = rec[1], rec[2]\n"
     "        if len(filter_keys(cands, H, g[\"place_keys\"], g[\"value\"][pinH])) == 1:\n"
     "            return rec\n"
     "    for rec in o:\n"
     "        pinH = rec[2]\n"
     "        if trap_of(cands, g[\"value\"][pinH], g[\"place_keys\"]) == 0:\n"
     "            return rec\n"
     "    return None",
     "def teacher_trap(o, cands, g):\n"
     "    for rec in o:\n"
     "        pinH, vis1 = rec[2], rec[3]\n"
     "        nxt = g[\"by_key\"].get(g[\"value\"][pinH], ())\n"
     "        if any(h not in vis1 for h in nxt):\n"
     "            return rec\n"
     "    return None",
     "2."),
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
