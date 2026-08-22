"""Check of 523: mix vs con stars by jacc. GATE mix extra flat, con extra > 0.05."""
from __future__ import annotations

from pathlib import Path

import _audit523_disagree as M

SRC = Path("_audit523_disagree.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit511_ring import cheap_rec, graph, mentions, pick_corpus" not in src:
        f.append("1. 511 reuse missing")
    if "from _audit517_window import comps" not in src:
        f.append("1. 517 comps missing")
    if "from _audit518_reldf import pct_band" not in src:
        f.append("1. 518 pct_band missing")
    if "def rec_of(" not in src:
        f.append("1. enumerate hop1 per half missing")
    if "j < 0.15" not in src or "j >= 0.40" not in src:
        f.append("1. mix/con jacc bands missing")
    if "k * g[\"n\"] / max(g[\"df\"][v]" not in src:
        f.append("1. relative walk budget missing")
    if "rec_of(g, by, v, A, cache)[:1]" in src:
        f.append("1. must not pin mix rec to one hop")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'mx["extra"] <= 0.05' not in gate:
        f.append("2. GATE missing mix extra <= 0.05")
    if 'cn["extra"] > 0.05' not in gate:
        f.append("2. GATE missing con extra > 0.05")
    if "hop1" in gate or "walk" in gate:
        f.append("2. raw hop1/walk must not gate alone")
    if 'mx["n"] < 20' not in src or 'cn["n"] < 20' not in src:
        f.append("2. VOID missing")
    if "pick_corpus" not in src:
        f.append("3. corpus pick missing")
    if "import torch" in src or "PickNet" in src or "score_w" in src:
        f.append("4. Phi/W leaked")
    return f


MUTANTS = (
    ("gate con only",
     '    gate = (not void) and (mx["extra"] <= 0.05) and (cn["extra"] > 0.05)',
     '    gate = (not void) and (cn["extra"] > 0.05)',
     "2."),
    ("no jacc split",
     "        if j < 0.15:\n            mix.append(row)\n        elif j >= 0.40:\n            con.append(row)",
     "        if j < 0.99:\n            mix.append(row)\n        elif j >= 0.40:\n            con.append(row)",
     "1."),
    ("pin mix",
     "        ra = set(rec_of(g, by, v, A, cache))",
     "        ra = set(rec_of(g, by, v, A, cache)[:1])",
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
