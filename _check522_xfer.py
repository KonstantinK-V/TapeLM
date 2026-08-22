"""Check of 522: home k from 70% train, 519 walk on foreign 30% tail."""
from __future__ import annotations

from pathlib import Path

import _audit522_xfer as M

SRC = Path("_audit522_xfer.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit519_highcap import eval_cap" not in src:
        f.append("1. 519 eval_cap reuse missing")
    if "from _audit518_reldf import pct_band" not in src:
        f.append("1. 518 pct_band missing")
    if "cut = int(0.7 * len(all_lines))" not in src:
        f.append("1. 70/30 home/foreign split missing")
    if "home_pool = all_lines[:cut]" not in src or "fore_pool = all_lines[cut:]" not in src:
        f.append("1. home vs foreign pools missing")
    if "k = 200.0 / max(g400[\"n\"]" not in src:
        f.append("1. k from home 400 missing")
    if "SIZES = (100, 400, 1200, 2400)" not in src:
        f.append("1. nested foreign curve missing")
    if "base[: min(n, len(base))]" not in src:
        f.append("1. nested prefix on foreign missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'm2400.get("d2", 0) > m100.get("d2", 0) + 1' not in gate:
        f.append("2. GATE missing mid d2 growth +1")
    if 'h2400.get("d2", 99) < 1.0' not in gate:
        f.append("2. GATE missing high d2 < 1")
    if 'm100.get("n", 0) < 10' not in src or 'm2400.get("n", 0) < 20' not in src:
        f.append("2. VOID missing")
    if "pick_corpus" not in src:
        f.append("3. corpus pick missing")
    if "import torch" in src or "PickNet" in src or "score_w" in src:
        f.append("4. Phi/W leaked")
    return f


MUTANTS = (
    ("gate growth only",
     "    gate = (not void) and (m2400.get(\"d2\", 0) > m100.get(\"d2\", 0) + 1) and (\n"
     "        h2400.get(\"d2\", 99) < 1.0)",
     "    gate = (not void) and (m2400.get(\"d2\", 0) > m100.get(\"d2\", 0) + 1)",
     "2."),
    ("same pool",
     "    fore_pool = all_lines[cut:][: args.lines]",
     "    fore_pool = home_pool",
     "1."),
    ("k on foreign",
     "    k = 200.0 / max(g400[\"n\"], 1)",
     "    k = 200.0 / max(g[\"n\"], 1)",
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
