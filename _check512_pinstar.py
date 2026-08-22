"""Check of 512: 436 pin walk vs random token. Separate JSON from 513."""
from __future__ import annotations

from pathlib import Path

import _audit512_pinstar as M

SRC = Path("_audit512_pinstar.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit511_ring import mentions, walk" not in src:
        f.append("1. 511 walk reuse missing")
    if "def const_pins(" not in src or "CAP" not in src:
        f.append("1. 436 const pin missing")
    if "rng2.choice(vocab)" not in src:
        f.append("1. random token control missing")
    if "pin_rep = mean_walk(g, by, pins, cache)" not in src:
        f.append("1. pin walk must use pins list")
    if "pick_by_q" in src or "def train(" in src or "Q[" in src or "\nQ =" in src:
        f.append("1. Q leaked into 512")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if 'pin_rep["d2"] > rnd_rep["d2"]' not in gate:
        f.append("2. GATE missing pin d2 > rnd d2")
    if "pin_rep[\"m2\"] - rnd_rep[\"m2\"] > 0.05" not in gate:
        f.append("2. GATE missing pin m2 delta > 0.05")
    if "_stage513" in src:
        f.append("3. must not mix 513 JSON")
    if 'pin_rep["n"] < 30' not in src:
        f.append("2. VOID pins < 30 missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("gate d2 only",
     "    gate = (not void) and (pin_rep[\"d2\"] > rnd_rep[\"d2\"]) and (\n"
     "        pin_rep[\"m2\"] - rnd_rep[\"m2\"] > 0.05)",
     "    gate = (not void) and (pin_rep[\"d2\"] > rnd_rep[\"d2\"])",
     "2."),
    ("no pin walk",
     "    pin_rep = mean_walk(g, by, pins, cache)",
     "    pin_rep = mean_walk(g, by, rnd, cache)",
     "1."),
    ("Q leaked",
     "OUT = Path(\"results/_stage512_pinstar.json\")",
     "OUT = Path(\"results/_stage512_pinstar.json\")\nQ = defaultdict(float)",
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
