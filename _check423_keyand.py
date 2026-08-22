"""Check of 423 key AND. Formation, not Phi. Three mutants.

  1. Designed they∩tasty → APPLES places (AND of two key traces).
  2. new_vs_A is 0 by construction (AND ⊆ A).
  3. Gate is and_minus_single > 0.05; single = better of A/B; hapax excluded.

    python _check423_keyand.py
"""
from __future__ import annotations

from pathlib import Path

import _audit390_address as A
import _audit417h_densepin as H
import _audit423_keyand as M

SRC = Path("_audit423_keyand.py")


def _pad(k):
    return " " + " ".join(f"p{k}x{j}" for j in range(24))


DESIGNED = [
    "peti has they tasty APPLES at home now" + _pad(0),
    "peti has they tasty APPLES at home two" + _pad(1),
    "basket holds they tasty APPLES today yes" + _pad(2),
    "basket holds they tasty APPLES today yes" + _pad(3),
    "trees grow APPLES they tasty more here" + _pad(4),
    "trees grow ORANGES they tasty more here" + _pad(5),
]


def designed():
    T = A.build_tape(DESIGNED, frame_max=3, min_fillers=1)
    hide = next(s for s, t in enumerate(T["toks"])
                if t == "APPLES" and T["owner"][s] == 4)
    bags = H.place_bags(T)
    st = H.step_of(T, bags, hide, cap=8, joint=2, min_keys=4)
    return T, hide, st, bags


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T, hide, st, bags = designed()
    if st is None or st.get("thin"):
        f.append("1. designed step thin/None")
        return f
    pair = M.two_keys(T, st["keys"])
    if pair is None:
        f.append("1. designed two_keys failed")
        return f
    ka, kb = pair
    # Force they/tasty path when both eligible — designed must include both
    if "they" not in st["keys"] or "tasty" not in st["keys"]:
        f.append("1. they/tasty not in designed keys")
    drop = set(T["on_line"][T["owner"][hide]])
    ok = M.eligible(T, st["qpid"], drop)
    PA = M.places_for_key(bags, "they", ok)
    PB = M.places_for_key(bags, "tasty", ok)
    PAND = PA & PB
    if not PAND:
        f.append("1. designed they∩tasty is empty")
    else:
        hit = any("APPLES" in H.fillers_place(T, j) for j in PAND)
        if not hit:
            f.append("1. designed they∩tasty does not reach APPLES places")
    if "PAND = PA & PB" not in src:
        f.append("1. AND is not set intersection")

    if '"new_vs_A": 0.0' not in src:
        f.append("2. new_vs_A is not fixed at 0")
    if "AND ⊆" not in src and "0 by construction" not in src:
        f.append("2. AND ⊆ A / construction note missing")

    if 'rep["and_minus_single"] > 0.05' not in src:
        f.append("3. gate and_minus_single > 0.05 missing")
    if "max((x for x, sz in singles" not in src and "max((x for x, sz in singles if x is not None)" not in src:
        if "lsng = max(" not in src:
            f.append("3. single is not the better of A/B")
    if 'T["freq"].get(v, 0) >= 2' not in src:
        f.append("3. hapax not excluded from two_keys")
    for ban in ("REACH_CANDS", "gate_walk_only", "n_vocab", "CrossEntropyLoss", "import torch"):
        if ban in src:
            f.append(f"3. artifact {ban}")
    if "Not composition" not in src and "not composition" not in src and "Not 390" not in src:
        f.append("3. file does not reject 390 composition")
    return f


MUTANTS = (
    ("AND becomes OR",
     "        PAND = PA & PB",
     "        PAND = PA | PB",
     "1."),
    ("new_vs_A nonzero",
     '        "new_vs_A": 0.0,',
     '        "new_vs_A": 1.0,',
     "2."),
    ("gate bar dropped",
     '    gate = (not void) and rep["and_minus_single"] > 0.05',
     '    gate = (not void) and rep["and_minus_single"] > -1.0',
     "3."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props(src)
    caught = 0
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor occurs {src.count(old)} times")
            continue
        mut = src.replace(old, new, 1)
        hit = [x for x in props(mut) if x.startswith(tag)]
        if not hit:
            fails.append(f"mutant not caught: {name}")
        else:
            caught += 1
    if fails:
        print("FAIL")
        for x in fails:
            print(" ", x)
        return 1
    print(f"all properties hold, and all {caught} re-introduced failures were caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
