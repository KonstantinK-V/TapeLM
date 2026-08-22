"""Check of 417 dense pin — teacher ceiling, not Phi. Designed peti/basket/trees.

Five properties, each mutation-verified:

  1. THE HOLE IS NOT A KEY. window_keys excludes the stream token.
  2. INDEX IS LEFT/RIGHT of the address, not filler w / vocab.
  3. dense_labels: |cands|+1, last=REFUSE, sums to 1; mass on hit places only.
  4. NO 289 HOLE / vocab CE in the audit (teacher is places+REFUSE).
  5. VOID/GATE/REFUSE thresholds are the declared ones (live<=0.05; ora-rnd>0.05; df1>df2).

    python _check417_densepin.py
"""
from __future__ import annotations

from pathlib import Path

import _audit390_address as A
import _audit417_densepin as M

SRC = Path("_audit417_densepin.py")


def _pad(k):
    return " " + " ".join(f"p{k}x{j}" for j in range(24))


DESIGNED = [
    "peti has tasty APPLES at home now" + _pad(0),
    "peti has tasty APPLES at home two" + _pad(1),
    "basket holds tasty APPLES today yes" + _pad(2),
    "basket holds tasty APPLES today yes" + _pad(3),
    "trees grow APPLES they tasty" + _pad(4),
    "trees grow ORANGES they tasty" + _pad(5),
]


def designed_step():
    T = A.build_tape(DESIGNED, frame_max=3, min_fillers=1)
    hide = next(s for s, t in enumerate(T["toks"])
                if t == "APPLES" and T["owner"][s] == 4)
    ix = M.by_ctx_of(T)
    st = M.step_of(T, ix, hide, cap=8)
    return T, hide, st


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T, hide, st = designed_step()
    if st is None:
        f.append("1. designed step is None")
        return f
    if T["toks"][hide] != "APPLES":
        f.append("1. designed hole is not APPLES")
    if "APPLES" in st["keys"]:
        f.append("1. the hole APPLES entered the keys")
    if "tasty" not in st["keys"]:
        f.append("1. tasty not in window keys on designed trees line")
    if "if v == hole or v in seen:" not in src:
        f.append("1. window_keys does not exclude the hole")

    if "for tok in set(L) | set(R):" not in src:
        f.append("2. by_ctx_of is not left/right")
    if "_w, L, R" not in src:
        f.append("2. filler w is not unpacked aside")

    y = st["y"]
    if len(y) != len(st["cands"]) + 1:
        f.append(f"3. dense_labels length {len(y)} != |cands|+1")
    if abs(sum(y) - 1.0) > 1e-9:
        f.append(f"3. dense_labels do not sum to 1: {sum(y)}")
    if st["hits"]:
        if y[-1] != 0.0:
            f.append("3. REFUSE mass non-zero when hits exist")
    else:
        if y[-1] != 1.0:
            f.append("3. REFUSE is not 1 when no hits")
    if "mass = 1.0 / len(hits)" not in src or "y[-1] = 1.0" not in src:
        f.append("3. dense_labels formula missing")

    for ban in ("reach_loss", "REACH_CANDS", "CrossEntropyLoss", "n_vocab", "gate_walk_only"):
        if ban in src:
            f.append(f"4. lab/vocab artifact in teacher file: {ban}")
    if "Phi is not" not in src:
        f.append("4. file does not declare Phi is out of scope")

    if 'rep["live"] <= 0.05' not in src:
        f.append("5. VOID threshold is not live <= 0.05")
    if 'rep["oracle_minus_random"] > 0.05' not in src:
        f.append("5. GATE is not oracle - random > 0.05")
    if "r1 > r2" not in src:
        f.append("5. refuse check is not df1 > df>=2")
    return f


MUTANTS = (
    ("hole enters keys",
     "        if v == hole or v in seen:",
     "        if v in seen:",
     "1."),
    ("index includes filler w",
     "        for tok in set(L) | set(R):",
     "        for tok in set(L) | set(R) | {_w}:",
     "2."),
    ("refuse mass broken",
     "        y[-1] = 1.0",
     "        y[-1] = 0.0",
     "3."),
    ("289 hole leaked into teacher",
     'OUT = Path("results/_stage417_densepin.json")',
     'OUT = Path("results/_stage417_densepin.json")\nREACH_CANDS = 8\ngate_walk_only = True',
     "4."),
    ("gate threshold dropped",
     '    gate = (not void) and rep["oracle_minus_random"] > 0.05',
     '    gate = (not void) and rep["oracle_minus_random"] > -1.0',
     "5."),
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
