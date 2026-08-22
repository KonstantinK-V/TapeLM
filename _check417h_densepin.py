"""Check of 417h honest dense pin — joint window teacher, not Phi.

Five properties, each mutation-verified:

  1. HOLE OUT OF KEY AND BAG. window_keys excludes stream token; overlap uses b-{token}.
  2. JOINT, NOT OR. retrieve needs overlap >= joint (default 2); they+tasty → peti/basket.
  3. dense_labels: |cands|+1, last=REFUSE, sums to 1; mass on hit places only.
  4. NO 289 HOLE / vocab CE (teacher is places+REFUSE). Phi out of scope.
  5. VOID/THIN/GATE/REFUSE: live<=0.05 on non-thin; thin skipped; ora-rnd>0.05; df1>df2.

    python _check417h_densepin.py
"""
from __future__ import annotations

from pathlib import Path

import _audit390_address as A
import _audit417h_densepin as M

SRC = Path("_audit417h_densepin.py")


def _pad(k):
    return " " + " ".join(f"p{k}x{j}" for j in range(24))


# they+tasty on the trees hole must jointly hit peti/basket bags (joint=2).
DESIGNED = [
    "peti has they tasty APPLES at home now" + _pad(0),
    "peti has they tasty APPLES at home two" + _pad(1),
    "basket holds they tasty APPLES today yes" + _pad(2),
    "basket holds they tasty APPLES today yes" + _pad(3),
    "trees grow APPLES they tasty more here" + _pad(4),
    "trees grow ORANGES they tasty more here" + _pad(5),
]


def designed_step():
    T = A.build_tape(DESIGNED, frame_max=3, min_fillers=1)
    hide = next(s for s, t in enumerate(T["toks"])
                if t == "APPLES" and T["owner"][s] == 4)
    bags = M.place_bags(T)
    st = M.step_of(T, bags, hide, cap=8, joint=2, min_keys=4)
    return T, hide, st, bags


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T, hide, st, bags = designed_step()
    if st is None or st.get("thin"):
        f.append("1. designed step is None/thin")
        return f
    if T["toks"][hide] != "APPLES":
        f.append("1. designed hole is not APPLES")
    if "APPLES" in st["keys"]:
        f.append("1. the hole APPLES entered the keys")
    if "if v == hole or v in seen:" not in src:
        f.append("1. window_keys does not exclude the hole")
    if "b - {token}" not in src and "(b - {token})" not in src:
        f.append("1. overlap does not subtract stream token from bag")

    if "they" not in st["keys"] or "tasty" not in st["keys"]:
        f.append("2. they/tasty missing from designed window keys")
    # peti lines 0–1, basket 2–3: joint they+tasty must land there (not only trees)
    hit_lines = set()
    for j in st["hits"]:
        hit_lines |= {T["owner"][x] for x in T["places"][j]}
    if not (hit_lines & {0, 1, 2, 3}):
        f.append("2. designed they+tasty did not reach peti/basket lines")
    if not st["hits"] or any(T["toks"][hide] not in M.fillers_place(T, j) for j in st["hits"]):
        f.append("2. designed hits do not carry the stream literal")
    if "ov >= joint" not in src:
        f.append("2. retrieve_joint is not overlap >= joint")
    if "default=2" not in src or "--joint" not in src:
        f.append("2. joint default is not 2")
    if "by_ctx_of" in src or "for tok in set(L) | set(R):" in src:
        f.append("2. single-token OR index leaked back (417 style)")
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
    if 'if st["thin"]:' not in src or "thin" not in src:
        f.append("5. THIN steps are not skipped out of live")
    if 'rep["oracle_minus_random"] > 0.05' not in src:
        f.append("5. GATE is not oracle - random > 0.05")
    if "r1 > r2" not in src:
        f.append("5. refuse check is not df1 > df>=2")
    if "min_keys" not in src or "default=4" not in src:
        f.append("5. min_keys default is not 4")
    return f


MUTANTS = (
    ("hole enters keys",
     "        if v == hole or v in seen:",
     "        if v in seen:",
     "1."),
    ("joint dropped to OR",
     "        if ov >= joint:",
     "        if ov >= 1:",
     "2."),
    ("refuse mass broken",
     "        y[-1] = 1.0",
     "        y[-1] = 0.0",
     "3."),
    ("289 hole leaked into teacher",
     'OUT = Path("results/_stage417h_densepin.json")',
     'OUT = Path("results/_stage417h_densepin.json")\nREACH_CANDS = 8\ngate_walk_only = True',
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
