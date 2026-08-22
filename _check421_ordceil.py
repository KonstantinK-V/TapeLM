"""Check of 421 ordered-match ceiling. No torch.

  1. Hole token never in W_L/W_R.
  2. bag = 417h ov; ceiling uses D417h.step_of.
  3. ordered uses address L,R + window sides — not fillers_place as feature.
  4. Gate bars > 0.05 for ordered-bag and ordered-random; no torch/REACH.

    python _check421_ordceil.py
"""
from __future__ import annotations

from pathlib import Path

import _audit390_address as A
import _audit417h_densepin as D417h
import _audit421_ordceil as M

SRC = Path("_audit421_ordceil.py")


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
    bags = D417h.place_bags(T)
    st = D417h.step_of(T, bags, hide, cap=8, joint=2, min_keys=4)
    return T, hide, st, bags


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T, hide, st, bags = designed()
    if st is None or st.get("thin"):
        f.append("1. designed step thin/None")
        return f
    W_L, W_R = M.window_sides(T, hide)
    hole = T["toks"][hide]
    if hole in W_L or hole in W_R:
        f.append("1. hole entered window sides")
    if "if v == hole:" not in src and "v == hole" not in src:
        f.append("1. window_sides does not skip hole")

    if "D417h.step_of" not in src:
        f.append("2. does not call D417h.step_of")
    if "bags[j] - {token}" not in src and "(bags[j] - {token})" not in src:
        f.append("2. bag score is not 417h ov")
    bs = src.find("def bag_score")
    be = src.find("\ndef ", bs + 1) if bs >= 0 else -1
    bag_body = src[bs:be] if bs >= 0 else ""
    if "fillers_place" in bag_body:
        f.append("2. fillers_place leaked into bag_score")
    if st["cands"]:
        j0 = st["cands"][0]
        ov = M.bag_score(bags, st["keys"], st["token"], j0)
        expect = len((bags[j0] - {st["token"]}) & set(st["keys"]))
        if ov != expect:
            f.append("2. bag_score mismatch vs 417h ov")

    if "T[\"addrs\"][j]" not in src and "T['addrs'][j]" not in src:
        f.append("3. ordered does not read place address L,R")
    if "order_agree" not in src or "window_sides" not in src:
        f.append("3. ordered machinery missing")
    if "fillers_place(" in src and "ordered_score" in src:
        # fillers_place may appear only if leaked into ordered — ban call inside ordered path
        pass
    # forbid fillers_place in ordered_score body
    start = src.find("def ordered_score")
    end = src.find("\ndef ", start + 1)
    body = src[start:end] if start >= 0 else ""
    if "fillers_place" in body:
        f.append("3. fillers_place leaked into ordered_score")
    for ban in ("torch", "REACH_CANDS", "gate_walk_only", "n_vocab", "CrossEntropyLoss"):
        if ban in src:
            f.append(f"4. lab/torch artifact {ban}")
    if "ordered_minus_bag\"] > 0.05" not in src and '["ordered_minus_bag"] > 0.05' not in src:
        if 'rep["ordered_minus_bag"] > 0.05' not in src:
            f.append("4. ordered-bag gate bar missing")
    if 'rep["ordered_minus_random"] > 0.05' not in src:
        f.append("4. ordered-random gate bar missing")
    if "Phi is not" not in src and "No Phi" not in src:
        f.append("4. file does not declare No Phi")
    return f


MUTANTS = (
    ("hole enters window sides",
     "        if v == hole:\n            continue",
     "        if False:\n            continue",
     "1."),
    ("bag becomes ordered-only leak",
     "    return len((bags[j] - {token}) & set(keys))",
     "    return len((bags[j] - {token}) & set(keys)) + len(D417h.fillers_place(T, j))",
     "2."),
    ("fillers_place in ordered",
     "def ordered_score(T, bags, keys, token, s, j):\n"
     "    W_L, W_R = window_sides(T, s)\n"
     "    _w, L, R = T[\"addrs\"][j]",
     "def ordered_score(T, bags, keys, token, s, j):\n"
     "    _ = D417h.fillers_place(T, j)\n"
     "    W_L, W_R = window_sides(T, s)\n"
     "    _w, L, R = T[\"addrs\"][j]",
     "3."),
    ("gate bar dropped",
     '    go = (not void) and rep["ordered_minus_bag"] > 0.05 and rep["ordered_minus_random"] > 0.05',
     '    go = (not void) and rep["ordered_minus_bag"] > -1.0 and rep["ordered_minus_random"] > -1.0',
     "4."),
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
        # bag mutant needs T in bag_score — that breaks signature; catch via fillers_place in src prop 2
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
