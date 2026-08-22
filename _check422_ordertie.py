"""Check of 422 order-tie ceiling. No Phi. Four mutants.

  1. Ties measured on max bag score; bag pick among tied is min(j).
  2. Ordered breaks only inside the tied set (pick_argmax(tied)).
  3. VOID bar tie_rate <= 0.05; GO bar Δ > 0.05 (fixed).
  4. unique_agree recorded, not a gate; no torch/REACH.

    python _check422_ordertie.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit422_ordertie.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "D417h.step_of" not in src:
        f.append("1. 417h step_of missing")
    if "scores[j] == mx" not in src and "scores[j]==mx" not in src:
        f.append("1. tie set is not max bag")
    if "j_bag = min(tied)" not in src:
        f.append("1. bag pick among tied is not min(j)")
    if "pick_argmax(tied" not in src:
        f.append("2. ordered does not pick inside tied set")
    if "pick_argmax(cands" in src[src.find("n_tie"): src.find("unique_agree")]:
        # after n_tie += 1, must not use full cands for ordered
        block = src[src.find("n_tie += 1"): src.find("bag_tie_h")]
        if "pick_argmax(cands" in block:
            f.append("2. ordered pick uses full cands on ties")
    if 'or rep["tie_rate"] <= 0.05' not in src:
        f.append("3. VOID tie_rate <= 0.05 missing")
    if 'rep["ordered_minus_bag_tie"] > 0.05' not in src:
        f.append("3. GO Δ > 0.05 missing")
    if "unique_agree" not in src:
        f.append("4. unique_agree missing")
    if "not gate" not in src:
        f.append("4. unique_agree not marked non-gate")
    for ban in ("REACH_CANDS", "gate_walk_only", "CrossEntropyLoss", "n_vocab"):
        if ban in src:
            f.append(f"4. artifact {ban}")
    if "import torch" in src:
        f.append("4. torch imported")
    if "No Phi" not in src:
        f.append("4. file does not say No Phi")
    return f


MUTANTS = (
    ("bag pick not min",
     "        j_bag = min(tied)  # bag cannot break; deterministic among tied",
     "        j_bag = max(tied)  # flipped",
     "1."),
    ("ordered on full cands",
     "        j_ord = C421.pick_argmax(tied, sc_ord)",
     "        j_ord = C421.pick_argmax(cands, sc_ord)",
     "2."),
    ("tie void bar dropped",
     '    void = rep["n_live"] == 0 or rep["teacher_live"] <= 0.05 or rep["tie_rate"] <= 0.05',
     '    void = rep["n_live"] == 0 or rep["teacher_live"] <= 0.05 or rep["tie_rate"] <= -1.0',
     "3."),
    ("go bar dropped",
     '    go = (not void) and rep["n_tie"] > 0 and rep["ordered_minus_bag_tie"] > 0.05',
     '    go = (not void) and rep["n_tie"] > 0 and rep["ordered_minus_bag_tie"] > -1.0',
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
