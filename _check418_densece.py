"""Check of 418: CE on 417 dense pins. No torch run required for source props.

  1. Teacher is 417 dense_labels / step_of — not 416 R, not cut.
  2. Loss is -sum y log_softmax(scores) over places+REFUSE.
  3. Hole token not in the query key (uses 417 window_keys).
  4. No REACH_CANDS / gate_walk_only / vocab CE in the train loop.
  5. GATE declared: mind_pin > random_pin; refuse_df1 > refuse_df2.
  6. Lab hole-seeds are not in the loss (only optional --lab-probe after freeze).

    python _check418_densece.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit418_densece.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "D417.step_of" not in src and "dense_labels" not in src:
        f.append("1. does not use 417 step / dense labels")
    if "D417.step_of" not in src:
        f.append("1. step_of from 417 missing")
    if 'sp["y"]' not in src or "log_softmax" not in src:
        f.append("2. CE is not -sum y log_softmax")
    if "sp[\"R\"]" in src or "expected reward" in src.lower():
        f.append("2. 416 reward term leaked into CE file")
    if "window_keys" not in src and "st[\"keys\"]" not in src:
        f.append("3. query is not from window keys")
    if "hash_fp([f\"K:{k}\" for k in st[\"keys\"]" not in src:
        f.append("3. query key construction missing")
    for ban in ("REACH_CANDS", "gate_walk_only", "reach_loss", "n_vocab", "CrossEntropyLoss"):
        # CrossEntropyLoss as nn module over vocab would be bad; our manual CE is fine
        if ban == "CrossEntropyLoss" and ban in src:
            f.append(f"4. {ban} present")
        elif ban != "CrossEntropyLoss" and ban in src:
            f.append(f"4. lab artifact {ban}")
    if "gate_mind_beats_random" not in src or "gate_refuse_one_gt_ge2" not in src:
        f.append("5. declared gates missing")
    if "mind_pin" not in src or "random_pin" not in src:
        f.append("5. pin metrics missing")
    if "mind_hit > rnd_hit" not in src:
        f.append("5. mind>random comparison missing from gate")
    if "--lab-probe" not in src:
        f.append("6. lab probe flag missing")
    # train loop must not call lab probe
    train = src[src.find("for step in range"): src.find("exam_rng")]
    if "lab_probe" in train or "reach_question" in train:
        f.append("6. lab hole entered the train loop")
    return f


MUTANTS = (
    ("416 reward instead of CE",
     "        logp = F.log_softmax(logits, 0)\n"
     "        loss = -(sp[\"y\"] * logp).sum()",
     "        logp = F.log_softmax(logits, 0)\n"
     "        loss = -(sp[\"R\"] * logp).sum()  # expected reward\n",
     "2."),
    ("lab artifact",
     'OUT = Path("results/_stage418_densece.json")',
     'OUT = Path("results/_stage418_densece.json")\nREACH_CANDS = 8\ngate_walk_only = True',
     "4."),
    ("gate dropped",
     '        "gate_mind_beats_random": bool(\n'
     '            tot["n"] and mind_hit == mind_hit and rnd_hit == rnd_hit\n'
     '            and mind_hit > rnd_hit),',
     '        "gate_mind_beats_random": bool(True),',
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
