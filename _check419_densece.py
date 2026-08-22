"""Check of 419: CE on 417h honest dense pins. No torch run required for source props.

  1. Teacher is 417h step_of / dense_labels (joint), not 417 OR, not 416 R, not cut.
  2. Loss is -sum y log_softmax(scores) over places+REFUSE.
  3. Hole token not in the query key (uses 417h window_keys / st keys).
  4. No REACH_CANDS / gate_walk_only / vocab CE in the train loop.
  5. GATE declared: mind_pin > random_pin; refuse_df1 > refuse_df2.
  6. Lab hole-seeds are not in the loss (only optional --lab-probe after freeze).
  7. Thin steps skipped; joint default 2 (not one-word OR).
  8. Standing JSON fields: mind_live, always_refuse, refuse_df1, refuse_df2.

    python _check419_densece.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("_audit419_densece.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "D417h.step_of" not in src:
        f.append("1. step_of from 417h missing")
    if "place_bags" not in src:
        f.append("1. place_bags from 417h missing")
    if "by_ctx_of" in src or "D417.step_of" in src:
        f.append("1. 417 one-word OR leaked into 419")
    if 'sp["y"]' not in src or "log_softmax" not in src:
        f.append("2. CE is not -sum y log_softmax")
    if 'sp["R"]' in src or "expected reward" in src.lower():
        f.append("2. 416 reward term leaked into CE file")
    if 'st["keys"]' not in src:
        f.append("3. query is not from window keys")
    if 'hash_fp([f"K:{k}" for k in st["keys"]' not in src:
        f.append("3. query key construction missing")
    for ban in ("REACH_CANDS", "gate_walk_only", "reach_loss", "n_vocab"):
        if ban in src:
            f.append(f"4. lab artifact {ban}")
    if "CrossEntropyLoss" in src:
        f.append("4. CrossEntropyLoss present")
    if "gate_mind_beats_random" not in src or "gate_refuse_one_gt_ge2" not in src:
        f.append("5. declared gates missing")
    if "mind_pin" not in src or "random_pin" not in src:
        f.append("5. pin metrics missing")
    if "mind_hit > rnd_hit" not in src:
        f.append("5. mind>random comparison missing from gate")
    if "--lab-probe" not in src:
        f.append("6. lab probe flag missing")
    train = src[src.find("for step in range"): src.find("exam_rng")]
    if "lab_probe" in train or "reach_question" in train:
        f.append("6. lab hole entered the train loop")
    if 'st.get("thin")' not in src:
        f.append("7. thin steps not skipped")
    if "--joint" not in src or "default=2" not in src:
        f.append("7. joint default is not 2")
    if "ce_dense_labels_417h" not in src:
        f.append("7. loss tag is not 417h")
    for need in ('"mind_live": mind_live', '"always_refuse": always_refuse',
                 '"refuse_df1": r1', '"refuse_df2": r2'):
        if need not in src:
            f.append(f"8. standing field missing: {need}")
    if '"standing": True' not in src:
        f.append("8. standing freeze flag missing")
    return f


MUTANTS = (
    ("416 reward instead of CE",
     "            logp = F.log_softmax(logits, 0)\n"
     "            loss = -(sp[\"y\"] * logp).sum()",
     "            logp = F.log_softmax(logits, 0)\n"
     "            loss = -(sp[\"R\"] * logp).sum()  # expected reward\n",
     "2."),
    ("lab artifact",
     'OUT = Path("results/_stage419_densece.json")',
     'OUT = Path("results/_stage419_densece.json")\nREACH_CANDS = 8\ngate_walk_only = True',
     "4."),
    ("gate dropped",
     '        "gate_mind_beats_random": bool(\n'
     '            tot["n"] and mind_hit == mind_hit and rnd_hit == rnd_hit\n'
     '            and mind_hit > rnd_hit),',
     '        "gate_mind_beats_random": bool(True),',
     "5."),
    ("joint dropped to 1",
     '    ap.add_argument("--joint", type=int, default=2)',
     '    ap.add_argument("--joint", type=int, default=1)',
     "7."),
    ("standing fields dropped",
     '        "mind_live": mind_live,\n'
     '        "always_refuse": always_refuse,\n',
     '        "mind_pin_only": mind_live,\n'
     '        "refuse_only": always_refuse,\n',
     "8."),
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
