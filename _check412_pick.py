"""Check of 412: who teaches the pick. No torch, no corpus - properties on the source.

`loss_for` reads OBJECTIVE only on the lookup path, so the reach pick has never had a
cross-entropy option: every reach number was trained by one scalar per question. These are the
ways a race between the two teachers can be unfair without looking unfair:

  1. OFF IS OFF. `stage` is the default and returns None, so no earlier run moves by a digit.
  2. THE ROUTER IS DETACHED IN BOTH COMPARED ARMS, and in neither under `stage`. If only one arm
     detached, the difference would be the router as much as the teacher.
  3. THE TERM IS ADDED AT BOTH RETURN POINTS - `--two-way` returns early, and a term on one path
     only would run the control on the standing arm.
  4. CE TARGETS REFUSE WHERE THE OFFER MISSES THE TRUTH. Without it CE trains on the ~12% of
     questions whose offer holds the truth while the payoff trains on all of them, and the
     comparison becomes one of coverage rather than of teachers.
  5. THE CONTROL IS THE SAME PAYOFF. `reward` uses R2/R1, not a re-derived one.
  6. THE NAMES ARE CUT TO THE LOGITS. `reach_names` appends REFUSE to both stages, and an index
     taken past the end of a shorter logit vector would silently teach the wrong world.
  7. IT IS IN THE ARM SIGNATURE and the live counters are reported.

    python _check412_pick.py
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("_stage289_derivation.py")


def code(src, name):
    m = re.search(rf"^def {name}\(.*?(?=\ndef )", src, re.S | re.M)
    return re.sub(r'"""(?:.|\n)*?"""', "", m.group(0)) if m else ""


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    pt, rl = code(src, "pick_term"), code(src, "reach_loss")

    if not re.search(r'^PICK_TEACHER = "stage"', src, re.M):
        f.append("1. PICK_TEACHER does not default to `stage`")
    if 'if PICK_TEACHER == "stage":\n        return None' not in pt:
        f.append("1. `stage` does not return None, so the default path is not the old one")

    for want in ("p2.detach() if _pk is not None else p2",
                 "so.detach() if _pk is not None else so"):
        if want not in rl:
            f.append(f"2. the router's value is not detached exactly when a pick term exists: "
                     f"{want!r} missing")
    got = len(re.findall(r"return -\(out if _pk is None else out \+ _pk\)", rl))
    if got != 2:
        f.append(f"3. the pick term is on {got} of the 2 return points")

    if 'head.index(REFUSE_LABEL) if REFUSE_LABEL in head else -1' not in pt:
        f.append("4. CE has no REFUSE target, so the two arms train on different populations")
    if "torch.log_softmax(lg, 0)[idx]" not in pt:
        f.append("4. the CE term is not the log-probability of the target")
    if "R2[:len(l2)]" not in pt or "R1[:len(lo)]" not in pt:
        f.append("5. the control does not use the stage's own payoff, cut to the logits")
    if "head = names[:len(lg)]" not in pt:
        f.append("6. the names are not cut to the logits - an index past the end would teach "
                 "another world")
    # Under reach-depth>1 the deep max is on l2 before pick_term; R2 must already hold v3 or the
    # reward arm crashes (9 vs 8) while CE silently survives - a false race.
    cat = rl.find("R2 = torch.cat([R2, v3.reshape(1)])")
    call = rl.find("_pk = pick_term(")
    if cat < 0 or call < 0 or cat > call:
        f.append("5. R2 is not extended with the deep value before pick_term")

    sig = src[src.find("# 341 IS IN THE SIGNATURE"):][:1000]
    if '"pick_teacher": PICK_TEACHER' not in sig:
        f.append("7. pick_teacher is not in the arm signature")
    for k in ("pick_live", "pick_target"):
        if f'"{k}"' not in src:
            f.append(f"7. {k} is not reported")
    return f


MUTANTS = (
    ("the default stops being the old path", 'PICK_TEACHER = "stage"', 'PICK_TEACHER = "ce"', "1."),
    ("only one arm detaches the router",
     "    v2 = REACH_GAMMA * ((p2.detach() if _pk is not None else p2) * R2).sum() - STEP_COST",
     "    v2 = REACH_GAMMA * (p2 * R2).sum() - STEP_COST", "2."),
    ("the term is dropped from the two-way return",
     "        out = out if mv is None else out + mv\n"
     "        return -(out if _pk is None else out + _pk)",
     "        return -(out if mv is None else out + mv)", "3."),
    ("CE loses its refusal target",
     "        idx = head.index(tv) if tv in head else (\n"
     "            head.index(REFUSE_LABEL) if REFUSE_LABEL in head else -1)",
     "        idx = head.index(tv) if tv in head else -1", "4."),
    ("the names are not cut to the logits",
     "        head = names[:len(lg)]", "        head = list(names)", "6."),
    ("not in the arm signature",
     '                "move_teach": MOVE_TEACH, "route_on": ROUTE_ON,\n'
     '                "pick_teacher": PICK_TEACHER,',
     '                "move_teach": MOVE_TEACH, "route_on": ROUTE_ON,', "7."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        if not any(g.startswith(tag) for g in props(src.replace(old, new, 1))):
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
