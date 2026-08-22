"""Check of 391's move term. No torch, no corpus - the properties are read off the source.

391 is the first time the move DECISION receives a gradient. Until now `reach_logits` kept the
chosen name and threw the ballot's logits away, so 385 and 386 measured an argmax of a scorer
trained to rank final names, applied to a choice nobody had taught. The ways that can go wrong
silently are specific, and all of them print an ordinary number:

  1. THE LOGITS ARE KEPT. If the ballot is not stashed with its tensor, the term has nothing to
     differentiate and the arm is its own control under the arm's name.
  2. THE BALLOT IS REBUILT ON EVERY PASS WHEN THE TERM IS ON. A batch draws with replacement, so
     a tensor stashed once would be backward'd through a freed graph on the second draw.
  3. THE CHOICE IS NOT REMADE. The offer is cached from the first pass; a move re-chosen against
     a newer Phi would disagree with the offer it is supposed to have opened.
  4. THE TEACHER NEVER TOUCHES THE CHOICE. `answerable` per lane is the term's label; if it
     reached the argmax, the mind would be picking moves by the answer.
  5. THE TERM IS ADDED, NOT FOLDED INTO v2. Multiplying the step's value by the move probability
     would reprice the ROUTER - stepping would look worse by exactly p(move) - and one lever
     would be moving two decisions.
  6. BOTH RETURN POINTS CARRY IT. The standing arm is `--two-way`, which returns early; a term
     added only at the far return would run the control on the very arm that matters.
  7. A DEAD BALLOT CONTRIBUTES NOTHING AND IS COUNTED. Fewer than two live moves, or every lane
     reaching the same, is a constant in l0 - zero gradient by construction. The count is the
     void check of the step and must be incremented only AFTER that guard.
  8. THE SCALE IS shift_reward's. 311a: a discount on a signed scale paid the mind to leave.
  9. OFF IS OFF, at the default, bit for bit.
 10. THE FLAG IS REFUSED WITHOUT A BALLOT rather than run as its own control.
 11. IT IS IN THE ARM SIGNATURE. It changes what the mind was trained to do, so a transplant
     across it is a different mind.
 12. THE VOID CHECK IS REPORTED, or "the term taught nothing" cannot be told from "the term
     taught the wrong thing".

Every property is a WRONG NUMBER on the source, and every one of them is verified the only way a
check can be: the failure it exists to catch is RE-INTRODUCED textually at the bottom of this
file and the check must fire on it.

    python _check391_moveteach.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path("_stage289_derivation.py")


def body(src, name, fails, tag):
    m = re.search(rf"^def {name}\(.*?(?=\n(?:def |@|# ---))", src, re.S | re.M)
    if not m:
        fails.append(f"{tag} {name} not found")
        return ""
    return m.group(0)


def code_of(text):
    """the function with its docstrings removed - the prose SAYS what the code must not do"""
    return re.sub(r'"""(?:.|\n)*?"""', "", text)


def props(src):
    f = []
    mp = code_of(body(src, "reach_move_pick", f, "1."))
    rl = code_of(body(src, "reach_logits", f, "1."))
    mt = code_of(body(src, "move_term", f, "1."))
    ls = code_of(body(src, "reach_loss", f, "5."))

    # 1: the logits are kept, and the term reads them
    if '_move_ballot"] = None if _l0 is None else (_l0, _mans)' not in rl:
        f.append("1. reach_logits does not stash the ballot with its logits - the move term has "
                 "nothing to differentiate and the arm is its own control")
    if '_move_ballot")' not in mt:
        f.append("1. move_term does not read the ballot")

    # 2: rebuilt on every pass while the term is on
    if 'if MOVES_ON and ("_move" not in q or MOVE_TEACH):' not in rl:
        f.append("2. the ballot is not rebuilt when the term is on - a tensor stashed on a "
                 "question is backward'd through a freed graph when a batch draws it twice")

    # 3: the choice is made once
    if 'if "_move" not in q:\n        q["_move"] = props[int(l0.argmax())][0]' not in mp:
        f.append("3. the move is re-chosen on every pass, so a question's move can drift away "
                 "from the offer that was cached for it")

    # 4: the teacher is not in the choice. The argmax must see l0 and nothing else, and no
    # ordering may read the flags.
    for ln in mp.splitlines():
        if "argmax" in ln and ("_a" in ln.replace("_argmax", "") or "ans" in ln):
            f.append(f"4. the lane's answerability is in the choosing expression: {ln.strip()!r}")
    if "sorted(props" in mp or "props.sort" in mp:
        f.append("4. the ballot is re-ordered before the argmax - the teacher can decide the "
                 "choice through the order")

    # 5 + 6: added at both return points, never folded into v2
    if "mv = move_term(q, device)" not in ls:
        f.append("5. reach_loss does not compute the move term")
    if re.search(r"v2 = [^\n]*move", ls) or re.search(r"move[^\n]*\* *v2", ls):
        f.append("5. the move probability is folded into v2 - that reprices the router, so one "
                 "lever would be moving two decisions")
    # 412 split the return into two lines so the pick term can be added beside the move term.
    # THE PROPERTY IS UNCHANGED - the move term must be folded in on BOTH paths - and only its
    # literal moved.
    got = len(re.findall(r"out = out if mv is None else out \+ mv", ls))
    if got != 2:
        f.append(f"6. the move term is on {got} of the 2 return points - `--two-way` returns "
                 f"early, so a term added only at the far one runs the control on the standing "
                 f"arm")

    # 7: the dead ballot, and the counter after the guard
    if "if len(ans) < 2 or len(set(ans)) < 2:\n        return None" not in mt:
        f.append("7. a ballot that cannot teach is not skipped - it adds a constant and the "
                 "void check stops meaning anything")
    i_guard = mt.find("len(set(ans)) < 2")
    i_live = mt.find('_MOVE_LIVE["live"]')
    i_n = mt.find('_MOVE_LIVE["n"]')
    if not (0 < i_n < i_guard < i_live):
        f.append(f"7. the live counter is not incremented after the guard (n={i_n} "
                 f"guard={i_guard} live={i_live}) - a dead ballot would count as live and the "
                 f"void check would read healthy on an arm that taught nothing")

    # 8: the scale
    if "shift_reward(" not in mt:
        f.append("8. the move reward does not go through shift_reward - a signed scale under a "
                 "discount is what paid the mind to leave in 311a")

    # 9: off is off
    if "if not MOVE_TEACH:\n        return None" not in mt:
        f.append("9. move_term does not return None when the flag is off")
    if not re.search(r"^MOVE_TEACH = 0\.0", src, re.M):
        f.append("9. MOVE_TEACH does not default to 0.0, so earlier runs are not bit for bit")

    # 10: refused without a ballot
    if "if MOVE_TEACH and not MOVES_ON:" not in src or "--move-teach needs --moves" not in src:
        f.append("10. --move-teach without --moves is accepted, and would report the control's "
                 "numbers under the arm's name")

    # 11 + 12: the signature and the void check in the report
    sig = src[src.find("# 341 IS IN THE SIGNATURE"):][:900]
    if '"move_teach": MOVE_TEACH' not in sig:
        f.append("11. move_teach is not in the arm signature - a mind trained with the term "
                 "would transplant onto one trained without it")
    for k in ("move_teach_live", "move_teach_ballot", "move_teach_seen"):
        if f'"{k}"' not in src:
            f.append(f"12. {k} is not reported - 'the term taught nothing' cannot be told from "
                     f"'the term taught the wrong thing'")

    # the module still parses as itself
    try:
        ast.parse(src)
    except SyntaxError as e:
        f.append(f"0. the source does not parse: {e}")
    return f


# ------------------------------------------------------------------ the checker checks itself

MUTANTS = (
    ("the logits are discarded again",
     '        q["_move_ballot"] = None if _l0 is None else (_l0, _mans)',
     '        q["_move_ballot"] = None', "1."),
    ("the ballot is stashed once and reused",
     'if MOVES_ON and ("_move" not in q or MOVE_TEACH):',
     'if MOVES_ON and "_move" not in q:', "2."),
    ("the move is re-chosen on every pass",
     '    if "_move" not in q:\n        q["_move"] = props[int(l0.argmax())][0]',
     '    q["_move"] = props[int(l0.argmax())][0]', "3."),
    ("the teacher decides the choice",
     "    if \"_move\" not in q:\n        q[\"_move\"] = props[int(l0.argmax())][0]",
     "    props = sorted(props, key=lambda e: -e[3])\n"
     "    if \"_move\" not in q:\n        q[\"_move\"] = props[int(l0.argmax())][0]", "4."),
    ("the move probability is folded into the router's value",
     "    mv = move_term(q, device)",
     "    mv = None\n    v2 = v2 * move_term(q, device)", "5."),
    ("only the far return carries the term",
     "        out = p1[0] * v_stay + p1[1] * v2\n"
     "        out = out if mv is None else out + mv",
     "        out = p1[0] * v_stay + p1[1] * v2", "6."),
    ("a dead ballot is counted as live",
     '    if len(ans) < 2 or len(set(ans)) < 2:\n        return None\n    _MOVE_LIVE["live"] += 1',
     '    _MOVE_LIVE["live"] += 1\n    if len(ans) < 2 or len(set(ans)) < 2:\n        return None',
     "7."),
    ("the raw signed reward, without the scale",
     "    R = shift_reward(torch.tensor([1.0 if a else -1.0 for a in ans],\n"
     "                                  device=device, dtype=l0.dtype))",
     "    R = torch.tensor([1.0 if a else -1.0 for a in ans],\n"
     "                     device=device, dtype=l0.dtype)", "8."),
    ("the flag is on by default",
     "MOVE_TEACH = 0.0", "MOVE_TEACH = 1.0", "9."),
    ("the flag is accepted without a ballot",
     "    if MOVE_TEACH and not MOVES_ON:", "    if False:", "10."),
    ("the term is not in the arm signature",
     '                "speak_batch": SPEAK_BATCH, "speak_weight": SPEAK_WEIGHT,\n'
     '                "move_teach": MOVE_TEACH,',
     '                "speak_batch": SPEAK_BATCH, "speak_weight": SPEAK_WEIGHT,', "11."),
    ("the void check is not reported",
     '"move_teach_live": (_MOVE_LIVE["live"] / max(1, _MOVE_LIVE["n"])',
     '"unused_live": (_MOVE_LIVE["live"] / max(1, _MOVE_LIVE["n"])', "12."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props(src)
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its own anchor occurs {src.count(old)} "
                         f"times - the mutation is not testing what it says")
            continue
        if not any(g.startswith(tag) for g in props(src.replace(old, new, 1))):
            fails.append(f"MUTATION {tag} ({name}): the failure was re-introduced and check "
                         f"{tag} did not fire - it is a comment, not a check")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures, {len(MUTANTS)} mutations" if fails
          else f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
