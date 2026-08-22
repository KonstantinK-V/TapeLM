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
v0 = v3('_stage289_derivation.py')

def body(v4, v5, v6, v7):
    v8 = v38.v24(f'^def {v5}\\(.*?(?=\\n(?:def |@|# ---))', v4, v38.v39 | v38.v40)
    if not v8:
        v6.v41(f'{v7} {v5} not found')
        return ''
    return v8.v25(0)

def code_of(v9):
    """the function with its docstrings removed - the prose SAYS what the code must not do"""
    return v38.v26('"""(?:.|\\n)*?"""', '', v9)

def props(v4):
    v10 = []
    v11 = v27(v42(v4, 'reach_move_pick', v10, '1.'))
    v12 = v27(v42(v4, 'reach_logits', v10, '1.'))
    v13 = v27(v42(v4, 'move_term', v10, '1.'))
    v14 = v27(v42(v4, 'reach_loss', v10, '5.'))
    if '_move_ballot"] = None if _l0 is None else (_l0, _mans)' not in v12:
        v10.v41('1. reach_logits does not stash the ballot with its logits - the move term has nothing to differentiate and the arm is its own control')
    if '_move_ballot")' not in v13:
        v10.v41('1. move_term does not read the ballot')
    if 'if MOVES_ON and ("_move" not in q or MOVE_TEACH):' not in v12:
        v10.v41("2. the ballot is not rebuilt when the term is on - a tensor stashed on a question is backward'd through a freed graph when a batch draws it twice")
    if 'if "_move" not in q:\n        q["_move"] = props[int(l0.argmax())][0]' not in v11:
        v10.v41("3. the move is re-chosen on every pass, so a question's move can drift away from the offer that was cached for it")
    for v15 in v11.v28():
        if 'argmax' in v15 and ('_a' in v15.v49('_argmax', '') or 'ans' in v15):
            v10.v41(f"4. the lane's answerability is in the choosing expression: {v15.v53()!r}")
    if 'sorted(props' in v11 or 'props.sort' in v11:
        v10.v41('4. the ballot is re-ordered before the argmax - the teacher can decide the choice through the order')
    if 'mv = move_term(q, device)' not in v14:
        v10.v41('5. reach_loss does not compute the move term')
    if v38.v24('v2 = [^\\n]*move', v14) or v38.v24('move[^\\n]*\\* *v2', v14):
        v10.v41('5. the move probability is folded into v2 - that reprices the router, so one lever would be moving two decisions')
    v16 = v29(v38.v43('return -\\(out if mv is None else out \\+ mv\\)', v14))
    if v16 != 2:
        v10.v41(f'6. the move term is on {v16} of the 2 return points - `--two-way` returns early, so a term added only at the far one runs the control on the standing arm')
    if 'if len(ans) < 2 or len(set(ans)) < 2:\n        return None' not in v13:
        v10.v41('7. a ballot that cannot teach is not skipped - it adds a constant and the void check stops meaning anything')
    v17 = v13.v30('len(set(ans)) < 2')
    v18 = v13.v30('_MOVE_LIVE["live"]')
    v19 = v13.v30('_MOVE_LIVE["n"]')
    if not 0 < v19 < v17 < v18:
        v10.v41(f'7. the live counter is not incremented after the guard (n={v19} guard={v17} live={v18}) - a dead ballot would count as live and the void check would read healthy on an arm that taught nothing')
    if 'shift_reward(' not in v13:
        v10.v41('8. the move reward does not go through shift_reward - a signed scale under a discount is what paid the mind to leave in 311a')
    if 'if not MOVE_TEACH:\n        return None' not in v13:
        v10.v41('9. move_term does not return None when the flag is off')
    if not v38.v24('^MOVE_TEACH = 0\\.0', v4, v38.v40):
        v10.v41('9. MOVE_TEACH does not default to 0.0, so earlier runs are not bit for bit')
    if 'if MOVE_TEACH and not MOVES_ON:' not in v4 or '--move-teach needs --moves' not in v4:
        v10.v41("10. --move-teach without --moves is accepted, and would report the control's numbers under the arm's name")
    v20 = v4[v4.v30('# 341 IS IN THE SIGNATURE'):][:900]
    if '"move_teach": MOVE_TEACH' not in v20:
        v10.v41('11. move_teach is not in the arm signature - a mind trained with the term would transplant onto one trained without it')
    for v21 in ('move_teach_live', 'move_teach_ballot', 'move_teach_seen'):
        if f'"{v21}"' not in v4:
            v10.v41(f"12. {v21} is not reported - 'the term taught nothing' cannot be told from 'the term taught the wrong thing'")
    try:
        v46.v44(v4)
    except v31 as e:
        v10.v41(f'0. the source does not parse: {v50}')
    return v10
v1 = (('the logits are discarded again', '        q["_move_ballot"] = None if _l0 is None else (_l0, _mans)', '        q["_move_ballot"] = None', '1.'), ('the ballot is stashed once and reused', 'if MOVES_ON and ("_move" not in q or MOVE_TEACH):', 'if MOVES_ON and "_move" not in q:', '2.'), ('the move is re-chosen on every pass', '    if "_move" not in q:\n        q["_move"] = props[int(l0.argmax())][0]', '    q["_move"] = props[int(l0.argmax())][0]', '3.'), ('the teacher decides the choice', '    if "_move" not in q:\n        q["_move"] = props[int(l0.argmax())][0]', '    props = sorted(props, key=lambda e: -e[3])\n    if "_move" not in q:\n        q["_move"] = props[int(l0.argmax())][0]', '4.'), ("the move probability is folded into the router's value", '    mv = move_term(q, device)', '    mv = None\n    v2 = v2 * move_term(q, device)', '5.'), ('only the far return carries the term', '        out = p1[0] * v_stay + p1[1] * v2\n        return -(out if mv is None else out + mv)', '        return -(p1[0] * v_stay + p1[1] * v2)', '6.'), ('a dead ballot is counted as live', '    if len(ans) < 2 or len(set(ans)) < 2:\n        return None\n    _MOVE_LIVE["live"] += 1', '    _MOVE_LIVE["live"] += 1\n    if len(ans) < 2 or len(set(ans)) < 2:\n        return None', '7.'), ('the raw signed reward, without the scale', '    R = shift_reward(torch.tensor([1.0 if a else -1.0 for a in ans],\n                                  device=device, dtype=l0.dtype))', '    R = torch.tensor([1.0 if a else -1.0 for a in ans],\n                     device=device, dtype=l0.dtype)', '8.'), ('the flag is on by default', 'MOVE_TEACH = 0.0', 'MOVE_TEACH = 1.0', '9.'), ('the flag is accepted without a ballot', '    if MOVE_TEACH and not MOVES_ON:', '    if False:', '10.'), ('the term is not in the arm signature', '                "speak_batch": SPEAK_BATCH, "speak_weight": SPEAK_WEIGHT,\n                "move_teach": MOVE_TEACH,', '                "speak_batch": SPEAK_BATCH, "speak_weight": SPEAK_WEIGHT,', '11.'), ('the void check is not reported', '"move_teach_live": (_MOVE_LIVE["live"] / max(1, _MOVE_LIVE["n"])', '"unused_live": (_MOVE_LIVE["live"] / max(1, _MOVE_LIVE["n"])', '12.'))

def main() -> v2:
    v4 = v0.v32(encoding='utf-8')
    v6 = v33(v4)
    for v5, v34, v35, v7 in v1:
        if v4.v47(v34) != 1:
            v6.v41(f'MUTATION {v7} ({v5}): its own anchor occurs {v4.v47(v34)} times - the mutation is not testing what it says')
            continue
        if not v48((v52.v51(v7) for v52 in v33(v4.v49(v34, v35, 1)))):
            v6.v41(f'MUTATION {v7} ({v5}): the failure was re-introduced and check {v7} did not fire - it is a comment, not a check')
    for v22 in v6:
        v36('FAIL ' + v22)
    v36(f'{v29(v6)} failures, {v29(v1)} mutations' if v6 else f'all properties hold, and all {v29(v1)} re-introduced failures were caught')
    return 1 if v6 else 0
if v23 == '__main__':
    raise v37(v45())