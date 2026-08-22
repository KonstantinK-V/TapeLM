"""THE CEILING OF A SESSION. Does answering question k help question k+1?

WHAT 353 SETTLED AND WHAT IT LEFT. Depth is established: two hops inside ONE question, one
price, one head, home kept (353 `--two-way`, max summary). That is a chain ACROSS HOPS. The
loop diagnosis (351) was about a chain ACROSS QUESTIONS, and nothing has touched it:

    question k+1 knows nothing of question k       no state
    the answer goes nowhere                        never written, never read again
    the reward is terminal and per-question        no consequence propagates

324 CLOSED MEMORY AND THE CLOSURE DOES NOT TRANSFER. It measured a PERFECT write-back's
marginal retrieval gain over INDEPENDENT questions - questions drawn at random from the whole
tape, which share nothing, so of course a previous answer bought nothing. It never measured a
SITUATION: several questions out of the same region, where an answer to one is a filler that
the next question's own paradigm does not contain.

WHAT IS MEASURED, torch-free, before anything is built:

    reach_own    the truth in the offer built from the question's OWN rows   (today)
    reach_W      the truth in the offer when the session's PREVIOUS ANSWERS are added as lenses
    chain_only   reached with the session and NOT without it                 THE NUMBER
    reach_R      the same with WRONG answers written back - random fillers from the same places

The budget is matched: the same top-m at the end, so a session cannot win by being offered more.
reach_R IS THE NULL AND IT IS LOAD-BEARING. Adding any lens widens the offer; if wrong answers
buy the same gain, the session is measuring offer size and not memory. The gain that counts is
reach_W - reach_R, not reach_W - reach_own.

W holds the TRUE previous answers - the ceiling of a perfect memory, the job the mind would be
given. If that ceiling is flat, no policy over it can be worth building and 324's closure
stands after all, for a reason 324 never established.

    python _audit354_session.py
    python _audit354_session.py --session 8 --session-lines 40
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage354_session.json')

def main() -> v2:
    v4 = v69.v25()
    v4.v26('--bytes', type=v2, default=30000000)
    v4.v26('--frame-max', type=v2, default=3)
    v4.v26('--min-fillers', type=v2, default=2)
    v4.v26('--addresses', type=v2, default=1500)
    v4.v26('--lines', type=v2, default=25000)
    v4.v26('--window-lines', type=v2, default=400)
    v4.v26('--topm', type=v2, default=8, help='values the offer keeps. The budget')
    v4.v26('--session', type=v2, default=6, help='questions in one situation')
    v4.v26('--session-lines', type=v2, default=40, help="how tight a situation is. 0 = draw from the whole tape, which is 324's independent-question setting and should reproduce its ~0")
    v4.v26('--seed', type=v2, default=1337)
    v4.v26('--sessions', type=v2, default=400)
    v4.v26('--corpus', default=v103(v0))
    v5 = v4.v27()
    v6 = v3(v5.v116).v104('r', encoding='utf-8', errors='ignore').v28(v5.v29)
    v7 = [v71.v70() for v71 in v6.v105('\n') if v43(v71.v70()) >= 80]
    v8 = v7[:v2(0.7 * v43(v7))][:v5.v8]
    v9 = v72.v30(v5.v31)
    v32, v33, v34 = v73.v35(v8, v5.v36, v5.v37)
    if v5.v10:
        v38 = v73.v74(v32, v34)
        v39 = v9.v75(v61(1, v43(v8)))
        v40 = v48(v49)
        for v41 in v57(v5.v10):
            for v106, v84 in v38.v79((v39 + v41) % v43(v8), ()):
                v40[v106].v78(v84)
        v32 = [(v106, v52(v53)) for v106, v53 in v40.v51() if v43({v33[v84] for v84 in v53}) >= v5.v37]
    if v5.v42 and v43(v32) > v5.v42:
        v32 = v9.v76(v32, v5.v42)
    if not v32:
        v67('no tape')
        return 1
    v11 = [v49(v45) for v107, v45 in v32]
    v12 = [[v33[v47] for v47 in v45] for v45 in v11]
    v13 = v43(v12)
    v14 = {}
    for v44, v45 in v46(v11):
        for v47 in v45:
            v14[v47] = v44
    v15 = v48(v49)
    for v44, v45 in v46(v11):
        for v47 in v45:
            v15[v33[v47]].v78(v47)
    v16 = {}
    for v44, v45 in v46(v11):
        v16[v44] = v77((v34[v47] for v47 in v45))
    v17 = v48(v49)
    for v44, v50 in v16.v51():
        v17[v50].v78(v44)
    v18 = v52(v17)
    v19 = {}

    def co(v53):
        v20 = v19.v79(v53)
        if v20 is None:
            v20 = v54()
            for v47 in v15[v53]:
                for v108 in v11[v14[v47]]:
                    if v33[v108] != v53:
                        v20[v33[v108]] += 1
            v19[v53] = v20
        return v20
    v20 = v54()
    v55, v56 = ([], [])
    for v21 in v57(v5.v58):
        if v5.v64 and v43(v18) > 1:
            v80 = v9.v75(v43(v18))
            v59 = [v44 for v50 in v18[v80:v80 + v5.v64] for v44 in v17[v50]]
        else:
            v59 = v49(v57(v13))
        v59 = [v44 for v44 in v59 if v43(v12[v44]) >= 2]
        if v43(v59) < 2:
            continue
        v9.v81(v59)
        v59 = v59[:v5.v63]
        v60 = [(v44, v9.v75(v43(v12[v44]))) for v44 in v59]
        v82, v83 = ([], [])
        for v44, v84 in v60:
            v85 = v12[v44][v84]
            v86 = v54(v12[v44])
            v86[v85] -= 1
            if v86[v85] <= 0:
                del v86[v85]
            v87 = v49(v86)[:6]
            if not v87:
                continue
            v88 = v109(v11[v44])

            def offer(v110):
                """the same exclusion every audit makes: this place cannot answer itself,
                at any hop and from any lens."""
                v111 = v54()
                v112 = v54((v33[v47] for v47 in v88))
                for v53 in v110:
                    for v98, v22 in v120(v53).v51():
                        if v98 in v112:
                            v22 -= v112[v98]
                        if v22 > 0 and v98 != v53:
                            v111[v98] += v22
                return v111
            v89 = v113(v87)
            v90 = v113(v87 + [v53 for v53 in v82 if v53 not in v87])
            v91 = v113(v87 + [v53 for v53 in v83 if v53 not in v87])
            v92 = lambda v47: {v98 for v98, v118 in v47.v119(v5.v62)}
            v93 = v85 in v92(v89)
            v94 = v85 in v92(v90)
            v95 = v85 in v92(v91)
            v96 = v97 = False
            for v98 in v82:
                if v98 not in v87 and v85 in v92(v113(v87 + [v98])):
                    v96 = True
                    break
            for v98 in v83:
                if v98 not in v87 and v85 in v92(v113(v87 + [v98])):
                    v97 = True
                    break
            v20['n'] += 1
            v20['reach_own'] += v93
            v20['reach_W'] += v94
            v20['reach_R'] += v95
            v20['oracle_W'] += v96 or v93
            v20['oracle_R'] += v97 or v93
            v20['held'] += v43(v82)
            v20['chain_only'] += v94 and (not v93)
            v20['lost'] += v93 and (not v94)
            v55.v78(v43(v89))
            v56.v78(v43(v90))
            v82.v78(v85)
            v99 = [v53 for v53 in v12[v44] if v53 != v85]
            v83.v78(v9.v117(v99) if v99 else v85)
    v22 = v61(1, v20['n'])
    v23 = {'places': v13, 'questions': v20['n'], 'topm': v5.v62, 'session': v5.v63, 'session_lines': v5.v64, 'reach_own': v20['reach_own'] / v22, 'reach_W': v20['reach_W'] / v22, 'reach_R': v20['reach_R'] / v22, 'chain_only': v20['chain_only'] / v22, 'lost': v20['lost'] / v22, 'offer_own': v114(v55) / v61(1, v43(v55)), 'offer_W': v114(v56) / v61(1, v43(v56)), 'oracle_W': v20['oracle_W'] / v22, 'oracle_R': v20['oracle_R'] / v22, 'held': v20['held'] / v22}
    v23['gain'] = v23['reach_W'] - v23['reach_own']
    v23['gain_over_null'] = v23['reach_W'] - v23['reach_R']
    v23['oracle_gain'] = v23['oracle_W'] - v23['oracle_R']
    v1.v100.v65(parents=True, exist_ok=True)
    v1.v66(v115.v101(v23, indent=1), encoding='utf-8')
    v67(f"tape     {v13} places, {v20['n']} questions, sessions of {v5.v63} over {v5.v64 or 'ALL'} lines, topm {v5.v62}")
    v67(f"OWN      reach {v23['reach_own']:.4f}   offer {v23['offer_own']:.0f}")
    v67(f"SESSION  reach {v23['reach_W']:.4f}   offer {v23['offer_W']:.0f}   gain {v23['gain']:+.4f}")
    v67(f"NULL     reach {v23['reach_R']:.4f}   (wrong answers written back)   gain over null {v23['gain_over_null']:+.4f}")
    v67(f"APART    only with the session {v23['chain_only']:.4f}   lost to the crowd {v23['lost']:.4f}")
    v67(f"ORACLE   ONE answer chosen perfectly {v23['oracle_W']:.4f}   null {v23['oracle_R']:.4f}   gain over null {v23['oracle_gain']:+.4f}   (choosing among {v23['held']:.1f} held)")
    if v23['oracle_gain'] > 0.05:
        v67(f"\nTHE SESSION CARRIES, BUT ONLY UNDER SELECTION. Pooling the answers gains {v23['gain_over_null']:+.4f}; CHOOSING ONE gains {v23['oracle_gain']:+.4f} over a wrong answer given the same number of shots. That is a well-posed problem - pick which remembered answer to read the next question through - and it is the kind of problem Phi already solves. 324's closure was scoped to independent questions and does not cover it.")
    elif v23['gain_over_null'] > 0.05:
        v67(f'\nTHE SESSION CARRIES POOLED but not under selection - unexpected, and the audit should be read again before it is believed.')
    else:
        v67("\nTHE SESSION DOES NOT CARRY, POOLED OR CHOSEN: a true previous answer buys no more than a wrong one, at matched budget and given the same number of shots. The identity of what was answered before carries nothing about the next question, and 324's closure stands - now for a reason 324 never established.")
    v67(f'\nwritten to {v1}')
    return 0
if v24 == '__main__':
    raise v68(v102())