"""IS THERE ANYTHING TO CARRY BETWEEN QUESTIONS THAT IS NOT A FACT?

354 CLOSED THE OBVIOUS KIND OF MEMORY. A previous answer used as a lens buys nothing: pooled it
LOSES (-0.038), and chosen perfectly by an oracle it gains +0.0025 / -0.0004 against a null that
writes WRONG answers back. A trained level cannot beat its own oracle, so no amount of
architecture makes remembered CONTENT pay. That is the substrate's answer and it is final.

BUT CONTENT IS NOT THE ONLY THING A MIND CARRIES, and this project of all projects should say
so, because its whole thesis is the split: FACTS LIVE ON THE TAPE, POLICY LIVES IN PHI. 354
measured a memory of facts. A memory of READING - "in this situation the tape is thin, do not
speak" - holds no facts, breaks no invariant, and has never been measured.

WHAT WOULD MAKE IT REAL. Difficulty must be a property of the SITUATION, not only of the
question. And it must be a property Phi CANNOT ALREADY SEE in the question in front of it -
otherwise a carried state is redundant and a level added for it is dead weight.

    lift        p(reach | the previous question in this situation reached)
                  minus p(reach | it did not)
    null        the same, with the "previous" question taken from ANOTHER situation
    WITHIN      the same, computed inside strata of the CURRENT question's own difficulty
                (bins of the top-1 minus top-2 count margin of its own offer - a count, no
                heuristic). THIS IS THE NUMBER THAT DECIDES: it is the part of the situation
                that the question in front of Phi does not already announce.

  GATE  within-stratum lift > 0.05 with the cross-situation null near zero. Then a carried
        state is worth a level, and it is a level that holds no facts.
  If WITHIN is flat while raw lift is not, situations do differ but the current question
  already says so, and the state is redundant - build nothing.

    python _audit355_carry.py --session 6 --session-lines 40
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage355_carry.json')

def rate(v4, v5):
    v6 = [v29 for v29 in v4 if v5(v29)]
    return (v121((v29['reach'] for v29 in v6)) / v64(v6), v64(v6)) if v6 else (0.0, 0)

def lift(v4, v7):
    """p(reach | previous reached) - p(reach | previous did not), and the two supports."""
    v33, v34 = v35(v4, lambda v29: v29[v7] == 1)
    v36, v37 = v35(v4, lambda v29: v29[v7] == 0)
    return (v33 - v36, v33, v34, v36, v37)

def within(v4, v7, v8):
    """the same difference computed INSIDE strata of the current question's own margin, then
    pooled by stratum size. A situation effect that survives this is one the question in front
    of Phi does not already announce."""
    v9 = v38((v29['margin'] for v29 in v4))
    if not v9:
        return 0.0
    v10 = [v9[v2(v64(v9) * (v41 + 1) / v8) - 1] for v41 in v74(v8)]
    v39, v40 = (0.0, 0)
    for v41, v42 in v43(v10):
        v44 = v10[v41 - 1] if v41 else -1e+18
        v45 = [v29 for v29 in v4 if v44 < v29['margin'] <= v42 or (v41 == 0 and v29['margin'] <= v42)]
        v62, v98, v34, v99, v37 = v80(v45, v7)
        if v34 and v37:
            v39 += v62 * (v34 + v37)
            v40 += v34 + v37
    return (v39 / v40 if v40 else 0.0, v40)

def main() -> v2:
    v11 = v100.v46()
    v11.v47('--bytes', type=v2, default=30000000)
    v11.v47('--frame-max', type=v2, default=3)
    v11.v47('--min-fillers', type=v2, default=2)
    v11.v47('--addresses', type=v2, default=1500)
    v11.v47('--lines', type=v2, default=25000)
    v11.v47('--window-lines', type=v2, default=400)
    v11.v47('--topm', type=v2, default=8)
    v11.v47('--session', type=v2, default=6)
    v11.v47('--session-lines', type=v2, default=40)
    v11.v47('--bins', type=v2, default=5)
    v11.v47('--seed', type=v2, default=1337)
    v11.v47('--sessions', type=v2, default=600)
    v11.v47('--corpus', default=v126(v0))
    v12 = v11.v48()
    v13 = v3(v12.v138).v127('r', encoding='utf-8', errors='ignore').v49(v12.v50)
    v14 = [v102.v101() for v102 in v13.v128('\n') if v64(v102.v101()) >= 80]
    v15 = v14[:v2(0.7 * v64(v14))][:v12.v15]
    v16 = v103.v51(v12.v52)
    v53, v54, v55 = v104.v56(v15, v12.v57, v12.v58)
    if v12.v17:
        v59 = v104.v105(v53, v55)
        v60 = v16.v106(v122(1, v64(v15)))
        v61 = v68(v69)
        for v62 in v74(v12.v17):
            for v78, v41 in v59.v110((v60 + v62) % v64(v15), ()):
                v61[v78].v109(v41)
        v53 = [(v78, v38(v72)) for v78, v72 in v61.v71() if v64({v54[v41] for v41 in v72}) >= v12.v58]
    if v12.v63 and v64(v53) > v12.v63:
        v53 = v16.v107(v53, v12.v63)
    if not v53:
        v96('no tape')
        return 1
    v18 = [v69(v66) for v129, v66 in v53]
    v19 = [[v54[v67] for v67 in v66] for v66 in v18]
    v20 = v64(v19)
    v21 = {}
    for v65, v66 in v43(v18):
        for v67 in v66:
            v21[v67] = v65
    v22 = v68(v69)
    for v65, v66 in v43(v18):
        for v67 in v66:
            v22[v54[v67]].v109(v67)
    v23 = {v65: v108((v55[v67] for v67 in v66)) for v65, v66 in v43(v18)}
    v24 = v68(v69)
    for v65, v70 in v23.v71():
        v24[v70].v109(v65)
    v25 = v38(v24)
    v26 = {}

    def co(v72):
        v73 = v26.v110(v72)
        if v73 is None:
            v73 = v130()
            for v67 in v22[v72]:
                for v131 in v18[v21[v67]]:
                    if v54[v131] != v72:
                        v73[v54[v131]] += 1
            v26[v72] = v73
        return v73
    v27 = []
    for v28 in v74(v12.v27):
        if v12.v93 and v64(v25) > 1:
            v33 = v16.v106(v64(v25))
            v75 = [v65 for v70 in v25[v33:v33 + v12.v93] for v65 in v24[v70]]
        else:
            v75 = v69(v74(v20))
        v75 = [v65 for v65 in v75 if v64(v19[v65]) >= 2]
        if v64(v75) < 2:
            continue
        v16.v111(v75)
        v76 = []
        for v65 in v75[:v12.v92]:
            v41 = v16.v106(v64(v19[v65]))
            v112 = v19[v65][v41]
            v113 = v130(v19[v65])
            v113[v112] -= 1
            if v113[v112] <= 0:
                del v113[v112]
            v114 = v69(v113)[:6]
            if not v114:
                continue
            v115 = v132(v18[v65])
            v116 = v130((v54[v67] for v67 in v115))
            v117 = v130()
            for v72 in v114:
                for v135, v136 in v139(v72).v71():
                    if v135 in v116:
                        v136 -= v116[v135]
                    if v136 > 0 and v135 != v72:
                        v117[v135] += v136
            v118 = v117.v133(v12.v91)
            v119 = v118[0][1] - v118[1][1] if v64(v118) > 1 else v118[0][1] if v118 else 0
            v76.v109({'reach': v2(v112 in {v135 for v135, v140 in v118}), 'margin': v119})
        if v64(v76) >= 2:
            v27.v109(v76)
    v4 = []
    for v77, v76 in v43(v27):
        for v78 in v74(1, v64(v76)):
            v4.v109({'reach': v76[v78]['reach'], 'margin': v76[v78]['margin'], 'prev': v76[v78 - 1]['reach'], 'sid': v77, 'pos': v78})
    for v29 in v4:
        for v79 in v74(8):
            v120 = v16.v106(v64(v27))
            if v120 != v29['sid'] and v64(v27[v120]) > v29['pos']:
                v29['prev_null'] = v27[v120][v29['pos'] - 1]['reach']
                break
        else:
            v29['prev_null'] = v29['prev']
    v62, v33, v34, v36, v37 = v80(v4, 'prev')
    v81, v82, v83, v84, v85 = v80(v4, 'prev_null')
    v86, v87 = v88(v4, 'prev', v12.v8)
    v89, v90 = v88(v4, 'prev_null', v12.v8)
    v30 = v121((v29['reach'] for v29 in v4)) / v122(1, v64(v4))
    v31 = {'places': v20, 'sessions': v64(v27), 'pairs': v64(v4), 'topm': v12.v91, 'session': v12.v92, 'session_lines': v12.v93, 'bins': v12.v8, 'base': v30, 'lift': v62, 'p_after_hit': v33, 'n_after_hit': v34, 'p_after_miss': v36, 'n_after_miss': v37, 'lift_null': v81, 'within': v86, 'within_null': v89, 'within_cov': v87}
    v1.v123.v94(parents=True, exist_ok=True)
    v1.v95(v134.v124(v31, indent=1), encoding='utf-8')
    v96(f'tape     {v20} places, {v64(v27)} situations, {v64(v4)} pairs, base reach {v30:.4f}')
    v96(f'LIFT     after a hit {v33:.4f} (n {v34})   after a miss {v36:.4f} (n {v37})   {v62:+.4f}')
    v96(f'NULL     previous taken from ANOTHER situation                        {v81:+.4f}')
    v96(f"WITHIN   inside strata of the question's OWN margin  {v86:+.4f}   null {v89:+.4f}   (coverage {v87} of {v64(v4)} pairs)")
    if v87 < 0.2 * v64(v4):
        v96('\nVOID, NOT FLAT. The strata almost never hold both a hit and a miss, so there is no within-stratum comparison to read. Re-run with fewer --bins before believing any WITHIN number.')
    elif v86 - v137(v89) > 0.05:
        v96(f"\nTHERE IS SOMETHING TO CARRY, AND IT IS NOT A FACT. A situation's difficulty is {v86:+.4f} predictable from the previous question BEYOND what the current question's own margin says. A state that carries it holds no facts and breaks no invariant - it changes HOW the tape is read, which is exactly Phi's half of the split.")
    elif v62 - v137(v81) > 0.05:
        v96(f"\nSITUATIONS DIFFER BUT THE QUESTION ALREADY SAYS SO. Raw lift {v62:+.4f} collapses to {v86:+.4f} inside strata of the current question's own margin. A carried state would be redundant with what Phi already sees. Build nothing.")
    else:
        v96("\nNOTHING TO CARRY. Answerability is not a property of the situation at all: one question's outcome says nothing about the next, in or out of a situation. Memory is closed on the substrate for CONTENT (354) and for READING (355).")
    v96(f'\nwritten to {v1}')
    return 0
if v32 == '__main__':
    raise v97(v125())