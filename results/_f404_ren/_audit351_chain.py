"""THE CEILING OF A CHAIN. Does a second step reach what one step cannot?

WHY THIS AND WHY NOW. Kostya's diagnosis, and it is right: we have laid the parts on the floor
and pressed a pedal that is lying next to the engine. The pipeline IS assembled - text, write
path, tape, question, walk, worlds, Phi, answer, reward, weights - but it is ASSEMBLED FOR ONE
SHOT. There is no loop:

    question k+1 knows nothing of question k          no state
    the answer goes nowhere - never written, never read again
    the reward is terminal and per-question           no consequence propagates
    WE choose the hole, at random                     an exam item, not a situation

AND EVERY CLOSED RESULT WAS CLOSED AS A SINGLE-SHOT OPERATION. Composition, in any mind, is
"answer A, then ask B through A" - we tried it as one joint world and it factorised by identity.
Generation is a partial result recombined with the store, iteratively. Revision literally
requires a second step. We have been diagnosing an engine with its crankshaft on the bench.

THE CLAIM THIS AUDIT TESTS. Generation may not need a new relation on the tape at all. It needs
a SECOND STEP. Substitution can only hand back what already stood where A stands - but a value
reached at step one is NOT at the question's place, so using IT as the next lens reaches things
the question's own paradigm never contained. Two rankings chained produce what neither can.

    step 1   the question's own rows -> what stands with them
    step 2   THOSE values -> what stands with THEM

THIS IS NOT L3. L3 intersected two lenses that were both the question's OWN rows - same
paradigm, narrowed. A chain leaves the paradigm at the first hop, which is the whole point.

AND IT IS NOT NEW EVIDENCE-FREE HOPE. 322 already measured depth on the WALK: reachable 0.54 at
depth 2 against 0.12 at depth 1 - the largest single movement of reach this project has ever
recorded. It was closed because CONFIRM collapsed (42/312) and because an honest rival read
2-3%. BOTH OF THOSE ARE SINGLE-SHOT FRAMING ARTEFACTS: CONFIRM broke because ONE decision had to
serve both "answer at home" and "chase depth", and the rival was weak because a one-hop rival
cannot follow a two-hop path. Neither says the chain does not reach.

WHAT IS MEASURED, all with the question's own place and lines excluded at every hop:

    reach1        the truth in step one's offer                    (today's ceiling)
    reach2        the truth in step two's offer
    chain_only    reached at two hops and NOT at one               THE NUMBER THAT DECIDES
    oracle2       the truth reachable by SOME path of length two   the ceiling of a perfect
                                                                   chooser, which is the job
                                                                   the mind would be given
    paths         how many paths there are to choose between       the size of that job

If chain_only is large and paths is not astronomical, the mind has a well-posed problem it has
never been given, and every single-shot closure has to be re-read as conditional. If chain_only
is small, chaining substitution does not generate either, and the substrate verdict stands.

    python _audit351_chain.py
    python _audit351_chain.py --branch 8 --topm 8      # budget-matched to today's offer
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage351_chain.json')

def main() -> v2:
    v4 = v76.v22()
    v4.v23('--bytes', type=v2, default=30000000)
    v4.v23('--frame-max', type=v2, default=3)
    v4.v23('--min-fillers', type=v2, default=2)
    v4.v23('--addresses', type=v2, default=1500)
    v4.v23('--lines', type=v2, default=25000)
    v4.v23('--window-lines', type=v2, default=400)
    v4.v23('--topm', type=v2, default=8, help='values a lens offers, per hop')
    v4.v23('--branch', type=v2, default=8, help="how many of step one's values are followed. The chooser's fan-out")
    v4.v23('--seed', type=v2, default=1337)
    v4.v23('--max-questions', type=v2, default=2000)
    v4.v23('--corpus', default=v96(v0))
    v5 = v4.v24()
    v6 = v3(v5.v108).v97('r', encoding='utf-8', errors='ignore').v25(v5.v26)
    v7 = [v78.v77() for v78 in v6.v98('\n') if v40(v78.v77()) >= 80]
    v8 = v7[:v2(0.7 * v40(v7))][:v5.v8]
    v9 = v79.v27(v5.v28)
    v29, v30, v31 = v80.v32(v8, v5.v33, v5.v34)
    if v5.v10:
        v35 = v80.v81(v29, v31)
        v36 = v9.v82(v69(1, v40(v8)))
        v37 = v45(v46)
        for v38 in v83(v5.v10):
            for v99, v53 in v35.v85((v36 + v38) % v40(v8), ()):
                v37[v99].v91(v53)
        v29 = [(v99, v106(v47)) for v99, v47 in v37.v101() if v40({v30[v53] for v53 in v47}) >= v5.v34]
    if v5.v39 and v40(v29) > v5.v39:
        v29 = v9.v84(v29, v5.v39)
    if not v29:
        v74('no tape')
        return 1
    v11 = [v46(v42) for v100, v42 in v29]
    v12 = [[v30[v44] for v44 in v42] for v42 in v11]
    v13 = v40(v12)
    v14 = {}
    for v41, v42 in v43(v11):
        for v44 in v42:
            v14[v44] = v41
    v15 = v45(v46)
    for v41, v42 in v43(v11):
        for v44 in v42:
            v15[v30[v44]].v91(v44)
    v16 = {}

    def co(v47):
        v18 = v16.v85(v47)
        if v18 is None:
            v18 = v49()
            for v44 in v15[v47]:
                for v63 in v11[v14[v44]]:
                    if v30[v63] != v47:
                        v18[v30[v63]] += 1
            v16[v47] = v18
        return v18
    v17 = [(v41, v53) for v41 in v83(v13) for v53 in v83(v40(v12[v41])) if v40(v12[v41]) >= 2]
    v9.v48(v17)
    v17 = v17[:v5.v86]
    v18 = v49()
    v50, v51, v52 = ([], [], [])
    for v41, v53 in v17:
        v54 = v12[v41][v53]
        v55 = v49(v12[v41])
        v55[v54] -= 1
        if v55[v54] <= 0:
            del v55[v54]
        v56 = v46(v55)[:6]
        if not v56:
            continue
        v18['n'] += 1
        v57 = v87(v11[v41])
        v58 = {v41}

        def offer(v47):
            """v's co-occurrence with THIS place subtracted - the same exclusion every other
            audit makes, applied at every hop so a chain cannot re-enter the question."""
            v88 = v49()
            v89 = {v30[v44]: 0 for v44 in v57}
            for v90, v19 in v109(v47).v101():
                if v90 in v89:
                    v19 -= v104((1 for v44 in v57 if v30[v44] == v90))
                if v19 > 0 and v90 != v47:
                    v88[v90] = v19
            return v88
        v59 = v49()
        for v47 in v56:
            v59 += v102(v47)
        v60 = [v90 for v90, v107 in v59.v103(v5.v70)]
        v51.v91(v40(v59))
        v61 = v54 in v87(v60)
        v62 = [v90 for v90, v107 in v59.v103(v5.v71)]
        v63 = v49()
        v64 = False
        v65 = 0
        for v66 in v62:
            v92 = v102(v66)
            v65 += 1
            for v90, v19 in v92.v103(v5.v70):
                v63[v90] += v19
                if v90 == v54:
                    v64 = True
        v67 = [v90 for v90, v107 in v63.v103(v5.v70)]
        v52.v91(v40(v63))
        v50.v91(v65)
        v68 = v54 in v87(v67)
        v18['reach1'] += v61
        v18['reach2'] += v68
        v18['union'] += v61 or v68
        v18['chain_only'] += v68 and (not v61)
        v18['one_only'] += v61 and (not v68)
        v18['oracle2'] += v64
        v18['oracle_or_1'] += v64 or v61
    v19 = v69(1, v18['n'])
    v20 = {'bytes': v5.v26, 'window_lines': v5.v10, 'places': v13, 'questions': v18['n'], 'topm': v5.v70, 'branch': v5.v71, 'reach1': v18['reach1'] / v19, 'reach2': v18['reach2'] / v19, 'union': v18['union'] / v19, 'chain_only': v18['chain_only'] / v19, 'one_only': v18['one_only'] / v19, 'oracle2': v18['oracle2'] / v19, 'oracle_or_1': v18['oracle_or_1'] / v19, 'paths': v104(v50) / v69(1, v40(v50)), 'step1_size': v104(v51) / v69(1, v40(v51)), 'step2_size': v104(v52) / v69(1, v40(v52))}
    v1.v93.v72(parents=True, exist_ok=True)
    v1.v73(v105.v94(v20, indent=1), encoding='utf-8')
    v74(f"tape    {v13} places, {v18['n']} questions, topm {v5.v70}, branch {v5.v71}")
    v74(f"ONE     reach {v20['reach1']:.4f}   offer {v20['step1_size']:.0f}")
    v74(f"TWO     reach {v20['reach2']:.4f}   offer {v20['step2_size']:.0f}   paths {v20['paths']:.1f}")
    v74(f"APART   only by TWO {v20['chain_only']:.4f}   only by ONE {v20['one_only']:.4f}   union {v20['union']:.4f}")
    v74(f"ORACLE  some 2-hop path lands on the truth {v20['oracle2']:.4f}   with one-hop {v20['oracle_or_1']:.4f}")
    if v20['oracle_or_1'] > v20['reach1'] + 0.05:
        v74(f"\nTHE CHAIN REACHES. A perfect chooser over {v20['paths']:.0f} paths would see {v20['oracle_or_1']:.4f} against one step's {v20['reach1']:.4f}. That is a well-posed problem the mind has never been given, and every single-shot closure - composition, generation, revision - has to be re-read as conditional on a loop that does not exist yet.")
    else:
        v74('\nCHAINING DOES NOT GENERATE EITHER: a second hop over the same relation reaches no more than the first. Substitution is closed as a substrate, one step or two.')
    v74(f'\nwritten to {v1}')
    return 0
if v21 == '__main__':
    raise v75(v95())