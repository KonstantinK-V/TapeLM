"""Could memory buy anything on this tape - measured before any memory is built.

WHAT MEMORY WOULD MEAN HERE. Every question is answered from nothing but the tape; nothing
found at question k is available at question k+1 except the weights. The cheapest real memory
is a WRITE-BACK: once a hole is resolved, that (place, value) becomes an ordinary row, visible
to every later question. Successes then accumulate and mistakes poison - which is what memory
is, and why it must be priced before it is built.

THE CEILING IS WHAT THIS MEASURES, not the mechanism. Assume a PERFECT memory: every earlier
hole resolved correctly and written back. How many later questions would that answer, over and
above what the question's own rows already give? If the answer is ~0, memory cannot pay on this
tape whatever mechanism carries it, and the right move is to say so instead of building it.

Three ceilings, from tightest to loosest:
  same place   - a later hole at the SAME place whose truth a remembered row already holds.
                 This overlaps CONFIRM by construction, so the number that matters is the part
                 NOT already in the question's own rows.
  shared       - a later hole at a place that SHARES A FILLER with a remembered place, and whose
                 truth is among that place's remembered values. This is the walk with memory.
  any          - the truth was remembered anywhere at all. A loose upper bound, printed so the
                 tighter numbers can be read against something.

    python _audit324_memory.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage324_memory.json')

def main() -> v2:
    v4 = v64.v23()
    v4.v24('--bytes', type=v2, default=30000000)
    v4.v24('--frame-max', type=v2, default=3)
    v4.v24('--min-fillers', type=v2, default=2)
    v4.v24('--addresses', type=v2, default=1500)
    v4.v24('--lines', type=v2, default=25000)
    v4.v24('--window-lines', type=v2, default=400)
    v4.v24('--sample', choices=('uniform', 'region'), default='region')
    v4.v24('--seed', type=v2, default=1337)
    v4.v24('--max-questions', type=v2, default=4000)
    v4.v24('--recall', type=v65, default=1.0, help='the share of earlier holes memory gets RIGHT. 1.0 is the ceiling; lower values show how fast the ceiling decays when memory is wrong, which is the risk a write-back actually carries')
    v5 = v4.v25()
    v6 = v0.v89('r', encoding='utf-8', errors='ignore').v26(v5.v27)
    v7 = [v67.v66() for v67 in v6.v90('\n') if v38(v67.v66()) >= 80]
    v8 = v7[:v2(0.7 * v38(v7))][:v5.v8]
    v9 = v68.v28(v5.v29)
    v11, v30, v31 = v69.v32(v8, v5.v33, v5.v34)
    if v5.v35 == 'region':
        if v5.v36:
            v70 = v69.v91(v11, v31)
            v71 = v9.v92(v58(1, v38(v8)))
            v72 = v39(v93)
            for v73 in v94(v5.v36):
                for v99, v53 in v70.v97((v71 + v73) % v38(v8), ()):
                    v72[v99].v78(v53)
            v11 = [(v99, v102(v44)) for v99, v44 in v72.v81() if v38({v30[v53] for v53 in v44}) >= v5.v34]
            if v5.v74 and v38(v11) > v5.v74:
                v11 = v9.v35(v11, v5.v74)
        else:
            v11 = v69.v95(v11, v30, v31, v38(v8), v5.v74, v9, v5.v34)
    elif v5.v74 and v38(v11) > v5.v74:
        v11 = v9.v35(v11, v5.v74)
    if not v11:
        v62('no tape')
        return 1
    v10 = []
    for (v75, v76, v77), v37 in v11:
        v10.v78([v30[v53] for v53 in v37])
    v12 = v38(v10)
    v13 = [v52(v44) for v44 in v10]
    v14 = v39(v40)
    for v41, v42 in v43(v13):
        for v44 in v42:
            v14[v44].v96(v41)
    v15 = [(v41, v53) for v41 in v94(v12) for v53 in v94(v38(v10[v41])) if v38(v10[v41]) >= 2]
    v9.v45(v15)
    v15 = v15[:v5.v79]
    v46, v47 = (8, 8)

    def walk_offer(v41, v48):
        v49 = v52()
        for v44, v80 in v48.v81():
            for v51 in v14[v44]:
                if v51 != v41:
                    v49[v51] += v80
        v50 = [v51 for v51, v100 in v49.v101(v46)]
        v82, v83 = (v40(), [])
        for v51 in v50:
            for v44 in v10[v51]:
                if v44 not in v82:
                    v82.v96(v44)
                    v83.v78(v44)
        return (v50, v40(v83[:v47]))
    v16 = v39(v40)
    v17 = v40()
    v18 = v52()
    for v41, v53 in v15:
        v54 = v10[v41][v53]
        v48 = v52(v10[v41])
        v48[v54] -= 1
        if v48[v54] <= 0:
            del v48[v54]
        v18['n'] += 1
        v55 = v54 in v48
        v18['in_own'] += v55
        v56 = v54 in v16.v97(v41, ())
        v50, v84 = v85(v41, v48)
        v18['walk_reach'] += v54 in v84
        v57 = False
        if not v56:
            for v51 in v50:
                if v54 in v16.v97(v51, ()):
                    v57 = True
                    break
        v18['same_place'] += v56 and (not v55)
        v18['shared'] += (v56 or v57) and (not v55) and (v54 not in v84)
        v18['shared_incl_walk'] += (v56 or v57) and (not v55)
        v18['any'] += v54 in v17 and (not v55) and (v54 not in v84)
        v18['answerable_wo_own'] += not v55
        v18['unreached'] += not v55 and v54 not in v84
        if v5.v59 >= 1.0 or v9.v68() < v5.v59:
            v16[v41].v96(v54)
            v17.v96(v54)
    v19 = v58(1, v18['n'])
    v20 = v58(1, v18['unreached'])
    v21 = {'bytes': v5.v27, 'sample': v5.v35, 'window_lines': v5.v36, 'recall': v5.v59, 'places': v12, 'questions': v18['n'], 'in_own': v18['in_own'] / v19, 'walk_reach': v18['walk_reach'] / v19, 'same_place_gain': v18['same_place'] / v19, 'shared_gain': v18['shared'] / v19, 'shared_incl_walk': v18['shared_incl_walk'] / v19, 'any_gain': v18['any'] / v19, 'shared_gain_of_unreached': v18['shared'] / v20}
    v1.v86.v60(parents=True, exist_ok=True)
    v1.v61(v98.v87(v21, indent=1), encoding='utf-8')
    v62(f"tape    {v12} places, {v18['n']} questions in order, in_own {v21['in_own']:.4f}")
    v62(f"CEILING beyond the question's own rows, with recall {v5.v59}:")
    v62(f"        walk already reaches {v21['walk_reach']:.4f}")
    v62(f"        same place {v21['same_place_gain']:.4f}   shared-and-remembered, BEYOND the walk {v21['shared_gain']:.4f}   (including what the walk gives: {v21['shared_incl_walk']:.4f})")
    v62(f"        anywhere, beyond the walk {v21['any_gain']:.4f}")
    v62(f"        as a share of what neither own rows nor the walk reach: {v21['shared_gain_of_unreached']:.4f}")
    v62(f'\nwritten to {v1}')
    return 0
if v22 == '__main__':
    raise v63(v88())