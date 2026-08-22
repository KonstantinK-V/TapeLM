"""How much of the truth is ON THE TAPE, against how much the walk reaches.

WHY THIS COULD OVERTURN A CONCLUSION WE HAVE ALREADY DRAWN. 310 read the composition failure as
the corpus's fault: strict pairs carry no joint signal, so the product of marginals is correct
and no mind should beat it. That reading rests on a ceiling of 0.17 - but 0.17 is the ceiling OF
OUR OFFER, eight candidates from eight places, not the ceiling of the tape. A hidden truth is a
filler, and a filler usually stands somewhere else too.

So: what share of hidden truths is present ANYWHERE on the tape, and what share does the walk
actually put in front of the mind? If presence is far above reach, the binding constraint is our
RETRIEVAL, not the corpus, and 310 has to be re-read - "there was no signal" becomes "we could
not get to it", which is a different and far more tractable problem.

The distances are nested, so the number to read is where the mass stops growing:
  own      - the question's own place, other rows. What a lookup gets for free.
  walk K   - the eight nearest places by fingerprint. What the mind is offered today.
  walk 2K  - the same compass, twice as far. Separates "our cap is tight" from "the direction
             is wrong": if 2K barely beats K, walking further is not the answer.
  shared   - places holding a filler in common, exactly. 323's second compass.
  any      - anywhere on the tape. The tape's own ceiling, and the number 310 assumed was 0.17.

    python _audit327_presence.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage327_presence.json')
v4, v5, v6 = (8, 8, 64)

def fp(v7: v29, v8: v2=v6):
    v9 = [0.0] * v8
    v10 = f'  {v7}  '
    for v11 in v30(v47(v10) - 2):
        v31 = v112.v108(v10[v11:v11 + 3].v113('utf-8'), digest_size=8).v71()
        v9[v2.v109(v31[:4], 'big') % v8] += 1.0
    v12 = v72((v94 * v94 for v94 in v9)) ** 0.5
    return [v94 / v12 for v94 in v9] if v12 else v9

def main() -> v2:
    v13 = v73.v32()
    v13.v33('--bytes', type=v2, default=30000000)
    v13.v33('--frame-max', type=v2, default=3)
    v13.v33('--min-fillers', type=v2, default=2)
    v13.v33('--addresses', type=v2, default=1500)
    v13.v33('--lines', type=v2, default=25000)
    v13.v33('--window-lines', type=v2, default=400)
    v13.v33('--sample', choices=('uniform', 'region'), default='region')
    v13.v33('--seed', type=v2, default=1337)
    v13.v33('--max-questions', type=v2, default=3000)
    v14 = v13.v34()
    v15 = v0.v95('r', encoding='utf-8', errors='ignore').v35(v14.v36)
    v16 = [v75.v74() for v75 in v15.v96('\n') if v47(v75.v74()) >= 80]
    v17 = v16[:v2(0.7 * v47(v16))][:v14.v17]
    v18 = v76.v37(v14.v38)
    v39, v40, v41 = v77.v42(v17, v14.v43, v14.v44)
    if v14.v45 == 'region':
        if v14.v46:
            v78 = v77.v97(v39, v41)
            v79 = v18.v98(v66(1, v47(v17)))
            v52 = v48(v99)
            for v80 in v30(v14.v46):
                for v110, v11 in v78.v111((v79 + v80) % v47(v17), ()):
                    v52[v110].v84(v11)
            v39 = [(v110, v87(v9)) for v110, v9 in v52.v82() if v47({v40[v11] for v11 in v9}) >= v14.v44]
            if v14.v81 and v47(v39) > v14.v81:
                v39 = v18.v45(v39, v14.v81)
        else:
            v39 = v77.v100(v39, v40, v41, v47(v17), v14.v81, v18, v14.v44)
    elif v14.v81 and v47(v39) > v14.v81:
        v39 = v18.v45(v39, v14.v81)
    if not v39:
        v69('no tape')
        return 1
    v19 = [[v40[v11] for v11 in v101] for v102, v101 in v39]
    v20 = v47(v19)
    v21 = [v54(v9) for v9 in v19]
    v22 = v48(v49)
    for v50, v24 in v51(v21):
        for v9 in v24:
            v22[v9].v103(v50)
    v23 = []
    for v24 in v21:
        v52 = [0.0] * v6
        for v9, v26 in v24.v82():
            v83 = v104(v9)
            for v11 in v30(v6):
                v52[v11] += v26 * v83[v11]
        v12 = v72((v94 * v94 for v94 in v52)) ** 0.5
        v23.v84([v94 / v12 for v94 in v52] if v12 else v52)
    v25 = [(v50, v11) for v50 in v30(v20) for v11 in v30(v47(v19[v50])) if v47(v19[v50]) >= 2]
    v18.v53(v25)
    v25 = v25[:v14.v85]
    v26 = v54()
    for v50, v11 in v25:
        v55 = v19[v50][v11]
        v56 = v54(v19[v50])
        v56[v55] -= 1
        if v56[v55] <= 0:
            del v56[v55]
        v26['n'] += 1
        v57 = v55 in v56
        v26['own'] += v57
        v52 = [0.0] * v6
        for v9, v86 in v56.v82():
            v83 = v104(v9)
            for v80 in v30(v6):
                v52[v80] += v86 * v83[v80]
        v58 = v72((v94 * v94 for v94 in v52)) ** 0.5
        v59 = [v94 / v58 for v94 in v52] if v58 else v52
        v60 = v87(((v72((v117 * v24 for v117, v24 in v118(v59, v23[v89]))), v89) for v89 in v30(v20) if v89 != v50), reverse=True)

        def offered(v88):
            v105, v106 = (v49(), [])
            for v89 in v88:
                for v9 in v19[v89]:
                    if v9 not in v105:
                        v105.v103(v9)
                        v106.v84(v9)
            return v49(v106[:v5])
        v61 = v90([v89 for v114, v89 in v60[:v4]])
        v62 = v90([v89 for v114, v89 in v60[:2 * v4]])
        v63 = v54()
        for v9, v86 in v56.v82():
            for v89 in v22[v9]:
                if v89 != v50:
                    v63[v89] += v86
        v64 = v90([v89 for v89, v115 in v63.v116(v4)])
        v65 = v47(v22.v111(v55, ())) > 1 or v57
        v26['walk_k'] += v55 in v61
        v26['walk_2k'] += v55 in v62
        v26['shared'] += v55 in v64
        v26['union'] += v55 in v61 | v64
        v26['anywhere'] += v65
        v26['present_unreached'] += v65 and v55 not in v62 | v64 and (not v57)
    v12 = v66(1, v26['n'])
    v27 = {'bytes': v14.v36, 'sample': v14.v45, 'window_lines': v14.v46, 'places': v20, 'questions': v26['n'], 'own': v26['own'] / v12, 'walk_k': v26['walk_k'] / v12, 'walk_2k': v26['walk_2k'] / v12, 'shared': v26['shared'] / v12, 'union_k_shared': v26['union'] / v12, 'anywhere': v26['anywhere'] / v12, 'present_unreached': v26['present_unreached'] / v12, 'reach_over_presence': v26['walk_k'] / v66(1, v26['anywhere'])}
    v1.v91.v67(parents=True, exist_ok=True)
    v1.v68(v107.v92(v27, indent=1), encoding='utf-8')
    v69(f"tape    {v20} places, {v26['n']} questions")
    v69(f"nested  own {v27['own']:.4f}  walk K {v27['walk_k']:.4f}  walk 2K {v27['walk_2k']:.4f}  shared {v27['shared']:.4f}  union {v27['union_k_shared']:.4f}")
    v69(f"TAPE    present anywhere {v27['anywhere']:.4f}   the walk shows the mind {v27['reach_over_presence']:.4f} of it")
    v69(f"GAP     on the tape and reached by nothing we offer: {v27['present_unreached']:.4f}")
    v69(f'\nwritten to {v1}')
    return 0
if v28 == '__main__':
    raise v70(v93())