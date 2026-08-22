"""Is there a PEAK to find? The ceiling of one lens and of two, measured before either is built.

WHY THIS EXISTS. Step 1 spent two levers and they failed in OPPOSITE directions:

    L1 raw count   answerable 0.0440   present@topm 0.1450   argmax goes to globally frequent
    L2 share       answerable 0.0040   present@topm 0.0365   argmax goes to SINGLETONS

Share is cooc(v,w)/total(w), so a value standing once in the whole tape and once beside the lens
scores a perfect 1.0. Count takes the frequent tail, share takes the singleton tail, and neither
is the truth. `chosen_share 0.13` in L1 already said it: THE CO-OCCURRENCE DISTRIBUTION AT A LENS
HAS NO PEAK. Not a wrong rule for picking the maximum - no maximum worth picking.

L3 (two lenses intersected) is the last named lever for step 1, and its mechanism is exactly
"sharpen a flat distribution". But if the tape is thin, two lenses intersect to NOTHING far more
often than they intersect to the truth, and L3 would be spent on an operation the tape cannot
support. 324 and 327 measured a ceiling before building; this does the same.

AND IT TESTS THE ONE THING THE CORPUS LEVER HAS NEVER BEEN ASKED. 335 swept the tape's WIDTH
(more places) and its DEPTH (more text per region) and found neither helps. It never asked for
THICKER PLACES: more mentions per place at a FIXED number of places. A flat co-occurrence
distribution is what thinness looks like from inside, so run this at two corpus sizes with
--addresses held fixed and read `support` - if the peak appears as places thicken, the corpus
lever is real for the first time in this project, and it is real for the constraint and not for
the walk.

    python _audit346_lens.py --bytes 30000000
    python _audit346_lens.py --bytes 120000000        # same --addresses: thicker, not wider
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v5('data/_wikitext103_train.txt')
v1 = v5('results/_stage346_lens.json')
v2 = 8
v3 = 6

def main() -> v4:
    v6 = v76.v27()
    v6.v28('--bytes', type=v4, default=30000000)
    v6.v28('--frame-max', type=v4, default=3)
    v6.v28('--min-fillers', type=v4, default=2)
    v6.v28('--addresses', type=v4, default=1500)
    v6.v28('--lines', type=v4, default=25000)
    v6.v28('--window-lines', type=v4, default=400)
    v6.v28('--seed', type=v4, default=1337)
    v6.v28('--max-questions', type=v4, default=3000)
    v6.v28('--corpus', default=v100(v0))
    v7 = v6.v29()
    v8 = v5(v7.v71).v101('r', encoding='utf-8', errors='ignore').v30(v7.v31)
    v9 = [v78.v77() for v78 in v8.v102('\n') if v45(v78.v77()) >= 80]
    v10 = v9[:v4(0.7 * v45(v9))][:v7.v10]
    v11 = v79.v32(v7.v33)
    v34, v35, v36 = v80.v37(v10, v7.v38, v7.v39)
    if v7.v12:
        v40 = v80.v81(v34, v36)
        v41 = v11.v82(v70(1, v45(v10)))
        v42 = v46(v47)
        for v43 in v83(v7.v12):
            for v103, v55 in v40.v86((v41 + v43) % v45(v10), ()):
                v42[v103].v90(v55)
        v34 = [(v103, v114(v22)) for v103, v22 in v42.v85() if v45({v35[v55] for v55 in v22}) >= v7.v39]
    if v7.v44 and v45(v34) > v7.v44:
        v34 = v11.v84(v34, v7.v44)
    if not v34:
        v74('no tape')
        return 1
    v13 = [[v35[v55] for v55 in v104] for v105, v104 in v34]
    v14 = v45(v13)
    v15 = [v48(v22) for v22 in v13]
    v16 = v46(v47)
    v17 = v48()
    for v49, v50 in v51(v15):
        for v22, v20 in v50.v85():
            v16[v22].v90((v49, v20))
            v17[v22] += v20
    v18 = {}

    def co(v22):
        v20 = v18.v86(v22)
        if v20 is None:
            v20 = v48()
            for v49, v106 in v16[v22]:
                for v95, v115 in v15[v49].v85():
                    v20[v95] += v115
            v18[v22] = v20
        return v20
    v19 = [(v49, v55) for v49 in v83(v14) for v55 in v83(v45(v13[v49])) if v45(v13[v49]) >= 2]
    v11.v52(v19)
    v19 = v19[:v7.v87]
    v20 = v48()
    v53, v54 = ([], [])
    for v49, v55 in v19:
        v56 = v13[v49][v55]
        v57 = v48(v13[v49])
        v57[v56] -= 1
        if v57[v56] <= 0:
            del v57[v56]
        v58 = v47(v57)[:v3]
        if not v58:
            continue
        v20['n'] += 1
        v20['in_own'] += v56 in v57
        v59 = v15[v49]

        def resolved(v22):
            """The lens's counter with THIS place taken out - the same subtraction the stage
            makes, and the reason a leak cannot flatter these numbers."""
            v88 = {}
            v89 = v59 if v116((v121 == v49 for v121, v106 in v16[v22])) else {}
            for v95, v24 in v119(v22).v85():
                if v95 == v22:
                    continue
                v24 -= v89.v86(v95, 0)
                if v24 > 0:
                    v88[v95] = v24
            return v88
        v60 = {v22: v107(v22) for v22 in v58}
        v53.v90(v96((v45(v122) for v122 in v60.v111())) / v45(v60))
        v61 = v62 = None
        v63 = v64 = -1.0
        v65 = False
        for v22, v91 in v60.v85():
            v92 = v114(v91.v85(), key=lambda v123: (-v123[1], v123[0]))[:v2]
            if v56 in {v95 for v95, v106 in v92}:
                v65 = True
            for v95, v24 in v91.v85():
                if v24 > v63:
                    v63, v61 = (v24, v95)
                v108 = v24 / v70(1, v17[v95])
                if v108 > v64:
                    v64, v62 = (v108, v95)
        v20['one_present_topm'] += v65
        v20['one_count_right'] += v61 == v56
        v20['one_share_right'] += v62 == v56
        v66 = v67 = v68 = 0
        v93, v94 = (None, -1.0)
        for v69 in v83(v45(v58)):
            for v50 in v83(v69 + 1, v45(v58)):
                v117, v118 = (v60[v58[v69]], v60[v58[v50]])
                if not v117 or not v118:
                    continue
                v109 = {v95: v120(v117[v95], v118[v95]) for v95 in v117.v124() & v118.v124()}
                if not v109:
                    continue
                v68 += 1
                v54.v90(v45(v109))
                v110 = v114(v109.v85(), key=lambda v123: (-v123[1], v123[0]))[:v2]
                if v56 in {v95 for v95, v106 in v110}:
                    v67 = 1
                for v95, v24 in v109.v85():
                    if v24 > v94:
                        v94, v93 = (v24, v95)
        v20['pair_nonempty'] += v68 > 0
        v20['pair_present_topm'] += v67
        v20['pair_count_right'] += v93 == v56
        v20['pair_pairs'] += v68
    v21 = v48()
    for v22 in v47(v18):
        for v95, v24 in v18[v22].v85():
            if v95 != v22:
                v21[v24] += 1
    v23 = v70(1, v96(v21.v111()))
    v24 = v70(1, v20['n'])
    v25 = {'bytes': v7.v31, 'corpus': v7.v71, 'places': v14, 'questions': v20['n'], 'mentions_per_place': v96((v45(v22) for v22 in v13)) / v14, 'mean_lens_offer': v96(v53) / v70(1, v45(v53)), 'in_own': v20['in_own'] / v24, 'one_present_topm': v20['one_present_topm'] / v24, 'one_count_right': v20['one_count_right'] / v24, 'one_share_right': v20['one_share_right'] / v24, 'pair_nonempty': v20['pair_nonempty'] / v24, 'pair_present_topm': v20['pair_present_topm'] / v24, 'pair_count_right': v20['pair_count_right'] / v24, 'pair_mean_size': v96(v54) / v45(v54) if v54 else v112('nan'), 'pairs_per_question': v20['pair_pairs'] / v24, 'support_1': v21[1] / v23, 'support_2plus': v96((v22 for v103, v22 in v21.v85() if v103 >= 2)) / v23, 'support_3plus': v96((v22 for v103, v22 in v21.v85() if v103 >= 3)) / v23}
    v1.v97.v72(parents=True, exist_ok=True)
    v1.v73(v113.v98(v25, indent=1), encoding='utf-8')
    v74(f"tape    {v14} places, {v25['mentions_per_place']:.2f} mentions each, {v20['n']} questions, in_own {v25['in_own']:.4f}")
    v74(f"THICK   co-occurring pairs seen once {v25['support_1']:.4f}   twice or more {v25['support_2plus']:.4f}   three or more {v25['support_3plus']:.4f}")
    v74(f"ONE     present@{v2} {v25['one_present_topm']:.4f}   count-argmax {v25['one_count_right']:.4f}   share-argmax {v25['one_share_right']:.4f}   offer {v25['mean_lens_offer']:.1f}")
    v74(f"TWO     non-empty {v25['pair_nonempty']:.4f}   present@{v2} {v25['pair_present_topm']:.4f}   count-argmax {v25['pair_count_right']:.4f}   size {v25['pair_mean_size']:.1f}   pairs/q {v25['pairs_per_question']:.1f}")
    if v25['support_2plus'] < 0.1:
        v74('\nNO PEAK TO FIND: over 90% of co-occurring pairs are seen exactly once, so the distribution a lens resolves is made of singletons. No rule for taking its maximum can work, and L3 would be spent on an operation the tape cannot support. Thicken the places before building it.')
    elif v25['pair_present_topm'] > v25['one_present_topm']:
        v74('\nTWO LENSES SHARPEN IT: the intersection reaches more of the truth than one lens does. L3 is worth a lever.')
    else:
        v74("\nTWO LENSES DO NOT SHARPEN IT: the intersection reaches no more than one lens, so L3's mechanism does not hold on this tape.")
    v74(f'\nwritten to {v1}')
    return 0
if v26 == '__main__':
    raise v75(v99())