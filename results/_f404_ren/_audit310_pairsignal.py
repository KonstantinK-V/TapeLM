"""Is there ANY pair signal left once co-occurrence is subtracted - measured before building.

WHY. 309b was a clean negative: evidence-at-fill doubled per-hole accuracy (0.26 -> 0.53) and
pair hits sat exactly on the product of the mind's own marginals, twice, on two different world
constructions. Before that is read as "Phi cannot compose", the other reading has to be priced:
COMP_STRICT is DEFINED as "the pair co-occurs nowhere" - we subtracted the joint statistic from
the subset by construction. walk_only was winnable because the information existed (the truth
stood at a reachable place); for a strict pair the only information that can remain is SECOND
ORDER: do the HOMES of the two values agree - shared rare words, similar places - more for the
true pair than for a wrong one? If yes, composition has fuel and the verb needs that channel.
If no, P(a,b) = P(a)P(b) is the CORRECT posterior on this tape and no mind should beat it -
the negative is a property of the corpus, not of Phi, and the lever is a denser corpus.

This is 305's shape exactly: Kostya's confirmation channel was validated by measuring
separation (0.647) before one line of the verb was written. Same discipline, one level up.

    python _audit310_pairsignal.py --bytes 30000000 --frame-max 3 --sample region         --window-lines 400 --min-fillers 2
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v5('data/_wikitext103_train.txt')
v1 = v5('results/_stage310_pairsignal.json')
v2 = 8
v3 = 3

def main() -> v4:
    v6 = v87.v30()
    v6.v31('--bytes', type=v4, default=30000000)
    v6.v31('--frame-max', type=v4, default=3)
    v6.v31('--min-fillers', type=v4, default=2)
    v6.v31('--addresses', type=v4, default=1500)
    v6.v31('--lines', type=v4, default=25000)
    v6.v31('--window-lines', type=v4, default=400)
    v6.v31('--sample', choices=('uniform', 'region'), default='region')
    v6.v31('--seed', type=v4, default=1337)
    v6.v31('--pairs-per-line', type=v4, default=4)
    v6.v31('--max-questions', type=v4, default=4000)
    v7 = v6.v32()
    v8 = v0.v112('r', encoding='utf-8', errors='ignore').v33(v7.v34)
    v9 = [v89.v88() for v89 in v8.v113('\n') if v51(v89.v88()) >= 80]
    v10 = v9[:v4(0.7 * v51(v9))][:v7.v10]
    v11 = v90.v35(v7.v36)
    v12, v37, v38 = v91.v39(v10, v7.v40, v7.v41)
    if v7.v42 == 'region':
        if v7.v43:
            v92 = v91.v114(v12, v38)
            v93 = v11.v115(v82(1, v51(v10)))
            v94 = v52(v53)
            for v95 in v55(v7.v43):
                for v119, v50 in v92.v117((v93 + v95) % v51(v10), ()):
                    v94[v119].v100(v50)
            v12 = [(v119, v56(v70)) for v119, v70 in v94.v68() if v51({v37[v50] for v50 in v70}) >= v7.v41]
        else:
            v12 = v91.v116(v12, v37, v38, v51(v10), v7.v96, v11, v7.v41)
    elif v7.v96 and v51(v12) > v7.v96:
        v12 = v11.v42(v12, v7.v96)
    if not v12:
        v85('no tape')
        return 1
    v44, v45, v46, v47 = ([], [], [], [])
    for (v97, v98, v99), v48 in v12:
        v49 = f"{' '.v120(v98)}|{' '.v120(v99)}"
        for v50 in v48:
            v44.v100(v49)
            v45.v100(v37[v50])
            v46.v100(v38[v50])
            v47.v100(v50)
    v13 = v51(v44)
    v14 = v52(v53)
    v15 = v52(v53)
    v16 = v52(v54)
    for v17 in v55(v13):
        v14[v46[v17]].v100(v17)
        if v46[v17] not in v15[v45[v17]]:
            v15[v45[v17]].v100(v46[v17])
        v16[v44[v17]].v101(v45[v17])
    v18 = v56({v46[v17] for v17 in v55(v13)})
    v19 = v57()
    v20 = {}
    for v21 in v18:
        v58 = v54(v10[v21].v113())
        v20[v21] = v58
        for v59 in v58:
            v19[v59] += 1
    for v21 in v18:
        v20[v21] = {v59 for v59 in v20[v21] if v19[v59] <= v3}

    def pair_signal(v60, v61, v62):
        """Best rare-word overlap between a home line of x and a home line of y - the two
        lines DIFFERENT and both different from the question's, so co-occurrence (the bag)
        and the question's own record can contribute nothing."""
        v63 = 0
        v64 = [v89 for v89 in v15[v60] if v89 != v62][:v2]
        v65 = [v89 for v89 in v15[v61] if v89 != v62][:v2]
        for v66 in v64:
            v102 = v20.v117(v66, ())
            if not v102:
                continue
            for v103 in v65:
                if v103 == v66:
                    continue
                v63 = v82(v63, v51(v102 & v20.v117(v103, v54())))
        return v63
    v22 = v52(v54)
    for v21, v67 in v14.v68():
        v69 = {v45[v17] for v17 in v67}
        for v70 in v69:
            v22[v70] |= v69
    v23 = []
    for v21, v67 in v14.v68():
        if v51(v67) < 2:
            continue
        v71 = [(v75, v76) for v121, v75 in v122(v67) for v76 in v67[v121 + 1:] if v44[v75] != v44[v76] and v125(v47[v75] - v47[v76]) > v7.v40]
        v11.v72(v71)
        v23.v104(v71[:v7.v123])
    v11.v72(v23)
    if v7.v24:
        v23 = v23[:v7.v24]
    v25 = v57()
    v73, v74 = ([], [])
    for v75, v76 in v23:
        v105, v106, v62 = (v45[v75], v45[v76], v46[v75])
        v25['n'] += 1
        v77 = v107((1 for v124 in v15[v105] if v124 in v54(v15[v106])))
        if v77 > 1:
            v25['cooccur'] += 1
            continue
        v78 = [v59 for v59 in v56(v16[v44[v76]]) if v59 != v106]
        if not v78:
            v25['no_distractor'] += 1
            continue
        v79 = v78[v11.v115(v51(v78))]
        v80 = v108(v105, v106, v62)
        v81 = v108(v105, v79, v62)
        v25['strict'] += 1
        v73.v100(v80)
        v74.v100(v81)
        if v80 != v81:
            v25['differ'] += 1
            v25['true_higher'] += v80 > v81
        v25['nonzero'] += v80 > 0 or v81 > 0
    v26 = v82(1, v25['strict'])
    v27 = v82(1, v25['differ'])
    v28 = {'bytes': v7.v34, 'frame_max': v7.v40, 'sample': v7.v42, 'window_lines': v7.v43, 'min_fillers': v7.v41, 'slots': v13, 'questions': v25['n'], 'strict': v25['strict'], 'cooccur_dropped': v25['cooccur'], 'no_distractor': v25['no_distractor'], 'signal_nonzero': v25['nonzero'] / v26, 'mean_true': v107(v73) / v26, 'mean_wrong': v107(v74) / v26, 'differ_rate': v25['differ'] / v26, 'separates': v25['true_higher'] / v27}
    v1.v109.v83(parents=True, exist_ok=True)
    v1.v84(v118.v110(v28, indent=1), encoding='utf-8')
    v85(f"tape    {v13} slots, {v25['n']} two-hole questions, {v25['strict']} strict (pair co-occurs nowhere else, distractor exists)")
    v85(f"signal  nonzero on {v28['signal_nonzero']:.4f} of strict   mean true {v28['mean_true']:.3f} vs wrong {v28['mean_wrong']:.3f}")
    v85(f"DECIDE  differ {v28['differ_rate']:.4f}   separates {v28['separates']:.4f}   (0.5 = coin; 305's confirm read 0.647 and was built)")
    v85(f'\nwritten to {v1}')
    return 0
if v29 == '__main__':
    raise v86(v111())