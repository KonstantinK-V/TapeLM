"""
Stage 264 — Balance alternatives: flat mean, idf-mean, votes, cascade, fusion.

Open-bank retrieval (261f exam, zero train) compares all modes on the same items.
Optional --glue runs 256 decode EM per mode (263 harness).

  python _stage264_retrieval_blend.py [--smoke] [--glue]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage24x_lib as L
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage261_nl_query import collect, ctx_words, jaccard
from _stage261f_word_votes import content
from _retrieval_modes import CASCADE_POOL, FUSION_LAM, cascade_order, cosine_scores, ctx_vector, fusion_scores, rank_from_order, rank_from_scores, vote_scores
from _tape_index import nway_strict, vote_arm_fields
v0 = v9('results')
v1 = v0 / 'stage264_decision.json'
v2 = v0 / 'stage264_mini.md'
v3 = v0 / '_stage264_log.txt'
v4 = v9('checkpoints/stage191_p1_curve.pt')
v5 = v9('data/_wikitext103_train.txt')
v6 = 2640
v7 = ('mean', 'idf_mean', 'votes', 'cascade', 'fusion')

def log(v10: v64) -> None:
    v11 = v10 if v10.v132('\n') else v10 + '\n'
    try:
        v133(v11, end='', flush=True)
    except v65:
        v133(v11.v197('ascii', 'replace').v180('ascii'), end='', flush=True)
    v3.v134.v66(parents=True, exist_ok=True)
    with v3.v135('a', encoding='utf-8') as v67:
        v67.v136(v11)

def build_open_bank(v12: v68, v13: v69):
    v14 = v137.v70(v6)
    v15 = 60 if v13 else 400
    v16 = 400 if v13 else 4000
    v17 = 3000 if v13 else 25000
    with v5.v135('r', encoding='utf-8', errors='ignore') as v67:
        v71 = v67.v138(3000000 if v13 else 20000000)
    v18 = [v163.v141() for v163 in v71.v181('\n') if 80 <= v83(v163.v141()) <= 400][:v17]
    v19 = v72(v18, v12)
    v20 = v85(v19)[:v15]
    v14.v73(v20)
    v21: v74[v64, v76[v8]] = v75(v76)
    v22: v76[v76[v64]] = []
    v23: v76[v64] = []
    v24 = []
    for v25 in v20:
        v77 = v19[v25]
        v139, v140 = (v77[0], v77[1])
        v78 = v139['line'][v122(0, v139['start'] - 140):v182(v83(v139['line']), v139['end'] + 140)]
        v79 = v140['line'][v122(0, v140['start'] - 200):v140['start']].v141()
        v32 = v142(v78, exclude=v25)
        v80 = v142(v79, exclude=v25)
        if v83(v32) < 4 or v83(v80) < 4:
            continue
        v81 = v83(v23)
        v23.v130(v25)
        v22.v130(v32)
        for v82 in v32:
            v21[v82].v130(v81)
        v24.v130({'ent': v25, 'cid': v81, 'qwords': v80, 'overlap': v183(v191(v78, v25), v191(v79, v25))})
    v26 = v83(v23)
    v27 = v84(v23)
    for v28 in v18:
        if v83(v23) >= v26 + v16:
            break
        for v10 in v164.v143(v28):
            v25 = v10.v165(1)
            if v83(v25) < 5 or v25 in v27:
                continue
            v166, v167 = (v122(0, v10.v198() - 140), v182(v83(v28), v10.v199() + 140))
            v32 = v142(v28[v166:v167], exclude=v25)
            if v83(v32) < 4:
                continue
            v81 = v83(v23)
            v23.v130(v25)
            v22.v130(v32)
            for v82 in v32:
                v21[v82].v130(v81)
            v27.v168(v25)
            if v83(v23) >= v26 + v16:
                break
    v29 = v85(v21)
    v30 = {v82: 1.0 / v184.v111(2.0 + v83(v21[v82])) for v82 in v29}
    return (v24, v22, v21, v30, v23, v26)

@v89.v36()
def precompute_keys(v12: v68, v22: v76[v76[v64]], v30: v74[v64, v90], v31):
    v86, v87 = ([], [])
    for v32 in v22:
        v86.v130(v146(v12, v32, None))
        v87.v130(v146(v12, v32, v30))
    v33 = v86[0] if v86[0] is not None else v89.v144(256, device=v31)
    v34 = v89.v192([v204 if v204 is not None else v33 for v204 in v86]).v90().v88(v31)
    v35 = v89.v192([v204 if v204 is not None else v33 for v204 in v87]).v90().v88(v31)
    return (v34, v35)

def eval_mode(v37: v64, v24, v22, v21, v30, v34, v35, v12, v38: v8, v39: v90):
    v40 = v83(v22)
    v41 = v137.v70(v6 + 5)
    v91, v92, v93 = ([], [], [])
    for v42 in v24:
        v94 = v42['qwords']
        v95 = v42['cid']
        v96 = v145(v94, v21, v30)
        v97 = v146(v12, v94, None)
        v98 = v146(v12, v94, v30)
        v99 = v42['overlap'] <= v39
        if v37 == 'mean':
            v147 = v169(v97, v34)
        elif v37 == 'idf_mean':
            v147 = v169(v98, v35)
        elif v37 == 'votes':
            v147 = v96
        elif v37 == 'cascade':
            v193 = v200(v96, v97, v34, v40, v128)
            v100 = v201(v193, v95)
            v91.v130(v100)
            v101 = [v170 for v170 in v41.v194(v203(v40), v182(v38 * 3, v40)) if v170 != v95][:v38 - 1]
            if v90(v96.v149(v95, 0.0)) <= 0.0:
                v92.v130(0)
            else:
                v147 = {v81: -v90(v207 + 1) for v207, v81 in v208(v193)}
                v102 = v147[v95]
                v92.v130(v8(v185(v102, (v147.v149(v170, -v90(v40 + 1)) for v170 in v101))))
            v93.v130({'gold_score': v90(v96.v149(v95, 0.0)), 'rank': v100, 'low_overlap': v99})
            continue
        elif v37 == 'fusion':
            v202 = v169(v97, v34)
            v147 = v205(v96, v202, v127)
        else:
            raise v206(v37)
        v100 = v148(v147, v95, v40)
        v91.v130(v100)
        v101 = [v170 for v170 in v41.v194(v203(v40), v182(v38 * 3, v40)) if v170 != v95][:v38 - 1]
        v102 = v147.v149(v95, 0.0)
        v92.v130(v8(v185(v102, (v147.v149(v170, 0.0) for v170 in v101))))
        v103 = v90(v96.v149(v95, 0.0)) if v37 in ('votes', 'fusion') else v90(v102)
        v93.v130({'gold_score': v103, 'rank': v100, 'low_overlap': v99})
    v43 = v104(v93)
    v44 = v150.v105(v91, dtype=v150.v151)
    return {'top1': v90(v150.v171(v44 == 1)), 'mrr': v90(v150.v171(1.0 / v44)), 'median_rank': v90(v150.v156(v44)), f'acc_{v38}way': v90(v150.v171(v92)), f'chance_{v38}way': 1.0 / v38, 'top1_low_overlap': v43['top1_low_overlap'], 'top1_high_overlap': v43['top1_high_overlap'], 'tie_at_zero_frac': v43['tie_at_zero_frac'], 'silence': v43, 'n': v83(v91)}

def main() -> v8:
    v45 = v152.v106()
    v45.v107('--smoke', action='store_true')
    v45.v107('--glue', action='store_true', help='256 decode EM per mode (slow)')
    v45.v107('--n-way', type=v8, default=20)
    v45.v107('--steps', type=v8, default=0)
    v45.v107('--topk', type=v8, default=8)
    v46 = v45.v108()
    v3.v109('', encoding='utf-8')
    v31 = v89.v31('cuda' if v89.v186.v172() else 'cpu')
    v47 = v110.v110()
    v111(f'Stage264 retrieval blend start {v195.v189(v196.v190).v159()} device={v31}')
    v112, v112, v113, v114 = v115()
    v48 = v153.v116(v64(v173.v154))
    v49 = v174(v114, v48.v187()).v88(v31)
    v49.v117(v89.v175(v4, map_location=v31, weights_only=False)['model'])
    v49.v118()
    for v50 in v49.v119():
        v50.v155(False)
    v12 = v68(v49, v113, v31)
    v24, v22, v21, v30, v23, v26 = v120(v12, v46.v13)
    if v83(v24) < 16:
        v111('not enough eval pairs')
        return 1
    v39 = v90(v150.v156([v42['overlap'] for v42 in v24]))
    v111(f'  candidates={v83(v23)} ({v26} exam) eval={v83(v24)} overlap_med={v39:.3f}')
    v34, v35 = v121(v12, v22, v30, v31)
    v51 = {}
    for v37 in v7:
        v51[v37] = v157(v37, v24, v22, v21, v30, v34, v35, v12, v46.v38, v39)
        v111(f'  [{v37}] ' + v179.v160(v51[v37]))
    v52 = v51['votes'][f'acc_{v46.v38}way']
    v53 = v51['mean'][f'acc_{v46.v38}way']
    v54 = v122(v7, key=lambda v10: v51[v10][f'acc_{v46.v38}way'])
    v55 = v51[v54][f'acc_{v46.v38}way']
    v56 = v122(v7, key=lambda v10: (v51[v10]['top1'], v51[v10][f'acc_{v46.v38}way']))
    v57 = v51['idf_mean'][f'acc_{v46.v38}way'] >= v53 + 0.05
    v58 = v123((v51[v10]['top1'] >= v51['votes']['top1'] + 0.03 for v10 in ('cascade', 'fusion')))
    v59 = v51['cascade']['top1'] >= v51['votes']['top1'] + 0.03
    if v58:
        v124 = 'BLEND_BEATS_SINGLE'
    elif v51['idf_mean'][f'acc_{v46.v38}way'] >= v53 + 0.1:
        v124 = 'IDF_MEAN_FIXES_FLAT'
    elif v56 == 'votes' and v51['votes']['top1'] >= v51['mean']['top1'] + 0.05:
        v124 = 'VOTES_BEST_OPEN_BANK'
    else:
        v124 = 'NO_CLEAR_WINNER'
    v60 = None
    if v46.v61:
        import _stage263_votes_vs_mean as s263
        v60 = {}
        v125 = v46.v125 or (200 if v46.v13 else 800)
        v126 = ('cosine', 'idf_mean', 'votes', 'cascade', 'fusion')
        for v37 in v126:
            v60[v37] = v188.v176(v37, v31, v46.v13, v125, v46.v177)
        v111('  glue summary: ' + v179.v160(v60))
    v62 = {'stage': 264, 'overall': v124, 'modes': v76(v7), 'fusion_lambda': v127, 'cascade_pool': v128, 'candidates': v83(v23), 'n_exam': v26, 'eval_pairs': v83(v24), 'gates': {'G_idf_mean_beats_flat_mean': v57, 'G_blend_beats_votes': v58, 'G_cascade_beats_votes': v59}, 'summary': {'retrieval': v51, 'read_full_bank': {v10: {'full_bank_top1': v51[v10]['top1'], 'full_bank_median_rank': v51[v10]['median_rank'], f'acc_{v46.v38}way': v51[v10].v149(f'acc_{v46.v38}way'), 'tie_at_zero_frac': v51[v10].v149('tie_at_zero_frac'), 'top1_low_overlap_given_vote': v51[v10].v149('silence', {}).v149('top1_low_overlap_given_vote'), 'low_overlap_miss_is_silence_frac': v51[v10].v149('silence', {}).v149('low_overlap_miss_is_silence_frac')} for v10 in v7}, 'best_mode_top1': v56, 'best_mode_20way': v54, 'reference_261f': {'votes_20way_strict': 0.432, 'votes_20way_legacy_ge': 0.601, 'mean_20way': 0.226, 'tie_at_zero_frac': 0.488}}, 'fp_version': v178.v158(), 'glue_em': v60, 'note': 'n-way pessimistic via nway_strict for ALL modes (cascade: score=-rank). Overall uses top1 as headline (20-way alone had crowned cascade after a metric bug). silence.* permanent. Low-overlap hole ≈ silence → route to sem-q (258) when votes silent.', 'timestamp': v195.v189(v196.v190).v159(), 'wall_s': v110.v110() - v47}
    v1.v109(v179.v160(v62, indent=2), encoding='utf-8')
    v18 = [f'# Stage 264 retrieval blend\n\n**{v124}** eval={v83(v24)} candidates={v83(v23)}\n\n']
    for v37 in v7:
        v44 = v51[v37]
        v129 = v44.v149('silence') or {}
        v18.v130(f"- **{v37}**: 20-way **{v44[f'acc_{v46.v38}way']:.3f}** top1 **{v44['top1']:.3f}** low-ov **{v44['top1_low_overlap']:.3f}** tie0 **{v44.v149('tie_at_zero_frac', v90('nan')):.3f}** miss=silence **{v129.v149('low_overlap_miss_is_silence_frac', v90('nan')):.3f}**\n")
    v18.v130(f"\nBest top1: **{v56}** ({v51[v56]['top1']:.3f}) · best 20-way: **{v54}** ({v55:.3f})\n")
    v2.v109(''.v161(v18), encoding='utf-8')
    v111(v179.v160({'overall': v124, 'best_top1': v56, 'best_20way': v54}, indent=2))
    return 0
if v63 == '__main__':
    raise v131(v162())