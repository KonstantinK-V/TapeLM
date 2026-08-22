"""
Stage 261f — One slot, one association: drop the averaging and drop the training.

  python _stage261f_word_votes.py [--smoke] [--soft] [--noise-typo 0.15]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage261_nl_query import collect, ctx_words, jaccard
import _stage24x_lib as L
v0 = v9('results')
v1 = v0 / '_stage261f_log.txt'
v2 = v9('checkpoints/stage191_p1_curve.pt')
v3 = v9('data/_wikitext103_train.txt')
v4 = v74.v10('[A-Za-z][a-z]{2,}')
v5 = 2610
v6 = {'the', 'and', 'that', 'was', 'were', 'for', 'with', 'from', 'his', 'her', 'its', 'their', 'this', 'there', 'which', 'have', 'has', 'had', 'been', 'are', 'not', 'but', 'also', 'who', 'into', 'after', 'before', 'when', 'while', 'than', 'then', 'they', 'them', 'she', 'him'}

def log(v11: v7) -> None:
    v12 = v11 if v11.v138('\n') else v11 + '\n'
    try:
        v139(v12, end='', flush=True)
    except v75:
        v139(v12.v213('ascii', 'replace').v189('ascii'), end='', flush=True)
    v1.v140.v76(parents=True, exist_ok=True)
    with v1.v141('a', encoding='utf-8') as v77:
        v77.v142(v12)
from _tape_index import context_words, nway_strict, vote_arm_fields, vote_rank

def content(v13: v7, v14: v7 | None=None, v15: v8=40) -> v16[v7]:
    return v78(v13, exclude=v14, cap=v15)

def typo(v17: v7, v18: v79, v19: v143.v80) -> v7:
    if v18 <= 0 or v119(v17) < 4:
        return v17
    v20 = v16(v17)
    for v21 in v81(v119(v20)):
        if v19.v143() < v18:
            v20[v21] = v19.v167('abcdefghijklmnopqrstuvwxyz')
    return ''.v82(v20)

def decision_paths(v22: v83, v23: v79) -> v27[v9, v9]:
    v24 = ''
    if v22:
        v24 += '_soft'
    if v23 > 0:
        v24 += f'_typo{v8(v206(v23 * 100)):03d}'
    v25 = v0 / (f'stage261f_decision{v24}.json' if v24 else 'stage261f_decision.json')
    v26 = v0 / (f'stage261f_mini{v24}.md' if v24 else 'stage261f_mini.md')
    return (v25, v26)

def main() -> v8:
    v28 = v144.v84()
    v28.v85('--smoke', action='store_true')
    v28.v85('--entities', type=v8, default=0)
    v28.v85('--distractor-entities', type=v8, default=0)
    v28.v85('--soft', action='store_true')
    v28.v85('--soft-k', type=v8, default=3)
    v28.v85('--noise-typo', type=v79, default=0.0)
    v28.v85('--n-way', type=v8, default=20)
    v29 = v28.v86()
    v87, v88 = v89(v29.v22, v29.v23)
    v1.v90('', encoding='utf-8')
    v30 = v145.v30('cuda' if v145.v190.v168() else 'cpu')
    v19 = v143.v80(v5)
    v145.v91(v5)
    v31 = v92.v92()
    v32 = v29.v93 or (60 if v29.v95 else 400)
    v33 = v29.v94 or (400 if v29.v95 else 4000)
    v34 = 3000 if v29.v95 else 25000
    v96(f'Stage261f word votes start {v211.v204(v212.v205).v164()} device={v30} soft={v29.v22} typo={v29.v23}')
    v97, v97, v98, v99 = v100()
    v35 = v146.v101(v7(v169.v147))
    v36 = v35.v102()
    v37 = v170(v99, v36).v103(v30)
    v37.v104(v145.v171(v2, map_location=v30, weights_only=False)['model'])
    v37.v105()
    for v18 in v37.v106():
        v18.v148(False)
    v38 = v107(v37, v98, v30)
    with v3.v141('r', encoding='utf-8', errors='ignore') as v77:
        v108 = v77.v149(3000000 if v29.v95 else 20000000)
    v39 = [v172.v152() for v172 in v108.v191('\n') if 80 <= v119(v172.v152()) <= 400][:v34]
    v40 = v109(v39, v38)
    v41 = v121(v40)[:v32]
    v19.v110(v41)
    v96(f'  entities with >=2 natural mentions: {v119(v40)} (using {v119(v41)})')
    if v119(v41) < 16:
        v96('  not enough multi-mention entities')
        return 1
    v42: v111[v7, v16[v8]] = v112(v16)
    v43: v16[v7] = []
    v44 = []
    for v45 in v41:
        v113 = v40[v45]
        v150, v151 = (v113[0], v113[1])
        v114 = v150['line'][v192(0, v150['start'] - 140):v193(v119(v150['line']), v150['end'] + 140)]
        v115 = v151['line'][v192(0, v151['start'] - 200):v151['start']].v152()
        v20 = v153(v114, exclude=v45)
        v116 = v153(v115, exclude=v45)
        if v119(v20) < 4 or v119(v116) < 4:
            continue
        v117 = v119(v43)
        v43.v154(v45)
        for v118 in v20:
            v42[v118].v154(v117)
        v44.v154({'ent': v45, 'cid': v117, 'qwords': v116, 'overlap': v194(v207(v114, v45), v207(v115, v45))})
    v46 = v119(v43)
    v47 = v120(v43)
    for v48 in v39:
        if v119(v43) >= v46 + v33:
            break
        for v11 in v173.v155(v48):
            v45 = v11.v174(1)
            if v119(v45) < 5 or v45 in v47:
                continue
            v175, v176 = (v192(0, v11.v214() - 140), v193(v119(v48), v11.v215() + 140))
            v20 = v153(v48[v175:v176], exclude=v45)
            if v119(v20) < 4:
                continue
            v117 = v119(v43)
            v43.v154(v45)
            for v118 in v20:
                v42[v118].v154(v117)
            v47.v177(v45)
            if v119(v43) >= v46 + v33:
                break
    v49 = v121(v42)
    v50 = {v118: 1.0 / v195.v96(2.0 + v119(v42[v118])) for v118 in v49}
    v51 = v79(v160.v178([v55['overlap'] for v55 in v44])) if v44 else 0.0
    v96(f'  candidates={v119(v43)} ({v46} asked + {v119(v43) - v46} distractor) | vocab={v119(v49)} postings={v162((v119(v184) for v184 in v42.v43()))} | eval={v119(v44)} overlap median={v51:.3f}')
    if v119(v44) < 16:
        v96('  not enough usable pairs')
        return 1
    v52 = None
    if v29.v22:
        v52 = v145.v216([v38.v218([v118])[0] for v118 in v49], 0).v103(v30).v79()
        v96(f'  soft mode: word vocabulary embedded ({v52.v208[0]} x {v52.v208[1]})')

    def vote(v122: v16[v7]) -> v111[v8, v79]:
        v123: v111[v8, v79] = v112(v79)
        for v118 in v122:
            if v29.v22 and v52 is not None:
                v179 = v38.v218([v118])[0].v103(v30).v79()
                v196, v197 = v145.v198(v52 @ v179, v193(v29.v136, v52.v217(0)))
                for v199, v181 in v200(v196.v209(), v197.v209()):
                    if v199 <= 0:
                        continue
                    v201 = v49[v181]
                    for v117 in v42[v201]:
                        v123[v117] += v199 * v50[v201]
            else:
                for v117 in v42.v185(v118, ()):
                    v123[v117] += v50[v118]
        return v123
    v53 = v143.v80(v5 + 3)
    v54 = v143.v80(v5 + 5)
    v124, v125, v126 = ([], [], [])
    for v55 in v44:
        v127 = [v180(v118, v29.v23, v53) for v118 in v55['qwords']]
        v123 = v156(v127)
        v157, v158 = v159(v123, v55['cid'], v119(v43))
        v124.v154(v158)
        v128 = [v181 for v181 in v54.v210(v81(v119(v43)), v193(v29.v134 * 3, v119(v43))) if v181 != v55['cid']][:v29.v134 - 1]
        v125.v154(v8(v202(v157, (v123.v185(v181, 0.0) for v181 in v128))))
        v126.v154({'gold_score': v79(v157), 'rank': v158, 'low_overlap': v55['overlap'] <= v51})
    v56 = v129(v126)
    v57 = v160.v130(v124, dtype=v160.v161)
    v58 = {'top1': v79(v160.v182(v57 == 1)), 'mrr': v79(v160.v182(1.0 / v57)), 'median_rank': v79(v160.v178(v57)), f'acc_{v29.v134}way': v79(v160.v182(v125)), f'chance_{v29.v134}way': 1.0 / v29.v134, 'top1_low_overlap': v56['top1_low_overlap'], 'top1_high_overlap': v56['top1_high_overlap'], 'tie_at_zero_frac': v56['tie_at_zero_frac'], 'n': v119(v124)}
    v96('votes: ' + v188.v165(v58))
    v96('silence: ' + v188.v165(v56))
    v59 = v16(v81(v119(v43)))
    v143.v80(v5 + 7).v110(v59)
    v60 = {v118: [v59[v183] for v183 in v184] for v118, v184 in v42.v44()}
    v131, v132, v133 = ([], [], [])
    v61 = v143.v80(v5 + 5)
    for v55 in v44:
        v123: v111[v8, v79] = v112(v79)
        for v118 in v55['qwords']:
            for v117 in v60.v185(v118, ()):
                v123[v117] += v50[v118]
        v157, v158 = v159(v123, v55['cid'], v119(v43))
        v131.v154(v158)
        v128 = [v181 for v181 in v61.v210(v81(v119(v43)), v193(v29.v134 * 3, v119(v43))) if v181 != v55['cid']][:v29.v134 - 1]
        v132.v154(v8(v202(v157, (v123.v185(v181, 0.0) for v181 in v128))))
        v133.v154({'gold_score': v79(v157), 'rank': v158, 'low_overlap': v55['overlap'] <= v51})
    v62 = v129(v133)
    v63 = {'top1': v79(v160.v182(v160.v130(v131) == 1)), f'acc_{v29.v134}way': v79(v160.v182(v132)), 'tie_at_zero_frac': v62['tie_at_zero_frac'], 'silence': v62}
    v96('popularity floor (postings repointed, counts preserved): ' + v188.v165({v203: v184 for v203, v184 in v63.v44() if v203 != 'silence'}))
    v64 = v58[f'acc_{v29.v134}way']
    v65 = 1.0 / v29.v134
    v66 = v64 >= 0.3
    v67 = v64 >= v65 + 0.1
    v68 = not v195.v186(v56['top1_low_overlap_given_vote']) and v56['top1_low_overlap_given_vote'] > 0.0
    v69 = v63['top1'] <= 0.02
    v70 = v64 >= v63[f'acc_{v29.v134}way'] + 0.15
    v71 = v58['top1'] >= 0.3
    if v69 and v70 and v71 and v68:
        v135 = 'WORD_VOTES_OK'
    elif v69 and v70 and v66:
        v135 = 'WORD_VOTES_BEATS_MEAN'
    elif v69 and v67:
        v135 = 'WORD_VOTES_SIGNAL_ONLY'
    else:
        v135 = 'WORD_VOTES_NO'
    v72 = {'stage': '261f', 'overall': v135, 'soft': v29.v22, 'soft_k': v29.v136, 'noise_typo': v29.v23, 'n_way': v29.v134, 'candidates': v119(v43), 'asked': v46, 'distractors': v119(v43) - v46, 'vocab': v119(v49), 'postings': v162((v119(v184) for v184 in v42.v43())), 'overlap_median': v51, 'trained_parameters': 0, 'fp_version': v187.v163(), 'read': {f'acc_{v29.v134}way': v58.v185(f'acc_{v29.v134}way'), 'full_bank_top1': v58['top1'], 'full_bank_mrr': v58['mrr'], 'full_bank_median_rank': v58['median_rank'], 'tie_at_zero_frac': v56['tie_at_zero_frac'], 'top1_low_overlap': v56['top1_low_overlap'], 'top1_low_overlap_given_vote': v56['top1_low_overlap_given_vote'], 'low_overlap_miss_is_silence_frac': v56['low_overlap_miss_is_silence_frac']}, 'silence': v56, 'gates': {'G_signal': v67, 'G_beats_mean_fp': v66, 'G_low_overlap_works': v68, 'G_open_top1': v71, 'G_causal_top1': v69, 'G_beats_popularity_20way': v70, 'G_low_overlap_uses': 'top1_low_overlap_given_vote'}, 'summary': {'votes': v58, 'popularity_floor': v63, 'reference_261_ctx_fp_mean': {'acc_20way': 0.226, 'top1': 0.034, 'top1_low_overlap': 0.0}}, 'note': 'Zero-train word postings + idf. n-way pessimistic (gold > distractor). silence.tie_at_zero_frac is permanent: share of queries with gold score 0 (index silent). top1_low_overlap hole is mostly silence — see low_overlap_miss_is_silence_frac and top1_low_overlap_given_vote. Causal read still on top1.', 'timestamp': v211.v204(v212.v205).v164(), 'wall_s': v92.v92() - v31}
    v87.v90(v188.v165(v72, indent=2), encoding='utf-8')
    v88.v90(f"# Stage 261f word votes (zero-train)\n\n**{v135}** candidates={v119(v43)} vocab={v119(v49)} soft={v29.v22} typo={v29.v23}\n\n- {v29.v134}-way **{v64:.3f}** (chance {v65:.3f}, popularity floor {v63[f'acc_{v29.v134}way']:.3f}) vs 261 ctx_fp mean **0.226**\n- open top1 **{v58['top1']:.3f}** vs popularity floor **{v63['top1']:.3f}** (261 mean: 0.034), mrr {v58['mrr']:.3f}, median rank {v58['median_rank']:.0f}\n- by overlap: low **{v56['top1_low_overlap']:.3f}** vs high **{v56['top1_high_overlap']:.3f}**\n- **silence:** tie_at_zero **{v56['tie_at_zero_frac']:.3f}** (low-ov **{v56['tie_at_zero_frac_low_overlap']:.3f}** / high **{v56['tie_at_zero_frac_high_overlap']:.3f}**); low-ov miss is silence **{v56['low_overlap_miss_is_silence_frac']:.3f}**; top1 low-ov | gold>0 **{v56['top1_low_overlap_given_vote']:.3f}**\n- trained parameters: **0**\n", encoding='utf-8')
    v96(v188.v165({'overall': v135, 'gates': v72['gates']}, indent=2))
    return 0
if v73 == '__main__':
    raise v137(v166())