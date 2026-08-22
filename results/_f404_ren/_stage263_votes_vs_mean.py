"""
Stage 263 — Was the context mean a bottleneck everywhere, or only in 261?

`ctx_fp(text) = normalize(mean over up to 40 word fingerprints)` was decided in 194 and never
questioned again. Every slot key since is `norm(fp(anchor) + ctx_fp(...))` - 194 through 198,
255, 256, 257, 258, 260. Seventy stages on one unexamined choice.

261f found the first exam where the ceiling of that mean is visible: one posting per content
word, zero trained parameters, scored 20-way 0.601 against the mean's 0.226 and open top1 0.246
against 0.034. A mean of forty unit vectors is nearly a constant direction; it cannot
discriminate. On the other stages the candidate set is closed - four ways, or eight relations of
one subject - so the mean is good enough and the ceiling never shows.

That is not evidence there is no ceiling. This runs 256's exam, which has a decode and a
published number to beat (em_glue 0.667), with ONE thing changed:

    cosine arm   retrieve by cos(q, key)              the mean, as shipped
    votes arm    retrieve by word postings + idf      one slot per word, nothing fitted

Everything else is identical and shared: same facts, same tape, same glue, same copy mixture,
same gate, same training loop, same seeds. Only the retrieval step differs, so a difference in
EM is a difference in retrieval and nothing else.

  python _stage263_votes_vs_mean.py [--smoke]
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
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, DEFAULT_CUE, DEFAULT_FACT_TMPL, SlotBias, SlotPostings, TapeView, copy_dist, full_bank_cue_summary, hidden_and_logits, mix_logprob
from _stage261f_word_votes import content
from _retrieval_modes import CASCADE_POOL, FUSION_LAM, cascade_order, cosine_scores, ctx_vector, fusion_scores, vote_scores
v0 = v11('results')
v1 = v0 / 'stage263_decision.json'
v2 = v0 / 'stage263_mini.md'
v3 = v0 / '_stage263_log.txt'
v4 = v11('checkpoints/stage191_p1_curve.pt')
v5 = v11('checkpoints/stage253_joint_l02.pt')
v6 = v11('data/_wikitext103_train.txt')
v7 = 256

def log(v12: v91) -> None:
    v13 = v12 if v12.v180('\n') else v12 + '\n'
    try:
        v181(v13, end='', flush=True)
    except v92:
        v181(v13.v274('ascii', 'replace').v194('ascii'), end='', flush=True)
    v3.v182.v93(parents=True, exist_ok=True)
    with v3.v183('a', encoding='utf-8') as v94:
        v94.v184(v13)

def build(v14: v95, v15: v96, v16: v10, v17, v18: v97):
    """256's data build, kept verbatim, plus the write-context words each slot came from."""
    v19 = v185.v98(v7)
    v20 = 8 if v18 else 48
    v21 = 150 if v18 else 1200
    v22 = 400 if v18 else 6000
    with v6.v183('r', encoding='utf-8', errors='ignore') as v94:
        v99 = v94.v186(1000000 if v18 else 6000000)
    v23 = v100(v9.v187((v12.v244(1) for v12 in v243.v191(v99) if v125(v12.v244(1)) >= 5)))
    v19.v101(v23)
    v24 = [v240.v147() for v240 in v99.v269('\n') if v125(v240.v147()) >= 60][:v22]
    v25 = [v114 for v114 in v270(v110(v23), v19, v20 + 30) if v125(v114) >= 5][:v20]
    v26 = [{'S': v188, 'value': v23[v142], 'sent': v271.v241(S=v188, V=v23[v142]), 'fid': f'f{v142}', 'glue_train': v142 % 2 == 0} for v142, v188 in v113(v25)]
    v102, v103, v104, v105 = ([], [], [], [])
    v106, v107 = ([], [])
    for v27 in v26:
        v108 = v14.v242([v27['S']])[0]
        v109 = v120(v27['sent'], exclude=v27['value'])
        v41 = v14.v189(v27['sent'], exclude=v27['value'])
        v102.v190(v250.v196(v108 + v41, dim=-1) if v41 is not None else v108)
        v105.v190(v108)
        v103.v190(v27['value'])
        v104.v190(v109)
    v28 = v110(v103)
    for v29 in v24:
        if v125(v103) >= v20 + v21:
            break
        for v12 in v243.v191(v29):
            v78 = v12.v244(1)
            if v125(v78) < 5 or v78 in v28:
                continue
            v245, v246 = (v215(0, v12.v291() - 120), v202(v125(v29), v12.v292() + 120))
            v41 = v14.v189(v29[v245:v246], exclude=v78)
            if v41 is None:
                continue
            v42 = [v114 for v114 in v195.v122(v29[v245:v12.v291()]) if v114 != v78]
            if not v42:
                continue
            v192 = v14.v242([v42[-1]])[0]
            v109 = v120(v29[v245:v246], exclude=v78)
            v102.v190(v250.v196(v192 + v41, dim=-1))
            v105.v190(v192)
            v193 = v14.v189(v29[v245:v12.v291()])
            if v193 is not None:
                v106.v190(v250.v196(v192 + v193, dim=-1))
                v107.v190(v125(v103))
            v103.v190(v78)
            v104.v190(v109)
            v28.v247(v78)
            if v125(v103) >= v20 + v21:
                break
    v30: v9[v91, v100[v10]] = v111(v100)
    for v112, v109 in v113(v104):
        for v114 in v109:
            v30[v114].v190(v112)
    v31 = {v114: 1.0 / v272.v69(2.0 + v125(v47)) for v114, v47 in v30.v248()}
    v32 = []
    for v115, v109 in v116(v105, v104):
        v117 = v121(v14, v109, v31)
        v32.v190(v250.v196(v115 + v117, dim=-1) if v117 is not None else v115)
    v33 = v118(v148.v249(v102, 0).v119(v17), v103, v15, v16, ctxw=v104)
    v34 = v148.v249(v32, 0).v119(v17)
    v35 = v148.v249(v106).v119(v17).v138() if v106 else None
    v36 = v148.v123(v107, device=v17) if v107 else None
    return (v26, v33, v104, v35, v36, v34)

class Votes(v8):
    """Alias for stage 263 ablation arms."""
    pass

def _adapted_q(v37, v14, v15, v38, v39, v31: v9[v91, v138] | None):
    v40 = v120(v15.v194(v38[-40:]))
    v41 = v121(v14, v40, v31)
    if v41 is None:
        return None
    v42 = v195.v122(v15.v194(v39))
    v43 = v250.v196(v14.v242([v42[-1]])[0] + v41, dim=-1) if v42 else v41
    return v250.v196(v37.v251(v43.v273(0)), dim=-1)[0]

def _topk_indices(v44: v9[v10, v138], v45: v10, v17):
    v46 = v197(v44, key=lambda v112: -v44[v112])[:v45]
    v47 = v148.v123([v44[v112] for v112 in v46], dtype=v148.v198, device=v17)
    v47 = v47 / v47.v215().v199(1e-06)
    return (v47, v148.v123(v46, dtype=v148.v252, device=v17))

def retrieve(v48, v37, v14, v15, v33, v49, v38, v39, v45, v34):
    """Retrieval mode: cosine | idf_mean | votes | cascade | fusion."""
    v40 = v120(v15.v194(v38[-60:]))
    v17 = v33.v124.v17
    v50 = v125(v33.v126)
    if v48 == 'votes':
        return v49.v67(v40, v45, v33.v200)
    v43 = v127(v37, v14, v15, v38, v39, None)
    if v43 is None:
        return None
    if v48 == 'cosine':
        return v33.v67(v43, v45)
    if v48 == 'idf_mean':
        v128 = v127(v37, v14, v15, v38, v39, v49.v31)
        if v128 is None:
            return None
        v129 = (v34.v138() @ v128).v201(~v33.v200, -10000.0)
        v130 = v202(v45, v10(v33.v200.v254()))
        v47, v46 = v148.v67(v129, v130)
        return (v47, v46)
    if v48 == 'cascade':
        v131 = v203(v40, v49.v30, v49.v31)
        v132 = v204(v131, v43, v33.v124.v138(), v50, v205)
        v133 = v132[:v45]
        v129 = v148.v123([v138(v33.v124[v112] @ v43) for v112 in v133], dtype=v148.v198, device=v17)
        v129 = v129 / v129.v215().v199(1e-06)
        return (v129, v148.v123(v133, dtype=v148.v252, device=v17))
    if v48 == 'fusion':
        v131 = v203(v40, v49.v30, v49.v31)
        v134 = v206(v43, v33.v124.v138())
        v135 = v207(v131, v134, v208)
        return v209(v135, v45, v17)
    raise v136(v48)

def step_logp(v48, v37, v51, v52, v15, v14, v33, v49, v38, v39, v53, v54, v55, v17, v45, v34):
    v56 = v137(v48, v37, v14, v15, v33, v49, v38, v39, v45, v34)
    if v56 is None:
        return (v148.v69(v250.v287(v54, -1) + 1e-09), v148.v253((), device=v17))
    v129, v46 = v56
    v57 = v138(-(v250.v287(v54, -1) * v250.v293(v54, -1)).v254())
    v139, v140 = v141(v37, v33, v129, v46, v38, v55, v17)
    v58 = v37.v58(v53, v138(v129.v215()), v138(v129.v226()), v57, v140)
    return (v210(v54, v58, v139, v140), v58)

@v148.v64()
def decode(v48, v37, v51, v52, v15, v14, v33, v49, v59, v16, v55, v17, v45, v60, v34, v61=True):
    v62 = [v142 for v142 in v15.v274(v266.v241(S=v59['S'])).v145 if v142 != v16]
    v38, v143 = (v100(v62), [])
    for v63 in v144(v60):
        v145 = v148.v123([v38[-v277:]], dtype=v148.v252, device=v17)
        v211, v212 = v213(v51, v52, v145, v16)
        v54 = v212[0, -1]
        if v61:
            v214, v63 = v255(v48, v37, v51, v52, v15, v14, v33, v49, v38, v62, v211[0, -1], v54, v55, v17, v45, v34)
        else:
            v214 = v148.v69(v250.v287(v54, -1) + 1e-09)
        v146 = v10(v214.v256())
        v143.v190(v146)
        v38.v190(v146)
    return v15.v194(v143).v147()

def em(v48, v37, v51, v52, v15, v14, v33, v49, v26, v16, v55, v17, v45, v60, v34, v61=True):
    v65 = 0
    for v27 in v26:
        v56 = v194(v48, v37, v51, v52, v15, v14, v33, v49, v27, v16, v55, v17, v45, v60, v34, v61)
        v65 += v10(v97(v56) and v56.v147().v269(' ')[0].v147(' .,;:') == v27['value'])
    return v65 / v215(1, v125(v26))

def run_single_mode(v48: v91, v17, v18: v97, v66: v10, v67: v10, *, v68=v69) -> v9:
    """Train glue once for a retrieval mode; return EM metrics."""
    v148.v149(v7)
    v70 = v150.v150()
    v60 = 4 if v18 else 6
    v45 = v67
    v63, v63, v151, v152 = v153()
    v15 = v96.v154(v91(v257.v216))
    v55 = v15.v155()
    v16 = v15.v217(v218) or 0
    v52 = v275.v258(v15, v151, v16, v55).v119(v17)
    v71 = v5 if v5.v219() else v4
    v51 = v259(v152, v55).v119(v17)
    v51.v156(v148.v260(v71, map_location=v17, weights_only=False)['model'])
    v51.v157()
    for v72 in v51.v158():
        v72.v220(False)
    v73 = v259(v152, v55).v119(v17)
    v73.v156(v148.v260(v4, map_location=v17, weights_only=False)['model'])
    v73.v157()
    for v72 in v73.v158():
        v72.v220(False)
    v14 = v95(v73, v151, v17)
    v26, v33, v104, v35, v36, v34 = v159(v14, v15, v16, v17, v18)
    v49 = v160(v104, v17)
    v74 = [v94 for v94 in v26 if v94['glue_train']]
    v75 = [v94 for v94 in v26 if not v94['glue_train']]
    v19 = v185.v98(v7 + 11)
    v37 = v161(2 * (v51.v276.v261 // 2), v17)
    v76 = v148.v221.v162(v37.v222(), lr=0.003, weight_decay=0.01)
    for v77 in v144(1, v66 + 1):
        v27 = v74[v19.v262(v125(v74))]
        v62 = [v142 for v142 in v15.v274(v266.v241(S=v27['S'])).v145 if v142 != v16]
        v163 = [v142 for v142 in v15.v274(' ' + v27['value']).v145 if v142 != v16]
        if not v62 or not v163:
            continue
        v38 = (v62 + v163)[-v277:]
        v164 = v125(v38) - v125(v163)
        v145 = v148.v123([v38], dtype=v148.v252, device=v17)
        v211, v212 = v213(v51, v52, v145, v16)
        v165 = []
        for v223, v224 in v113(v163):
            v225 = v164 + v223 - 1
            if v225 < 0 or v225 >= v212.v278(1):
                break
            v214, v63 = v255(v48, v37, v51, v52, v15, v14, v33, v49, v38[:v225 + 1], v62, v211[0, v225], v212[0, v225], v55, v17, v45, v34)
            v165.v190(-v214[v224])
        if not v165:
            continue
        v166 = v148.v249(v165).v226()
        if v48 != 'votes' and v35 is not None:
            v227 = v148.v263(0, v35.v278(0), (v202(32, v35.v278(0)),), device=v17)
            v228 = v250.v196(v37.v251(v35[v227]), dim=-1)
            v166 = v166 + v250.v279(v228 @ v33.v124.v138().v225() / 0.05, v36[v227])
        v76.v229(set_to_none=True)
        v166.v230()
        v148.v280.v264.v231(v37.v222(), 1.0)
        v76.v77()
    v37.v157()
    v78 = v167(v48, v37, v51, v52, v15, v14, v33, v49, v75, v16, v55, v17, v45, v60, v34)
    v79 = []
    for v27 in v75:
        v168 = v33.v232()
        v168.v233(v27['value'])
        v169 = v49
        if v48 == 'votes':
            v169 = v160([[] if v33.v126[v112] == v27['value'] else v104[v112] for v112 in v144(v125(v104))], v17)
        v79.v190(v167(v48, v37, v51, v52, v15, v14, v168, v169, [v27], v16, v55, v17, v45, v60, v34))
    v80 = {'em': v78, 'em_after_delete': v138(v288.v226(v79)) if v79 else v138('nan'), 'wall_s': v150.v150() - v70}
    v68(f'[263/{v48}] ' + v268.v238(v80))
    return v80

def main() -> v10:
    v81 = v234.v170()
    v81.v171('--smoke', action='store_true')
    v81.v171('--steps', type=v10, default=0)
    v81.v171('--topk', type=v10, default=8)
    v82 = v81.v172()
    v3.v173('', encoding='utf-8')
    v17 = v148.v17('cuda' if v148.v281.v265() else 'cpu')
    v148.v149(v7)
    v70 = v150.v150()
    v66 = v82.v66 or (200 if v82.v18 else 800)
    v60 = 4 if v82.v18 else 6
    v45 = v82.v67
    v69(f'Stage263 votes vs mean start {v289.v285(v290.v286).v237()} device={v17}')
    v63, v63, v151, v152 = v153()
    v15 = v96.v154(v91(v257.v216))
    v55 = v15.v155()
    v16 = v15.v217(v218) or 0
    v52 = v275.v258(v15, v151, v16, v55).v119(v17)
    v71 = v5 if v5.v219() else v4
    v51 = v259(v152, v55).v119(v17)
    v51.v156(v148.v260(v71, map_location=v17, weights_only=False)['model'])
    v51.v157()
    for v72 in v51.v158():
        v72.v220(False)
    v73 = v259(v152, v55).v119(v17)
    v73.v156(v148.v260(v4, map_location=v17, weights_only=False)['model'])
    v73.v157()
    for v72 in v73.v158():
        v72.v220(False)
    v14 = v95(v73, v151, v17)
    v26, v33, v104, v35, v36, v34 = v159(v14, v15, v16, v17, v82.v18)
    v49 = v160(v104, v17)
    v74 = [v94 for v94 in v26 if v94['glue_train']]
    v75 = [v94 for v94 in v26 if not v94['glue_train']]
    v69(f'  trunk={v71.v178} slots={v125(v33.v126)} fit={v125(v74)} eval={v125(v75)} | vocab={v125(v49.v30)} postings={v254((v125(v47) for v47 in v49.v30.v126()))} | W_q pairs={(0 if v35 is None else v35.v278(0))}')

    def run(v48):
        v148.v149(v7)
        v19 = v185.v98(v7 + 11)
        v37 = v161(2 * (v51.v276.v261 // 2), v17)
        v76 = v148.v221.v162(v37.v222(), lr=0.003, weight_decay=0.01)
        for v77 in v144(1, v66 + 1):
            v27 = v74[v19.v262(v125(v74))]
            v62 = [v142 for v142 in v15.v274(v266.v241(S=v27['S'])).v145 if v142 != v16]
            v163 = [v142 for v142 in v15.v274(' ' + v27['value']).v145 if v142 != v16]
            if not v62 or not v163:
                continue
            v38 = (v62 + v163)[-v277:]
            v164 = v125(v38) - v125(v163)
            v145 = v148.v123([v38], dtype=v148.v252, device=v17)
            v211, v212 = v213(v51, v52, v145, v16)
            v165 = []
            for v223, v224 in v113(v163):
                v225 = v164 + v223 - 1
                if v225 < 0 or v225 >= v212.v278(1):
                    break
                v214, v63 = v255(v48, v37, v51, v52, v15, v14, v33, v49, v38[:v225 + 1], v62, v211[0, v225], v212[0, v225], v55, v17, v45, v34)
                v165.v190(-v214[v224])
            if not v165:
                continue
            v166 = v148.v249(v165).v226()
            if v48 != 'votes' and v35 is not None:
                v227 = v148.v263(0, v35.v278(0), (v202(32, v35.v278(0)),), device=v17)
                v228 = v250.v196(v37.v251(v35[v227]), dim=-1)
                v166 = v166 + v250.v279(v228 @ v33.v124.v138().v225() / 0.05, v36[v227])
            v76.v229(set_to_none=True)
            v166.v230()
            v148.v280.v264.v231(v37.v222(), 1.0)
            v76.v77()
            if v77 % v215(1, v66 // 4) == 0:
                v69(f'  [{v48}] step {v77}/{v66} loss={v138(v166):.3f} ({v150.v150() - v70:.0f}s)')
        v37.v157()
        v174 = v235(v48, v37, v14, v15, v33, v75, v16, cue_tmpl=v266)
        v78 = v167(v48, v37, v51, v52, v15, v14, v33, v49, v75, v16, v55, v17, v45, v60, v34)
        v175 = v167(v48, v37, v51, v52, v15, v14, v33.v282(v7 + 1), v49, v75, v16, v55, v17, v45, v60, v34) if v48 == 'cosine' else None
        v79 = []
        for v27 in v75:
            v168 = v33.v232()
            v168.v233(v27['value'])
            v169 = v49
            if v48 == 'votes':
                v169 = v160([[] if v33.v126[v112] == v27['value'] else v104[v112] for v112 in v144(v125(v104))], v17)
            v79.v190(v167(v48, v37, v51, v52, v15, v14, v168, v169, [v27], v16, v55, v17, v45, v60, v34))
        v80 = {'em': v78, 'em_after_delete': v138(v288.v226(v79)) if v79 else v138('nan'), **v174}
        if v175 is not None:
            v80['em_shuffled_keys'] = v175
        v69(f'[{v48}] ' + v268.v238(v80))
        return v80
    v83 = v176('cosine')
    v84 = v176('votes')
    v85 = v83['em'] >= 0.5
    v86 = v84['em'] - v83['em']
    v87 = v84['em_after_delete'] <= 0.1
    v88 = v83['em_after_delete'] <= 0.1
    if not v85:
        v177 = 'COSINE_BASELINE_INVALID'
    elif v86 >= 0.1 and v87:
        v177 = 'VOTES_BEAT_MEAN'
    elif v283(v86) < 0.1:
        v177 = 'VOTES_TIE_MEAN'
    else:
        v177 = 'MEAN_BEATS_VOTES'
    v89 = {'stage': 263, 'overall': v177, 'trunk': v71.v178, 'steps': v66, 'topk': v45, 'fp_version': v267.v236(), 'slots': v125(v33.v126), 'n_fit': v125(v74), 'n_eval': v125(v75), 'vocab': v125(v49.v30), 'gates': {'G_votes_causal': v87, 'G_cosine_causal': v88, 'G_cosine_reproduces_256': v85}, 'summary': {'cosine_mean': v83, 'word_votes': v84, 'delta_em': v86, 'delta_full_bank_top1': v84.v284('full_bank_top1', v138('nan')) - v83.v284('full_bank_top1', v138('nan')), 'reference_256_em_glue': 0.667, 'reference_261f': {'votes_20way': 0.601, 'mean_20way': 0.226}}, 'note': "256's exam with one line changed: retrieval by cos(q, key) over the context mean, or by word postings with an idf weight and nothing fitted. Cosine arm gets wiki InfoNCE on W_q like 256; votes arm does not. If G_cosine_reproduces_256 is false, delta_em is not readable.", 'timestamp': v289.v285(v290.v286).v237(), 'wall_s': v150.v150() - v70}
    v1.v173(v268.v238(v89, indent=2), encoding='utf-8')
    v2.v173(f"# Stage 263 word votes vs context mean (256's exam)\n\n**{v177}** slots={v125(v33.v126)} eval={v125(v75)}\n\n- EM: cosine/mean **{v83['em']:.3f}** -> votes **{v84['em']:.3f}** (delta {v86:+.3f}; 256 published 0.667; G_cosine_reproduces_256={v85})\n- slot deleted: cosine {v83['em_after_delete']:.3f} | votes {v84['em_after_delete']:.3f}\n- votes are zero-train; cosine arm also trains W_q via wiki InfoNCE\n", encoding='utf-8')
    v69(v268.v238({'overall': v177, 'delta_em': v86, 'G_cosine_reproduces_256': v85}, indent=2))
    return 0
if v90 == '__main__':
    raise v179(v239())