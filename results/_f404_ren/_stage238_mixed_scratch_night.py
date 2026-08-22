"""
Stage 238 — Mixed multi-domain L1 **from scratch** (hypothesis check).

Train two SelfModelXL from random init (matched steps):
  A) prose/wiki-only
  B) interleaved wiki+code

Then: write facts in each arm's own fp; apply code arc_enc shift; fit qmap W;
compare W-recall and next_tok.

Does **not** touch `stage191_p1_curve.pt`.
Writes: `checkpoints/stage238_prose_scratch.pt`, `checkpoints/stage238_mixed_scratch.pt`

  python _stage238_mixed_scratch_night.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
from _stage191_night import EVAL_EVERY, EXAM_V3, LR, MICRO, PAD, SelfModelXL, W_SELF, load_data, lr_at, sample_windows, score_items, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, mean_core_cos
v0 = v13('results')
v1 = v13('checkpoints')
v2 = v1 / 'stage238_prose_scratch.pt'
v3 = v1 / 'stage238_mixed_scratch.pt'
v4 = v0 / 'stage238_decision.json'
v5 = v0 / 'stage238_mini.md'
v6 = v0 / '_stage238_log.txt'
v7 = v13('data/_wikitext103_train.txt')
v8 = 238
v9 = 10000
v10 = 3.6 * 3600

def log(v14: v77) -> None:
    v15 = v14 if v14.v140('\n') else v14 + '\n'
    try:
        v141(v15, end='', flush=True)
    except v78:
        v141(v15.v222('ascii', 'replace').v198('ascii'), end='', flush=True)
    try:
        v6.v182.v142(parents=True, exist_ok=True)
        with v6.v143('a', encoding='utf-8') as v81:
            v81.v183(v15)
    except v79:
        pass

def build_mixed_flat(v16, v17: v12, v18: v12, v19: v80):
    with v7.v143('r', encoding='utf-8', errors='ignore') as v81:
        v82 = [v145 for v145 in v81.v176(8000000 if v19 else 40000000).v199() if v145.v200()][:v18]
    v20 = v144.v83(v153.v91(v8 + 1), v19)
    v21 = [v145 for v145 in v20.v199() if v145.v200()][:v18]
    v22: v84[v77] = []
    for v23 in v85(v146(v184(v82), v184(v21))):
        if v23 < v184(v82):
            v22.v185(v82[v23])
        if v23 < v184(v21):
            v22.v185(v21[v23])
    return v147.v86('\n'.v148(v22), v16, v17, max_lines=v184(v22), min_line_len=20)

def train_from_scratch(v24: v77, v25, v26, v27, v17: v12, v28: v12, v29: v12, v30: v84, v31: v149.v31, v32: v12, v33: v87, v34: v12) -> v42[v108, v11]:
    v149.v88(v34)
    v35 = v108(v28, v29).v89(v31)
    v36 = v149.v150.v90(v35.v151(), lr=v152, weight_decay=0.01)
    v37 = v153.v91(v34)
    v38 = v92.v92()
    v93, v94, v95 = (-1.0, 0, 0)
    v39 = None
    v35.v96()
    for v40 in v85(1, v32 + 1):
        for v97 in v36.v98:
            v97['lr'] = v186(v40, v32)
        v99 = v201(v25, v26, v202, v37, v17).v89(v31)
        v100 = v99 == v17
        v154, v155, v156 = v35.v157(v27[v99], v100, ids=v99)
        v101 = v99[:, 1:]
        v102 = ~v100[:, :-1] & ~v100[:, 1:]
        v103 = v187.v158(v154[:, :-1][v102], v101[v102])
        v104 = v103 + v188 * v156[~v100].v203()
        v36.v159(set_to_none=True)
        v104.v160()
        v204.v189.v161(v35.v151(), 1.0)
        v36.v40()
        v39 = v87(v103) if v39 is None else 0.95 * v39 + 0.05 * v87(v103)
        if v40 % (40 if v32 <= 200 else v215) == 0 or v40 == v32:
            v35.v107()
            v162 = v190(lambda v216, v217: v218(v35, v27, v17, v216, v217, v31), v30, 'next_tok')
            v163 = v162.v175('next_tok_acc', 0)
            v164 = v92.v92() - v38
            v122(f'  [{v24}] step {v40}/{v32}: ce~{v39:.3f} next_tok(mid)={v163:.3f} ({v164:.0f}s)')
            if v163 > v93 + 1e-06:
                v93, v94, v95 = (v163, v40, 0)
                v149.v174({'model': v35.v209(), 'step': v40, 'mid': v163, 'tag': v24}, v1 / f'_tmp_238_{v24}.pt')
            else:
                v95 += 1
            v35.v96()
            if v164 > v33:
                v122(f'  [{v24}] budget hit')
                break
            if v95 >= 2 and v40 >= v32 // 2:
                v122(f'  [{v24}] early stop (flat)')
                break
    v41 = v1 / f'_tmp_238_{v24}.pt'
    if v41.v105():
        v106 = v149.v165(v41, map_location=v31, weights_only=False)
        v35.v166(v106['model'])
    v35.v107()
    return (v35, {'best_mid': v93, 'best_step': v94, 'ce': v39, 'wall_s': v92.v92() - v38})

def arm_memory_exam(v35: v108, v43, v27, v17: v12, v31: v149.v31, v44: v84[v77], v45: v84[v77], v46: v84[v77], v47, v48, v49: v12, v50: v12, v37: v153.v91, v51: v12) -> v11:
    v52 = v109(v35, v43, v31)
    v53 = v167.v110(v52, v44)
    v111, v29 = v167.v112(v52, v45, v46, v37)
    v54 = v167.v113(v35, v47, v48, v27, v17, v31, v49, v51)
    v55 = v109(v54, v43, v31)
    v56 = v114(v52, v55, v44)
    v115, v116 = v167.v117(v205(256).v89(v31), v167.v110(v55, v44), v53, v37, v50, v31)
    v57 = v144.v118(v111, v29, v55, v45, v46, v37, query_x=v144.v191(v115))
    v58 = v144.v118(v111, v29, v55, v45, v46, v37)
    return {'mean_cos_after_code_shift': v56, 'W_align': v116, 'recall_W': v57, 'recall_no_W': v58}

def main() -> v12:
    v59 = v168.v119()
    v59.v120('--smoke', action='store_true')
    v60 = v59.v121()
    try:
        v6.v138('', encoding='utf-8')
    except v79:
        pass
    v31 = v149.v31('cuda' if v149.v206.v192() else 'cpu')
    v32 = 120 if v60.v19 else v9
    v61 = 600 if v60.v19 else v10
    v18 = 400 if v60.v19 else 20000
    v62 = 60 if v60.v19 else 400
    v63 = 10 if v60.v19 else 50
    v49 = 40 if v60.v19 else 400
    v50 = 60 if v60.v19 else 800
    v37 = v153.v91(v8)
    v122(f'Stage238 start {v220.v213(v221.v214).v179()} device={v31} steps/arm={v32}')
    v123, v124, v43, v28 = v125()
    v16 = v169.v126(v77(v193.v170))
    v29 = v16.v127()
    v17 = v16.v171(v172) or 0
    v27 = v207.v194(v16, v43, v17, v29).v89(v31)
    v128, v129 = v130(v16, v17, v18, v60.v19)
    v64 = []
    if v173.v105():
        with v173.v143(encoding='utf-8') as v81:
            for v15 in v81:
                v195 = v197.v208(v15)
                if v195.v175('type') == 'next_tok':
                    v64.v185(v195)
                if v184(v64) >= (40 if v60.v19 else 80):
                    break
    v122('train prose/wiki from scratch …')
    v131, v132 = v133('prose', v123, v124, v27, v17, v28, v29, v64, v31, v32, v61, v8 + 2)
    v122('train mixed wiki+code from scratch …')
    v134, v135 = v133('mixed', v128, v129, v27, v17, v28, v29, v64, v31, v32, v61, v8 + 3)
    if not v60.v19:
        v1.v142(parents=True, exist_ok=True)
        v149.v174({'model': v131.v209(), 'train': v132, 'stage': 238}, v2)
        v149.v174({'model': v134.v209(), 'train': v135, 'stage': 238}, v3)
    v65 = v87(v190(lambda v216, v217: v218(v131, v27, v17, v216, v217, v31), v64, 'next_tok').v175('next_tok_acc', 0))
    v66 = v87(v190(lambda v216, v217: v218(v134, v27, v17, v216, v217, v31), v64, 'next_tok').v175('next_tok_acc', 0))
    v122(f'next_tok prose={v65:.3f} mixed={v66:.3f}')
    with v7.v143('r', encoding='utf-8', errors='ignore') as v81:
        v136 = v81.v176(2000000)
    v44 = v84(v11.v177((v210 for v210 in v225.v223('[A-Za-z][a-z]{2,}', v136) if v184(v210) <= 14)))[:v62]
    v67 = v84(v11.v177((v212.v211(1) for v212 in v224.v219(v136) if v184(v212.v211(1)) >= 5)))
    v45 = v178(v196(v67), v37, v63 + 10)[:v63]
    v46 = v67[:v63]
    v47, v48 = v147.v86(v144.v83(v153.v91(v8 + 9), v60.v19), v16, v17, max_lines=v18, min_line_len=20)
    v122('memory exam prose arm …')
    v68 = v137(v131, v43, v27, v17, v31, v44, v45, v46, v47, v48, v49, v50, v37, v8 + 11)
    v122('memory exam mixed arm …')
    v69 = v137(v134, v43, v27, v17, v31, v44, v45, v46, v47, v48, v49, v50, v37, v8 + 12)
    v122(f'mem prose={v197.v180(v68)} mixed={v197.v180(v69)}')
    v70 = v66 >= v65 - 0.03
    v71 = v69['recall_W'] >= v68['recall_W'] + 0.12
    v72 = v69['recall_W'] >= 0.55
    v73 = v69['mean_cos_after_code_shift'] >= v68['mean_cos_after_code_shift'] + 0.05
    v74 = 'MIXED_SCRATCH_OK' if v70 and v71 and v72 else 'MIXED_SCRATCH_PARTIAL' if v70 and (v71 or v73 or v69['recall_W'] >= v68['recall_W']) else 'MIXED_SCRATCH_NO'
    v75 = {'stage': 238, 'overall': v74, 'gates': {'G_mixed_next_tok_not_worse': v70, 'G_mixed_W_beats_prose_by_0p12': v71, 'G_mixed_W_floor_0p55': v72, 'G_mixed_cos_more_stable': v73}, 'steps_per_arm': v32, 'train_prose': v132, 'train_mixed': v135, 'next_tok_prose': v65, 'next_tok_mixed': v66, 'memory_prose': v68, 'memory_mixed': v69, 'margin_W_mixed_minus_prose': v69['recall_W'] - v68['recall_W'], 'hypothesis': 'mixed-from-scratch improves post-code-shift W recall vs prose-from-scratch', 'note': 'Does not replace stage191_p1_curve.pt', 'timestamp': v220.v213(v221.v214).v179()}
    v4.v138(v197.v180(v75, indent=2), encoding='utf-8')
    v5.v138(f"# Stage 238 mixed scratch\n\n**{v74}** nt_m={v66:.3f} W_m={v69['recall_W']:.3f} W_p={v68['recall_W']:.3f} Δ={v69['recall_W'] - v68['recall_W']:+.3f}\n", encoding='utf-8')
    v122(v197.v180(v75, indent=2))
    return 0
if v76 == '__main__':
    raise v139(v181())