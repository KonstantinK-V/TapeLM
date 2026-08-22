"""
Stage 187 — Self-model: predict-own-next-state + surprise-gated writes.

Same clean-CE recipe as 185-endpoint (the healthy baseline), but the slow channel
now has a self-model:
  - pred(slow_t) = expectation of its own next write
  - surprise_t   = 1 - cos(expectation, actual write)
  - write gate  *= sigmoid(k * (surprise - 0.5))  → routine ink barely writes, novelty writes hard

Aux loss = self-prediction only (predictor head, detached target) — does NOT push
representations directly (lesson of 185: hand losses on representations are poison).

Gates (judge = calibrated Exam v2 from 186):
  G1 CE preserved : next_tok >= endpoint_185(v2) - 0.03
  G2 novelty      : mean surprise on UNSEEN hold docs > on seen train docs
  G3 calibration  : predictive entropy after FAKE entity > after real entity (knows-it-doesn't-know)

SELF_MODEL_YES = G1 & G2 & G3.

  python _stage187_self_model.py
  python _stage187_self_model.py --steps 3000
"""
from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage181_ce_control as s181
import _stage185_tape_read as s185
v0 = v24('results')
v1 = v24('data')
v2 = v24('checkpoints')
v3 = v0 / '_stage187_log.txt'
v4 = v0 / 'stage187_decision.json'
v5 = v0 / 'stage187_mini.md'
v6 = v1 / 'stage186_exam_v2.jsonl'
v7 = v0 / 'stage186_decision.json'
v8 = v2 / 'stage187_self_model.pt'
v9 = v25.v9
v10 = 185
v11 = 128
v12 = v25.v12
v13 = 16
v14 = 0.0003
v15 = 1000
v16 = 3000
v17 = 0.1
v18 = 60
v19 = '[PAD]'
v20 = ['Zorblax', 'Quenith', 'Marbune', 'Xaldera', 'Kessari', 'Vornak', 'Talmidex', 'Orsiphon', 'Pholmar', 'Girenth']

def log(v26: v77) -> None:
    v27 = v26 if v26.v146('\n') else v26 + '\n'
    try:
        v147(v27, end='', flush=True)
    except v78:
        v147(v27.v232('ascii', 'replace').v222('ascii'), end='', flush=True)
    v3.v148.v79(parents=True, exist_ok=True)
    with v3.v149('a', encoding='utf-8') as v80:
        v80.v84(v27)

class SurpriseWriter(v28.v21):
    """Write-budget memory where write strength is gated by self-prediction error."""

    def __init__(v81, v82: v23, v83: v23=v11):
        v223().v150()
        v81.v84 = v28.v151(v28.v152(v82 + v83, v83), v28.v202(), v28.v152(v83, v83))
        v81.v85 = v28.v152(v82 + v83, 1)
        v81.v86 = v28.v151(v28.v152(v83, v83), v28.v202(), v28.v152(v83, v83))
        v81.v87 = v28.v153(v101.v107(4.0))
        v81.v88 = v28.v154(v83)
        v81.v89 = v28.v153(v101.v203(v83))

    def forward(v81, v90: v101.v29, v44: v101.v29):
        v155, v156, v38 = v90.v91
        v92 = v81.v89.v206(0).v224(v155, -1).v157()
        v93 = v101.v158(v155, 1, device=v90.v34)
        v159, v160, v161 = ([], [], [])
        for v94 in v102(v156):
            v43 = v90[:, v94]
            v162 = v101.v204([v43, v92], dim=-1)
            v163 = v81.v86(v92)
            v164 = v81.v84(v162)
            v165 = 1.0 - v178.v225(v163, v164, dim=-1)
            v166 = v101.v205(v81.v87 * (v165.v206(-1) - 0.5))
            v167 = (~v44[:, v94]).v22().v206(-1)
            v168 = v101.v205(v81.v85(v162)) * v93 * v166 * v167
            v92 = v81.v88(v92 + v168 * v164)
            v93 = (v93 - v168).v207(min=0.0)
            v159.v175(v92)
            v160.v175(v165 * v167.v229(-1))
            v161.v175((1.0 - v178.v225(v163, v164.v236(), dim=-1)) * v167.v229(-1))
        return (v101.v208(v159, 1), v101.v208(v160, 1), v101.v208(v161, 1))

class SelfModel(v28.v21):

    def __init__(v81, v95: v23, v54: v23):
        v223().v150()
        v81.v96 = v25.v169(v95, d=v11)
        v81.v97 = v25.v170(d=v11)
        v81.v92 = v171(v11, v11)
        v81.v98 = v28.v152(2 * v11, v54, bias=False)

    def forward_all(v81, v99: v101.v29, v44: v101.v29):
        v90 = v81.v96(v99)
        v97 = v81.v97(v90, pad_mask=v44)
        v92, v165, v172 = v81.v92(v90, v44)
        v45 = v81.v98(v101.v204([v97, v92], dim=-1))
        return (v45, v165, v172)

    def logits(v81, v99: v101.v29, v44: v101.v29, v100: v173=False) -> v101.v29:
        return v81.v174(v99, v44)[0]

@v101.v39()
def mean_surprise(v30, v31, v32, v33, v34, v35, v36=100) -> v22:
    v37 = []
    for v38 in v102(v36):
        v103 = v31[v35.v209(0, v210(v31) - 1)]
        if v210(v103) < 8:
            continue
        v104 = v103[:v12]
        if v210(v104) < v12:
            v104 = v104 + [v33] * (v12 - v210(v104))
        v105 = v101.v107([v104], dtype=v101.v177, device=v34)
        v44 = v105 == v33
        v38, v165, v38 = v30.v174(v32[v105], v44)
        v106 = ~v44
        v37.v175(v22(v165[v106].v176()))
    return v22(v211.v176(v37))

@v101.v39()
def entropy_after(v30, v32, v33, v40, v41, v34) -> v22:
    v42 = (v40 + v41)[-v12:]
    v43 = v101.v107([v42], dtype=v101.v177, device=v34)
    v44 = v43 == v33
    v45 = v30.v45(v32[v43], v44)[0, v210(v42) - 1]
    v46 = v178.v108(v45, dim=-1)
    return v22(-(v46 * v101.v113(v46 + 1e-09)).v212())

def main() -> v23:
    v47 = v179.v109()
    v47.v110('--steps', type=v23, default=v16)
    v47.v110('--device', default='cuda' if v101.v230.v226() else 'cpu')
    v48 = v47.v111()
    v0.v79(parents=True, exist_ok=True)
    v3.v112('', encoding='utf-8')
    v113(f'Stage187 start {v233.v227(v234.v228).v198()}')
    v113('Self-model: predict-own-next-state + surprise-gated slow writes')
    v49 = [v181.v114(v180) for v180 in v6.v182(encoding='utf-8').v213() if v180.v214()]
    v50 = [v70 for v70 in v49 if v70['type'] == 'next_tok'][:v18]
    v51 = v181.v114(v7.v182(encoding='utf-8'))
    v52 = v51['results']['endpoint_185']['next_tok_acc']
    v113(f'exam v2 items={v210(v49)}; baseline endpoint_185 next_tok={v52:.3f}')
    v34 = v101.v34(v48.v34)
    v53 = v183.v115(v77(v9))
    v54 = v53.v116()
    v33 = v53.v184(v19) or 0
    v55 = v185.v117(max_chars=20000000)
    v56 = v118(v215(v55) | {' '})
    v57 = ['<pad>'] + v56
    v58 = {v119: v186 + 1 for v186, v119 in v216(v56)}
    v31 = v187.v120(v53, v55)
    v59 = v31[:v23(0.8 * v210(v31))] or v31
    v60 = v31[v23(0.8 * v210(v31)):] or v31[-100:]
    v32 = v197.v217(v53, v58, v33, v54).v121(v34)
    v113(f'docs={v210(v31)} V={v54} n_char={v210(v57)}')
    v101.v122(v10)
    v30 = v218(v210(v57), v54).v121(v34)
    v61 = v101.v188.v123(v30.v189(), lr=v14, weight_decay=0.01)
    v35 = v190.v124(v10)
    v125, v126 = (None, None)
    v62 = v127.v127()
    v30.v128()
    for v63 in v102(1, v48.v144 + 1):
        v105 = v197.v191(v59, v13, v35, v34, v33)
        v44 = v105 == v33
        v45, v165, v172 = v30.v174(v32[v105], v44)
        v129 = v105[:, 1:]
        v106 = ~v44[:, :-1] & ~v44[:, 1:]
        v130 = v178.v192(v45[:, :-1][v106], v129[v106])
        v131 = v172[~v44].v176()
        v132 = v130 + v17 * v131
        v61.v193(set_to_none=True)
        v132.v194()
        v28.v219.v195(v30.v189(), 1.0)
        v61.v63()
        v125 = v22(v130) if v125 is None else 0.95 * v125 + 0.05 * v22(v130)
        v126 = v22(v131) if v126 is None else 0.95 * v126 + 0.05 * v22(v131)
        if v63 % v15 == 0 or v63 == v48.v144:
            v30.v133()
            v196 = v197.v134(v30, v32, v33, v50, v34, only_type='next_tok')
            v113(f"  step {v63}: ce~{v125:.3f} self~{v126:.3f} k={v22(v30.v92.v87):.2f} next_tok(mid)={v196.v135('next_tok_acc', 0):.3f} ({v127.v127() - v62:.0f}s)")
            v30.v128()
            v101.v220({'model': v30.v231(), 'step': v63}, v8)
    v30.v133()
    v64 = v197.v134(v30, v32, v33, v49, v34)
    v65 = v64.v135('next_tok_acc', 0.0)
    v66 = v136(v30, v59, v32, v33, v34, v190.v124(1))
    v67 = v136(v30, v60, v32, v33, v34, v190.v124(2))
    v68 = [v70 for v70 in v49 if v70['type'] == 'entity'][:80]
    v69 = v190.v124(3)
    v137, v138 = ([], [])
    for v70 in v68:
        v139 = v70['cand_ids'][v70['gold_idx']]
        v140 = v20[v69.v209(0, v210(v20) - 1)]
        v141 = [v186 for v186 in v53.v232(' ' + v140).v105 if v186 != v33]
        v137.v175(v221(v30, v32, v33, v70['ctx_ids'], v139, v34))
        v138.v175(v221(v30, v32, v33, v70['ctx_ids'], v141, v34))
    v142, v143 = (v22(v211.v176(v137)), v22(v211.v176(v138)))
    v71 = v65 >= v52 - 0.03
    v72 = v67 > v66
    v73 = v143 > v142
    v74 = 'SELF_MODEL_YES' if v71 and v72 and v73 else 'SELF_MODEL_PARTIAL_' + ''.v200((v36 for v36, v235 in (('1', v71), ('2', v72), ('3', v73)) if not v235))
    v75 = {'timestamp': v233.v227(v234.v228).v198(), 'protocol': 'self_model_surprise_187', 'overall': v74, 'gates': {'G1_ce_preserved': {'next_tok': v65, 'baseline': v52, 'ok': v71}, 'G2_novelty': {'surprise_seen_train': v66, 'surprise_unseen_hold': v67, 'ok': v72}, 'G3_calibration': {'entropy_after_real': v142, 'entropy_after_fake': v143, 'ok': v73}}, 'exam_full': v64, 'k_final': v22(v30.v92.v87), 'steps': v48.v144, 'note': 'aux loss = predictor only (detached target); no representation pushing'}
    v4.v112(v181.v199(v75, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v112('\n'.v200(['# Stage187 — self-model (surprise-gated writes)', '', f'**Overall:** `{v74}`', '', f'- G1 CE preserved: next_tok={v65:.3f} vs baseline {v52:.3f} → {v71}', f'- G2 novelty: surprise seen={v66:.4f} vs unseen={v67:.4f} → {v72}', f'- G3 calibration: entropy real={v142:.3f} vs fake={v143:.3f} → {v73}', f"- entity={v64.v135('entity_acc', 0):.3f} ood={v64.v135('ood_acc', 0):.3f} k={v22(v30.v92.v87):.2f}", '']), encoding='utf-8')
    v113(f'[187] {v74} | G1 {v65:.3f}/{v52:.3f} | G2 {v66:.4f}<{v67:.4f}? | G3 {v142:.3f}<{v143:.3f}?')
    return 0
if v76 == '__main__':
    raise v145(v201())