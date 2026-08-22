"""
Stage 178 — Objective flip: force prefix (not new units).

Reuse Stage177 ByteLevel BPE pieces as arcs.
Change the teacher so the goal *requires* the prefix:

  1) RETENTION — same last piece, different prefixes → push final states apart
  2) PAST-BAG — from state_t predict mean of earlier arcs (exclude last)
  3) PREDICT-FAR — from state_t predict arc_{t+k}
  4) INSTANCE — random cue only on prefix arcs; recover cue from final state

Next-local kept weak (so dynamics don't die). NO text CE.

Gate A: same last BPE piece / different prefix (must move if retention works).

  python _stage178_curve_retention.py
  python _stage178_curve_retention.py --steps 12000
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
v0 = v29('results')
v1 = v29('checkpoints')
v2 = v0 / '_stage178_log.txt'
v3 = v0 / 'stage178_decision.json'
v4 = v0 / 'stage178_mini.md'
v5 = v1 / 'stage178_curve_retention.pt'
v6 = v30.v6
v7 = v0 / 'plan_curve_dynamics.md'
v8 = 178
v9 = v30.v9
v10 = v30.v10
v11 = 16
v12 = 8
v13 = 4
v14 = 1
v15 = 0.0003
v16 = 1500
v17 = 12000
v18 = 0.15
v19 = 0.5
v20 = 1.0
v21 = 1.5
v22 = 1.0
v23 = 0.985
v24 = 0.9

def log(v31: v102) -> None:
    v32 = v31 if v31.v177('\n') else v31 + '\n'
    try:
        v178(v32, end='', flush=True)
    except v103:
        v178(v32.v268('ascii', 'replace').v251('ascii'), end='', flush=True)
    v2.v179.v104(parents=True, exist_ok=True)
    with v2.v180('a', encoding='utf-8') as v105:
        v105.v181(v32)

def write_json(v33: v29, v34: v40) -> None:
    v33.v179.v104(parents=True, exist_ok=True)
    v33.v106(v231.v182(v34, indent=2, ensure_ascii=False), encoding='utf-8')

class RetentionModel(v35.v25):

    def __init__(v107, v108: v28):
        v252().v183()
        v107.v109 = v30.v184(v108)
        v107.v110 = v30.v185()
        v107.v111 = v35.v186(v35.v232(v9, v9), v35.v233(), v35.v232(v9, v9))
        v107.v112 = v35.v186(v35.v232(v9, v9), v35.v233(), v35.v232(v9, v9))
        v107.v113 = v35.v186(v35.v232(v9, v9), v35.v233(), v35.v232(v9, v9))
        v107.v114 = v35.v186(v35.v232(v9, v9), v35.v233(), v35.v232(v9, v9))

    def encode_arcs(v107, v57: v51.v26) -> v51.v26:
        return v107.v109(v57)

    def forward_states(v107, v115: v51.v26, v116: v51.v26 | None=None) -> v51.v26:
        return v107.v110(v115, pad_mask=v116)

def build_same_last_index(v36: v118[v118[v102]], v37: v28=40) -> v40[v102, v118[v118[v102]]]:
    v38: v40[v102, v118[v118[v102]]] = v117(v118)
    for v39 in v36:
        if v208(v39) < 12:
            continue
        for v119 in v127(10, v234(v208(v39), 80)):
            v128 = v39[v119]
            v187 = v39[v204(0, v119 - (v10 - 1)):v119 + 1]
            if v208(v38[v128]) < v37:
                v235 = v253(v187[:-1])
                if v254((v253(v269[:-1]) != v235 for v269 in v38[v128])):
                    v38[v128].v152(v187)
    return {v120: v121 for v120, v121 in v38.v236() if v208(v121) >= 2}

def sample_retention_pair_batch(v41: v40[v102, v118[v118[v102]]], v42: v40, v43: v28, v44: v188.v122, v45):
    v46 = [v120 for v120, v121 in v41.v236() if v208(v121) >= 2]
    if not v46:
        return None
    v123, v124, v125, v126 = ([], [], [], [])
    for v47 in v127(v43):
        v128 = v46[v44.v237(0, v208(v46) - 1)]
        v129 = v41[v128]
        v189, v190 = v44.v191(v129, 2)

        def pack(v187):
            v187 = v187[-v10:]
            v192 = v10 - v208(v187)
            v193 = v187 + [''] * v192
            return (v30.v255(v193, v42), v51.v256([v216 == '' for v216 in v193], dtype=v51.v209))
        v194, v195 = v196(v189)
        v197, v198 = v196(v190)
        v123.v152(v194)
        v124.v152(v197)
        v125.v152(v195)
        v126.v152(v198)
    return (v51.v206(v123).v157(v45), v51.v206(v124).v157(v45), v51.v206(v125).v157(v45), v51.v206(v126).v157(v45))

def last_valid_index(v48: v51.v26) -> v51.v26:
    v130, v131 = v48.v49
    v50 = (~v48).v153(dim=1).v132(min=1)
    return (v50 - 1).v133()

def gather_last(v52: v51.v26, v48: v51.v26) -> v51.v26:
    v53 = v134(v48)
    return v52[v51.v238(v52.v257(0), device=v52.v45), v53]

def cos_loss_match(v54: v51.v26, v55: v51.v26) -> v51.v26:
    return (1.0 - v199.v156(v54, v55.v207(), dim=-1)).v135()

def dynamics_bundle(v56: v136, v57: v51.v26, v48: v51.v26, v58=None):
    """next (weak) + far + past-bag + instance-on-prefix."""
    v130, v131, v47 = v57.v49
    v45 = v57.v45
    v59 = v56.v137(v57)
    v60 = v199.v138(v51.v200(v130, v9, device=v45), dim=-1)
    v50 = (~v48).v153(dim=1).v132(min=2)
    v61 = (v50.v27() * 0.5).v133().v132(min=1)
    v62 = v59.v139()
    for v63 in v127(v130):
        v140 = v28(v61[v63].v239())
        v62[v63, :v140] = v62[v63, :v140] + 0.35 * v60[v63]
    v52 = v56.v92(v62, pad_mask=v48)
    v64 = {}
    v65 = []
    v66 = ~v48[:, :-1] & ~v48[:, 1:]
    if v66.v153() > 0:
        v141 = v56.v111(v52[:, :-1])
        v142 = v151(v141[v66], v59[:, 1:][v66])
        v65.v152(v18 * v142)
        v64['cos_next'] = v27(v199.v156(v141[v66], v59[:, 1:][v66].v207(), dim=-1).v135())
    else:
        v64['cos_next'] = 0.0
    if v131 > v13 + 1:
        v143 = ~v48[:, :-v13] & ~v48[:, v13:]
        if v143.v153() > 0:
            v201 = v56.v112(v52[:, :-v13])
            v202 = v59[:, v13:]
            v203 = v151(v201[v143], v202[v143])
            v65.v152(v19 * v203)
            v64['cos_far'] = v27(v199.v156(v201[v143], v202[v143].v207(), dim=-1).v135())
        else:
            v64['cos_far'] = 0.0
    else:
        v64['cos_far'] = 0.0
    v67 = []
    v68 = []
    for v69 in v127(3, v131):
        v144 = ~v48[:, v69]
        if v144.v153() < 1:
            continue
        v145 = v204(1, v69 - v14)
        v146 = []
        for v63 in v127(v130):
            if v48[v63, v69]:
                v146.v152(v51.v158(v9, device=v45))
                continue
            v97 = v59[v63, :v145]
            v205 = ~v48[v63, :v145]
            if v205.v153() < 1:
                v146.v152(v51.v158(v9, device=v45))
            else:
                v146.v152(v97[v205].v135(dim=0))
        v147 = v51.v206(v146, 0)
        v148 = v56.v113(v52[:, v69])
        v79 = (1.0 - v199.v156(v148[v144], v147[v144].v207(), dim=-1)).v135()
        v68.v152(v79)
        v67.v152(v27(v199.v156(v148[v144], v147[v144].v207(), dim=-1).v135()))
    if v68:
        v149 = v51.v206(v68).v135()
        v65.v152(v20 * v149)
        v64['cos_past'] = v27(v258.v135(v67))
    else:
        v64['cos_past'] = 0.0
    v70 = v150(v52, v48)
    v71 = v56.v114(v70)
    v72 = v151(v71, v60)
    v65.v152(v22 * v72)
    v64['cos_inst'] = v27(v199.v156(v71, v60, dim=-1).v135())
    if not v65:
        return (v52.v153() * 0.0, v64)
    v73 = v153(v65)
    v64['loss_dyn'] = v27(v73.v207())
    return (v73, v64)

def retention_loss(v56: v136, v74, v75=None):
    v123, v124, v154, v155 = v74
    v76 = v150(v56.v92(v56.v137(v123), v154), v154)
    v77 = v150(v56.v92(v56.v137(v124), v155), v155)
    v78 = v199.v156(v76, v77, dim=-1)
    v79 = v199.v259(v78 - 0.5).v135() + 0.25 * v78.v135()
    return (v21 * v79, {'ret_cos': v27(v78.v135().v207()), 'ret_hinge': v27(v79.v207())})

@v51.v81()
def encode_seq(v56, v80, v42, v45):
    v80 = v80[-v10:] or ['.']
    v57 = v30.v255(v80, v42).v240(0).v157(v45)
    v48 = v51.v158(1, v208(v80), dtype=v51.v209, device=v45)
    return v56.v92(v56.v137(v57), v48)[0]

def cos(v82, v63) -> v27:
    return v27(v199.v156(v199.v138(v82, dim=0), v199.v138(v63, dim=0), dim=0))

def gate_A(v56, v36, v42, v45, v44, v43=80):
    return v30.v159(v56, v36, v42, v45, v44, n_pairs=v43)

def main() -> v28:
    v83 = v210.v160()
    v83.v161('--steps', type=v28, default=v17)
    v83.v161('--device', default='cuda' if v51.v264.v260() else 'cpu')
    v84 = v83.v162()
    v0.v104(parents=True, exist_ok=True)
    v1.v104(parents=True, exist_ok=True)
    v2.v106('', encoding='utf-8')
    v163(f'Stage178 start {v266.v262(v267.v263).v228()}')
    v163('Objective flip: retention + past-bag + far + prefix-instance (BPE units from 177)')
    v163(f'weights next={v18} far={v19} past={v20} ret={v21} inst={v22} k_far={v13}')
    if not v6.v211():
        raise v212(f'need {v6} from Stage177')
    v85 = v213.v164(v102(v6))
    v86 = v214.v165(max_chars=20000000)
    v87 = v166(v241(v86) | {' '})
    v88 = ['<pad>'] + v87
    v42 = {v167: v119 + 1 for v119, v167 in v242(v87)}
    v36 = v30.v168(v85, v86)
    v89 = v36[v28(0.8 * v208(v36)):] or v36[-100:]
    v90 = v36[:v28(0.8 * v208(v36))] or v36
    v41 = v169(v90)
    v163(f'docs={v208(v36)} same-last keys={v208(v41)} V_bpe={v85.v261()}')
    v45 = v51.v45(v84.v45)
    v51.v170(v8)
    v188.v171(v8)
    v56 = v136(v208(v88)).v157(v45)
    v56.v91 = v56.v92

    class GateWrap(v35.v25):

        def __init__(v107, v215):
            v252().v183()
            v107.v215 = v215

        def encode_arcs(v107, v216):
            return v107.v215.v137(v216)

        def forward_states(v107, v57, v116=None):
            return v107.v215.v92(v107.v215.v137(v57), v116)
    v93 = v243(v56).v157(v45)
    v94 = v51.v217.v172(v56.v218(), lr=v15, weight_decay=0.0001)
    v44 = v188.v122(v8)
    v95 = v159(v93, v89, v42, v45, v188.v122(v8))
    v163(f"  init A: same={v95['mean_cos_same_last_piece']:.3f} diff={v95['mean_cos_diff_last_piece']:.3f} → {v95['verdict']}")
    v56.v90()
    v96 = v95
    v97 = None
    v98 = None
    for v99 in v127(1, v84.v219 + 1):
        v216, v48 = v30.v220(v90, v42, v11, v44, v45)
        v221, v222 = v223(v56, v216, v48)
        v173 = v224(v41, v42, v12, v44, v45)
        if v173 is not None:
            v244, v245 = v246(v56, v173)
            v73 = v221 + v244
            v222.v247(v245)
        else:
            v73 = v221
            v222['ret_cos'] = 1.0
        v94.v225(set_to_none=True)
        v73.v226()
        v35.v248.v227(v56.v218(), 1.0)
        v94.v99()
        v98 = v27(v73.v207()) if v98 is None else 0.95 * v98 + 0.05 * v27(v73.v207())
        if v99 % v16 == 0 or v99 == v84.v219:
            v56.v249()
            v96 = v159(v93, v89, v42, v45, v188.v122(v8 + v99))
            v163(f"  step {v99}: loss~{v98:.3f} next={v222.v270('cos_next', 0):.3f} far={v222.v270('cos_far', 0):.3f} past={v222.v270('cos_past', 0):.3f} inst={v222.v270('cos_inst', 0):.3f} ret_cos={v222.v270('ret_cos', 0):.3f} A_same={v96['mean_cos_same_last_piece']:.3f} A_diff={v96['mean_cos_diff_last_piece']:.3f} → {v96['verdict']}")
            v56.v90()
            v51.v250({'model': v56.v265(), 'stoi': v42, 'step': v99, 'A': v96}, v5)
            if v99 >= 4500 and v96['mean_cos_same_last_piece'] >= v23:
                v97 = 'EARLY_FAIL_STILL_WIPE'
                v163(f'  [{v97}] stop @ {v99}')
                break
            if v96['mean_cos_same_last_piece'] < v24 and 'PASS' in v96['verdict']:
                v97 = 'EARLY_PASS_PREFIX'
                v163(f'  [{v97}] stop @ {v99}')
                break
    if 'PASS' in v96['verdict']:
        v174 = 'CURVE_RETENTION_CONTEXT_YES'
    elif v96['mean_cos_same_last_piece'] >= 0.95:
        v174 = 'CURVE_RETENTION_CONTEXT_NULL'
    else:
        v174 = 'CURVE_RETENTION_CONTEXT_WEAK'
    v100 = {'timestamp': v266.v262(v267.v263).v228(), 'protocol': 'curve_retention_178', 'overall': v174, 'early': v97, 'steps_ran': v99, 'objective': {'retention': 'same-last-piece pairs → push endpoints apart', 'past_bag': 'state_t → mean of earlier arcs (exclude last)', 'predict_far': f'state_t → arc_t+{v13}', 'instance': 'random cue on prefix half only → recover from final', 'next_local_weight': v18}, 'units': 'Stage177 ByteLevel BPE pieces (unchanged)', 'A': v96, 'init_A': v95, 'note': 'Falsify: can objective-flip beat last-unit wipe without new tokenizer?', 'next': 'If YES: harden + gate B. If NULL: this objective family insufficient at this scale — need stronger instance/handwriting or abandon context-from-curve under local ink.'}
    v175(v3, v100)
    v4.v106('\n'.v229(['# Stage178 — retention objective', '', f'**Overall:** `{v174}`' + (f' ({v97})' if v97 else ''), '', f"- A: {v96['verdict']} same={v96['mean_cos_same_last_piece']:.3f} diff={v96['mean_cos_diff_last_piece']:.3f}", '- losses: retention + past-bag + far + prefix-instance (next weak)', f"- {v100['next']}", '']), encoding='utf-8')
    v163(f'[178] {v174}')
    return 0
if v101 == '__main__':
    raise v176(v230())