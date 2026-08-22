"""
Stage 175 — Causal Transformer pen for context (gate A/B).

Replace GRU compression with causal self-attn pen so past arc can remain
visible at the endpoint. Short fit → freeze pen → rerun 174-style A (and B).

Contract: still NO char/word CE as teacher. Loss = latent Δ / next-z / contrastive.

  python _stage175_attn_pen_context_gate.py
  python _stage175_attn_pen_context_gate.py --steps 20000
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import _stage170_curve_dynamics as s170
v0 = v27('results')
v1 = v27('checkpoints')
v2 = v0 / '_stage175_log.txt'
v3 = v0 / 'stage175_decision.json'
v4 = v0 / 'stage175_mini.md'
v5 = v1 / 'stage175_attn_pen.pt'
v6 = v0 / 'plan_curve_dynamics.md'
v7 = 175
v8 = 96
v9 = 2
v10 = 4
v11 = 128
v12 = 160
v13 = 16
v14 = 0.0003
v15 = 2000
v16 = 20000
v17 = (1, 2, 4, 8)
v18 = 24
v19 = 96

def log(v28: v90) -> None:
    v29 = v28 if v28.v179('\n') else v28 + '\n'
    try:
        v180(v29, end='', flush=True)
    except v91:
        v180(v29.v120('ascii', 'replace').v246('ascii'), end='', flush=True)
    v2.v181.v92(parents=True, exist_ok=True)
    with v2.v182('a', encoding='utf-8') as v93:
        v93.v183(v29)

def write_json(v30: v27, v31: v23) -> None:
    v30.v181.v92(parents=True, exist_ok=True)
    v30.v94(v231.v184(v31, indent=2), encoding='utf-8')

class CausalAttnBlock(v32.v20):

    def __init__(v95, v96: v26, v97: v26):
        v247().v185()
        v95.v98 = v32.v186(v96, v97, batch_first=True, dropout=0.1)
        v95.v99 = v32.v187(v96)
        v95.v100 = v32.v188(v32.v232(v96, 4 * v96), v32.v233(), v32.v232(4 * v96, v96), v32.v234(0.1))
        v95.v101 = v32.v187(v96)

    def forward(v95, v41: v39.v21) -> v39.v21:
        v102 = v41.v189(1)
        v103 = v39.v190(v39.v235(v102, v102, device=v41.v54, dtype=v39.v248), diagonal=1)
        v111, v67 = v95.v98(v41, v41, v41, attn_mask=v103)
        v41 = v95.v99(v41 + v111)
        return v95.v101(v41 + v95.v100(v41))

class AttnPen(v32.v20):
    """Char stream → curve z_t with causal attention (context-capable ink)."""

    def __init__(v95, v104: v26, v96: v26=v8, v105: v26=v9, v106: v26=512):
        v247().v185()
        v95.v107 = v32.v191(v104, v96)
        v95.v108 = v32.v191(v106, v96)
        v95.v109 = v32.v192([v249(v96, v10) for v67 in v136(v105)])
        v95.v110 = v32.v187(v96)
        v95.v106 = v106

    def forward(v95, v41: v39.v21) -> v39.v21:
        v121, v102 = v41.v42
        if v102 > v95.v106:
            v41 = v41[:, -v95.v106:]
            v102 = v41.v189(1)
        v108 = v39.v118(v102, device=v41.v54).v198(0).v193(v121, v102)
        v111 = v95.v107(v41) + v95.v108(v108)
        for v112 in v95.v109:
            v111 = v112(v111)
        return v95.v110(v111)

class DynHead(v32.v20):

    def __init__(v95, v96: v26=v8, v113: v51[v26, ...]=v17):
        v247().v185()
        v95.v113 = v113
        v95.v93 = v32.v188(v32.v232(v96 * 2, v96 * 2), v32.v233(), v32.v232(v96 * 2, v96 * 2))
        v95.v114 = v32.v194({v90(v49): v32.v232(v96 * 2, v96) for v49 in v113})
        v95.v115 = v32.v194({v90(v49): v32.v232(v96 * 2, v96) for v49 in v113})

    def forward(v95, v45: v39.v21) -> v23[v90, v39.v21]:
        v195, v131 = (v45[:, -1], v45.v131(1))
        v111 = v95.v93(v39.v236([v195, v131], -1))
        v66 = {}
        for v49 in v95.v113:
            v66[f'delta_{v49}'] = v95.v114[v90(v49)](v111)
            v66[f'z_{v49}'] = v95.v115[v90(v49)](v111)
        return v66

class Curve175(v32.v20):

    def __init__(v95, v104: v26):
        v247().v185()
        v95.v116 = v196(v104)
        v95.v117 = v197()

    def encode(v95, v41: v39.v21) -> v39.v21:
        return v95.v116(v41)

def ctx_windows(v33: v39.v21, v34: v26, v35: v26, v36: v26=v11) -> v39.v21:
    v37 = v39.v118(v34, v34 + v35, device=v33.v54)
    v38 = v37.v198(1) - v39.v118(v36 - 1, -1, -1, device=v33.v54).v198(0)
    return v33[:, v38.v237(min=0)]

def train_loss(v40: v119, v41: v39.v21) -> v51[v39.v21, v23]:
    v33 = v40.v120(v41)
    v121, v102, v96 = v33.v42
    v43 = v122(v17)
    v34 = v11
    v44 = v102 - 1 - v43
    v35 = v44 - v34 + 1
    if v35 < 1:
        raise v199('seq too short')
    v45 = v123(v33, v34, v35)
    v46 = v40.v117(v45.v200(v121 * v35, v11, v96))
    v47 = 0.0
    v48 = {}
    for v49 in v17:
        v124 = v33[:, v34:v34 + v35]
        v125 = v33[:, v34 + v49:v34 + v35 + v49]
        v126 = v125 - v124
        v127 = v46[f'delta_{v49}'].v201(v121, v35, v96)
        v128 = v46[f'z_{v49}'].v201(v121, v35, v96)
        v129 = 1.0 - v240.v203(v127, v126.v202(), dim=-1).v131()
        v130 = 1.0 - v240.v203(v128, v125.v202(), dim=-1).v131()
        v47 = v47 + 1.0 / v258.v254(v49) * (v129 + v130)
        v48[f'cos_d_k{v49}'] = v22(v240.v203(v127, v126.v202(), dim=-1).v131().v202())
    v50 = (v33[:, 1:] - v33[:, :-1]).v238(2).v131()
    v47 = v47 + 0.01 * v240.v239(0.05 - v50)
    v48['loss'] = v22(v47.v202())
    v48['energy'] = v22(v50.v202())
    return (v47, v48)

@v39.v56()
def encode_text(v40, v52: v90, v53: v23, v54) -> v39.v21:
    v55 = v39.v132([[v53.v250(v243, 0) for v243 in v52]], device=v54)
    return v40.v120(v55)[0]

def cos(v57: v39.v21, v58: v39.v21) -> v22:
    return v22(v240.v203(v240.v211(v57, dim=0), v240.v211(v58, dim=0), dim=0))

def mine_same_suffix_pairs(v52: v90, v59: v204.v133, v60: v26=100):
    v61 = v19 + v18
    v62 = v134(v135)
    for v63 in v136(0, v209(v52) - v61 - 1, 17):
        v137 = v52[v63:v63 + v61]
        v62[v137[-v18:]].v205(v137)
    v64 = []
    for v138, v139 in v62.v140():
        v141 = {}
        for v142 in v139:
            v141[v142[:-v18]] = v142
            if v209(v141) >= 2:
                break
        if v209(v141) < 2:
            continue
        v143 = v135(v141.v241())
        v64.v205((v143[0], v143[1]))
        if v209(v64) >= v60 * 2:
            break
    v59.v144(v64)
    return v64[:v60]

def mine_diff_suffix_pairs(v52: v90, v59: v204.v133, v65: v26):
    v61 = v19 + v18
    v66 = []
    for v67 in v136(v65 * 4):
        v63 = v59.v206(0, v209(v52) - v61 - 1)
        v145 = v59.v206(0, v209(v52) - v61 - 1)
        v57, v58 = (v52[v63:v63 + v61], v52[v145:v145 + v61])
        if v57[-v18:] != v58[-v18:]:
            v66.v205((v57, v58))
        if v209(v66) >= v65:
            break
    return v66

@v39.v56()
def gate_A(v40, v52, v53, v54, v59) -> v23:
    v68 = v146(v52, v59, 100)
    v69 = v147(v52, v59, 100)
    v148, v149, v150 = ([], [], [])
    for v57, v58 in v68:
        v207, v208 = (v210(v40, v57, v53, v54), v210(v40, v58, v53, v54))
        v148.v205(v212(v207[-1], v208[-1]))
        v63 = v19 - 1
        v150.v205(v212(v207[v63], v208[v63]))
    for v57, v58 in v69:
        v207, v208 = (v210(v40, v57, v53, v54), v210(v40, v58, v53, v54))
        v149.v205(v212(v207[-1], v208[-1]))
    v70 = v22(v217.v131(v148))
    v71 = v22(v217.v131(v149))
    v72 = v22(v217.v131(v150))
    if v70 >= 0.98:
        v151 = 'A_FAIL_STILL_SUFFIX_WIPED'
    elif v70 < 0.9 and v70 - v71 < 0.35:
        v151 = 'A_PASS_PREFIX_VISIBLE_AT_ENDPOINT'
    else:
        v151 = 'A_WEAK_PARTIAL'
    return {'verdict': v151, 'mean_cos_endpoint_same_suffix': v70, 'mean_cos_endpoint_diff_suffix': v71, 'mean_cos_at_prefix_end': v72, 'n_same': v209(v148), 'n_diff': v209(v149)}
v24 = [('The cat sat on the mat.', 'A cat was sitting on the mat.'), ('She quickly opened the door.', 'She opened the door quickly.'), ('He bought a new car yesterday.', 'Yesterday he purchased a new automobile.'), ('The weather is very cold today.', 'It is extremely chilly outside today.'), ('Children are playing in the park.', 'Kids are playing at the park.'), ('I need to finish this work soon.', 'I must complete this task shortly.'), ('Please close the window.', 'Could you shut the window?'), ('The train leaves at noon.', 'The train departs at midday.'), ('He is afraid of spiders.', 'Spiders scare him.'), ('The film was long and boring.', 'The movie was lengthy and dull.'), ('We should start the meeting now.', "Let's begin the meeting now."), ('His answer was completely wrong.', 'His reply was totally incorrect.')]
v25 = [('The cat sat on the mat.', 'The car sat on the mat.'), ('She opened the door quickly.', 'She opened the book quickly.'), ('He bought a new car yesterday.', 'He bought a new cat yesterday.'), ('The weather is very cold today.', 'The weather is very warm today.'), ('The train leaves at noon.', 'The plane leaves at noon.'), ('She teaches mathematics at school.', 'She teaches history at school.')]

@v39.v56()
def gate_B(v40, v53, v54, v59) -> v23:

    def summ(v52):
        v33 = v210(v40, v52, v53, v54)
        return v240.v211(v39.v236([v33[-1], v33.v131(0)], 0), dim=0)
    v73 = [v212(v242(v57), v242(v58)) for v57, v58 in v24]
    v74 = [v212(v242(v57), v242(v58)) for v57, v58 in v25]
    v75 = []
    for v57, v58 in v24:
        v75.v213([v242(v57), v242(v58)])
    v76 = []
    for v67 in v136(v209(v73) * 4):
        v63, v145 = v59.v214(v136(v209(v75)), 2)
        v76.v205(v212(v75[v63], v75[v145]))
    v152, v153, v154 = (v22(v217.v131(v73)), v22(v217.v131(v76)), v22(v217.v131(v74)))
    v155, v156 = (v152 - v153, v152 - v154)
    if v155 > 0.05 and v156 > 0.03:
        v151 = 'B_PASS_MEANING_STRUCTURE'
    elif v156 <= 0.0 and v155 > 0.02:
        v151 = 'B_FAIL_FORM_NOT_MEANING'
    elif v155 <= 0.02:
        v151 = 'B_FAIL_NO_PARAPHRASE_CLUSTER'
    else:
        v151 = 'B_WEAK_MIXED'
    return {'verdict': v151, 'mean_cos_paraphrase': v152, 'mean_cos_random': v153, 'mean_cos_hard_spelling': v154, 'lift_vs_random': v155, 'lift_vs_hard': v156}

def main() -> v26:
    v77 = v215.v157()
    v77.v158('--steps', type=v26, default=v16)
    v77.v158('--device', default='cuda' if v39.v255.v251() else 'cpu')
    v78 = v77.v159()
    v0.v92(parents=True, exist_ok=True)
    v1.v92(parents=True, exist_ok=True)
    v2.v94('', encoding='utf-8')
    v160(f'Stage175 start {v256.v252(v257.v253).v228()}')
    v160('Attn pen (causal Transformer) → fit dynamics → freeze → gates A/B')
    v160(f'plan={v6}')
    v79 = v216.v161(max_chars=20000000)
    v53, v162 = v216.v163(v79)
    v55 = v217.v164((v53[v243] for v243 in v79), dtype=v217.v218, count=v209(v79))
    v160(f'corpus={v209(v55)} vocab={v209(v162)} pen=AttnL{v9}d{v8} seq={v12}')
    v54 = v39.v54(v78.v54)
    v39.v165(v7)
    v204.v166(v7)
    v217.v204.v166(v7)
    v40 = v119(v209(v162)).v167(v54)
    v80 = v39.v219.v168(v40.v171(), lr=v14, weight_decay=0.0001)
    v59 = v204.v133(v7)
    v81 = v79[v26(0.7 * v209(v79)):v26(0.7 * v209(v79)) + 2000000]
    v82 = v169(v40, v81, v53, v54, v204.v133(v7))
    v160(f"  init A: same={v82['mean_cos_endpoint_same_suffix']:.3f} diff={v82['mean_cos_endpoint_diff_suffix']:.3f} → {v82['verdict']}")
    v40.v170()
    v83 = None
    for v84 in v136(1, v78.v176 + 1):
        v41 = v216.v220(v55, v13, v12, v59, v54)
        v47, v221 = v222(v40, v41)
        v80.v223(set_to_none=True)
        v47.v224()
        v32.v244.v225(v40.v171(), 1.0)
        v80.v84()
        v83 = v221['loss'] if v83 is None else 0.95 * v83 + 0.05 * v221['loss']
        if v84 % v15 == 0 or v84 == v78.v176:
            v40.v172()
            v226 = v169(v40, v81, v53, v54, v204.v133(v7 + v84))
            v160(f"  step {v84}: loss~{v83:.3f} cos_d1={v221.v250('cos_d_k1', 0):.3f} A_same={v226['mean_cos_endpoint_same_suffix']:.3f} A_diff={v226['mean_cos_endpoint_diff_suffix']:.3f} → {v226['verdict']}")
            v40.v170()
    for v85 in v40.v116.v171():
        v85.v227(False)
    v40.v116.v172()
    v160('pen FROZEN — final A/B gates')
    v86 = v169(v40, v81, v53, v54, v204.v133(v7 + 99))
    v87 = v173(v40, v53, v54, v204.v133(v7 + 100))
    v160(f"  FINAL A: same={v86['mean_cos_endpoint_same_suffix']:.3f} diff={v86['mean_cos_endpoint_diff_suffix']:.3f} pref={v86['mean_cos_at_prefix_end']:.3f} → {v86['verdict']}")
    v160(f"  FINAL B: para={v87['mean_cos_paraphrase']:.3f} rand={v87['mean_cos_random']:.3f} hard={v87['mean_cos_hard_spelling']:.3f} → {v87['verdict']}")
    if 'PASS' in v86['verdict'] and 'PASS' in v87['verdict']:
        v174 = 'ATTN_PEN_CONTEXT_YES'
    elif 'PASS' in v86['verdict']:
        v174 = 'ATTN_PEN_CONTEXT_PARTIAL'
    elif 'FAIL' in v86['verdict']:
        v174 = 'ATTN_PEN_CONTEXT_NULL'
    else:
        v174 = 'ATTN_PEN_CONTEXT_WEAK'
    v39.v175({'model': v40.v245(), 'stoi': v53, 'itos': v162, 'step': v78.v176, 'pen': 'causal_transformer', 'A': v86, 'B': v87}, v5)
    v66 = {'timestamp': v256.v252(v257.v253).v228(), 'protocol': 'attn_pen_context_gate_175', 'overall': v174, 'steps': v78.v176, 'arch': {'pen': 'causal_transformer', 'd': v8, 'layers': v9, 'heads': v10}, 'A': v86, 'B': v87, 'init_A': v82, 'note': 'Compared to GRU pen A_FAIL (same_suf cos=1.0). Gate is endpoint context retention.', 'next': 'If PARTIAL/YES: harden freeze+dyn, extend B. If NULL: try multi-state/memory pen, not deeper same attn only.'}
    v177(v3, v66)
    v88 = [f'`{v174}` steps={v78.v176}', f"A: {v86['verdict']} same={v86['mean_cos_endpoint_same_suffix']:.3f} diff={v86['mean_cos_endpoint_diff_suffix']:.3f}", f"B: {v87['verdict']} para={v87['mean_cos_paraphrase']:.3f} hard={v87['mean_cos_hard_spelling']:.3f}", f'GRU baseline A was same≈1.0 — attn must beat that wipe', v66['next']]
    v4.v94('\n'.v229(['# Stage175 — attn pen context gate', '', f'**Overall:** `{v174}`', ''] + [f'- {v58}' for v58 in v88] + ['']), encoding='utf-8')
    v160(f'[175] {v174}')
    return 0
if v89 == '__main__':
    raise v178(v230())