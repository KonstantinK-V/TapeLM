"""
Stage 176 — Curve-as-tokens (arcs).

Split the drawn path into arc-tokens (whitespace/punct segments).
Each arc → one vector (local char encoder).
Causal Transformer over arc sequence predicts next arc / Δ.
NO char/word CE teacher.

Gate A (arc-level): same last arc string, different prefix arcs →
  does final state still wipe? (compare to char-GRU/attn wipe)

  python _stage176_curve_tokens.py
  python _stage176_curve_tokens.py --steps 15000
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import _stage170_curve_dynamics as s170
v0 = v23('results')
v1 = v23('checkpoints')
v2 = v0 / '_stage176_log.txt'
v3 = v0 / 'stage176_decision.json'
v4 = v0 / 'stage176_mini.md'
v5 = v1 / 'stage176_curve_tokens.pt'
v6 = v0 / 'plan_curve_dynamics.md'
v7 = 176
v8 = 128
v9 = 4
v10 = 4
v11 = 64
v12 = 24
v13 = 24
v14 = 0.0003
v15 = 1500
v16 = 15000
v17 = v81.v24('\\S+|\\s+')

def log(v25: v32) -> None:
    v26 = v25 if v25.v167('\n') else v25 + '\n'
    try:
        v168(v26, end='', flush=True)
    except v82:
        v168(v26.v249('ascii', 'replace').v235('ascii'), end='', flush=True)
    v2.v169.v83(parents=True, exist_ok=True)
    with v2.v170('a', encoding='utf-8') as v84:
        v84.v171(v26)

def write_json(v27: v23, v28: v21) -> None:
    v27.v169.v83(parents=True, exist_ok=True)
    v27.v85(v219.v172(v28, indent=2), encoding='utf-8')

def split_arcs(v29: v32) -> v31[v32]:
    """Token-like arcs: non-space chunks; drop pure whitespace (boundary only)."""
    v30 = v17.v86(v29)
    return [v87 for v87 in v30 if v87.v220()]

def arcs_to_char_ids(v33: v31[v32], v34: v21, v35: v22=v12) -> v38.v18:
    """[n_arcs, max_chars] with pad 0; also lengths."""
    v36 = []
    for v37 in v33:
        v88 = [v34.v221(v156, 0) for v156 in v37[:v35]]
        if v198(v88) < v35:
            v88 = v88 + [0] * (v35 - v198(v88))
        v36.v173(v88)
    return v38.v89(v36, dtype=v38.v174)

class ArcEncoder(v39.v19):
    """Local char → one arc vector (mean pool over chars). Ink for one token."""

    def __init__(v90, v91: v22, v92: v22=v8):
        v236().v175()
        v90.v93 = v39.v176(v91, v92, padding_idx=0)
        v90.v94 = v39.v177(v39.v222(v92, v92), v39.v223(), v39.v222(v92, v92))
        v90.v95 = v39.v178(v92)

    def forward(v90, v50: v38.v18) -> v38.v18:
        v96 = v90.v93(v50)
        v97 = (v50 != 0).v20().v179(-1)
        v98 = (v96 * v97).v180(dim=-2)
        v99 = v97.v180(dim=-2).v181(min=1.0)
        v100 = v98 / v99
        return v90.v95(v90.v94(v100))

class CausalBlock(v39.v19):

    def __init__(v90, v92: v22, v101: v22):
        v236().v175()
        v90.v102 = v39.v182(v92, v101, batch_first=True, dropout=0.1)
        v90.v103 = v39.v178(v92)
        v90.v94 = v39.v177(v39.v222(v92, 4 * v92), v39.v223(), v39.v222(4 * v92, v92))
        v90.v104 = v39.v178(v92)

    def forward(v90, v47: v38.v18, v105: v38.v18 | None=None) -> v38.v18:
        v106 = v47.v183(1)
        v107 = v38.v184(v38.v224(v106, v106, device=v47.v43, dtype=v38.v199), diagonal=1)
        v96, v46 = v90.v102(v47, v47, v47, attn_mask=v107, key_padding_mask=v105)
        v47 = v90.v103(v47 + v96)
        return v90.v104(v47 + v90.v94(v47))

class ArcTransformer(v39.v19):

    def __init__(v90, v92: v22=v8, v108: v22=v9):
        v236().v175()
        v90.v109 = v39.v176(v11, v92)
        v90.v110 = v39.v185([v237(v92, v10) for v46 in v119(v108)])

    def forward(v90, v111: v38.v18, v112: v38.v18 | None=None) -> v38.v18:
        v186, v187, v92 = v111.v113
        v109 = v38.v250(v187, device=v111.v43).v179(0).v188(v186, v187)
        v47 = v111 + v90.v109(v109)
        for v114 in v90.v110:
            v47 = v114(v47, key_padding_mask=v112)
        return v47

class CurveTokenModel(v39.v19):

    def __init__(v90, v91: v22):
        v236().v175()
        v90.v115 = v189(v91)
        v90.v116 = v190()
        v90.v117 = v39.v177(v39.v222(v8, v8), v39.v223(), v39.v222(v8, v8))

    def encode_arcs(v90, v50: v38.v18) -> v38.v18:
        return v90.v115(v50)

    def forward_states(v90, v50: v38.v18, v112: v38.v18 | None=None) -> v38.v18:
        v33 = v90.v191(v50)
        return v90.v116(v33, pad_mask=v112)

def sample_arc_batch(v40: v31[v31[v32]], v34: v21, v41: v22, v42: v192.v118, v43):
    """Sample contiguous arc windows."""
    v44 = []
    v45 = []
    for v46 in v119(v41):
        v60 = v40[v42.v225(0, v198(v40) - 1)]
        if v198(v60) < 8:
            v60 = v60 * 4
        v120 = v193(0, v198(v60) - v11)
        v121 = v42.v225(0, v120) if v120 > 0 else 0
        v122 = v60[v121:v121 + v11]
        v123 = v11 - v198(v122)
        if v123 > 0:
            v122 = v122 + [''] * v123
        v50 = v194(v122, v34)
        v44.v173(v50)
        v97 = v38.v89([v37 == '' for v37 in v122], dtype=v38.v199)
        v45.v173(v97)
    v47 = v38.v226(v44, 0).v124(v43)
    v48 = v38.v226(v45, 0).v124(v43)
    return (v47, v48)

def train_loss(v49: v125, v50: v38.v18, v48: v38.v18) -> v51[v38.v18, v21]:
    with v38.v195(True):
        v126 = v49.v191(v50)
        v55 = v49.v116(v126, pad_mask=v48)
        v127 = ~v48[:, :-1] & ~v48[:, 1:]
        if v127.v180() < 1:
            return (v55.v180() * 0.0, {'loss': 0.0, 'cos': 0.0})
        v117 = v49.v117(v55[:, :-1])
        v128 = v126[:, 1:]
        v129 = v117 - v55[:, :-1]
        v130 = v126[:, 1:] - v126[:, :-1]
        v131 = v227.v200(v117[v127], v128[v127].v239(), dim=-1).v196()
        v132 = v227.v200(v129[v127], v130[v127].v239(), dim=-1).v196()
        v133 = 1.0 - v131 + (1.0 - v132)
        v133 = v133 + 0.1 * v227.v238(v117[v127], v128[v127].v239())
        v134 = {'loss': v20(v133.v239()), 'cos': v20(v131.v239()), 'cos_d': v20(v132.v239())}
        return (v133, v134)

def build_arc_corpus(v29: v32, v52: v22=4000) -> v31[v31[v32]]:
    v53 = []
    for v54 in v29.v135('\n\n'):
        v33 = v197(v54)
        if v198(v33) >= 16:
            v53.v173(v33)
        if v198(v53) >= v52:
            break
    if v198(v53) < 50:
        v33 = v197(v29)
        for v136 in v119(0, v198(v33) - 64, 32):
            v53.v173(v33[v136:v136 + 128])
            if v198(v53) >= v52:
                break
    return v53

@v38.v56()
def encode_arc_seq(v49, v33: v31[v32], v34, v43) -> v38.v18:
    v33 = v33[-v11:]
    if not v33:
        v33 = ['.']
    v50 = v194(v33, v34).v179(0).v124(v43)
    v48 = v38.v137(1, v198(v33), dtype=v38.v199, device=v43)
    v55 = v49.v138(v50, pad_mask=v48)
    return v55[0]

def cos(v37, v57) -> v20:
    return v20(v227.v200(v227.v228(v37, dim=0), v227.v228(v57, dim=0), dim=0))

def gate_A_arcs(v49, v53: v31[v31[v32]], v34, v43, v42, v58: v22=80) -> v21:
    """Same last arc string, different prefixes → endpoint state cos."""
    v59 = v139(v31)
    for v60 in v53:
        if v198(v60) < 12:
            continue
        for v136 in v119(8, v198(v60)):
            v201 = v51(v60[v193(0, v136 - 24):v136])
            v140 = v60[v136]
            v59[v140].v173(v31(v201) + [v140])
    v61 = []
    for v140, v141 in v59.v142():
        v143 = {}
        for v121 in v141:
            v202 = v51(v121[:-1])
            if v202 not in v143:
                v143[v202] = v121
            if v198(v143) >= 2:
                break
        if v198(v143) >= 2:
            v203 = v31(v143.v240())
            v61.v173((v203[0], v203[1]))
        if v198(v61) >= v58:
            break
    v42.v144(v61)
    v61 = v61[:v58]
    v62 = []
    v63 = [v121 for v141 in v31(v59.v240())[:200] for v121 in v141[:3]]
    for v46 in v119(v58 * 3):
        if v198(v63) < 2:
            break
        v37, v57 = v42.v204(v63, 2)
        if v37[-1] != v57[-1]:
            v62.v173((v37, v57))
        if v198(v62) >= v58:
            break
    v145, v146 = ([], [])
    for v37, v57 in v61:
        v147 = v229(v49, v37, v34, v43)[-1]
        v148 = v229(v49, v57, v34, v43)[-1]
        v145.v173(v230(v147, v148))
    for v37, v57 in v62:
        v147 = v229(v49, v37, v34, v43)[-1]
        v148 = v229(v49, v57, v34, v43)[-1]
        v146.v173(v230(v147, v148))
    v64 = v20(v241.v196(v145)) if v145 else 1.0
    v65 = v20(v241.v196(v146)) if v146 else 0.0
    if v64 >= 0.98:
        v149 = 'A_FAIL_LAST_ARC_WIPES'
    elif v64 < 0.9 and v64 - v65 < 0.35:
        v149 = 'A_PASS_PREFIX_VISIBLE'
    else:
        v149 = 'A_WEAK_PARTIAL'
    return {'verdict': v149, 'mean_cos_same_last_arc': v64, 'mean_cos_diff_last_arc': v65, 'n_same': v198(v145), 'n_diff': v198(v146)}

def main() -> v22:
    v66 = v205.v150()
    v66.v151('--steps', type=v22, default=v16)
    v66.v151('--device', default='cuda' if v38.v245.v242() else 'cpu')
    v67 = v66.v152()
    v0.v83(parents=True, exist_ok=True)
    v1.v83(parents=True, exist_ok=True)
    v2.v85('', encoding='utf-8')
    v153(f'Stage176 start {v247.v243(v248.v244).v216()}')
    v153('Curve-as-tokens: arcs (whitespace segments) + causal Transformer next-arc/Δ')
    v153(f'plan={v6}')
    v29 = v206.v154(max_chars=20000000)
    v68 = v155(v207(v29))
    v69 = ['<pad>'] + v68
    v34 = {v156: v136 + 1 for v136, v156 in v231(v68)}
    v53 = v157(v29)
    v153(f'docs={v198(v53)} vocab={v198(v69)} max_arcs={v11} d={v8} layers={v9}')
    v43 = v38.v43(v67.v43)
    v38.v158(v7)
    v192.v159(v7)
    v49 = v125(v198(v69)).v124(v43)
    v70 = v38.v208.v160(v49.v209(), lr=v14, weight_decay=0.0001)
    v42 = v192.v118(v7)
    v71 = v53[v22(0.8 * v198(v53)):] or v53[-100:]
    v72 = v53[:v22(0.8 * v198(v53))] or v53
    v73 = v161(v49, v71, v34, v43, v192.v118(v7))
    v153(f"  init A: same={v73['mean_cos_same_last_arc']:.3f} diff={v73['mean_cos_diff_last_arc']:.3f} → {v73['verdict']}")
    v49.v162()
    v74 = None
    v75 = v73
    for v76 in v119(1, v67.v164 + 1):
        v47, v48 = v210(v72, v34, v13, v42, v43)
        v133, v211 = v212(v49, v47, v48)
        v70.v213(set_to_none=True)
        v133.v214()
        v39.v232.v215(v49.v209(), 1.0)
        v70.v76()
        v74 = v211['loss'] if v74 is None else 0.95 * v74 + 0.05 * v211['loss']
        if v76 % v15 == 0 or v76 == v67.v164:
            v49.v233()
            v187 = v161(v49, v71, v34, v43, v192.v118(v7 + v76))
            v75 = v187
            v153(f"  step {v76}: loss~{v74:.3f} cos_next={v211['cos']:.3f} cos_d={v211['cos_d']:.3f} A_same={v187['mean_cos_same_last_arc']:.3f} A_diff={v187['mean_cos_diff_last_arc']:.3f} → {v187['verdict']}")
            v49.v162()
            v38.v234({'model': v49.v246(), 'stoi': v34, 'itos': v69, 'step': v76, 'A': v187}, v5)
    v77 = v75
    if 'PASS' in v77['verdict']:
        v163 = 'CURVE_TOKENS_CONTEXT_YES'
    elif 'WEAK' in v77['verdict']:
        v163 = 'CURVE_TOKENS_CONTEXT_WEAK'
    else:
        v163 = 'CURVE_TOKENS_CONTEXT_NULL'
    v78 = {'timestamp': v247.v243(v248.v244).v216(), 'protocol': 'curve_as_tokens_176', 'overall': v163, 'steps': v67.v164, 'arc_definition': 'whitespace-separated non-space chunks (word/punct-like)', 'unit': 'arc vector from local char mean-pool; sequence model = causal Transformer', 'loss': 'next-arc cosine + arc-Δ cosine (no CE)', 'A': v77, 'init_A': v73, 'note': 'Analog of BPE tokens but continuous arc embeddings on the curve.', 'next': 'If YES/WEAK: add B paraphrase + harden. If NULL: arc unit still local-wipe — try longer memory / retention loss.'}
    v165(v3, v78)
    v79 = [f'`{v163}`', f"A: {v77['verdict']} same_last_arc={v77['mean_cos_same_last_arc']:.3f} diff={v77['mean_cos_diff_last_arc']:.3f}", 'arcs = whitespace segments; Transformer predicts next arc/Δ', v78['next']]
    v4.v85('\n'.v217(['# Stage176 — curve as tokens', '', f'**Overall:** `{v163}`', ''] + [f'- {v57}' for v57 in v79] + ['']), encoding='utf-8')
    v153(f'[176] {v163}')
    return 0
if v80 == '__main__':
    raise v166(v218())