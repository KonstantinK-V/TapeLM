"""
Stage 177 — Curve-BPE (real statistical tokens on the path).

Unlike 176 (whitespace words), this mirrors GPT/BPE:
  - merges by corpus frequency, not meaning / grammar rules
  - frequent forms often = 1 piece; rare/new = several pieces
  - leading space is INSIDE the token (ByteLevel Ġ / decoded space)

Each BPE piece → continuous arc vector (local char pool on the ink of that piece).
Causal Transformer predicts next-arc / Δ. NO BPE-id CE teacher.

Gate A: same last BPE piece string, different prefixes → endpoint wipe?

  python _stage177_curve_bpe.py
  python _stage177_curve_bpe.py --steps 15000 --vocab 4096
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import _stage170_curve_dynamics as s170
v0 = v26('results')
v1 = v26('checkpoints')
v2 = v0 / '_stage177_log.txt'
v3 = v0 / 'stage177_decision.json'
v4 = v0 / 'stage177_mini.md'
v5 = v0 / 'stage177_curve_bpe_tokenizer.json'
v6 = v1 / 'stage177_curve_bpe.pt'
v7 = v0 / 'plan_curve_dynamics.md'
v8 = 177
v9 = 128
v10 = 4
v11 = 4
v12 = 64
v13 = 24
v14 = 24
v15 = 0.0003
v16 = 1500
v17 = 15000
v18 = 4096
v19 = '[PAD]'

def log(v27: v45) -> None:
    v28 = v27 if v27.v196('\n') else v27 + '\n'
    try:
        v197(v28, end='', flush=True)
    except v109:
        v197(v28.v120('ascii', 'replace').v209('ascii'), end='', flush=True)
    v2.v198.v110(parents=True, exist_ok=True)
    with v2.v199('a', encoding='utf-8') as v111:
        v111.v200(v28)

def write_json(v29: v26, v30: v21) -> None:
    v29.v198.v110(parents=True, exist_ok=True)
    v29.v112(v253.v201(v30, indent=2, ensure_ascii=False), encoding='utf-8')

def train_or_load_bpe(v31: v45, v32: v25, v33: v113=False) -> v20:
    """GPT-2-style ByteLevel BPE: space lives inside tokens (Ġ…)."""
    if v5.v202() and (not v33):
        v34 = v20.v203(v45(v5))
        v114(f'[bpe] reuse {v5.v279} V={v34.v278()}')
        return v34
    v114(f'[bpe] train ByteLevel BPE V={v32} on {v208(v31):,} chars')
    v34 = v20(v254.v204(unk_token=None))
    v34.v35 = v205.v115(add_prefix_space=False)
    v34.v36 = v206.v115()
    v37 = v207.v116(vocab_size=v32, special_tokens=[v19], show_progress=False, initial_alphabet=v205.v115.v255())
    v38 = [v117 for v117 in v31.v256() if v117.v231()]
    if v208(v38) < 100:
        v38 = [v31[v155:v155 + 256] for v155 in v159(0, v291(v208(v31), 5000000), 128)]
    v34.v118(v38, trainer=v37)
    v5.v198.v110(parents=True, exist_ok=True)
    v34.v119(v45(v5))
    v114(f'[bpe] saved {v5} V={v34.v278()}')
    return v34

def encode_pieces(v34: v20, v39: v45) -> v44[v45]:
    """Return decoded surface strings of BPE pieces (space may be inside)."""
    v40 = v34.v120(v39)
    v41 = []
    for v42 in v40.v43:
        if v42 == v34.v257(v19):
            continue
        v121 = v34.v209([v42], skip_special_tokens=False)
        if v121 == '' and v42 is not None:
            v210 = {v258: v259 for v259, v258 in v34.v299().v171()}
            v121 = v210.v260(v42, '')
        if v121:
            v41.v215(v121)
    return v41

def bpe_stats(v34: v20, v46: v45, v47: v25=2000) -> v21:
    """Show frequent vs rare word fragmentation (BPE property check)."""
    import re
    v48 = v211.v122('\\S+', v46[:800000])
    if v208(v48) > v47:
        v48 = v48[:v47]
    v49 = v123(v48)
    v50 = [v55 for v55, v70 in v49.v261(40)]
    v51 = [v55 for v55, v185 in v49.v261() if v185 == 1][:40]
    if v208(v51) < 20:
        v51 = [v55 for v55, v185 in v49.v261() if v185 <= 2][:40]

    def n_pieces(v55: v45, v124: v113) -> v25:
        v121 = ' ' + v55 if v124 else v55
        return v212(1, v208(v214(v34, v121)))
    v52 = [v213(v55, True) for v55 in v50]
    v53 = [v213(v55, True) for v55 in v51] if v51 else [0]
    v54 = []
    for v55 in v50[:5] + v51[:5]:
        v125 = v214(v34, ' ' + v55)
        v54.v215({'word': v55, 'pieces': v125, 'n': v208(v125)})
    v56 = v126((1 for v97 in v54 for v262 in v97['pieces'] if v262.v280(' ') or v262.v280('Ġ') or 'Ġ' in v262))
    v57 = 0
    for v55 in v50[:30]:
        v125 = v214(v34, ' the ' + v55)
        if v125 and (v125[0].v280(' ') or v281((v262.v280(' ') for v262 in v125[:2]))):
            v57 += 1
    return {'freq_mean_pieces': v24(v287.v165(v52)) if v52 else 0.0, 'rare_mean_pieces': v24(v287.v165(v53)) if v53 else 0.0, 'freq_frac_single': v24(v287.v165([v297 == 1 for v297 in v52])) if v52 else 0.0, 'rare_frac_single': v24(v287.v165([v297 == 1 for v297 in v53])) if v53 else 0.0, 'examples': v54[:10], 'note': 'freq should be closer to 1 piece; rare more fragmented; space inside pieces'}

def pieces_to_char_ids(v41: v44[v45], v58: v21, v59: v25=v13) -> v62.v22:
    v60 = []
    for v61 in v41:
        v43 = [v58.v260(v185, 0) for v185 in v61[:v59]]
        if v208(v43) < v59:
            v43 = v43 + [0] * (v59 - v208(v43))
        v60.v215(v43)
    if not v60:
        v60 = [[0] * v59]
    return v62.v127(v60, dtype=v62.v216)

class ArcEncoder(v63.v23):

    def __init__(v128, v129: v25, v130: v25=v9):
        v282().v217()
        v128.v131 = v63.v218(v129, v130, padding_idx=0)
        v128.v132 = v63.v219(v63.v263(v130, v130), v63.v264(), v63.v263(v130, v130))
        v128.v133 = v63.v220(v130)

    def forward(v128, v72: v62.v22) -> v62.v22:
        v134 = v128.v131(v72)
        v135 = (v72 != 0).v24().v221(-1)
        v136 = (v134 * v135).v126(dim=-2) / v135.v126(dim=-2).v265(min=1.0)
        return v128.v133(v128.v132(v136))

class CausalBlock(v63.v23):

    def __init__(v128, v130: v25, v137: v25):
        v282().v217()
        v128.v138 = v63.v222(v130, v137, batch_first=True, dropout=0.1)
        v128.v139 = v63.v220(v130)
        v128.v132 = v63.v219(v63.v263(v130, 4 * v130), v63.v264(), v63.v263(4 * v130, v130))
        v128.v140 = v63.v220(v130)

    def forward(v128, v141: v62.v22, v142: v62.v22 | None=None) -> v62.v22:
        v143 = v141.v223(1)
        v144 = v62.v224(v62.v266(v143, v143, device=v141.v69, dtype=v62.v113), diagonal=1)
        v134, v70 = v128.v138(v141, v141, v141, attn_mask=v144, key_padding_mask=v142)
        v141 = v128.v139(v141 + v134)
        return v128.v140(v141 + v128.v132(v141))

class ArcTransformer(v63.v23):

    def __init__(v128, v130: v25=v9, v145: v25=v10):
        v282().v217()
        v128.v146 = v63.v218(v12, v130)
        v128.v147 = v63.v225([v283(v130, v11) for v70 in v159(v145)])

    def forward(v128, v148: v62.v22, v149: v62.v22 | None=None) -> v62.v22:
        v226, v227, v70 = v148.v150
        v146 = v62.v298(v227, device=v148.v69).v221(0).v228(v226, v227)
        v141 = v148 + v128.v146(v146)
        for v151 in v128.v147:
            v141 = v151(v141, key_padding_mask=v149)
        return v141

class CurveBPEModel(v63.v23):

    def __init__(v128, v129: v25):
        v282().v217()
        v128.v152 = v229(v129)
        v128.v153 = v230()
        v128.v77 = v63.v219(v63.v263(v9, v9), v63.v264(), v63.v263(v9, v9))

    def encode_arcs(v128, v72: v62.v22) -> v62.v22:
        return v128.v152(v72)

    def forward_states(v128, v72: v62.v22, v149: v62.v22 | None=None) -> v62.v22:
        return v128.v153(v128.v164(v72), pad_mask=v149)

def build_piece_docs(v34: v20, v39: v45, v64: v25=4000) -> v44[v44[v45]]:
    v65 = []
    for v66 in v39.v154('\n\n'):
        v66 = v66.v231()
        if v208(v66) < 40:
            continue
        v125 = v214(v34, v66)
        if v208(v125) >= 16:
            v65.v215(v125)
        if v208(v65) >= v64:
            break
    if v208(v65) < 50:
        v125 = v214(v34, v39[:2000000])
        for v155 in v159(0, v212(1, v208(v125) - 64), 48):
            v65.v215(v125[v155:v155 + 128])
            if v208(v65) >= v64:
                break
    return v65

def sample_batch(v65: v44[v44[v45]], v58: v21, v67: v25, v68: v232.v156, v69):
    v157, v158 = ([], [])
    for v70 in v159(v67):
        v88 = v65[v68.v267(0, v208(v65) - 1)]
        if v208(v88) < 8:
            v88 = v88 * 4
        v160 = v212(0, v208(v88) - v12)
        v121 = v68.v267(0, v160) if v160 > 0 else 0
        v161 = v88[v121:v121 + v12]
        v162 = v12 - v208(v161)
        if v162 > 0:
            v161 = v161 + [''] * v162
        v157.v215(v268(v161, v58))
        v158.v215(v62.v127([v61 == '' for v61 in v161], dtype=v62.v113))
    return (v62.v284(v157, 0).v166(v69), v62.v284(v158, 0).v166(v69))

def train_loss(v71: v163, v72: v62.v22, v73: v62.v22):
    v74 = v71.v164(v72)
    v75 = v71.v153(v74, pad_mask=v73)
    v76 = ~v73[:, :-1] & ~v73[:, 1:]
    if v76.v126() < 1:
        return (v75.v126() * 0.0, {'loss': 0.0, 'cos': 0.0, 'cos_d': 0.0})
    v77 = v71.v77(v75[:, :-1])
    v78 = v74[:, 1:]
    v79 = v77 - v75[:, :-1]
    v80 = v74[:, 1:] - v74[:, :-1]
    v81 = v270.v234(v77[v76], v78[v76].v285(), dim=-1).v165()
    v82 = v270.v234(v79[v76], v80[v76].v285(), dim=-1).v165()
    v83 = 1.0 - v81 + (1.0 - v82) + 0.1 * v270.v269(v77[v76], v78[v76].v285())
    return (v83, {'loss': v24(v83.v285()), 'cos': v24(v81.v285()), 'cos_d': v24(v82.v285())})

@v62.v84()
def encode_seq(v71, v41: v44[v45], v58, v69) -> v62.v22:
    v41 = v41[-v12:] or ['.']
    v72 = v268(v41, v58).v221(0).v166(v69)
    v73 = v62.v167(1, v208(v41), dtype=v62.v113, device=v69)
    return v71.v233(v72, pad_mask=v73)[0]

def cos(v61, v85) -> v24:
    return v24(v270.v234(v270.v271(v61, dim=0), v270.v271(v85, dim=0), dim=0))

def gate_A(v71, v65: v44[v44[v45]], v58, v69, v68, v86: v25=80) -> v21:
    v87 = v168(v44)
    for v88 in v65:
        if v208(v88) < 12:
            continue
        for v155 in v159(8, v208(v88)):
            v235 = v272(v88[v212(0, v155 - 24):v155])
            v169 = v88[v155]
            v87[v169].v215(v44(v235) + [v169])
    v89 = []
    for v169, v170 in v87.v171():
        v172 = {}
        for v121 in v170:
            v236 = v272(v121[:-1])
            if v236 not in v172:
                v172[v236] = v121
            if v208(v172) >= 2:
                break
        if v208(v172) >= 2:
            v237 = v44(v172.v286())
            v89.v215((v237[0], v237[1]))
        if v208(v89) >= v86:
            break
    v68.v173(v89)
    v89 = v89[:v86]
    v90 = []
    v91 = [v121 for v170 in v44(v87.v286())[:200] for v121 in v170[:3]]
    for v70 in v159(v86 * 4):
        if v208(v91) < 2:
            break
        v61, v85 = v68.v238(v91, 2)
        if v61[-1] != v85[-1]:
            v90.v215((v61, v85))
        if v208(v90) >= v86:
            break
    v174, v175 = ([], [])
    for v61, v85 in v89:
        v174.v215(v273(v292(v71, v61, v58, v69)[-1], v292(v71, v85, v58, v69)[-1]))
    for v61, v85 in v90:
        v175.v215(v273(v292(v71, v61, v58, v69)[-1], v292(v71, v85, v58, v69)[-1]))
    v92 = v24(v287.v165(v174)) if v174 else 1.0
    v93 = v24(v287.v165(v175)) if v175 else 0.0
    if v92 >= 0.98:
        v176 = 'A_FAIL_LAST_PIECE_WIPES'
    elif v92 < 0.9 and v92 - v93 < 0.35:
        v176 = 'A_PASS_PREFIX_VISIBLE'
    else:
        v176 = 'A_WEAK_PARTIAL'
    return {'verdict': v176, 'mean_cos_same_last_piece': v92, 'mean_cos_diff_last_piece': v93, 'n_same': v208(v174), 'n_diff': v208(v175)}

def main() -> v25:
    v94 = v239.v177()
    v94.v178('--steps', type=v25, default=v17)
    v94.v178('--vocab', type=v25, default=v18)
    v94.v178('--force-bpe', action='store_true')
    v94.v178('--device', default='cuda' if v62.v293.v288() else 'cpu')
    v95 = v94.v179()
    v0.v110(parents=True, exist_ok=True)
    v1.v110(parents=True, exist_ok=True)
    v2.v112('', encoding='utf-8')
    v114(f'Stage177 start {v295.v289(v296.v290).v250()}')
    v114('Curve-BPE: statistical merges + space-in-token; continuous next-arc (no CE)')
    v114(f'plan={v7}')
    v39 = v240.v180(max_chars=20000000)
    v34 = v181(v39, v95.v182, force=v95.v241)
    v96 = v183(v34, v39)
    v114(f"[bpe] freq_mean_pcs={v96['freq_mean_pieces']:.2f} rare_mean_pcs={v96['rare_mean_pieces']:.2f} freq_single={v96['freq_frac_single']:.2f} rare_single={v96['rare_frac_single']:.2f}")
    for v97 in v96['examples'][:6]:
        v114(f"  ex {v97['word']!r} → {v97['pieces']!r} (n={v97['n']})")
    v98 = v184(v274(v39) | {' '})
    v99 = ['<pad>'] + v98
    v58 = {v185: v155 + 1 for v155, v185 in v275(v98)}
    v65 = v186(v34, v39)
    v114(f'docs={v208(v65)} char_vocab={v208(v99)} max_arcs={v12} d={v9}')
    v69 = v62.v69(v95.v69)
    v62.v187(v8)
    v232.v188(v8)
    v71 = v163(v208(v99)).v166(v69)
    v100 = v62.v242.v189(v71.v243(), lr=v15, weight_decay=0.0001)
    v68 = v232.v156(v8)
    v101 = v65[v25(0.8 * v208(v65)):] or v65[-100:]
    v102 = v65[:v25(0.8 * v208(v65))] or v65
    v103 = v190(v71, v101, v58, v69, v232.v156(v8))
    v114(f"  init A: same={v103['mean_cos_same_last_piece']:.3f} diff={v103['mean_cos_diff_last_piece']:.3f} → {v103['verdict']}")
    v71.v191()
    v104 = None
    v105 = v103
    for v106 in v159(1, v95.v193 + 1):
        v141, v73 = v244(v102, v58, v14, v68, v69)
        v83, v245 = v246(v71, v141, v73)
        v100.v247(set_to_none=True)
        v83.v248()
        v63.v276.v249(v71.v243(), 1.0)
        v100.v106()
        v104 = v245['loss'] if v104 is None else 0.95 * v104 + 0.05 * v245['loss']
        if v106 % v16 == 0 or v106 == v95.v193:
            v71.v277()
            v105 = v190(v71, v101, v58, v69, v232.v156(v8 + v106))
            v114(f"  step {v106}: loss~{v104:.3f} cos_next={v245['cos']:.3f} cos_d={v245['cos_d']:.3f} A_same={v105['mean_cos_same_last_piece']:.3f} A_diff={v105['mean_cos_diff_last_piece']:.3f} → {v105['verdict']}")
            v71.v191()
            v62.v119({'model': v71.v294(), 'stoi': v58, 'itos': v99, 'step': v106, 'A': v105, 'vocab': v95.v182}, v6)
    if 'PASS' in v105['verdict']:
        v192 = 'CURVE_BPE_CONTEXT_YES'
    elif 'WEAK' in v105['verdict']:
        v192 = 'CURVE_BPE_CONTEXT_WEAK'
    else:
        v192 = 'CURVE_BPE_CONTEXT_NULL'
    v107 = {'timestamp': v295.v289(v296.v290).v250(), 'protocol': 'curve_bpe_177', 'overall': v192, 'practical': 'STILL_LAST_UNIT_WIPE' if v105['mean_cos_same_last_piece'] >= 0.95 else 'PARTIAL' if 'WEAK' in v105['verdict'] else v105['verdict'], 'steps': v95.v193, 'bpe': {'style': 'ByteLevel BPE (GPT-2-like); space inside tokens', 'vocab_size': v34.v278(), 'tokenizer': v45(v5), 'stats': {v259: v258 for v259, v258 in v96.v171() if v259 != 'examples'}, 'examples': v96['examples']}, 'vs_176': '176=whitespace words; 177=statistical merges + space-in-token', 'loss': 'next-piece cosine + piece-Delta cosine (no BPE-id CE)', 'A': v105, 'init_A': v103, 'next': 'If still wipe: retention/instance channel. If PASS: gate B paraphrase.'}
    v194(v3, v107)
    v4.v112('\n'.v251(['# Stage177 — curve BPE', '', f'**Overall:** `{v192}`', '', f"- ByteLevel BPE V={v34.v278()}; freq_pcs={v96['freq_mean_pieces']:.2f} rare_pcs={v96['rare_mean_pieces']:.2f}", f"- A: {v105['verdict']} same={v105['mean_cos_same_last_piece']:.3f} diff={v105['mean_cos_diff_last_piece']:.3f}", f'- vs 176: statistical merges + space-in-token (not whitespace words)', f"- {v107['next']}", '']), encoding='utf-8')
    v114(f'[177] {v192}')
    return 0
if v108 == '__main__':
    raise v195(v252())