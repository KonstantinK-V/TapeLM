"""
Stage 185 — Addressable tape vs endpoint (falsify "endpoint = ceiling").

Two matched models, identical budget/seed/data, pure next-piece CE (no hand losses):
  ENDPOINT : logits = head([fast_t ; slow_t])           — memory = one point
  TAPE     : logits = head([fast_t ; read_t]),
             read_t = causal attention of fast_t over slow_1..slow_t — memory = addressable tape

Judge = calibrated Stage184 exam (log-prob). Reference: GPT next_tok=0.758.
Ablation for TAPE: shuffle slow tape along time at eval → accuracy must drop,
else the tape is decorative.

Verdicts:
  TAPE_READ_YES        gain>=+0.03 over endpoint AND shuffle drop>=0.05
  TAPE_GAIN_BUT_DECOR  gain without ablation drop (suspicious)
  TAPE_USED_NO_GAIN    ablation drop but no gain
  ENDPOINT_ENOUGH_HERE neither

  python _stage185_tape_read.py
  python _stage185_tape_read.py --steps 3000
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
import _stage180_dual_channel as s180
import _stage181_ce_control as s181
v0 = v26('results')
v1 = v26('data')
v2 = v26('checkpoints')
v3 = v0 / '_stage185_log.txt'
v4 = v0 / 'stage185_decision.json'
v5 = v0 / 'stage185_mini.md'
v6 = v1 / 'stage184_exam.jsonl'
v7 = v0 / 'stage184_decision.json'
v8 = v27.v8
v9 = 185
v10 = v28.v10
v11 = v28.v11
v12 = v27.v12
v13 = v27.v14
v15 = 16
v16 = 0.0003
v17 = 1000
v18 = 3000
v19 = 60
v20 = '[PAD]'

def log(v29: v99) -> None:
    v30 = v29 if v29.v157('\n') else v29 + '\n'
    try:
        v158(v30, end='', flush=True)
    except v100:
        v158(v30.v226('ascii', 'replace').v202('ascii'), end='', flush=True)
    v3.v159.v101(parents=True, exist_ok=True)
    with v3.v160('a', encoding='utf-8') as v102:
        v102.v161(v30)

def write_json(v31: v26, v32: v24) -> None:
    v31.v159.v101(parents=True, exist_ok=True)
    v31.v103(v201.v162(v32, indent=2, ensure_ascii=False), encoding='utf-8')

def build_char_table(v33: v104, v34: v24, v35: v25, v36: v25) -> v39.v21:
    """[V, MAX_CHARS] char-id rows for every BPE id; pad row = zeros."""
    v37 = v39.v105(v36, v13, dtype=v39.v163)
    for v38 in v106(v36):
        if v38 == v35:
            continue
        v107 = v33.v202([v38], skip_special_tokens=False) or ''
        for v164, v148 in v123(v107[:v13]):
            v37[v38, v164] = v34.v181(v148, 0)
    return v37

class TapeReadModel(v40.v22):
    """mode='endpoint' → head([fast;slow_t]); mode='tape' → head([fast;read over slow_1..t])."""

    def __init__(v108, v74: v25, v36: v25, v70: v99):
        v218().v165()
        assert v70 in ('endpoint', 'tape')
        v108.v70 = v70
        v108.v109 = v28.v166(v74)
        if v70 == 'tape':
            v108.v167 = v40.v203(v11, num_heads=4, batch_first=True)
        v108.v110 = v40.v168(v10 + v11, v36, bias=False)

    def logits(v108, v51: v39.v21, v50: v39.v21, v58: v169=False) -> v39.v21:
        v46, v170, v112 = v108.v109.v171(v51, v50)
        if v108.v70 == 'endpoint':
            v115 = v39.v175([v170, v112], dim=-1)
            return v108.v110(v115)
        v111 = v112
        if v58:
            v113 = v112.v173(1)
            v172 = v39.v204(v113, device=v112.v44)
            v111 = v112[:, v172]
        v113 = v170.v173(1)
        v114 = v39.v174(v39.v152(v113, v113, dtype=v39.v169, device=v170.v44), diagonal=1)
        v167, v46 = v108.v167(v170, v111, v111, attn_mask=v114, key_padding_mask=v50, need_weights=False)
        v115 = v39.v175([v170, v167], dim=-1)
        return v108.v110(v115)

def sample_id_batch(v41, v42, v43, v44, v35):
    v45 = []
    for v46 in v106(v42):
        v89 = v41[v43.v205(0, v179(v41) - 1)]
        if v179(v89) < 8:
            v89 = v89 * 4
        v116 = v176(0, v179(v89) - v12)
        v117 = v43.v205(0, v116) if v116 > 0 else 0
        v118 = v89[v117:v117 + v12]
        if v179(v118) < v12:
            v118 = v118 + [v35] * (v12 - v179(v118))
        v45.v177(v118)
    return v39.v119(v45, dtype=v39.v163, device=v44)

def ce_step(v47, v48, v49, v35):
    v50 = v48 == v35
    v51 = v49[v48]
    v52 = v47.v52(v51, v50)
    v53 = v48[:, 1:]
    v54 = ~v50[:, :-1] & ~v50[:, 1:]
    v55 = v178.v120(v52[:, :-1][v54], v53[v54])
    return v55

@v39.v64()
def span_logprob(v47, v49, v35, v56, v57, v44, v58=False) -> v23:
    v59 = (v56 + v57)[-v12:]
    v60 = v179(v59) - v179(v57)
    v61 = v39.v119([v59], dtype=v39.v163, device=v44)
    v50 = v61 == v35
    v52 = v47.v52(v49[v61], v50, shuffle_tape=v58)[0]
    v62 = v178.v121(v52, dim=-1)
    v63 = 0.0
    for v122, v38 in v123(v57):
        v124 = v60 + v122 - 1
        v63 += v23(v62[v124, v38])
    return v63 / v176(1, v179(v57))

@v39.v64()
def score_exam(v47, v49, v35, v65, v44, v58=False, v66=None, v67='') -> v24:
    v47.v125()
    v68 = {}
    for v126, v92 in v123(v65):
        v127 = v92['type']
        if v66 and v127 != v66:
            continue
        v128 = [v206(v47, v49, v35, v92['ctx_ids'], v148, v44, v58) for v148 in v92['cand_ids']]
        v129 = v25(v197.v207(v128))
        v180, v130 = v68.v181(v127, (0, 0))
        v68[v127] = (v180 + v25(v129 == v92['gold_idx']), v130 + 1)
    v69 = {}
    for v127, (v180, v130) in v68.v65():
        v69[f'{v127}_acc'] = v180 / v176(1, v130)
        v69[f'{v127}_n'] = v130
    return v69

def train_variant(v70: v99, v71, v72, v73, v49, v35, v36, v74, v75, v44):
    v39.v131(v9)
    v47 = v208(v74, v36, v70).v132(v44)
    v76 = v39.v182.v133(v47.v183(), lr=v16, weight_decay=0.01)
    v43 = v184.v134(v9)
    v77 = None
    v78 = v135.v135()
    v47.v136()
    for v79 in v106(1, v75 + 1):
        v48 = v185(v71, v15, v43, v44, v35)
        v55 = v186(v47, v48, v49, v35)
        v76.v187(set_to_none=True)
        v55.v188()
        v40.v209.v189(v47.v183(), 1.0)
        v76.v79()
        v77 = v23(v55) if v77 is None else 0.95 * v77 + 0.05 * v23(v55)
        if v79 % v17 == 0 or v79 == v75:
            v190 = v137(v47, v49, v35, v72, v44, only_type='next_tok')
            v143(f"  [{v70}] step {v79}: ce~{v77:.3f} next_tok(mid)={v190.v181('next_tok_acc', 0):.3f} ({v135.v135() - v78:.0f}s)")
            v47.v136()
    v47.v125()
    v80 = v137(v47, v49, v35, v73, v44)
    v81 = {'ce_final': v77, **v80}
    if v70 == 'tape':
        v138 = v137(v47, v49, v35, v73, v44, shuffle_tape=True, only_type='next_tok')
        v81['next_tok_shuffled'] = v138.v181('next_tok_acc', 0.0)
    v39.v139({'model': v47.v210(), 'mode': v70, 'res': v81}, v2 / f'stage185_{v70}.pt')
    return v81

def main() -> v25:
    v82 = v191.v140()
    v82.v141('--steps', type=v25, default=v18)
    v82.v141('--device', default='cuda' if v39.v223.v219() else 'cpu')
    v83 = v82.v142()
    v0.v101(parents=True, exist_ok=True)
    v3.v103('', encoding='utf-8')
    v143(f'Stage185 start {v224.v221(v225.v222).v199()}')
    v143('Addressable tape (query read) vs endpoint — matched CE, judge = calibrated 184 exam')
    if not v6.v150():
        v143('FATAL: run _stage184_exam_logprob.py first (needs data/stage184_exam.jsonl)')
        return 1
    v73 = [v201.v192(v193) for v193 in v6.v215(encoding='utf-8').v211() if v193.v212()]
    v72 = [v92 for v92 in v73 if v92['type'] == 'next_tok'][:v19]
    v143(f'exam items={v179(v73)} mid={v179(v72)}')
    v44 = v39.v44(v83.v44)
    v33 = v104.v144(v99(v8))
    v36 = v33.v145()
    v35 = v33.v194(v20) or 0
    v84 = v195.v146(max_chars=20000000)
    v85 = v147(v213(v84) | {' '})
    v86 = ['<pad>'] + v85
    v34 = {v148: v126 + 1 for v126, v148 in v123(v85)}
    v41 = v196.v149(v33, v84)
    v71 = v41[:v25(0.8 * v179(v41))] or v41
    v143(f'docs={v179(v41)} V={v36} n_char={v179(v86)}')
    v49 = v214(v33, v34, v35, v36).v132(v44)
    v143('char table ready')
    v87 = None
    if v7.v150():
        v151 = v201.v192(v7.v215(encoding='utf-8'))
        v87 = v151['results']['ce_gpt_181']
    v88 = v197.v152(v36)
    for v89 in v71:
        for v127 in v89:
            v88[v127] += 1
    v90 = v197.v143(v88 / v88.v216())
    v91 = {'next_tok': [0, 0], 'entity': [0, 0], 'ood': [0, 0]}
    for v92 in v73:
        v128 = [v23(v197.v220([v90[v127] for v127 in v148])) for v148 in v92['cand_ids']]
        v180, v130 = v91[v92['type']]
        v91[v92['type']] = (v180 + v25(v25(v197.v207(v128)) == v92['gold_idx']), v130 + 1)
    v93 = {f'{v127}_acc': v180 / v176(1, v130) for v127, (v180, v130) in v91.v65()}
    v143(f"unigram baseline: next_tok={v93['next_tok_acc']:.3f} entity={v93['entity_acc']:.3f}")
    v94 = {}
    for v70 in ('endpoint', 'tape'):
        v143(f'train {v70} …')
        v94[v70] = v198(v70, v71, v72, v73, v49, v35, v36, v179(v86), v83.v75, v44)
        v153 = v94[v70]
        v143(f"  {v70} FINAL: next_tok={v153['next_tok_acc']:.3f} entity={v153.v181('entity_acc', 0):.3f} ood={v153.v181('ood_acc', 0):.3f}" + (f" shuffled={v153['next_tok_shuffled']:.3f}" if 'next_tok_shuffled' in v153 else ''))
    v95 = v94['tape']['next_tok_acc'] - v94['endpoint']['next_tok_acc']
    v96 = v94['tape']['next_tok_acc'] - v94['tape'].v181('next_tok_shuffled', v94['tape']['next_tok_acc'])
    if v95 >= 0.03 and v96 >= 0.05:
        v154 = 'TAPE_READ_YES'
    elif v95 >= 0.03:
        v154 = 'TAPE_GAIN_BUT_DECOR'
    elif v96 >= 0.05:
        v154 = 'TAPE_USED_NO_GAIN'
    else:
        v154 = 'ENDPOINT_ENOUGH_HERE'
    v69 = {'timestamp': v224.v221(v225.v222).v199(), 'protocol': 'tape_read_vs_endpoint_185', 'overall': v154, 'gain_tape_minus_endpoint': v95, 'shuffle_drop': v96, 'gpt_ref_184': v87, 'unigram_baseline': v93, 'steps': v83.v75, 'results': v94, 'note': 'matched budget/seed/data; pure CE; no retention/hand losses'}
    v155(v4, v69)
    v97 = ['# Stage185 — addressable tape vs endpoint', '', f'**Overall:** `{v154}`  (gain={v95:+.3f}, shuffle_drop={v96:+.3f})', '', f"GPT ref next_tok: {v87['next_tok_acc']:.3f}" if v87 else '', f"Unigram (no context): next_tok={v93['next_tok_acc']:.3f} — context credit is only what's above this"]
    for v70, v153 in v94.v65():
        v97.v177(f"- `{v70}`: next_tok={v153['next_tok_acc']:.3f} entity={v153.v181('entity_acc', 0):.3f} ood={v153.v181('ood_acc', 0):.3f}" + (f" shuffled={v153['next_tok_shuffled']:.3f}" if 'next_tok_shuffled' in v153 else ''))
    v5.v103('\n'.v217(v97) + '\n', encoding='utf-8')
    v143(f'[185] {v154} gain={v95:+.3f} drop={v96:+.3f}')
    return 0
if v98 == '__main__':
    raise v156(v200())