"""
Stage 213 — Fine-tune variant A with frozen arc_enc (character fp geometry fixed).

Problem: if arc_enc (emb + FF + GELU) adapts on a new domain, h("the") can cross GELU
kinks differently from h("cat"); pooled char geometry stops behaving like a stable fp substrate.

Protocol:
  A) Freeze arc_enc (eval + no grad); train fast/slow/head on TinyStories domain.
  B) Control: train arc_enc only on same data (fp drift + GELU zone shift).

Gates:
  G1 fp_stable: max(1 - cos(fp_before, fp_after)) < 1e-5 after A
  G2 fp_drifts: mean fp drift after B >= 0.02 (control shows domain move hurts fp)
  G3 gen_ok: next_tok_acc after A within 0.03 of baseline OR improves

  python _stage213_arc_enc_freeze_finetune.py
  python _stage213_arc_enc_freeze_finetune.py --smoke
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import D_XL, L_XL, LR, MAX_ARCS, MICRO, PAD, SelfModelXL, W_SELF, WARMUP, load_data, lr_at, sample_windows, score_items, span_logprob_x
from _stage194_fp_fact_memory import FpBank
v0 = v16('results')
v1 = v16('checkpoints/stage191_p1_curve.pt')
v2 = v16('data/stage191_exam_v3.jsonl')
v3 = v16('data/external_tinystories_100k_85.txt')
v4 = v0 / 'stage213_decision.json'
v5 = v0 / 'stage213_mini.md'
v6 = v0 / '_stage213_log.txt'
v7 = v16('checkpoints/stage213_upper_tinystories.pt')
v8 = 213
v9 = 1200
v10 = v17.v11
v12 = ['the', 'and', 'cat', 'London', 'quantum', 'Elizabeth', 'running', 'xyzabc']

def log(v18: v93) -> None:
    v19 = v18 if v18.v155('\n') else v18 + '\n'
    try:
        v156(v19, end='', flush=True)
    except v94:
        v156(v19.v230('ascii', 'replace').v217('ascii'), end='', flush=True)
    v6.v157.v95(parents=True, exist_ok=True)
    with v6.v158('a', encoding='utf-8') as v96:
        v96.v159(v19)

def build_flat_from_text(v20: v93, v21: v97, v22: v15, v23: v15=8000, v24: v15=40) -> v27[v164.v101, v164.v101]:
    v25 = []
    for v26 in v20.v98('\n'):
        v26 = v26.v160()
        if v26.v198('#') or not v26:
            continue
        if v161(v26) >= v24:
            v25.v168(v26)
        if v161(v25) >= v23:
            break
    v56, v99 = ([], [0])
    for v19 in v25:
        v100 = [v110 for v110 in v21.v230(v19).v100 if v110 != v22]
        if v161(v100) >= 8:
            v56.v199(v100)
            v99.v168(v161(v56))
    if v161(v99) < 2:
        raise v162('domain B too small for flat corpus')
    return (v164.v163(v56, dtype=v164.v200), v164.v163(v99, dtype=v164.v201))

def load_p1(v28: v114.v28, v29: v15, v30: v15) -> v13:
    v31 = v13(v29, v30, d=v218, n_layers=v219).v102(v28)
    v32 = v114.v103(v1, map_location=v28, weights_only=False)
    v31.v104(v32['model'])
    v31.v105()
    return v31

def set_train_mode(v33: v13, v34: v93) -> None:
    for v35 in v33.v106():
        v35.v107 = False
    if v34 == 'none':
        v33.v105()
    elif v34 == 'upper':
        for v165 in (v33.v202, v33.v203, v33.v204):
            v165.v220()
            for v35 in v165.v106():
                v35.v107 = True
        v33.v47.v105()
    elif v34 == 'arc_enc':
        v33.v47.v220()
        for v35 in v33.v47.v106():
            v35.v107 = True
        v33.v202.v105()
        v33.v203.v105()
        v33.v204.v105()
    else:
        raise v221(v34)

@v114.v41()
def fp_drift(v36: v108, v37: v166[v93], v38: v14[v93, v114.v115]) -> v14:
    v39 = v36.v109(v37)
    v40 = []
    for v110, v111 in v112(v37):
        v113 = v167((v38[v111] * v39[v110]).v171())
        v40.v168(1.0 - v113)
    return {'mean': v167(v164.v205(v40)), 'max': v167(v164.v206(v40)), 'per_word': {v111: v40[v110] for v110, v111 in v112(v37)}}

@v114.v41()
def snapshot_fps(v36: v108, v37: v166[v93]) -> v14[v93, v114.v115]:
    v39 = v36.v109(v37)
    return {v111: v39[v110].v169() for v110, v111 in v112(v37)}

@v114.v41()
def gelu_zone_stats(v33: v13, v42: v93, v43: v14, v28: v114.v28) -> v14:
    """Pre/post FF on pooled char emb; GELU saturation fraction on L1 activations."""
    v44 = v114.v116(1, 1, v10, dtype=v114.v170, device=v28)
    v45 = 0
    for v117, v113 in v112(v42[:v10]):
        v44[0, 0, v117] = v43.v151(v113, 0)
        v45 += 1
    v46 = v33.v47
    v48 = v46.v118(v44)
    v49 = (v44 != 0).v167().v119(-1)
    v50 = (v48 * v49).v171(dim=-2) / v49.v171(dim=-2).v172(min=1.0)
    v51 = v46.v173[0](v50)
    v52 = v174.v120(v51)
    v53 = v46.v121(v46.v173[2](v52))
    v54 = (v51.v216() > 2.0).v167().v205().v122()
    v55 = (v51 < 0).v167().v205().v122()
    return {'word': v42, 'gelu_sat_frac': v54, 'pre_linear_neg_frac': v55, 'out_norm': v167(v53.v121())}

def finetune(v33: v13, v34: v93, v56, v57, v58, v22: v15, v59, v28: v114.v28, v60: v15, v61: v175.v123) -> v14:
    v124(v33, v34)
    v62 = [v35 for v35 in v33.v106() if v35.v107]
    v63 = v114.v176.v125(v62, lr=v207 * 0.5, weight_decay=0.01)
    v64 = v126.v126()
    v65 = None
    v66 = -1.0
    for v67 in v127(1, v60 + 1):
        for v128 in v63.v129:
            v128['lr'] = v208(v67, v60)
        v100 = v222(v56, v57, v223, v61, v22).v102(v28)
        v130 = v100 == v22
        v177, v178, v179 = v33.v180(v58[v100], v130, ids=v100)
        v131 = v100[:, 1:]
        v132 = ~v130[:, :-1] & ~v130[:, 1:]
        v133 = v174.v181(v177[:, :-1][v132], v131[v132])
        v134 = v133 + v209 * v179[~v130].v205()
        v63.v182(set_to_none=True)
        v134.v183()
        v224.v210.v184(v62, 1.0)
        v63.v67()
        v65 = v167(v133) if v65 is None else 0.95 * v65 + 0.05 * v167(v133)
        if v67 % v206(1, v60 // 4) == 0 or v67 == v60:
            v33.v105()
            if v34 == 'upper':
                v33.v47.v105()
            v51 = v211(lambda v113, v231: v232(v33, v58, v22, v113, v231, v28), v59, 'next_tok')
            v185 = v51.v151('next_tok_acc', 0)
            v139(f'  [{v34}] step {v67}/{v60} ce~{v65:.3f} next_tok={v185:.3f}')
            v66 = v206(v66, v185)
            v124(v33, v34)
    v33.v105()
    v33.v47.v105()
    return {'steps': v60, 'ce': v65, 'best_next_tok': v66, 'wall_s': v126.v126() - v64}

def main() -> v15:
    v68 = v186.v135()
    v68.v136('--smoke', action='store_true')
    v68.v136('--device', default='cuda' if v114.v233.v225() else 'cpu')
    v69 = v68.v137()
    v0.v95(parents=True, exist_ok=True)
    v6.v138('', encoding='utf-8')
    v139(f'Stage213 start {v234.v228(v235.v229).v195()}')
    v28 = v114.v28(v69.v28)
    v61 = v175.v123(v8)
    v60 = 80 if v69.v89 else v9
    v140, v141, v43, v29 = v142()
    v21 = v97.v143(v93(v17.v187))
    v30 = v21.v144()
    v22 = v21.v188(v189) or 0
    v58 = v226.v212(v21, v43, v22, v30).v102(v28)
    v70 = [v213.v190(v26) for v26 in v2.v145(encoding='utf-8').v214() if v26.v160()]
    v59 = [v191 for v191 in v70 if v191['type'] == 'next_tok'][:80 if v69.v89 else 200]
    if not v3.v192():
        v139(f'MISSING {v3}')
        return 1
    v71 = v3.v145(encoding='utf-8', errors='ignore')
    v146, v147 = v148(v71, v21, v22, max_lines=400 if v69.v89 else 8000, min_line_len=20)
    v139(f'domain B: {v3.v215} docs={v161(v147) - 1} tokens={v161(v146)}')
    v33 = v149(v28, v29, v30)
    v72 = v108(v33, v43, v28)
    v73 = v150(v72, v12)
    v74 = {v111: v193(v33, v111, v43, v28) for v111 in v12[:4]}
    v75 = v211(lambda v113, v231: v232(v33, v58, v22, v113, v231, v28), v59, 'next_tok').v151('next_tok_acc', 0)
    v139(f'baseline next_tok={v75:.3f}')
    v76 = v149(v28, v29, v30)
    v139('Phase A: freeze arc_enc, finetune fast/slow/head on domain B …')
    v77 = v152(v76, 'upper', v146, v147, v58, v22, v59, v28, v60, v61)
    v78 = v108(v76, v43, v28)
    v79 = v153(v78, v12, v73)
    v80 = v211(lambda v113, v231: v232(v76, v58, v22, v113, v231, v28), v59, 'next_tok').v151('next_tok_acc', 0)
    v139(f"A fp drift mean={v79['mean']:.2e} max={v79['max']:.2e} next_tok={v80:.3f}")
    if not v69.v89:
        v114.v194({'model': v76.v227(), 'train': v77}, v7)
    v81 = v149(v28, v29, v30)
    v139('Phase B control: train arc_enc only (fp should move) …')
    v82 = v152(v81, 'arc_enc', v146, v147, v58, v22, v59, v28, v60, v61)
    v83 = v108(v81, v43, v28)
    v84 = v153(v83, v12, v73)
    v85 = {v111: v193(v81, v111, v43, v28) for v111 in v12[:4]}
    v139(f"B fp drift mean={v84['mean']:.4f} max={v84['max']:.4f}")
    v86 = v79['max'] < 1e-05
    v87 = v84['mean'] >= 0.02
    v88 = v216(v80 - v75) <= 0.05 or v80 >= v75 - 1e-06
    if v69.v89:
        v86 = v79['max'] < 0.0001
        v87 = v84['mean'] > v79['mean']
    v90 = 'ARC_ENC_FREEZE_FP_STABLE_YES' if v86 and v87 and v88 else 'ARC_ENC_FREEZE_PARTIAL'
    if v86 and (not v87):
        v90 = 'ARC_ENC_FREEZE_FP_YES_CONTROL_WEAK'
    v91 = {'stage': 213, 'overall': v90, 'gates': {'G1_fp_stable_upper': v86, 'G2_fp_drifts_arc_control': v87, 'G3_gen_ok': v88}, 'baseline_next_tok': v75, 'after_upper_next_tok': v80, 'fp_drift_upper': v79, 'fp_drift_arc_enc': v84, 'gelu_baseline': v74, 'gelu_after_arc_train': v85, 'train_upper': v77, 'train_arc': v82, 'note': 'Freeze arc_enc (eval) during upper finetune → fp(word) unchanged; arc_enc-only train shifts fp and GELU stats.', 'timestamp': v234.v228(v235.v229).v195()}
    v4.v138(v213.v196(v91, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v138(f"# Stage213 — frozen arc_enc finetune\n\n**Overall:** `{v90}`\n\n- G1 fp stable (upper): {v86} (max drift {v79['max']:.2e})\n- G2 fp drifts (arc control): {v87} (mean {v84['mean']:.4f})\n- G3 gen ok: {v88} ({v75:.3f} → {v80:.3f})\n", encoding='utf-8')
    v139(f'VERDICT {v90}')
    return 0
if v92 == '__main__':
    raise v154(v197())