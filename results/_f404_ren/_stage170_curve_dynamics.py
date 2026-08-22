"""
Stage 170 — Curve dynamics smoke.

Contract: text only draws a latent curve; train on curve changes (no char/word CE).
Plan: results/plan_curve_dynamics.md

  python _stage170_curve_dynamics.py
  python _stage170_curve_dynamics.py --steps 30000
"""
from __future__ import annotations
import argparse
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
v0 = v96(v198).v92().v1
v2 = v0 / 'results'
v3 = v0 / 'data'
v4 = v0 / 'checkpoints'
v5 = v3 / '_wikitext103_train.txt'
v6 = v2 / 'plan_curve_dynamics.md'
v7 = v2 / '_stage170_log.txt'
v8 = v2 / 'stage170_decision.json'
v9 = v2 / 'stage170_mini.md'
v10 = v4 / 'stage170_curve.pt'
v11 = 170
v12 = 96
v13 = 1
v14 = 96
v15 = 128
v16 = 32
v17 = 0.0003
v18 = 2000
v19 = 30000

def log(v25: v20) -> None:
    v26 = v25 if v25.v152('\n') else v25 + '\n'
    try:
        v153(v26, end='', flush=True)
    except v93:
        v153(v26.v124('ascii', 'replace').v213('ascii'), end='', flush=True)
    v7.v1.v94(parents=True, exist_ok=True)
    with v7.v154('a', encoding='utf-8') as v95:
        v95.v155(v26)

def write_json(v27: v96, v28: v23) -> None:
    v27.v1.v94(parents=True, exist_ok=True)
    v27.v97(v199.v156(v28, indent=2), encoding='utf-8')

def build_charset(v29: v20) -> v33[v23[v20, v24], v158[v20]]:
    v30 = v98(v157(v29))
    v31 = {v99: v100 for v100, v99 in v200(v30)}
    v32 = v30
    return (v31, v32)

def load_corpus(v34: v24=20000000) -> v20:
    if not v5.v159():
        raise v160(f'missing {v5} — need local wiki text for pen')
    v35 = v5.v101(encoding='utf-8', errors='ignore')
    if v119(v35) > v34:
        v35 = v35[:v34]
    return v35

class CurvePen(v36.v21):
    """Char → latent curve z_t (the 'pen'). No char CE anywhere."""

    def __init__(v102, v103: v24, v104: v24=v12, v105: v24=v13):
        v214().v161()
        v102.v106 = v36.v162(v103, v104)
        v102.v107 = v36.v163(v104, v104, num_layers=v105, batch_first=True)
        v102.v108 = v36.v164(v104)

    def forward(v102, v48: v46.v22) -> v46.v22:
        v109 = v102.v106(v48)
        v49, v45 = v102.v107(v109)
        return v102.v108(v49)

class DynamicsHead(v36.v21):
    """From past curve window, predict next latent and next delta."""

    def __init__(v102, v104: v24=v12, v110: v24=v14):
        v214().v161()
        v102.v110 = v110
        v102.v95 = v36.v165(v36.v201(v104 * 2, v104 * 2), v36.v202(), v36.v201(v104 * 2, v104 * 2))

    def forward(v102, v55: v46.v22) -> v33[v46.v22, v46.v22]:
        v111 = v55[:, -1]
        v112 = v55.v112(dim=1)
        v109 = v102.v95(v46.v203([v111, v112], dim=-1))
        v104 = v111.v166(-1)
        v56 = v109[:, :v104]
        v57 = v109[:, v104:]
        return (v56, v57)

class CurveModel(v36.v21):

    def __init__(v102, v103: v24):
        v214().v161()
        v102.v113 = v167(v103)
        v102.v114 = v168()

    def encode(v102, v48: v46.v22) -> v46.v22:
        return v102.v113(v48)

    def predict_from_prefix(v102, v49: v46.v22, v115: v24) -> v33[v46.v22, v46.v22]:
        v116 = v169(0, v115 + 1 - v14)
        v55 = v49[:, v116:v115 + 1]
        if v55.v166(1) < v14:
            v170 = v55[:, :1].v204(-1, v14 - v55.v166(1), -1)
            v55 = v46.v203([v170, v55], dim=1)
        return v102.v114(v55)

def sample_char_batch(v37: v171.v117, v38: v24, v39: v24, v40: v172.v118, v41) -> v46.v22:
    v42 = v119(v37)
    v43 = v42 - v39 - 1
    v44 = []
    for v45 in v120(v38):
        v121 = v40.v173(0, v43)
        v44.v147(v37[v121:v121 + v39].v205(v171.v206))
    return v46.v122(v171.v174(v44, 0), dtype=v46.v175, device=v41)

def dynamics_loss(v47: v123, v48: v46.v22) -> v33[v46.v22, v23]:
    """
    Teacher path: pen encodes full window (detach optional on early pen — here joint).
    Loss only on latent next-z / delta — never char logits.
    """
    v49 = v47.v124(v48)
    v125, v126, v104 = v49.v50
    v51 = v14
    if v126 <= v51 + 2:
        raise v176('seq too short for ctx')
    v52 = v126 - 1 - v51
    v53 = v46.v127(v51, v126 - 1, device=v49.v41)
    v54 = v53.v177(1) - v46.v127(v14 - 1, -1, -1, device=v49.v41).v177(0)
    v54 = v54.v128(min=0)
    v55 = v49[:, v54]
    v56, v57 = v47.v114(v55.v178(v125 * v52, v14, v104))
    v56 = v56.v129(v125, v52, v104)
    v57 = v57.v129(v125, v52, v104)
    v58 = v49[:, v51 + 1:v126]
    v59 = v49[:, v51:v126 - 1]
    v60 = v58 - v59
    v61 = 1.0 - v179.v215(v56, v58.v180(), dim=-1).v112()
    v62 = 1.0 - v179.v215(v57, v60.v180(), dim=-1).v112()
    v63 = v179.v130(v57, v60.v180())
    v64 = v60.v207(2).v112()
    v65 = v61 + v62 + 0.1 * v63 + 0.01 * v179.v208(0.05 - v64)
    v66 = {'loss': v181(v65.v180()), 'loss_z': v181(v61.v180()), 'loss_d': v181(v62.v180()), 'cos_z': v181(v179.v215(v56, v58.v180(), dim=-1).v112().v180()), 'cos_d': v181(v179.v215(v57, v60.v180(), dim=-1).v112().v180()), 'energy': v181(v64.v180())}
    return (v65, v66)

@v46.v78()
def eval_hold(v47: v123, v37: v171.v117, v41, v67: v24=64) -> v23:
    v47.v131()
    v40 = v172.v118(v11 + 7)
    v68 = v24(0.9 * v119(v37))
    v69 = []
    v70 = []
    v71 = []
    v72 = []
    v73 = []
    v74 = []
    v75 = []
    for v45 in v120(32):
        v121 = v40.v173(0, v169(1, v68 - v15 - 2))
        v48 = v46.v122(v37[v121:v121 + v15][None].v205(v171.v206), device=v41)
        v49 = v47.v124(v48)
        v75.v147((v49[:, 1:] - v49[:, :-1]).v112(dim=(0, 1)))
    v76 = v46.v174(v75, 0).v112(0)
    for v45 in v120(v67):
        v121 = v68 + v40.v173(0, v169(1, v119(v37) - v68 - v15 - 2))
        v48 = v46.v122(v37[v121:v121 + v15][None].v205(v171.v206), device=v41)
        v49 = v47.v124(v48)
        v125, v126, v104 = v49.v50
        v115 = v126 - 2
        v56, v57 = v47.v182(v49, v115)
        v58 = v49[:, v115 + 1]
        v59 = v49[:, v115]
        v60 = v58 - v59
        v132 = v49[:, v115] - v49[:, v115 - 1]
        v69.v147(v181(v179.v215(v56, v58, dim=-1).v112()))
        v70.v147(v181(v179.v215(v57, v60, dim=-1).v112()))
        v71.v147(v181(v179.v215(v46.v224(v60), v60, dim=-1).v112()))
        v72.v147(v181(v179.v215(v76.v177(0), v60, dim=-1).v112()))
        v73.v147(v181(v179.v215(v132, v60, dim=-1).v112()))

    def avg(v44):
        return v209(v44) / v169(v119(v44), 1)
    v77 = {'cos_z': v183(v69), 'cos_delta': v183(v70), 'base_zero_delta': v183(v71), 'base_mean_delta': v183(v72), 'base_copy_delta': v183(v73)}
    v77['lift_vs_mean_delta'] = v77['cos_delta'] - v77['base_mean_delta']
    v77['lift_vs_copy_delta'] = v77['cos_delta'] - v77['base_copy_delta']
    v47.v133()
    return v77

def main() -> v24:
    v79 = v184.v134()
    v79.v135('--steps', type=v24, default=v19)
    v79.v135('--device', default='cuda' if v46.v219.v216() else 'cpu')
    v80 = v79.v136()
    v2.v94(parents=True, exist_ok=True)
    v4.v94(parents=True, exist_ok=True)
    v7.v97('', encoding='utf-8')
    v137(f'Stage170 start {v221.v217(v222.v218).v195()}')
    v137(f'plan={v6}')
    v137('contract: text draws curve; loss=latent dynamics only; NO char/word CE')
    v29 = v138()
    v31, v32 = v139(v29)
    v37 = v171.v140((v31[v99] for v99 in v29), dtype=v171.v185, count=v119(v29))
    v137(f'corpus chars={v119(v37)} vocab={v119(v32)} file={v5.v210}')
    v41 = v46.v41(v80.v41)
    v46.v141(v11)
    v172.v142(v11)
    v171.v172.v142(v11)
    v47 = v123(v119(v32)).v143(v41)
    v81 = v46.v186.v144(v47.v187(), lr=v17, weight_decay=0.0001)
    v40 = v172.v118(v11)
    v51 = v145.v145()
    v82 = None
    v83 = []
    v84 = v146(v47, v37, v41)
    v137(f"  step 0: cos_z={v84['cos_z']:.3f} cos_d={v84['cos_delta']:.3f} lift_mean={v84['lift_vs_mean_delta']:+.3f} lift_copy={v84['lift_vs_copy_delta']:+.3f}")
    v83.v147({'step': 0, **v84})
    v47.v133()
    for v85 in v120(1, v80.v149 + 1):
        v48 = v188(v37, v16, v15, v40, v41)
        v65, v189 = v190(v47, v48)
        v81.v191(set_to_none=True)
        v65.v192()
        v36.v211.v193(v47.v187(), 1.0)
        v81.v85()
        v82 = v189['loss'] if v82 is None else 0.95 * v82 + 0.05 * v189['loss']
        if v85 % v18 == 0 or v85 == v80.v149:
            v194 = v146(v47, v37, v41)
            v83.v147({'step': v85, **v194, 'loss_ema': v82})
            v137(f"  step {v85}: loss~{v82:.3f} cos_z={v194['cos_z']:.3f} cos_d={v194['cos_delta']:.3f} lift_mean={v194['lift_vs_mean_delta']:+.3f} lift_copy={v194['lift_vs_copy_delta']:+.3f} base_mean={v194['base_mean_delta']:.3f} base_copy={v194['base_copy_delta']:.3f}")
            v46.v212({'model': v47.v220(), 'stoi': v31, 'itos': v32, 'step': v85, 'curve': v83}, v10)
    v86 = (v145.v145() - v51) / 3600
    v87 = v83[-1]
    v88 = v87['lift_vs_mean_delta'] > 0.02
    v89 = v87['lift_vs_copy_delta'] > 0.02
    if v88 and v89:
        v148 = 'CURVE_DYN_SMOKE_YES'
    elif v88 or v89:
        v148 = 'CURVE_DYN_SMOKE_MIXED'
    else:
        v148 = 'CURVE_DYN_SMOKE_NULL'
    v77 = {'timestamp': v221.v217(v222.v218).v195(), 'protocol': 'curve_dynamics_smoke', 'plan': v20(v6), 'verdict': v148, 'wall_hours': v86, 'steps': v80.v149, 'arch': {'d': v12, 'gru_layers': v13, 'ctx': v14, 'seq': v15, 'micro': v16}, 'corpus_chars': v24(v119(v37)), 'char_vocab': v119(v32), 'final': v87, 'curve': v83, 'note': 'No char/word CE. Gate = latent delta prediction vs mean/copy baselines.', 'next': 'If YES/MIXED: stronger pen / longer soak. If NULL: redesign dynamics loss, not revive 169 CE.', 'frozen_prior': 'stage169 word-CE FROZEN'}
    v150(v8, v77)
    v90 = [f'verdict `{v148}` wall={v86:.2f}h steps={v80.v149}', f"final cos_delta={v87['cos_delta']:.3f} lift_vs_mean={v87['lift_vs_mean_delta']:+.3f} lift_vs_copy={v87['lift_vs_copy_delta']:+.3f}", f"baselines mean={v87['base_mean_delta']:.3f} copy={v87['base_copy_delta']:.3f} zero={v87['base_zero_delta']:.3f}", 'loss = latent next-z + delta only (no text CE)', '169 frozen; do not resume word-battery path unless reopened']
    v9.v97('\n'.v196(['# Stage170 — curve dynamics smoke', '', f'**Verdict:** `{v148}`', ''] + [f'- {v223}' for v223 in v90] + ['']), encoding='utf-8')
    v137(f'[170] {v148}')
    return 0 if v148 != 'CURVE_DYN_SMOKE_NULL' else 0
if v91 == '__main__':
    raise v151(v197())