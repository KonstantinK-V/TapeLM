"""
Stage 171 — Curve dynamics HARDENING (contract).

Frozen pen + multi-step Δ + contrastive arcs.
Separates "we predict Δ" from "we painted an easy curve."

Plan/contract: results/plan_curve_dynamics.md , results/stage170_contract.json

  python _stage171_curve_harden.py
  python _stage171_curve_harden.py --steps 40000
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
import _stage170_curve_dynamics as s170
v0 = v100(v222).v95().v1
v2 = v0 / 'results'
v3 = v0 / 'checkpoints'
v4 = v2 / 'plan_curve_dynamics.md'
v5 = v2 / 'stage170_contract.json'
v6 = v2 / '_stage171_log.txt'
v7 = v2 / 'stage171_decision.json'
v8 = v2 / 'stage171_mini.md'
v9 = v3 / 'stage170_curve.pt'
v10 = v3 / 'stage171_curve.pt'
v11 = 171
v12 = v25.v12
v13 = v25.v13
v14 = 160
v15 = 24
v16 = 0.0003
v17 = 2000
v18 = 40000
v19 = (1, 2, 4, 8)
v20 = 0.07

def log(v26: v96) -> None:
    v27 = v26 if v26.v177('\n') else v26 + '\n'
    try:
        v178(v27, end='', flush=True)
    except v97:
        v178(v27.v192('ascii', 'replace').v239('ascii'), end='', flush=True)
    v6.v1.v98(parents=True, exist_ok=True)
    with v6.v179('a', encoding='utf-8') as v99:
        v99.v180(v27)

def write_json(v28: v100, v29: v22) -> None:
    v28.v1.v98(parents=True, exist_ok=True)
    v28.v101(v223.v181(v29, indent=2), encoding='utf-8')

class MultiStepDynamics(v30.v21):
    """From past arc, predict Δ over several horizons + arc embedding for contrastive."""

    def __init__(v102, v103: v24=v12, v104: v63[v24, ...]=v19):
        v240().v182()
        v102.v104 = v104
        v102.v105 = v30.v183(v30.v224(v103 * 2, v103 * 2), v30.v225(), v30.v224(v103 * 2, v103 * 2), v30.v225())
        v102.v106 = v30.v184({v96(v51): v30.v224(v103 * 2, v103) for v51 in v104})
        v102.v107 = v30.v184({v96(v51): v30.v224(v103 * 2, v103) for v51 in v104})
        v102.v108 = v30.v183(v30.v224(v103 * 2, v103), v30.v225(), v30.v224(v103, v103))

    def encode_ctx(v102, v47: v42.v23) -> v42.v23:
        v109 = v47[:, -1]
        v110 = v47.v110(dim=1)
        return v102.v105(v42.v226([v109, v110], dim=-1))

    def forward(v102, v47: v42.v23) -> v22[v96, v42.v23]:
        v111 = v102.v185(v47)
        v73 = {'arc': v195.v142(v102.v108(v111), dim=-1)}
        for v51 in v102.v104:
            v73[f'delta_{v51}'] = v102.v106[v96(v51)](v111)
            v73[f'z_{v51}'] = v102.v107[v96(v51)](v111)
        return v73

class HardenModel(v30.v21):

    def __init__(v102, v112: v24):
        v240().v182()
        v102.v113 = v25.v186(v112)
        v102.v114 = v187()

    def encode(v102, v43: v42.v23) -> v42.v23:
        return v102.v113(v43)

def freeze_pen(v31: v115) -> None:
    for v32 in v31.v113.v116():
        v32.v188(False)
    v31.v113.v117()

def load_pen_from_170(v31: v115, v28: v100, v33) -> v22:
    if not v28.v189():
        raise v190(f'need Stage170 ckpt at {v28}')
    v34 = v42.v118(v28, map_location='cpu', weights_only=False)
    v35 = v34['model']
    v36 = {v51[v230('pen.'):]: v119 for v51, v119 in v35.v227() if v51.v228('pen.')}
    v120, v121 = v31.v113.v122(v36, strict=True)
    return {'step170': v34.v162('step'), 'stoi': v34.v162('stoi'), 'itos': v34.v162('itos'), 'missing': v219(v120) if v120 else [], 'unexpected': v219(v121) if v121 else []}

def ctx_windows(v37: v42.v23, v38: v24, v39: v24) -> v42.v23:
    """z [B,T,d] → windows ending at t0..t0+n_pred-1 → [B,n_pred,CTX,d]."""
    v40 = v42.v123(v38, v38 + v39, device=v37.v33)
    v41 = v40.v191(1) - v42.v123(v13 - 1, -1, -1, device=v37.v33).v191(0)
    v41 = v41.v124(min=0)
    return v37[:, v41]

def harden_loss(v31: v115, v43: v42.v23) -> v63[v42.v23, v22]:
    """Frozen pen → z; train dyn on multi-step Δ/z + contrastive future arcs."""
    with v42.v74():
        v37 = v31.v192(v43)
    v125, v126, v103 = v37.v44
    v45 = v127(v19)
    v38 = v13
    v46 = v126 - 1 - v45
    if v46 < v38:
        raise v193('seq too short for multi-step')
    v39 = v46 - v38 + 1
    v47 = v128(v37, v38, v39)
    v48 = v31.v114(v47.v144(v125 * v39, v13, v103))
    v49 = 0.0
    v50: v22 = {}
    for v51 in v19:
        v129 = v37[:, v38:v38 + v39]
        v130 = v37[:, v38 + v51:v38 + v39 + v51]
        v131 = v130 - v129
        v132 = v48[f'z_{v51}'].v143(v125, v39, v103)
        v133 = v48[f'delta_{v51}'].v143(v125, v39, v103)
        v134 = 1.0 - v195.v244(v132, v130, dim=-1).v110()
        v135 = 1.0 - v195.v244(v133, v131, dim=-1).v110()
        v136 = v195.v194(v133, v131)
        v137 = 1.0 / v51
        v49 = v49 + v137 * (v134 + v135 + 0.1 * v136)
        v50[f'cos_d_k{v51}'] = v146(v195.v244(v133, v131, dim=-1).v110().v197())
        v50[f'cos_z_k{v51}'] = v146(v195.v244(v132, v130, dim=-1).v110().v197())
    v52 = v138(8, v45)
    v53 = []
    for v54 in v139(v39):
        v140 = v38 + v54 + 1
        v53.v172(v37[:, v140:v140 + v52].v110(dim=1))
    v55 = v42.v141(v53, dim=1)
    v55 = v195.v142(v55, dim=-1)
    v56 = v48['arc'].v143(v125, v39, v103)
    v57 = v56.v144(v125 * v39, v103)
    v58 = v55.v144(v125 * v39, v103)
    v59 = v57 @ v58.v149() / v20
    v60 = v42.v123(v57.v196(0), device=v57.v33)
    v61 = v195.v145(v59, v60)
    v49 = v49 + 0.5 * v61
    v50['loss_contrast'] = v146(v61.v197())
    v50['loss'] = v146(v49.v197())
    v62 = (v37[:, 1:] - v37[:, :-1]).v229(2).v110()
    v50['energy'] = v146(v62.v197())
    return (v49, v50)

@v42.v74()
def eval_hold(v31: v115, v64: v198.v147, v33, v65: v24=80) -> v22:
    v31.v113.v117()
    v31.v114.v117()
    v66 = v199.v148(v11 + 11)
    v67 = v24(0.9 * v230(v64))
    v68 = {v51: [] for v51 in v19}
    for v69 in v139(40):
        v140 = v66.v200(0, v127(1, v67 - v14 - 2))
        v43 = v42.v201(v64[v140:v140 + v14][None].v231(v198.v232), device=v33)
        v37 = v31.v192(v43)
        for v51 in v19:
            v68[v51].v172((v37[:, v51:] - v37[:, :-v51]).v110(dim=(0, 1)))
    v70 = {v51: v42.v141(v119, 0).v110(0) for v51, v119 in v68.v227()}
    v71 = {v51: {'cos': [], 'base_mean': [], 'base_copy': [], 'base_zero': []} for v51 in v19}
    v72 = []
    for v69 in v139(v65):
        v140 = v67 + v66.v200(0, v127(1, v230(v64) - v67 - v14 - 2))
        v43 = v42.v201(v64[v140:v140 + v14][None].v231(v198.v232), device=v33)
        v37 = v31.v192(v43)
        v126 = v37.v196(1)
        v149 = v126 - 1 - v127(v19) - 1
        v149 = v127(v149, v13)
        v47 = v37[:, v149 + 1 - v13:v149 + 1]
        if v47.v196(1) < v13:
            v202 = v47[:, :1].v233(1, v13 - v47.v196(1), -1)
            v47 = v42.v226([v202, v47], dim=1)
        v48 = v31.v114(v47)
        for v51 in v19:
            if v149 + v51 >= v126:
                continue
            v203 = v37[:, v149 + v51] - v37[:, v149]
            v133 = v48[f'delta_{v51}']
            v204 = v37[:, v149] - v37[:, v149 - v138(v51, v149)]
            v71[v51]['cos'].v172(v146(v195.v244(v133, v203, dim=-1).v110()))
            v71[v51]['base_mean'].v172(v146(v195.v244(v70[v51].v191(0), v203, dim=-1).v110()))
            v71[v51]['base_copy'].v172(v146(v195.v244(v204, v203, dim=-1).v110()))
            v71[v51]['base_zero'].v172(0.0)
        v52 = 8
        v150 = v195.v142(v37[:, v149 + 1:v149 + 1 + v52].v110(dim=1), dim=-1)
        v151 = v195.v142(v37[:, v13:v13 + v52].v110(dim=1), dim=-1)
        v56 = v48['arc']
        v72.v172(v146((v195.v244(v56, v150) > v195.v244(v56, v151)).v146().v110()))

    def avg(v152):
        return v234(v152) / v127(v230(v152), 1)
    v73: v22 = {'contrast_pref': v205(v72)}
    for v51 in v19:
        v153 = v205(v71[v51]['cos'])
        v154 = v205(v71[v51]['base_mean'])
        v155 = v205(v71[v51]['base_copy'])
        v73[f'k{v51}'] = {'cos_delta': v153, 'base_mean': v154, 'base_copy': v155, 'lift_mean': v153 - v154, 'lift_copy': v153 - v155}
    v73['min_lift_mean'] = v138((v73[f'k{v51}']['lift_mean'] for v51 in v19))
    v73['min_lift_copy'] = v138((v73[f'k{v51}']['lift_copy'] for v51 in v19))
    v31.v114.v156()
    return v73

def main() -> v24:
    v75 = v206.v157()
    v75.v158('--steps', type=v24, default=v18)
    v75.v158('--device', default='cuda' if v42.v245.v241() else 'cpu')
    v76 = v75.v159()
    v2.v98(parents=True, exist_ok=True)
    v3.v98(parents=True, exist_ok=True)
    v6.v101('', encoding='utf-8')
    v160(f'Stage171 start {v247.v242(v248.v243).v218()}')
    v160(f'plan={v4}')
    v160(f'contract={v5}')
    v160('HARDEN: frozen pen + multi-step Δ + contrastive arcs; NO text CE')
    if not v9.v189():
        v160(f'FATAL missing {v9}')
        return 1
    v77 = v25.v161()
    v78 = v42.v118(v9, map_location='cpu', weights_only=False)
    v79 = v78.v162('stoi')
    v80 = v78.v162('itos')
    if not v79 or not v80:
        v79, v80 = v25.v207(v77)
    v81 = 0
    v64 = v198.v163((v79.v162(v235, v81) for v235 in v77), dtype=v198.v208, count=v230(v77))
    v160(f'corpus chars={v230(v64)} vocab={v230(v80)} pen_from={v9.v236}')
    v33 = v42.v33(v76.v33)
    v42.v164(v11)
    v199.v165(v11)
    v198.v199.v165(v11)
    v31 = v115(v230(v80)).v166(v33)
    v82 = v167(v31, v9, v33)
    v168(v31)
    v160(f"loaded pen from 170 step={v82.v162('step170')}; pen FROZEN")
    v83 = v42.v209.v169(v31.v114.v116(), lr=v16, weight_decay=0.0001)
    v66 = v199.v148(v11)
    v38 = v170.v170()
    v84 = None
    v85 = []
    v86 = v171(v31, v64, v33)
    v160(f"  step 0: min_lift_mean={v86['min_lift_mean']:+.3f} min_lift_copy={v86['min_lift_copy']:+.3f} contrast_pref={v86['contrast_pref']:.3f} k1_cos={v86['k1']['cos_delta']:.3f} k8_cos={v86['k8']['cos_delta']:.3f}")
    v85.v172({'step': 0, **v86})
    v31.v114.v156()
    for v87 in v139(1, v76.v174 + 1):
        v43 = v25.v210(v64, v15, v14, v66, v33)
        v49, v211 = v212(v31, v43)
        v83.v213(set_to_none=True)
        v49.v214()
        v30.v237.v215(v31.v114.v116(), 1.0)
        v83.v87()
        v84 = v211['loss'] if v84 is None else 0.95 * v84 + 0.05 * v211['loss']
        if v87 % v17 == 0 or v87 == v76.v174:
            v216 = v171(v31, v64, v33)
            v217 = {'step': v87, **v216, 'loss_ema': v84, 'energy': v211.v162('energy')}
            v85.v172(v217)
            v160(f"  step {v87}: loss~{v84:.3f} min_lift_mean={v216['min_lift_mean']:+.3f} min_lift_copy={v216['min_lift_copy']:+.3f} contrast={v216['contrast_pref']:.3f} k1={v216['k1']['cos_delta']:.3f} k4={v216['k4']['cos_delta']:.3f} k8={v216['k8']['cos_delta']:.3f} energy={v211.v162('energy', 0):.4f}")
            v42.v238({'model': v31.v246(), 'stoi': v79, 'itos': v80, 'step': v87, 'curve': v85, 'pen_frozen': True, 'horizons': v219(v19), 'from_170': v96(v9)}, v10)
    v88 = (v170.v170() - v38) / 3600
    v89 = v85[-1]
    v90 = v89['min_lift_mean'] > 0.02
    v91 = v89['min_lift_copy'] > 0.02
    v72 = v89['contrast_pref'] > 0.55
    v92 = v89['k8']['lift_mean'] > 0.0 and v89['k4']['lift_mean'] > 0.02
    if v90 and v91 and v72 and v92:
        v173 = 'CURVE_DYN_HARDEN_YES'
    elif v90 and (v72 or v92):
        v173 = 'CURVE_DYN_HARDEN_MIXED'
    else:
        v173 = 'CURVE_DYN_HARDEN_NULL'
    v73 = {'timestamp': v247.v242(v248.v243).v218(), 'protocol': 'curve_dynamics_harden_171', 'plan': v96(v4), 'contract': v96(v5), 'verdict': v173, 'wall_hours': v88, 'steps': v76.v174, 'pen_frozen': True, 'horizons': v219(v19), 'from_170': v96(v9), 'final': v89, 'curve': v85, 'gates': {'beat_mean': v90, 'beat_copy': v91, 'contrast_ok': v72, 'far_ok': v92}, 'note': 'Frozen pen from 170; dynamics-only train; multi-step + contrastive. No text CE.', 'next': 'If YES: scale data / stronger dyn. If MIXED: tighten. If NULL: redesign — do not add CE.'}
    v175(v7, v73)
    v93 = [f'`{v173}` wall={v88:.2f}h steps={v76.v174} pen=FROZEN from 170', f"min_lift_mean={v89['min_lift_mean']:+.3f} min_lift_copy={v89['min_lift_copy']:+.3f} contrast={v89['contrast_pref']:.3f}", f"k1 cos={v89['k1']['cos_delta']:.3f} lift_m={v89['k1']['lift_mean']:+.3f}", f"k4 cos={v89['k4']['cos_delta']:.3f} lift_m={v89['k4']['lift_mean']:+.3f}", f"k8 cos={v89['k8']['cos_delta']:.3f} lift_m={v89['k8']['lift_mean']:+.3f}", f"gates={v73['gates']}"]
    v8.v101('\n'.v220(['# Stage171 — curve harden (frozen pen)', '', f'**Verdict:** `{v173}`', ''] + [f'- {v249}' for v249 in v93] + ['']), encoding='utf-8')
    v160(f'[171] {v173}')
    return 0
if v94 == '__main__':
    raise v176(v221())