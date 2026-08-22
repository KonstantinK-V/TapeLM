"""
Stage 172 — Curve dynamics SCALE (contract).

After 171 HARDEN_YES:
  - more char stream
  - stronger dynamics (deeper + ctx attention)
  - longer horizons (incl. k=16)
  - weak decoder probe = readout only (stop-grad on z; NEVER teaches dynamics)

Pen stays FROZEN from 170. No text CE into pen/dyn.

  python _stage172_curve_scale.py
  python _stage172_curve_scale.py --steps 80000
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
import _stage170_curve_dynamics as s170
v0 = v108(v254).v103().v1
v2 = v0 / 'results'
v3 = v0 / 'checkpoints'
v4 = v2 / 'plan_curve_dynamics.md'
v5 = v2 / 'stage170_contract.json'
v6 = v2 / '_stage172_log.txt'
v7 = v2 / 'stage172_decision.json'
v8 = v2 / 'stage172_mini.md'
v9 = v3 / 'stage170_curve.pt'
v10 = v3 / 'stage171_curve.pt'
v11 = v3 / 'stage172_curve.pt'
v12 = 172
v13 = 128
v14 = 1
v15 = v30.v13
v16 = 128
v17 = 192
v18 = 16
v19 = 0.0002
v20 = 0.001
v21 = 4000
v22 = 80000
v23 = (1, 2, 4, 8, 16)
v24 = 0.07
v25 = 80000000

def log(v31: v104) -> None:
    v32 = v31 if v31.v200('\n') else v31 + '\n'
    try:
        v201(v32, end='', flush=True)
    except v105:
        v201(v32.v221('ascii', 'replace').v274('ascii'), end='', flush=True)
    v6.v1.v106(parents=True, exist_ok=True)
    with v6.v202('a', encoding='utf-8') as v107:
        v107.v203(v32)

def write_json(v33: v108, v34: v27) -> None:
    v33.v1.v106(parents=True, exist_ok=True)
    v33.v109(v255.v204(v34, indent=2), encoding='utf-8')

class CtxAttention(v35.v26):

    def __init__(v110, v111: v29, v112: v29=4):
        v275().v205()
        v110.v113 = v35.v206(v111, v112, batch_first=True, dropout=0.1)
        v110.v114 = v35.v207(v111)
        v110.v115 = v35.v208(v35.v211(v111, v111 * 2), v35.v256(), v35.v211(v111 * 2, v111))
        v110.v116 = v35.v207(v111)

    def forward(v110, v48: v47.v28) -> v47.v28:
        v117 = v48.v209(1)
        v118 = v47.v210(v47.v257(v117, v117, device=v48.v72, dtype=v47.v276), diagonal=1)
        v130, v77 = v110.v113(v48, v48, v48, attn_mask=v118)
        v48 = v110.v114(v48 + v130)
        return v110.v116(v48 + v110.v115(v48))

class StrongDynamics(v35.v26):

    def __init__(v110, v119: v29=v15, v111: v29=v13, v120: v70[v29, ...]=v23):
        v275().v205()
        v110.v120 = v120
        v110.v121 = v35.v211(v119, v111)
        v110.v62 = v35.v212(v47.v258(1, v16, v111))
        v35.v259.v213(v110.v62, std=0.02)
        v110.v122 = v35.v214([v277(v111, n_heads=4) for v77 in v167(2)])
        v110.v123 = v35.v208(v35.v211(v111 * 2, v111 * 2), v35.v256(), v35.v211(v111 * 2, v111 * 2))
        v110.v124 = v35.v215({v104(v56): v35.v211(v111 * 2, v119) for v56 in v120})
        v110.v125 = v35.v215({v104(v56): v35.v211(v111 * 2, v119) for v56 in v120})
        v110.v126 = v35.v208(v35.v211(v111 * 2, v111), v35.v256(), v35.v211(v111, v119))

    def forward(v110, v52: v47.v28) -> v27[v104, v47.v28]:
        v48 = v110.v121(v52) + v110.v62[:, :v52.v209(1)]
        for v127 in v110.v122:
            v48 = v127(v48)
        v128 = v48[:, -1]
        v129 = v48.v129(dim=1)
        v130 = v110.v123(v47.v260([v128, v129], dim=-1))
        v80 = {'arc': v226.v156(v110.v126(v130), dim=-1)}
        for v56 in v110.v120:
            v80[f'delta_{v56}'] = v110.v124[v104(v56)](v130)
            v80[f'z_{v56}'] = v110.v125[v104(v56)](v130)
        return v80

class WeakCharProbe(v35.v26):
    """Read-out only: z → next char. Trained with stop-grad on z."""

    def __init__(v110, v119: v29, v131: v29):
        v275().v205()
        v110.v115 = v35.v208(v35.v211(v119, v119), v35.v256(), v35.v211(v119, v131))

    def forward(v110, v41: v47.v28) -> v47.v28:
        return v110.v115(v41)

class ScaleModel(v35.v26):

    def __init__(v110, v131: v29):
        v275().v205()
        v110.v132 = v30.v216(v131, d=v15, n_layers=v14)
        v110.v133 = v217(d_in=v15, d=v13, horizons=v23)
        v110.v134 = v218(v15, v131)

    def encode(v110, v48: v47.v28) -> v47.v28:
        return v110.v132(v48)

def freeze_pen(v36: v135) -> None:
    for v37 in v36.v132.v136():
        v37.v219(False)
    v36.v132.v137()

def load_pen(v36: v135, v33: v108) -> v27:
    v38 = v47.v138(v33, map_location='cpu', weights_only=False)
    v39 = v38['model']
    v40 = {v56[v265('pen.'):]: v139 for v56, v139 in v39.v261() if v56.v262('pen.')}
    v36.v132.v140(v40, strict=True)
    return {'step': v38.v182('step'), 'stoi': v38.v182('stoi'), 'itos': v38.v182('itos')}

def ctx_windows(v41: v47.v28, v42: v29, v43: v29, v44: v29=v16) -> v47.v28:
    v45 = v47.v141(v42, v42 + v43, device=v41.v72)
    v46 = v45.v220(1) - v47.v141(v44 - 1, -1, -1, device=v41.v72).v220(0)
    v46 = v46.v142(min=0)
    return v41[:, v46]

def scale_loss(v36: v135, v48: v47.v28) -> v70[v47.v28, v47.v28, v27]:
    with v47.v81():
        v41 = v36.v221(v48)
    v143, v117, v111 = v41.v49
    v50 = v144(v23)
    v42 = v16
    v51 = v117 - 1 - v50
    if v51 < v42:
        raise v222('seq too short')
    v43 = v51 - v42 + 1
    v52 = v145(v41, v42, v43)
    v53 = v36.v133(v52.v158(v143 * v43, v16, v111))
    v54 = 0.0
    v55: v27 = {}
    for v56 in v23:
        v146 = v41[:, v42:v42 + v43]
        v147 = v41[:, v42 + v56:v42 + v43 + v56]
        v148 = v147 - v146
        v149 = v53[f'z_{v56}'].v157(v143, v43, v111)
        v150 = v53[f'delta_{v56}'].v157(v143, v43, v111)
        v151 = 1.0 - v226.v283(v149, v147, dim=-1).v129()
        v152 = 1.0 - v226.v283(v150, v148, dim=-1).v129()
        v153 = v226.v223(v150, v148)
        v154 = 1.0 / v278.v263(v56)
        v54 = v54 + v154 * (v151 + v152 + 0.1 * v153)
        v55[f'cos_d_k{v56}'] = v224(v226.v283(v150, v148, dim=-1).v129().v161())
    v57 = v155(8, v50)
    v58 = [v41[:, v42 + v225 + 1:v42 + v225 + 1 + v57].v129(dim=1) for v225 in v167(v43)]
    v59 = v226.v156(v47.v227(v58, dim=1), dim=-1)
    v60 = v53['arc'].v157(v143, v43, -1)
    if v60.v209(-1) != v59.v209(-1):
        pass
    v61 = v226.v156(v60, dim=-1).v158(v143 * v43, -1)
    v62 = v59.v158(v143 * v43, -1)
    if v61.v209(-1) != v62.v209(-1):
        v159 = v155(v61.v209(-1), v62.v209(-1))
        v61, v62 = (v61[:, :v159], v62[:, :v159])
    v63 = v61 @ v62.v170() / v24
    v64 = v47.v141(v61.v209(0), device=v61.v72)
    v65 = v226.v160(v63, v64)
    v54 = v54 + 0.5 * v65
    v66 = v41.v161()
    v67 = v36.v134(v66[:, :-1])
    v68 = v226.v160(v67.v158(-1, v67.v209(-1)), v48[:, 1:].v158(-1))
    with v47.v81():
        v162 = v67.v228(-1)
        v163 = v224((v162 == v48[:, 1:]).v224().v129())
    v69 = (v41[:, 1:] - v41[:, :-1]).v264(2).v129()
    v55.v164({'loss_dyn': v224(v54.v161()), 'loss_contrast': v224(v65.v161()), 'loss_probe': v224(v68.v161()), 'probe_acc': v163, 'energy': v224(v69.v161())})
    return (v54, v68, v55)

@v47.v81()
def eval_hold(v36: v135, v71: v229.v165, v72, v73: v29=64) -> v27:
    v36.v132.v137()
    v36.v133.v137()
    v36.v134.v137()
    v74 = v230.v166(v12 + 3)
    v75 = v29(0.9 * v265(v71))
    v76 = {v56: [] for v56 in v23}
    for v77 in v167(32):
        v168 = v74.v231(0, v144(1, v75 - v17 - 2))
        v48 = v47.v232(v71[v168:v168 + v17][None].v266(v229.v267), device=v72)
        v41 = v36.v221(v48)
        for v56 in v23:
            v76[v56].v195((v41[:, v56:] - v41[:, :-v56]).v129(dim=(0, 1)))
    v78 = {v56: v47.v227(v139, 0).v129(0) for v56, v139 in v76.v261()}
    v79 = {v56: {'cos': [], 'base_mean': [], 'base_copy': []} for v56 in v23}
    v98, v169 = ([], [])
    for v77 in v167(v73):
        v168 = v75 + v74.v231(0, v144(1, v265(v71) - v75 - v17 - 2))
        v48 = v47.v232(v71[v168:v168 + v17][None].v266(v229.v267), device=v72)
        v41 = v36.v221(v48)
        v117 = v41.v209(1)
        v170 = v144(v16, v117 - 1 - v144(v23) - 1)
        v52 = v41[:, v170 + 1 - v16:v170 + 1]
        if v52.v209(1) < v16:
            v233 = v52[:, :1].v268(1, v16 - v52.v209(1), -1)
            v52 = v47.v260([v233, v52], dim=1)
        v53 = v36.v133(v52)
        for v56 in v23:
            if v170 + v56 >= v117:
                continue
            v234 = v41[:, v170 + v56] - v41[:, v170]
            v150 = v53[f'delta_{v56}']
            v235 = v41[:, v170] - v41[:, v144(0, v170 - v56)]
            v79[v56]['cos'].v195(v224(v226.v283(v150, v234, dim=-1).v129()))
            v79[v56]['base_mean'].v195(v224(v226.v283(v78[v56].v220(0), v234, dim=-1).v129()))
            v79[v56]['base_copy'].v195(v224(v226.v283(v235, v234, dim=-1).v129()))
        v57 = 8
        v171 = v226.v156(v41[:, v170 + 1:v170 + 1 + v57].v129(dim=1), dim=-1)
        v172 = v226.v156(v41[:, v16:v16 + v57].v129(dim=1), dim=-1)
        v60 = v226.v156(v53['arc'][:, :v171.v209(-1)], dim=-1)
        if v60.v209(-1) != v171.v209(-1):
            v159 = v155(v60.v209(-1), v171.v209(-1))
            v60, v171, v172 = (v60[:, :v159], v171[:, :v159], v172[:, :v159])
        v98.v195(v224((v226.v283(v60, v171) > v226.v283(v60, v172)).v224().v129()))
        v63 = v36.v134(v41[:, :-1])
        v169.v195(v224((v63.v228(-1) == v48[:, 1:]).v224().v129()))

    def avg(v173):
        return v269(v173) / v144(v265(v173), 1)
    v80: v27 = {'contrast_pref': v236(v98), 'probe_acc': v236(v169)}
    for v56 in v23:
        v174 = v236(v79[v56]['cos'])
        v175 = v236(v79[v56]['base_mean'])
        v176 = v236(v79[v56]['base_copy'])
        v80[f'k{v56}'] = {'cos_delta': v174, 'base_mean': v175, 'base_copy': v176, 'lift_mean': v174 - v175, 'lift_copy': v174 - v176}
    v80['min_lift_mean'] = v155((v80[f'k{v56}']['lift_mean'] for v56 in v23))
    v80['min_lift_copy'] = v155((v80[f'k{v56}']['lift_copy'] for v56 in v23))
    v80['lift_mean_k1'] = v80['k1']['lift_mean']
    v80['lift_mean_k16'] = v80['k16']['lift_mean']
    v36.v133.v177()
    v36.v134.v177()
    return v80

def main() -> v29:
    v82 = v237.v178()
    v82.v179('--steps', type=v29, default=v22)
    v82.v179('--device', default='cuda' if v47.v284.v279() else 'cpu')
    v83 = v82.v180()
    v2.v106(parents=True, exist_ok=True)
    v3.v106(parents=True, exist_ok=True)
    v6.v109('', encoding='utf-8')
    v181(f'Stage172 start {v286.v281(v287.v282).v250()}')
    v181(f'plan={v4} contract={v5}')
    v181('SCALE: frozen pen + stronger dyn + more data + weak probe readout; probe CE does NOT teach dyn/pen')
    if not (v2 / 'stage171_decision.json').v238():
        v181('FATAL: need stage171_decision.json')
        return 1
    v84 = v255.v270((v2 / 'stage171_decision.json').v280(encoding='utf-8')).v182('verdict')
    if v84 != 'CURVE_DYN_HARDEN_YES':
        v181(f'WARN: 171 verdict was {v84}, continuing anyway per user request')
    if not v9.v238():
        v181(f'FATAL missing pen {v9}')
        return 1
    v85 = v30.v183(max_chars=v25)
    v86 = v47.v138(v9, map_location='cpu', weights_only=False)
    v184, v185 = (v86.v182('stoi'), v86.v182('itos'))
    if not v184 or not v185:
        v184, v185 = v30.v239(v85)
    v71 = v229.v186((v184.v182(v271, 0) for v271 in v85), dtype=v229.v240, count=v265(v85))
    v181(f'corpus chars={v265(v71)} vocab={v265(v185)} d_dyn={v13} ctx={v16} seq={v17} K={v251(v23)}')
    v72 = v47.v72(v83.v72)
    v47.v187(v12)
    v230.v188(v12)
    v229.v230.v188(v12)
    v36 = v135(v265(v185)).v189(v72)
    v87 = v190(v36, v9)
    v191(v36)
    v181(f"pen loaded from 170 step={v87.v182('step')}; FROZEN")
    v88 = v47.v241.v192(v36.v133.v136(), lr=v19, weight_decay=0.0001)
    v89 = v47.v241.v192(v36.v134.v136(), lr=v20, weight_decay=0.0)
    v74 = v230.v166(v12)
    v42 = v193.v193()
    v90 = None
    v91 = []
    v92 = v194(v36, v71, v72)
    v181(f"  step 0: min_lift_mean={v92['min_lift_mean']:+.3f} min_lift_copy={v92['min_lift_copy']:+.3f} contrast={v92['contrast_pref']:.3f} probe={100 * v92['probe_acc']:.1f}% k1={v92['k1']['cos_delta']:.3f} k16={v92['k16']['cos_delta']:.3f}")
    v91.v195({'step': 0, **v92})
    v36.v133.v177()
    v36.v134.v177()
    for v93 in v167(1, v83.v197 + 1):
        v48 = v30.v242(v71, v18, v17, v74, v72)
        v54, v68, v243 = v244(v36, v48)
        v88.v245(set_to_none=True)
        v54.v246()
        v35.v272.v247(v36.v133.v136(), 1.0)
        v88.v93()
        v89.v245(set_to_none=True)
        v68.v246()
        v89.v93()
        v90 = v243['loss_dyn'] if v90 is None else 0.95 * v90 + 0.05 * v243['loss_dyn']
        if v93 % v21 == 0 or v93 == v83.v197:
            v248 = v194(v36, v71, v72)
            v249 = {'step': v93, **v248, 'loss_ema': v90, 'energy': v243.v182('energy'), 'probe_train': v243.v182('probe_acc')}
            v91.v195(v249)
            v181(f"  step {v93}: loss_dyn~{v90:.3f} min_lift_mean={v248['min_lift_mean']:+.3f} min_lift_copy={v248['min_lift_copy']:+.3f} contrast={v248['contrast_pref']:.3f} probe={100 * v248['probe_acc']:.1f}% k1={v248['k1']['cos_delta']:.3f} k8={v248['k8']['cos_delta']:.3f} k16={v248['k16']['cos_delta']:.3f} energy={v243.v182('energy', 0):.3f}")
            v47.v273({'model': v36.v285(), 'stoi': v184, 'itos': v185, 'step': v93, 'curve': v91, 'pen_frozen': True, 'horizons': v251(v23), 'd_model': v13, 'from_pen': v104(v9)}, v11)
    v94 = (v193.v193() - v42) / 3600
    v95 = v91[-1]
    v96 = v95['min_lift_mean'] > 0.02
    v97 = v95['min_lift_copy'] > 0.02
    v98 = v95['contrast_pref'] > 0.55
    v99 = v95['k16']['lift_mean'] > 0.0 and v95['k8']['lift_mean'] > 0.02
    v100 = v95['probe_acc']
    if v96 and v97 and v98 and v99:
        v196 = 'CURVE_DYN_SCALE_YES'
    elif v96 and v98:
        v196 = 'CURVE_DYN_SCALE_MIXED'
    else:
        v196 = 'CURVE_DYN_SCALE_NULL'
    v80 = {'timestamp': v286.v281(v287.v282).v250(), 'protocol': 'curve_dynamics_scale_172', 'plan': v104(v4), 'contract': v104(v5), 'verdict': v196, 'wall_hours': v94, 'steps': v83.v197, 'pen_frozen': True, 'horizons': v251(v23), 'arch': {'pen_d': v15, 'dyn_d': v13, 'ctx': v16, 'seq': v17, 'attn_blocks': 2}, 'corpus_chars': v29(v265(v71)), 'final': v95, 'curve': v91, 'gates': {'beat_mean': v96, 'beat_copy': v97, 'contrast_ok': v98, 'far_ok': v99}, 'probe_acc_hold': v100, 'probe_role': 'readout_only_stopgrad_z_never_teaches_dyn_or_pen', 'note': 'Scale+stronger dyn on frozen 170 pen. Weak char probe is diagnostic only.', 'next': 'If YES: longer soak / domain transfer. Decoder stays weak. Do not revive 169 CE.'}
    v198(v7, v80)
    v101 = [f'`{v196}` wall={v94:.2f}h steps={v83.v197}', f'corpus={v265(v71)} pen=FROZEN dyn_d={v13} K={v251(v23)}', f"min_lift_mean={v95['min_lift_mean']:+.3f} min_lift_copy={v95['min_lift_copy']:+.3f} contrast={v95['contrast_pref']:.3f}", f"k1={v95['k1']['cos_delta']:.3f} k8={v95['k8']['cos_delta']:.3f} k16={v95['k16']['cos_delta']:.3f}", f'weak probe hold acc={100 * v100:.1f}% (readout only, not gate)', f"gates={v80['gates']}"]
    v8.v109('\n'.v252(['# Stage172 — curve scale', '', f'**Verdict:** `{v196}`', ''] + [f'- {v288}' for v288 in v101] + ['']), encoding='utf-8')
    v181(f'[172] {v196}')
    return 0
if v102 == '__main__':
    raise v199(v253())