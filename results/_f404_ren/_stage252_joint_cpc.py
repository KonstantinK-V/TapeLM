"""
Stage 252 — Joint CE + lambda*CPC on the upper trunk (fix for 251).

251 read: CPC alone moved `fast` while `slow`/`head` kept the old representation
(head sees cat([fast, slow])), so holdout PPL blew up 72 -> 245. Fix = train both
objectives on the same steps so downstream re-aligns under the drift.

Arms: lambda in {0.0, 0.05, 0.2}; lambda=0 is the CAL control (pure CE) measured
in-run, so gates compare against it instead of a number from a previous stage.

Per arm:
  loss = CE(next token) + W_SELF*pred_loss + lambda * CPC(consequence prediction)
  CPC negatives: in-batch + one hard negative per anchor from the SAME document
  budget counted in CE tokens; probes on held-out docs; keep best-by-holdout-CE snapshot

Metrics: exam next_tok, holdout CE/PPL, 179 para/hard/gap, uniformity (collapse check),
slot recall, parametric leak.

  python _stage252_joint_cpc.py [--smoke] [--token-budget N] [--lambdas 0,0.05,0.2]
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
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
import _stage251_cpc_understand as s251
from _stage191_night import MICRO, PAD, SelfModelXL, W_SELF, load_data
from _stage194_fp_fact_memory import FpBank
v0 = v15('results')
v1 = v0 / 'stage252_decision.json'
v2 = v0 / 'stage252_mini.md'
v3 = v0 / '_stage252_log.txt'
v4 = v15('checkpoints/stage191_p1_curve.pt')
v5 = 252
v6 = 64
v7 = v16.v7
v8 = v16.v8
v9 = 0.0002
v10 = 8
v11 = 2
v12 = 0.15

def log(v17: v111) -> None:
    v18 = v17 if v17.v175('\n') else v17 + '\n'
    try:
        v176(v18, end='', flush=True)
    except v112:
        v176(v18.v272('ascii', 'replace').v258('ascii'), end='', flush=True)
    v3.v177.v113(parents=True, exist_ok=True)
    with v3.v178('a', encoding='utf-8') as v114:
        v114.v179(v18)

def cpc_draw_hard(v19, v20, v21, v22: v180.v115, v23: v14, v24: v14, v25: v14):
    """anchor xa, continuation xb, and a hard negative xn from the same document."""
    v26 = v181.v116((v24, v25), v23, v181.v117)
    v27 = v181.v116((v24, v25), v23, v181.v117)
    v28 = v181.v116((v24, v25), v23, v181.v117)
    v29 = [v56 for v56 in v21 if v20[v56 + 1] - v20[v56] >= 4 * v25]
    v30 = v29 or [v56 for v56 in v21 if v20[v56 + 1] - v20[v56] >= 2 * v25] or v21
    for v31 in v118(v24):
        v56 = v30[v22.v237(0, v265(v30) - 1)]
        v182, v183 = (v20[v56], v20[v56 + 1])
        v119 = v183 - v182
        v120 = v182 + v22.v237(0, v135(0, v119 - 2 * v25))
        v26[v31] = v19[v120:v120 + v25]
        v27[v31] = v19[v120 + v25:v120 + 2 * v25]
        v121 = False
        if v119 >= 4 * v25:
            for v184 in v118(6):
                v186 = v182 + v22.v237(0, v119 - v25)
                if v186 + v25 <= v120 or v186 >= v120 + 2 * v25:
                    v28[v31] = v19[v186:v186 + v25]
                    v121 = True
                    break
        if not v121:
            v185 = v30[v22.v237(0, v265(v30) - 1)]
            v238, v239 = (v20[v185], v20[v185 + 1])
            v186 = v238 + v22.v237(0, v135(0, v239 - v238 - v25))
            v28[v31] = v19[v186:v186 + v25]
    return (v122.v187(v26), v122.v187(v27), v122.v187(v28))

def make_hold_batches(v19, v20, v32, v23, v33: v14, v34: v14) -> v35[v122.v36]:
    """One fixed held-out window set reused by every probe and every arm."""
    v22 = v180.v115(v34)
    return [v16.v188(v19, v20, v189, v22, v23, v32) for v184 in v118(v33)]

@v122.v41()
def fixed_hold_ce(v37, v38: v35[v122.v36], v39, v23, v40) -> v13:
    v123, v42 = (0.0, 0)
    for v31 in v38:
        v43 = v31.v127(v40)
        v124 = v43 == v23
        v190, v184, v184 = v37.v191(v39[v43], v124, ids=v43)
        v125 = v43[:, 1:]
        v126 = ~v124[:, :-1] & ~v124[:, 1:]
        if v126.v170():
            v123 += v13(v192.v202(v190[:, :-1][v126], v125[v126]))
            v42 += 1
    return v123 / v135(1, v42)

@v122.v41()
def uniformity(v37, v19, v20, v39, v23, v40, v21, v42: v14, v34: v14) -> v13:
    """Mean pairwise cosine over random windows; high value = representation collapse."""
    v22 = v180.v115(v34)
    v43 = v16.v188(v19, v20, v42, v22, v23, v21).v127(v40)
    v44 = v192.v128(v16.v193(v37, v39, v43, v23), dim=-1)
    v45 = v44 @ v44.v129
    v46 = v45[~v122.v240(v42, dtype=v122.v130, device=v40)]
    return v13(v46.v194())

def train_joint(v47, v19, v20, v39, v23, v40, v48, v49: v13, v34: v14, v50: v111, v51, v52, v53, v54: v130=True, v55: v14=v10):
    v17 = v195.v131(v47)
    v196.v132(v17, 'upper')
    v56 = v17.v197.v133 // 2
    v57 = v16.v241(v56).v127(v40)
    v58 = [v73 for v73 in v17.v141() if v73.v198]
    v59 = v58 + (v35(v57.v141()) if v49 > 0 else [])
    v60 = v122.v199.v134(v59, lr=v9, weight_decay=0.01)
    v22 = v180.v115(v34)
    v61 = v180.v115(v34 + 7)
    v62 = v135(1, v48 // v55)
    v63 = v62
    v64 = 0
    v65 = 0
    v66 = 0
    v67 = []
    v68 = {'hold_ce': v13('inf'), 'sd': None, 'tokens': 0}
    v69 = 0
    v70 = None
    v71 = {'ce': 0.0, 'cpc': 0.0}
    while v64 < v48:
        v43 = v16.v188(v19, v20, v189, v22, v23, v51).v127(v40)
        v64 += v16.v200(v43, v23)
        v124 = v43 == v23
        v190, v184, v201 = v17.v191(v39[v43], v124, ids=v43)
        v125 = v43[:, 1:]
        v126 = ~v124[:, :-1] & ~v124[:, 1:]
        v83 = v192.v202(v190[:, :-1][v126], v125[v126])
        v136 = v83 + v242 * v201[~v124].v194()
        v137 = v13('nan')
        if v49 > 0:
            v26, v27, v28 = v243(v19, v20, v51, v61, v23, v6, v7)
            v65 += v16.v200(v26, v23) + v16.v200(v27, v23)
            v26, v27, v28 = (v26.v127(v40), v27.v127(v40), v28.v127(v40))
            v203 = v192.v128(v16.v193(v17, v39, v26, v23), dim=-1)
            v204 = v192.v128(v16.v193(v17, v39, v27, v23), dim=-1)
            v205 = v192.v128(v16.v193(v17, v39, v28, v23), dim=-1)
            v206 = v57(v203)
            v207 = v122.v244([v57(v204), v57(v205)], dim=0)
            v208 = v206 @ v207.v129 / v8
            v209 = v122.v245(v26.v259(0), device=v40)
            v210 = v192.v202(v208, v209)
            v136 = v136 + v49 * v210
            v137 = v13(v210)
        v60.v211(set_to_none=True)
        v136.v212()
        v122.v260.v246.v213(v59, 1.0)
        v60.v66()
        v66 += 1
        v71 = {'ce': v13(v83), 'cpc': v137}
        if v64 >= v63:
            v63 += v62
            v17.v138()
            v214 = v139(v17, v52, v39, v23, v40)
            v215 = v16.v217(v17, v39, v23, v53, v40)
            v67.v173({'tokens_ce': v64, 'hold_ce': v214, 'next_tok': v215, 'ce': v71['ce'], 'cpc': v71['cpc']})
            v150(f"  {v50} tok={v64}/{v48} ce={v71['ce']:.3f} cpc={v71['cpc']:.3f} hold={v214:.3f} nt={v215:.3f}")
            if v214 < v68['hold_ce']:
                v68 = {'hold_ce': v214, 'sd': {v247: v269.v276().v275().v273() for v247, v269 in v17.v274().v77()}, 'tokens': v64}
                v69 = 0
            else:
                v69 += 1
                if v54 and v69 >= v11 and (v214 > v68['hold_ce'] + v12):
                    v70 = f'holdout_ce_rose_at_{v64}'
                    v150(f"  {v50} early stop: hold {v214:.3f} > best {v68['hold_ce']:.3f} + {v12}")
                    v196.v132(v17, 'upper')
                    break
            v196.v132(v17, 'upper')
    v17.v138()
    v72 = v139(v17, v52, v39, v23, v40)
    if v68['sd'] is not None and v68['hold_ce'] < v72 - 1e-06:
        v17.v156({v247: v269.v127(v40) for v247, v269 in v68['sd'].v77()})
        v140 = True
    else:
        v140 = False
    v17.v138()
    for v73 in v17.v141():
        v73.v216(False)
    v74 = {'tokens_ce': v64, 'tokens_cpc': v65, 'steps': v66, 'curve': v67, 'early_stop': v70, 'restored_best': v140, 'best_hold_ce': v68['hold_ce'], 'best_at_tokens': v68['tokens'], 'final_hold_ce_before_restore': v72}
    return (v17, v74)

def evaluate(v17, v39, v23, v75, v76, v40, v19, v20, v32, v52, v77, v78, v79, v80, v81):
    v82 = v142(v17, v76, v40)
    v83 = v139(v17, v52, v39, v23, v40)
    return {'next_tok': v16.v217(v17, v39, v23, v77, v40), 'hold_ce': v83, 'hold_ppl': v248.v218(v249(v83, 20)), 'inversion': v16.v219(v17, v39, v23, v75, v40), 'uniformity': v220(v17, v19, v20, v39, v23, v40, v32, 48, v5 + 32), 'slot_mem': v229.v221(v78, v79, v82, v80, v81, v5), 'param_leak': v16.v222(v17, v39, v23, v75, v78, v79, v40, v5 + 33)}

def main() -> v14:
    v84 = v223.v143()
    v84.v144('--smoke', action='store_true')
    v84.v144('--token-budget', type=v14, default=0)
    v84.v144('--lambdas', type=v111, default='0,0.05,0.2')
    v85 = v84.v145()
    v3.v146('', encoding='utf-8')
    v40 = v122.v40('cuda' if v122.v261.v250() else 'cpu')
    v22 = v180.v115(v5)
    v122.v147(v5)
    v86 = v148.v148()
    v87 = v85.v48 or (120000 if v85.v149 else 4000000)
    v88 = [v13(v224) for v224 in v85.v262.v251(',') if v224.v263() != '']
    v89 = 8 if v85.v149 else 20
    v90 = 40 if v85.v149 else 120
    v91 = 24 if v85.v149 else 60
    v92 = 8 if v85.v149 else 24
    v150(f'Stage252 start {v270.v267(v271.v268).v234()} budget={v87} lambdas={v88}')
    v19, v20, v76, v151 = v152()
    v51, v32 = v16.v153(v20)
    v75 = v225.v154(v111(v252.v226))
    v93 = v75.v155()
    v23 = v75.v227(v228) or 0
    v39 = v264.v253(v75, v76, v23, v93).v127(v40)
    v94 = v254(v151, v93).v127(v40)
    v94.v156(v122.v255(v4, map_location=v40, weights_only=False)['model'])
    v94.v138()
    for v73 in v94.v141():
        v73.v216(False)
    v150(f'corpus docs train={v265(v51)} hold={v265(v32)}')
    v78, v79 = v16.v157(v22, v89, v85.v149)
    v95 = v142(v94, v76, v40)
    v80, v81 = v229.v158(v95, v78)
    v77 = v16.v159(v90)
    v53 = v77[:v91]
    v52 = v160(v19, v20, v32, v23, v92, v5 + 5)
    v150(f'fixed holdout set: {v92} batches x {v189} windows')
    v96 = v161(v94, v39, v23, v75, v76, v40, v19, v20, v32, v52, v77, v78, v79, v80, v81)
    v150(f"baseline nt={v96['next_tok']:.3f} hold_ce={v96['hold_ce']:.3f} gap={v96['inversion']['gap_hard_minus_para']:+.3f} unif={v96['uniformity']:.3f} mem={v96['slot_mem']:.3f} leak={v96['param_leak']:.3f}")
    v97 = {}
    for v49 in v88:
        v50 = f'lam{v49:g}'
        v150(f'arm {v50}: joint CE + {v49:g}*CPC')
        v17, v74 = v230(v94, v19, v20, v39, v23, v40, v87, v49, v5 + v14(v49 * 1000) + 1, v50, v51, v52, v53)
        v162 = v161(v17, v39, v23, v75, v76, v40, v19, v20, v32, v52, v77, v78, v79, v80, v81)
        v97[v50] = {'lambda': v49, **v74, **v162}
        v150(f"  {v50} DONE nt={v162['next_tok']:.3f} hold_ce={v162['hold_ce']:.3f} gap={v162['inversion']['gap_hard_minus_para']:+.3f} unif={v162['uniformity']:.3f} mem={v162['slot_mem']:.3f} leak={v162['param_leak']:.3f} ({v148.v148() - v86:.0f}s)")
        if v49 == 0.0:
            v15('checkpoints').v113(exist_ok=True)
            if not v85.v149:
                v122.v266({'model': v17.v274(), 'stage': 252, 'lambda': 0.0}, 'checkpoints/stage252_ce_upper.pt')
    v98 = v97.v163('lam0')
    v99 = v98['next_tok'] if v98 else v96['next_tok']
    v100 = v98['hold_ce'] if v98 else v96['hold_ce']
    v101 = v96['inversion']['gap_hard_minus_para']
    v102 = v96['uniformity']
    v103 = []
    for v50, v164 in v97.v77():
        if v164['lambda'] == 0.0:
            continue
        v165 = v164['next_tok'] >= v99 - 0.01 and v164['hold_ce'] <= v100 + 0.05
        v166 = v164['inversion']['gap_hard_minus_para'] <= v101 - 0.02 or v164['inversion']['inversion']
        v167 = v164['uniformity'] <= v102 + 0.1
        v168 = v164['slot_mem'] >= 0.75 and v164['param_leak'] <= 0.4
        v164['gates'] = {'G_language_kept': v165, 'G_meaning_gain': v166, 'G_no_collapse': v167, 'G_memory_clean': v168}
        v103.v173((v50, v165 and v166 and v167 and v168, v165, v166, v167))
    v104 = [v169 for v169, v231, *v184 in v103 if v231]
    v105 = v170((v232 for v184, v184, v232, v184, v184 in v103))
    v106 = v170((v233 for v184, v184, v184, v233, v184 in v103))
    v107 = v130(v98 and v98['next_tok'] >= v96['next_tok'] + 0.01)
    if v104:
        v171 = 'JOINT_CPC_OK'
    elif v105 and v106:
        v171 = 'JOINT_CPC_PARTIAL'
    elif v107:
        v171 = 'JOINT_CPC_NO_CE_ONLY_WINS'
    else:
        v171 = 'JOINT_CPC_NO'
    if v104:
        v172 = 'SCALE_JOINT_TOKENS'
    elif v105 and (not v106):
        v172 = 'TRY_PAWS_202_SUPERVISION'
    elif v106 and (not v105):
        v172 = 'LOWER_LAMBDA_OR_LR'
    else:
        v172 = 'SCALE_CE_ONLY_16M'
    v108 = {'stage': 252, 'overall': v171, 'fork_next': v172, 'token_budget_per_arm': v87, 'joint_lr': v9, 'cpc': {'batch': v6, 'L': v7, 'temp': v8, 'hard_negatives': 'same-document window'}, 'winners': v104, 'reference': {'control_arm': 'lam0', 'ref_next_tok': v99, 'ref_hold_ce': v100}, 'baseline': v96, 'arms': v97, 'timestamp': v270.v267(v271.v268).v234(), 'wall_s': v148.v148() - v86, 'note': '251 fix: CPC alone drifted `fast` while slow/head stayed; joint CE re-aligns downstream. lambda=0 arm is the in-run CAL control; gates compare against it.'}
    v1.v146(v256.v235(v108, indent=2), encoding='utf-8')
    v109 = ['# Stage 252 joint CE + CPC', '', f'**{v171}** fork={v172} budget={v87} tok/arm', '']
    v109.v173(f"- baseline: nt={v96['next_tok']:.3f} hold_ce={v96['hold_ce']:.3f} gap={v101:+.3f} unif={v102:.3f}")
    for v50, v164 in v97.v77():
        v109.v173(f"- {v50} (λ={v164['lambda']:g}): nt={v164['next_tok']:.3f} hold_ce={v164['hold_ce']:.3f} gap={v164['inversion']['gap_hard_minus_para']:+.3f} unif={v164['uniformity']:.3f} mem={v164['slot_mem']:.3f} leak={v164['param_leak']:.3f}" + (f" [early stop: {v164['early_stop']}]" if v164.v163('early_stop') else ''))
    v2.v146('\n'.v257(v109) + '\n', encoding='utf-8')
    v150(v256.v235({'overall': v171, 'fork': v172, 'winners': v104}, indent=2))
    return 0
if v110 == '__main__':
    raise v174(v236())