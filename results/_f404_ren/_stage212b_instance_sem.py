"""
Stage 212b — instance disambiguation, retried with the 258 channel.

212 closed THESIS_NO: a read-only 2-layer MLP over frozen P1 state could not tell WHICH
occurrence of a surface form we are in, so it could not resolve collisions a surface-keyed
memory is blind to (`g1_collision` false). That was one scale, one architecture, no control —
the same shape of evidence 210-212 were criticised for.

Since then 258 built the mechanism 212 was missing: instead of a standalone head scored on its
own contrastive objective, project the trunk state INTO KEY SPACE and blend it with the fp
query, trained by InfoNCE against the actual bank:

    q = normalize( (1 - a) * W_q(fp query) + a * W_sem(h_t) ),   a = sigmoid(MLP([h_t, fp conf]))

Collisions are the case where fp is blind BY CONSTRUCTION: one surface form, four occurrences,
four different values, so every candidate key carries the identical fingerprint. Chance 0.25.
Anything above it has to come from context, which is exactly what the semantic channel carries.

Store from one half of an occurrence window, query from the DISJOINT other half — no lexical
overlap shortcut inside an occurrence, same rule 212 used.

  fp_only     the 256 path — pinned at chance by construction
  fp + sem    the 258 channel
  gpt2 + sem  matched control, so a negative separates scale from architecture
  shuffled    keys permuted — causal floor

  python _stage212b_instance_sem.py [--smoke] [--no-gpt-control]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import hidden_and_logits
v0 = v13('results')
v1 = v0 / 'stage212b_decision.json'
v2 = v0 / 'stage212b_mini.md'
v3 = v0 / '_stage212b_log.txt'
v4 = v13('checkpoints/stage191_p1_curve.pt')
v5 = v13('checkpoints/stage253_joint_l02.pt')
v6 = v13('data/_wikitext103_train.txt')
v7 = v67.v14('[A-Za-z][a-z]{2,}')
v8 = 212
v9 = 4
v10 = 1.0 / v9

def log(v15: v68) -> None:
    v16 = v15 if v15.v137('\n') else v15 + '\n'
    try:
        v138(v16, end='', flush=True)
    except v69:
        v138(v16.v238('ascii', 'replace').v222('ascii'), end='', flush=True)
    v3.v139.v70(parents=True, exist_ok=True)
    with v3.v140('a', encoding='utf-8') as v71:
        v71.v141(v16)

class SemQuery(v17.v11):

    def __init__(v72, v73: v12, v27):
        v223().v142()
        v72.v74 = v17.v224(v73, 256).v101(v27)
        v72.v75 = v17.v225(v17.v224(v73 + 2, 64), v17.v237(), v17.v224(64, 1)).v101(v27)
        v17.v190.v143(v72.v75[-1].v144)
        v17.v190.v145(v72.v75[-1].v146, -2.0)

    def q(v72, v76):
        return v191.v147(v72.v74(v76), dim=-1)

    def a(v72, v76, v77):
        return v82.v226(v72.v75(v82.v246([v76, v77], dim=-1))).v148(-1)

def fp_conf(v18, v19):
    v20 = v18 @ v19.v149()
    v21 = v82.v150(v20, v192(2, v20.v151(-1)), dim=-1).v22
    if v21.v151(-1) < 2:
        return v82.v78([v21[..., 0], v21[..., 0]], dim=-1)
    return v82.v78([v21[..., 0], v21[..., 0] - v21[..., 1]], dim=-1)

@v82.v30()
def state(v23, v24, v25, v26, v27, v28):
    v29 = [v152 for v152 in v25.v238(v28).v29 if v152 != v26][-v193:]
    if not v29:
        return None
    v76, v79 = v80(v23, v24, v82.v120([v29], device=v27), v26)
    return v76[0, -1].v194().v81()

def collisions(v31, v32: v12, v33: v12=220):
    """Surface forms with >= N_SIB occurrences; each occurrence contributes two DISJOINT halves
    of its own window: one writes the key, the other asks."""
    v34 = v83(v84)
    for v35 in v31:
        for v15 in v195.v153(v35):
            v85 = v15.v196(1)
            if v166(v85) < 5:
                continue
            v197, v198 = (v215(0, v15.v247() - v33), v192(v166(v35), v15.v248() + v33))
            v154 = v35[v197:v198]
            if v166(v7.v239(v154)) < 12:
                continue
            v155 = v166(v154) // 2
            v34[v85].v199((v154[:v155], v154[v155:]))
    v36 = {v85: v156[:v9] for v85, v156 in v34.v110() if v166(v156) >= v9}
    return v86(v84(v36.v110())[:v32])

def main() -> v12:
    v37 = v157.v87()
    v37.v88('--smoke', action='store_true')
    v37.v88('--steps', type=v12, default=0)
    v37.v88('--tau', type=v81, default=0.05)
    v37.v88('--forms', type=v12, default=0)
    v37.v88('--no-gpt-control', action='store_true')
    v38 = v37.v89()
    v3.v90('', encoding='utf-8')
    v27 = v82.v27('cuda' if v82.v227.v200() else 'cpu')
    v39 = v158.v91(v8)
    v82.v92(v8)
    v40 = v93.v93()
    v41 = v38.v41 or (200 if v38.v94 else 800)
    v42 = v38.v49 or (24 if v38.v94 else 120)
    v43 = 4000 if v38.v94 else 30000
    v95(f'Stage212b instance-sem start {v244.v235(v245.v236).v187()} device={v27}')
    v79, v79, v96, v97 = v98()
    v25 = v159.v99(v68(v201.v160))
    v44 = v25.v100()
    v26 = v25.v161(v162) or 0
    v24 = v228.v202(v25, v96, v26, v44).v101(v27)
    v45 = v5 if v5.v163() else v4
    v23 = v203(v97, v44).v101(v27)
    v23.v102(v82.v204(v45, map_location=v27, weights_only=False)['model'])
    v23.v103()
    for v46 in v23.v104():
        v46.v164(False)
    v47 = v203(v97, v44).v101(v27)
    v47.v102(v82.v204(v4, map_location=v27, weights_only=False)['model'])
    v47.v103()
    for v46 in v47.v104():
        v46.v164(False)
    v48 = v105(v47, v96, v27)
    with v6.v140('r', encoding='utf-8', errors='ignore') as v71:
        v106 = v71.v165(4000000 if v38.v94 else 25000000)
    v31 = [v206.v205() for v206 in v106.v229('\n') if 200 <= v166(v206.v205()) <= 600][:v43]
    v49 = v107(v31, v42)
    v95(f'  surface forms with >={v9} occurrences: {v166(v49)}')
    if v166(v49) < 8:
        v95('  not enough colliding forms')
        return 1
    v108, v109, v110 = ([], [], [])
    for v85, v111 in v49.v110():
        v112 = v48.v207([v85])[0]
        v113 = []
        for v167, (v208, v209) in v168(v111):
            v169 = v48.v210(v208, exclude=v85)
            if v169 is None:
                continue
            v108.v199(v191.v147(v112 + v169, dim=-1))
            v113.v199(v166(v109))
            v109.v199({'form': v85, 'occ': v167, 'slot': v166(v109)})
        for v170, (v208, v209) in v171(v113, v111):
            v172 = v48.v210(v209, exclude=v85)
            v173 = v211(v23, v24, v25, v26, v27, v209)
            if v172 is None or v173 is None:
                continue
            v110.v199({'form': v85, 'slot': v170, 'sib': v113, 'raw': v191.v147(v112 + v172, dim=-1), 'h': v173, 'qtext': v209})
    if v166(v110) < 16:
        v95('  not enough usable occurrences')
        return 1
    v19 = v82.v78(v108, 0).v101(v27).v81()
    v39.v114(v110)
    v50 = v166(v110) // 2
    v115, v116 = (v110[:v50], v110[v50:])
    v95(f'  slots={v166(v109)} fit={v166(v115)} eval={v166(v116)} chance={v10:.3f}')
    v51 = v117(v12(v115[0]['h'].v212()), v27)
    v52 = v174.v118(v27)
    v53 = v82.v175.v119(v84(v51.v104()) + v84(v52.v104()), lr=0.002, weight_decay=0.01)
    v54 = v82.v78([v128['raw'] for v128 in v115]).v101(v27).v81()
    v55 = v82.v78([v128['h'] for v128 in v115]).v101(v27).v81()
    v56 = v82.v120([v128['slot'] for v128 in v115], device=v27)
    for v57 in v121(1, v41 + 1):
        v122 = v82.v176(0, v54.v151(0), (v192(32, v54.v151(0)),), device=v27)
        v18 = v191.v147(v52(v54[v122]), dim=-1)
        v123 = v51.v123(v55[v122], v240(v18, v19)).v177(-1)
        v124 = v191.v147((1 - v123) * v18 + v123 * v51.v124(v55[v122]), dim=-1)
        v125 = v191.v178(v124 @ v19.v149() / v38.v213, v56[v122])
        v53.v179(set_to_none=True)
        v125.v180()
        v82.v17.v214.v181(v84(v51.v104()) + v84(v52.v104()), 1.0)
        v53.v57()
        if v57 % v215(1, v41 // 5) == 0:
            v95(f'  step {v57}/{v41} loss={v81(v125):.3f} a={v81(v123.v230()):.3f}')
    v51.v103()

    @v82.v30()
    def score(v126, v127=v19):
        v182, v183 = ([], [])
        for v128 in v116:
            v18 = v191.v147(v52(v128['raw'].v177(0)), dim=-1)[0]
            if v126:
                v123 = v51.v123(v128['h'], v240(v18, v127))
                v124 = v191.v147((1 - v123) * v18 + v123 * v51.v124(v128['h']), dim=-1)
                v183.v199(v81(v123))
            else:
                v124 = v18
            v20 = v127 @ v124
            v184 = v215(v128['sib'], key=lambda v167: v81(v20[v167]))
            v182.v199(v12(v184 == v128['slot']))
        return {'collision_4way': v81(v241.v230(v182)), 'n': v166(v182), 'alpha': v81(v241.v230(v183)) if v183 else 0.0}
    v129, v130 = (v132(False), v132(True))
    v58 = v82.v131(v19.v151(0), generator=v82.v242().v92(v8 + 1))
    v59 = v132(True, Kmat=v19[v58.v101(v19.v27)])
    v95(f"fp_only={v129['collision_4way']:.3f} sem={v130['collision_4way']:.3f} shuffled={v59['collision_4way']:.3f} (chance {v10:.3f})")
    v60 = None
    if not v38.v133:
        try:
            v185 = v174.v216(v27)
            v182 = True
            for v128 in v110:
                v85 = v174.v231(v185, v25, v26, v27, [v152 for v152 in v25.v238(v128['qtext']).v29 if v152 != v26])
                if v85 is None:
                    v182 = False
                    break
                v128['h_gpt'] = v85.v194().v81()
            if v182:
                v217 = v117(v12(v110[0]['h_gpt'].v212()), v27)
                v218 = v174.v118(v27)
                v219 = v82.v175.v119(v84(v217.v104()) + v84(v218.v104()), lr=0.002)
                v220 = v82.v78([v128['h_gpt'] for v128 in v115]).v101(v27).v81()
                for v79 in v121(v41):
                    v122 = v82.v176(0, v54.v151(0), (v192(32, v54.v151(0)),), device=v27)
                    v232 = v191.v147(v218(v54[v122]), dim=-1)
                    v123 = v217.v123(v220[v122], v240(v232, v19)).v177(-1)
                    v124 = v191.v147((1 - v123) * v232 + v123 * v217.v124(v220[v122]), dim=-1)
                    v233 = v191.v178(v124 @ v19.v149() / v38.v213, v56[v122])
                    v219.v179(set_to_none=True)
                    v233.v180()
                    v219.v57()
                v217.v103()
                with v82.v30():
                    v234 = []
                    for v128 in v116:
                        v232 = v191.v147(v218(v128['raw'].v177(0)), dim=-1)[0]
                        v123 = v217.v123(v128['h_gpt'], v240(v232, v19))
                        v124 = v191.v147((1 - v123) * v232 + v123 * v217.v124(v128['h_gpt']), dim=-1)
                        v243 = v19 @ v124
                        v234.v199(v12(v215(v128['sib'], key=lambda v167: v81(v243[v167])) == v128['slot']))
                    v60 = {'collision_4way': v81(v241.v230(v234))}
                v95(f"gpt2+sem={v60['collision_4way']:.3f}")
        except v186 as e:
            v95(f'  gpt control unavailable: {v249(v85).v66}: {v85}')
    v61 = v129['collision_4way'] <= v10 + 0.1
    v62 = v130['collision_4way'] >= v10 + 0.2
    v63 = v130['collision_4way'] >= v129['collision_4way'] + 0.15
    v64 = v59['collision_4way'] <= v10 + 0.1
    v65 = v60 is not None and v60['collision_4way'] < v10 + 0.2
    if not v61:
        v134 = 'INSTANCE_SEM_INVALID'
    elif v62 and v63 and v64:
        v134 = 'INSTANCE_SEM_OK'
    elif v65:
        v134 = 'INSTANCE_SEM_NO_AT_SCALE'
    else:
        v134 = 'INSTANCE_SEM_NO'
    v36 = {'stage': '212b', 'overall': v134, 'trunk': v45.v135, 'chance': v10, 'steps': v41, 'slots': v166(v109), 'n_fit': v166(v115), 'n_eval': v166(v116), 'gates': {'G_fp_blind_by_construction': v61, 'G_sem_above_chance': v62, 'G_beats_fp_only': v63, 'G_tape_causal': v64}, 'summary': {'fp_only': v129, 'fp_plus_sem': v130, 'shuffled_keys': v59, 'gpt_control': v60}, 'note': 'Retry of 212 with the mechanism it lacked. 212 scored a standalone read-only MLP on its own contrastive objective; here the trunk state is projected INTO KEY SPACE and blended with the fp query, trained by InfoNCE against the real bank (the 258 channel). Collisions are where fp is blind by construction — one surface form, four occurrences, four values, identical fingerprint on every sibling key — so G_fp_blind must hold or the exam leaked. Store and query halves of an occurrence window are disjoint, as in 212. P1 and trunk frozen; only W_q, W_sem and the blend train.', 'timestamp': v244.v235(v245.v236).v187(), 'wall_s': v93.v93() - v40}
    v1.v90(v221.v188(v36, indent=2), encoding='utf-8')
    v2.v90(f"# Stage 212b instance channel, retried via W_sem\n\n**{v134}** chance={v10:.2f} slots={v166(v109)} eval={v166(v116)}\n\n- collision 4-way: fp-only **{v129['collision_4way']:.3f}** -> fp+sem **{v130['collision_4way']:.3f}** (shuffled {v59['collision_4way']:.3f})\n- blend a {v130['alpha']:.3f}\n" + (f"- matched GPT-2: {v60['collision_4way']:.3f}\n" if v60 else '- matched GPT-2: not run\n'), encoding='utf-8')
    v95(v221.v188({'overall': v134, 'gates': v36['gates']}, indent=2))
    return 0
if v66 == '__main__':
    raise v136(v189())