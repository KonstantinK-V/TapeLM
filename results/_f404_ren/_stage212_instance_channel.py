"""
Stage 212 — instance / content-invariant channel on the frozen tape (final pre-publish stage).

Question: can a narrow read-only channel over the SAME frozen tape encode WHICH OCCURRENCE
of a surface form we are in (episode identity), and thereby resolve collisions that a
surface-keyed memory cannot? (old hop2/joint debt: dense H1 collisions, soft-disambig failed)

Setup (no CE touched, P1 frozen, channel read-only):
  h(crop)   = [fast_last ; slow_last] from frozen P1 over a text crop
  inst      = normalize(g(h)), g = 2-layer MLP trained CONTRASTIVELY:
                positive = the OTHER (disjoint) half of the same occurrence window
                hard neg = halves of OTHER occurrences of the SAME surface form

T1 collision (4-way, chance 0.25): one surface form S with 4 distinct occurrences, each
  carrying a distinct value label. Store key from first half, query from the DISJOINT second
  half. Candidates = the 4 sibling values → surface key alone is blind by construction.
  Baselines: fp_only (surface key), ctx_blend (197 M3 subject+ctx), soft_rerank (weak ctx),
             inst_random (untrained g — does learning matter?)
T2 para/hard invariance on 179 pairs (held out from corpus training).
T3/G5 next_tok unchanged + anti-CF assert.

  python _stage212_instance_channel.py
"""
from __future__ import annotations
import json
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
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, WORD_RE, FpBank
v0 = v23('results')
v1 = v23('checkpoints/stage191_p1_curve.pt')
v2 = v23('data/_wikitext103_train.txt')
v3 = v23('data/stage191_exam_v3.jsonl')
v4 = v0 / 'stage212_decision.json'
v5 = v0 / 'stage212_mini.md'
v6 = v0 / '_stage212_log.txt'
v7 = 212
v8 = 150000000
v9 = 70000000
v10 = 6000000
v11 = 4
v12 = 320
v13 = 220
v14 = 3000
v15 = 16
v16 = 0.001
v17 = 128
v18 = 0.07
v19 = 0.25

def log(v24: v96) -> None:
    v25 = v24 if v24.v172('\n') else v24 + '\n'
    try:
        v173(v25, end='', flush=True)
    except v97:
        v173(v25.v280('ascii', 'replace').v254('ascii'), end='', flush=True)
    v6.v174.v98(parents=True, exist_ok=True)
    with v6.v175('a', encoding='utf-8') as v99:
        v99.v176(v25)

class InstanceHead(v26.v20):
    """Narrow read-only channel: tape state -> instance code."""

    def __init__(v100, v101: v22, v102: v22=v17):
        v255().v177()
        v100.v103 = v26.v178(v26.v223(v101, v101 // 2), v26.v224(), v26.v223(v101 // 2, v102))

    def forward(v100, v104: v37.v21) -> v37.v21:
        return v225.v179(v100.v103(v104), dim=-1)

@v37.v36()
def tape_states(v27, v28, v29, v30, v31: v116[v96], v32, v33: v22=32) -> v37.v21:
    """[N, 2d] = [fast_last ; slow_last] over each text crop (frozen P1)."""
    v34 = []
    for v35 in v105(0, v180(v31), v33):
        v106 = v31[v35:v35 + v33]
        v181, v182 = ([], [])
        for v107 in v106:
            v183 = [v256 for v256 in v29.v280(v107).v183 if v256 != v30][:v257]
            if not v183:
                v183 = [v30]
            v182.v189(v180(v183))
            v181.v189(v183 + [v30] * (v257 - v180(v183)))
        v108 = v37.v184(v181, dtype=v37.v226, device=v32)
        v109 = v108 == v30
        v110 = v27.v185(v28[v108], v108)
        v111 = v27.v111(v110, pad_mask=v109)
        v186, v187, v187 = v27.v186(v110, v109)
        v71 = v37.v184([v204(0, v258 - 1) for v258 in v182], device=v32)
        v112 = v37.v188(v180(v106), device=v32)
        v34.v189(v37.v113([v111[v112, v71], v186[v112, v71]], dim=-1))
    return v37.v113(v34, 0)

def collect_occurrences(v38: v116[v96], v39: v190.v114):
    """surface -> list of OCC_PER_SURFACE windows, each split into two disjoint halves."""
    v40 = v115(v116)
    for v41 in v38:
        for v117 in v227.v191(v41):
            v118 = v117.v228(1)
            if v180(v118) < 4:
                continue
            v192 = v204(0, v117.v274() - v13 // 2)
            v193 = v229(v180(v41), v117.v275() + v13 // 2)
            v194 = v41[v192:v193]
            if v180(v281.v276(v194)) < 12:
                continue
            v40[v118].v189(v194)
    v42 = []
    for v118, v119 in v40.v42():
        v120 = v116(v259.v230(v119))
        if v180(v120) < v11:
            continue
        v39.v160(v120)
        v121 = []
        for v122 in v120[:v11]:
            v72 = v180(v122) // 2
            v231, v232 = (v122[:v72].v214(), v122[v72:].v214())
            if v180(v281.v276(v231)) < 4 or v180(v281.v276(v232)) < 4:
                v121 = []
                break
            v121.v189((v231, v232))
        if v180(v121) == v11:
            v42.v189((v118, v121))
        if v180(v42) >= v12:
            break
    return v42

def train_instance(v43: v123, v44, v32, v39):
    """InfoNCE: positive = disjoint other half of same occurrence; hard negs = same surface."""
    v45 = v37.v195.v124(v43.v156(), lr=v16, weight_decay=0.01)
    v43.v125()
    v46 = None
    for v47 in v105(1, v14 + 1):
        v126 = [v44[v39.v260(0, v180(v44) - 1)] for v187 in v105(v15)]
        v196, v197 = ([], [])
        for v127 in v126:
            for v233, v234 in v127:
                v196.v189(v233)
                v197.v189(v234)
        v128 = v43(v37.v235(v196))
        v129 = v43(v37.v235(v197))
        v130 = v128 @ v129.v236 / v18
        v131 = v37.v188(v128.v237(0), device=v32)
        v76 = 0.5 * (v225.v261(v130, v131) + v225.v261(v130.v236, v131))
        v45.v198(set_to_none=True)
        v76.v199()
        v45.v47()
        if v47 % 500 == 0:
            v144(f'  inst step {v47} loss~{v133(v76):.3f}')
        v46 = v133(v76) if v46 is None else 0.98 * v46 + 0.02 * v133(v76)
    v43.v132()
    return v46

@v37.v36()
def eval_collision(v48, v49, v50, v51, v43, v52: v96, v39, v53: v133=1.0):
    """4-way among the 4 sibling values of the SAME surface (chance 0.25)."""
    v54 = v55 = 0
    for v134, (v118, v121) in v135(v48):
        v136 = v51.v238([v118])[0]
        v137 = []
        for v138 in v105(v11):
            if v52 in ('inst', 'inst_random'):
                v137.v189(v43(v49[v134][v138].v282(0))[0])
            elif v52 == 'ctx_blend':
                v262 = v51.v277(v121[v138][0])
                v137.v189(v225.v179(v136 + v262, dim=-1) if v262 is not None else v136)
            elif v52 == 'soft_rerank':
                v262 = v51.v277(v121[v138][0])
                v137.v189(v225.v179(v136 + 0.25 * v262, dim=-1) if v262 is not None else v136)
            else:
                v137.v189(v136)
        for v138 in v105(v11):
            if v52 in ('inst', 'inst_random'):
                v239 = v43(v50[v134][v138].v282(0))[0]
            elif v52 == 'ctx_blend':
                v262 = v51.v277(v121[v138][1])
                v239 = v225.v179(v136 + v262, dim=-1) if v262 is not None else v136
            elif v52 == 'soft_rerank':
                v262 = v51.v277(v121[v138][1])
                v239 = v225.v179(v136 + 0.25 * v262, dim=-1) if v262 is not None else v136
            else:
                v239 = v136
            v200 = [v133(v263 @ v239) for v263 in v137]
            v201 = v116(v105(v11))
            v39.v160(v201)
            v202 = [v200[v35] for v35 in v201]
            v203 = v201.v240(v138)
            v54 += v22(v22(v265.v267(v202)) == v203)
            v55 += 1
    return v54 / v204(1, v55)

@v37.v36()
def eval_para_hard(v27, v28, v29, v30, v43, v32):

    def code(v139):
        v104 = v162(v27, v28, v29, v30, [v139], v32)
        return v43(v104)[0]
    v56 = [v133(v264(v231) @ v264(v232)) for v231, v232 in v241.v205]
    v57 = [v133(v264(v231) @ v264(v232)) for v231, v232 in v241.v206]
    v140, v141 = (v133(v265.v242(v56)), v133(v265.v242(v57)))
    return {'para': v140, 'hard': v141, 'gap_hard_minus_para': v141 - v140, 'inversion_para_gt_hard': v140 > v141}

@v37.v36()
def next_tok_acc(v27, v28, v30, v32, v55=100):
    if not v3.v207():
        return None
    v42 = []
    with v3.v175('r', encoding='utf-8') as v99:
        for v25 in v99:
            v58 = v253.v243(v25)
            if v58.v266('type') == 'next_tok':
                v42.v189(v58)
            if v180(v42) >= v55:
                break
    if not v42:
        return None
    v54 = 0
    for v58 in v42:
        v142 = [v244(v27, v28, v30, v58['ctx_ids'], v245, v32) for v245 in v58['cand_ids']]
        v54 += v22(v265.v267(v142) == v58['gold_idx'])
    return v54 / v180(v42)

def main() -> v22:
    v0.v98(parents=True, exist_ok=True)
    v6.v143('', encoding='utf-8')
    v144(f'Stage212 start {v278.v272(v279.v273).v219()}')
    v32 = v37.v32('cuda' if v37.v268.v246() else 'cpu')
    v39 = v190.v114(v7)
    v37.v145(v7)
    v59 = v146.v146()
    v147, v148, v149, v150 = v151()
    v29 = v208.v152(v96(v247.v209))
    v60 = v29.v153()
    v30 = v29.v210(v211) or 0
    v28 = v269.v248(v29, v149, v30, v60).v154(v32)
    v27 = v249(v150, v60).v154(v32)
    v27.v155(v37.v250(v1, map_location=v32, weights_only=False)['model'])
    v27.v132()
    for v41 in v27.v156():
        v41.v212(False)
    v61 = v157((v133(v41.v218().v157()) for v41 in v27.v156()))
    v51 = v158(v27, v149, v32)
    v62 = v27.v43.v63
    v144(f'P1 frozen, tape state dim={v62} ({v146.v146() - v59:.0f}s)')
    v64 = v159(v27, v28, v30, v32)
    with v2.v175('r', encoding='utf-8', errors='ignore') as v99:
        v66 = v99.v213(v8)
    v65 = v66[v9:v9 + v10]
    del v66
    v38 = [v41.v214() for v41 in v65.v251('\n') if 200 < v180(v41.v214()) < 1200]
    v39.v160(v38)
    v42 = v161(v38, v39)
    v144(f'surfaces={v180(v42)} x{v11} occurrences ({v146.v146() - v59:.0f}s)')
    if v180(v42) < 40:
        v144('[212] ABORT not enough colliding surfaces')
        return 1
    v67 = [v104[0] for v187, v121 in v42 for v104 in v121]
    v68 = [v104[1] for v187, v121 in v42 for v104 in v121]
    v69 = v162(v27, v28, v29, v30, v67, v32)
    v70 = v162(v27, v28, v29, v30, v68, v32)
    v144(f'tape states {v270(v69.v271)} ({v146.v146() - v59:.0f}s)')
    v163, v164 = ([], [])
    for v35 in v105(v180(v42)):
        v165 = v215(v35 * v11, (v35 + 1) * v11)
        v163.v189(v69[v165])
        v164.v189(v70[v165])
    v71 = v116(v105(v180(v42)))
    v39.v160(v71)
    v72 = v22(0.7 * v180(v71))
    v166, v167 = (v71[:v72], v71[v72:])
    v73 = [[(v163[v35][v252], v164[v35][v252]) for v252 in v105(v11)] for v35 in v166]
    v48 = [v42[v35] for v35 in v167]
    v74 = [v163[v35] for v35 in v167]
    v75 = [v164[v35] for v35 in v167]
    v144(f'train surfaces={v180(v166)} test surfaces={v180(v167)}')
    v43 = v123(v62).v154(v32)
    v76 = v168(v43, v73, v32, v190.v114(v7))
    v144(f'instance head trained loss~{v76:.3f}')
    v77 = v123(v62).v154(v32)
    v77.v132()
    v78 = lambda v52, v104, v216: v217(v48, v74, v75, v51, v104, v52, v190.v114(v216))
    v79 = v78('inst', v43, v7 + 1)
    v80 = v78('inst_random', v77, v7 + 1)
    v81 = v78('fp_only', v43, v7 + 1)
    v82 = v78('ctx_blend', v43, v7 + 1)
    v83 = v78('soft_rerank', v43, v7 + 1)
    v144(f'T1 collision: inst={v79:.3f} inst_random={v80:.3f} fp_only={v81:.3f} ctx_blend={v82:.3f} soft_rerank={v83:.3f}')
    v84 = v169(v27, v28, v29, v30, v43, v32)
    v144(f"T2 para={v84['para']:.3f} hard={v84['hard']:.3f} gap={v84['gap_hard_minus_para']:.3f} inversion={v84['inversion_para_gt_hard']}")
    v85 = v159(v27, v28, v30, v32)
    v86 = v157((v133(v41.v218().v157()) for v41 in v27.v156()))
    v87 = v218(v61 - v86) < 0.001
    v88 = None if v64 is None or v85 is None else v218(v64 - v85)
    v89 = v79 >= 0.7 and v81 <= 0.45
    v90 = v84['inversion_para_gt_hard'] or v84['gap_hard_minus_para'] < 0
    v91 = v79 >= v80 + 0.1
    v92 = v79 >= v83 + 0.1
    v93 = v87 and (v88 is None or v88 <= 0.005)
    v94 = v79 >= v82 + 0.05
    if v89 and v91 and v92 and v93 and v94:
        v170 = 'THESIS_YES'
    elif v89 and v93:
        v170 = 'ENGINEERING_ONLY'
    else:
        v170 = 'THESIS_NO_AT_SCALE'
    v34 = {'timestamp': v278.v272(v279.v273).v219(), 'protocol': 'instance_channel_212', 'overall': v170, 't1_collision_4way': {'instance_learned': v79, 'instance_random_untrained': v80, 'fp_only_surface_key': v81, 'ctx_blend_197': v82, 'soft_rerank': v83, 'chance': v19}, 't2_para_hard': v84, 'next_tok': {'before': v64, 'after': v85, 'delta': v88}, 'gates': {'g1_collision': v89, 'g2_invariance': v90, 'g3_learning_matters': v91, 'g4_beats_soft_rerank': v92, 'g5_no_ce_cost': v93, 'g6_beats_ctx_blend': v94}, 'anticf_frozen': v87, 'surfaces': {'total': v180(v42), 'train': v180(v166), 'test': v180(v167), 'occ_per_surface': v11}, 'note': 'instance channel = read-only 2-layer head on frozen tape state; collision test is 4-way among siblings of the SAME surface form, store/query crops are DISJOINT halves (no lexical overlap shortcut)'}
    v4.v143(v253.v220(v34, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v143('\n'.v221(['# Stage212 — instance / content-invariant channel', '', f'**Overall:** `{v170}`', '', f'- T1 collision (4-way, chance 0.25): instance **{v79:.3f}** | untrained {v80:.3f} | fp_only {v81:.3f} | ctx_blend {v82:.3f} | soft_rerank {v83:.3f}', f"- T2 para={v84['para']:.3f} hard={v84['hard']:.3f} inversion={v84['inversion_para_gt_hard']}", f'- next_tok {v64} -> {v85} (delta {v88}), anti-CF {v87}', f"- gates: {v34['gates']}"]), encoding='utf-8')
    v144(f'[212] {v170} | inst={v79:.3f} blend={v82:.3f} fp={v81:.3f} ({v146.v146() - v59:.0f}s)')
    return 0
if v95 == '__main__':
    raise v171(v222())