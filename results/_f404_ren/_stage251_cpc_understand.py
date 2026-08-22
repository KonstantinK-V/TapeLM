"""
Stage 251 — Understanding ladder: calibrate instrument, then CPC-upper on real corpus.

Plan (ingest-forks follow-up):
  0) CAL: upper CE on unmasked load_data() for a fixed *token budget* (is exam next_tok movable?)
  1) CPC: upper + frozen arc_enc; CPC on load_data windows; facts live in slots only (no binding CE)
  2) Fixed eval: exam next_tok + held-out wiki mean CE/PPL + 179 inversion + slot mem + parametric leak

Corpus: load_data() flat/offsets (multi-document), NOT single-doc join from 248/250.

  python _stage251_cpc_understand.py [--smoke] [--cal-only] [--skip-cal]
  python _stage251_cpc_understand.py --token-budget 4000000
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import re
import time
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
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
from _stage191_night import MICRO, MAX_ARCS, PAD, SelfModelXL, W_SELF, load_data, sample_windows, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
v0 = v22('results')
v1 = v0 / 'stage251_decision.json'
v2 = v0 / 'stage251_mini.md'
v3 = v0 / '_stage251_log.txt'
v4 = v22('checkpoints/stage191_p1_curve.pt')
v5 = v22('data/_wikitext103_train.txt')
v6 = v22('data/stage191_exam_v3.jsonl')
v7 = 251
v8 = 32
v9 = 64
v10 = 0.07
v11 = 0.0003
v12 = 0.0003
v13 = 0.05
v14 = ['The passage continues with other institutional details.', 'Later sections discuss unrelated regional history.', 'The narrative shifts to broader economic context.', 'A following paragraph covers administrative procedure.', 'The article mentions general background without specifics.', 'Subsequent lines treat a different topic entirely.']

def log(v23: v15) -> None:
    v24 = v23 if v23.v214('\n') else v23 + '\n'
    try:
        v215(v24, end='', flush=True)
    except v130:
        v215(v24.v305('ascii', 'replace').v288('ascii'), end='', flush=True)
    v3.v216.v131(parents=True, exist_ok=True)
    with v3.v217('a', encoding='utf-8') as v56:
        v56.v218(v24)

def strip_fact_bindings(v25: v15, v26, v27: v219.v132) -> v15:
    """Remove binding sentences; per-fact placeholders (no repeated single stub)."""
    v28 = v25
    for v133, v56 in v134(v26):
        v135 = v14[v133 % v221(v14)]
        v28 = v28.v220(v56['sent'], v135)
        v28 = v28.v220(v56['S'], f'PersonAlpha{v133}')
        v28 = v28.v220(v56['value'], f'RegionBeta{v133}')
    return v28

def split_train_hold(v29, v30: v20=v13) -> v35[v19[v17], v19[v17]]:
    v31 = v221(v29) - 1
    v32 = v136(12, v17(v31 * v30))
    v33 = v19(v138(v31 - v32))
    v34 = v19(v138(v31 - v32, v31))
    return (v33, v34)

def sample_windows_docs(v36, v29, v37, v27: v219.v132, v38: v17, v39: v19[v17]) -> v42.v16:
    v40 = v222.v137((v37, v223), v38, dtype=v222.v165)
    for v41 in v138(v37):
        v82 = v39[v27.v268(0, v221(v39) - 1)]
        v224, v225 = (v29[v82], v29[v82 + 1])
        v139 = v225 - v224
        if v139 <= v223:
            v40[v41, :v139] = v36[v224:v225]
        else:
            v166 = v224 + v27.v268(0, v139 - v223)
            v40[v41] = v36[v166:v166 + v223]
    return v42.v140(v40)

def count_valid_tokens(v43: v42.v16, v38: v17) -> v17:
    return v17((v43 != v38).v246().v226())

class Predictor(v44.v18):

    def __init__(v141, v82: v17):
        v289().v227()
        v141.v142 = v44.v228(v44.v269(v82, v82), v44.v270(), v44.v269(v82, v82))

    def forward(v141, v143):
        return v271.v229(v141.v142(v143), dim=-1)

def load_exam_next(v45: v17) -> v19:
    v46 = []
    if not v6.v230():
        return v46
    with v6.v217(encoding='utf-8') as v56:
        for v24 in v56:
            v51 = v287.v272(v24)
            if v51.v290('type') == 'next_tok':
                v46.v254(v51)
            if v221(v46) >= v45:
                break
    return v46

def next_tok_acc(v47, v48, v38, v46, v49) -> v20:
    if not v46:
        return v20('nan')
    v50 = 0
    for v51 in v46:
        v144 = [v273(v47, v48, v38, v51['ctx_ids'], v274, v49) for v274 in v51['cand_ids']]
        v50 += v17(v17(v222.v304(v144)) == v51['gold_idx'])
    return v50 / v221(v46)

def curve_param_recall(v47, v48, v38, v52, v26, v53, v49, v54: v17) -> v20:
    v55 = v219.v132(v54 + 11)
    v50 = 0
    for v56 in v26:
        v145 = [v133 for v133 in v52.v305(f"In the report {v56['S']} was linked to the organization of").v43 if v133 != v38]
        v146 = [v153 for v153 in v53 if v153 != v56['value']]
        v55.v176(v146)
        v147 = [v56['value']] + v146[:3]
        v148 = v19(v138(4))
        v55.v176(v148)
        v149 = [v147[v133] for v133 in v148]
        v144 = [v273(v47, v48, v38, v145, [v133 for v133 in v52.v305(' ' + v274).v43 if v133 != v38], v49) for v274 in v149]
        v50 += v17(v17(v222.v304(v144)) == v148.v291(0))
    return v50 / v136(1, v221(v26))

@v42.v60()
def wiki_mean_ce(v47, v36, v29, v48, v38, v49, v39: v19[v17], v57: v17, v54: v17) -> v20:
    v27 = v219.v132(v54)
    v58 = []
    for v59 in v138(v57):
        v43 = v292(v36, v29, 1, v27, v38, v39).v168(v49)
        v74 = v43 == v38
        v173, v59, v59 = v47.v231(v48[v43], v74, ids=v43)
        v150 = v43[:, 1:]
        v151 = ~v74[:, :-1] & ~v74[:, 1:]
        if v151.v232():
            v58.v254(v20(v271.v241(v173[:, :-1][v151], v150[v151])))
    return v20(v222.v234(v58)) if v58 else v20('nan')

def inversion_fast(v47, v48, v38, v52, v49) -> v21:

    @v42.v60()
    def z(v152: v15) -> v42.v16:
        v43 = [v133 for v133 in v52.v305(v152).v43 if v133 != v38][:v223]
        v153 = v42.v233([v43], device=v49)
        v74 = v153 == v38
        v75 = v47.v163(v48[v153], v153)
        v76 = v47.v76(v75, pad_mask=v74)
        v23 = (~v74).v20().v164(-1)
        v72 = (v76 * v23).v246(1) / v23.v246(1).v247(min=1.0)
        return v271.v229(v72[0], dim=-1)
    v61 = v20(v222.v234([v20(v271.v306(v143(v307), v143(v41), dim=-1)) for v307, v41 in v308.v293]))
    v62 = v20(v222.v234([v20(v271.v306(v143(v307), v143(v41), dim=-1)) for v307, v41 in v308.v294]))
    return {'para': v61, 'hard': v62, 'gap_hard_minus_para': v62 - v61, 'inversion': v61 > v62}

def train_ce_token_budget(v63, v36, v29, v48, v38, v49, v64, v54, v65, v66) -> v35[v162, v17]:
    v23 = v235.v154(v63)
    v236.v155(v23, 'upper')
    v67 = [v72 for v72 in v23.v160() if v72.v237]
    v68 = v42.v238.v156(v67, lr=v12, weight_decay=0.01)
    v27 = v219.v132(v54)
    v69 = 0
    v70 = 0
    v71 = 0.0
    while v69 < v64:
        v43 = v292(v36, v29, v295, v27, v38, v66).v168(v49)
        v69 += v239(v43, v38)
        v74 = v43 == v38
        v173, v59, v240 = v23.v231(v48[v43], v74, ids=v43)
        v150 = v43[:, 1:]
        v151 = ~v74[:, :-1] & ~v74[:, 1:]
        v157 = v271.v241(v173[:, :-1][v151], v150[v151])
        v158 = v157 + v275 * v240[~v74].v234()
        v68.v242(set_to_none=True)
        v158.v243()
        v42.v44.v276.v244(v67, 1.0)
        v68.v70()
        v70 += 1
        v71 = v20(v157)
        if v70 % v136(50, v64 // (v295 * v223 * 20)) == 0:
            v161(f'  {v65} tokens={v69}/{v64} step={v70} ce={v71:.3f}')
    v23.v159()
    for v72 in v23.v160():
        v72.v245(False)
    v161(f'  {v65} done tokens={v69} steps={v70} ce={v71:.3f}')
    return (v23, v69)

def pooled_fast_train(v47, v48, v73: v42.v16, v38: v17) -> v42.v16:
    v74 = v73 == v38
    v75 = v47.v163(v48[v73], v73)
    v76 = v47.v76(v75, pad_mask=v74)
    v23 = (~v74).v20().v164(-1)
    return (v76 * v23).v246(1) / v23.v246(1).v247(min=1.0)

def cpc_draw(v36, v29, v77, v27: v219.v132, v38: v17, v37: v17, v78: v17):
    v79 = v222.v137((v37, v78), v38, v222.v165)
    v80 = v222.v137((v37, v78), v38, v222.v165)
    v81 = [v82 for v82 in v77 if v29[v82 + 1] - v29[v82] >= 2 * v78]
    if not v81:
        v81 = v77
    for v41 in v138(v37):
        v82 = v81[v27.v268(0, v221(v81) - 1)]
        v224, v225 = (v29[v82], v29[v82 + 1])
        v166 = v224 + v27.v268(0, v136(0, v225 - v224 - 2 * v78))
        v79[v41] = v36[v166:v166 + v78]
        v80[v41] = v36[v166 + v78:v166 + 2 * v78]
    return (v42.v140(v79), v42.v140(v80))

def train_cpc_token_budget(v63, v36, v29, v48, v38, v49, v64, v54, v65, v66) -> v35[v162, v17]:
    v23 = v235.v154(v63)
    v236.v155(v23, 'upper')
    v82 = v23.v248.v167 // 2
    v83 = v277(v82).v168(v49)
    v67 = [v72 for v72 in v23.v160() if v72.v237] + v19(v83.v160())
    v68 = v42.v238.v156(v67, lr=v11, weight_decay=0.01)
    v27 = v219.v132(v54)
    v69 = 0
    v70 = 0
    v84 = 0.0
    while v69 < v64:
        v79, v80 = v249(v36, v29, v66, v27, v38, v9, v8)
        v69 += v239(v79, v38) + v239(v80, v38)
        v79, v80 = (v79.v168(v49), v80.v168(v49))
        v169 = v271.v229(v278(v23, v48, v79, v38), dim=-1)
        v170 = v271.v229(v278(v23, v48, v80, v38), dim=-1)
        v171 = v83(v169)
        v172 = v83(v170)
        v173 = v171 @ v172.v279 / v10
        v174 = v42.v250(v79.v280(0), device=v49)
        v158 = v271.v241(v173, v174)
        v68.v242(set_to_none=True)
        v158.v243()
        v42.v44.v276.v244(v67, 1.0)
        v68.v70()
        v70 += 1
        v84 = v20(v158)
        if v70 % v136(40, v64 // (v9 * v8 * 30)) == 0:
            v161(f'  {v65} tokens={v69}/{v64} step={v70} cpc={v84:.3f}')
    v23.v159()
    v83.v159()
    for v72 in v23.v160():
        v72.v245(False)
    v161(f'  {v65} done tokens={v69} steps={v70} cpc={v84:.3f}')
    return (v23, v69)

def make_facts(v27: v219.v132, v85: v17, v86: v175) -> v35[v19, v19]:
    with v5.v217('r', encoding='utf-8', errors='ignore') as v56:
        v25 = v56.v251(2000000 if v86 else 8000000)
    v87 = v19(v21.v252((v23.v296(1) for v23 in v313.v309(v25) if v221(v23.v296(1)) >= 5)))
    v27.v176(v87)
    v88 = [v253 for v253 in v297(v310(v87), v27, v85 + 20) if v221(v253) >= 5][:v85]
    v26 = []
    for v133, v177 in v134(v88):
        v178 = v87[v133]
        v26.v254({'S': v177, 'value': v178, 'sent': f'{v177} was appointed director of {v178} in 1987 .', 'fid': v133})
    v53 = [v56['value'] for v56 in v26] + v87[v85:v85 + 40]
    return (v26, v53)

def main() -> v17:
    v89 = v255.v179()
    v89.v180('--smoke', action='store_true')
    v89.v180('--cal-only', action='store_true')
    v89.v180('--skip-cal', action='store_true')
    v89.v180('--token-budget', type=v17, default=0, help='per phase (cal and cpc); 0 = preset')
    v90 = v89.v181()
    v3.v182('', encoding='utf-8')
    v49 = v42.v49('cuda' if v42.v298.v281() else 'cpu')
    v27 = v219.v132(v7)
    v42.v183(v7)
    v91 = v184.v184()
    v92 = v90.v64 or (120000 if v90.v86 else 2000000)
    v85 = 8 if v90.v86 else 20
    v93 = 40 if v90.v86 else 120
    v94 = 24 if v90.v86 else 80
    v161(f'Stage251 start {v311.v301(v312.v302).v264()} token_budget/phase={v92} smoke={v90.v86}')
    v36, v29, v185, v186 = v187()
    v66, v188 = v189(v29)
    v161(f'corpus docs train={v221(v66)} hold={v221(v188)} tokens~{v221(v36)}')
    v52 = v256.v190(v15(v282.v257))
    v95 = v52.v191()
    v38 = v52.v258(v259) or 0
    v48 = v299.v283(v52, v185, v38, v95).v168(v49)
    v96 = v162(v186, v95).v168(v49)
    v96.v192(v42.v284(v4, map_location=v49, weights_only=False)['model'])
    v96.v159()
    for v72 in v96.v160():
        v72.v245(False)
    v26, v53 = v193(v27, v85, v90.v86)
    v97 = v194(v96, v185, v49)
    v195, v196 = v78.v197(v97, v26)
    v46 = v198(v93)
    v98 = v78.v199(v26, v53, v97, v195, v196, v7)
    v99 = v200(v96, v48, v38, v46, v49)
    v100 = v201(v96, v48, v38, v52, v26, v53, v49, v7)
    v101 = v202(v96, v48, v38, v52, v49)
    v102 = v203(v96, v36, v29, v48, v38, v49, v188, v94, v7 + 1)
    v103 = v260.v204(v261(v102, 20))
    v161(f"baseline nt={v99:.3f} mem={v98['four_way']:.3f} fb_top1={v98['full_bank_top1']:.3f} leak={v100:.3f} inv={v101['inversion']} gap={v101['gap_hard_minus_para']:+.3f} hold_ce={v102:.3f} ppl~{v103:.1f}")
    v104 = v96
    v105 = 0
    v106 = v99
    v107 = v101
    v108 = v102
    if not v90.v205:
        v161('phase0 CAL: upper CE on unmasked load_data (instrument)')
        v104, v105 = v262(v96, v36, v29, v48, v38, v49, v92, v7 + 10, 'cal_ce', v66)
        v106 = v200(v104, v48, v38, v46, v49)
        v107 = v202(v104, v48, v38, v52, v49)
        v108 = v203(v104, v36, v29, v48, v38, v49, v188, v94, v7 + 11)
        v161(f'  CAL nt {v99:.3f}->{v106:.3f} delta={v106 - v99:+.3f} hold_ce={v108:.3f}')
    v109 = v106 - v99
    if v109 >= 0.015:
        v206 = 'CAL_MOVES_EXAM'
    elif v109 <= 0.005 and v109 >= -0.005:
        v206 = 'CAL_CEILING'
    elif v109 < -0.02:
        v206 = 'CAL_HURTS'
    else:
        v206 = 'CAL_FLAT'
    v110 = v104
    v111 = 0
    v112 = v106
    v113 = v107
    v114 = v108
    v115 = v98
    v116 = v100
    if not v90.v128:
        v161('phase1 CPC: upper on load_data; facts in slots only')
        v110, v111 = v263(v96, v36, v29, v48, v38, v49, v92, v7 + 20, 'cpc_upper', v66)
        v112 = v200(v110, v48, v38, v46, v49)
        v113 = v202(v110, v48, v38, v52, v49)
        v114 = v203(v110, v36, v29, v48, v38, v49, v188, v94, v7 + 21)
        v207 = v194(v110, v185, v49)
        v115 = v78.v199(v26, v53, v207, v195, v196, v7)
        v116 = v201(v110, v48, v38, v52, v26, v53, v49, v7 + 22)
    with v5.v217('r', encoding='utf-8', errors='ignore') as v56:
        v208 = v56.v251(500000)
    v117 = [v72.v285() for v72 in v208.v300('\n') if v221(v72.v285()) > 160][:v85 + 5]
    v118 = ' '.v209(v117[:20])
    v59, v210 = v236.v211(v118, v52, v38, max_lines=500, min_line_len=16)
    v119 = {'load_data_docs': v221(v29) - 1, 'bad_joined_docs': v221(v210) - 1, '248_250_single_doc_risk': v221(v210) - 1 <= 2}
    v120 = v115['four_way'] >= 0.75
    v121 = v116 <= 0.4
    v122 = v114 <= v102 * 1.12 + 0.05
    v123 = v112 >= v99 - 0.03
    v124 = v112 >= v99 + 0.01
    v125 = v113['gap_hard_minus_para'] < v101['gap_hard_minus_para'] - 0.01 or v113['inversion']
    v126 = v206 in ('CAL_MOVES_EXAM', 'CAL_CEILING')
    v127 = None
    if v125 and (not v124) and (v206 == 'CAL_CEILING'):
        v127 = 'TRY_ARC_UNFREEZE_202B'
    elif v206 == 'CAL_HURTS':
        v127 = 'ADD_KL_REPLAY_ANCHOR'
    elif v124 and v120 and v121:
        v127 = 'SCALE_CPC_TOKENS'
    elif not v125 and (not v124):
        v127 = 'TRY_PAWS_202'
    if v90.v128:
        v212 = 'CAL_' + v206.v220('CAL_', '') if v206.v286('CAL_') else v206
    elif v120 and v121 and v122 and v123 and (v124 or v125):
        v212 = 'CPC_UNDERSTAND_OK'
    elif v120 and v121 and v123 and (v124 or v125 or v125):
        v212 = 'CPC_UNDERSTAND_PARTIAL'
    elif v126 and (not v124) and (not v125):
        v212 = 'CPC_UNDERSTAND_NO'
    else:
        v212 = 'CPC_UNDERSTAND_PARTIAL'
    v28 = {'stage': 251, 'overall': v212, 'token_budget_per_phase': v92, 'cal_verdict': v206, 'fork_next': v127, 'mask_corpus_note': v119, 'gates': {'G_mem': v120, 'G_no_param_leak': v121, 'G_holdout_ppl_ok': v122, 'G_exam_not_worse': v123, 'G_exam_gain': v124, 'G_inversion_gain': v125, 'G_cal_instrument': v126}, 'baseline': {'next_tok': v99, 'slot_mem': v98, 'param_leak': v100, 'inversion': v101, 'hold_ce': v102, 'hold_ppl': v103}, 'cal_ce': {'tokens': v105, 'next_tok': v106, 'delta_nt': v109, 'inversion': v107, 'hold_ce': v108}, 'cpc_upper': {'tokens': v111, 'next_tok': v112, 'delta_nt_vs_base': v112 - v99, 'slot_mem': v115, 'param_leak': v116, 'inversion': v113, 'hold_ce': v114, 'hold_ppl': v260.v204(v261(v114, 20))}, 'timestamp': v311.v301(v312.v302).v264(), 'wall_s': v184.v184() - v91, 'note': 'Facts in slots only; train corpus = load_data(). CAL=unmasked CE; CPC=consequence prediction on upper.'}
    v1.v182(v287.v265(v28, indent=2), encoding='utf-8')
    v2.v182(f"# Stage 251 CPC understand\n\n**{v212}** cal={v206} fork={v127}\nnt {v99:.3f}->{v106:.3f}(cal)->{v112:.3f}(cpc) mem={v115['four_way']:.3f} fb_rank={v115['full_bank_median_rank']:.0f} leak={v116:.3f}\ninv gap {v101['gap_hard_minus_para']:+.3f}->{v113['gap_hard_minus_para']:+.3f}\n", encoding='utf-8')
    v161(v287.v265({'overall': v212, 'cal_verdict': v206, 'fork': v127}, indent=2))
    if not v90.v86 and v92 >= 500000 and (not v90.v128):
        v22('checkpoints').v131(exist_ok=True)
        v42.v266({'model': v110.v303(), 'stage': 251, 'tokens': v111}, 'checkpoints/stage251_cpc_upper.pt')
    return 0
if v129 == '__main__':
    raise v213(v267())