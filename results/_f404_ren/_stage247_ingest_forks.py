"""
Stage 247 — Fork map: where do unknown facts go?

One domain stream with planted novel facts. Three ingest policies:

  P_ce     : everything into CE (bindings enter weights)
  P_slots  : everything novel into slots; CE only on binding-stripped filler
  P_hop    : hop-similarity gate → slots if cos(fp(fact), hop_query) high;
             else skip slot; CE on binding-stripped filler (same as P_slots CE)

Then score each policy on:
  M_acquire   fact recall after ingest
  M_edit      overwrite one fact; target updates; retained collateral
  M_cf        after short code-domain CE on the *same* weights (P_ce) or
              query-shift+W (P_slots/P_hop); retained fact recall
  M_under    next_tok on exam v3 (understanding / language proxy)

Not a shipping trunk stage — map of forks for TapeLM evolution.

  python _stage247_ingest_forks.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import re
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
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage191_night import MICRO, PAD, SelfModelXL, W_SELF, load_data, lr_at, sample_windows, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
from _tapelm_ext import DomainAdapter
v0 = v10('results')
v1 = v0 / 'stage247_decision.json'
v2 = v0 / 'stage247_mini.md'
v3 = v0 / '_stage247_log.txt'
v4 = v10('checkpoints/stage191_p1_curve.pt')
v5 = v10('data/_wikitext103_train.txt')
v6 = v10('data/stage191_exam_v3.jsonl')
v7 = 247

def log(v11: v8) -> None:
    v12 = v11 if v11.v217('\n') else v11 + '\n'
    try:
        v218(v12, end='', flush=True)
    except v126:
        v218(v12.v288('ascii', 'replace').v273('ascii'), end='', flush=True)
    v3.v219.v127(parents=True, exist_ok=True)
    with v3.v220('a', encoding='utf-8') as v16:
        v16.v221(v12)

def mask_bindings(v13: v8, v14) -> v8:
    v15 = v13
    for v16 in v14:
        v15 = v15.v222(v16['sent'], f'The chronicle continues without naming the director.')
        v15 = v15.v222(v16['S'], 'Someone')
        v15 = v15.v222(v16['value'], 'somewhere')
    return v15

def build_stream(v17, v14, v18: v223.v128) -> v8:
    """Interleave filler paragraphs with fact sentences."""
    v19 = []
    v20 = 0
    for v129, v16 in v130(v14):
        if v20 < v238(v17):
            v19.v224(v17[v20][:280])
            v20 += 1
        v19.v224(v16['sent'])
        if v129 % 2 == 0 and v20 < v238(v17):
            v19.v224(v17[v20][:200])
            v20 += 1
    while v20 < v225(v238(v17), v238(v14) + 8):
        v19.v224(v17[v20][:240])
        v20 += 1
    v18.v131(v19)
    v19 = []
    for v129, v16 in v130(v14):
        if v129 < v238(v17):
            v19.v224(v17[v129][:260])
        v19.v224(v16['sent'])
    return ' '.v132(v19)

def ce_train(v21, v22, v23, v24, v25, v26, v27, v28, v29, v30=True):
    v31 = v226.v133(v21)
    if v30:
        v251.v227(v31, 'upper')
    else:
        v31.v144()
        for v35 in v31.v143():
            v35.v236(True)
    v32 = [v35 for v35 in v31.v143() if v35.v228]
    v33 = v240.v229.v134(v32, lr=0.0003, weight_decay=0.01)
    v18 = v223.v128(v28)
    for v34 in v135(1, v27 + 1):
        v136 = v274(v22, v23, v275, v18, v25).v166(v26)
        v137 = v136 == v25
        v230, v161, v231 = v31.v232(v24[v136], v137, ids=v136)
        v138 = v136[:, 1:]
        v139 = ~v137[:, :-1] & ~v137[:, 1:]
        v140 = v260.v233(v230[:, :-1][v139], v138[v139])
        v141 = v140 + v261 * v231[~v137].v276()
        v33.v234(set_to_none=True)
        v141.v235()
        v33.v34()
        if v34 % v176(20, v27 // 4) == 0:
            v160(f'  {v29} step {v34}: ce={v149(v140):.3f}')
    v31.v142()
    for v35 in v31.v143():
        v35.v236(False)
    return v31

def gpt_ce_train(v36, v22, v23, v25, v26, v27, v28, v29):
    v37 = v226.v133(v36)
    v33 = v240.v229.v134(v37.v143(), lr=0.0003, weight_decay=0.01)
    v18 = v223.v128(v28)
    v37.v144()
    for v34 in v135(1, v27 + 1):
        v136 = v274(v22, v23, v275, v18, v25).v166(v26)
        v141 = v37(input_ids=v136, labels=v136).v141
        v33.v234(set_to_none=True)
        v141.v235()
        v33.v34()
        if v34 % v176(20, v27 // 4) == 0:
            v160(f'  {v29} step {v34}: loss={v149(v141):.3f}')
    v37.v142()
    return v37

def write_slots(v38, v14):
    return v237.v145(v38, v14)

def slot_recall(v14, v39, v38, v40, v41, v28, v42=None):
    return v237.v146(v14, v39, v38, v40, v41, v28, W_bwd=v42)

def gpt_recall(v36, v43, v25, v14, v39, v26, v28):
    return v237.v147(v36, v43, v25, v14, v39, v26, v28)

def curve_fact_recall(v21, v44, v26, v14, v39, v28, v40=None, v41=None, v38=None, v42=None):
    if v38 is None:
        v38 = v168(v21, v44, v26)
    if v40 is None:
        v40, v41 = v194(v38, v14)
    return (v196(v14, v39, v38, v40, v41, v28, W=v42), v40, v41, v38)

def next_tok_acc(v21, v24, v25, v45, v26):
    if not v45:
        return v149('nan')
    v46 = 0
    for v47 in v45:
        v148 = [v262(v21, v24, v25, v47['ctx_ids'], v103, v26) for v103 in v47['cand_ids']]
        v46 += v9(v9(v291.v285(v148)) == v47['gold_idx'])
    return v46 / v238(v45)

def hop_gate(v38, v14, v48: v8, v49: v149):
    """Admit fact if subject/ctx fp is close to hop query fp."""
    v50 = v38.v150(v48)
    if v50 is None:
        v50 = v38.v255(['organization'])[0]
    v51 = []
    v52 = []
    for v16 in v14:
        v102 = v38.v255([v16['S']])[0]
        v103 = v38.v150(v16['sent'], exclude=v16['value'])
        v151 = v260.v256(v102 + v103, dim=-1) if v103 is not None else v102
        v152 = v149((v151 * v50).v263())
        v52.v224(v152)
        if v152 >= v49:
            v51.v224(v16)
    return (v51, v52)

def main() -> v9:
    v53 = v239.v153()
    v53.v154('--smoke', action='store_true')
    v54 = v53.v155()
    v3.v156('', encoding='utf-8')
    v26 = v240.v26('cuda' if v240.v277.v264() else 'cpu')
    v18 = v223.v128(v7)
    v240.v157(v7)
    v55 = v158.v158()
    v56 = 10 if v54.v159 else 24
    v57 = 80 if v54.v159 else 600
    v58 = 60 if v54.v159 else 400
    v59 = 40 if v54.v159 else 300
    v60 = 40 if v54.v159 else 400
    v61 = 30 if v54.v159 else 80
    v62 = 200 if v54.v159 else 4000
    v63 = 40 if v54.v159 else 200
    v64 = 0.15
    v160(f'Stage247 start {v289.v283(v290.v284).v257()} device={v26}')
    v161, v161, v44, v162 = v163()
    v43 = v241.v164(v8(v265.v242))
    v65 = v43.v165()
    v25 = v43.v243(v244) or 0
    v24 = v278.v266(v43, v44, v25, v65).v166(v26)
    v66 = v267(v162, v65).v166(v26)
    v66.v167(v240.v268(v4, map_location=v26, weights_only=False)['model'])
    v66.v142()
    for v35 in v66.v143():
        v35.v236(False)
    v67 = v168(v66, v44, v26)
    v68 = v226.v133(v245(v26))
    v68.v142()
    with v5.v220('r', encoding='utf-8', errors='ignore') as v16:
        v13 = v16.v246(3000000 if v54.v159 else 12000000)
    v69 = v169(v198.v247((v31.v279(1) for v31 in v292.v286(v13) if v238(v31.v279(1)) >= 5)))
    v18.v131(v69)
    v17 = [v35.v248() for v35 in v13.v269('\n') if v238(v35.v248()) > 180]
    v70 = v169(v198.v247((v250 for v250 in v294.v293('[A-Za-z][a-z]{2,}', v13) if v238(v250) <= 14)))[:v63]
    v71 = v249.v170(v67, v70)
    v72 = [v250 for v250 in v280(v287(v69), v18, v56 + 30) if v238(v250) >= 5][:v56]
    v14 = []
    for v129, v171 in v130(v72):
        v172 = v69[v129]
        v14.v224({'S': v171, 'value': v172, 'sent': f'{v171} was appointed director of {v172} in 1987 .', 'fid': v129})
    v39 = [v16['value'] for v16 in v14] + v69[v56:v56 + 60]
    v48 = 'In the report the organization appointed a new director linked to governance.'
    v161, v173 = v174(v67, v14, v48, thresh=-1.0)
    v73 = v175(v135(v238(v14)), key=lambda v129: v173[v129], reverse=True)
    v74 = v176(2, v238(v14) // 2)
    v75 = {v14[v129]['fid'] for v129 in v73[:v74]}
    v76 = [v16 for v16 in v14 if v16['fid'] in v75]
    v77 = [v16 for v16 in v14 if v16['fid'] not in v75]
    v160(f'facts={v238(v14)} hop_admit={v238(v76)} hop_score_mean={v149(v291.v276(v173)):.3f}')
    v78 = v177(v17, v14, v18)
    v79 = v178(v78, v14)
    v179, v180 = v251.v181(v78, v43, v25, max_lines=v62, min_line_len=16)
    v182, v183 = v251.v181(v79, v43, v25, max_lines=v62, min_line_len=16)
    v45 = []
    if v6.v184():
        with v6.v220(encoding='utf-8') as v16:
            for v12 in v16:
                v47 = v272.v281(v12)
                if v47.v271('type') == 'next_tok':
                    v45.v224(v47)
                if v238(v45) >= v61:
                    break
    v80 = v252.v185(v223.v128(v7 + 1), v54.v159)
    v186, v187 = v251.v181(v80, v43, v25, max_lines=v62, min_line_len=20)
    v81 = {}
    v160('P_ce: GPT CE on full stream (bindings in weights)')
    v82 = v188(v68, v179, v180, v25, v26, v57, v7 + 3, 'P_ce')
    v83 = v189(v82, v43, v25, v14, v39, v26, v7)
    v84 = v14[:v176(2, v56 // 5)]
    v85 = v14[v238(v84):]
    v86 = [[v129 for v129 in v43.v288(v16['sent']).v136 if v129 != v25] for v16 in v84]
    v87 = v226.v133(v82)
    v33 = v240.v229.v134(v87.v143(), lr=5e-05)
    v87.v144()
    for v34 in v135(1, (20 if v54.v159 else 40) + 1):
        v190 = v237.v253(v223.v128(v7 + v34), v86, [], 4, 64, v26, mix_real=False)
        v141 = -v87(input_ids=v190, labels=v190).v141
        v33.v234(set_to_none=True)
        v141.v235()
        v33.v34()
    v87.v142()
    v88 = v189(v87, v43, v25, v84, v39, v26, v7)
    v89 = v189(v87, v43, v25, v85, v39, v26, v7)
    v90 = v188(v82, v186, v187, v25, v26, v58, v7 + 4, 'P_ce_cf')
    v91 = v189(v90, v43, v25, v14, v39, v26, v7)
    v92 = v237.v191(v90, v45, v26)
    v81['P_ce'] = {'acquire': v83, 'edit_target_after_unlearn': v88, 'edit_retained': v89, 'edit_collateral': v254(v89 - v189(v82, v43, v25, v85, v39, v26, v7)), 'cf_retain': v91, 'cf_drop': v83 - v91, 'understand_next_tok': v92, 'carrier': 'gpt_weights'}
    v160(f'  P_ce acq={v83:.3f} cf={v91:.3f} edit_ret={v89:.3f} under={v92:.3f}')
    v160('P_slots: write all facts to slots; CE on masked stream')
    v192, v193 = v194(v67, v14)
    v93 = v195(v66, v182, v183, v24, v25, v26, v57, v7 + 5, 'P_slots', True)
    v94 = v168(v93, v44, v26)
    v95 = v196(v14, v39, v67, v192, v193, v7)
    v96 = v192.v197()
    v97 = v169(v193)
    v98 = 0
    v99 = v69[v56 + 3]
    v100 = v97[v98]
    v97[v98] = v99
    v101 = v198(v14[v98])
    v101['value'] = v99
    v101['sent'] = f"{v101['S']} was appointed director of {v99} in 1987 ."
    v102 = v67.v255([v101['S']])[0]
    v103 = v67.v150(v101['sent'], exclude=v99)
    v96[v98] = v260.v256(v102 + v103, dim=-1) if v103 is not None else v102
    v104 = [v101]
    v105 = [v14[v98]]
    v106 = v14[1:]
    v107 = v196(v104, v39 + [v99], v67, v96, v97, v7)
    v108 = v196(v105, v39, v67, v96, v97, v7)
    v109 = v196(v106, v39, v67, v96, v97, v7)
    v110 = v249.v199(v93, v186, v187, v24, v25, v26, v59, v7 + 6)
    v111 = v168(v110, v44, v26)
    v42, v200 = v249.v201(v282(256).v166(v26), v249.v170(v111, v70), v71, v18, v60, v26)
    v112 = v196(v14, v39, v111, v192, v193, v7, W=v42)
    v113 = v202(v93, v24, v25, v45, v26)
    v81['P_slots'] = {'acquire': v95, 'edit_new_ok': v107, 'edit_old_gone': 1.0 - v108, 'edit_retained': v109, 'edit_collateral': v254(v109 - v196(v106, v39, v67, v192, v193, v7)), 'cf_retain': v112, 'cf_drop': v95 - v112, 'understand_next_tok': v113, 'W_align': v200, 'carrier': 'slots+masked_CE'}
    v160(f'  P_slots acq={v95:.3f} cf={v112:.3f} edit_new={v107:.3f} under={v113:.3f}')
    v160('P_hop: hop-similar facts → slots only; CE masked')
    v203, v204 = v194(v67, v76) if v76 else (v240.v270(1, 256, device=v26), [])
    v114 = v195(v66, v182, v183, v24, v25, v26, v57, v7 + 7, 'P_hop', True)
    v115 = v196(v76, v39, v67, v203, v204, v7) if v76 else 0.0
    v116 = v196(v77, v39, v67, v203, v204, v7) if v77 and v76 else 0.0
    if v238(v76) >= 2:
        v205 = v203.v197()
        v206 = v169(v204)
        v206[0] = v69[v56 + 5]
        v207 = v198(v76[0])
        v207['value'] = v206[0]
        v207['sent'] = f"{v207['S']} was appointed director of {v206[0]} in 1987 ."
        v102 = v67.v255([v207['S']])[0]
        v103 = v67.v150(v207['sent'], exclude=v206[0])
        v205[0] = v260.v256(v102 + v103, dim=-1) if v103 is not None else v102
        v208 = v196([v207], v39 + [v206[0]], v67, v205, v206, v7)
        v209 = v196(v76[1:], v39, v67, v205, v206, v7)
        v210 = v254(v209 - v196(v76[1:], v39, v67, v203, v204, v7))
    else:
        v208, v209, v210 = (v149('nan'), v149('nan'), v149('nan'))
    v117 = v249.v199(v114, v186, v187, v24, v25, v26, v59, v7 + 8)
    v118 = v168(v117, v44, v26)
    v211, v212 = v249.v201(v282(256).v166(v26), v249.v170(v118, v70), v71, v18, v60, v26)
    v119 = v196(v76, v39, v118, v203, v204, v7, W=v211) if v76 else 0.0
    v120 = v202(v114, v24, v25, v45, v26)
    v81['P_hop'] = {'acquire_admitted': v115, 'acquire_rejected_should_be_low': v116, 'edit_new_ok': v208, 'edit_retained': v209, 'edit_collateral': v210, 'cf_retain': v119, 'cf_drop': v115 - v119, 'understand_next_tok': v120, 'n_admitted': v238(v76), 'n_rejected': v238(v77), 'W_align': v212, 'carrier': 'hop_gated_slots+masked_CE'}
    v160(f'  P_hop in={v115:.3f} out={v116:.3f} cf={v119:.3f} edit_new={v208} under={v120:.3f}')
    v140, v213, v214 = (v81['P_ce'], v81['P_slots'], v81['P_hop'])
    v121 = v213['cf_retain'] >= v140['cf_retain'] + 0.15
    v122 = v213['edit_collateral'] <= 0.05
    v123 = v214['acquire_admitted'] >= 0.7 and v214['acquire_rejected_should_be_low'] <= v214['acquire_admitted'] - 0.2
    v124 = v213['understand_next_tok'] + 0.05 >= v140.v271('understand_next_tok', 0) or v213['understand_next_tok'] >= 0.55
    if v121 and v122 and v123:
        v215 = 'INGEST_FORK_SLOTS_AND_HOP'
    elif v121 and v122:
        v215 = 'INGEST_FORK_SLOTS_BEATS_CE'
    elif v123:
        v215 = 'INGEST_FORK_HOP_SELECTIVE'
    else:
        v215 = 'INGEST_FORK_MIXED'
    v15 = {'stage': 247, 'overall': v215, 'gates': {'G_slots_cf_beats_ce_0p15': v121, 'G_slots_edit_low_collateral': v122, 'G_hop_admits_and_rejects': v123, 'G_slots_under_not_ruined': v124}, 'policies': v81, 'n_facts': v56, 'ce_steps': v57, 'cf_steps': v58, 'note': 'Fork map only. P_ce=parametric GPT; P_slots/P_hop=TapeLM slots + masked CE. Evolution hint: keep bindings out of CE; use hop-sim as admission, not as CE curriculum.', 'timestamp': v289.v283(v290.v284).v257(), 'wall_s': v158.v158() - v55}
    v0.v127(parents=True, exist_ok=True)
    v1.v156(v272.v258(v15, indent=2), encoding='utf-8')
    v2.v156(f"# Stage 247 ingest forks\n\n**{v215}**\n\n| policy | acquire | cf_retain | edit_collateral | under |\n|--------|---------|-----------|-----------------|-------|\n| P_ce | {v140['acquire']:.2f} | {v140['cf_retain']:.2f} | {v140['edit_collateral']:.2f} | {v140['understand_next_tok']:.2f} |\n| P_slots | {v213['acquire']:.2f} | {v213['cf_retain']:.2f} | {v213['edit_collateral']:.2f} | {v213['understand_next_tok']:.2f} |\n| P_hop | {v214['acquire_admitted']:.2f} (out {v214['acquire_rejected_should_be_low']:.2f}) | {v214['cf_retain']:.2f} | {v214['edit_collateral']} | {v214['understand_next_tok']:.2f} |\n", encoding='utf-8')
    v160(v272.v258({'overall': v215, 'gates': v15['gates']}, indent=2))
    return 0
if v125 == '__main__':
    raise v216(v259())