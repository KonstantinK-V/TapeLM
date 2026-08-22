"""
Stage 254 — Continual curriculum with ONE shared upper (fix for 246).

246 trained a fresh head per domain from canonical P1, so `tape_next_tok` for wiki was the
identical 0.2998 in every phase and all drops were exactly 0.0: retention was trivially true
and understanding had nowhere to accumulate. Here a SINGLE upper walks wiki -> stories -> med
-> news with the 253 recipe (CE + 0.2*CPC), so forgetting is possible and growth is possible.

Per phase d:
  - inject fact sentences into domain lines, then STRIP bindings from the CE text
    (typed placeholders, not one repeated stub) -> facts never enter the weights
  - hop-gate admission -> admitted facts written to the shared canonical slot bank
  - train shared upper: joint CE + lam*CPC on (domain + replay of past domains)
  - learn W_bwd[d] (shifted arc_enc -> canonical) for reading slots after query drift

After every phase, for EVERY seen domain: held-out CE/PPL, slot recall against the
ACCUMULATED bank, parametric leak. Global: exam next_tok, uniformity.
Cross-domain 2-hop over the accumulated tape = the "thought" metric (chains are planted so a
value from domain i-1 is the subject of a fact in domain i).

Note: internal latent hops are closed (210/212 **THESIS_NO_AT_SCALE**) — hops here are external fp loops.

  python _stage254_continual_understand.py [--smoke] [--operators-only] [--token-budget N] [--domains wiki,med]

  --operators-only: frozen P1 upper; only W_query (+ growing tape). No joint CE/CPC, no arc shift.
"""
from __future__ import annotations
import argparse
import json
import math
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
import _stage224_far_shift as s224
import _stage246_domain_curriculum as s246
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
v0 = v19('results')
v1 = v0 / 'stage254_decision.json'
v2 = v0 / 'stage254_mini.md'
v3 = v0 / '_stage254_log.txt'
v4 = v19('checkpoints/stage191_p1_curve.pt')
v5 = v19('checkpoints/stage254_continual_upper.pt')
v6 = v19('data/_wikitext103_train.txt')
v7 = 254
v8 = 0.2
v9 = 0.25
v10 = 'In the report the organization appointed a new director linked to governance.'
v11 = 'The recipe for {S} calls for {V} simmered slowly in a copper pan .'
v12 = '{S} was appointed director of {V} in the {D} chronicle of 1987 .'
v13 = ['The chronicle continues with other institutional details .', 'A later passage turns to unrelated regional history .', 'The record notes routine administrative procedure .', 'Following lines cover general background only .', 'The section closes without naming any official .']

def log(v20: v16) -> None:
    v21 = v20 if v20.v206('\n') else v20 + '\n'
    try:
        v207(v21, end='', flush=True)
    except v124:
        v207(v21.v335('ascii', 'replace').v305('ascii'), end='', flush=True)
    v3.v208.v125(parents=True, exist_ok=True)
    with v3.v209('a', encoding='utf-8') as v44:
        v44.v210(v21)
v14 = v19('data/_tinystories_raw_100k.txt')
v15 = 40

def build_filtered(v22: v19, v23, v24: v18, v25: v18=48) -> v16:
    """Domain slice of wikitext; 246's caches were capped at 8000 lines, too small here."""
    if v22.v211() and v22.v306().v212 > 100000:
        return v22.v213(encoding='utf-8')
    v26: v30[v16] = []
    with v6.v209('r', encoding='utf-8', errors='ignore') as v44:
        for v21 in v44:
            v21 = v21.v215()
            if v159(v21) < v25 or not v23.v324(v21):
                continue
            v26.v217(v21)
            if v159(v26) >= v24:
                break
    v22.v126('\n'.v128(v26), encoding='utf-8')
    v127(f'  built {v22.v27}: {v159(v26)} lines')
    return '\n'.v128(v26)

def domain_lines(v27: v16, v24: v18, v28: v129, v29: v214.v130) -> v30[v16]:
    if v27 == 'stories' and v14.v211():
        v131 = v14.v213(encoding='utf-8', errors='ignore')[:4000000 if not v28 else 400000]
    elif v27 == 'med' and (not v28):
        v131 = v271(v19('data/_stage254_med.txt'), v307.v272, v24)
    elif v27 == 'news' and (not v28):
        v131 = v271(v19('data/_stage254_news.txt'), v325.v308, v24)
    else:
        v131 = v325.v309(v27, v24, v28, v29)
    v26 = [v216.v215() for v216 in v131.v273('\n') if v159(v216.v215()) >= v15]
    return v26[:v24]

def plant_facts(v31, v32, v33, v34, v35, v29):
    """On-theme facts (some chained from the previous domain) + off-theme gate distractors.

    Half of each group carries wq_train=True: those fit W_query, the rest score recall.
    """
    v36 = [v132 for v132 in v274(v310(v31), v29, v32 + v33 + 30) if v159(v132) >= 5]
    v37 = []
    v38 = 0
    for v39 in v133(v32):
        v134 = v35[v39] if v39 < v159(v35) else v36[v38]
        if v39 >= v159(v35):
            v38 += 1
        v135 = v31[v29.v275(v159(v31))]
        v37.v217({'S': v134, 'value': v135, 'sent': v12.v311(S=v134, V=v135, D=v34), 'domain': v34, 'theme': 'org', 'chained': v39 < v159(v35), 'wq_train': v39 % 2 == 0, 'fid': f'{v34}_on_{v39}'})
    for v39 in v133(v33):
        v134 = v36[v38]
        v38 += 1
        v135 = v31[v29.v275(v159(v31))]
        v37.v217({'S': v134, 'value': v135, 'sent': v11.v311(S=v134, V=v135), 'domain': v34, 'theme': 'food', 'chained': False, 'wq_train': v39 % 2 == 0, 'fid': f'{v34}_off_{v39}'})
    return v37

def hop_gate(v40, v37, v41='median'):
    v42 = v40.v136(v10)
    if v42 is None:
        v42 = v40.v276(['organization'])[0]
    v43 = []
    for v44 in v37:
        v137 = v40.v276([v44['S']])[0]
        v138 = v40.v136(v44['sent'], exclude=v44['value'])
        v139 = v312.v277(v137 + v138, dim=-1) if v138 is not None else v137
        v43.v217(v218((v139 * v42).v241()))
    v45 = v218(v224.v278(v43)) if v41 == 'median' else 0.0
    v46 = [v44 for v44, v140 in v141(v37, v43) if v140 >= v45]
    for v44, v140 in v141(v37, v43):
        if v44['chained'] and v44 not in v46:
            v46.v217(v44)
    return (v46, v43, v45)

def inject_and_mask(v26: v30[v16], v37, v29: v214.v130) -> v50[v16, v16]:
    """Append fact sentences into real lines; masked copy replaces only those sentences with placeholders."""
    v47 = v30(v26)
    v48 = v30(v26)
    if not v47:
        raise v219('empty domain')
    v49 = v142(1, v159(v47) // v142(1, v159(v37)))
    for v39, v44 in v143(v37):
        v144 = v202(v159(v47) - 1, v39 * v49)
        v47[v144] = v47[v144] + ' ' + v44['sent']
        v48[v144] = v48[v144] + ' ' + v13[v39 % v159(v13)]
    return ('\n'.v128(v47), '\n'.v128(v48))

def concat_corpora(v51: v17) -> v50[v224.v149, v224.v149, v17]:
    """One global flat/offsets so replay is just a doc-id mix."""
    v145, v146, v147 = ([], [0], {})
    for v91, (v220, v221) in v51.v96():
        v148 = v159(v146) - 1
        v101 = v146[-1]
        v145.v217(v220)
        for v39 in v133(v159(v221) - 1):
            v146.v217(v101 + v18(v221[v39 + 1]))
        v147[v91] = (v148, v159(v146) - 1)
    return (v224.v222(v145), v224.v223(v146, dtype=v224.v279), v147)

def retrieve_value(v52, v53, v54, v55: v16, v56=None) -> v16:
    v42 = v52.v136(f'In the report {v55} was linked to the organization.')
    if v42 is None:
        v42 = v52.v276([v55])[0]
    if v56 is not None:
        v42 = v312.v277(v56.v313(v42.v326(0)), dim=-1)[0]
    return v54[v18((v53 @ v42).v280())]

def two_hop_acc(v40, v53, v54, v57, v58, v59: v18) -> v17:
    """S_a -> mid -> final over the accumulated tape (external fp loop, 203-style)."""
    if not v57:
        return {'strict': v218('nan'), 'four_way': v218('nan'), 'n': 0, 'hop1': v218('nan')}
    v29 = v214.v130(v59)
    v60 = v61 = v62 = 0
    for v150, v151, v104 in v57:
        v152 = v225(v40, v53, v54, v150)
        v62 += v18(v152 == v151)
        v153 = v225(v40, v53, v54, v152)
        v60 += v18(v153 == v104)
        v154 = [v226 for v226 in v58 if v226 != v104]
        v29.v179(v154)
        v155 = [v104] + v154[:3]
        v156 = v30(v133(4))
        v29.v179(v156)
        v157 = [v155[v39] for v39 in v156]
        v42 = v40.v136(f'In the report {v152} was linked to the organization.')
        if v42 is None:
            v42 = v40.v276([v152])[0]
        v158 = []
        for v138 in v157:
            v227 = [v144 for v144, v226 in v143(v54) if v226 == v138]
            v158.v217(v218((v53[v227] @ v42).v142()) if v227 else -1.0)
        v61 += v18(v18(v224.v280(v158)) == v156.v314(0))
    v63 = v159(v57)
    return {'strict': v60 / v63, 'four_way': v61 / v63, 'hop1': v62 / v63, 'n': v63}

def main() -> v18:
    v64 = v228.v160()
    v64.v161('--smoke', action='store_true')
    v64.v161('--operators-only', action='store_true', help='frozen upper; train W_query only (tape grows). Skips joint CE/CPC and per-domain arc/W_bwd.')
    v64.v161('--token-budget', type=v18, default=0, help='CE tokens per domain (joint mode only)')
    v64.v161('--domains', type=v16, default='wiki,stories,med,news')
    v65 = v64.v162()
    global DECISION, MINI
    if v65.v66:
        v163 = 'operators_smoke' if v65.v28 else 'operators'
        v1 = v0 / f'stage254_decision_{v163}.json'
        v2 = v0 / f'stage254_mini_{v163}.md'
    v3.v126('', encoding='utf-8')
    v67 = v229.v67('cuda' if v229.v315.v281() else 'cpu')
    v29 = v214.v130(v7)
    v229.v164(v7)
    v68 = v165.v165()
    v69 = [v91.v215() for v91 in v65.v69.v273(',') if v91.v215()]
    v70 = v65.v166 or (150000 if v65.v28 else 4000000)
    v32 = 6 if v65.v28 else 16
    v33 = 4 if v65.v28 else 10
    v71 = 2 if v65.v28 else 6
    v72 = 40 if v65.v28 else 120
    v73 = 24 if v65.v28 else 60
    v74 = 6 if v65.v28 else 16
    v75 = 4 if v65.v28 else 8
    v24 = 300 if v65.v28 else 40000
    v76 = 3.0
    v77 = 50 if v65.v28 else 250
    v78 = 40 if v65.v28 else 300
    v79 = 40 if v65.v28 else 400
    v127(f'Stage254 start {v333.v321(v334.v322).v267()} domains={v69} budget/domain={v70} operators_only={v65.v66}')
    v167, v167, v168, v169 = v170()
    v80 = v230.v171(v16(v282.v231))
    v81 = v80.v172()
    v82 = v80.v232(v233) or 0
    v83 = v316.v283(v80, v168, v82, v81).v173(v67)
    v84 = v284(v169, v81).v173(v67)
    v84.v174(v229.v285(v4, map_location=v67, weights_only=False)['model'])
    v84.v175()
    for v85 in v84.v176():
        v85.v234(False)
    v40 = v177(v84, v168, v67)
    with v6.v209('r', encoding='utf-8', errors='ignore') as v44:
        v178 = v44.v235(2000000 if v65.v28 else 8000000)
    v31 = v30(v17.v236((v20.v317(1) for v20 in v336.v327(v178) if v159(v20.v317(1)) >= 5)))
    v29.v179(v31)
    v86 = v30(v17.v236((v132 for v132 in v339.v337('[A-Za-z][a-z]{2,}', v178) if v159(v132) <= 14)))[:v77]
    v87 = v237.v180(v40, v86)
    v181, v182, v183 = ({}, {}, {})
    v88 = []
    v89 = {}
    for v39, v91 in v143(v69):
        v26 = v238(v91, v24, v65.v28, v29)
        v35 = v88[:v71] if v39 > 0 else []
        v37 = v239(v31, v32, v33, v91, v35, v29)
        v46, v43, v45 = v240(v40, v37)
        v184 = v241((1 for v44 in v46 if v44['theme'] == 'org'))
        v185 = v241((1 for v44 in v46 if v44['theme'] == 'food'))
        v89[v91] = {'n_facts': v159(v37), 'admitted': v159(v46), 'on_theme_admitted': v184, 'off_theme_admitted': v185, 'thresh': v45}
        v242, v48 = v243(v26, v37, v29)
        v220, v221 = v286.v244(v48, v80, v82, max_lines=v24, min_line_len=20)
        v183[v91] = (v220, v221)
        v181[v91], v182[v91] = (v37, v46)
        v88 = [v44['value'] for v44 in v37 if v44['theme'] == 'org']
        v127(f'  {v91}: docs={v159(v221) - 1} facts={v159(v37)} admitted={v159(v46)} (on={v184}/{v32}, off={v185}/{v33})')
    v90 = {}
    for v91 in v69:
        for v44 in v181[v91]:
            v90.v287(v44['S'], v44)
    v92 = []
    for v91 in v69:
        for v44 in v181[v91]:
            if v44['theme'] != 'org':
                continue
            v245 = v90.v288(v44['value'])
            if v245 is not None and v245['domain'] != v44['domain']:
                v92.v217((v44['S'], v44['value'], v245['value']))
    v127(f'cross-domain 2-hop chains: {v159(v92)}')
    v93 = {}
    for v91 in v69:
        v186 = v18(v159(v183[v91][0]))
        if v186 < (20000 if v65.v28 else 200000):
            raise v219(f'domain {v91} too small ({v186} tokens) — fix the source before running')
        v93[v91] = v18(v202(v70, v76 * v186))
        v127(f'  {v91}: corpus_tokens={v186} budget={v93[v91]} (~{v93[v91] / v186:.1f} epochs)')
    v187, v188, v147 = v189(v183)
    v94 = {}
    for v91 in v69:
        v246, v247 = v147[v91]
        v190 = v247 - v246
        v191 = v142(4, v18(v190 * 0.05))
        v94[v91] = (v30(v133(v246, v247 - v191)), v30(v133(v247 - v191, v247)))
        v127(f'  {v91}: train_docs={v159(v94[v91][0])} hold_docs={v159(v94[v91][1])}')
    v95 = {v91: v289.v248(v187, v188, v94[v91][1], v82, v74, v7 + 5 + v39) for v39, v91 in v143(v69)}
    v96 = v249.v192(v72)
    v97 = v96[:v73]
    v98 = v30(v17.v236((v44['value'] for v91 in v69 for v44 in v181[v91])))
    v99 = v229.v193(0, v87.v250(-1), device=v67)
    v100: v30[v16] = []
    v101 = {'exam_next_tok': v249.v251(v84, v83, v82, v96, v67), 'uniformity': v289.v252(v84, v187, v188, v83, v82, v67, v94[v69[0]][1], 48, v7 + 9), 'hold_ce': {v91: v289.v290(v84, v95[v91], v83, v82, v67) for v91 in v69}, 'leak': {v91: v249.v291(v84, v83, v82, v80, v181[v91], v98, v67, v7 + 300) for v91 in v69}}
    v127(f"baseline exam_nt={v101['exam_next_tok']:.3f} hold={ {v137: v328(v226, 3) for v137, v226 in v101['hold_ce'].v96()}} leak={ {v137: v328(v226, 3) for v137, v226 in v101['leak'].v96()}}")
    v102 = 40 if v65.v28 else 180
    v103 = v253.v194(v67)
    v20 = v84
    v195, v196, v197, v122 = ({}, {}, {}, [])
    for v39, v91 in v143(v69):
        v127(f'\n=== PHASE {v39 + 1}/{v159(v69)}: {v91} ===')
        v198 = []
        for v44 in v182[v91]:
            v137 = v40.v276([v44['S']])[0]
            v138 = v40.v136(v44['sent'], exclude=v44['value'])
            v198.v217(v312.v277(v137 + v138, dim=-1) if v138 is not None else v137)
        if v198:
            v99 = v229.v292([v99, v229.v329(v198, 0)], 0)
            v100 = v100 + [v44['value'] for v44 in v182[v91]]
        v127(f'  slots: +{v159(v198)} -> bank={v159(v100)}')
        if not v65.v66:
            v293, v294 = v183[v91]
            v254 = v237.v295(v84, v293, v294, v83, v82, v67, v78, v7 + 10 + v39)
            v255 = v177(v254, v168, v67)
            v56, v296 = v237.v297(v338(256).v173(v67), v237.v180(v255, v86), v87, v29, v79, v67)
            v195[v91], v196[v91] = (v56, v255)
            v127(f'  W[{v91}] align={v296:.3f}')
            v256 = v30(v94[v91][0])
            if v39 > 0 and v9 > 0:
                v298 = [v318 for v201 in v69[:v39] for v318 in v94[v201][0]]
                v299 = v202(v159(v298), v18(v159(v256) * v9 / v142(1e-06, 1 - v9)))
                v256 = v256 + v214.v130(v7 + 77 + v39).v330(v298, v299)
                v127(f'  replay: +{v299} past docs ({v9:.0%} target)')
            v257 = [v300 for v201 in v69[:v39 + 1] for v300 in v95[v201]]
            v20, v258 = v289.v301(v20, v187, v188, v83, v82, v67, v93[v91], v8, v7 + 100 + v39, f'phase_{v91}', v256, v257, v97, early_stop=False, n_probes=v75)
        else:
            v196[v91] = v40
            v195[v91] = None
            v258 = {'tokens_ce': 0, 'tokens_cpc': 0, 'steps': 0}
            v127('  operators-only: upper frozen — skip arc/W_bwd and joint train')
        v199 = [v44 for v302 in v69[:v39 + 1] for v44 in v182[v302] if v44['wq_train']]
        if v199 and v159(v100) > 0:
            v253.v303(v103, v40, v199, v99, v100, v67, v102, v7 + 400 + v39)
        v200 = {'after_phase': v91, 'bank_size': v159(v100), 'domains': {}}
        for v201 in v69[:v39 + 1]:
            v259 = v289.v290(v20, v95[v201], v83, v82, v67)
            v260 = [v44 for v44 in v182[v201] if not v44['wq_train']]
            v261 = v253.v319(v260, v98, v40, v99, v100, v7, W_bwd=v103) if v260 else {'four_way': v218('nan'), 'full_bank_top1': v218('nan'), 'full_bank_mrr': v218('nan'), 'full_bank_median_rank': v218('nan')}
            v262 = v253.v319(v260, v98, v196[v201], v99, v100, v7, W_bwd=v195[v201]) if v260 and v195.v288(v201) is not None else {'four_way': v218('nan'), 'full_bank_top1': v218('nan'), 'full_bank_mrr': v218('nan'), 'full_bank_median_rank': v218('nan')}
            v263 = v249.v291(v20, v83, v82, v80, v181[v201], v98, v67, v7 + 300)
            v200['domains'][v201] = {'hold_ce': v259, 'hold_ppl': v331.v320(v202(v259, 20)), 'mem': v261, 'mem_shift': v262, 'leak': v263}
            v197.v287(v201, v17(v200['domains'][v201]))
        v200['exam_next_tok'] = v249.v251(v20, v83, v82, v96, v67)
        v200['uniformity'] = v289.v252(v20, v187, v188, v83, v82, v67, v94[v91][1], 48, v7 + 9)
        v200['inversion'] = v249.v264(v20, v83, v82, v80, v67)
        v200['hop'] = v265(v40, v99, v100, v92, v98, v7 + v39)
        v200['train_meta'] = {v137: v258[v137] for v137 in ('tokens_ce', 'tokens_cpc', 'steps')}
        v122.v217(v200)
        v127(f"  after {v91}: exam={v200['exam_next_tok']:.3f} hop2={v200['hop']['four_way']:.3f} bank={v159(v100)} " + ' '.v128((f"{v201}[ce={v200['domains'][v201]['hold_ce']:.3f} mem={v200['domains'][v201]['mem']:.2f}]" for v201 in v69[:v39 + 1])) + f' ({v165.v165() - v68:.0f}s)')
    v104 = v122[-1]
    v105 = {v201: v104['domains'][v201]['hold_ce'] - v197[v201]['hold_ce'] for v201 in v104['domains']}
    v106 = {v201: v104['domains'][v201]['hold_ce'] - v101['hold_ce'][v201] for v201 in v104['domains']}
    v107 = v142(v105.v266())
    v108 = v142(v106.v266())
    v109 = v202((v226['mem'] for v226 in v104['domains'].v266() if not v331.v332(v226['mem'])))
    v110 = v142((v226['leak'] for v226 in v104['domains'].v266()))
    v111 = v142((v104['domains'][v201]['leak'] - v101['leak'][v201] for v201 in v104['domains']))
    v112 = [v121['exam_next_tok'] for v121 in v122]
    v113 = v108 <= 0.15
    v114 = v107 <= 0.15
    v115 = v104['exam_next_tok'] >= v101['exam_next_tok'] - 0.01
    v116 = v109 >= 0.75
    v117 = v111 <= 0.12
    v118 = v104['hop']['four_way'] >= 0.5
    v119 = v104['uniformity'] <= v101['uniformity'] + 0.1
    if v113 and v115 and v116 and v117 and v118:
        v203 = 'CONTINUAL_UNDERSTAND_OK'
    elif v113 and v116 and v117 and (v115 or v118):
        v203 = 'CONTINUAL_UNDERSTAND_PARTIAL'
    else:
        v203 = 'CONTINUAL_UNDERSTAND_NO'
    v47 = {'stage': 254, 'mode': 'operators_only' if v65.v66 else 'joint_upper', 'overall': v203, 'domains': v69, 'lambda': v8, 'replay_frac': v9, 'token_budget_per_domain': v70, 'budget_used_per_domain': v93, 'max_epochs_per_domain': v76, 'gates': {'G_no_forget_vs_P1': v113, 'G_peak_hold_regress': v114, 'G_understanding_holds': v115, 'G_mem_holds_full_bank': v116, 'G_no_param_leak': v117, 'G_cross_domain_hop': v118, 'G_no_collapse': v119}, 'summary': {'exam_curve': v112, 'exam_base': v101['exam_next_tok'], 'max_forget_hold_ce_vs_first_phase': v107, 'max_forget_hold_ce_vs_P1': v108, 'forget_per_domain_vs_first': v105, 'forget_per_domain_vs_P1': v106, 'min_mem_full_bank': v109, 'max_leak': v110, 'max_leak_delta_vs_P1': v111, 'baseline_leak': v101['leak'], 'hop_final': v104['hop'], 'bank_size_final': v104['bank_size']}, 'gate_stats': v89, 'baseline': v101, 'matrix': v122, 'note': 'operators_only: frozen P1 upper; local-mask CE corpus unused for weight updates; only W_query trains on wq_train facts; tape grows via hop-gate. mem=canonical+W_q.' if v65.v66 else 'One shared upper across domains. Canonical slot KEYS frozen; W_query trains each phase on wq_train facts only, mem is scored on the held-out half. mem=canonical+W_q; mem_shift=arc-shift+W_bwd. Leak gate: delta vs P1 baseline (fixed seed), not absolute 0.40.', 'timestamp': v333.v321(v334.v322).v267(), 'wall_s': v165.v165() - v68}
    v1.v126(v304.v268(v47, indent=2), encoding='utf-8')
    v120 = ['# Stage 254 continual understanding' + (' (operators-only)' if v65.v66 else ' (shared upper)') + '', f"**{v203}** domains={'->'.v128(v69)} budget={v70} tok/domain", '', f"- exam: {v101['exam_next_tok']:.3f} -> " + ' -> '.v128((f'{v318:.3f}' for v318 in v112)), f"- max forget vs P1 (hold CE): {v108:+.3f} | vs first phase: {v107:+.3f} | min mem @bank {v104['bank_size']}: {v109:.3f} | max leak: {v110:.3f}", f"- cross-domain 2-hop: 4way={v104['hop']['four_way']:.3f} strict={v104['hop']['strict']:.3f} n={v104['hop']['n']}", '', '| after \\ domain | ' + ' | '.v128(v69) + ' |', '|' + '---|' * (v159(v69) + 1)]
    for v121 in v122:
        v204 = []
        for v201 in v69:
            v226 = v121['domains'].v288(v201)
            v204.v217(f"ce {v226['hold_ce']:.2f} / mem {v226['mem']:.2f}" if v226 else '-')
        v120.v217(f"| {v121['after_phase']} | " + ' | '.v128(v204) + ' |')
    v2.v126('\n'.v128(v120) + '\n', encoding='utf-8')
    v127(v304.v268({'overall': v203, 'exam_curve': v112, 'hop': v104['hop']}, indent=2))
    if not v65.v28:
        v5.v208.v125(exist_ok=True)
        v229.v269({'model': v20.v323(), 'stage': 254, 'domains': v69, 'W_query': v103.v323()}, v5)
    return 0
if v123 == '__main__':
    raise v205(v270())