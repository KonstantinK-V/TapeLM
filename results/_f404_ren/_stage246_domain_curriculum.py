"""
Stage 246 — Multi-domain sequential retention (full TapeLM stack vs GPT).

Curriculum (default order):
  wiki → tinystories → med → news(ag-like)
Each phase trains for --steps (smoke small; full default 3000; use --steps 30000 for paper-scale).

TapeLM (product-faithful, full stack):
  - frozen canonical P1 arc_enc + shared slot bank
  - per domain: learn W_bwd (qmap), train head_domain (225, arc frozen), write domain facts into slots
  - after each phase: for EVERY past domain measure
      gen = window next_tok with matched head
      mem = slot recall with matched W @ shifted query encoder

GPT (parametric continuum):
  - sequential CE on the same domain flats
  - after each phase: for EVERY past domain measure window CE→PPL and planted-fact recall

Output: retention matrix phase × domain × {tape_gen, tape_mem, gpt_ppl, gpt_fact}.

  python _stage246_domain_curriculum.py [--smoke] [--steps N]
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
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage224_far_shift as s224
import _stage225_family_fork as s225
import _stage24x_lib as L
from _stage191_night import MICRO, PAD, SelfModelXL, load_data, sample_windows
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import load_gpt
from _tapelm_ext import DomainAdapter
v0 = v14('results')
v1 = v14('data')
v2 = v0 / 'stage246_decision.json'
v3 = v0 / 'stage246_mini.md'
v4 = v0 / '_stage246_log.txt'
v5 = v14('checkpoints/stage191_p1_curve.pt')
v6 = v14('data/_wikitext103_train.txt')
v7 = v14('data/external_tinystories_100k_85.txt')
v8 = 246
v9 = v94.v15('\\b(said|reuters|reported|minister|election|president|government|officials|according to|announced|senate|parliament|campaign)\\b', v94.v16)
v10 = ('wiki', 'stories', 'med', 'news')

def log(v17: v11) -> None:
    v18 = v17 if v17.v155('\n') else v17 + '\n'
    try:
        v156(v18, end='', flush=True)
    except v95:
        v156(v18.v252('ascii', 'replace').v230('ascii'), end='', flush=True)
    v4.v157.v96(parents=True, exist_ok=True)
    with v4.v158('a', encoding='utf-8') as v51:
        v51.v159(v18)

def ensure_news(v19: v13) -> v11:
    v20 = v1 / '_stage246_news_corpus.txt'
    if v20.v160() and v20.v231().v161 > 5000:
        return v20.v100(encoding='utf-8')
    v21: v97[v11] = []
    with v6.v158('r', encoding='utf-8', errors='ignore') as v51:
        for v18 in v51:
            v18 = v18.v210()
            if v162(v18) < 48 or not v9.v244(v18):
                continue
            v21.v152(v18)
            if v162(v21) >= v19:
                break
    if v162(v21) < 80:
        raise v163('news corpus too small')
    v20.v98('\n'.v164(v21), encoding='utf-8')
    v99(f'news corpus lines={v162(v21)}')
    return v20.v100(encoding='utf-8')

def domain_text(v22: v11, v19: v13, v23: v101, v24: v165.v102) -> v11:
    if v22 == 'wiki':
        with v6.v158('r', encoding='utf-8', errors='ignore') as v51:
            return v51.v186(2000000 if v23 else 12000000)
    if v22 == 'stories':
        return v7.v100(encoding='utf-8', errors='ignore')
    if v22 == 'med':
        return v211.v166(max_lines=v19)
    if v22 == 'news':
        return v167(v19)
    raise v103(v22)

@v109.v34()
def window_ce(v25, v26, v27, v28, v29, v30, v24, v31=12) -> v12:
    v32 = []
    for v33 in v104(v31):
        v105 = v232(v26, v27, v233, v24, v29).v137(v30)
        v106 = v105 == v29
        v168, v33, v33 = v25.v169(v28[v105], v106, ids=v105)
        v107 = v105[:, 1:]
        v108 = ~v106[:, :-1] & ~v106[:, 1:]
        if v108.v212() == 0:
            continue
        v32.v152(v12(v245.v234(v168[:, :-1][v108], v107[v108])))
    return v12(v235.v213(v32)) if v32 else v12('nan')

@v109.v34()
def gpt_window_ce(v35, v26, v27, v29, v30, v24, v31=12) -> v12:
    v32 = []
    for v33 in v104(v31):
        v105 = v232(v26, v27, v233, v24, v29).v137(v30)
        v92 = v35(input_ids=v105, labels=v105)
        v32.v152(v12(v92.v112))
    return v12(v235.v213(v32)) if v32 else v12('nan')

def train_gpt_domain(v35, v26, v27, v29, v30, v36, v37, v38, v39, v40):
    v41 = v109.v170.v110(v35.v139(), lr=v37, weight_decay=0.01)
    v24 = v165.v102(v38)
    v35.v111()
    for v42 in v104(1, v36 + 1):
        v105 = v232(v26, v27, v233, v24, v29).v137(v30)
        v112 = v35(input_ids=v105, labels=v105).v112
        v41.v171(set_to_none=True)
        v112.v172()
        v41.v42()
        if v42 % v39 == 0:
            v99(f'  gpt {v40} step {v42}: loss={v12(v112):.3f}')
    v35.v113()

def train_head(v25, v26, v27, v28, v29, v30, v36, v38):
    return v173.v114(v25, v26, v27, v28, v29, v30, v36, v38)

def plant_domain_facts(v43, v44, v45, v24, v46: v11, v47: v13):
    v48 = v165.v102(v8 + v47)
    v49 = [v174 for v174 in v236(v246(v44), v48, v45 + 20) if v162(v174) >= 5][:v45]
    v50 = []
    for v115, v116 in v117(v49):
        v118 = v44[(v115 + v47) % v162(v44)]
        v119 = f'{v116} was appointed director of {v118} in the {v46} chronicle of 1987 .'
        v50.v152({'S': v116, 'value': v118, 'sent': v119, 'domain': v46, 'fid': f'{v46}_{v115}'})
    v120, v121 = ([], [])
    for v51 in v50:
        v122 = v43.v214([v51['S']])[0]
        v123 = v43.v175(v51['sent'], exclude=v51['value'])
        v120.v152(v245.v237(v122 + v123, dim=-1) if v123 is not None else v122)
        v121.v152(v51['value'])
    return (v50, v109.v176(v120, 0), v121)

def mem_recall(v50, v52, v53, v54, v55, v38: v13) -> v12:
    v56 = v97(v215.v177(v54))
    return v178.v124(v50, v56 + v56, v52, v53, v54, v38, W_bwd=v55)

def gpt_fact_acc(v35, v57, v29, v50, v30, v38: v13) -> v12:
    v56 = [v51['value'] for v51 in v50]
    return v178.v125(v35, v57, v29, v50, v56 + v56, v30, v38)

def main() -> v13:
    v58 = v179.v126()
    v58.v127('--smoke', action='store_true')
    v58.v127('--steps', type=v13, default=0, help='steps/domain (0 => smoke 120 / full 3000)')
    v59 = v58.v128()
    v4.v98('', encoding='utf-8')
    v30 = v109.v30('cuda' if v109.v238.v216() else 'cpu')
    v24 = v165.v102(v8)
    v109.v129(v8)
    v60 = v130.v130()
    v36 = v59.v36 or (120 if v59.v23 else 3000)
    v61 = 40 if v59.v23 else v180(400, v131(80, v36 // 4))
    v62 = 40 if v59.v23 else v180(600, v131(100, v36 // 3))
    v63 = 40 if v59.v23 else v180(800, v131(100, v36 // 2))
    v64 = 0.0003
    v45 = 8 if v59.v23 else 24
    v19 = 250 if v59.v23 else 8000
    v65 = 50 if v59.v23 else 300
    v66 = 6 if v59.v23 else 16
    v39 = v131(20, v36 // 4)
    v99(f'Stage246 start {v250.v242(v251.v243).v207()} device={v30} steps/domain={v36} domains={v97(v10)}')
    from _stage191_night import load_data
    v33, v33, v132, v133 = v134()
    v57 = v181.v135(v11(v217.v182))
    v67 = v57.v136()
    v29 = v57.v183(v184) or 0
    v28 = v239.v218(v57, v132, v29, v67).v137(v30)
    v68 = v219(v133, v67).v137(v30)
    v68.v138(v109.v220(v5, map_location=v30, weights_only=False)['model'])
    v68.v113()
    for v69 in v68.v139():
        v69.v185(False)
    v70 = v140(v68, v132, v30)
    with v6.v158('r', encoding='utf-8', errors='ignore') as v51:
        v141 = v51.v186(4000000 if v59.v23 else 20000000)
    v44 = v97(v215.v177((v204.v240(1) for v204 in v253.v247(v141) if v162(v204.v240(1)) >= 5)))
    v24.v142(v44)
    v71 = v97(v215.v177((v174 for v174 in v94.v254('[A-Za-z][a-z]{2,}', v141) if v162(v174) <= 14)))[:v65]
    v72 = v187.v143(v70, v71)
    v73 = {}
    v74 = {}
    for v115, v144 in v117(v10):
        v145 = v188(v144, v19, v59.v23, v24)
        v26, v27 = v221.v189(v145, v57, v29, max_lines=v19, min_line_len=20)
        v73[v144] = v145
        v74[v144] = (v26, v27)
        v99(f'domain {v144}: tokens={v162(v26)} docs={v162(v27) - 1}')
    v35 = v190.v146(v191(v30))
    v35.v113()
    v75 = {}
    v55 = {}
    v76 = {}
    v77 = {}
    v78 = {}
    v79 = {}
    v80 = []
    for v147, v144 in v117(v10):
        v99(f'\n=== PHASE {v147 + 1}/{v162(v10)}: {v144} ({v36} steps) ===')
        v26, v27 = v74[v144]
        v148 = v187.v192(v68, v26, v27, v28, v29, v30, v61, v8 + 10 + v147)
        v52 = v140(v148, v132, v30)
        v76[v144] = v52
        v149, v193 = v187.v194(v248(256).v137(v30), v187.v143(v52, v71), v72, v24, v62, v30)
        v55[v144] = v149
        v75[v144] = v195(v68, v26, v27, v28, v29, v30, v63, v8 + 50 + v147)
        v50, v53, v54 = v196(v70, v44, v45, v24, v144, seed_off=100 + v147 * 17)
        v79[v144], v77[v144], v78[v144] = (v50, v53, v54)
        v99(f'  tape {v144}: W_align={v193:.3f} facts={v162(v50)}')
        v197(v35, v26, v27, v29, v30, v36, v64, v8 + 200 + v147, v39, v144)
        v150 = v10[:v147 + 1]
        v83 = {'after_phase': v144, 'domains': {}}
        for v85 in v150:
            v222, v223 = v74[v85]
            v198 = v224(v75[v85], v222, v223, v28, v29, v30, v165.v102(v8 + 7), v66)
            v199 = v173.v225(v75[v85], v222, v223, v28, v29, v30, v165.v102(v8 + 8), v66)
            v200 = v226(v79[v85], v76[v85], v77[v85], v78[v85], v55[v85], v8 + v147)
            v201 = v227(v35, v222, v223, v29, v30, v165.v102(v8 + 9), v66)
            v202 = v249.v241(v180(20.0, v201)) if v201 == v201 else v12('nan')
            v203 = v228(v35, v57, v29, v79[v85], v30, v8 + v147)
            v83['domains'][v85] = {'tape_ce': v198, 'tape_next_tok': v199, 'tape_mem': v200, 'gpt_ce': v201, 'gpt_ppl': v202, 'gpt_fact': v203}
            v99(f'  eval[{v85}]: tape_nt={v199:.3f} tape_mem={v200:.3f} | gpt_ppl={v202:.2f} gpt_fact={v203:.3f}')
        v80.v152(v83)
        v99(f'phase {v144} done ({v130.v130() - v60:.0f}s)')
    v81 = v80[-1]['domains']
    v82 = {}
    for v83 in v80:
        for v85, v204 in v83['domains'].v205():
            if v85 not in v82:
                v82[v85] = v204
    v84 = {}
    for v85 in v10:
        if v85 not in v81 or v85 not in v82:
            continue
        v84[v85] = {'tape_mem_drop': v82[v85]['tape_mem'] - v81[v85]['tape_mem'], 'tape_nt_drop': v82[v85]['tape_next_tok'] - v81[v85]['tape_next_tok'], 'gpt_fact_drop': v82[v85]['gpt_fact'] - v81[v85]['gpt_fact'], 'gpt_ppl_rise': v81[v85]['gpt_ppl'] - v82[v85]['gpt_ppl']}
    v86 = v10[0]
    v87 = v81.v206(v86, {}).v206('tape_mem', 0) >= 0.7
    v88 = v81.v206(v86, {}).v206('tape_next_tok', 0) >= 0.5
    v89 = v84.v206(v86, {}).v206('gpt_fact_drop', 0) >= 0.1
    v90 = v84.v206(v86, {}).v206('gpt_ppl_rise', 0) >= 0.5
    v91 = v81.v206(v86, {}).v206('tape_mem', 0) - v81.v206(v86, {}).v206('gpt_fact', 0) >= 0.15
    if v87 and v88 and (v89 or v90) and v91:
        v151 = 'DOMAIN_CURRICULUM_OK'
    elif v87 and (v89 or v91):
        v151 = 'DOMAIN_CURRICULUM_PARTIAL'
    else:
        v151 = 'DOMAIN_CURRICULUM_NO'
    v92 = {'stage': 246, 'overall': v151, 'steps_per_domain': v36, 'domains': v97(v10), 'gates': {'G_tape_keeps_wiki_mem_ge_0p70': v87, 'G_tape_keeps_wiki_gen_ge_0p50': v88, 'G_gpt_wiki_fact_drop_ge_0p10': v89, 'G_gpt_wiki_ppl_rise_ge_0p5': v90, 'G_gap_mem_ge_0p15': v91}, 'matrix': v80, 'drops_first_to_final': v84, 'note': 'TapeLM = frozen P1 + per-domain {W, head, slots}. GPT = single weight trajectory. news = AG-like wiki filter (not HF AG News download). Pass --steps 30000 for paper-scale curriculum.', 'timestamp': v250.v242(v251.v243).v207(), 'wall_s': v130.v130() - v60}
    v0.v96(parents=True, exist_ok=True)
    v2.v98(v229.v208(v92, indent=2), encoding='utf-8')
    v21 = [f'# Stage 246 domain curriculum\n\n**{v151}** steps/domain={v36}\n']
    v21.v152('| after \\ domain | ' + ' | '.v164(v10) + ' |')
    v21.v152('|' + '---|' * (v162(v10) + 1))
    for v83 in v80:
        v153 = []
        for v85 in v10:
            v204 = v83['domains'].v206(v85)
            if not v204:
                v153.v152('-')
            else:
                v153.v152(f"tMem{v204['tape_mem']:.2f}/gPPL{v204['gpt_ppl']:.1f}")
        v21.v152(f"| {v83['after_phase']} | " + ' | '.v164(v153) + ' |')
    v3.v98('\n'.v164(v21) + '\n', encoding='utf-8')
    v99(v229.v208({'overall': v151, 'drops': v84}, indent=2))
    return 0
if v93 == '__main__':
    raise v154(v209())