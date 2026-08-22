"""
Stage 191 — NIGHT-9h scale run (plan: results/plan_stage191_night9h.md).

Phases (idempotent — done phase = json exists, rerun skips):
  P0 data+exam : 150M chars → proper line-split id docs (npz cache) + Exam v3 (freq-matched)
  P1 curve-XL  : self-model dual-channel d256/6L, clean CE + read-only surprise, ≤15k steps
  P2 gpt-XL    : matched GPT-2 (d256/6L/T64), same data/steps
  P3 rarity    : P1 + char-trigram rarity feature in ink + surprise temperature (S3-G3 fix)
  P4 sweep     : gate B + doclink for P1/P2/P3/187-old (does scale move meaning?)
  P5 report    : night report + verdicts

NOTE: 181's build_id_docs split on 

 (absent in this wiki file) → all prior runs actually
trained on a ~2M-char fallback slice. Tonight = ~75x data.

  python _stage191_night.py --phase all
  python _stage191_night.py --phase all --smoke   (fast end-to-end check)
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
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
import _stage186_exam_v2 as s186
import _stage187_self_model as s187
v0 = v26('results')
v1 = v26('data')
v2 = v26('checkpoints')
v3 = v0 / '_stage191_log.txt'
v4 = v26('data/_wikitext103_train.txt')
v5 = v1 / 'stage191_docs.npz'
v6 = v1 / 'stage191_charset.json'
v7 = v1 / 'stage191_exam_v3.jsonl'
v8 = v0 / 'stage191_night_report.md'
v9 = v32.v9
v10 = 191
v11 = 256
v12 = 6
v13 = v32.v13
v14 = 16
v15 = 0.0003
v16 = 200
v17 = 0.1
v18 = '[PAD]'
v19 = 150000000
v20 = 3000000
v21 = {'p1': 3.2 * 3600, 'p2': 1.8 * 3600, 'p3': 1.7 * 3600}
v22 = {'p1': 15000, 'p2': 15000, 'p3': 10000}
v23 = 2500
v24 = 80
v25 = v33.v25

def log(v34: v123) -> None:
    v35 = v34 if v34.v232('\n') else v34 + '\n'
    try:
        v233(v35, end='', flush=True)
    except v124:
        v233(v35.v356('ascii', 'replace').v318('ascii'), end='', flush=True)
    v3.v234.v125(parents=True, exist_ok=True)
    with v3.v235('a', encoding='utf-8') as v126:
        v126.v236(v35)

def pj(v36: v123) -> v26:
    return v0 / f'stage191_{v36}.json'

def save_phase(v36: v123, v37: v29) -> None:
    v37['timestamp'] = v341.v298(v342.v299).v127()
    v300(v36).v128(v249.v237(v37, indent=2, ensure_ascii=False), encoding='utf-8')

def phase0(v38: v129) -> None:
    if v300('p0').v130():
        v132('P0 done, skip')
        return
    v39 = v131.v131()
    v40 = 6000000 if v38 else v19
    v132(f'P0: reading {v40 // 1000000}M chars …')
    with v4.v235('r', encoding='utf-8', errors='ignore') as v126:
        v133 = v126.v238(v40)
    v41 = v134(v301(v133) | {' '})
    v6.v128(v249.v237(v41, ensure_ascii=False), encoding='utf-8')
    v132(f'  charset={v251(v41) + 1}')
    v42 = v191.v135(v123(v9))
    v43 = v42.v239(v18) or 0
    v44 = v133[:-v20] if v251(v133) > v20 * 2 else v133[:v251(v133) // 2]
    v45 = [v241.v240() for v241 in v44.v302('\n') if v251(v241.v240()) >= 120]
    v132(f'  lines={v251(v45)}; tokenizing …')
    v46 = v42.v136(v45)
    v62, v137 = ([], [0])
    for v47 in v46:
        v138 = [v195 for v195 in v47.v138 if v195 != v43]
        if v251(v138) >= 24:
            v62.v303(v138)
            v137.v288(v251(v62))
    v48 = v242.v139(v62, dtype=v242.v243)
    v49 = v242.v139(v137, dtype=v242.v244)
    v242.v140(v5, flat=v48, offsets=v49)
    v132(f'  docs={v251(v49) - 1} tokens={v251(v48)} ({v131.v131() - v39:.0f}s)')
    v50 = v133[-v20:]
    v51 = v42.v141()
    v52 = v242.v192(v48, minlength=v51).v245(v242.v246) + 1.0
    v247.v142, v247.v143, v247.v144 = (60, 30, 20) if v38 else (300, 150, 100)
    v53 = v248.v145(v10)
    v54 = v247.v146(v50, v42, v43, v52, v53)
    with v7.v235('w', encoding='utf-8') as v126:
        for v88 in v54:
            v126.v236(v249.v237(v88, ensure_ascii=False) + '\n')
    v55 = {v147: v184((1 for v195 in v54 if v195['type'] == v147)) for v147 in ('next_tok', 'entity', 'ood')}
    v56 = v242.v132(v52 / v52.v184())
    v57 = v247.v148(lambda v152, v304: v28(v242.v281([v56[v147] for v147 in v304])), v54)
    v58 = v248.v145(0)
    v59 = v247.v148(lambda v152, v304: v58.v248(), v54)
    v132(f"  exam v3 {v55} | unigram next_tok={v57['next_tok_acc']:.3f} random={v59['next_tok_acc']:.3f}")
    v149('p0', {'counts': v55, 'unigram': v57, 'random': v59, 'docs': v251(v49) - 1, 'tokens': v31(v251(v48)), 'charset': v251(v41) + 1})

def load_data():
    v60 = v242.v150(v5)
    v62, v63 = (v60['flat'], v60['offsets'])
    v41 = v249.v151(v6.v250(encoding='utf-8'))
    v61 = {v152: v195 + 1 for v195, v152 in v305(v41)}
    return (v62, v63, v61, v251(v41) + 1)

def sample_windows(v62, v63, v64, v53, v43):
    v65 = v251(v63) - 1
    v66 = v242.v101((v64, v13), v43, dtype=v242.v244)
    for v67 in v153(v64):
        v154 = v53.v252(0, v65 - 1)
        v199, v47 = (v63[v154], v63[v154 + 1])
        v155 = v47 - v199
        if v155 <= v13:
            v66[v67, :v155] = v62[v199:v47]
        else:
            v253 = v199 + v53.v252(0, v155 - v13)
            v66[v67] = v62[v253:v253 + v13]
    return v109.v156(v66)

def lr_at(v68, v69):
    if v68 < v16:
        return v15 * v68 / v16
    v70 = (v68 - v16) / v254(1, v69 - v16)
    return v15 * 0.5 * (1 + v343.v210(v343.v344 * v70))

class SelfModelXL(v71.v27):

    def __init__(v157, v158: v31, v51: v31, v154: v31=v11, v159: v31=v12, v110: v109.v30 | None=None, v160: v129=False):
        v345().v255()
        v157.v161 = v32.v256(v158, d=v154)
        v157.v162 = v32.v257(d=v154, n_layers=v159)
        v157.v163 = v33.v258(v154, v154)
        v157.v164 = v71.v259(2 * v154, v51, bias=False)
        v157.v160 = v160
        if v160:
            v157.v260 = v71.v306(v109.v168(4.0))
            v157.v261 = v71.v306(v109.v168(-2.0))
        if v110 is not None:
            v157.v307('rarity', v110)
            v157.v262 = v71.v259(1, v154)
        else:
            v157.v110 = None

    def _arcs(v157, v165, v138=None):
        v166 = v157.v161(v165)
        if v157.v110 is not None and v138 is not None:
            v166 = v166 + v157.v262(v157.v110[v138].v346(-1))
        return v166

    def forward_all(v157, v165, v81, v138=None):
        v166 = v157.v263(v165, v138)
        v162 = v157.v162(v166, pad_mask=v81)
        v163, v264, v265 = v157.v163(v166, v81)
        v82 = v157.v164(v109.v308([v162, v163], dim=-1))
        if v157.v160:
            v266 = 1.0 + v269.v362(v157.v260 * v264.v369() + v157.v261).v346(-1)
            v82 = v82 / v266
        return (v82, v264, v265)

def make_logits_fn(v72, v73, v43, v74: v129):
    """expose .logits(char_ids,pad) compatible with s185.span_logprob via wrapper obj"""

    class W:

        def eval(v157):
            v72.v180()
            return v157

        def logits(v157, v165, v81, v267=False):
            return v72.v273(v165, v81, ids=None)[0]
    return v167()

@v109.v84()
def span_logprob_x(v72, v73, v43, v75, v76, v77) -> v28:
    v78 = (v75 + v76)[-v13:]
    v79 = v251(v78) - v251(v76)
    v80 = v109.v168([v78], dtype=v109.v268, device=v77)
    v81 = v80 == v43
    v82 = v72.v273(v73[v80], v81, ids=v80)[0][0]
    v83 = v269.v169(v82, dim=-1)
    return v184((v28(v83[v79 + v224 - 1, v107]) for v224, v107 in v305(v76))) / v254(1, v251(v76))

@v109.v84()
def score_items(v85, v54, v86=None) -> v29:
    v87 = {}
    for v88 in v54:
        v147 = v88['type']
        if v86 and v147 != v86:
            continue
        v170 = [v85(v88['ctx_ids'], v152) for v152 in v88['cand_ids']]
        v214, v213 = v87.v270(v147, (0, 0))
        v87[v147] = (v214 + v31(v31(v242.v366(v170)) == v88['gold_idx']), v213 + 1)
    return {f'{v147}_acc': v214 / v254(1, v213) for v147, (v214, v213) in v87.v54()} | {f'{v147}_n': v213 for v147, (v214, v213) in v87.v54()}

def train_curve(v89, v72, v62, v63, v73, v43, v90, v77, v91, v92, v38):
    if v38:
        v91 = 120
    v93 = v109.v271.v171(v72.v272(), lr=v15, weight_decay=0.01)
    v53 = v248.v145(v10)
    v39 = v131.v131()
    v94 = -1.0
    v95 = 0
    v96 = 0
    v97 = None
    v72.v172()
    for v68 in v153(1, v91 + 1):
        for v173 in v93.v174:
            v173['lr'] = v309(v68, v91)
        v138 = v347(v62, v63, v14, v53, v43).v182(v77)
        v81 = v138 == v43
        v82, v264, v265 = v72.v273(v73[v138], v81, ids=v138)
        v175 = v138[:, 1:]
        v176 = ~v81[:, :-1] & ~v81[:, 1:]
        v177 = v269.v274(v82[:, :-1][v176], v175[v176])
        v178 = v177 + v17 * v265[~v81].v281()
        v93.v275(set_to_none=True)
        v178.v276()
        v71.v310.v277(v72.v272(), 1.0)
        v93.v68()
        v97 = v28(v177) if v97 is None else 0.95 * v97 + 0.05 * v28(v177)
        if v68 % (40 if v38 else v23) == 0 or v68 == v91:
            v72.v180()
            v278 = v186(lambda v152, v304: v315(v72, v73, v43, v152, v304, v77), v90, 'next_tok')
            v87 = v278.v270('next_tok_acc', 0)
            v279 = v131.v131() - v39
            v132(f'  [{v89}] step {v68}/{v91}: ce~{v97:.3f} next_tok(mid)={v87:.3f} ({v279:.0f}s)')
            if v87 > v94 + 1e-06:
                v94, v95, v96 = (v87, v68, 0)
                v109.v348({'model': v72.v363(), 'step': v68, 'mid': v87}, v2 / f'stage191_{v89}.pt')
            else:
                v96 += 1
            v72.v172()
            if v279 > v92:
                v132(f'  [{v89}] budget hit, stop')
                break
            if v96 >= 2 and v68 >= v91 // 2:
                v132(f'  [{v89}] early stop (flat)')
                break
    v98 = v109.v150(v2 / f'stage191_{v89}.pt', map_location=v77, weights_only=False)
    v72.v179(v98['model'])
    v72.v180()
    return {'best_mid': v94, 'best_step': v95, 'ce': v97, 'wall_s': v131.v131() - v39}

def phase1(v38, v77):
    if v300('p1').v130():
        v132('P1 done, skip')
        return
    v62, v63, v61, v158 = v181()
    v42 = v191.v135(v123(v9))
    v51 = v42.v141()
    v43 = v42.v239(v18) or 0
    v73 = v349.v311(v42, v61, v43, v51).v182(v77)
    v54 = [v249.v151(v241) for v241 in v7.v250(encoding='utf-8').v312()]
    v90 = [v88 for v88 in v54 if v88['type'] == 'next_tok'][:v24]
    v109.v183(v10)
    v72 = v313(v158, v51).v182(v77)
    v99 = v184((v70.v314() for v70 in v72.v272()))
    v132(f'P1 curve-XL d{v11}/L{v12} params={v99 / 1000000.0:.1f}M')
    v100 = v185('p1_curve', v72, v62, v63, v73, v43, v90, v77, v22['p1'], v21['p1'], v38)
    v101 = v186(lambda v152, v304: v315(v72, v73, v43, v152, v304, v77), v54)
    v132(f"  P1 FINAL: next_tok={v101.v270('next_tok_acc', 0):.3f} entity={v101.v270('entity_acc', 0):.3f} ood={v101.v270('ood_acc', 0):.3f}")
    v149('p1', {'train': v100, 'exam': v101, 'params_m': v99 / 1000000.0})

def phase2(v38, v77):
    if v300('p2').v130():
        v132('P2 done, skip')
        return
    v62, v63, v61, v158 = v181()
    v42 = v191.v135(v123(v9))
    v51 = v42.v141()
    v43 = v42.v239(v18) or 0
    v54 = [v249.v151(v241) for v241 in v7.v250(encoding='utf-8').v312()]
    v90 = [v88 for v88 in v54 if v88['type'] == 'next_tok'][:v24]
    v102 = v187(vocab_size=v51, n_positions=v13, n_embd=v11, n_layer=v12, n_head=8, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1)
    v109.v183(v10)
    v72 = v316(v102).v182(v77)
    v99 = v184((v70.v314() for v70 in v72.v272()))
    v132(f'P2 gpt-XL params={v99 / 1000000.0:.1f}M')

    @v109.v84()
    def gpt_span(v188, v189):
        v78 = (v188 + v189)[-v13:]
        v79 = v251(v78) - v251(v189)
        v80 = v109.v168([v78], device=v77)
        v83 = v269.v169(v72(input_ids=v80).v82[0], dim=-1)
        return v184((v28(v83[v79 + v224 - 1, v107]) for v224, v107 in v305(v189))) / v254(1, v251(v189))
    v91 = 120 if v38 else v22['p2']
    v93 = v109.v271.v171(v72.v272(), lr=v15, weight_decay=0.01)
    v53 = v248.v145(v10)
    v39 = v131.v131()
    v94, v95, v96, v97 = (-1.0, 0, 0, None)
    v72.v172()
    for v68 in v153(1, v91 + 1):
        for v173 in v93.v174:
            v173['lr'] = v309(v68, v91)
        v138 = v347(v62, v63, v14, v53, v43).v182(v77)
        v114 = v72(input_ids=v138, labels=v138)
        v178 = v114.v178
        v93.v275(set_to_none=True)
        v178.v276()
        v71.v310.v277(v72.v272(), 1.0)
        v93.v68()
        v97 = v28(v178) if v97 is None else 0.95 * v97 + 0.05 * v28(v178)
        if v68 % (40 if v38 else v23) == 0 or v68 == v91:
            v72.v180()
            v278 = v186(v190, v90, 'next_tok')
            v87 = v278.v270('next_tok_acc', 0)
            v279 = v131.v131() - v39
            v132(f'  [p2_gpt] step {v68}/{v91}: ce~{v97:.3f} next_tok(mid)={v87:.3f} ({v279:.0f}s)')
            if v87 > v94 + 1e-06:
                v94, v95, v96 = (v87, v68, 0)
                v109.v348({'model': v72.v363(), 'conf': v102.v364(), 'step': v68}, v2 / 'stage191_p2_gpt.pt')
            else:
                v96 += 1
            v72.v172()
            if v279 > v21['p2'] or (v96 >= 2 and v68 >= v91 // 2):
                v132('  [p2_gpt] stop')
                break
    v98 = v109.v150(v2 / 'stage191_p2_gpt.pt', map_location=v77, weights_only=False)
    v72.v179(v98['model'])
    v72.v180()
    v101 = v186(v190, v54)
    v132(f"  P2 FINAL: next_tok={v101.v270('next_tok_acc', 0):.3f} entity={v101.v270('entity_acc', 0):.3f} ood={v101.v270('ood_acc', 0):.3f}")
    v149('p2', {'best_mid': v94, 'best_step': v95, 'exam': v101, 'params_m': v99 / 1000000.0})

def build_rarity(v42: v191, v51: v31, v62, v77) -> v109.v30:
    """z-scored char-trigram novelty per token id."""
    from collections import Counter
    v103: v104 = v104()
    v105 = v62[:v317(v251(v62), 3000000)]
    v106 = [v42.v318([v31(v147)], skip_special_tokens=False) or '' for v147 in v242.v193(v105)]
    v52 = v242.v192(v105, minlength=v51)
    for v107 in v242.v193(v105):
        v70 = v42.v318([v31(v107)], skip_special_tokens=False) or ''
        v194 = v31(v52[v107])
        for v195 in v153(v251(v70) - 2):
            v103[v70[v195:v195 + 3]] += v194
    v69 = v184(v103.v319()) or 1
    v108 = v242.v196(v51, dtype=v242.v280)
    for v107 in v153(v51):
        v70 = v42.v318([v31(v107)], skip_special_tokens=False) or ''
        v197 = [v70[v195:v195 + 3] for v195 in v153(v251(v70) - 2)]
        if not v197:
            v108[v107] = 0.0
            continue
        v108[v107] = v28(v242.v281([-v242.v132((v103.v270(v147, 0) + 1) / (v69 + 1)) for v147 in v197]))
    v198, v199 = (v108.v281(), v108.v320() + 1e-06)
    return v109.v168((v108 - v198) / v199, device=v77)

def phase3(v38, v77):
    if v300('p3').v130():
        v132('P3 done, skip')
        return
    v62, v63, v61, v158 = v181()
    v42 = v191.v135(v123(v9))
    v51 = v42.v141()
    v43 = v42.v239(v18) or 0
    v73 = v349.v311(v42, v61, v43, v51).v182(v77)
    v54 = [v249.v151(v241) for v241 in v7.v250(encoding='utf-8').v312()]
    v90 = [v88 for v88 in v54 if v88['type'] == 'next_tok'][:v24]
    v132('P3: building rarity table …')
    v110 = v200(v42, v51, v62, v77)
    v109.v183(v10)
    v72 = v313(v158, v51, rarity=v110, surprise_temp=True).v182(v77)
    v100 = v185('p3_rarity', v72, v62, v63, v73, v43, v90, v77, v22['p3'], v21['p3'], v38)
    v101 = v186(lambda v152, v304: v315(v72, v73, v43, v152, v304, v77), v54)
    v111 = [v88 for v88 in v54 if v88['type'] == 'entity'][:80]
    v112 = v248.v145(3)
    v201, v202, v203, v204 = ([], [], [], [])

    @v109.v84()
    def probe(v188, v205):
        v78 = (v188 + v205)[-v13:]
        v79 = v251(v78) - v251(v205)
        v80 = v109.v168([v78], dtype=v109.v268, device=v77)
        v81 = v80 == v43
        v82, v264, v215 = v72.v273(v73[v80], v81, ids=v80)
        v70 = v269.v282(v82[0, v251(v78) - 1], dim=-1)
        return (v28(-(v70 * v109.v132(v70 + 1e-09)).v184()), v28(v264[0, v79:].v281()))
    for v88 in v111:
        v206 = v88['cand_ids'][v88['gold_idx']]
        v207 = v25[v112.v252(0, v251(v25) - 1)]
        v208 = [v195 for v195 in v42.v356(' ' + v207).v138 if v195 != v43]
        v283, v284 = v285(v88['ctx_ids'], v206)
        v286, v287 = v285(v88['ctx_ids'], v208)
        v201.v288(v283)
        v202.v288(v286)
        v203.v288(v284)
        v204.v288(v287)
    v113 = {'entropy_real': v28(v242.v281(v201)), 'entropy_fake': v28(v242.v281(v202)), 'surprise_real': v28(v242.v281(v203)), 'surprise_fake': v28(v242.v281(v204)), 'entropy_ok': v28(v242.v281(v202)) > v28(v242.v281(v201)), 'surprise_ok': v28(v242.v281(v204)) > v28(v242.v281(v203))}
    v132(f"  P3 FINAL: next_tok={v101.v270('next_tok_acc', 0):.3f} | G3 {v113}")
    v149('p3', {'train': v100, 'exam': v101, 'g3': v113})

def phase4(v38, v77):
    if v300('p4').v130():
        v132('P4 done, skip')
        return
    v62, v63, v61, v158 = v181()
    v42 = v191.v135(v123(v9))
    v51 = v42.v141()
    v43 = v42.v239(v18) or 0
    v73 = v349.v311(v42, v61, v43, v51).v182(v77)

    def curve_z(v72):

        @v109.v84()
        def z_of_ids(v289):
            v80 = v109.v168([v289[-v13:]], dtype=v109.v268, device=v77)
            v81 = v80 == v43
            v166 = v72.v263(v73[v80], v80)
            v163, v215, v215 = v72.v163(v166, v81)
            v155 = v31((~v81).v184())
            return v163[0, v155 - 1]
        return v209

    @v109.v84()
    def gate_B(v209):

        def z_text(v147):
            return v209([v195 for v195 in v42.v356(v147).v138 if v195 != v43])
        v210 = lambda v291, v67: v28(v269.v350(v291, v67, dim=-1))
        v211 = [v210(v351(v291), v351(v67)) for v291, v67 in v352.v321]
        v212 = [v210(v351(v291), v351(v67)) for v291, v67 in v352.v322]
        return {'para': v28(v242.v281(v211)), 'hard': v28(v242.v281(v212)), 'gap': v28(v242.v281(v212) - v242.v281(v211))}

    @v109.v84()
    def doclink(v209, v213=80):
        v53 = v248.v145(7)
        v65 = v251(v63) - 1
        v214 = 0
        for v215 in v153(v213):
            v323, v324 = (v53.v252(0, v65 - 1), v53.v252(0, v65 - 1))
            v325, v326 = (v63[v323], v63[v323 + 1])
            v327, v328 = (v63[v324], v63[v324 + 1])
            if v326 - v325 < v13 + 16 or v328 - v327 < v13:
                continue
            v290 = (v325 + v326) // 2
            v291 = v62[v325:v317(v325 + v13, v290)].v329()
            v67 = v62[v290:v290 + v13].v329()
            v152 = v62[v327:v327 + v13].v329()
            v330, v331, v332 = (v209(v291), v209(v67), v209(v152))
            v214 += v31(v28(v269.v350(v330, v331, dim=-1)) > v28(v269.v350(v330, v332, dim=-1)))
        return v214 / v254(1, v213)
    v114 = {}
    for v89, v216, v217 in (('p1_curve', v2 / 'stage191_p1_curve.pt', {}), ('p3_rarity', v2 / 'stage191_p3_rarity.pt', {'surprise_temp': True, 'need_rarity': True})):
        if not v216.v130():
            continue
        v110 = v200(v42, v51, v62, v77) if v217.v270('need_rarity') else None
        v198 = v313(v158, v51, rarity=v110, surprise_temp=v217.v270('surprise_temp', False)).v182(v77)
        v198.v179(v109.v150(v216, map_location=v77, weights_only=False)['model'])
        v198.v180()
        v60 = v292(v198)
        v114[v89] = {'gateB': v333(v60), 'doclink': v334(v60)}
        v132(f'  P4 {v89}: {v114[v89]}')
    v115 = v2 / 'stage191_p2_gpt.pt'
    if v115.v130():
        v98 = v109.v150(v115, map_location=v77, weights_only=False)
        v218 = v316(v187(**v98['conf'])).v182(v77)
        v218.v179(v98['model'])
        v218.v180()

        @v109.v84()
        def gpt_z(v289):
            v80 = v109.v168([v289[-v13:]], device=v77)
            v293 = v218.v357(input_ids=v80).v335[0]
            return v293.v281(dim=0)
        v114['p2_gpt'] = {'gateB': v333(v336), 'doclink': v334(v336)}
        v132(f"  P4 p2_gpt: {v114['p2_gpt']}")
    v116 = v2 / 'stage187_self_model.pt'
    if v116.v130() and (not v38):
        import _stage170_curve_dynamics as s170
        v219 = v337.v294(max_chars=20000000)
        v220 = v134(v301(v219) | {' '})
        v221 = {v152: v195 + 1 for v195, v152 in v305(v220)}
        v222 = v349.v311(v42, v221, v43, v51).v182(v77)
        v223 = v33.v353(v251(v220) + 1, v51).v182(v77)
        v223.v179(v109.v150(v116, map_location=v77, weights_only=False)['model'])
        v223.v180()

        @v109.v84()
        def z_old(v289):
            v80 = v109.v168([v289[-v13:]], dtype=v109.v268, device=v77)
            v81 = v80 == v43
            v166 = v223.v161(v222[v80])
            v163, v215, v215 = v223.v163(v166, v81)
            v155 = v31((~v81).v184())
            return v163[0, v155 - 1]
        v114['old_187_d128_2M'] = {'gateB': v333(v338), 'doclink': v334(v338)}
        v132(f"  P4 old_187: {v114['old_187_d128_2M']}")
    v149('p4', v114)

def phase5(v38):
    v70 = {v224: v249.v151(v300(v224).v250(encoding='utf-8')) for v224 in ('p0', 'p1', 'p2', 'p3', 'p4') if v300(v224).v130()}
    v117 = []
    if 'p1' in v70 and 'p2' in v70:
        v152, v173 = (v70['p1']['exam'].v270('next_tok_acc', 0), v70['p2']['exam'].v270('next_tok_acc', 0))
        v154 = v152 - v173
        v117.v288('NIGHT_PARITY_HELD' if v358(v154) <= 0.03 else 'NIGHT_CURVE_AHEAD' if v154 > 0 else 'NIGHT_GPT_AHEAD')
    if 'p3' in v70:
        v113 = v70['p3']['g3']
        if v113['entropy_ok'] and v113['surprise_ok']:
            v117.v288('NIGHT_G3_FIXED')
        elif v113['surprise_ok']:
            v117.v288('NIGHT_G3_SURPRISE_ONLY')
    if 'p4' in v70 and 'old_187_d128_2M' in v70['p4']:
        v225 = {v224: v227['gateB']['gap'] for v224, v227 in v70['p4'].v54() if v359(v227, v29) and 'gateB' in v227}
        v226 = v225.v270('old_187_d128_2M')
        if v226 is not None and v339((v227 < v226 - 0.01 for v224, v227 in v225.v54() if v224 != 'old_187_d128_2M')):
            v117.v288('NIGHT_MEANING_MOVES')
    v45 = [f'# Stage191 night report ({v341.v298(v342.v299).v127()})', '', f"**Verdicts:** {', '.v295(v117) or 'incomplete'}", '']
    for v224, v227 in v70.v54():
        v45.v288(f'## {v224}')
        v45.v288('```json')
        v45.v288(v249.v237(v227, indent=2, ensure_ascii=False)[:3000])
        v45.v288('```')
    v8.v128('\n'.v295(v45), encoding='utf-8')
    v149('p5', {'verdicts': v117})
    v132(f"[191] NIGHT DONE: {', '.v295(v117) or 'incomplete'}")

def main() -> v31:
    v118 = v296.v228()
    v118.v229('--phase', default='all')
    v118.v229('--smoke', action='store_true')
    v118.v229('--device', default='cuda' if v109.v360.v354() else 'cpu')
    v119 = v118.v230()
    v77 = v109.v77(v119.v77)
    v0.v125(parents=True, exist_ok=True)
    v2.v125(parents=True, exist_ok=True)
    v132(f'Stage191 start {v341.v298(v342.v299).v127()} phase={v119.v36} smoke={v119.v38}')
    v120 = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5'] if v119.v36 == 'all' else [v119.v36]
    for v121 in v120:
        if v121 == 'p0':
            v340(v119.v38)
        elif v121 == 'p1':
            v355(v119.v38, v77)
        elif v121 == 'p2':
            v361(v119.v38, v77)
        elif v121 == 'p3':
            v365(v119.v38, v77)
        elif v121 == 'p4':
            v367(v119.v38, v77)
        elif v121 == 'p5':
            v368(v119.v38)
    return 0
if v122 == '__main__':
    raise v231(v297())