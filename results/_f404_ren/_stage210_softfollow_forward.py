"""
Stage 210 — structured hop composition INSIDE forward, answer via CE tokens.

Extends 203: SoftFollow / HopReader retrieve over frozen fp tape, but inject retrieved
vector as one memory-arc into P1's arc stream and read answer with span_logprob (tokens),
not cosine in fp-space.

Trainable (P1 frozen): reader (SoftFollow or HopReader), inject_proj (zero-init), log_gate.
Anti-CF: encoder untouched; tape K/V non-gradient.

Gates (pre_publish_frontier.md):
  G1 soft_follow_token test k2>=0.70 k3>=0.60
  G2 free_form overfits train>=0.90 test<=0.45
  G3 next_tok delta vs P1 <= 0.01 with inject path (mem_arc=0 on generic text)
  G4 gate=0 bit-identical logits vs P1
  G5 no_memory (gate=0 on hop task) <= 0.35

  python _stage210_softfollow_forward.py
"""
from __future__ import annotations
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage203_internal_hops as s203
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
v0 = v25('results')
v1 = v25('checkpoints/stage191_p1_curve.pt')
v2 = v25('checkpoints/stage191_p2_gpt.pt')
v3 = v25('data/_wikitext103_train.txt')
v4 = v25('data/stage191_exam_v3.jsonl')
v5 = v0 / 'stage210_decision.json'
v6 = v0 / 'stage210_mini.md'
v7 = v0 / '_stage210_log.txt'
v8 = 210
v9 = 40000000
v10 = 90000000
v11 = 240
v12 = 4
v13 = 4000
v14 = 3
v15 = 1500
v16 = 800
v17 = 128
v18 = 8
v19 = 0.001
v20 = 'Chain start {a0} hops {k} answer '

def log(v26: v118) -> None:
    v27 = v26 if v26.v206('\n') else v26 + '\n'
    try:
        v207(v27, end='', flush=True)
    except v119:
        v207(v27.v278('ascii', 'replace').v276('ascii'), end='', flush=True)
    v7.v208.v120(parents=True, exist_ok=True)
    with v7.v209('a', encoding='utf-8') as v121:
        v121.v210(v27)

class InjectPack(v28.v21):
    """Reader + zero-init inject into arc stream; gate=0 => zero mem arc."""

    def __init__(v122, v123: v28.v21, v79: v24):
        v277().v211()
        v122.v123 = v123
        v122.v124 = v28.v212(v79, v79, bias=False)
        v28.v253.v213(v122.v124.v214)
        v122.v124.v214.v254.v215(0.02)
        v122.v125 = v28.v216(v60.v141(-2.0))

    def mem_arc(v122, v126, v43, v64, v65, v127: v130=False):
        if v127:
            v217 = v60.v255(v126)
            return (v217, 0.0)
        v128 = v122.v123(v126, v43, v64, v65)
        v129 = v60.v218(v122.v125)
        return (v129 * v122.v124(v128), v129)

def forward_inject_logits(v29, v30, v31: v60.v23, v32: v24, v33: v60.v23, v34: v24, v35: v22, v36: v130=False):
    """bpe_ids [1,L], mem_arc [1,d]; insert mem at inject_pos when gate>0 or always_insert."""
    v37 = v31 == v32
    if not v36 and v35 <= 1e-06:
        return v29.v240(v30[v31], v37, ids=v31)[0]
    v38 = v29.v131(v30[v31], v31)
    v39 = v33.v132(1)
    v38 = v60.v133([v38[:, :v34], v39, v38[:, v34:]], dim=1)
    v37 = v60.v133([v37[:, :v34], v60.v160(1, 1, dtype=v60.v130, device=v37.v52), v37[:, v34:]], dim=1)
    v40 = v29.v40(v38, pad_mask=v37)
    v134, v135, v135 = v29.v134(v38, v37)
    return v29.v136(v60.v133([v40, v134], dim=-1))

def encode_query(v41: v137, v32: v24, v42: v118, v43: v24) -> v45[v24]:
    v44 = v20.v138(a0=v42, k=v43)
    return [v97 for v97 in v41.v278(v44).v219 if v97 != v32]

def encode_word(v41: v137, v32: v24, v46: v118) -> v45[v24]:
    return [v97 for v97 in v41.v278(' ' + v46).v219 if v97 != v32]

def pack_ctx_cand(v47: v45[v24], v48: v45[v24]) -> v51[v45[v24], v24]:
    """Fit in MAX_ARCS-1 tokens (+1 inject arc). Return seq, inject_pos."""
    v49 = (v47 + v48)[-(v279 - 1):]
    v50 = v139(v220(v48), v220(v49))
    v34 = v220(v49) - v50
    return (v49, v34)

def span_logprob_inject(v29, v30, v32: v24, v47: v45[v24], v48: v45[v24], v33: v60.v23, v35: v22, v52, v36: v130=False) -> v22:
    v49, v34 = v140(v47, v48)
    v50 = v220(v49) - v34
    v53 = v60.v141([v49], dtype=v60.v221, device=v52)
    v54 = v22(v35) if not v256(v35, v60.v23) else v22(v35.v162())
    v55 = v161(v29, v30, v53, v32, v33, v34, v54, always_insert=v36)[0]
    v56 = v222.v142(v55, dim=-1)
    v57 = 0.0
    for v58 in v143(v50):
        v144 = v49[v34 + v58]
        v72 = v34 + v58 if v36 or v54 > 1e-06 else v34 + v58 - 1
        if v72 < 0:
            continue
        v57 += v22(v56[v72, v144])
    return v57 / v223(1, v50)

def span_logprob_inject_train(v29, v30, v32: v24, v47: v45[v24], v48: v45[v24], v33: v60.v23, v35: v22, v52, v36: v130=True) -> v60.v23:
    v49, v34 = v140(v47, v48)
    v50 = v220(v49) - v34
    v53 = v60.v141([v49], dtype=v60.v221, device=v52)
    v54 = v22(v35) if not v256(v35, v60.v23) else v22(v35.v257())
    v55 = v161(v29, v30, v53, v32, v33, v34, v54, always_insert=v36)[0]
    v56 = v222.v142(v55, dim=-1)
    v59 = v60.v141(0.0, device=v52)
    for v58 in v143(v50):
        v144 = v49[v34 + v58]
        v72 = v34 + v58 if v36 or v54 > 1e-06 else v34 + v58 - 1
        if v72 >= 0:
            v59 = v59 - v56[v72, v144]
    return v59 / v223(1, v50)

def train_reader_fp(v61: v145, v62, v63, v64, v65, v52, v66):
    """Phase A: same fp CE as 203 (cheap — no P1 forward in loop)."""
    v67 = v60.v224.v146(v61.v123.v154(), lr=v19, weight_decay=0.01)
    v61.v123.v147()
    v68 = None
    for v69 in v143(1, v15 + 1):
        v148 = [v62[v66.v280(0, v220(v62) - 1)] for v135 in v143(v17)]
        v149 = v63.v190([v281[0] for v281 in v148])
        v150 = v60.v141([v281[1] for v281 in v148], device=v52)
        v151 = v63.v190([v281[2] for v281 in v148])
        v152 = v61.v123(v149, v150, v64, v65)
        v59 = v222.v225(v152 @ v151.v282 / 0.1, v60.v258(v220(v148), device=v52))
        v67.v226(set_to_none=True)
        v59.v227()
        v67.v69()
        if v69 % 500 == 0:
            v170(f'  reader step {v69} loss~{v22(v59):.3f}')
        v68 = v22(v59) if v68 is None else 0.98 * v68 + 0.02 * v22(v59)
    v61.v123.v153()
    return v68

def train_inject_only(v61: v145, v62, v63, v64, v65, v41, v32, v29, v30, v52, v66):
    """Phase B: frozen reader; train inject_proj + log_gate through P1 (small batch)."""
    for v70 in v61.v123.v154():
        v70.v228(False)
    v67 = v60.v224.v146([v61.v124.v214, v61.v125], lr=v19, weight_decay=0.01)
    v61.v153()
    v68 = None
    for v69 in v143(1, v16 + 1):
        v148 = [v62[v66.v280(0, v220(v62) - 1)] for v135 in v143(v18)]
        v59 = v60.v141(0.0, device=v52)
        for v42, v43, v229 in v148:
            v149 = v63.v190([v42]).v257()
            v150 = v60.v141([v43], device=v52)
            v259, v129 = v61.v33(v149, v150, v64, v65)
            v230 = v260(v41, v32, v42, v43)
            v231 = v261(v41, v32, v229)
            v59 = v59 + v283(v29, v30, v32, v230, v231, v259, v129, v52, True)
        v59 = v59 / v18
        v67.v226(set_to_none=True)
        v59.v227()
        v67.v69()
        if v69 % 50 == 0:
            v170(f'  inject step {v69} loss~{v22(v59):.3f} gate={v22(v60.v218(v61.v125)):.3f}')
        v68 = v22(v59) if v68 is None else 0.98 * v68 + 0.02 * v22(v59)
    return v68

def train_pack(v61: v145, v62, v63, v64, v65, v41, v32, v29, v30, v52, v66):
    v71 = v155(v61, v62, v63, v64, v65, v52, v66)
    v72 = v156(v61, v62, v63, v64, v65, v41, v32, v29, v30, v52, v66)
    return (v71, v72)

@v60.v77()
def eval_token_pack(v61, v73, v63, v64, v65, v41, v32, v29, v30, v52, v66, v74, v75: v130):
    v76 = {}
    for v43 in v143(1, v12):
        v86 = [v232 for v232 in v73 if v232[1] == v43]
        v157 = 0
        for v42, v135, v229 in v86:
            v149 = v63.v190([v42])
            v150 = v60.v141([v43], device=v52)
            v259, v129 = v61.v33(v149, v150, v64, v65, force_zero=v75)
            v230 = v260(v41, v32, v42, v43)
            v166 = [v229] + [v74[v66.v280(0, v220(v74) - 1)] for v135 in v143(3)]
            v233 = v45(v143(4))
            v66.v159(v233)
            v234 = [v166[v97] for v97 in v233]
            v235 = v233.v262(0)
            v236 = [v265(v29, v30, v32, v230, v261(v41, v32, v96), v259, v129, v52, always_insert=not v75) for v96 in v234]
            v157 += v24(v24(v291.v285(v236)) == v235)
        v76[v43] = v157 / v223(1, v220(v86))
    return v76

@v60.v77()
def hand_loop_cosine(v73, v63, v64, v65, v66, v74, v78):
    v76 = {}
    for v43 in v143(1, v12):
        v86 = [v232 for v232 in v73 if v232[1] == v43]
        v157 = 0
        for v42, v135, v229 in v86:
            v237 = v63.v190([v42])[0]
            for v135 in v143(v43):
                v237 = v65[v24((v64 @ v237).v285())]
            v238 = v74[v24((v78 @ v237).v285())]
            v157 += v24(v238 == v229)
        v76[v43] = v157 / v223(1, v220(v86))
    return v76

@v60.v77()
def bit_identity_check(v29, v30, v32, v52, v79):
    v66 = v239.v158(0)
    v80 = v45(v143(100, 200))
    v66.v159(v80)
    v49 = v80[:32]
    v53 = v60.v141([v49], dtype=v60.v221, device=v52)
    v37 = v53 == v32
    v81 = v29.v240(v30[v53], v37, ids=v53)[0]
    v82 = v60.v160(1, v79, device=v52)
    v83 = v161(v29, v30, v53, v32, v82, inject_pos=16, gate=0.0)
    v84 = (v81 - v83).v248().v223().v162()
    return (v84 < 1e-05, v84)

@v60.v77()
def next_tok_slice(v29, v30, v32, v52, v85=80):
    if not v4.v241():
        return (None, None)
    v86 = []
    with v4.v209('r', encoding='utf-8') as v121:
        for v27 in v121:
            v86.v246(v275.v284(v27))
    v86 = [v87 for v87 in v86 if v87.v273('type') == 'next_tok'][:v85]
    if not v86:
        return (None, None)
    v163, v164, v85 = (0, 0, 0)
    v79 = v29.v136.v165 // 2
    v82 = v60.v160(1, v79, device=v52)
    for v87 in v86:
        v166 = v87['cand_ids']
        v167 = [v263(v29, v30, v32, v87['ctx_ids'], v264, v52) for v264 in v166]
        v168 = [v265(v29, v30, v32, v87['ctx_ids'], v264, v82, 0.0, v52) for v264 in v166]
        v163 += v24(v291.v285(v167) == v87['gold_idx'])
        v164 += v24(v291.v285(v168) == v87['gold_idx'])
        v85 += 1
    return (v163 / v85, v164 / v85)

def main() -> v24:
    v0.v120(parents=True, exist_ok=True)
    v7.v169('', encoding='utf-8')
    v170(f'Stage210 start {v295.v289(v296.v290).v249()}')
    v52 = v60.v52('cuda' if v60.v286.v266() else 'cpu')
    v66 = v239.v158(v8)
    v60.v171(v8)
    v88 = v172.v172()
    v173, v174, v175, v176 = v177()
    v41 = v137.v178(v118(v267.v242))
    v89 = v41.v179()
    v32 = v41.v243(v244) or 0
    v30 = v287.v268(v41, v175, v32, v89).v180(v52)
    v29 = v269(v176, v89).v180(v52)
    v29.v181(v60.v270(v1, map_location=v52, weights_only=False)['model'])
    v29.v153()
    for v70 in v29.v154():
        v70.v228(False)
    v79 = v29.v136.v165 // 2
    v90 = v182((v22(v70.v248().v182()) for v70 in v29.v292.v154()))
    v63 = v183(v29, v175, v52)
    v170(f'P1 frozen d={v79} ({v172.v172() - v88:.0f}s)')
    with v3.v209('r', encoding='utf-8', errors='ignore') as v121:
        v44 = v121.v271(v9)[v10 % v9:]
    v91 = v45(v272.v247((v294.v293(1) for v294 in v300.v297(v44) if v220(v294.v293(1)) >= 4)))[:3000]
    del v44
    v92 = v184(v245(), v66, v11 * v12 + 50)
    v93 = [v92[v97 * v12:(v97 + 1) * v12] for v97 in v143(v11)]
    v93 = [v96 for v96 in v93 if v220(v96) == v12]
    v66.v159(v93)
    v94 = v93[:v24(0.75 * v220(v93))]
    v95 = v93[v24(0.75 * v220(v93)):]
    v185, v186 = ([], [])
    for v96 in v93:
        for v97 in v143(v220(v96) - 1):
            v185.v246(v63.v190([v96[v97]])[0])
            v186.v246(v63.v190([v96[v97 + 1]])[0])
    for v97 in v143(v139(v13, v220(v91) - 1)):
        v185.v246(v63.v190([v91[v97]])[0])
        v186.v246(v63.v190([v91[v97 + 1]])[0])
    v64 = v60.v187(v185, 0)
    v65 = v60.v187(v186, 0)

    def samples(v188):
        v116 = []
        for v96 in v188:
            for v43 in v143(1, v12):
                v116.v246((v96[0], v43, v96[v43]))
        return v116
    v62 = v189(v94)
    v98 = v189(v95)
    v74 = v45(v272.v247([v85 for v96 in v93 for v85 in v96]))
    v78 = v63.v190(v74)
    v170(f'tape={v64.v288[0]} train/test chains={v220(v94)}/{v220(v95)} ({v172.v172() - v88:.0f}s)')
    v99 = v145(v301.v298(v79, v14).v180(v52), v79).v180(v52)
    v191, v192 = v193(v99, v62, v63, v64, v65, v41, v32, v29, v30, v52, v239.v158(v8))
    v170(f'soft_follow reader~{v191:.3f} inject~{v192:.3f} gate={v22(v60.v218(v99.v125)):.3f}')
    v100 = v145(v301.v299(v79, v14, v12).v180(v52), v79).v180(v52)
    v194, v195 = v193(v100, v62, v63, v64, v65, v41, v32, v29, v30, v52, v239.v158(v8 + 1))
    v170(f'free_form reader~{v194:.3f} inject~{v195:.3f}')
    v101 = v239.v158(v8 + 2)
    v102 = v196(v99, v62, v63, v64, v65, v41, v32, v29, v30, v52, v101, v74, False)
    v103 = v196(v99, v98, v63, v64, v65, v41, v32, v29, v30, v52, v239.v158(v8 + 3), v74, False)
    v104 = v196(v100, v62, v63, v64, v65, v41, v32, v29, v30, v52, v239.v158(v8 + 4), v74, False)
    v105 = v196(v100, v98, v63, v64, v65, v41, v32, v29, v30, v52, v239.v158(v8 + 5), v74, False)
    v106 = v196(v99, v98, v63, v64, v65, v41, v32, v29, v30, v52, v239.v158(v8 + 6), v74, gate_zero=True)
    v107 = v197(v98, v63, v64, v65, v239.v158(v8 + 7), v74, v78)
    v108 = v182((v22(v70.v248().v182()) for v70 in v29.v292.v154()))
    v109 = v248(v90 - v108) < 0.001
    v198, v199 = v200(v29, v30, v32, v52, v79)
    v201, v202 = v203(v29, v30, v32, v52)
    v110 = v248(v201 - v202) if v201 is not None else None
    v111 = v103.v273(2, 0) >= 0.7 and v103.v273(3, 0) >= 0.6
    v112 = v139(v104.v274()) >= 0.9 and v223(v105.v274()) <= 0.45
    v113 = v110 is None or v110 <= 0.01
    v114 = v198 and v109
    v115 = v223(v106.v274()) <= 0.35
    if v111 and v112 and v113 and v114 and v115:
        v204 = 'THESIS_YES'
    elif v111 and v114 and v115:
        v204 = 'ENGINEERING_ONLY'
    else:
        v204 = 'THESIS_NO_AT_SCALE'
    v170(f'soft token test k2={v103.v273(2):.3f} k3={v103.v273(3):.3f}')
    v170(f'free  token test k2={v105.v273(2):.3f} k3={v105.v273(3):.3f} train k2={v104.v273(2):.3f}')
    v170(f'no_mem test max={v223(v106.v274()):.3f} external cosine={v107}')
    v170(f'next_tok base={v201} inject_mem0={v202} delta={v110} bit_diff={v199}')
    v116 = {'timestamp': v295.v289(v296.v290).v249(), 'protocol': 'softfollow_forward_tokens_210', 'overall': v204, 'soft_follow_token': {'train': v102, 'test': v103}, 'free_form_token': {'train': v104, 'test': v105}, 'no_memory_test': v106, 'external_loop_cosine_test': v107, 'next_tok': {'p1_acc': v201, 'inject_mem0_acc': v202, 'delta': v110}, 'bit_identity_max_diff': v199, 'gates': {'g1_generalize': v111, 'g2_structure': v112, 'g3_no_ce_cost': v113, 'g4_bit_identity': v114, 'g5_needs_memory': v115}, 'anticf_encoder_frozen': v109, 'chance': 0.25, 'interpretation': 'THESIS_YES = structured hops inside forward answer via CE tokens without P1 grad'}
    v5.v169(v275.v250(v116, indent=2, ensure_ascii=False), encoding='utf-8')
    v6.v169('\n'.v251(['# Stage210 — SoftFollow in forward, token answers', '', f'**Overall:** `{v204}`', '', f'- soft-follow token test: k2={v103.v273(2):.3f} k3={v103.v273(3):.3f}', f'- free-form token test: k2={v105.v273(2):.3f} k3={v105.v273(3):.3f} (train k2={v104.v273(2):.3f})', f'- no_memory max={v223(v106.v274()):.3f} | external cosine {v107}', f'- next_tok delta={v110} bit_diff={v199}', f"- gates: {v116['gates']}"]), encoding='utf-8')
    v170(f'[210] {v204} ({v172.v172() - v88:.0f}s)')
    return 0
if v117 == '__main__':
    raise v205(v252())