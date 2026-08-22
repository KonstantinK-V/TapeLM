"""
Stage 182 — Hybrid: dual-channel tape + CE on SLOW («какой следующий id куска?»).

Matches what we want:
  - FAST/SLOW draw the BPE tape (ink + write-budget memory)
  - From SLOW state at t, predict next BPE piece id (CE) — GPT-like use of context
  - FAST keeps weak local ink (optional, low weight)

Same tokenizer/corpus/scale as 180/181.
Gates: A/B on slow (+ combined); prefix-shuffle ablation on slow-CE (like 181).

  python _stage182_slow_ce_tape.py
  python _stage182_slow_ce_tape.py --steps 10000
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage178_curve_retention as s178
import _stage179_curve_harden_B as s179
import _stage180_dual_channel as s180
import _stage181_ce_control as s181
v0 = v24('results')
v1 = v24('checkpoints')
v2 = v0 / '_stage182_log.txt'
v3 = v0 / 'stage182_decision.json'
v4 = v0 / 'stage182_mini.md'
v5 = v1 / 'stage182_slow_ce_tape.pt'
v6 = v25.v6
v7 = v0 / 'plan_curve_dynamics.md'
v8 = v0 / 'stage181_decision.json'
v9 = 182
v10 = v26.v10
v11 = v26.v11
v12 = v25.v12
v13 = 16
v14 = 0.0003
v15 = 1500
v16 = 10000
v17 = 1.0
v18 = 0.15
v19 = 0.3
v20 = '[PAD]'

def log(v27: v99) -> None:
    v28 = v27 if v27.v188('\n') else v27 + '\n'
    try:
        v189(v28, end='', flush=True)
    except v100:
        v189(v28.v287('ascii', 'replace').v269('ascii'), end='', flush=True)
    v2.v190.v101(parents=True, exist_ok=True)
    with v2.v191('a', encoding='utf-8') as v102:
        v102.v192(v28)

def write_json(v29: v24, v30: v103) -> None:
    v29.v190.v101(parents=True, exist_ok=True)
    v29.v104(v246.v193(v30, indent=2, ensure_ascii=False), encoding='utf-8')

class DualSlowCE(v31.v21):

    def __init__(v105, v106: v23, v107: v23):
        v270().v194()
        v105.v108 = v26.v195(v106)
        v105.v109 = v31.v196(v11, v107, bias=False)

    def forward_channels(v105, v48, v47, v110=None):
        return v105.v108.v128(v48, v47, inst_prefix=v110)

    def logits_from_slow(v105, v111: v39.v22) -> v39.v22:
        return v105.v109(v111)

def ids_to_char_batch(v32: v112, v33: v39.v22, v34: v103, v35: v23) -> v39.v22:
    """[B,A] token ids → [B,A,C] char ids for arc ink encoder."""
    v113, v114 = v33.v36
    v37 = []
    for v38 in v115(v113):
        v116 = []
        for v117 in v115(v114):
            v197 = v23(v33[v38, v117].v271())
            if v197 == v35:
                v116.v198('')
            else:
                v116.v198(v32.v269([v197], skip_special_tokens=False) or '')
        v37.v198(v25.v247(v116, v34))
    return v39.v118(v37, 0)

def sample_id_batch(v40: v141[v141[v23]], v41: v23, v42: v199.v119, v43, v35: v23):
    v44 = []
    for v45 in v115(v41):
        v65 = v40[v42.v217(0, v215(v40) - 1)]
        if v215(v65) < 8:
            v65 = v65 * 4
        v120 = v200(0, v215(v65) - v12)
        v121 = v42.v217(0, v120) if v120 > 0 else 0
        v122 = v65[v121:v121 + v12]
        if v215(v122) < v12:
            v122 = v122 + [v35] * (v12 - v215(v122))
        v44.v198(v122)
    return v39.v123(v44, dtype=v39.v201, device=v43)

def train_loss(v46: v124, v32, v33, v34, v35, v43):
    v47 = v33 == v35
    v48 = v248(v32, v33, v34, v35).v125(v43)
    v126, v127, v111 = v46.v128(v48, v47, inst_prefix=None)
    v49 = v46.v129(v111[:, :-1])
    v50 = v33[:, 1:]
    v51 = ~v47[:, :-1] & ~v47[:, 1:]
    if v51.v202() < 1:
        return (v127.v202() * 0.0, {'ce': 0.0, 'ppl': 0.0, 'cos_next': 0.0})
    v52 = v203.v130(v49[v51], v50[v51])
    v53 = v17 * v52
    v54 = {'ce': v204(v52.v249()), 'ppl': v204(v39.v250(v52.v249().v272(max=20)))}
    v55 = v51
    if v55.v202() > 0:
        v131 = v46.v108.v205(v127[:, :-1])
        v132 = (1.0 - v203.v135(v131[v55], v126[:, 1:][v55].v249(), dim=-1)).v206()
        v53 = v53 + v18 * v132
        v54['cos_next'] = v204(v203.v135(v131[v55], v126[:, 1:][v55].v249(), dim=-1).v206())
    return (v53, v54)

def retention_slow(v46: v124, v32, v34, v35, v56, v57, v43):
    """batch_ids: [P,A] each"""

    def run(v66):
        v47 = v66 == v35
        v48 = v248(v32, v66, v34, v35).v125(v43)
        v45, v45, v111 = v46.v128(v48, v47)
        return v26.v207(v111, v47)
    v133, v134 = (v208(v56), v208(v57))
    v58 = v203.v135(v133, v134, dim=-1)
    v59 = v203.v273(v58 - 0.5).v206() + 0.15 * v58.v206()
    return (v19 * v59, {'ret_cos': v204(v58.v206().v249())})

def sample_ret_id_pairs(v60: v103, v61: v23, v42: v199.v119, v35: v23, v43):
    v62 = [v136 for v136, v143 in v60.v147() if v215(v143) >= 2]
    if not v62:
        return None
    v137, v138 = ([], [])
    for v45 in v115(v61):
        v139 = v62[v42.v217(0, v215(v62) - 1)]
        v209, v210 = v42.v211(v60[v139], 2)

        def pack(v150):
            v150 = v150[-v12:]
            if v215(v150) < v12:
                v150 = v150 + [v35] * (v12 - v215(v150))
            return v150
        v137.v198(v251(v209))
        v138.v198(v251(v210))
    return (v39.v123(v137, dtype=v39.v201, device=v43), v39.v123(v138, dtype=v39.v201, device=v43))

def build_same_last_id_index(v40: v141[v141[v23]], v63: v23=40):
    from collections import defaultdict
    v64 = v140(v141)
    for v65 in v40:
        if v215(v65) < 12:
            continue
        for v142 in v115(10, v252(v215(v65), 80)):
            v139 = v65[v142]
            v150 = v65[v200(0, v142 - (v12 - 1)):v142 + 1]
            if v215(v64[v139]) < v63:
                v253 = v255(v150[:-1])
                if v274((v255(v121[:-1]) != v253 for v121 in v64[v139])):
                    v64[v139].v198(v150)
    return {v136: v143 for v136, v143 in v64.v147() if v215(v143) >= 2}

class SlowGateWrap(v31.v21):

    def __init__(v105, v46: v124, v32, v34, v35, v144='slow'):
        v270().v194()
        v105.v46 = v46
        v105.v32 = v32
        v105.v34 = v34
        v105.v35 = v35
        v105.v144 = v144

    def forward_states(v105, v48, v145=None):
        if v145 is None:
            v145 = v39.v254(v48.v275(0), v48.v275(1), dtype=v39.v276, device=v48.v43)
        v45, v127, v111 = v105.v46.v128(v48, v145)
        if v105.v144 == 'fast':
            return v127
        if v105.v144 == 'combined':
            return 0.5 * v127 + 0.5 * v111
        return v111

@v39.v68()
def encode_id_seq_slow(v46, v32, v34, v35, v66: v141[v23], v43):
    v66 = v66[-v12:]
    v67 = v39.v123([v66], dtype=v39.v201, device=v43)
    v47 = v67 == v35
    v48 = v248(v32, v67, v34, v35).v125(v43)
    v45, v45, v111 = v46.v128(v48, v47)
    return v111[0]

def gate_A_ids(v46, v40, v32, v34, v35, v43, v42, v61=80):
    from collections import defaultdict
    v64 = v140(v141)
    for v65 in v40:
        if v215(v65) < 12:
            continue
        for v142 in v115(8, v252(v215(v65), 80)):
            v139 = v65[v142]
            v150 = v65[v200(0, v142 - (v12 - 1)):v142 + 1]
            if v215(v64[v139]) < 40:
                v253 = v255(v150[:-1])
                if v274((v255(v121[:-1]) != v253 for v121 in v64[v139])):
                    v64[v139].v198(v150)
    v69 = []
    for v139, v146 in v64.v147():
        v148 = {}
        for v121 in v146:
            v212 = v255(v121[:-1])
            if v212 not in v148:
                v148[v212] = v121
            if v215(v148) >= 2:
                break
        if v215(v148) >= 2:
            v213 = v141(v148.v277())
            v69.v198((v213[0], v213[1]))
        if v215(v69) >= v61:
            break
    v42.v149(v69)
    v69 = v69[:v61]
    v70 = [v121 for v146 in v141(v64.v277())[:200] for v121 in v146[:3]]
    v71 = []
    for v45 in v115(v61 * 4):
        if v215(v70) < 2:
            break
        v152, v38 = v42.v211(v70, 2)
        if v152[-1] != v38[-1]:
            v71.v198((v152, v38))
        if v215(v71) >= v61:
            break

    def last_h(v150):
        v151 = v214(v46, v32, v34, v35, v150, v43)
        return v151[-1]

    def c(v152, v38):
        return v204(v203.v135(v203.v278(v152, dim=0), v203.v278(v38, dim=0), dim=0))
    v72 = [v169(v256(v152), v256(v38)) for v152, v38 in v69]
    v73 = [v169(v256(v152), v256(v38)) for v152, v38 in v71]
    v74 = v204(v279.v206(v72)) if v72 else 1.0
    v75 = v204(v279.v206(v73)) if v73 else 0.0
    if v74 >= 0.98:
        v153 = 'A_FAIL_LAST_TOKEN_WIPES'
    elif v74 < 0.9 and v74 - v75 < 0.35:
        v153 = 'A_PASS_PREFIX_VISIBLE'
    else:
        v153 = 'A_WEAK_PARTIAL'
    return {'verdict': v153, 'mean_cos_same_last_piece': v74, 'mean_cos_diff_last_piece': v75, 'n_same': v215(v72), 'n_diff': v215(v73)}

def gate_B_slow(v46, v32, v34, v35, v43, v42):
    v76 = v257(v46, v32, v34, v35, mode='slow').v125(v43)
    return v216.v154(v76, v32, v34, v43, v42)

@v39.v68()
def slow_ce_ablation(v46, v32, v34, v40, v35, v43, v42, v77=40):
    """CE of next-ids from slow: natural vs prefix-shuffled (same suffix)."""
    v155, v156 = ([], [])
    for v45 in v115(v77 * 2):
        if v215(v155) >= v77:
            break
        v65 = v40[v42.v217(0, v215(v40) - 1)]
        if v215(v65) < v12:
            continue
        v121 = v42.v217(0, v215(v65) - v12)
        v122 = v65[v121:v121 + v12]
        v157 = v200(8, v12 // 3)
        v218, v219 = (v122[:-v157], v122[-v157:])
        v158 = v218.v220()
        v42.v149(v158)

        def ce_of(v66):
            v67 = v39.v123([v66], dtype=v39.v201, device=v43)
            v47 = v67 == v35
            v48 = v248(v32, v67, v34, v35).v125(v43)
            v45, v45, v111 = v46.v128(v48, v47)
            v49 = v46.v129(v111[:, :-1])
            v50 = v67[:, 1:]
            v221 = v200(0, v12 - v157 - 1)
            v222 = v49[:, v221:]
            v223 = v50[:, v221:]
            v224 = v47[:, 1:][:, v221:]
            v51 = ~v224
            if v51.v202() < 1:
                return None
            return v204(v203.v130(v222[v51], v223[v51]))
        v152, v38 = (v258(v122), v258(v158 + v219))
        if v152 is not None and v38 is not None:
            v155.v198(v152)
            v156.v198(v38)
    if not v155:
        return {'delta_shuf_minus_nat': 0.0, 'nat': 0.0, 'shuf': 0.0, 'n': 0}
    v159, v160 = (v204(v279.v206(v155)), v204(v279.v206(v156)))
    return {'mean_ce_natural': v159, 'mean_ce_prefix_shuffled': v160, 'delta_shuf_minus_nat': v160 - v159, 'n': v215(v155), 'note': 'positive => slow-CE uses prefix (context)'}

def main() -> v23:
    v78 = v225.v161()
    v78.v162('--steps', type=v23, default=v16)
    v78.v162('--device', default='cuda' if v39.v283.v280() else 'cpu')
    v79 = v78.v163()
    v0.v101(parents=True, exist_ok=True)
    v1.v101(parents=True, exist_ok=True)
    v2.v104('', encoding='utf-8')
    v164(f'Stage182 start {v285.v281(v286.v282).v243()}')
    v164('Hybrid: dual-channel tape + SLOW predicts next BPE id (CE)')
    v164(f'plan={v7} | unlock: piece-id CE from slow only')
    if not v6.v184():
        raise v226(v6)
    v32 = v112.v165(v99(v6))
    v35 = v32.v227(v20) or 0
    v80 = v32.v166()
    v81 = v228.v167(max_chars=20000000)
    v82 = v168(v259(v81) | {' '})
    v83 = ['<pad>'] + v82
    v34 = {v169: v142 + 1 for v142, v169 in v260(v82)}
    v40 = v229.v170(v32, v81)
    v84 = v40[v23(0.8 * v215(v40)):] or v40[-100:]
    v85 = v40[:v23(0.8 * v215(v40))] or v40
    v86 = v171(v85)
    v164(f'docs={v215(v40)} V={v80} same-last={v215(v86)} d={v10}')
    v43 = v39.v43(v79.v43)
    v39.v172(v9)
    v199.v173(v9)
    v46 = v124(v215(v83), v80).v125(v43)
    v87 = v39.v230.v174(v46.v231(), lr=v14, weight_decay=0.01)
    v42 = v199.v119(v9)
    v46.v175()
    v88 = v176(v46, v84, v32, v34, v35, v43, v199.v119(v9))
    v89 = v177(v46, v32, v34, v35, v43, v199.v119(v9 + 1))
    v90 = v178(v46, v32, v34, v84, v35, v43, v199.v119(v9 + 2))
    v164(f"  init A: same={v88['mean_cos_same_last_piece']:.3f} → {v88['verdict']}")
    v164(f"  init B: para={v89['mean_cos_paraphrase']:.3f} hard={v89['mean_cos_hard_spelling']:.3f} → {v89['verdict']}")
    v164(f"  init ablΔ={v90['delta_shuf_minus_nat']:+.4f}")
    v91 = []
    v179, v180, v181 = (v88, v89, v90)
    v92 = None
    v46.v85()
    for v93 in v115(1, v79.v232 + 1):
        v66 = v233(v85, v13, v42, v43, v35)
        v53, v234 = v235(v46, v32, v66, v34, v35, v43)
        v182 = v236(v86, 4, v42, v35, v43)
        if v182 is not None:
            v261, v262 = v263(v46, v32, v34, v35, v182[0], v182[1], v43)
            v53 = v53 + v261
            v234.v264(v262)
        v87.v237(set_to_none=True)
        v53.v238()
        v31.v265.v239(v46.v231(), 1.0)
        v87.v93()
        v92 = v234['ce'] if v92 is None else 0.95 * v92 + 0.05 * v234['ce']
        if v93 % v15 == 0 or v93 == v79.v232:
            v46.v175()
            v179 = v176(v46, v84, v32, v34, v35, v43, v199.v119(v9 + v93))
            v180 = v177(v46, v32, v34, v35, v43, v199.v119(v9 + v93 + 3))
            v181 = v178(v46, v32, v34, v84, v35, v43, v199.v119(v9 + v93 + 5))
            v240 = v180['mean_cos_hard_spelling'] - v180['mean_cos_paraphrase']
            v241 = {'step': v93, 'ce': v92, 'A_same': v179['mean_cos_same_last_piece'], 'A': v179['verdict'], 'para': v180['mean_cos_paraphrase'], 'hard': v180['mean_cos_hard_spelling'], 'gap': v240, 'B': v180['verdict'], 'ablation_delta': v181['delta_shuf_minus_nat']}
            v91.v198(v241)
            v164(f"  step {v93}: ce~{v92:.3f} A_same={v241['A_same']:.3f}→{v241['A']} | para={v241['para']:.3f} hard={v241['hard']:.3f} gap={v241['gap']:.3f} | ablΔ={v241['ablation_delta']:+.4f}")
            v46.v85()
            v39.v266({'model': v46.v284(), 'step': v93, 'A': v179, 'B': v180, 'Ab': v181}, v5)
    v94 = v181['delta_shuf_minus_nat'] > 0.5
    v95 = v179['mean_cos_same_last_piece'] < 0.9
    if v94 and v95:
        v183 = 'SLOW_CE_TAPE_CONTEXT_YES'
    elif v94:
        v183 = 'SLOW_CE_ABLATION_YES_A_WEAK'
    elif v95:
        v183 = 'SLOW_CE_A_YES_ABLATION_WEAK'
    else:
        v183 = 'SLOW_CE_FLAT'
    v96 = None
    if v8.v184():
        v185 = v246.v242(v8.v267(encoding='utf-8'))
        v96 = {'gpt_ablation': v185.v268('final_ablation', {}).v268('delta_shuf_minus_nat'), 'gpt_A_same': v185.v268('final_A', {}).v268('mean_cos_same_last_piece'), 'tape_ablation': v181['delta_shuf_minus_nat'], 'tape_A_same': v179['mean_cos_same_last_piece']}
    v97 = {'timestamp': v285.v281(v286.v282).v243(), 'protocol': 'slow_ce_dual_tape_182', 'overall': v183, 'design': 'fast/slow draw BPE tape; slow state → next piece-id CE (main); weak fast ink + light retention', 'contract_note': 'Piece-id CE unlocked on SLOW only — hybrid path, not 169 word-CE revival', 'final_A': v179, 'final_B': v180, 'final_ablation': v181, 'init_A': v88, 'init_B': v89, 'history': v91, 'vs_181_gpt': v96, 'next': 'If ablation approaches GPT: hybrid works. If A wipe returns: raise W_RET / write-budget. If flat: CE head underpowered vs GPT attn.'}
    v186(v3, v97)
    v4.v104('\n'.v244(['# Stage182 — slow-CE on dual tape', '', f'**Overall:** `{v183}`', '', f"- A: {v179['verdict']} same={v179['mean_cos_same_last_piece']:.3f}", f"- B: para={v180['mean_cos_paraphrase']:.3f} hard={v180['mean_cos_hard_spelling']:.3f}", f"- ablation Δ={v181['delta_shuf_minus_nat']:+.4f}", f'- vs GPT181: {v96}', f"- {v97['next']}", '']), encoding='utf-8')
    v164(f'[182] {v183}')
    return 0
if v98 == '__main__':
    raise v187(v245())