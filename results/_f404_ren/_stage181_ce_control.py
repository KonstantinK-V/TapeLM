"""
Stage 181 — Matched CE-Transformer control (dataset context ceiling).

Same ByteLevel BPE (Stage177) + same wiki chunk as curve stages.
Train a small GPT-2 with standard next-token CE.
Probe the SAME gates as curve:
  A) same last piece / different prefix → hidden-state wipe?
  B) paraphrase vs hard spelling (micro-signal: para↑, gap hard-para↓)
  Ablation) CE loss with natural vs prefix-shuffled context

Question: does THIS dataset+scale support any context signal under ordinary LM training?
If CE control also flat on B-micro → don't call a curve wall.
If CE control shows A/B-micro and curve doesn't → curve objective/arch lag.

  python _stage181_ce_control.py
  python _stage181_ce_control.py --steps 10000
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
v0 = v23('results')
v1 = v23('checkpoints')
v2 = v0 / '_stage181_log.txt'
v3 = v0 / 'stage181_decision.json'
v4 = v0 / 'stage181_mini.md'
v5 = v1 / 'stage181_ce_control.pt'
v6 = v24.v6
v7 = v0 / 'plan_curve_dynamics.md'
v8 = v0 / 'stage180_decision.json'
v9 = 181
v10 = 128
v11 = 4
v12 = 4
v13 = 64
v14 = 24
v15 = 0.0003
v16 = 1500
v17 = 10000
v18 = '[PAD]'

def log(v25: v88) -> None:
    v26 = v25 if v25.v156('\n') else v25 + '\n'
    try:
        v157(v26, end='', flush=True)
    except v89:
        v157(v26.v221('ascii', 'replace').v211('ascii'), end='', flush=True)
    v2.v158.v90(parents=True, exist_ok=True)
    with v2.v159('a', encoding='utf-8') as v91:
        v91.v160(v26)

def write_json(v27: v23, v28: v21) -> None:
    v27.v158.v90(parents=True, exist_ok=True)
    v27.v92(v201.v161(v28, indent=2, ensure_ascii=False), encoding='utf-8')

def cos(v29: v52.v20, v30: v52.v20) -> v19:
    return v19(v169.v162(v169.v108(v29.v19(), dim=0), v169.v108(v30.v19(), dim=0), dim=0))

def build_id_docs(v31: v93, v32: v88, v33: v22=4000) -> v37[v37[v22]]:
    v34 = v31.v94(v18)
    v35 = []
    for v36 in v32.v95('\n\n'):
        v36 = v36.v163()
        if v164(v36) < 40:
            continue
        v46 = [v96 for v96 in v31.v221(v36).v46 if v96 != v34]
        if v164(v46) >= 16:
            v35.v167(v46)
        if v164(v35) >= v33:
            break
    if v164(v35) < 50:
        v46 = [v96 for v96 in v31.v221(v32[:2000000]).v46 if v96 != v34]
        for v96 in v98(0, v166(1, v164(v46) - 64), 48):
            v35.v167(v46[v96:v96 + 128])
            if v164(v35) >= v33:
                break
    return v35

def sample_batch(v35: v37[v37[v22]], v38: v22, v39: v165.v97, v40, v34: v22):
    v41 = []
    for v42 in v98(v38):
        v55 = v35[v39.v180(0, v164(v35) - 1)]
        if v164(v55) < 8:
            v55 = v55 * 4
        v99 = v166(0, v164(v55) - v13)
        v100 = v39.v180(0, v99) if v99 > 0 else 0
        v101 = v55[v100:v100 + v13]
        if v164(v101) < v13:
            v101 = v101 + [v34] * (v13 - v164(v101))
        v41.v167(v101)
    v43 = v52.v102(v41, dtype=v52.v105, device=v40)
    v44 = v43.v103()
    v44[v44 == v34] = -100
    return v43

@v52.v51()
def hidden_last(v45: v104, v46: v37[v22], v40, v34: v22) -> v52.v20:
    v46 = v46[-v13:]
    v43 = v52.v102([v46], dtype=v52.v105, device=v40)
    v47 = (v43 != v34).v105()
    v48 = v45.v106(input_ids=v43, attention_mask=v47)
    v49 = v48.v107[0]
    v50 = v22(v47[0].v212().v168())
    return v49[v50 - 1]

@v52.v51()
def hidden_summary(v45, v46: v37[v22], v40, v34: v22) -> v52.v20:
    v46 = v46[-v13:]
    v43 = v52.v102([v46], dtype=v52.v105, device=v40)
    v47 = (v43 != v34).v105()
    v49 = v45.v106(input_ids=v43, attention_mask=v47).v107[0]
    v50 = v22(v47[0].v212().v168())
    v49 = v49[:v50]
    return v169.v108(v52.v170([v49[-1], v49.v206(0)], 0), dim=0)

def gate_A(v45, v35: v37[v37[v22]], v40, v34: v22, v39: v165.v97, v53: v22=80) -> v21:
    v54: v21[v22, v37[v37[v22]]] = v109(v37)
    for v55 in v35:
        if v164(v55) < 12:
            continue
        for v96 in v98(8, v202(v164(v55), 80)):
            v110 = v55[v96]
            v171 = v55[v166(0, v96 - (v13 - 1)):v96 + 1]
            if v164(v54[v110]) < 40:
                v203 = v204(v171[:-1])
                if v213((v204(v100[:-1]) != v203 for v100 in v54[v110])):
                    v54[v110].v167(v171)
    v56 = []
    for v110, v111 in v54.v112():
        v113 = {}
        for v100 in v111:
            v172 = v204(v100[:-1])
            if v172 not in v113:
                v113[v172] = v100
            if v164(v113) >= 2:
                break
        if v164(v113) >= 2:
            v173 = v37(v113.v214())
            v56.v167((v173[0], v173[1]))
        if v164(v56) >= v53:
            break
    v39.v114(v56)
    v56 = v56[:v53]
    v57 = [v100 for v111 in v37(v54.v214())[:200] for v100 in v111[:3]]
    v58 = []
    for v42 in v98(v53 * 4):
        if v164(v57) < 2:
            break
        v29, v30 = v39.v174(v57, 2)
        if v29[-1] != v30[-1]:
            v58.v167((v29, v30))
        if v164(v58) >= v53:
            break
    v59 = [v175(v205(v45, v29, v40, v34), v205(v45, v30, v40, v34)) for v29, v30 in v56]
    v60 = [v175(v205(v45, v29, v40, v34), v205(v45, v30, v40, v34)) for v29, v30 in v58]
    v61 = v19(v215.v206(v59)) if v59 else 1.0
    v62 = v19(v215.v206(v60)) if v60 else 0.0
    if v61 >= 0.98:
        v115 = 'A_FAIL_LAST_TOKEN_WIPES'
    elif v61 < 0.9 and v61 - v62 < 0.35:
        v115 = 'A_PASS_PREFIX_VISIBLE'
    else:
        v115 = 'A_WEAK_PARTIAL'
    return {'verdict': v115, 'mean_cos_same_last_piece': v61, 'mean_cos_diff_last_piece': v62, 'n_same': v164(v59), 'n_diff': v164(v60)}

def gate_B(v45, v31: v93, v40, v34: v22, v39: v165.v97) -> v21:

    def enc(v32: v88):
        v46 = [v96 for v96 in v31.v221(v32).v46 if v96 != v34]
        return v176(v45, v46 or [v31.v94('a') or 1], v40, v34)
    v63 = [v175(v207(v29), v207(v30)) for v29, v30 in v116.v65]
    v64 = [v175(v207(v29), v207(v30)) for v29, v30 in v116.v177]
    v57 = []
    for v29, v30 in v116.v65:
        v57.v178([v207(v29), v207(v30)])
    v66 = []
    for v42 in v98(v164(v63) * 4):
        v96, v179 = v39.v174(v98(v164(v57)), 2)
        v66.v167(v175(v57[v96], v57[v179]))
    v117, v118, v119 = (v19(v215.v206(v63)), v19(v215.v206(v66)), v19(v215.v206(v64)))
    v67 = v119 - v117
    v68 = v117 - v118
    if v68 > 0.05 and v117 - v119 > 0.03:
        v115 = 'B_PASS_MEANING_STRUCTURE'
    elif v67 > 0.05 and v68 <= 0.05:
        v115 = 'B_FORM_DOMINANT'
    elif v68 <= 0.02:
        v115 = 'B_NO_PARA_CLUSTER'
    else:
        v115 = 'B_MICRO_MIXED'
    return {'verdict': v115, 'mean_cos_paraphrase': v117, 'mean_cos_random': v118, 'mean_cos_hard_spelling': v119, 'gap_hard_minus_para': v67, 'lift_vs_random': v68, 'n_para': v164(v63)}

@v52.v51()
def prefix_ablation(v45, v35, v40, v34, v39, v69: v22=40) -> v21:
    """CE on last positions: natural vs shuffled prefix (same suffix)."""
    v120, v121 = ([], [])
    for v42 in v98(v69):
        v55 = v35[v39.v180(0, v164(v35) - 1)]
        if v164(v55) < v13:
            continue
        v100 = v39.v180(0, v164(v55) - v13)
        v101 = v55[v100:v100 + v13]
        v122 = v166(8, v13 // 3)
        v181, v182 = (v101[:-v122], v101[-v122:])
        v123 = v181.v183()
        v39.v114(v123)
        for v46, v184 in ((v101, v120), (v123 + v182, v121)):
            v43 = v52.v102([v46], dtype=v52.v105, device=v40)
            v48 = v45(input_ids=v43, labels=v43)
            v184.v167(v19(v48.v146))
    if not v120:
        return {'delta_shuf_minus_nat': 0.0, 'nat': 0.0, 'shuf': 0.0}
    v124, v125 = (v19(v215.v206(v120)), v19(v215.v206(v121)))
    return {'mean_ce_natural': v124, 'mean_ce_prefix_shuffled': v125, 'delta_shuf_minus_nat': v125 - v124, 'n': v164(v120), 'note': 'positive delta => model used prefix (context helps CE)'}

def main() -> v22:
    v70 = v185.v126()
    v70.v127('--steps', type=v22, default=v17)
    v70.v127('--device', default='cuda' if v52.v222.v216() else 'cpu')
    v71 = v70.v128()
    v0.v90(parents=True, exist_ok=True)
    v1.v90(parents=True, exist_ok=True)
    v2.v92('', encoding='utf-8')
    v129(f'Stage181 start {v225.v219(v226.v220).v198()}')
    v129('Matched CE GPT-2 control — dataset context ceiling vs curve stages')
    v129(f'match: BPE={v6} d={v10} L={v11} H={v12} T={v13} steps={v71.v190}')
    if not v6.v147():
        raise v186(v6)
    v31 = v93.v130(v88(v6))
    v34 = v31.v94(v18)
    if v34 is None:
        v34 = 0
    v72 = v31.v131()
    v32 = v187.v132(max_chars=20000000)
    v35 = v133(v31, v32)
    v73 = v35[v22(0.8 * v164(v35)):] or v35[-100:]
    v74 = v35[:v22(0.8 * v164(v35))] or v35
    v129(f'docs={v164(v35)} V={v72} pad={v34}')
    v40 = v52.v40(v71.v40)
    v52.v134(v9)
    v165.v135(v9)
    v75 = v136(vocab_size=v72, n_positions=v13, n_embd=v10, n_layer=v11, n_head=v12, n_inner=4 * v10, bos_token_id=v34, eos_token_id=v34, pad_token_id=v34)
    v45 = v104(v75).v137(v40)
    v76 = v52.v188.v138(v45.v189(), lr=v15, weight_decay=0.01)
    v39 = v165.v97(v9)
    v45.v139()
    v77 = v140(v45, v73, v40, v34, v165.v97(v9))
    v78 = v141(v45, v31, v40, v34, v165.v97(v9 + 1))
    v79 = v142(v45, v73, v40, v34, v165.v97(v9 + 2))
    v129(f"  init A: same={v77['mean_cos_same_last_piece']:.3f} diff={v77['mean_cos_diff_last_piece']:.3f} → {v77['verdict']}")
    v129(f"  init B: para={v78['mean_cos_paraphrase']:.3f} hard={v78['mean_cos_hard_spelling']:.3f} gap={v78['gap_hard_minus_para']:.3f} lift_r={v78['lift_vs_random']:+.3f} → {v78['verdict']}")
    v129(f"  init ablation Δ(shuf-nat)={v79['delta_shuf_minus_nat']:+.4f}")
    v80 = []
    v143, v144, v145 = (v77, v78, v79)
    v81 = None
    v45.v74()
    for v82 in v98(1, v71.v190 + 1):
        v43 = v191(v74, v14, v39, v40, v34)
        v48 = v45(input_ids=v43, labels=v43)
        v146 = v48.v146
        v76.v192(set_to_none=True)
        v146.v193()
        v52.v217.v208.v194(v45.v189(), 1.0)
        v76.v82()
        v81 = v19(v146.v218()) if v81 is None else 0.95 * v81 + 0.05 * v19(v146.v218())
        if v82 % v16 == 0 or v82 == v71.v190:
            v45.v139()
            v143 = v140(v45, v73, v40, v34, v165.v97(v9 + v82))
            v144 = v141(v45, v31, v40, v34, v165.v97(v9 + v82 + 3))
            v145 = v142(v45, v73, v40, v34, v165.v97(v9 + v82 + 5))
            v195 = {'step': v82, 'ce': v81, 'A_same': v143['mean_cos_same_last_piece'], 'A': v143['verdict'], 'para': v144['mean_cos_paraphrase'], 'hard': v144['mean_cos_hard_spelling'], 'gap': v144['gap_hard_minus_para'], 'lift_r': v144['lift_vs_random'], 'B': v144['verdict'], 'ablation_delta': v145['delta_shuf_minus_nat']}
            v80.v167(v195)
            v129(f"  step {v82}: ce~{v81:.3f} A_same={v195['A_same']:.3f}→{v195['A']} | para={v195['para']:.3f} hard={v195['hard']:.3f} gap={v195['gap']:.3f}→{v195['B']} | ablΔ={v195['ablation_delta']:+.4f}")
            v45.v74()
            v52.v209({'model': v45.v223(), 'conf': v75.v224(), 'step': v82, 'A': v143, 'B': v144}, v5)
    v83 = None
    if v8.v147():
        v148 = v201.v196(v8.v210(encoding='utf-8'))
        v149 = v148.v197('trajectory', {})
        v150 = v149.v197('4500', {})
        v83 = {'curve_peak_para': v150.v197('para'), 'curve_peak_gap': v150.v197('gap'), 'curve_A_slow_best': v149.v197('1500', {}).v197('A_slow'), 'ce_final_para': v144['mean_cos_paraphrase'], 'ce_final_gap': v144['gap_hard_minus_para'], 'ce_final_A': v143['mean_cos_same_last_piece'], 'ce_ablation_delta': v145['delta_shuf_minus_nat']}
    if v164(v80) >= 2:
        v151 = v80[-1]['para'] - v80[0]['para']
        v152 = v80[-1]['gap'] - v80[0]['gap']
    else:
        v151 = v152 = 0.0
    v84 = v143['mean_cos_same_last_piece'] < 0.9
    v85 = v145['delta_shuf_minus_nat'] > 0.02
    v86 = v151 > 0.01 or v152 < -0.01
    if v84 and (v85 or v86):
        v153 = 'CE_CONTROL_CONTEXT_SIGNAL_YES'
    elif v84:
        v153 = 'CE_CONTROL_A_ONLY'
    elif v85:
        v153 = 'CE_CONTROL_ABLATION_ONLY'
    else:
        v153 = 'CE_CONTROL_FLAT'
    v48 = {'timestamp': v225.v219(v226.v220).v198(), 'protocol': 'ce_transformer_control_181', 'overall': v153, 'matched': {'tokenizer': v88(v6), 'corpus_chars': 20000000, 'd': v10, 'layers': v11, 'heads': v12, 'seq': v13, 'steps': v71.v190, 'objective': 'GPT2 next-token CE'}, 'init_A': v77, 'init_B': v78, 'init_ablation': v79, 'final_A': v143, 'final_B': v144, 'final_ablation': v145, 'b_micro': {'para_delta_first_to_last_eval': v151, 'gap_delta_first_to_last_eval': v152}, 'history': v80, 'vs_180': v83, 'interpretation': {'FLAT': "this dataset+scale barely teaches context even for CE — don't wall the curve", 'A_ONLY': 'prefix lives in state; meaning micro weak — same regime as curve', 'CONTEXT_SIGNAL_YES': 'data supports context under CE — curve should chase this ceiling'}, 'next': 'Use CE control as ceiling. If FLAT/A_ONLY: scale data before blaming curve. If YES: put semantic pressure on slow channel.'}
    v154(v3, v48)
    v4.v92('\n'.v199(['# Stage181 — CE Transformer control', '', f'**Overall:** `{v153}`', '', f"- A: {v143['verdict']} same={v143['mean_cos_same_last_piece']:.3f}", f"- B: {v144['verdict']} para={v144['mean_cos_paraphrase']:.3f} hard={v144['mean_cos_hard_spelling']:.3f} gap={v144['gap_hard_minus_para']:.3f}", f"- ablation Δ={v145['delta_shuf_minus_nat']:+.4f}", f'- B micro: paraΔ={v151:+.3f} gapΔ={v152:+.3f}', f"- {v48['next']}", '']), encoding='utf-8')
    v129(f'[181] {v153}')
    return 0
if v87 == '__main__':
    raise v155(v200())