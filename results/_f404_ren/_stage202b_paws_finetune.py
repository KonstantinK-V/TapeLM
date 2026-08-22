"""
Stage 202b — decisive B-capability test: fine-tune the encoder END-TO-END on PAWS.

202 showed a head on FROZEN features plateaus (~0.65) for BOTH curve and GPT — the
frozen small encoder is the bottleneck. Here we UNFREEZE the encoder (on a COPY; product
P1 stays frozen) and train encoder+head jointly on PAWS. Fair GPT control fine-tuned the
same way. Question: given a meaning signal AND a trainable encoder, can the CURVE reach
semantic invariance (inversion para>hard) and match GPT?

Runs on 4GB: d256/6L, short sentences (<=64 tok), small batch.

Gates:
  G_paws       curve PAWS test acc >= 0.75
  G_inversion  179 para_sim > hard_sim
  G_parity     |curve_acc - gpt_acc| <= 0.03
  verdict: G_paws & G_inversion -> SEM_B_CAP_CONFIRMED ; G_paws -> SEM_B_CAP_PARTIAL ; else SEM_B_CAP_NO

  python _stage202b_paws_finetune.py
"""
from __future__ import annotations
from datasets import load_dataset
import copy
import json
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
from transformers import GPT2Config, GPT2LMHeadModel
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data, score_items, span_logprob_x
v0 = v18('results')
v1 = v18('checkpoints/stage191_p1_curve.pt')
v2 = v18('checkpoints/stage191_p2_gpt.pt')
v3 = v18('data/stage191_exam_v3.jsonl')
v4 = v0 / 'stage202b_decision.json'
v5 = v0 / 'stage202b_mini.md'
v6 = v0 / '_stage202b_log.txt'
v7 = 2022
v8 = 128
v9 = 64
v10 = 3
v11 = 48
v12 = 0.0001
v13 = 0.0005
v14 = 0.4
v15 = v44.v19('[A-Za-z]+')

def log(v20: v45) -> None:
    v21 = v20 if v20.v98('\n') else v20 + '\n'
    try:
        v99(v21, end='', flush=True)
    except v46:
        v99(v21.v210('ascii', 'replace').v173('ascii'), end='', flush=True)
    v6.v100.v47(parents=True, exist_ok=True)
    with v6.v101('a', encoding='utf-8') as v48:
        v48.v102(v21)

class SemHead(v22.v16):

    def __init__(v49, v31, v50=v8):
        v174().v103()
        v49.v51 = v22.v104(v31, 1)
        v49.v52 = v22.v105(v22.v104(v31, v31), v22.v147(), v22.v104(v31, v50))

    def forward(v49, v53, v54):
        v55 = v49.v51(v53).v175(-1).v106(~v54, -1000000000.0)
        v56 = v55.v176(-1).v107(-1)
        return v148.v108(v49.v52((v56 * v53).v177(1)), dim=-1)

def main() -> v17:
    v0.v47(parents=True, exist_ok=True)
    v6.v57('', encoding='utf-8')
    v58(f'Stage202b start {v208.v200(v209.v201).v143()}')
    v58('decisive B: fine-tune encoder end-to-end on PAWS (copy; product P1 frozen)')
    v23 = v109.v23('cuda' if v109.v178.v149() else 'cpu')
    v109.v59(v7)
    v24 = v110.v60(v7)
    v25 = v61.v61()
    v62, v63, v64, v65 = v66()
    v26 = v111.v67(v45(v150.v112))
    v27 = v26.v68()
    v28 = v26.v113(v114) or 0
    v29 = v179.v151(v26, v64, v28, v27).v69(v23)
    v30 = v152(v65, v27).v69(v23)
    v30.v70(v109.v153(v1, map_location=v23, weights_only=False)['model'])
    v30.v71()
    v31 = v30.v86.v72 // 2
    v32 = v73('paws', 'labeled_final')
    v33 = [(v115['sentence1'], v115['sentence2'], v17(v115['label'])) for v115 in v32['train']]
    v34 = [(v115['sentence1'], v115['sentence2'], v17(v115['label'])) for v115 in v32['validation']]
    v35 = [(v115['sentence1'], v115['sentence2'], v17(v115['label'])) for v115 in v32['test']]
    v58(f'PAWS train={v180(v33)} val={v180(v34)} test={v180(v35)} ({v61.v61() - v25:.0f}s)')

    def ids_of(v74):
        v75 = [[v118 for v118 in v26.v210(v154).v75 if v118 != v28][:v9] or [v28] for v154 in v74]
        v76 = v116((v180(v77) for v77 in v75))
        v77 = v109.v117((v180(v75), v76), v28, dtype=v109.v155, device=v23)
        for v118, v119 in v120(v75):
            v77[v118, :v180(v119)] = v109.v156(v119, device=v23)
        return v77

    def curve_states(v77, v78):
        v79 = v77 == v28
        v80 = v78.v121(v29[v77], v77)
        v81 = v78.v81(v80, pad_mask=v79)
        return (v81, ~v79)

    def gpt_states(v77, v78):
        v82 = v77 != v28
        v83 = v78.v157(input_ids=v77, attention_mask=v82.v155()).v84
        return (v83, v82)

    def run(v85):
        if v85 == 'curve':
            v78 = v181.v158(v30)
            v78.v33()
            v122 = v123
            v124 = v128(v78.v211.v182()) + v128(v78.v81.v182())
        else:
            v125 = v109.v153(v2, map_location=v23, weights_only=False)
            v78 = v202(v212(**v125['conf'])).v69(v23)
            v78.v70(v125['model'])
            v78.v33()
            v122 = v126
            v124 = v128(v78.v157.v182())
        v86 = v183(v31).v69(v23)
        v87 = v109.v159.v127([{'params': v124, 'lr': v12}, {'params': v86.v182(), 'lr': v13}], weight_decay=0.01)
        v88 = v128(v129(v180(v33)))
        for v89 in v129(v10):
            v24.v160(v88)
            v130 = None
            for v131 in v129(0, v180(v88), v11):
                v161 = [v33[v118] for v118 in v88[v131:v131 + v11]]
                v162 = v109.v156([v77[2] for v77 in v161], dtype=v109.v138, device=v23)
                v184, v185 = v122(v194([v77[0] for v77 in v161]), v78)
                v186, v187 = v122(v194([v77[1] for v77 in v161]), v78)
                v163 = (v86(v184, v185) * v86(v186, v187)).v177(-1)
                v164 = (v162 * (1 - v163) + (1 - v162) * v148.v223(v163 - v14)).v170()
                v87.v188(set_to_none=True)
                v164.v189()
                v22.v203.v190(v124 + v128(v86.v182()), 1.0)
                v87.v191()
                v130 = v138(v164) if v130 is None else 0.98 * v130 + 0.02 * v138(v164)
            v58(f'  [{v85}] epoch {v89 + 1}/{v10} loss~{v130:.4f} ({v61.v61() - v25:.0f}s)')
        v78.v71()
        v86.v71()

        @v109.v133()
        def cos_of(v132):
            v42 = []
            for v131 in v129(0, v180(v132), 128):
                v165 = v132[v131:v131 + 128]
                v184, v185 = v122(v194([v77[0] for v77 in v165]), v78)
                v186, v187 = v122(v194([v77[1] for v77 in v165]), v78)
                v42.v192((v86(v184, v185) * v86(v186, v187)).v177(-1).v220().v204())
            return v193.v166(v42)
        v134, v135 = (v193.v167([v77[2] for v77 in v34]), v168(v34))
        v90 = v116(v193.v169(-0.2, 0.95, 60), key=lambda v139: ((v135 >= v139).v213(v17) == v134).v170())
        v136, v137 = (v193.v167([v77[2] for v77 in v35]), v168(v35))
        v91 = v138(((v137 >= v90).v213(v17) == v136).v170())

        @v109.v133()
        def z(v139):
            v171, v82 = v122(v194([v139]), v78)
            return v86(v171, v82)[0]
        v92 = v138(v193.v170([v138(v148.v214(v221(v215), v221(v131), dim=-1)) for v215, v131 in v216.v205]))
        v93 = v138(v193.v170([v138(v148.v214(v221(v215), v221(v131), dim=-1)) for v215, v131 in v216.v206]))
        v94 = {'paws_acc': v91, 'para': v92, 'hard': v93, 'inversion': v92 > v93}
        if v85 == 'curve':
            v140 = [v172.v195(v196) for v196 in v3.v222(encoding='utf-8').v207()]
            v141 = [v197 for v197 in v140 if v197['type'] == 'next_tok'][:120]
            v94['next_tok_copy'] = v198(lambda v217, v218: v219(v78, v29, v28, v217, v218, v23), v141, 'next_tok')['next_tok_acc']
        return v94
    v36 = v95('curve')
    v58(f"curve: paws={v36['paws_acc']:.3f} para={v36['para']:.3f} hard={v36['hard']:.3f} inv={v36['inversion']} nt_copy={v36.v199('next_tok_copy')}")
    v37 = v95('gpt')
    v58(f"gpt:   paws={v37['paws_acc']:.3f} para={v37['para']:.3f} hard={v37['hard']:.3f} inv={v37['inversion']}")
    v38 = v36['paws_acc'] >= 0.75
    v39 = v36['inversion']
    v40 = v142(v36['paws_acc'] - v37['paws_acc']) <= 0.03
    if v38 and v39:
        v96 = 'SEM_B_CAP_CONFIRMED'
    elif v38:
        v96 = 'SEM_B_CAP_PARTIAL'
    else:
        v96 = 'SEM_B_CAP_NO'
    v41 = {'g_paws': v38, 'g_inversion': v39, 'g_parity': v40}
    v42 = {'timestamp': v208.v200(v209.v201).v143(), 'protocol': 'paws_finetune_202b', 'overall': v96, 'gates': v41, 'curve': v36, 'gpt': v37, 'note': 'encoder fine-tuned end-to-end on PAWS (copy; product P1 frozen); decisive B-capability test'}
    v4.v57(v172.v144(v42, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v57('\n'.v145(['# Stage202b — decisive B: encoder fine-tune on PAWS', '', f'**Overall:** `{v96}`', '', f"- curve: PAWS **{v36['paws_acc']:.3f}** | 179 para {v36['para']:.3f} / hard {v36['hard']:.3f} (**inversion={v36['inversion']}**) | next_tok(copy) {v36.v199('next_tok_copy')}", f"- gpt:   PAWS {v37['paws_acc']:.3f} | para {v37['para']:.3f} / hard {v37['hard']:.3f} (inversion={v37['inversion']})", '', f'gates: {v41}', '', 'Encoder fine-tuned end-to-end (copy); product P1 frozen. Tests whether the curve substrate CAN reach semantic invariance given a meaning signal + trainable encoder, at parity with GPT.']), encoding='utf-8')
    v58(f"[202b] {v96} | curve paws={v36['paws_acc']:.3f} inv={v36['inversion']} gpt={v37['paws_acc']:.3f}")
    return 0
if v43 == '__main__':
    raise v97(v146())