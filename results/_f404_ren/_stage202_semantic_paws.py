"""
Stage 202 — B capability test (path A): can the FROZEN curve representation be made
semantically invariant when given a real meaning signal? Confirm-or-refute on 4GB.

Data: PAWS (adversarial paraphrase — high lexical overlap, label = same-meaning or not).
This is exactly the hard-pair problem: surface says "same", meaning may differ.

Method (NON-DESTRUCTIVE): P1 encoder FROZEN. Train only a semantic head with ATTENTION
pooling over per-token fast states (learns to down-weight shared/function words that make
hard pairs collapse under mean-pool) -> z_sem. Online-contrastive loss on PAWS labels.

Eval:
  - PAWS test accuracy (best cos threshold on val) vs lexical-overlap baseline (~chance by design)
  - INVERSION on the 179 pairs: para_sim > hard_sim (meaning finally beats spelling)
  - fair GPT baseline: identical head on GPT hidden states (is the curve competitive?)

Gates:
  G_paws       curve test acc >= 0.70 (chance/lexical ~0.55)
  G_inversion  179 para_sim > hard_sim
  G_vs_gpt     curve acc >= gpt acc - 0.03
  verdict: G_paws & G_inversion -> SEM_B_CONFIRMED ; G_paws only -> SEM_B_PARTIAL ; else SEM_B_NO

  python _stage202_semantic_paws.py
"""
from __future__ import annotations
from datasets import load_dataset
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
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data
from _stage196_tapelm import load_gpt
v0 = v15('results')
v1 = v15('checkpoints/stage191_p1_curve.pt')
v2 = v0 / 'stage202_decision.json'
v3 = v0 / 'stage202_mini.md'
v4 = v0 / '_stage202_log.txt'
v5 = 202
v6 = 128
v7 = 64
v8 = 4
v9 = 128
v10 = 0.0005
v11 = 0.4
v12 = v46.v16('[A-Za-z]+')

def log(v17: v47) -> None:
    v18 = v17 if v17.v114('\n') else v17 + '\n'
    try:
        v115(v18, end='', flush=True)
    except v48:
        v115(v18.v212('ascii', 'replace').v187('ascii'), end='', flush=True)
    v4.v116.v49(parents=True, exist_ok=True)
    with v4.v117('a', encoding='utf-8') as v50:
        v50.v118(v18)

class SemHead(v19.v13):

    def __init__(v51, v32, v52=v6):
        v188().v119()
        v51.v53 = v19.v120(v32, 1)
        v51.v54 = v19.v121(v19.v120(v32, v32), v19.v157(), v19.v120(v32, v52))

    def forward(v51, v55, v56):
        v57 = v51.v53(v55).v189(-1).v122(~v56, -1000000000.0)
        v58 = v57.v190(-1).v123(-1)
        v59 = (v58 * v55).v124(1)
        return v158.v125(v51.v54(v59), dim=-1)

def lexical_overlap(v20, v21):
    v60, v61 = (v126(v12.v159(v20.v191())), v126(v12.v159(v21.v191())))
    return v127(v60 & v61) / v128(1, v127(v60 | v61))

def main() -> v14:
    v0.v49(parents=True, exist_ok=True)
    v4.v62('', encoding='utf-8')
    v63(f'Stage202 start {v210.v204(v211.v205).v153()}')
    v63('B capability via PAWS on FROZEN curve encoder + attention head (4GB)')
    v22 = v129.v22('cuda' if v129.v192.v160() else 'cpu')
    v129.v64(v5)
    v23 = v130.v65(v5)
    v24 = v66.v66()
    v67, v68, v69, v70 = v71()
    v25 = v131.v72(v47(v161.v132))
    v26 = v25.v73()
    v27 = v25.v133(v134) or 0
    v28 = v193.v162(v25, v69, v27, v26).v74(v22)
    v29 = v163(v70, v26).v74(v22)
    v29.v75(v129.v164(v1, map_location=v22, weights_only=False)['model'])
    v29.v76()
    for v30 in v29.v77():
        v30.v135(False)
    v31 = v78(v22)
    v32 = v29.v98.v79 // 2
    v63(f'models loaded (fast dim={v32}) ({v66.v66() - v24:.0f}s)')
    v33 = v80('paws', 'labeled_final')
    v34 = [(v136['sentence1'], v136['sentence2'], v14(v136['label'])) for v136 in v33['train']]
    v35 = [(v136['sentence1'], v136['sentence2'], v14(v136['label'])) for v136 in v33['validation']]
    v36 = [(v136['sentence1'], v136['sentence2'], v14(v136['label'])) for v136 in v33['test']]
    v63(f'PAWS train={v127(v34)} val={v127(v35)} test={v127(v36)} ({v66.v66() - v24:.0f}s)')

    def lex_acc(v81):
        v82 = 0.0
        v83 = v165.v137([v194(v20, v21) for v20, v21, v206 in v81])
        v84 = v165.v137([v166 for v206, v206, v166 in v81])
        for v85 in v165.v138(0.1, 0.95, 40):
            v82 = v128(v82, v128(((v83 >= v85) == v84).v184(), ((v83 < v85) == v84).v184()))
        return v139(v82)
    v37 = v86(v36)
    v63(f'lexical-overlap baseline test acc={v37:.3f}')

    def curve_states(v87):
        v88 = [[v141 for v141 in v25.v212(v167).v88 if v141 != v27][:v7] or [v27] for v167 in v87]
        v89 = v128((v127(v90) for v90 in v88))
        v90 = v129.v140((v127(v88), v89), v27, dtype=v129.v168, device=v22)
        for v141, v142 in v143(v88):
            v90[v141, :v127(v142)] = v129.v169(v142, device=v22)
        v91 = v90 == v27
        with v129.v95():
            v144 = v29.v170(v28[v90], v90)
            v145 = v29.v145(v144, pad_mask=v91)
        return (v145, ~v91)

    @v129.v95()
    def gpt_states(v87):
        v88 = [[v141 for v141 in v25.v212(v167).v88 if v141 != v27][:v7] or [v27] for v167 in v87]
        v89 = v128((v127(v90) for v90 in v88))
        v90 = v129.v140((v127(v88), v89), v27, dtype=v129.v168, device=v22)
        v92 = v129.v146((v127(v88), v89), dtype=v129.v171, device=v22)
        for v141, v142 in v143(v88):
            v90[v141, :v127(v142)] = v129.v169(v142, device=v22)
            v92[v141, :v127(v142)] = True
        v93 = v31.v172(input_ids=v90, attention_mask=v92.v168()).v94
        return (v93, v92)

    def train_head(v96, v97):
        v98 = v195(v32).v74(v22)
        v99 = v129.v173.v147(v98.v77(), lr=v10, weight_decay=0.01)
        v100 = v148(v149(v127(v34)))
        for v101 in v149(v8):
            v23.v174(v100)
            v150 = None
            for v21 in v149(0, v127(v100), v9):
                v175 = [v34[v141] for v141 in v100[v21:v21 + v9]]
                v176 = [v90[0] for v90 in v175]
                v177 = [v90[1] for v90 in v175]
                v84 = v129.v169([v90[2] for v90 in v175], dtype=v129.v139, device=v22)
                v196, v197 = v96(v176)
                v198, v199 = v96(v177)
                v178 = v98(v196, v197)
                v179 = v98(v198, v199)
                v180 = (v178 * v179).v124(-1)
                v181 = (v84 * (1 - v180) + (1 - v84) * v158.v218(v180 - v11)).v184()
                v99.v200(set_to_none=True)
                v181.v201()
                v99.v202()
                v150 = v139(v181) if v150 is None else 0.98 * v150 + 0.02 * v139(v181)
            v63(f'  [{v97}] epoch {v101 + 1}/{v8} loss~{v150:.4f} ({v66.v66() - v24:.0f}s)')
        v98.v76()

        @v129.v95()
        def cos_of(v81):
            v44 = []
            for v21 in v149(0, v127(v81), 256):
                v182 = v81[v21:v21 + 256]
                v196, v197 = v96([v90[0] for v90 in v182])
                v198, v199 = v96([v90[1] for v90 in v182])
                v44.v203((v98(v196, v197) * v98(v198, v199)).v124(-1).v216().v207())
            return v165.v183(v44)
        v102 = v165.v137([v90[2] for v90 in v35])
        v103 = v151(v35)
        v85 = v128(v165.v138(-0.2, 0.95, 60), key=lambda v152: ((v103 >= v152).v213(v14) == v102).v184())
        v104 = v165.v137([v90[2] for v90 in v36])
        v105 = v151(v36)
        v106 = v139(((v105 >= v85).v213(v14) == v104).v184())

        @v129.v95()
        def z(v152):
            v185, v92 = v96([v152])
            return v98(v185, v92)[0]
        v107 = v139(v165.v184([v139(v158.v214(v217(v20), v217(v21), dim=-1)) for v20, v21 in v215.v208]))
        v108 = v139(v165.v184([v139(v158.v214(v217(v20), v217(v21), dim=-1)) for v20, v21 in v215.v209]))
        return {'paws_acc': v106, 'thr': v139(v85), 'para': v107, 'hard': v108, 'inversion': v107 > v108}
    v38 = v109(v110, 'curve')
    v63(f"curve: paws={v38['paws_acc']:.3f} para={v38['para']:.3f} hard={v38['hard']:.3f} inv={v38['inversion']}")
    v39 = v109(v111, 'gpt')
    v63(f"gpt:   paws={v39['paws_acc']:.3f} para={v39['para']:.3f} hard={v39['hard']:.3f} inv={v39['inversion']}")
    v40 = v38['paws_acc'] >= 0.7
    v41 = v38['inversion']
    v42 = v38['paws_acc'] >= v39['paws_acc'] - 0.03
    if v40 and v41:
        v112 = 'SEM_B_CONFIRMED'
    elif v40:
        v112 = 'SEM_B_PARTIAL'
    else:
        v112 = 'SEM_B_NO'
    v43 = {'g_paws': v40, 'g_inversion': v41, 'g_vs_gpt': v42}
    v44 = {'timestamp': v210.v204(v211.v205).v153(), 'protocol': 'semantic_paws_202', 'overall': v112, 'gates': v43, 'curve': v38, 'gpt_baseline': v39, 'lexical_overlap_test_acc': v37, 'note': 'FROZEN P1 encoder + attention-pool semantic head trained on PAWS; non-destructive (generation/memory/calibration untouched); tests B CAPABILITY, not free emergence'}
    v2.v62(v186.v154(v44, indent=2, ensure_ascii=False), encoding='utf-8')
    v3.v62('\n'.v155(['# Stage202 — B capability via PAWS (frozen encoder + attention head)', '', f'**Overall:** `{v112}`', '', f"- curve: PAWS test acc **{v38['paws_acc']:.3f}** | 179 para {v38['para']:.3f} / hard {v38['hard']:.3f} (**inversion={v38['inversion']}**)", f"- gpt baseline: PAWS acc {v39['paws_acc']:.3f} | para {v39['para']:.3f} / hard {v39['hard']:.3f} (inv={v39['inversion']})", f'- lexical-overlap baseline: {v37:.3f} (PAWS adversarial ~chance)', '', f'gates: {v43}', '', 'Non-destructive: P1 frozen; head is a separate branch. Confirms whether the curve representation CAN encode meaning over spelling given a meaning signal.']), encoding='utf-8')
    v63(f"[202] {v112} | curve paws={v38['paws_acc']:.3f} inv={v38['inversion']} (para {v38['para']:.2f}/hard {v38['hard']:.2f})")
    return 0
if v45 == '__main__':
    raise v113(v156())