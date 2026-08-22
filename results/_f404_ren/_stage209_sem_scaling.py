"""
Stage 209 — on 3050: refute "variant A is structurally blind to meaning"?

Two probes (same PAWS harness as 202):
  1) Scaling grid: CE-pretrained curve + matched GPT at d128/2L, d192/4L, d256/6L (P1/P2 ckpt).
     Frozen encoder + PAWS attention head. Gates: monotonic curve PAWS acc; parity vs GPT each scale.
  2) Teacher sufficiency (MiniLM-L6-v2): train head to match teacher cosine (MSE) from frozen states.
     Gate: curve teacher-corr >= gpt - 0.05 at d256 (is semantic info in the substrate?).

  python _stage209_sem_scaling.py
"""
from __future__ import annotations
from datasets import load_dataset
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
from transformers import GPT2Config, GPT2LMHeadModel
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
import _stage191_night as s191
from _stage191_night import PAD, SelfModelXL, load_data, lr_at, sample_windows, span_logprob_x
from _stage202_semantic_paws import SemHead, lexical_overlap
v0 = v21('results')
v1 = v21('checkpoints')
v2 = v1 / 'stage191_p1_curve.pt'
v3 = v1 / 'stage191_p2_gpt.pt'
v4 = v0 / 'stage209_decision.json'
v5 = v0 / 'stage209_mini.md'
v6 = v0 / '_stage209_log.txt'
v7 = 209
v8 = 64
v9 = 24
v10 = 0.0003
v11 = 0.05
v12 = 3
v13 = 128
v14 = 0.0005
v15 = 0.4
v16 = 3
v17 = 2500
v18 = v71.v22('[A-Za-z]+')
v19 = ({'d': 128, 'L': 2, 'steps': 3200, 'budget_s': 720, 'gpt_heads': 4}, {'d': 192, 'L': 4, 'steps': 4200, 'budget_s': 900, 'gpt_heads': 4}, {'d': 256, 'L': 6, 'steps': 0, 'budget_s': 0, 'gpt_heads': 8})

def log(v23: v72) -> None:
    v24 = v23 if v23.v140('\n') else v23 + '\n'
    try:
        v141(v24, end='', flush=True)
    except v73:
        v141(v24.v172('ascii', 'replace').v227('ascii'), end='', flush=True)
    v6.v142.v74(parents=True, exist_ok=True)
    with v6.v143('a', encoding='utf-8') as v75:
        v75.v144(v24)

def train_curve_quick(v25, v26, v27, v28, v29, v30, v31, v32, v33):
    if v31 <= 0:
        return
    v34 = v156.v145.v76(v25.v146(), lr=v10, weight_decay=0.01)
    v35 = v147.v77(v7 + v33.v228() % 10000)
    v36 = v78.v78()
    v25.v51()
    v37 = None
    for v38 in v79(1, v31 + 1):
        for v55 in v34.v80:
            v55['lr'] = v188(v38, v31)
        v81 = v229(v26, v27, v9, v35, v29).v90(v30)
        v82 = v81 == v29
        v148, v149, v150 = v25.v151(v28[v81], v82, ids=v81)
        v83 = v81[:, 1:]
        v84 = ~v82[:, :-1] & ~v82[:, 1:]
        v85 = v189.v152(v148[:, :-1][v84], v83[v84])
        v86 = v85 + v11 * v150[~v82].v207()
        v34.v153(set_to_none=True)
        v86.v154()
        v230.v190.v155(v25.v146(), 1.0)
        v34.v38()
        v37 = v169(v85) if v37 is None else 0.95 * v37 + 0.05 * v169(v85)
        if v38 % 800 == 0 or v38 == v31:
            v125(f'    [{v33}] pretrain step {v38}/{v31} ce~{v37:.3f} ({v78.v78() - v36:.0f}s)')
        if v78.v78() - v36 > v32:
            v125(f'    [{v33}] pretrain budget stop @ {v38}')
            break
    v25.v87()
    v156.v88({'model': v25.v191()}, v1 / f'stage209_{v33}.pt')

def train_gpt_quick(v39, v40, v41, v42, v29, v26, v27, v30, v31, v32, v33):
    if v31 <= 0:
        return
    v43 = v89(vocab_size=v42, n_positions=v8, n_embd=v39, n_layer=v40, n_head=v41, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1)
    v44 = v192(v43).v90(v30)
    v34 = v156.v145.v76(v44.v146(), lr=v10, weight_decay=0.01)
    v35 = v147.v77(v7 + 17 + v39)
    v36 = v78.v78()
    v44.v51()
    for v38 in v79(1, v31 + 1):
        for v55 in v34.v80:
            v55['lr'] = v188(v38, v31)
        v81 = v229(v26, v27, v9, v35, v29).v90(v30)
        v86 = v44(input_ids=v81, labels=v81).v86
        v34.v153(set_to_none=True)
        v86.v154()
        v34.v38()
        if v38 % 800 == 0 or v38 == v31:
            v125(f'    [{v33}] gpt step {v38}/{v31} loss={v169(v86):.3f} ({v78.v78() - v36:.0f}s)')
        if v78.v78() - v36 > v32:
            break
    v44.v87()
    v156.v88({'model': v44.v191(), 'conf': v43.v193()}, v1 / f'stage209_{v33}.pt')
    return v44

def load_gpt_ckpt(v45, v30):
    v46 = v156.v91(v45, map_location=v30, weights_only=False)
    v43 = v89(**v46['conf']) if 'conf' in v46 else None
    if v43 is None:
        v43 = v89(**v46.v231('config', {}))
    v44 = v192(v43).v90(v30)
    v44.v92(v46['model'])
    v44.v87()
    return v44

def paws_eval(v47, v48, v28, v29, v49, v50, v30, v51, v52, v53, v35):
    v93, v94, v95 = (v51, v52, v53)

    def curve_states(v96, v97):
        v81 = [[v159 for v159 in v49.v172(v194).v81 if v159 != v29][:v8] or [v29] for v194 in v97]
        v98 = v157((v211(v99) for v99 in v81))
        v99 = v156.v158((v211(v81), v98), v29, dtype=v156.v195, device=v30)
        for v159, v160 in v161(v81):
            v99[v159, :v211(v160)] = v156.v196(v160, device=v30)
        v82 = v99 == v29
        with v156.v103():
            v162 = v96.v162(v96.v232(v28[v99], v99), pad_mask=v82)
        return (v162, ~v82)

    @v156.v103()
    def gpt_states(v44, v97):
        v81 = [[v159 for v159 in v49.v172(v194).v81 if v159 != v29][:v8] or [v29] for v194 in v97]
        v98 = v157((v211(v99) for v99 in v81))
        v99 = v156.v158((v211(v81), v98), v29, dtype=v156.v195, device=v30)
        v100 = v156.v163((v211(v81), v98), dtype=v156.v197, device=v30)
        for v159, v160 in v161(v81):
            v99[v159, :v211(v160)] = v156.v196(v160, device=v30)
            v100[v159, :v211(v160)] = True
        v101 = v44.v198(input_ids=v99, attention_mask=v100.v195()).v102
        return (v101, v100)

    def train_head(v104, v33):
        v105 = v233(v50).v90(v30)
        v34 = v156.v145.v76(v105.v146(), lr=v14, weight_decay=0.01)
        v106 = v164(v79(v211(v93)))
        for v107 in v79(v12):
            v35.v199(v106)
            for v165 in v79(0, v211(v106), v13):
                v200 = [v93[v159] for v159 in v106[v165:v165 + v13]]
                v234, v235 = v104([v99[0] for v99 in v200])
                v236, v237 = v104([v99[1] for v99 in v200])
                v201 = v156.v196([v99[2] for v99 in v200], dtype=v156.v169, device=v30)
                v238, v239 = (v105(v234, v235), v105(v236, v237))
                v202 = (v238 * v239).v240(-1)
                v86 = (v201 * (1 - v202) + (1 - v201) * v189.v260(v202 - v15)).v207()
                v34.v153(set_to_none=True)
                v86.v154()
                v34.v38()
        v105.v87()

        @v156.v103()
        def cos_of(v166):
            v69 = []
            for v165 in v79(0, v211(v166), 256):
                v203 = v166[v165:v165 + 256]
                v234, v235 = v104([v99[0] for v99 in v203])
                v236, v237 = v104([v99[1] for v99 in v203])
                v69.v183((v105(v234, v235) * v105(v236, v237)).v240(-1).v241().v173())
            return v205.v204(v69)
        v108 = v205.v167([v99[2] for v99 in v94])
        v109 = v168(v94)
        v110 = v157(v205.v206(-0.2, 0.95, 60), key=lambda v170: ((v109 >= v170).v255(v20) == v108).v207())
        v111 = v205.v167([v99[2] for v99 in v95])
        v112 = v169(((v168(v95) >= v110).v255(v20) == v111).v207())

        def zsent(v170):
            v208, v100 = v104([v170])
            return v105(v208, v100)[0]
        v113 = v169(v205.v207([v169(v189.v256(v259(v257), v259(v165), dim=-1)) for v257, v165 in v258.v248]))
        v114 = v169(v205.v207([v169(v189.v256(v259(v257), v259(v165), dim=-1)) for v257, v165 in v258.v249]))
        return {'paws_acc': v112, 'para': v113, 'hard': v114, 'inversion': v113 > v114}
    v54 = v115(lambda v194: v209(v47, v194), 'curve')
    v55 = v115(lambda v194: v210(v48, v194), 'gpt')
    return (v54, v55)

def teacher_probe(v47, v48, v28, v29, v49, v50, v30, v56, v51, v53, v35):
    """Train head to match MiniLM cosine; report Pearson r on test."""
    v57 = v35.v116(v51, v171(v17, v211(v51)))
    v117, v118 = ([v99[0] for v99 in v57], [v99[1] for v99 in v57])
    with v156.v103():
        v119 = v56.v172(v117, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
        v120 = v56.v172(v118, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
        v121 = v189.v256(v119, v120, dim=-1).v241().v173()

    def curve_states(v97):
        v81 = [[v159 for v159 in v49.v172(v194).v81 if v159 != v29][:v8] or [v29] for v194 in v97]
        v98 = v157((v211(v99) for v99 in v81))
        v99 = v156.v158((v211(v81), v98), v29, dtype=v156.v195, device=v30)
        for v159, v160 in v161(v81):
            v99[v159, :v211(v160)] = v156.v196(v160, device=v30)
        v82 = v99 == v29
        with v156.v103():
            v162 = v47.v162(v47.v232(v28[v99], v99), pad_mask=v82)
        return (v162, ~v82)

    @v156.v103()
    def gpt_states(v44, v97):
        v81 = [[v159 for v159 in v49.v172(v194).v81 if v159 != v29][:v8] or [v29] for v194 in v97]
        v98 = v157((v211(v99) for v99 in v81))
        v99 = v156.v158((v211(v81), v98), v29, dtype=v156.v195, device=v30)
        v100 = v156.v163((v211(v81), v98), dtype=v156.v197, device=v30)
        for v159, v160 in v161(v81):
            v99[v159, :v211(v160)] = v156.v196(v160, device=v30)
            v100[v159, :v211(v160)] = True
        v101 = v48.v198(input_ids=v99, attention_mask=v100.v195()).v102
        return (v101, v100)

    def fit(v104):
        v105 = v233(v50).v90(v30)
        v34 = v156.v145.v76(v105.v146(), lr=v14)
        for v107 in v79(v16):
            v174 = v35.v116(v79(v211(v57)), v211(v57))
            for v165 in v79(0, v211(v174), v13):
                v212 = v174[v165:v165 + v13]
                v200 = [v57[v159] for v159 in v212]
                v234, v235 = v104([v99[0] for v99 in v200])
                v236, v237 = v104([v99[1] for v99 in v200])
                v213 = (v105(v234, v235) * v105(v236, v237)).v240(-1)
                v86 = v189.v242(v213, v156.v196(v121[v212], device=v30))
                v34.v153(set_to_none=True)
                v86.v154()
                v34.v38()
        v105.v87()

        @v156.v103()
        def pred_split(v166):
            v214, v215 = ([], [])
            for v165 in v79(0, v211(v166), 128):
                v203 = v166[v165:v165 + 128]
                v216 = [v99[0] for v99 in v203]
                v217 = [v99[1] for v99 in v203]
                v234, v235 = v104(v216)
                v236, v237 = v104(v217)
                v214.v183((v105(v234, v235) * v105(v236, v237)).v240(-1).v241().v173())
                v119 = v56.v172(v216, batch_size=64, show_progress_bar=False)
                v120 = v56.v172(v217, batch_size=64, show_progress_bar=False)
                v215.v183(v205.v167([v169(v189.v256(v156.v196(v119[v159]), v156.v196(v120[v159]), dim=0)) for v159 in v79(v211(v216))]))
            return (v205.v204(v214), v205.v204(v215))
        v135, v170 = v175(v53)
        v122 = v169(v205.v250(v135, v170)[0, 1]) if v211(v135) > 2 else 0.0
        return v122
    v58 = v123(lambda v194: v209(v194))
    v59 = v123(lambda v194: v210(v48, v194))
    return {'curve_teacher_r': v58, 'gpt_teacher_r': v59}

def main() -> v20:
    v0.v74(parents=True, exist_ok=True)
    v1.v74(parents=True, exist_ok=True)
    v6.v124('', encoding='utf-8')
    v125(f'Stage209 start {v252.v246(v253.v247).v184()}')
    v125('scaling + MiniLM teacher probe: is A structurally blocked on meaning?')
    v30 = v156.v30('cuda' if v156.v243.v218() else 'cpu')
    v35 = v147.v77(v7)
    v156.v126(v7)
    v36 = v78.v78()
    v26, v27, v127, v128 = v129()
    v49 = v176.v130(v72(v219.v177))
    v42 = v49.v131()
    v29 = v49.v178(v179) or 0
    v28 = v244.v220(v49, v127, v29, v42).v90(v30)
    v60 = v132('paws', 'labeled_final')
    v51 = [(v122['sentence1'], v122['sentence2'], v20(v122['label'])) for v122 in v60['train']]
    v52 = [(v122['sentence1'], v122['sentence2'], v20(v122['label'])) for v122 in v60['validation']]
    v53 = [(v122['sentence1'], v122['sentence2'], v20(v122['label'])) for v122 in v60['test']]
    v125(f'PAWS loaded ({v78.v78() - v36:.0f}s)')
    try:
        from sentence_transformers import SentenceTransformer
        v56 = v180('all-MiniLM-L6-v2', device=v72(v30))
        v125(f'MiniLM teacher on {v30} ({v78.v78() - v36:.0f}s)')
    except v133 as e:
        v125(f'ERROR: need sentence-transformers: {v251}')
        return 1
    v61 = []
    v62 = None
    for v63 in v19:
        v39, v40 = (v63['d'], v63['L'])
        v33 = f'd{v39}_L{v40}'
        v125(f'=== scale {v33} ===')
        v134 = v245(v128, v42, d=v39, n_layers=v40).v90(v30)
        if v39 == 256 and v40 == 6:
            v134.v92(v156.v91(v2, map_location=v30, weights_only=False)['model'])
            v181 = v221(v3, v30)
            v125(f'  loaded P1 + P2 ckpts ({v78.v78() - v36:.0f}s)')
        else:
            v222(v134, v26, v27, v28, v29, v30, v63['steps'], v63['budget_s'], f'curve_{v33}')
            v134.v92(v156.v91(v1 / f'stage209_curve_{v33}.pt', map_location=v30, weights_only=False)['model'])
            v223(v39, v40, v63['gpt_heads'], v42, v29, v26, v27, v30, v63['steps'], v63['budget_s'], f'gpt_{v33}')
            v181 = v221(v1 / f'stage209_gpt_{v33}.pt', v30)
        for v135 in v134.v146():
            v135.v224(False)
        v50 = v39
        v54, v55 = v182(v134, v181, v28, v29, v49, v50, v30, v51, v52, v53, v35)
        v125(f"  PAWS curve={v54['paws_acc']:.3f} gpt={v55['paws_acc']:.3f} para/hard {v54['para']:.2f}/{v54['hard']:.2f}")
        v136 = {'scale': v33, 'd': v39, 'L': v40, 'curve': v54, 'gpt': v55, 'delta': v54['paws_acc'] - v55['paws_acc']}
        v61.v183(v136)
        if v39 == 256:
            v62 = v225(v134, v181, v28, v29, v49, v50, v30, v56, v51, v53, v35)
            v125(f"  teacher r: curve={v62['curve_teacher_r']:.3f} gpt={v62['gpt_teacher_r']:.3f}")
    v64 = [v122['curve']['paws_acc'] for v122 in v61]
    v65 = v64[2] + 0.02 >= v64[1] >= v64[0] - 0.02
    v66 = v137((v122['curve']['paws_acc'] >= v122['gpt']['paws_acc'] - 0.03 for v122 in v61))
    v67 = v62 and v62['curve_teacher_r'] >= v62['gpt_teacher_r'] - 0.05
    v68 = v62 and v62['curve_teacher_r'] >= 0.25
    if v66 and v65 and v67 and v68:
        v138 = 'STRUCTURAL_BLOCK_NO'
    elif v66 and (v65 or v64[2] >= 0.65):
        v138 = 'STRUCTURAL_BLOCK_UNLIKELY'
    else:
        v138 = 'STRUCTURAL_BLOCK_UNCLEAR'
    v69 = {'timestamp': v252.v246(v253.v247).v184(), 'protocol': 'sem_scaling_teacher_209', 'overall': v138, 'grid': v61, 'teacher_d256': v62, 'gates': {'g_monotone_paws': v65, 'g_parity_all_scales': v66, 'g_teacher_parity': v67, 'g_teacher_signal': v68}, 'interpretation': "STRUCTURAL_BLOCK_NO means: curve PAWS tracks GPT at every scale, accuracy tends to rise with scale, and curve states carry MiniLM geometry at least as well as GPT — 3050 cannot prove B at scale, but refutes 'A can never be meaningful'."}
    v4.v124(v226.v185(v69, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v124('\n'.v186([f'# Stage209 — scaling + teacher probe\n\n**Overall:** `{v138}`\n', '| scale | curve PAWS | gpt PAWS | Δ |', '|-------|------------|----------|---|'] + [f"| {v122['scale']} | {v122['curve']['paws_acc']:.3f} | {v122['gpt']['paws_acc']:.3f} | {v122['delta']:+.3f} |" for v122 in v61] + (['', f"Teacher Pearson r @ d256: curve **{v62['curve_teacher_r']:.3f}** vs gpt {v62['gpt_teacher_r']:.3f}", f'gates: mono={v65} parity={v66} teacher={v67} signal={v68}'] if v62 else [])), encoding='utf-8')
    v125(f'[209] {v138} | paws {[v254(v99, 3) for v99 in v64]} | gates mono={v65} parity={v66} teacher={v67}')
    return 0
if v70 == '__main__':
    raise v139(v187())