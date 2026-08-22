"""
Stage 179 — Harden retention (178) + Gate B (meaning vs form).

Continue the retention objective from Stage178, longer train.
Track:
  A) same last BPE piece / different prefix (must stay PASS-ish)
  B) paraphrase proximity vs random vs hard spelling cousins
     (174-style: is z about meaning or just form?)

Optional mid anneal of retention weight to see if A holds without constant push.

  python _stage179_curve_harden_B.py
  python _stage179_curve_harden_B.py --steps 10000
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
v0 = v27('results')
v1 = v27('checkpoints')
v2 = v0 / '_stage179_log.txt'
v3 = v0 / 'stage179_decision.json'
v4 = v0 / 'stage179_mini.md'
v5 = v28.v6
v6 = v1 / 'stage179_curve_harden.pt'
v7 = v29.v7
v8 = v0 / 'plan_curve_dynamics.md'
v9 = 179
v10 = v29.v10
v11 = v29.v11
v12 = 16
v13 = 8
v14 = 0.0002
v15 = 1500
v16 = 10000
v17 = 4500
v18 = 1.5
v19 = 0.4
v20 = [('The cat sat on the mat.', 'A cat was sitting on the mat.'), ('She quickly opened the door.', 'She opened the door quickly.'), ('He bought a new car yesterday.', 'Yesterday he purchased a new automobile.'), ('The weather is very cold today.', 'It is extremely chilly outside today.'), ('Children are playing in the park.', 'Kids are playing at the park.'), ('I need to finish this work soon.', 'I must complete this task shortly.'), ('The book was written by a famous author.', 'A famous writer wrote the book.'), ('They arrived at the station early.', 'They got to the station early.'), ('Please close the window.', 'Could you shut the window?'), ('The train leaves at noon.', 'The train departs at midday.'), ('He is afraid of spiders.', 'Spiders scare him.'), ('She teaches mathematics at school.', 'She is a math teacher at the school.'), ('The film was long and boring.', 'The movie was lengthy and dull.'), ('We should start the meeting now.', "Let's begin the meeting now."), ('The river flows into the sea.', 'The river runs into the ocean.'), ('His answer was completely wrong.', 'His reply was totally incorrect.'), ('The store opens at nine.', 'The shop opens at 9.'), ('Birds fly south in winter.', 'In winter birds migrate south.'), ('She drank a cup of tea.', 'She had a cup of tea.'), ('The problem is difficult to solve.', 'Solving the problem is hard.'), ('He forgot his keys at home.', 'He left his keys at home.'), ('The sun rises in the east.', 'In the east the sun comes up.'), ('The bridge connects the two cities.', 'The two cities are linked by the bridge.'), ('Water boils at one hundred degrees.', 'Water boils at 100 degrees.'), ('The dog chased the ball across the yard.', 'Across the yard the dog ran after the ball.')]
v21 = [('The cat sat on the mat.', 'The car sat on the mat.'), ('She opened the door quickly.', 'She opened the book quickly.'), ('He bought a new car yesterday.', 'He bought a new cat yesterday.'), ('The weather is very cold today.', 'The weather is very warm today.'), ('Children are playing in the park.', 'Children are studying in the park.'), ('The train leaves at noon.', 'The plane leaves at noon.'), ('Water boils at one hundred degrees.', 'Oil boils at one hundred degrees.'), ('She teaches mathematics at school.', 'She teaches history at school.')]

def log(v30: v73) -> None:
    v31 = v30 if v30.v122('\n') else v30 + '\n'
    try:
        v123(v31, end='', flush=True)
    except v74:
        v123(v31.v195('ascii', 'replace').v181('ascii'), end='', flush=True)
    v2.v124.v75(parents=True, exist_ok=True)
    with v2.v125('a', encoding='utf-8') as v76:
        v76.v126(v31)

def write_json(v32: v27, v33: v25) -> None:
    v32.v124.v75(parents=True, exist_ok=True)
    v32.v77(v168.v127(v33, indent=2, ensure_ascii=False), encoding='utf-8')

class GateWrap(v34.v22):

    def __init__(v78, v79: v28.v128):
        v182().v129()
        v78.v79 = v79

    def forward_states(v78, v43, v80=None):
        return v78.v79.v130(v78.v79.v169(v43), v80)

def cos(v35: v46.v24, v36: v46.v24) -> v23:
    return v23(v134.v131(v134.v86(v35, dim=0), v134.v86(v36, dim=0), dim=0))

@v46.v45()
def encode_text_states(v37: v81, v38: v82, v39: v73, v40: v25, v41) -> v46.v24:
    v42 = v29.v83(v38, v39)
    if not v42:
        v42 = ['.']
    v42 = v42[-v11:]
    v43 = v29.v190(v42, v40).v170(0).v84(v41)
    v44 = v46.v85(1, v132(v42), dtype=v46.v133, device=v41)
    return v37.v130(v43, pad_mask=v44)[0]

def z_summary(v47: v46.v24) -> v46.v24:
    return v134.v86(v46.v135([v47[-1], v47.v172(0)], 0), dim=0)

@v46.v45()
def gate_B(v37: v81, v38: v82, v40: v25, v41, v48: v136.v87) -> v25:
    v49 = []
    v50 = []
    for v35, v36 in v20:
        v88 = v137(v171(v37, v38, v35, v40, v41))
        v89 = v137(v171(v37, v38, v36, v40, v41))
        v49.v138(v140(v88, v89))
        v50.v139([v88, v89])
    v51 = [v140(v137(v171(v37, v38, v35, v40, v41)), v137(v171(v37, v38, v36, v40, v41))) for v35, v36 in v21]
    v52 = []
    for v53 in v90(v132(v49) * 4):
        v141, v142 = v48.v143(v90(v132(v50)), 2)
        v52.v138(v140(v50[v141], v50[v142]))
    v91, v92, v93 = (v23(v183.v172(v49)), v23(v183.v172(v52)), v23(v183.v172(v51)))
    v94, v95 = (v91 - v92, v91 - v93)
    if v94 > 0.05 and v95 > 0.03:
        v96 = 'B_PASS_MEANING_STRUCTURE'
    elif v94 > 0.03 and v95 <= 0.02:
        v96 = 'B_FAIL_FORM_NOT_MEANING'
    elif v95 <= 0.0 and v94 > 0.02:
        v96 = 'B_FAIL_FORM_NOT_MEANING'
    elif v94 <= 0.02:
        v96 = 'B_FAIL_NO_PARAPHRASE_CLUSTER'
    else:
        v96 = 'B_WEAK_MIXED'
    return {'verdict': v96, 'mean_cos_paraphrase': v91, 'mean_cos_random': v92, 'mean_cos_hard_spelling': v93, 'lift_vs_random': v94, 'lift_vs_hard': v95, 'n_para': v132(v49)}

def main() -> v26:
    v54 = v144.v97()
    v54.v98('--steps', type=v26, default=v16)
    v54.v98('--device', default='cuda' if v46.v191.v184() else 'cpu')
    v54.v98('--from-scratch', action='store_true')
    v55 = v54.v99()
    v0.v75(parents=True, exist_ok=True)
    v1.v75(parents=True, exist_ok=True)
    v2.v77('', encoding='utf-8')
    v100(f'Stage179 start {v193.v188(v194.v189).v165()}')
    v100('Harden 178 retention + gate B (paraphrase vs hard spelling)')
    v100(f'plan={v8}')
    if not v7.v145():
        raise v146(v7)
    v38 = v82.v101(v73(v7))
    v39 = v147.v102(max_chars=20000000)
    v56 = v103(v173(v39) | {' '})
    v57 = ['<pad>'] + v56
    v40 = {v104: v141 + 1 for v141, v104 in v174(v56)}
    v58 = v29.v105(v38, v39)
    v59 = v58[v26(0.8 * v132(v58)):] or v58[-100:]
    v60 = v58[:v26(0.8 * v132(v58))] or v58
    v61 = v28.v106(v60)
    v100(f'docs={v132(v58)} same-last={v132(v61)} V={v38.v185()}')
    v41 = v46.v41(v55.v41)
    v46.v107(v9)
    v136.v108(v9)
    v37 = v28.v128(v132(v57)).v84(v41)
    v62 = 0
    if v5.v145() and (not v55.v148):
        v109 = v46.v149(v5, map_location=v41, weights_only=False)
        v37.v150(v109['model'], strict=True)
        v62 = v26(v109.v175('step', 0))
        v100(f'loaded {v5} step={v62}')
    else:
        v100('training from scratch (no 178 ckpt)')
    v63 = v81(v37).v84(v41)
    v64 = v46.v151.v110(v37.v152(), lr=v14, weight_decay=0.0001)
    v48 = v136.v87(v9)
    v37.v111()
    v65 = v29.v112(v63, v59, v40, v41, v136.v87(v9))
    v66 = v113(v63, v38, v40, v41, v136.v87(v9 + 1))
    v100(f"  init A: same={v65['mean_cos_same_last_piece']:.3f} diff={v65['mean_cos_diff_last_piece']:.3f} → {v65['verdict']}")
    v100(f"  init B: para={v66['mean_cos_paraphrase']:.3f} rand={v66['mean_cos_random']:.3f} hard={v66['mean_cos_hard_spelling']:.3f} lift_r={v66['lift_vs_random']:+.3f} lift_h={v66['lift_vs_hard']:+.3f} → {v66['verdict']}")
    v67 = []
    v114, v115 = (v65, v66)
    v68 = None
    v37.v60()
    for v69 in v90(1, v55.v119 + 1):
        v116 = v18 if v69 < v17 else v19
        v153, v44 = v29.v154(v60, v40, v12, v48, v41)
        v155, v156 = v28.v157(v37, v153, v44)
        v117 = v28.v158(v61, v40, v13, v48, v41)
        if v117 is not None:
            v159, v176 = v28.v177(v37, v117)
            v159 = v159 * (v116 / v28.v186)
            v160 = v155 + v159
            v156.v178(v176)
        else:
            v160 = v155
            v156['ret_cos'] = 1.0
        v64.v161(set_to_none=True)
        v160.v162()
        v34.v179.v163(v37.v152(), 1.0)
        v64.v69()
        v68 = v23(v160.v187()) if v68 is None else 0.95 * v68 + 0.05 * v23(v160.v187())
        if v69 % v15 == 0 or v69 == v55.v119:
            v37.v111()
            v114 = v29.v112(v63, v59, v40, v41, v136.v87(v9 + v69))
            v115 = v113(v63, v38, v40, v41, v136.v87(v9 + v69 + 7))
            v164 = {'step': v69, 'w_ret': v116, 'A_same': v114['mean_cos_same_last_piece'], 'A_diff': v114['mean_cos_diff_last_piece'], 'A': v114['verdict'], 'B_para': v115['mean_cos_paraphrase'], 'B_rand': v115['mean_cos_random'], 'B_hard': v115['mean_cos_hard_spelling'], 'B': v115['verdict']}
            v67.v138(v164)
            v100(f"  step {v69}: loss~{v68:.3f} w_ret={v116:.2f} ret_cos={v156.v175('ret_cos', 0):.3f} past={v156.v175('cos_past', 0):.3f} inst={v156.v175('cos_inst', 0):.3f} A_same={v164['A_same']:.3f} A_diff={v164['A_diff']:.3f}→{v164['A']} | B para={v164['B_para']:.3f} rand={v164['B_rand']:.3f} hard={v164['B_hard']:.3f}→{v164['B']}")
            v37.v60()
            v46.v180({'model': v37.v192(), 'stoi': v40, 'step': v62 + v69, 'A': v114, 'B': v115}, v6)
    v70 = v114['mean_cos_same_last_piece'] < 0.9 and 'FAIL' not in v114['verdict']
    if 'PASS' in v115['verdict'] and v70:
        v118 = 'HARDEN_A_YES_B_YES'
    elif v70 and 'WEAK' in v115['verdict']:
        v118 = 'HARDEN_A_YES_B_WEAK'
    elif v70:
        v118 = 'HARDEN_A_YES_B_FAIL'
    elif 'PASS' in v115['verdict']:
        v118 = 'HARDEN_A_FAIL_B_YES'
    else:
        v118 = 'HARDEN_A_FAIL_B_FAIL'
    v71 = {'timestamp': v193.v188(v194.v189).v165(), 'protocol': 'curve_harden_B_179', 'overall': v118, 'steps': v55.v119, 'loaded_178': v5.v145() and (not v55.v148), 'anneal': {'at': v17, 'w_ret_before': v18, 'w_ret_after': v19}, 'init_A': v65, 'init_B': v66, 'final_A': v114, 'final_B': v115, 'history': v67, 'note': 'A=prefix visible; B=paraphrase vs hard spelling (meaning vs form).', 'next': 'If A_YES_B_FAIL: retention≠meaning — need semantic/instance channel, not more A soak. If B_YES: scale carefully + longer context probes. If A collapses after anneal: retention still a crutch.'}
    v120(v3, v71)
    v4.v77('\n'.v166(['# Stage179 — harden + gate B', '', f'**Overall:** `{v118}`', '', f"- A init→final: {v65['mean_cos_same_last_piece']:.3f}→{v114['mean_cos_same_last_piece']:.3f} ({v114['verdict']})", f"- B: {v115['verdict']} para={v115['mean_cos_paraphrase']:.3f} rand={v115['mean_cos_random']:.3f} hard={v115['mean_cos_hard_spelling']:.3f}", f"- lift_rand={v115['lift_vs_random']:+.3f} lift_hard={v115['lift_vs_hard']:+.3f}", f"- {v71['next']}", '']), encoding='utf-8')
    v100(f'[179] {v118}')
    return 0
if v72 == '__main__':
    raise v121(v167())