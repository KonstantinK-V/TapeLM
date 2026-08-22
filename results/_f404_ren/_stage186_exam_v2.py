"""
Stage 186 — Exam v2: kill the unigram shortcut.

v1 flaw (found in 185): next_tok gold is usually a frequent token, distractors random
→ pure frequency scores 0.65. v2: distractors are FREQUENCY-MATCHED to gold
(nearest ranks in the corpus frequency table). Entity distractors matched by
entity-frequency rank too.

Gates:
  EXAM2_OK  = unigram <= chance+0.10 (shortcut dead) AND GPT >= chance+0.20 (still calibrated)

Systems: unigram, random, ce_gpt_181, endpoint_185, tape_185.

  python _stage186_exam_v2.py
"""
from __future__ import annotations
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage181_ce_control as s181
import _stage184_exam_logprob as s184
import _stage185_tape_read as s185
v0 = v20('results')
v1 = v20('data')
v2 = v20('checkpoints')
v3 = v0 / '_stage186_log.txt'
v4 = v1 / 'stage186_exam_v2.jsonl'
v5 = v0 / 'stage186_decision.json'
v6 = v0 / 'stage186_mini.md'
v7 = v21.v7
v8 = 186
v9 = 40
v10 = 150
v11 = 100
v12 = 60
v13 = 4
v14 = 800000
v15 = 40
v16 = '[PAD]'
v17 = v78.v22('\\b([A-Z][a-z]{3,}|\\d{3,4})\\b')

def log(v23: v79) -> None:
    v24 = v23 if v23.v127('\n') else v23 + '\n'
    try:
        v128(v24, end='', flush=True)
    except v80:
        v128(v24.v185('ascii', 'replace').v172('ascii'), end='', flush=True)
    v3.v129.v81(parents=True, exist_ok=True)
    with v3.v130('a', encoding='utf-8') as v82:
        v82.v131(v24)

def freq_matched_pool(v25: v50[v19], v26: v18[v19, v19], v27: v19, v28: v132.v83, v29: v19):
    """k distractors with frequency rank near gold's."""
    v30 = v26[v27]
    v31 = v84(0, v30 - v15)
    v32 = v85(v133(v25), v30 + v15 + 1)
    v33 = [v86 for v86 in v25[v31:v32] if v86 != v27]
    v28.v87(v33)
    return v33[:v29]

def build_exam_v2(v34: v79, v35: v88, v36: v19, v37: v134.v89, v28: v132.v83) -> v50[v18]:
    v38 = [v43.v158() for v43 in v34.v173('\n') if 120 < v133(v43.v158()) < 1000][:1200]
    v39 = v38[v19(0.8 * v133(v38)):]
    v40 = ' '.v171(v38[:v19(0.8 * v133(v38))])[:200000].v90()
    v91(f'  paras={v133(v38)} hold={v133(v39)}')
    v25 = v50(v134.v135(-v37))
    v25 = [v19(v86) for v86 in v25 if v19(v86) != v36]
    v26 = {v86: v92 for v92, v86 in v159(v25)}
    v41 = []
    v42 = 0
    for v43 in v39 * 3:
        if v42 >= v10:
            break
        v93 = [v92 for v92 in v35.v185(v43).v93 if v92 != v36]
        if v133(v93) < v9 + 2:
            continue
        v94 = v28.v136(v9, v133(v93) - 2)
        v95 = v93[v84(0, v94 - v9):v94]
        v27 = v93[v94]
        v96 = v137(v25, v26, v27, v28, v13 - 1)
        if v133(v96) < v13 - 1:
            continue
        v97 = [[v27]] + [[v174] for v174 in v96]
        v98 = v50(v160(v133(v97)))
        v28.v87(v98)
        v41.v138({'type': 'next_tok', 'ctx_ids': v95, 'cand_ids': [v97[v29] for v29 in v98], 'gold_idx': v98.v175(0)})
        v42 += 1
    v44: v18[v79, v19] = {}
    for v43 in v38:
        for v99 in v17.v139(v43):
            v44[v99.v141(1)] = v44.v144(v99.v141(1), 0) + 1
    v45 = [v100 for v100, v161 in v116(v44.v41(), key=lambda v189: -v189[1])]
    v46 = {v100: v92 for v92, v100 in v159(v45)}
    v47 = 0
    for v43 in v39 * 2:
        if v47 >= v11:
            break
        v99 = v17.v140(v43, 60)
        if not v99:
            continue
        v101 = v99.v141(1)
        if v101 not in v46:
            continue
        v95 = [v92 for v92 in v35.v185(v43[:v99.v190()]).v93 if v92 != v36][-v9:]
        if v133(v95) < 8:
            continue
        v30 = v46[v101]
        v31, v32 = (v84(0, v30 - v15), v85(v133(v45), v30 + v15 + 1))
        v102 = [v100 for v100 in v45[v31:v32] if v100 != v101]
        v28.v87(v102)
        v96 = v102[:v13 - 1]
        v103 = [v92 for v92 in v35.v185(' ' + v101).v93 if v92 != v36]
        if v133(v96) < v13 - 1 or not v103:
            continue
        v104 = [v103] + [[v92 for v92 in v35.v185(' ' + v174).v93 if v92 != v36] for v174 in v96]
        v98 = v50(v160(v133(v104)))
        v28.v87(v98)
        v41.v138({'type': 'entity', 'ctx_ids': v95, 'cand_ids': [v104[v29] for v29 in v98], 'gold_idx': v98.v175(0)})
        v47 += 1
    v48 = [v82 for v82 in ['Zorblax', 'Quenith', 'Marbune', 'Xaldera', 'Kessari', 'Vornak', 'Talmidex', 'Orsiphon', 'Pholmar', 'Girenth'] if v82.v90() not in v40]
    v49 = 0
    for v43 in v39:
        if v49 >= v12 or v133(v48) < v13:
            break
        v99 = v17.v140(v43, 60)
        if not v99:
            continue
        v95 = [v92 for v92 in v35.v185(v43[:v99.v190()]).v93 if v92 != v36][-v9:]
        if v133(v95) < 8:
            continue
        v105 = v28.v142(v48, v13)
        v104 = [[v92 for v92 in v35.v185(' ' + v162).v93 if v92 != v36] for v162 in v105]
        v41.v138({'type': 'ood', 'ctx_ids': v95, 'cand_ids': v104, 'gold_idx': v28.v136(0, v13 - 1)})
        v49 += 1
    return v41

def score_with(v51, v41) -> v18:
    v52 = {}
    for v53 in v41:
        v106 = [v51(v53['ctx_ids'], v117) for v117 in v53['cand_ids']]
        v107 = v19(v134.v163(v106))
        v143, v108 = v52.v144(v53['type'], (0, 0))
        v52[v53['type']] = (v143 + v19(v107 == v53['gold_idx']), v108 + 1)
    v54 = {}
    for v86, (v143, v108) in v52.v41():
        v54[f'{v86}_acc'] = v143 / v84(1, v108)
        v54[f'{v86}_n'] = v108
    return v54

def main() -> v19:
    v55 = v145.v109()
    v55.v110('--device', default='cuda' if v146.v186.v176() else 'cpu')
    v56 = v55.v111()
    v0.v81(parents=True, exist_ok=True)
    v3.v112('', encoding='utf-8')
    v91(f'Stage186 start {v187.v183(v188.v184).v155()}')
    v91('Exam v2: frequency-matched distractors')
    v28 = v132.v83(v8)
    v57 = v146.v57(v56.v57)
    v35 = v88.v113(v79(v7))
    v58 = v35.v114()
    v36 = v35.v147(v16) or 0
    v34 = v148.v115(max_chars=20000000)
    v59 = v116(v164(v34) | {' '})
    v60 = ['<pad>'] + v59
    v61 = {v117: v92 + 1 for v92, v117 in v159(v59)}
    v62 = v149.v118(v35, v34)
    v63 = v62[:v19(0.8 * v133(v62))] or v62
    v37 = v134.v119(v58)
    for v64 in v63:
        for v86 in v64:
            v37[v86] += 1
    v91(f'docs={v133(v62)} V={v58}')
    v41 = v120(v34[:v14], v35, v36, v37, v28)
    with v4.v130('w', encoding='utf-8') as v82:
        for v53 in v41:
            v82.v131(v170.v156(v53, ensure_ascii=False) + '\n')
    v65 = {v86: v150((1 for v92 in v41 if v92['type'] == v86)) for v86 in ('next_tok', 'entity', 'ood')}
    v91(f'exam v2 n={v133(v41)} {v65}')
    v66 = {}
    v67 = v134.v91(v37 / v37.v150())
    v66['unigram'] = v121(lambda v117, v165: v166(v134.v177([v67[v86] for v86 in v165])), v41)
    v68 = v132.v83(0)
    v66['random'] = v121(lambda v117, v165: v68.v132(), v41)
    v91('score ce_gpt_181 …')
    v69 = v151.v122(v57)
    v66['ce_gpt_181'] = v121(lambda v117, v165: v151.v167(v69, v117, v165, v57), v41)
    v70 = v178.v168(v35, v61, v36, v58).v123(v57)
    for v71 in ('endpoint', 'tape'):
        v124 = v2 / f'stage185_{v71}.pt'
        if not v124.v169():
            v91(f'skip {v71} (no ckpt)')
            continue
        v91(f'score {v71}_185 …')
        v99 = v178.v179(v133(v60), v58, v71).v123(v57)
        v99.v152(v146.v180(v124, map_location=v57, weights_only=False)['model'])
        v99.v153()
        v66[f'{v71}_185'] = v121(lambda v117, v165, v181=v99: v178.v182(v181, v70, v36, v117, v165, v57), v41)
        if v71 == 'tape':
            v154 = v121(lambda v117, v165, v181=v99: v178.v182(v181, v70, v36, v117, v165, v57, shuffle_tape=True), [v53 for v53 in v41 if v53['type'] == 'next_tok'])
            v66['tape_185']['next_tok_shuffled'] = v154['next_tok_acc']
            v91(f"  tape shuffle ablation: next_tok={v154['next_tok_acc']:.3f}")
    v72 = 1.0 / v13
    for v125, v30 in v66.v41():
        v91(f"  {v125}: next_tok={v30.v144('next_tok_acc', 0):.3f} entity={v30.v144('entity_acc', 0):.3f} ood={v30.v144('ood_acc', 0):.3f}")
    v73 = v66['unigram']['next_tok_acc'] <= v72 + 0.1
    v74 = v66['ce_gpt_181']['next_tok_acc'] >= v72 + 0.2
    v75 = 'EXAM2_OK' if v73 and v74 else 'EXAM2_SHORTCUT_ALIVE' if not v73 else 'EXAM2_GPT_LOST_SIGNAL'
    v54 = {'timestamp': v187.v183(v188.v184).v155(), 'protocol': 'exam_v2_freq_matched_186', 'overall': v75, 'chance': v72, 'gates': {'unigram<=chance+0.10': v73, 'gpt>=chance+0.20': v74}, 'counts': v65, 'results': v66, 'note': 'context credit is now score minus chance (unigram shortcut removed)'}
    v5.v112(v170.v156(v54, indent=2, ensure_ascii=False), encoding='utf-8')
    v76 = ['# Stage186 — Exam v2 (freq-matched distractors)', '', f'**Overall:** `{v75}`  chance={v72:.2f}', '']
    for v125, v30 in v66.v41():
        v76.v138(f"- `{v125}`: next_tok={v30.v144('next_tok_acc', 0):.3f} entity={v30.v144('entity_acc', 0):.3f} ood={v30.v144('ood_acc', 0):.3f}")
    v6.v112('\n'.v171(v76) + '\n', encoding='utf-8')
    v91(f'[186] {v75}')
    return 0
if v77 == '__main__':
    raise v126(v157())