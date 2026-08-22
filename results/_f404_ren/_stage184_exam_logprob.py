"""
Stage 184 — Exam calibration via LOG-PROB (fix 183's broken cos scorer).

Two cloze flavors:
  1) next_tok  — next-token multiple choice. CALIBRATION: an LM must beat chance.
                 If GPT fails this, the harness lies → stop, fix harness.
  2) entity    — mask a content span (entity/number); candidates are real spans;
                 score by log-prob of the span in place (dataset-answer question).
  + OOD entity — gold span never in corpus; should stay ~chance.

Scoring = length-normalized log-prob of the candidate given context.
Systems with a vocab head: ce_gpt_181, hybrid_182. (dual_180 has no head → deferred to S2.)
random_uniform baseline = chance.

  python _stage184_exam_logprob.py
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
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage180_dual_channel as s180
import _stage182_slow_ce_tape as s182
v0 = v22('results')
v1 = v22('data')
v2 = v22('checkpoints')
v3 = v0 / '_stage184_log.txt'
v4 = v1 / 'stage184_exam.jsonl'
v5 = v0 / 'stage184_decision.json'
v6 = v0 / 'stage184_mini.md'
v7 = v0 / 'plan_curve_dynamics.md'
v8 = v23.v8
v9 = 184
v10 = 40
v11 = 120
v12 = 100
v13 = 60
v14 = 4
v15 = 800000
v16 = '[PAD]'

def log(v24: v17) -> None:
    v25 = v24 if v24.v147('\n') else v24 + '\n'
    try:
        v148(v25, end='', flush=True)
    except v83:
        v148(v25.v191('ascii', 'replace').v189('ascii'), end='', flush=True)
    v3.v149.v84(parents=True, exist_ok=True)
    with v3.v150('a', encoding='utf-8') as v85:
        v85.v151(v25)

def write_json(v26: v22, v27: v20) -> None:
    v26.v149.v84(parents=True, exist_ok=True)
    v26.v86(v178.v152(v27, indent=2, ensure_ascii=False), encoding='utf-8')

def load_text_fast(v28: v21) -> v17:
    v26 = v87.v29
    if not v26.v153():
        raise v154(v26)
    with v26.v150('r', encoding='utf-8', errors='ignore') as v85:
        return v85.v155(v28)
v18 = v88.v30('\\b([A-Z][a-z]{3,}|\\d{3,4})\\b')

def build_exam(v31: v17, v32: v89, v33: v21, v34: v156.v90) -> v49[v20]:
    v35 = [v41.v179() for v41 in v31.v190('\n') if 120 < v91(v41.v179()) < 1000][:1200]
    v36 = v91(v35)
    v37 = v35[v21(0.8 * v36):]
    v38 = v35[:v21(0.8 * v36)]
    v39 = ' '.v176(v38)[:200000].v92()
    v93(f'  paras={v36} hold={v91(v37)}')
    v40 = v94()
    for v41 in v38[:300]:
        v40.v157(v32.v191(v41).v97)
    v40.v95(v33)
    v42 = [v96 for v96 in v40 if v96 != v33]
    v43 = []
    v44 = 0
    for v41 in v37:
        if v44 >= v11:
            break
        v97 = [v96 for v96 in v32.v191(v41).v97 if v96 != v33]
        if v91(v97) < v10 + 2:
            continue
        v98 = v34.v158(v10, v91(v97) - 2)
        v51 = v97[v171(0, v98 - v10):v98]
        v99 = v97[v98]
        v100 = []
        for v101 in v159(30):
            v160 = v42[v34.v158(0, v91(v42) - 1)]
            if v160 != v99 and v160 not in v100:
                v100.v163(v160)
            if v91(v100) >= v14 - 1:
                break
        if v91(v100) < v14 - 1:
            continue
        v102 = [[v99]] + [[v160] for v160 in v100]
        v103 = 0
        v104 = v49(v159(v91(v102)))
        v34.v161(v104)
        v102 = [v102[v113] for v113 in v104]
        v103 = v104.v162(0)
        v43.v163({'type': 'next_tok', 'ctx_ids': v51, 'cand_ids': v102, 'gold_idx': v103})
        v44 += 1
    v45 = 0
    v46 = []
    for v41 in v37:
        for v105 in v18.v164(v41):
            v46.v163(v105.v167(1))
    v46 = v49(v20.v165(v46))
    for v41 in v37:
        if v45 >= v12:
            break
        v105 = v18.v166(v41, 60)
        if not v105:
            continue
        v106 = v105.v167(1)
        v107 = v41[:v105.v192()]
        v51 = [v96 for v96 in v32.v191(v107).v97 if v96 != v33][-v10:]
        if v91(v51) < 8:
            continue
        v108 = [v96 for v96 in v32.v191(' ' + v106).v97 if v96 != v33]
        v109 = []
        for v101 in v159(40):
            v168 = v46[v34.v158(0, v91(v46) - 1)]
            if v168 != v106 and v168 not in v109:
                v109.v163(v168)
            if v91(v109) >= v14 - 1:
                break
        if v91(v109) < v14 - 1 or not v108:
            continue
        v52 = [v108] + [[v96 for v96 in v32.v191(' ' + v160).v97 if v96 != v33] for v160 in v109]
        v104 = v49(v159(v91(v52)))
        v34.v161(v104)
        v52 = [v52[v113] for v113 in v104]
        v103 = v104.v162(0)
        v43.v163({'type': 'entity', 'ctx_ids': v51, 'cand_ids': v52, 'gold_idx': v103})
        v45 += 1
    v47 = [v85 for v85 in ['Zorblax', 'Quenith', 'Marbune', 'Xaldera', 'Kessari', 'Vornak', 'Talmidex', 'Orsiphon', 'Pholmar', 'Girenth'] if v85.v92() not in v39]
    v48 = 0
    for v41 in v37:
        if v48 >= v13 or v91(v47) < v14:
            break
        v105 = v18.v166(v41, 60)
        if not v105:
            continue
        v51 = [v96 for v96 in v32.v191(v41[:v105.v192()]).v97 if v96 != v33][-v10:]
        if v91(v51) < 8:
            continue
        v110 = v34.v169(v47, v14)
        v52 = [[v96 for v96 in v32.v191(' ' + v180).v97 if v96 != v33] for v180 in v110]
        v103 = v34.v158(0, v14 - 1)
        v43.v163({'type': 'ood', 'ctx_ids': v51, 'cand_ids': v52, 'gold_idx': v103})
        v48 += 1
    return v43

@v116.v59()
def gpt_span_logprob(v50, v51: v49[v21], v52: v49[v21], v53) -> v19:
    v54 = v51 + v52
    v55 = v116.v111([v54], device=v53)
    v56 = v50(input_ids=v55).v56[0]
    v57 = v170.v112(v56, dim=-1)
    v58 = 0.0
    for v113, v114 in v115(v52):
        v98 = v91(v51) + v113 - 1
        v58 += v19(v57[v98, v114])
    return v58 / v171(1, v91(v52))

@v116.v59()
def hybrid_span_logprob(v50, v32, v60, v33, v51: v49[v21], v52: v49[v21], v53) -> v19:
    v54 = v51 + v52
    v55 = v116.v111([v54], device=v53)
    v61 = v55 == v33
    v62 = v193.v181(v32, v55, v60, v33).v117(v53)
    v101, v101, v118 = v50.v119(v62, v61)
    v56 = v50.v120(v118[0])
    v57 = v170.v112(v56, dim=-1)
    v58 = 0.0
    for v113, v114 in v115(v52):
        v98 = v91(v51) + v113 - 1
        v58 += v19(v57[v98, v114])
    return v58 / v171(1, v91(v52))

def score_system(v63, v64, v43) -> v20:
    v65 = {'next_tok': [0, 0], 'entity': [0, 0], 'ood': [0, 0]}
    for v96, v121 in v115(v43):
        if v96 % 50 == 0:
            v93(f'    {v63} {v96}/{v91(v43)}')
        v122 = [v64(v121['ctx_ids'], v182) for v182 in v121['cand_ids']]
        v123 = v21(v194.v183(v122))
        v124 = v121['type']
        v65[v124][1] += 1
        v65[v124][0] += v21(v123 == v121['gold_idx'])
    v66 = {}
    for v124, (v172, v36) in v65.v43():
        v66[f'{v124}_acc'] = v172 / v171(1, v36)
        v66[f'{v124}_n'] = v36
    v66['chance'] = 1.0 / v14
    return v66

def load_gpt(v53):
    v67 = v116.v125(v2 / 'stage181_ce_control.pt', map_location=v53, weights_only=False)
    v50 = v184(v195(**v67['conf'])).v117(v53)
    v50.v126(v67['model'])
    v50.v127()
    return v50

def load_hybrid(v53):
    v26 = v2 / 'stage182_slow_ce_tape.pt'
    if not v26.v153():
        return (None, None)
    v67 = v116.v125(v26, map_location=v53, weights_only=False)
    v60 = v67.v128('stoi')
    if not v60:
        v129 = v116.v125(v2 / 'stage180_dual_channel.pt', map_location=v53, weights_only=False)
        v60 = v129['stoi']
    v68 = v171(v60.v185()) + 1
    v69 = v89.v134(v17(v8)).v130()
    v50 = v193.v186(v68, v69).v117(v53)
    v50.v126(v67['model'], strict=True)
    v50.v127()
    return (v50, v60)

def main() -> v21:
    v70 = v173.v131()
    v70.v132('--device', default='cuda' if v116.v200.v196() else 'cpu')
    v71 = v70.v133()
    v0.v84(parents=True, exist_ok=True)
    v1.v84(parents=True, exist_ok=True)
    v3.v86('', encoding='utf-8')
    v93(f'Stage184 start {v201.v198(v202.v199).v175()}')
    v93('Exam calibration via log-prob; GPT-beats-chance gate')
    v34 = v156.v90(v9)
    v53 = v116.v53(v71.v53)
    v32 = v89.v134(v17(v8))
    v33 = v32.v174(v16) or 0
    v31 = v135(v15)
    v93(f'text_chars={v91(v31)}')
    v43 = v136(v31, v32, v33, v34)
    with v4.v150('w', encoding='utf-8') as v85:
        for v121 in v43:
            v85.v151(v178.v152(v121, ensure_ascii=False) + '\n')
    v72 = v137((1 for v96 in v43 if v96['type'] == 'next_tok'))
    v73 = v137((1 for v96 in v43 if v96['type'] == 'entity'))
    v74 = v137((1 for v96 in v43 if v96['type'] == 'ood'))
    v93(f'exam n={v91(v43)} next_tok={v72} entity={v73} ood={v74}')
    v75 = {}
    v93('load ce_gpt_181 …')
    v76 = v138(v53)
    v75['ce_gpt_181'] = v139('gpt', lambda v182, v187: v188(v76, v182, v187, v53), v43)
    v140, v60 = v141(v53)
    if v140 is not None:
        v93('load hybrid_182 …')
        v75['hybrid_182'] = v139('hybrid', lambda v182, v187: v197(v140, v32, v60, v33, v182, v187, v53), v43)
    else:
        v93('skip hybrid_182 (no ckpt)')
    v77 = v156.v90(0)
    v75['random'] = v139('random', lambda v182, v187: v77.v156(), v43)
    for v63, v142 in v75.v43():
        v93(f"  {v63}: next_tok={v142['next_tok_acc']:.3f} entity={v142['entity_acc']:.3f} ood={v142['ood_acc']:.3f} (chance={v142['chance']:.2f})")
    v78 = v75['ce_gpt_181']['next_tok_acc']
    v79 = 1.0 / v14
    v80 = v78 >= v79 + 0.2
    if not v80:
        v143 = 'HARNESS_STILL_BROKEN'
    else:
        v144 = []
        for v63, v142 in v75.v43():
            if v63 == 'random':
                continue
            if v142['entity_acc'] >= v79 + 0.1 and v142['ood_acc'] <= v79 + 0.1:
                v144.v163(v63)
        v143 = 'CALIBRATED_ENTITY_SIGNAL:' + ','.v176(v144) if v144 else 'CALIBRATED_NO_ENTITY_SIGNAL'
    v66 = {'timestamp': v201.v198(v202.v199).v175(), 'protocol': 'exam_logprob_184', 'overall': v143, 'calibrated': v80, 'gate': 'GPT next_tok >= chance+0.20', 'chance': v79, 'counts': {'next_tok': v72, 'entity': v73, 'ood': v74}, 'results': v75, 'note': 'next_tok = harness calibration (LM must win). entity = dataset-answer. ood must stay ~chance.', 'next': 'If HARNESS_STILL_BROKEN: fix scorer/data before any curve claim. If CALIBRATED: entity_acc is now a trustworthy dataset-answer number; proceed to S2 (addressable tape).'}
    v145(v5, v66)
    v81 = ['# Stage184 — log-prob exam calibration', '', f'**Overall:** `{v143}`', '', f'chance={v79:.2f}', '']
    for v63, v142 in v75.v43():
        v81.v163(f"- `{v63}`: next_tok={v142['next_tok_acc']:.3f} entity={v142['entity_acc']:.3f} ood={v142['ood_acc']:.3f}")
    v81 += ['', v66['next'], '']
    v6.v86('\n'.v176(v81), encoding='utf-8')
    v93(f'[184] {v143}')
    return 0
if v82 == '__main__':
    raise v146(v177())