"""
Stage 183 — Fast dataset-answer exam (not LM ceiling).

Build small cloze / doc-link / OOD packs from wiki hold.
Score frozen backbones by embedding similarity (no long probe train):
  - cloze: pick candidate that best matches context state (cos)
  - doc-link: same-doc pairs should be closer than cross-doc
  - OOD: cloze with answers never in train — should stay near chance

Systems: ce_gpt_181, dual_180, hybrid_182 (if ckpt).

Speed: N small, encode once, vectorized scoring.

  python _stage183_dataset_answer_exam.py
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
v0 = v23('results')
v1 = v23('data')
v2 = v23('checkpoints')
v3 = v0 / '_stage183_log.txt'
v4 = v1 / 'stage183_exam.jsonl'
v5 = v0 / 'stage183_decision.json'
v6 = v0 / 'stage183_mini.md'
v7 = v0 / 'plan_stage183_dataset_answer.md'
v8 = v24.v8
v9 = 183
v10 = 48
v11 = 40
v12 = 24
v13 = 16
v14 = '[PAD]'
v15 = 800000

def load_text_fast(v25: v22) -> v16:
    """Avoid read_text() of full multi‑100MB wiki."""
    v26 = v99.v27
    if not v26.v164():
        raise v165(v26)
    with v26.v166('r', encoding='utf-8', errors='ignore') as v100:
        return v100.v167(v25)

def log(v28: v16) -> None:
    v29 = v28 if v28.v168('\n') else v28 + '\n'
    try:
        v169(v29, end='', flush=True)
    except v101:
        v169(v29.v236('ascii', 'replace').v219('ascii'), end='', flush=True)
    v3.v170.v102(parents=True, exist_ok=True)
    with v3.v166('a', encoding='utf-8') as v100:
        v100.v171(v29)

def write_json(v26: v23, v30: v21) -> None:
    v26.v170.v102(parents=True, exist_ok=True)
    v26.v103(v203.v172(v30, indent=2, ensure_ascii=False), encoding='utf-8')
v17 = v104.v31('\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+){0,2})\\b')
v18 = v104.v31('\\b(\\d{3,4})\\b')

def split_paras(v32: v16) -> v34[v16]:
    v33 = [v42.v173() for v42 in v32.v204('\n\n') if v111(v42.v173()) > 80]
    return v33

def pick_answer(v35: v16) -> v109[v16, v16] | None:
    """Return (context_with_mask, answer) or None."""
    for v36 in (v17, v18):
        v105 = v34(v36.v205(v35))
        if not v105:
            continue
        v106 = v174(v105, key=lambda v55: v111(v55.v175(1)))
        v107 = v106.v175(1)
        if v111(v107) < 3:
            continue
        v108 = v35[:v106.v237()] + ' [MASK] ' + v35[v106.v231():]
        if v111(v108) < 20:
            continue
        return (v108.v173(), v107)
    return None

def build_exam(v32: v16, v37: v176.v110) -> v34[v21]:
    v33 = [v42.v173() for v42 in v32.v204('\n\n') if 100 < v111(v42.v173()) < 800][:500]
    if v111(v33) < 40:
        v33 = [v32[v48:v48 + 300] for v48 in v119(0, v180(v111(v32), 120000), 300)]
    v38 = v111(v33)
    v39 = v33[v22(0.8 * v38):]
    v40 = v33[:v22(0.8 * v38)]
    v41 = '\n'.v201(v40[:100])[:50000].v112()
    v113(f'  paras={v38} hold={v111(v39)}')
    v46, v114 = ([], [])
    for v42 in v39:
        v35 = v42.v206('\n', ' ')[:200]
        v106 = v104.v177('\\b([A-Z][a-z]{3,})\\b', v35)
        if not v106:
            v106 = v104.v177('\\b(\\d{3,4})\\b', v35)
        if not v106:
            continue
        v107 = v106.v175(1)
        v108 = (v35[:v106.v237()] + ' [MASK] ' + v35[v106.v231():]).v173()
        v46.v178({'context': v108, 'gold': v107})
        v114.v178(v107)
        if v111(v46) >= v11 * 2:
            break
    v37.v115(v46)
    v113(f'  cloze_raw={v111(v46)}')
    v43 = []
    v44 = 0
    for v45 in v46:
        if v44 >= v11:
            break
        v116 = v45['gold']
        v117 = []
        for v49 in v119(20):
            v179 = v114[v37.v181(0, v111(v114) - 1)]
            if v179 != v116 and v179 not in v117:
                v117.v178(v179)
            if v111(v117) >= 3:
                break
        if v111(v117) < 3:
            continue
        v118 = [v116] + v117
        v37.v115(v118)
        v43.v178({'type': 'cloze', 'context': v45['context'], 'gold': v116, 'candidates': v118, 'ood': False})
        v44 += 1
    v47 = [v100 for v100 in ['Zorblax', 'Quenith', 'Marbune', 'Xaldera', '9191', 'Kessari', 'Vornak', 'Talmidex'] if v100.v112() not in v41]
    for v48 in v119(v180(v13, v111(v46), v174(0, v111(v47) - 3))):
        v116 = v47[v48]
        v117 = v47[v48 + 1:v48 + 4]
        v118 = [v116] + v117
        v37.v115(v118)
        v43.v178({'type': 'cloze', 'context': v46[-(v48 + 1)]['context'], 'gold': v116, 'candidates': v118, 'ood': True})
    for v49 in v119(v12):
        v48 = v37.v181(0, v111(v39) - 1)
        v42 = v39[v48]
        v65, v66 = (v42[:v111(v42) // 3], v42[v111(v42) // 2:2 * v111(v42) // 3])
        if v111(v65) < 30 or v111(v66) < 30:
            continue
        v43.v178({'type': 'doc_link', 'text_a': v65, 'text_b': v66, 'same': True})
        v120 = (v48 + 1 + v37.v181(0, v174(1, v111(v39) - 2))) % v111(v39)
        v43.v178({'type': 'doc_link', 'text_a': v65, 'text_b': v39[v120][:v111(v65)], 'same': False})
    return v43

@v58.v57()
def enc_gpt(v50, v51, v32: v16, v52, v53: v22) -> v58.v19:
    v54 = [v48 for v48 in v51.v236(v32).v54 if v48 != v53][-v10:] or [1]
    v55 = v58.v121([v54], device=v52)
    v56 = v50.v207(input_ids=v55).v122[0]
    return v182.v123(v58.v183([v56[-1], v56.v210(0)], 0), dim=0)

@v58.v57()
def enc_dual(v59, v51, v60, v32: v16, v52, v61: v16='slow') -> v58.v19:
    v62 = v24.v208(v51, v32)[-v10:] or ['.']
    v63 = v24.v232(v62, v60).v209(0).v124(v52)
    v64 = v58.v125(1, v111(v62), dtype=v58.v184, device=v52)
    v49, v126, v127 = v59.v128(v63, v64)
    v56 = v127[0] if v61 == 'slow' else 0.5 * v126[0] + 0.5 * v127[0]
    return v182.v123(v58.v183([v56[-1], v56.v210(0)], 0), dim=0)

def cos(v65, v66) -> v20:
    return v20(v182.v220(v65.v209(0), v66.v209(0)).v185())

def score_cloze(v67, v43: v34[v21], v68: v21 | None=None) -> v21:
    v68 = v68 if v68 is not None else {}

    def enc(v129: v16):
        if v129 not in v68:
            v68[v129] = v67(v129)
        return v68[v129]
    v69 = v70 = v71 = v72 = 0
    v73 = [v74 for v74 in v43 if v74['type'] == 'cloze']
    for v48, v74 in v130(v73):
        if v48 % 20 == 0:
            v113(f'    cloze {v48}/{v111(v73)}')
        v108 = v155(v74['context'])
        v77 = []
        for v131 in v74['candidates']:
            v186 = v74['context'].v206('[MASK]', v131)
            v77.v178(v187(v108, v155(v186)))
        v132 = v74['candidates'][v22(v222.v221(v77))]
        if v74.v140('ood'):
            v72 += 1
            v71 += v22(v132 == v74['gold'])
        else:
            v70 += 1
            v69 += v22(v132 == v74['gold'])
    return {'cloze_in_acc': v69 / v174(1, v70), 'cloze_in_n': v70, 'cloze_ood_acc': v71 / v174(1, v72), 'cloze_ood_n': v72, 'chance': 0.25}

def score_doclink(v67, v43: v34[v21], v68: v21 | None=None) -> v21:
    v68 = v68 if v68 is not None else {}

    def enc(v129: v16):
        if v129 not in v68:
            v68[v129] = v67(v129)
        return v68[v129]
    v133, v134 = ([], [])
    for v74 in v43:
        if v74['type'] != 'doc_link':
            continue
        v135 = v187(v155(v74['text_a']), v155(v74['text_b']))
        (v133 if v74['same'] else v134).v178(v135)
    if not v133 or not v134:
        return {'doc_acc': 0.0, 'doc_n': 0, 'gap_same_minus_diff': 0.0}
    v75 = 0.5 * (v20(v222.v210(v133)) + v20(v222.v210(v134)))
    v76 = v148((1 for v135 in v133 if v135 >= v75)) + v148((1 for v135 in v134 if v135 < v75))
    v38 = v111(v133) + v111(v134)
    return {'doc_acc': v76 / v38, 'doc_n': v38, 'mean_same': v20(v222.v210(v133)), 'mean_diff': v20(v222.v210(v134)), 'gap_same_minus_diff': v20(v222.v210(v133) - v222.v210(v134))}

def verdict_for(v77: v21) -> v16:
    v78 = v77['cloze_in_acc']
    v79 = v77['cloze_ood_acc']
    v80 = v77['chance']
    if v78 >= v80 + 0.1 and v79 <= v80 + 0.05:
        return 'DATASET_ANSWER_SIGNAL'
    if v78 >= v80 + 0.1 and v79 > v80 + 0.1:
        return 'LEAK_OR_PRIOR'
    if v77.v140('doc_acc', 0) >= 0.65 and v77.v140('gap_same_minus_diff', 0) > 0.05:
        return 'DOC_BINDING_ONLY'
    return 'NO_DATASET_ANSWER'

def load_gpt(v52):
    v26 = v2 / 'stage181_ce_control.pt'
    v81 = v58.v136(v26, map_location=v52, weights_only=False)
    v82 = v137(**v81['conf'])
    v50 = v211(v82).v124(v52)
    v50.v138(v81['model'])
    v50.v139()
    return v50

def load_dual180(v52):
    v26 = v2 / 'stage180_dual_channel.pt'
    v81 = v58.v136(v26, map_location=v52, weights_only=False)
    v60 = v81.v140('stoi')
    if not v60:
        raise v188('stage180 ckpt missing stoi')
    v83 = v174(v60.v212()) + 1
    v50 = v223.v213(v83).v124(v52)
    v50.v138(v81['model'], strict=True)
    v50.v139()
    return (v50, v60)

def load_hybrid182(v52, v60):
    v26 = v2 / 'stage182_slow_ce_tape.pt'
    if not v26.v164():
        return None
    v83 = v174(v60.v212()) + 1
    v51 = v189.v141(v16(v8))
    v84 = v51.v142()
    v50 = v224.v214(v83, v84).v124(v52)
    v81 = v58.v136(v26, map_location=v52, weights_only=False)
    v50.v138(v81['model'], strict=True)
    v50.v139()
    return v50

def main() -> v22:
    v85 = v190.v143()
    v85.v144('--device', default='cuda' if v58.v233.v225() else 'cpu')
    v85.v144('--rebuild-exam', action='store_true')
    v86 = v85.v145()
    v0.v102(parents=True, exist_ok=True)
    v1.v102(parents=True, exist_ok=True)
    v3.v103('', encoding='utf-8')
    v113(f'Stage183 start {v234.v229(v235.v230).v200()}')
    v113(f'plan={v7} | fast embedding exam (no long probe)')
    v37 = v176.v110(v9)
    v113('loading wiki slice …')
    v32 = v146(v15)
    v113(f'text_chars={v111(v32)}')
    if v86.v147 or not v4.v164():
        v113('building exam …')
        v43 = v191(v32, v37)
        with v4.v166('w', encoding='utf-8') as v100:
            for v74 in v43:
                v100.v171(v203.v172(v74, ensure_ascii=False) + '\n')
        v113(f'exam built → {v4} n={v111(v43)}')
    else:
        v43 = [v203.v215(v216) for v216 in v4.v238(encoding='utf-8').v226() if v216.v173()]
        v113(f'exam loaded n={v111(v43)}')
    v87 = v148((1 for v48 in v43 if v48['type'] == 'cloze' and (not v48.v140('ood'))))
    v88 = v148((1 for v48 in v43 if v48['type'] == 'cloze' and v48.v140('ood')))
    v89 = v148((1 for v48 in v43 if v48['type'] == 'doc_link'))
    v113(f'counts cloze_in={v87} ood={v88} doc_link={v89}')
    v51 = v189.v141(v16(v8))
    v53 = v51.v192(v14) or 0
    v52 = v58.v52(v86.v52)
    v90 = {}
    v113('load ce_gpt_181 …')
    v91 = v149(v52)
    v90['ce_gpt_181'] = lambda v129: v193(v91, v51, v129, v52, v53)
    v113('load dual_180 …')
    v150, v60 = v151(v52)
    v90['dual_180'] = lambda v129: v194(v150, v51, v60, v129, v52, mode='slow')
    v92 = v152(v52, v60)
    if v92 is not None:
        v113('load hybrid_182 …')
        v90['hybrid_182'] = lambda v129: v194(v92.v217, v51, v60, v129, v52, mode='slow')
    else:
        v113('skip hybrid_182 (no ckpt)')

    def enc_rand(v129: v16):
        v153 = v58.v227().v195(v228(v129) % (2 ** 31 - 1))
        return v182.v123(v58.v218(256, generator=v153), dim=0)
    v90['random_hash'] = v93
    v94 = {}
    for v154, v155 in v90.v43():
        v113(f'score {v154} …')
        v68: v21 = {}
        v156 = v196(v155, v43, v68)
        v157 = v197(v155, v43, v68)
        v158 = {**v156, **v157}
        v158['verdict'] = v198(v158)
        v94[v154] = v158
        v113(f"  {v154}: cloze_in={v158['cloze_in_acc']:.3f} ood={v158['cloze_ood_acc']:.3f} doc={v158['doc_acc']:.3f} gap={v158.v140('gap_same_minus_diff', 0):.3f} → {v158['verdict']}")
    v95 = [v159 for v159, v162 in v94.v43() if v159 != 'random_hash' and v162['verdict'] == 'DATASET_ANSWER_SIGNAL']
    if v95:
        v160 = 'EXAM_SIGNAL_PRESENT'
    elif v199((v162['verdict'] == 'DOC_BINDING_ONLY' for v159, v162 in v94.v43() if v159 != 'random_hash')):
        v160 = 'EXAM_DOC_ONLY'
    else:
        v160 = 'EXAM_NO_SIGNAL_YET'
    v96 = {'timestamp': v234.v229(v235.v230).v200(), 'protocol': 'dataset_answer_exam_183_fast', 'overall': v160, 'exam': v16(v4), 'counts': {'cloze_in': v87, 'ood': v88, 'doc_link': v89}, 'results': v94, 'note': 'Win = cloze_in≫chance and OOD~chance. LM ceiling (CE/ablation) is NOT this exam.', 'next': 'If NO_SIGNAL on all: strengthen fact-oriented non-text teacher or richer exam. If only GPT signals: principle gap. If dual/hybrid signal: north star alive.'}
    v161(v5, v96)
    v97 = ['# Stage183 — dataset-answer exam (fast)', '', f'**Overall:** `{v160}`', '']
    for v159, v162 in v94.v43():
        v97.v178(f"- `{v159}`: cloze_in={v162['cloze_in_acc']:.3f} ood={v162['cloze_ood_acc']:.3f} doc={v162['doc_acc']:.3f} → **{v162['verdict']}**")
    v97 += ['', v96['next'], '']
    v6.v103('\n'.v201(v97), encoding='utf-8')
    v113(f'[183] {v160}')
    return 0
if v98 == '__main__':
    raise v163(v202())