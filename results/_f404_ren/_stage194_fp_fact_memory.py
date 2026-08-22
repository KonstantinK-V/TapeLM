"""
Stage 194 — FP fact memory: entity recall via episodic fingerprint slots.

North-star attack: entity cloze is at chance for ALL systems (LM path = weight
memorization = needs billions of params). FP path (old SOTE SoftPhraseMemory,
hop1 only): while READING text, write slots
    key = normalize(mean fp(context words around entity)),  value = entity
At exam: query = normalize(mean fp(question context words));
score(candidate) = max cos(query, key) over slots whose value == candidate.

Zero training; fp = frozen 191-P1 arc encoder (as 192/193).

Gates:
  G1 acc >= 0.50 on entity items (chance 0.25; night models 0.27-0.30)
  G2 falsification control: memory built WITHOUT the read tail → acc <= 0.35
     (proves answers come from reading, not priors)

  python _stage194_fp_fact_memory.py
"""
from __future__ import annotations
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
v0 = v17('results')
v1 = v17('data')
v2 = v17('checkpoints/stage191_p1_curve.pt')
v3 = v17('data/_wikitext103_train.txt')
v4 = v0 / 'stage194_decision.json'
v5 = v0 / 'stage194_mini.md'
v6 = v0 / '_stage194_log.txt'
v7 = v1 / 'stage191_exam_v3.jsonl'
v8 = 194
v9 = v18.v10
v11 = 150000000
v12 = 3000000
from _tape_index import CTX_WIN, context_words
v13 = v53.v19('\\b([A-Z][a-z]{3,}|\\d{3,4})\\b')
v14 = v53.v19('[A-Za-z][a-z]{2,}')

def log(v20: v54) -> None:
    v21 = v20 if v20.v102('\n') else v20 + '\n'
    try:
        v103(v21, end='', flush=True)
    except v55:
        v103(v21.v168('ascii', 'replace').v119('ascii'), end='', flush=True)
    v6.v104.v56(parents=True, exist_ok=True)
    with v6.v105('a', encoding='utf-8') as v57:
        v57.v106(v21)

class FpBank:

    def __init__(v58, v38, v59, v36):
        v58.v38 = v38
        v58.v59 = v59
        v58.v36 = v36
        v58.v60: v15[v54, v64.v22] = {}

    @v64.v63()
    def fp(v58, v61: v113[v54]) -> v64.v22:
        v62 = [v107 for v107 in v61 if v107 not in v58.v60]
        if v62:
            v108 = v64.v117(v137(v62), 1, v9, dtype=v64.v151)
            for v71, v107 in v73(v62):
                for v152, v153 in v73(v107[:v9]):
                    v108[v71, 0, v152] = v58.v59.v131(v153, 0)
            v51 = v138.v112(v58.v38.v160(v108.v89(v58.v36))[:, 0], dim=-1)
            for v107, v72 in v136(v62, v51):
                v58.v60[v107] = v72
        return v64.v109([v58.v60[v107] for v107 in v61], 0)

    @v64.v63()
    def ctx_fp(v58, v43: v54, v65: v54 | None=None) -> v64.v22 | None:
        from _tape_index import CONTEXT_WORD_MIN, context_words
        v61 = v110(v43, exclude=v65)
        if v137(v61) < v111:
            return None
        return v138.v112(v58.v161(v61).v139(0), dim=-1)

def build_memory(v23: v113[v54], v24: v66, v25: v54) -> v29[v64.v22, v113[v54]]:
    v67, v33 = ([], [])
    v26 = v68.v68()
    for v27 in v23:
        for v69 in v13.v114(v27):
            v115 = v69.v140(1)
            v141, v142 = (v145(0, v69.v169() - v162), v154(v137(v27), v69.v170() + v162))
            v116 = v24.v120(v27[v141:v142], exclude=v115)
            if v116 is not None:
                v67.v118(v116)
                v33.v118(v115)
    v28 = v64.v109(v67, 0) if v67 else v64.v117(0, 256, device=v24.v36)
    v70(f'  memory[{v25}]: slots={v137(v33)} ({v68.v68() - v26:.0f}s)')
    return (v28, v33)

def score_entity_items(v30, v31, v32, v24: v66, v28: v64.v22, v33: v113[v54]) -> v15:
    v34: v15[v54, v113[v16]] = {}
    for v71, v72 in v73(v33):
        v34.v155(v72, []).v118(v71)
    v74, v75, v76 = (0, 0, 0)
    for v35 in v30:
        if v35['type'] != 'entity':
            continue
        v77 = v31.v119(v35['ctx_ids'], skip_special_tokens=False)
        v78 = v24.v120(v77)
        if v78 is None:
            continue
        v79 = v28 @ v78 if v137(v33) else v64.v117(0)
        v80 = []
        for v81 in v35['cand_ids']:
            v121 = v31.v119(v81, skip_special_tokens=False).v143()
            v122 = v13.v144(v121)
            v121 = v122.v140(1) if v122 else v121
            v123 = v34.v131(v121, [])
            v80.v118(v163(v79[v123].v145()) if v123 else -1.0)
        if v145(v80) <= -1.0:
            v76 += 1
            continue
        v74 += v16(v16(v171.v164(v80)) == v35['gold_idx'])
        v75 += 1
    return {'acc': v74 / v145(1, v75), 'n': v75, 'abstain': v76}

def main() -> v16:
    v0.v56(parents=True, exist_ok=True)
    v6.v82('', encoding='utf-8')
    v70(f'Stage194 start {v166.v158(v167.v159).v132()}')
    v70('FP fact memory (episodic slots, zero training) vs entity cloze')
    v36 = v64.v36('cuda' if v64.v156.v146() else 'cpu')
    v26 = v68.v68()
    v83, v84, v59, v85 = v86()
    v31 = v124.v87(v54(v18.v125))
    v37 = v31.v88()
    v32 = v31.v126(v127) or 0
    v38 = v147(v85, v37).v89(v36)
    v38.v90(v64.v148(v2, map_location=v36, weights_only=False)['model'])
    v38.v91()
    v24 = v66(v38, v59, v36)
    with v3.v105('r', encoding='utf-8', errors='ignore') as v57:
        v43 = v57.v128(v11)
    v39 = v43[-v12:]
    v40 = [v27.v143() for v27 in v39.v157('\n') if 120 < v137(v27.v143()) < 1000][:1200]
    v41 = v43[60000000:60000000 + v12]
    v42 = [v27.v143() for v27 in v41.v157('\n') if 120 < v137(v27.v143()) < 1000][:1200]
    del v43
    v70(f'tail paras={v137(v40)} ctrl paras={v137(v42)} ({v68.v68() - v26:.0f}s)')
    v30 = [v149.v129(v130) for v130 in v7.v165(encoding='utf-8').v150()]
    v44 = v92((1 for v35 in v30 if v35['type'] == 'entity'))
    v70(f'entity items={v44}')
    v93, v94 = v95(v40, v24, 'read-tail')
    v45 = v96(v30, v31, v32, v24, v93, v94)
    v70(f"  READ memory: acc={v45['acc']:.3f} n={v45['n']} abstain={v45['abstain']}")
    v97, v98 = v95(v42, v24, 'control-unread')
    v46 = v96(v30, v31, v32, v24, v97, v98)
    v70(f"  CONTROL memory: acc={v46['acc']:.3f} n={v46['n']} abstain={v46['abstain']}")
    v47 = None
    v48 = v0 / 'stage191_p1.json'
    if v48.v99():
        v47 = v149.v129(v48.v165(encoding='utf-8'))['exam'].v131('entity_acc')
    v49 = v45['acc'] >= 0.5 and v45['n'] >= 50
    v50 = v46['acc'] <= 0.35 or v46['n'] < 20
    if v49 and v50:
        v100 = 'FACT_MEMORY_YES'
    elif v45['acc'] >= 0.35 and v50:
        v100 = 'FACT_MEMORY_WEAK'
    else:
        v100 = 'FACT_MEMORY_NO'
    v51 = {'timestamp': v166.v158(v167.v159).v132(), 'protocol': 'fp_fact_memory_194', 'overall': v100, 'read_memory': v45, 'control_unread_memory': v46, 'chance': 0.25, 'lm_baseline_p1_entity': v47, 'slots': {'read': v137(v94), 'control': v137(v98)}, 'note': 'zero training; fp = frozen P1 arc_enc; hop1 retrieval (old SOTE SoftPhraseMemory style)'}
    v4.v82(v149.v133(v51, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v82('\n'.v134(['# Stage194 — FP fact memory (entity recall)', '', f'**Overall:** `{v100}`', '', f"- READ memory: acc={v45['acc']:.3f} (n={v45['n']}, abstain={v45['abstain']}) — chance 0.25, LM baseline {v47}", f"- CONTROL (unread) memory: acc={v46['acc']:.3f} (n={v46['n']}, abstain={v46['abstain']})", '']), encoding='utf-8')
    v70(f"[194] {v100} | read={v45['acc']:.3f} ctrl={v46['acc']:.3f} chance=0.25 lm={v47}")
    return 0
if v52 == '__main__':
    raise v101(v135())