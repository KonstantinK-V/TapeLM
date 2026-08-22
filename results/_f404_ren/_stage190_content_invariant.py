"""
Stage 190 — S4: content invariant on slow (attack MEANING).

Base = 187 self-model (best curve config). Add ONE representation objective:
doc-level InfoNCE on the slow endpoint — two windows of the SAME doc should map
close, windows of other docs far. This is the only natural "same content, different
surface" supervision the corpus gives for free (no handcrafted paraphrases).

Per the 185 rule, a representation loss is allowed ONLY if it survives the A/B:
  G1 next_tok(v2) >= 0.727 - 0.03      (CE not poisoned — else revert)
  G2 doc-link: same-doc vs cross-doc pairing acc > 187 baseline (invariant learned)
  G3 gate B (179 pairs): (hard - para) gap shrinks vs 187 baseline;
     strong win = para > hard (meaning beats form — never achieved before)

  python _stage190_content_invariant.py
  python _stage190_content_invariant.py --steps 3000
"""
from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage181_ce_control as s181
import _stage185_tape_read as s185
import _stage187_self_model as s187
v0 = v27('results')
v1 = v27('data')
v2 = v27('checkpoints')
v3 = v0 / '_stage190_log.txt'
v4 = v0 / 'stage190_decision.json'
v5 = v0 / 'stage190_mini.md'
v6 = v1 / 'stage186_exam_v2.jsonl'
v7 = v2 / 'stage190_content_invariant.pt'
v8 = v2 / 'stage187_self_model.pt'
v9 = v28.v9
v10 = 185
v11 = v28.v11
v12 = 16
v13 = 8
v14 = 0.2
v15 = 0.2
v16 = 0.1
v17 = 0.0003
v18 = 1000
v19 = 3000
v20 = 60
v21 = '[PAD]'
v22 = 0.727

def log(v29: v82) -> None:
    v30 = v29 if v29.v139('\n') else v29 + '\n'
    try:
        v140(v30, end='', flush=True)
    except v83:
        v140(v30.v215('ascii', 'replace').v204('ascii'), end='', flush=True)
    v3.v141.v84(parents=True, exist_ok=True)
    with v3.v142('a', encoding='utf-8') as v85:
        v85.v143(v30)

def slow_endpoint(v31: v144.v86, v32: v38.v23, v33, v34) -> v38.v23:
    v35 = v32 == v34
    v36 = v31.v87(v33[v32])
    v88, v89, v89 = v31.v88(v36, v35)
    v37 = (~v35).v186(dim=1).v90(min=1)
    return v88[v38.v98(v32.v151(0), device=v32.v41), v37 - 1]

def sample_doc_windows(v39, v40, v34, v41, v42=v13):
    """Two disjoint windows per doc: one from first half, one from second half."""
    v91, v92 = ([], [])
    v43 = 0
    while v187(v91) < v42 and v43 < 200:
        v43 += 1
        v93 = v39[v40.v145(0, v187(v39) - 1)]
        if v187(v93) < v11 + 16:
            continue
        v94 = v187(v93) // 2
        v95 = v40.v145(0, v155(0, v94 - v11 // 2))
        v96 = v40.v145(v94, v155(v94, v187(v93) - v11 // 2))

        def pack(v146):
            v147 = v93[v146:v146 + v11]
            if v187(v147) < v11:
                v147 = v147 + [v34] * (v11 - v187(v147))
            return v147
        v91.v148(v188(v95))
        v92.v148(v188(v96))
    if not v91:
        return None
    return (v38.v149(v91, dtype=v38.v189, device=v41), v38.v149(v92, dtype=v38.v189, device=v41))

def infonce(v44: v38.v23, v45: v38.v23) -> v38.v23:
    v44 = v150.v97(v44, dim=-1)
    v45 = v150.v97(v45, dim=-1)
    v46 = v44 @ v45.v190() / v14
    v47 = v38.v98(v44.v151(0), device=v44.v41)
    return 0.5 * (v150.v176(v46, v47) + v150.v176(v46.v190(), v47))

@v38.v51()
def doclink_acc(v31, v39, v33, v34, v41, v40, v48=100) -> v24:
    v49 = 0
    v50 = 0
    while v50 < v48:
        v99 = v152(v39, v40, v34, v41, n_docs=2)
        if v99 is None:
            break
        v103, v104 = v99
        v44 = v153(v31, v103, v33, v34)
        v45 = v153(v31, v104, v33, v34)
        v100 = v150.v154(v44[0], v45[0], dim=-1)
        v101 = v150.v154(v44[0], v45[1], dim=-1)
        v49 += v26(v24(v100) > v24(v101))
        v50 += 1
    return v49 / v155(1, v50)

@v38.v51()
def gate_B_slow(v31, v52, v33, v34, v41) -> v25:

    def z_of(v60: v82) -> v38.v23:
        v32 = [v166 for v166 in v52.v215(v60).v32 if v166 != v34][-v11:]
        v102 = v38.v149([v32], dtype=v38.v189, device=v41)
        return v153(v31, v102, v33, v34)[0]

    def cos(v103, v104):
        return v24(v150.v154(v103, v104, dim=-1))
    v53 = [v156(v191(v103), v191(v104)) for v103, v104 in v192.v157]
    v54 = [v156(v191(v103), v191(v104)) for v103, v104 in v192.v158]
    return {'para': v24(v205.v193(v53)), 'hard': v24(v205.v193(v54)), 'gap_hard_minus_para': v24(v205.v193(v54) - v205.v193(v53))}

def main() -> v26:
    v55 = v159.v105()
    v55.v106('--steps', type=v26, default=v19)
    v55.v106('--device', default='cuda' if v38.v210.v206() else 'cpu')
    v56 = v55.v107()
    v0.v84(parents=True, exist_ok=True)
    v3.v108('', encoding='utf-8')
    v109(f'Stage190 start {v213.v208(v214.v209).v182()}')
    v109('S4: doc-level content invariant (InfoNCE on slow endpoint) + gate B')
    v57 = [v194.v160(v161) for v161 in v6.v211(encoding='utf-8').v195() if v161.v196()]
    v58 = [v162 for v162 in v57 if v162['type'] == 'next_tok'][:v20]
    v41 = v38.v41(v56.v41)
    v52 = v163.v110(v82(v9))
    v59 = v52.v111()
    v34 = v52.v164(v21) or 0
    v60 = v165.v112(max_chars=20000000)
    v61 = v113(v197(v60) | {' '})
    v62 = ['<pad>'] + v61
    v63 = {v114: v166 + 1 for v166, v114 in v198(v61)}
    v39 = v167.v115(v52, v60)
    v64 = v39[:v26(0.8 * v187(v39))] or v39
    v65 = v39[v26(0.8 * v187(v39)):] or v39[-100:]
    v33 = v181.v199(v52, v63, v34, v59).v116(v41)
    v109(f'docs={v187(v39)} V={v59} n_char={v187(v62)}')
    v66 = v144.v86(v187(v62), v59).v116(v41)
    v66.v117(v38.v200(v8, map_location=v41, weights_only=False)['model'])
    v66.v118()
    v67 = v119(v66, v65, v33, v34, v41, v170.v123(7))
    v68 = v120(v66, v52, v33, v34, v41)
    v109(f"187 baseline: doclink={v67:.3f} B para={v68['para']:.3f} hard={v68['hard']:.3f} gap={v68['gap_hard_minus_para']:.3f}")
    v38.v121(v10)
    v31 = v144.v86(v187(v62), v59).v116(v41)
    v69 = v38.v168.v122(v31.v169(), lr=v17, weight_decay=0.01)
    v40 = v170.v123(v10)
    v124, v125 = (None, None)
    v70 = v126.v126()
    v31.v127()
    for v71 in v128(1, v56.v137 + 1):
        v32 = v181.v171(v64, v12, v40, v41, v34)
        v35 = v32 == v34
        v172, v173, v174 = v31.v175(v33[v32], v35)
        v129 = v32[:, 1:]
        v130 = ~v35[:, :-1] & ~v35[:, 1:]
        v131 = v150.v176(v172[:, :-1][v130], v129[v130])
        v132 = v131 + v16 * v174[~v35].v193()
        v99 = v152(v64, v40, v34, v41)
        v133 = v38.v149(0.0, device=v41)
        if v99 is not None:
            v44 = v153(v31, v99[0], v33, v34)
            v45 = v153(v31, v99[1], v33, v34)
            v133 = v201(v44, v45)
            v132 = v132 + v15 * v133
        v69.v177(set_to_none=True)
        v132.v178()
        v207.v202.v179(v31.v169(), 1.0)
        v69.v71()
        v124 = v24(v131) if v124 is None else 0.95 * v124 + 0.05 * v24(v131)
        v125 = v24(v133) if v125 is None else 0.95 * v125 + 0.05 * v24(v133)
        if v71 % v18 == 0 or v71 == v56.v137:
            v31.v118()
            v180 = v181.v134(v31, v33, v34, v58, v41, only_type='next_tok')
            v74 = v119(v31, v65, v33, v34, v41, v170.v123(7), n=40)
            v109(f"  step {v71}: ce~{v124:.3f} con~{v125:.3f} next_tok(mid)={v180.v135('next_tok_acc', 0):.3f} doclink={v74:.3f} ({v126.v126() - v70:.0f}s)")
            v31.v127()
            v38.v203({'model': v31.v212(), 'step': v71}, v7)
    v31.v118()
    v72 = v181.v134(v31, v33, v34, v57, v41)
    v73 = v72.v135('next_tok_acc', 0.0)
    v74 = v119(v31, v65, v33, v34, v41, v170.v123(7))
    v75 = v120(v31, v52, v33, v34, v41)
    v76 = v73 >= v22 - 0.03
    v77 = v74 > v67
    v78 = v75['gap_hard_minus_para'] < v68['gap_hard_minus_para']
    v79 = v75['para'] > v75['hard']
    if v76 and v77 and v79:
        v136 = 'MEANING_OVER_FORM_YES'
    elif v76 and v77 and v78:
        v136 = 'CONTENT_INV_PROGRESS'
    else:
        v136 = 'CONTENT_INV_PARTIAL_' + ''.v184((v48 for v48, v49 in (('1', v76), ('2', v77), ('3', v78)) if not v49))
    v80 = {'timestamp': v213.v208(v214.v209).v182(), 'protocol': 'content_invariant_190', 'overall': v136, 'gates': {'G1_ce_preserved': {'next_tok': v73, 'baseline_187': v22, 'ok': v76}, 'G2_doclink': {'model': v74, 'baseline_187': v67, 'ok': v77}, 'G3_gateB': {'model': v75, 'baseline_187': v68, 'gap_shrunk': v78, 'para_gt_hard': v79}}, 'exam_full': v72, 'w_con': v15, 'steps': v56.v137}
    v4.v108(v194.v183(v80, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v108('\n'.v184(['# Stage190 — content invariant (doc-level InfoNCE on slow)', '', f'**Overall:** `{v136}`', '', f'- G1: next_tok={v73:.3f} vs 0.727 → {v76}', f'- G2: doclink={v74:.3f} vs 187 {v67:.3f} → {v77}', f"- G3: para={v75['para']:.3f} hard={v75['hard']:.3f} gap={v75['gap_hard_minus_para']:.3f} (187 gap {v68['gap_hard_minus_para']:.3f}) shrunk={v78} para>hard={v79}", '']), encoding='utf-8')
    v109(f"[190] {v136} | G1 {v73:.3f} | G2 {v74:.3f}/{v67:.3f} | G3 para={v75['para']:.3f} hard={v75['hard']:.3f} (187: {v68['para']:.3f}/{v68['hard']:.3f})")
    return 0
if v81 == '__main__':
    raise v138(v185())