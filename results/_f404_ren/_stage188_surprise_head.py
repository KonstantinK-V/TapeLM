"""
Stage 188 — S3b: wire surprise into the output (fix 187's G3).

187 showed: internal surprise EXISTS (G2 pass) but is not visible in the output
distribution — the model gets MORE confident after fake entities (G3 fail),
while GPT gets less confident (rarity signal for free from BPE).

Fix: surprise-conditioned temperature. Per position t:
  T_t = 1 + softplus(w * surprise_t + b)     (learnable w,b)
  logits_t = head([fast_t ; slow_t]) / T_t
CE itself calibrates w,b: when surprised, softening predictions lowers loss on
hard positions. No new hand loss.

Extra diagnostic: mean surprise AT fake span vs AT real span — does the ink
channel even see fakes as unusual? If not, no head wiring can fix G3 and the
next target is a rarity signal in the ink encoder itself.

Gates (judge = Exam v2):
  G1 next_tok >= 0.727 - 0.03 (don't lose 187's CE parity)
  G2 surprise unseen > seen (keep novelty)
  G3 entropy after fake > after real (the fix target)

  python _stage188_surprise_head.py
  python _stage188_surprise_head.py --steps 3000
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
import _stage181_ce_control as s181
import _stage185_tape_read as s185
import _stage187_self_model as s187
v0 = v22('results')
v1 = v22('data')
v2 = v22('checkpoints')
v3 = v0 / '_stage188_log.txt'
v4 = v0 / 'stage188_decision.json'
v5 = v0 / 'stage188_mini.md'
v6 = v1 / 'stage186_exam_v2.jsonl'
v7 = v0 / 'stage187_decision.json'
v8 = v2 / 'stage188_surprise_head.pt'
v9 = v23.v9
v10 = 185
v11 = v23.v11
v12 = 16
v13 = 0.0003
v14 = 1000
v15 = 3000
v16 = 0.1
v17 = 60
v18 = '[PAD]'

def log(v24: v69) -> None:
    v25 = v24 if v24.v130('\n') else v24 + '\n'
    try:
        v131(v25, end='', flush=True)
    except v70:
        v131(v25.v196('ascii', 'replace').v185('ascii'), end='', flush=True)
    v3.v132.v71(parents=True, exist_ok=True)
    with v3.v133('a', encoding='utf-8') as v72:
        v72.v134(v25)

class SurpriseHeadModel(v26.v19):

    def __init__(v73, v74: v21, v44: v21):
        v186().v135(v74, v44)
        v73.v75 = v169.v136(v86.v82(4.0))
        v73.v76 = v169.v136(v86.v82(-2.0))

    def forward_all(v73, v77: v86.v137, v36: v86.v137):
        v78 = v73.v138(v77)
        v79 = v73.v79(v78, pad_mask=v36)
        v139, v84, v140 = v73.v139(v78, v36)
        v80 = v73.v141(v86.v170([v79, v139], dim=-1))
        v81 = 1.0 + v179.v192(v73.v75 * v84 + v73.v76).v171(-1)
        return (v80 / v81, v84, v140)

@v86.v37()
def surprise_at_span(v27, v28, v29, v30, v31, v32) -> v20:
    v33 = (v30 + v31)[-v11:]
    v34 = v142(v33) - v142(v31)
    v35 = v86.v82([v33], dtype=v86.v143, device=v32)
    v36 = v35 == v29
    v83, v84, v83 = v27.v85(v28[v35], v36)
    return v20(v84[0, v34:].v144())

def main() -> v21:
    v38 = v145.v87()
    v38.v88('--steps', type=v21, default=v15)
    v38.v88('--device', default='cuda' if v86.v193.v187() else 'cpu')
    v39 = v38.v89()
    v0.v71(parents=True, exist_ok=True)
    v3.v90('', encoding='utf-8')
    v91(f'Stage188 start {v197.v190(v198.v191).v165()}')
    v91('S3b: surprise-conditioned temperature on the head')
    v40 = [v172.v146(v147) for v147 in v6.v194(encoding='utf-8').v173() if v147.v174()]
    v41 = [v62 for v62 in v40 if v62['type'] == 'next_tok'][:v17]
    v42 = v172.v146(v7.v194(encoding='utf-8'))['gates']['G1_ce_preserved']['next_tok']
    v91(f'exam v2 items={v142(v40)}; baseline (187) next_tok={v42:.3f}')
    v32 = v86.v32(v39.v32)
    v43 = v148.v92(v69(v9))
    v44 = v43.v93()
    v29 = v43.v149(v18) or 0
    v45 = v150.v94(max_chars=20000000)
    v46 = v95(v175(v45) | {' '})
    v47 = ['<pad>'] + v46
    v48 = {v96: v151 + 1 for v151, v96 in v176(v46)}
    v49 = v152.v97(v43, v45)
    v50 = v49[:v21(0.8 * v142(v49))] or v49
    v51 = v49[v21(0.8 * v142(v49)):] or v49[-100:]
    v28 = v162.v177(v43, v48, v29, v44).v98(v32)
    v91(f'docs={v142(v49)} V={v44} n_char={v142(v47)}')
    v86.v99(v10)
    v27 = v178(v142(v47), v44).v98(v32)
    v52 = v86.v153.v100(v27.v154(), lr=v13, weight_decay=0.01)
    v53 = v155.v101(v10)
    v102, v103 = (None, None)
    v54 = v104.v104()
    v27.v105()
    for v55 in v106(1, v39.v128 + 1):
        v107 = v162.v156(v50, v12, v53, v32, v29)
        v36 = v107 == v29
        v80, v84, v140 = v27.v85(v28[v107], v36)
        v108 = v107[:, 1:]
        v109 = ~v36[:, :-1] & ~v36[:, 1:]
        v110 = v179.v157(v80[:, :-1][v109], v108[v109])
        v111 = v140[~v36].v144()
        v112 = v110 + v16 * v111
        v52.v158(set_to_none=True)
        v112.v159()
        v169.v180.v160(v27.v154(), 1.0)
        v52.v55()
        v102 = v20(v110) if v102 is None else 0.95 * v102 + 0.05 * v20(v110)
        v103 = v20(v111) if v103 is None else 0.95 * v103 + 0.05 * v20(v111)
        if v55 % v14 == 0 or v55 == v39.v128:
            v27.v113()
            v161 = v162.v114(v27, v28, v29, v41, v32, only_type='next_tok')
            v91(f"  step {v55}: ce~{v102:.3f} self~{v103:.3f} tw={v20(v27.v75):.2f} tb={v20(v27.v76):.2f} next_tok(mid)={v161.v115('next_tok_acc', 0):.3f} ({v104.v104() - v54:.0f}s)")
            v27.v105()
            v86.v181({'model': v27.v195(), 'step': v55}, v8)
    v27.v113()
    v56 = v162.v114(v27, v28, v29, v40, v32)
    v57 = v56.v115('next_tok_acc', 0.0)
    v58 = v26.v116(v27, v50, v28, v29, v32, v155.v101(1))
    v59 = v26.v116(v27, v51, v28, v29, v32, v155.v101(2))
    v60 = [v62 for v62 in v40 if v62['type'] == 'entity'][:80]
    v61 = v155.v101(3)
    v117, v118, v119, v120 = ([], [], [], [])
    for v62 in v60:
        v121 = v62['cand_ids'][v62['gold_idx']]
        v122 = v26.v163[v61.v182(0, v142(v26.v163) - 1)]
        v123 = [v151 for v151 in v43.v196(' ' + v122).v107 if v151 != v29]
        v117.v164(v26.v183(v27, v28, v29, v62['ctx_ids'], v121, v32))
        v118.v164(v26.v183(v27, v28, v29, v62['ctx_ids'], v123, v32))
        v119.v164(v184(v27, v28, v29, v62['ctx_ids'], v121, v32))
        v120.v164(v184(v27, v28, v29, v62['ctx_ids'], v123, v32))
    v124, v125 = (v20(v188.v144(v117)), v20(v188.v144(v118)))
    v126, v127 = (v20(v188.v144(v119)), v20(v188.v144(v120)))
    v63 = v57 >= v42 - 0.03
    v64 = v59 > v58
    v65 = v125 > v124
    v66 = 'SURPRISE_HEAD_YES' if v63 and v64 and v65 else 'SURPRISE_HEAD_PARTIAL_' + ''.v167((v189 for v189, v199 in (('1', v63), ('2', v64), ('3', v65)) if not v199))
    v67 = {'timestamp': v197.v190(v198.v191).v165(), 'protocol': 'surprise_head_188', 'overall': v66, 'gates': {'G1_ce_preserved': {'next_tok': v57, 'baseline_187': v42, 'ok': v63}, 'G2_novelty': {'surprise_seen_train': v58, 'surprise_unseen_hold': v59, 'ok': v64}, 'G3_calibration': {'entropy_after_real': v124, 'entropy_after_fake': v125, 'ok': v65}}, 'diagnostic_surprise_at_span': {'real': v126, 'fake': v127, 'fake_gt_real': v127 > v126}, 'temp': {'w': v20(v27.v75), 'b': v20(v27.v76)}, 'exam_full': v56, 'steps': v39.v128}
    v4.v90(v172.v166(v67, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v90('\n'.v167(['# Stage188 — surprise-conditioned head', '', f'**Overall:** `{v66}`', '', f'- G1: next_tok={v57:.3f} vs 187 {v42:.3f} → {v63}', f'- G2: surprise seen={v58:.4f} unseen={v59:.4f} → {v64}', f'- G3: entropy real={v124:.3f} fake={v125:.3f} → {v65}', f'- diag: surprise@span real={v126:.4f} fake={v127:.4f} (fake>real={v127 > v126})', f'- temp w={v20(v27.v75):.2f} b={v20(v27.v76):.2f}', '']), encoding='utf-8')
    v91(f'[188] {v66} | G1 {v57:.3f} | G2 {v58:.4f}<{v59:.4f} | G3 {v124:.3f}<{v125:.3f} | diag span s_real={v126:.4f} s_fake={v127:.4f}')
    return 0
if v68 == '__main__':
    raise v129(v168())