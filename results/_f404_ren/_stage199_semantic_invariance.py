"""
Stage 199 — semantic invariance via consequence-prediction, WITHOUT breaking the stack.

Goal B (open since 179): representation groups by MEANING not spelling — paraphrases
close, surface-similar-different-meaning (car/cat) far. Route 1 (cheapest, no new data):
"meaning = what comes next". Train a contrastive predictive head so a window's embedding
predicts its own continuation vs other continuations (CPC). Paraphrases predict similar
futures -> pulled together; car/cat predict different futures -> pushed apart.

NON-DESTRUCTION (user constraint): P1 encoder stays FROZEN. Only a separate semantic
head z_sem is trained on top. Generation / FP-memory / calibration never touch z_sem, so
they cannot regress — verified by a parity regression gate (next_tok unchanged).

SCALABILITY (user constraint): train the SAME head at 3 data budgets (5% / 25% / 100% of
tokens), fixed steps. Monotone improvement of the semantic gap = evidence it scales;
recipe is size-invariant.

Metrics on 179 pairs (cosine):
  para_sim (want HIGH), hard_sim (want LOW). inversion = para_sim > hard_sim (true semantic win).
  baseline = raw frozen fast mean-pool (191b regime: hard≈0.89 > para≈0.71).

Gates:
  G_nondestruct : next_tok parity preserved (~0.867; encoder frozen by construction)
  G_semantic    : at full budget para_sim > hard_sim (INVERSION)   -> SEM_INV_YES
                  else head shrinks (hard-para) gap vs raw baseline -> SEM_INV_TREND
  G_scale       : (hard-para) gap decreases monotonically across budgets

  python _stage199_semantic_invariance.py
"""
from __future__ import annotations
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
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data, score_items, span_logprob_x
v0 = v17('results')
v1 = v17('checkpoints/stage191_p1_curve.pt')
v2 = v17('data/stage191_exam_v3.jsonl')
v3 = v0 / 'stage199_decision.json'
v4 = v0 / 'stage199_mini.md'
v5 = v0 / '_stage199_log.txt'
v6 = 199
v7 = 32
v8 = 128
v9 = 0.07
v10 = 1200
v11 = 128
v12 = 0.0003
v13 = [0.05, 0.25, 1.0]
v14 = 64

def log(v18: v48) -> None:
    v19 = v18 if v18.v100('\n') else v18 + '\n'
    try:
        v101(v19, end='', flush=True)
    except v49:
        v101(v19.v187('ascii', 'replace').v165('ascii'), end='', flush=True)
    v5.v102.v50(parents=True, exist_ok=True)
    with v5.v103('a', encoding='utf-8') as v51:
        v51.v104(v19)

class SemHead(v20.v15):

    def __init__(v52, v29, v53=v8):
        v166().v105()
        v52.v54 = v20.v106(v20.v141(v29, v29), v20.v142(), v20.v141(v29, v53))

    def forward(v52, v55):
        return v143.v107(v52.v54(v55), dim=-1)

class Predictor(v20.v15):

    def __init__(v52, v53=v8):
        v166().v105()
        v52.v54 = v20.v106(v20.v141(v53, v53), v20.v142(), v20.v141(v53, v53))

    def forward(v52, v56):
        return v143.v107(v52.v54(v56), dim=-1)

def main() -> v16:
    v0.v50(parents=True, exist_ok=True)
    v5.v57('', encoding='utf-8')
    v58(f'Stage199 start {v185.v179(v186.v180).v137()}')
    v58('semantic invariance via CPC on a FROZEN encoder + scale trend')
    v21 = v78.v21('cuda' if v78.v167.v144() else 'cpu')
    v22 = v59.v59()
    v60, v61, v62, v63 = v64()
    v23 = v108.v65(v48(v145.v109))
    v24 = v23.v66()
    v25 = v23.v110(v111) or 0
    v26 = v168.v146(v23, v62, v25, v24).v67(v21)
    v27 = v147(v63, v24).v67(v21)
    v27.v68(v78.v148(v1, map_location=v21, weights_only=False)['model'])
    v27.v69()
    for v28 in v27.v70():
        v28.v112(False)
    v29 = v27.v87.v71 // 2
    v58(f'encoder frozen (fast dim={v29}) ({v59.v59() - v22:.0f}s)')

    @v78.v77()
    def pooled_fast(v72: v78.v30) -> v78.v30:
        v73 = v72 == v25
        v74 = v27.v113(v26[v72], v72)
        v75 = v27.v75(v74, pad_mask=v73)
        v76 = (~v73).v120().v114(-1)
        return (v75 * v76).v149(1) / v76.v149(1).v150(min=1.0)

    @v78.v77()
    def pooled_text(v79: v48) -> v78.v30:
        v80 = [v151 for v151 in v23.v187(v79).v80 if v151 != v25][:v14]
        v81 = v78.v115([v80], device=v21)
        return v152(v81)[0]
    v31 = v82(v60)

    def eligible_docs(v83):
        return [v116 for v116 in v126(v82(v61) - 1) if v61[v116 + 1] <= v83 and v61[v116 + 1] - v61[v116] >= 2 * v7]

    def sampler(v84, v85):

        def draw():
            v117 = v169.v36((v11, v7), v25, v169.v153)
            v118 = v169.v36((v11, v7), v25, v169.v153)
            for v96 in v126(v11):
                v116 = v84[v85.v181(0, v82(v84) - 1)]
                v170, v171 = (v61[v116], v61[v116 + 1])
                v154 = v170 + v85.v181(0, v171 - v170 - 2 * v7)
                v117[v96] = v60[v154:v154 + v7]
                v118[v96] = v60[v154 + v7:v154 + 2 * v7]
            return (v78.v188(v117).v67(v21), v78.v188(v118).v67(v21))
        return v86

    def measure_B(v87: v119, v88=False):

        def z(v79):
            v28 = v155(v79)
            return v143.v107(v28, dim=-1) if v88 else v87(v28.v114(0))[0]
        v89 = v120(v169.v156([v120(v143.v189(v56(v190), v56(v96), dim=-1)) for v190, v96 in v191.v182]))
        v90 = v120(v169.v156([v120(v143.v189(v56(v190), v56(v96), dim=-1)) for v190, v96 in v191.v183]))
        return {'para': v89, 'hard': v90, 'gap_hard_minus_para': v90 - v89, 'inversion': v89 > v90}
    v32 = v91(None, raw=True)
    v58(f"raw frozen fast baseline: para={v32['para']:.3f} hard={v32['hard']:.3f} gap={v32['gap_hard_minus_para']:+.3f}")
    v33 = {}
    v34 = {}
    for v35 in v13:
        v83 = v16(v31 * v35)
        v84 = v121(v83)
        v85 = v157.v122(v6)
        v87 = v119(v29).v67(v21)
        v92 = v172().v67(v21)
        v93 = v78.v158.v123(v173(v87.v70()) + v173(v92.v70()), lr=v12, weight_decay=0.01)
        v86 = v124(v84, v85)
        v87.v125()
        v92.v125()
        v94 = None
        for v95 in v126(1, v10 + 1):
            v117, v118 = v86()
            v127 = v87(v152(v117))
            v128 = v87(v152(v118))
            v129 = v92(v127)
            v130 = v129 @ v128.v174 / v9
            v131 = v78.v159(v117.v175(0), device=v21)
            v132 = v143.v160(v130, v131)
            v93.v161(set_to_none=True)
            v132.v162()
            v93.v95()
            v94 = v120(v132) if v94 is None else 0.97 * v94 + 0.03 * v120(v132)
        v87.v69()
        v96 = v91(v87)
        v33[f'{v35:.2f}'] = {'docs': v82(v84), 'budget_tokens': v83, 'cpc_loss': v94, **v96}
        v34[v35] = v87
        v58(f"  budget {v35:.2f} ({v82(v84)} docs): para={v96['para']:.3f} hard={v96['hard']:.3f} gap={v96['gap_hard_minus_para']:+.3f} inv={v96['inversion']} cpc~{v94:.3f} ({v59.v59() - v22:.0f}s)")
    v36 = v33[f'{v13[-1]:.2f}']
    v37 = [v163.v133(v134) for v134 in v2.v184(encoding='utf-8').v164()]
    v38 = [v135 for v135 in v37 if v135['type'] == 'next_tok'][:120]
    v39 = v136(lambda v176, v177: v178(v27, v26, v25, v176, v177, v21), v38, 'next_tok')['next_tok_acc']
    v58(f'non-destruct: next_tok(frozen)={v39:.3f} (expected ~0.867; head is a separate branch)')
    v40 = [v33[f'{v51:.2f}']['gap_hard_minus_para'] for v51 in v13]
    v41 = v97((v40[v151 + 1] <= v40[v151] + 1e-06 for v151 in v126(v82(v40) - 1)))
    v42 = v39 >= 0.8
    v43 = v36['inversion']
    v44 = v36['gap_hard_minus_para'] < v32['gap_hard_minus_para'] - 0.02
    if v42 and v43:
        v98 = 'SEM_INV_YES'
    elif v42 and v44 and v41:
        v98 = 'SEM_INV_TREND'
    elif v42 and v44:
        v98 = 'SEM_INV_PARTIAL'
    else:
        v98 = 'SEM_INV_NO'
    v45 = {'g_nondestruct': v42, 'g_scale_monotone': v41, 'inversion_at_full': v43, 'head_beats_raw': v44}
    v46 = {'timestamp': v185.v179(v186.v180).v137(), 'protocol': 'semantic_invariance_199', 'overall': v98, 'gates': v45, 'raw_frozen_baseline': v32, 'scale_trend': v33, 'next_tok_parity_frozen': v39, 'note': 'frozen P1 encoder + separate CPC-trained semantic head; consequence-prediction route to B; scale probe = same head at 5/25/100% token budgets'}
    v3.v57(v163.v138(v46, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v57('\n'.v139(['# Stage199 — semantic invariance (CPC on frozen encoder)', '', f'**Overall:** `{v98}`', '', f"- raw frozen baseline: para {v32['para']:.3f} / hard {v32['hard']:.3f} (gap {v32['gap_hard_minus_para']:+.3f})", '- scale trend (para / hard / gap hard−para):', *[f"  - budget {v51:.2f}: {v33[f'{v51:.2f}']['para']:.3f} / {v33[f'{v51:.2f}']['hard']:.3f} / {v33[f'{v51:.2f}']['gap_hard_minus_para']:+.3f} (inv={v33[f'{v51:.2f}']['inversion']})" for v51 in v13], '', f'- non-destruct: next_tok(frozen) = {v39:.3f} (generation/memory/calib untouched)', f'- gates: {v45}']), encoding='utf-8')
    v58(f"[199] {v98} | full para={v36['para']:.3f} hard={v36['hard']:.3f} gap={v36['gap_hard_minus_para']:+.3f} parity={v39:.3f}")
    return 0
if v47 == '__main__':
    raise v99(v140())