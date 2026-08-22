"""
Stage 237 — Overnight mixed-domain L1 step (not a 191 replacement).

From frozen P1: continue CE training on interleaved prose+code for several hours,
then measure next_tok mid + post-hoc code-shift W recall vs a prose-only continue control.

Does **not** overwrite `stage191_p1_curve.pt`. Writes `checkpoints/stage237_mixed_l1.pt`.

  python _stage237_mixed_l1_night.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
from _stage191_night import EVAL_EVERY, EXAM_V3, LR, MICRO, PAD, SelfModelXL, load_data, lr_at, sample_windows, score_items, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, mean_core_cos
v0 = v11('results')
v1 = v11('checkpoints/stage191_p1_curve.pt')
v2 = v11('checkpoints/stage237_mixed_l1.pt')
v3 = v11('checkpoints/stage237_prose_continue.pt')
v4 = v0 / 'stage237_decision.json'
v5 = v0 / 'stage237_mini.md'
v6 = v0 / '_stage237_log.txt'
v7 = v11('data/external_tinystories_100k_85.txt')
v8 = v11('data/_wikitext103_train.txt')
v9 = 237

def log(v12: v74) -> None:
    v13 = v12 if v12.v135('\n') else v12 + '\n'
    try:
        v136(v13, end='', flush=True)
    except v75:
        v136(v13.v212('ascii', 'replace').v190('ascii'), end='', flush=True)
    try:
        v6.v171.v137(parents=True, exist_ok=True)
        with v6.v164('a', encoding='utf-8') as v117:
            v117.v172(v13)
    except v76:
        pass

def build_mixed_flat(v14, v15: v10, v16: v10, v17: v77):
    v18 = v7.v160(encoding='utf-8', errors='ignore').v138()[:v16]
    v19 = v139.v78(v146.v85(v9 + 1), v17)
    v20 = v19.v138()[:v16]
    v21: v79[v74] = []
    for v22 in v80(v140(v173(v18), v173(v20))):
        if v22 < v173(v18) and v18[v22].v174():
            v21.v175(v18[v22])
        if v22 < v173(v20) and v20[v22].v174():
            v21.v175(v20[v22])
    return v141.v81('\n'.v142(v21), v14, v15, max_lines=v16 * 2, min_line_len=20)

def continue_train(v23, v24, v25, v26, v15, v27, v28: v10, v29: v10, v30: v74):
    v31 = v143.v82(v23)
    v31.v83()
    v32 = v79(v31.v144())
    v33 = v155.v145.v84(v32, lr=v176 * 0.5)
    v34 = v146.v85(v29)
    v35 = v86.v86()
    v36 = 0.0
    for v37 in v80(1, v28 + 1):
        for v87 in v33.v88:
            v87['lr'] = v177(v37, v28)
        v89 = v191(v24, v25, v192, v34, v15).v105(v27)
        v90 = v89 == v15
        v147, v126, v148 = v31.v149(v26[v89], v90, ids=v89)
        v91 = v89[:, 1:]
        v92 = ~v90[:, :-1] & ~v90[:, 1:]
        v93 = v178.v150(v147[:, :-1][v92], v91[v92])
        v94 = v93 + 0.1 * v148[~v90].v193()
        v33.v151(set_to_none=True)
        v94.v152()
        v155.v194.v179.v153(v32, 1.0)
        v33.v37()
        v36 = 0.95 * v36 + 0.05 * v50(v93.v195()) if v37 > 1 else v50(v93.v195())
        if v37 % v140(1, v215 // 2 if v28 > 500 else 40) == 0 or v37 == v28:
            v100(f'  [{v30}] step {v37}/{v28} ce~{v36:.3f} ({v86.v86() - v35:.0f}s)')
    v31.v95()
    return v31

def main() -> v10:
    v38 = v154.v96()
    v38.v97('--smoke', action='store_true')
    v39 = v38.v98()
    v6.v99('', encoding='utf-8')
    v27 = v155.v27('cuda' if v155.v196.v180() else 'cpu')
    v28 = 120 if v39.v17 else 6000
    v16 = 400 if v39.v17 else 12000
    v40 = 60 if v39.v17 else 400
    v41 = 10 if v39.v17 else 50
    v42 = 40 if v39.v17 else 400
    v43 = 60 if v39.v17 else 800
    v44 = v146.v85(v9)
    v100(f'Stage237 start {v210.v207(v211.v208).v168()} device={v27} steps={v28}')
    v24, v25, v101, v102 = v103()
    v14 = v156.v104(v74(v181.v157))
    v15 = v14.v158(v159) or 0
    v26 = v197.v182(v14, v101, v15, v14.v198()).v105(v27)
    v45 = v183(v102, v14.v198()).v105(v27)
    v46 = v155.v106(v1, map_location=v27, weights_only=False)
    v45.v107(v46['model'])
    v45.v95()
    v108, v109 = v141.v81(v7.v160(encoding='utf-8', errors='ignore'), v14, v15, max_lines=v16)
    v110, v111 = v112(v14, v15, v16, v39.v17)
    v100('continue prose …')
    v47 = v113(v45, v108, v109, v26, v15, v27, v28, v9 + 2, 'prose')
    v100('continue mixed …')
    v48 = v113(v45, v110, v111, v26, v15, v27, v28, v9 + 3, 'mixed')
    if not v39.v17:
        v2.v171.v137(parents=True, exist_ok=True)
        v155.v161({'model': v48.v199(), 'stage': 237, 'steps': v28}, v2)
        v155.v161({'model': v47.v199(), 'stage': 237, 'tag': 'prose_continue', 'steps': v28}, v3)
    v49 = []
    if v162.v114():
        with v162.v164(encoding='utf-8') as v117:
            for v13 in v117:
                v184 = v189.v200(v13)
                if v184.v185('type') == 'next_tok':
                    v49.v175(v184)
                if v173(v49) >= (40 if v39.v17 else 120):
                    break

    def next_tok_acc(v23) -> v50:
        if not v49:
            return v50('nan')
        v115 = v163(lambda v201, v202: v203(v23, v26, v15, v201, v202, v27), v49, 'next_tok')
        return v50(v115.v185('next_tok_acc', 0.0))
    v51 = v116(v47)
    v52 = v116(v48)
    v100(f'next_tok mid prose={v51:.3f} mixed={v52:.3f}')
    with v8.v164('r', encoding='utf-8', errors='ignore') as v117:
        v118 = v117.v165(2000000)
    v53 = v79(v204.v186((v205 for v205 in v216.v213('[A-Za-z][a-z]{2,}', v118) if v173(v205) <= 14)))[:v40]
    v54 = v119(v45, v101, v27)
    v55 = v119(v47, v101, v27)
    v56 = v119(v48, v101, v27)
    v57 = v166.v120(v54, v53)
    v121, v122 = v141.v81(v139.v78(v146.v85(v9 + 7), v39.v17), v14, v15, max_lines=v16, min_line_len=20)
    v58 = v166.v123(v47, v121, v122, v26, v15, v27, v42, v9 + 11)
    v59 = v166.v123(v48, v121, v122, v26, v15, v27, v42, v9 + 12)
    v60 = v119(v58, v101, v27)
    v61 = v119(v59, v101, v27)
    v62 = v124(v54, v60, v53)
    v63 = v124(v54, v61, v53)
    v125, v126 = v166.v127(v206(256).v105(v27), v166.v120(v60, v53), v57, v44, v43, v27)
    v128, v126 = v166.v127(v206(256).v105(v27), v166.v120(v61, v53), v57, v44, v43, v27)
    with v8.v164('r', encoding='utf-8', errors='ignore') as v117:
        v129 = v79(v204.v186((v31.v209(1) for v31 in v217.v214(v117.v165(2000000)) if v173(v31.v209(1)) >= 5)))
    v64 = v167(v187(v129), v44, v41 + 10)[:v41]
    v65 = v129[:v41]
    v130, v131 = v166.v132(v54, v64, v65, v44)
    v66 = v139.v133(v130, v131, v60, v64, v65, v44, query_x=v139.v188(v125))
    v67 = v139.v133(v130, v131, v61, v64, v65, v44, query_x=v139.v188(v128))
    v68 = not v52 != v52 and v52 + 1e-09 >= v51 - 0.03
    v69 = v63 >= v62 + 0.02
    v70 = v67 >= v66 - 0.05
    v71 = 'MIXED_L1_NIGHT_OK' if v68 and (v69 or v70) and (v67 >= 0.7) else 'MIXED_L1_NIGHT_PARTIAL' if v68 or v70 else 'MIXED_L1_NIGHT_NO'
    v72 = {'stage': 237, 'overall': v71, 'gates': {'G_mixed_next_tok_not_worse': v77(v68), 'G_mixed_code_shift_closer_to_can': v77(v69), 'G_mixed_W_recall_not_worse': v77(v70)}, 'steps': v28, 'next_tok_prose_continue': v51, 'next_tok_mixed_continue': v52, 'mean_cos_can_after_code_shift_prose': v62, 'mean_cos_can_after_code_shift_mixed': v63, 'recall_W_after_code_prose': v66, 'recall_W_after_code_mixed': v67, 'ckpt': v74(v2) if v2.v114() else None, 'note': 'Does not replace stage191_p1_curve.pt; overnight scale step toward multi-domain L1', 'timestamp': v210.v207(v211.v208).v168()}
    v4.v99(v189.v169(v72, indent=2), encoding='utf-8')
    v5.v99(f'# Stage 237 mixed L1 night\n\n**{v71}** nt_m={v52:.3f} W_m={v67:.3f} cos_m={v63:.3f}\n', encoding='utf-8')
    v100(v189.v169(v72, indent=2))
    return 0
if v73 == '__main__':
    raise v134(v170())