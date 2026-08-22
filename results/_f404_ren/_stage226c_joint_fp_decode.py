"""
Stage 226c — Joint gen + mem with official 228c fp decode (e2e trunk).

Fixes 226 protocol gaps:
  - Retrieval: 4-way qmap recall (227), not global argmax (226 was ~0.60).
  - Utilization: fp_retrieved_4way at code return position (228c), vs head_only.

  python _stage226c_joint_fp_decode.py [--smoke]
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
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage225_family_fork as s225
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, apply_qmap, fp_cos_scores, fp_decode_pick_retrieved_4way, slot_retrieve_4way
v0 = v8('results')
v1 = v0 / 'stage226c_decision.json'
v2 = v0 / 'stage226c_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/_wikitext103_train.txt')
v5 = 2263

@v63.v22()
def head_pick(v9, v10, v11, v12, v13, v14: v6, v15: v110[v6]) -> v6:
    v16 = v10.v111(v14).v16
    if not v16:
        return v15[0]
    v17 = v63.v58([v16], dtype=v63.v112, device=v13)
    v18 = v17 == v11
    v59, v60, v60 = v9.v61(v12[v17], v18, ids=v17)
    v19 = v59[0, -1]
    v20 = []
    for v21 in v15:
        v62 = v10.v111(v21).v16
        v20.v113(v134(v19[v62[0]]) if v62 else -1000000000.0)
    return v15[v7(v148.v138(v20))]

def main() -> v7:
    v23 = v114.v64()
    v23.v65('--smoke', action='store_true')
    v24 = v23.v66()
    v13 = v63.v13('cuda' if v63.v149.v139() else 'cpu')
    v25 = 80 if v24.v67 else v115.v68
    v26 = 80 if v24.v67 else 600
    v27 = 100 if v24.v67 else v115.v69
    v28 = 80 if v24.v67 else v115.v70
    v29 = 12 if v24.v67 else 60
    v30 = 400 if v24.v67 else 8000
    v31 = v116.v71(v5)
    v72, v73, v74, v75 = v76()
    v10 = v117.v77(v6(v140.v118))
    v11 = v10.v119(v120) or 0
    v12 = v150.v141(v10, v74, v11, v10.v151()).v78(v13)
    with v4.v121('r', encoding='utf-8', errors='ignore') as v79:
        v80 = v79.v122(2000000)
    v32 = v110(v152.v142((v153 for v153 in v167.v164('[A-Za-z][a-z]{2,}', v80) if v166(v153) <= 14)))[:v28]
    v33 = v143(v75, v10.v151()).v78(v13)
    v33.v81(v63.v144(v3, map_location=v13, weights_only=False)['model'])
    v33.v82()
    v34 = v83(v33, v74, v13)
    v35 = v115.v84(v34, v32)
    with v4.v121('r', encoding='utf-8', errors='ignore') as v79:
        v85 = v110(v152.v142((v159.v158(1) for v159 in v168.v165(v79.v122(4000000)) if v166(v159.v158(1)) >= 5)))
    v36 = v123(v145(v85), v31, v29 + 10)[:v29]
    v37 = v85[:v29]
    v86, v87 = v115.v88(v34, v36, v37, v31)
    v38 = v124.v89(v116.v71(v5 + 1), v24.v67)
    v90, v91 = v125.v92(v38, v10, v11, max_lines=v30, min_line_len=20)
    v39 = v115.v93(v33, v90, v91, v12, v11, v13, v25, v5 + 2)
    v40 = v83(v39, v74, v13)
    v41 = v115.v84(v40, v32)
    v94, v95 = v115.v96(v154(256).v78(v13), v41, v35, v31, v27, v13)
    v42 = v124.v97(v33, v90, v91, v12, v11, v13, v26, v5 + 3)
    v43 = 'In the report {S} was linked to the organization.'
    v44 = 'def org_of_{S}():\n    return '
    v45 = v46 = v47 = v48 = 0
    for v98, v99 in v100(v36, v37):
        v15 = [v99] + [v37[(v155 + 1) % v166(v37)] for v155 in v160(3)]
        v31.v126(v15)
        v101 = v43.v127(S=v98)
        v102 = v40.v128(v101, exclude=v99)
        if v102 is None:
            continue
        v103 = v129(v94, v102)
        v104 = v130(v86, v87, v103, v15)
        v45 += v7(v104 == v99)
        v60, v131 = v132(v34, v86, v87, v94, v40, v101, v99, v15)
        v46 += v7(v131 == v99)
        v14 = v44.v127(S=v98)
        v105 = v133(v42, v10, v11, v12, v13, v14, v15)
        v47 += v7(v105 == v99)
        v48 += 1
    v48 = v106(1, v48)
    v49 = v45 / v48
    v50 = v46 / v48
    v51 = v47 / v48
    v52 = v50 - v51
    v53 = v49 >= 0.7
    v54 = v50 >= 0.7 and v52 >= 0.08
    v55 = 'JOINT_FP_DECODE_OK' if v53 and v54 else 'JOINT_FP_DECODE_PARTIAL' if v53 or v54 else 'JOINT_FP_DECODE_NO'
    v56 = {'stage': '226c', 'overall': v55, 'gates': {'G_recall_4way': v53, 'G_fp_decode_util': v54}, 'align_W_bwd': v95, 'mean_cos_code_shift': v134((v35 * v41).v161(-1).v146()), 'recall_4way_qmap': v49, 'accuracy': {'head_only': v51, 'fp_retrieved_4way': v50}, 'lift_fp_minus_head': v52, 'n_items': v48, 'note': '226 e2e: canonical bank + code qmap + 228c decode at return token', 'timestamp': v162.v156(v163.v157).v135()}
    v1.v107(v147.v136(v56, indent=2), encoding='utf-8')
    v2.v107(f'# Stage 226c joint fp decode\n\n**{v55}** recall4={v49:.3f} fp={v50:.3f} head={v51:.3f} lift={v52:.3f}\n', encoding='utf-8')
    v108(v147.v136(v56, indent=2))
    return 0
if v57 == '__main__':
    raise v109(v137())