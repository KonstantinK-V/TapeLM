"""
Stage 228c — Fix 228b: align retrieve with 227, then fp-guided decode.

228b failed because global argmax retrieve was ~33% exact while 227 4-way qmap
was ~0.95. Oracle fp-scorer was 1.0 → mechanism OK, protocol mismatch.

Modes (4-way org pick at code return position):
  head_only          — LM first-BPE logit
  fp_query           — cos(fp_can(c), W_bwd(q_code))   [= 227 recall as decode]
  fp_retrieved_4way  — 4-way retrieve value, then cos(fp(c), fp(retrieved))
  fp_retrieved_global— 228b broken protocol (global argmax), for contrast
  fp_oracle          — cos(fp(c), fp(gold))
  hybrid_query       — zscore(head) + zscore(fp_query)

  python _stage228c_fp_decode_fix.py [--smoke]
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
from _tapelm_ext import DomainAdapter, apply_qmap, fp_cos_scores, slot_retrieve_4way, slot_retrieve_global
v0 = v8('results')
v1 = v0 / 'stage228c_decision.json'
v2 = v0 / 'stage228c_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/_wikitext103_train.txt')
v5 = 2283

def zscore(v9: v11[v12]) -> v11[v12]:
    v10 = v129.v63(v9, dtype=v129.v130)
    if v131(v10) < 2:
        return v9
    v64, v65 = (v10.v132(), v10.v133())
    if v65 < 1e-09:
        return [0.0] * v131(v9)
    return ((v10 - v64) / v65).v66()

@v73.v26()
def head_scores(v13, v14, v15, v16, v17, v18: v67, v19: v11[v67]) -> v11[v12]:
    v20 = v14.v134(v18).v20
    if not v20:
        return [-1000000000.0] * v131(v19)
    v21 = v73.v68([v20], dtype=v73.v135, device=v17)
    v22 = v21 == v15
    v69, v70, v70 = v13.v71(v16[v21], v22, ids=v21)
    v23 = v69[0, -1]
    v24 = []
    for v25 in v19:
        v72 = v14.v134(v25).v20
        v24.v136(v12(v23[v72[0]]) if v72 else -1000000000.0)
    return v24

@v73.v26()
def fp_vs_anchor(v27: v74, v28: v73.v75, v19: v11[v67]) -> v11[v12]:
    v29 = v27.v76(v19)
    return [v12((v29[v137] * v28).v161()) for v137 in v162(v131(v19))]

def pick(v30: v11[v12], v19: v11[v67], v31: v67) -> v6:
    return v19[v7(v129.v173(v30))] == v31

def main() -> v7:
    v32 = v138.v77()
    v32.v78('--smoke', action='store_true')
    v33 = v32.v79()
    v17 = v73.v17('cuda' if v73.v174.v163() else 'cpu')
    v34 = 80 if v33.v80 else v139.v81
    v35 = 80 if v33.v80 else 600
    v36 = 100 if v33.v80 else v139.v82
    v37 = 80 if v33.v80 else v139.v83
    v38 = 12 if v33.v80 else 60
    v39 = 400 if v33.v80 else 8000
    v40 = v140.v84(v5)
    v85, v86, v87, v88 = v89()
    v14 = v141.v90(v67(v164.v142))
    v15 = v14.v143(v144) or 0
    v16 = v175.v165(v14, v87, v15, v14.v176()).v91(v17)
    with v4.v145('r', encoding='utf-8', errors='ignore') as v92:
        v93 = v92.v146(2000000)
    v41 = v11(v177.v166((v178 for v178 in v188.v186('[A-Za-z][a-z]{2,}', v93) if v131(v178) <= 14)))[:v37]
    v42 = v167(v88, v14.v176()).v91(v17)
    v42.v94(v73.v168(v3, map_location=v17, weights_only=False)['model'])
    v42.v95()
    v43 = v74(v42, v87, v17)
    v44 = v139.v96(v43, v41)
    v45 = v140.v84(227)
    with v4.v145('r', encoding='utf-8', errors='ignore') as v92:
        v97 = v11(v177.v166((v64.v182(1) for v64 in v189.v187(v92.v146(4000000)) if v131(v64.v182(1)) >= 5)))
    v46 = v147(v169(v97), v45, v38 + 10)[:v38]
    v47 = v97[:v38]
    v98, v99 = v139.v100(v43, v46, v47, v45)
    v48 = v148.v101(v140.v84(v5 + 1), v33.v80)
    v102, v103 = v149.v104(v48, v14, v15, max_lines=v39, min_line_len=20)
    v49 = v139.v105(v42, v102, v103, v16, v15, v17, v34, v5 + 2)
    v50 = v74(v49, v87, v17)
    v51 = v139.v96(v50, v41)
    v106, v107 = v139.v108(v179(256).v91(v17), v51, v44, v45, v36, v17)
    v52 = v148.v109(v42, v102, v103, v16, v15, v17, v35, v5 + 3)
    v53 = ['head_only', 'fp_query', 'fp_retrieved_4way', 'fp_retrieved_global', 'fp_oracle', 'hybrid_query']
    v54 = {v110: 0 for v110 in v53}
    v55 = v56 = 0
    v57 = 0
    for v111, v31 in v112(v46, v47):
        v19 = [v31] + [v47[(v137 + 1) % v131(v47)] for v137 in v162(3)]
        v40.v150(v19)
        v18 = f'def org_of_{v111}():\n    return '
        v113 = v50.v151(f'In the report {v111} was linked to the organization.', exclude=v31)
        if v113 is None:
            continue
        v114 = v152(v106, v113)
        v115 = v153(v98, v99, v114, v19)
        v116 = v154(v98, v99, v114)
        v55 += v7(v115 == v31)
        v56 += v7(v116 == v31)
        v117 = v155(v52, v14, v15, v16, v17, v18, v19)
        v118 = v156(v43, v114, v19)
        v119 = v157(v43, v115, v19)
        v120 = v157(v43, v116, v19)
        v121 = v157(v43, v31, v19)
        v122 = [v10 + v170 for v10, v170 in v112(v183(v117), v183(v118))]
        v54['head_only'] += v7(v171(v117, v19, v31))
        v54['fp_query'] += v7(v171(v118, v19, v31))
        v54['fp_retrieved_4way'] += v7(v171(v119, v19, v31))
        v54['fp_retrieved_global'] += v7(v171(v120, v19, v31))
        v54['fp_oracle'] += v7(v171(v121, v19, v31))
        v54['hybrid_query'] += v7(v171(v122, v19, v31))
        v57 += 1
    v57 = v123(1, v57)
    v58 = {v110: v54[v110] / v57 for v110 in v53}
    v124, v125 = (v55 / v57, v56 / v57)
    v59 = v58['fp_retrieved_4way'] - v58['head_only']
    v60 = v58['fp_retrieved_4way'] >= 0.7 and v59 >= 0.08
    v61 = 'FP_DECODE_FIX_YES' if v60 else 'FP_DECODE_FIX_PARTIAL' if v59 > 0.03 else 'FP_DECODE_FIX_NO'
    v24 = {'stage': '228c', 'overall': v61, 'contract': 'decode-time: 4-way slot retrieve (227) → fp-scorer vs retrieved value; not raw query·fp(c)', 'gates': {'G_fp_ret4_ge_0p70': v58['fp_retrieved_4way'] >= 0.7, 'G_lift_ge_0p08': v59 >= 0.08, 'G_ret4_exact': v124 >= 0.9}, 'retrieve_exact': {'four_way': v124, 'global_argmax': v125}, 'align_W_bwd': v107, 'mean_cos_code': v12((v44 * v51).v161(-1).v132()), 'accuracy_4way': v58, 'lift_fp_retrieved_4way_minus_head': v59, 'lift_fp_query_minus_head': v58['fp_query'] - v58['head_only'], 'n_items': v57, 'note': 'fp_retrieved_4way=YES path; fp_query fails because qq is ctx mix not value fp; fp_retrieved_global reproduces 228b (~global argmax)', 'timestamp': v184.v180(v185.v181).v158()}
    v1.v126(v172.v159(v24, indent=2), encoding='utf-8')
    v2.v126(f"# Stage 228c fp decode fix\n\n**{v61}** head={v58['head_only']:.3f} fp_query={v58['fp_query']:.3f} ret4={v58['fp_retrieved_4way']:.3f} ret_global={v58['fp_retrieved_global']:.3f} oracle={v58['fp_oracle']:.3f}\n", encoding='utf-8')
    v127(v172.v159(v24, indent=2))
    return 0
if v62 == '__main__':
    raise v128(v160())