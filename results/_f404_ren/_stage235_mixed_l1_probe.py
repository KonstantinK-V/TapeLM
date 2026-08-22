"""
Stage 235 — Mixed-domain L1 pretrain probe (scale branch).

Short arc_enc finetune on interleaved prose+code vs prose-only, then measure
cross-domain fp stability (mean core cos) and post-hoc W recall vs frozen P1.

Not full multi-domain pretrain — bounded exam for whether mixed L1 reduces W need.

  python _stage235_mixed_l1_probe.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, mean_core_cos
v0 = v8('results')
v1 = v0 / 'stage235_decision.json'
v2 = v0 / 'stage235_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/external_tinystories_100k_85.txt')
v5 = v8('data/_wikitext103_train.txt')
v6 = 235

def main() -> v7:
    v9 = v94.v51()
    v9.v52('--smoke', action='store_true')
    v10 = v9.v53()
    v11 = v95.v11('cuda' if v95.v126.v114() else 'cpu')
    v12 = 50 if v10.v54 else 400
    v13 = 60 if v10.v54 else 800
    v14 = 60 if v10.v54 else 400
    v15 = 10 if v10.v54 else 50
    v16 = 250 if v10.v54 else 6000
    v17 = v96.v55(v6)
    v56, v57, v58, v59 = v60()
    v18 = v97.v61(v76(v115.v98))
    v19 = v18.v99(v100) or 0
    v20 = v127.v116(v18, v58, v19, v18.v128()).v62(v11)
    with v5.v101('r', encoding='utf-8', errors='ignore') as v63:
        v64 = v63.v102(2000000)
    v21 = v75(v129.v117((v130 for v130 in v140.v138('[A-Za-z][a-z]{2,}', v64) if v120(v130) <= 14)))[:v14]
    v22 = v118(v59, v18.v128()).v62(v11)
    v22.v65(v95.v119(v3, map_location=v11, weights_only=False)['model'])
    v22.v66()
    v23 = v67(v22, v58, v11)
    v24 = v103.v68(v23, v21)
    v69, v70 = v104.v71(v4.v105(encoding='utf-8', errors='ignore'), v18, v19, max_lines=v16)
    v25 = v106.v72(v96.v55(v6 + 1), v10.v54)
    v73, v74 = v104.v71(v25, v18, v19, max_lines=v16, min_line_len=20)
    v26 = v4.v105(encoding='utf-8', errors='ignore').v107()[:v16]
    v27 = v25.v107()[:v16]
    v28: v75[v76] = []
    for v29 in v77(v108(v120(v26), v120(v27))):
        if v29 < v120(v26) and v26[v29].v121():
            v28.v122(v26[v29])
        if v29 < v120(v27) and v27[v29].v121():
            v28.v122(v27[v29])
    v78, v79 = v104.v71('\n'.v109(v28), v18, v19, max_lines=v16 * 2, min_line_len=20)
    v30 = v103.v80(v22, v69, v70, v20, v19, v11, v12, v6 + 2)
    v31 = v103.v80(v22, v78, v79, v20, v19, v11, v12, v6 + 3)
    v32 = v103.v80(v22, v73, v74, v20, v19, v11, v12, v6 + 4)
    v33 = v67(v30, v58, v11)
    v34 = v67(v31, v58, v11)
    v35 = v67(v32, v58, v11)
    v36 = v81(v23, v33, v21)
    v37 = v81(v23, v34, v21)
    v38 = v81(v23, v35, v21)
    v39 = v81(v35, v34, v21)
    v40 = v81(v35, v33, v21)
    v82, v83 = v103.v84(v131(256).v62(v11), v103.v68(v34, v21), v24, v17, v13, v11)
    v85, v83 = v103.v84(v131(256).v62(v11), v103.v68(v33, v21), v24, v17, v13, v11)
    with v5.v101('r', encoding='utf-8', errors='ignore') as v63:
        v86 = v75(v129.v117((v135.v134(1) for v135 in v141.v139(v63.v102(2000000)) if v120(v135.v134(1)) >= 5)))
    v41 = v110(v123(v86), v17, v15 + 10)[:v15]
    v42 = v86[:v15]
    v87, v88 = v103.v89(v23, v41, v42, v17)
    v43 = v106.v90(v87, v88, v35, v41, v42, v17, query_x=v106.v124(v82))
    v44 = v106.v90(v87, v88, v35, v41, v42, v17, query_x=v106.v124(v85))
    v45 = v39 >= v40 + 0.03
    v46 = v43 >= v44 - 0.05
    v47 = v37 >= v36 - 0.02
    v48 = 'MIXED_L1_PROBE_OK' if v45 and v46 and (v43 >= 0.72) else 'MIXED_L1_PROBE_PARTIAL' if v45 or v46 else 'MIXED_L1_PROBE_NO'
    v49 = {'stage': 235, 'branch': 'pretrain_L1_mixed_domain_probe', 'overall': v48, 'gates': {'G_mixed_closer_to_code_than_prose_only': v45, 'G_mixed_W_recall_not_worse': v46, 'G_mixed_cos_can_not_hurt': v47}, 'mean_cos_can_prose_ft': v36, 'mean_cos_can_mixed_ft': v37, 'mean_cos_code_vs_mixed': v39, 'mean_cos_code_vs_prose_ft': v40, 'recall_code_query_W_mixed': v43, 'recall_code_query_W_prose_ft': v44, 'arc_steps': v12, 'timestamp': v136.v132(v137.v133).v111()}
    v1.v91(v125.v112(v49, indent=2), encoding='utf-8')
    v2.v91(f'# Stage 235 mixed L1 probe\n\n**{v48}** mixed_recall={v43:.3f} prose_W={v44:.3f}\n', encoding='utf-8')
    v92(v125.v112(v49, indent=2))
    return 0
if v50 == '__main__':
    raise v93(v113())