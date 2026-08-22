"""
Stage 234 — Compositional W (228 algebra branch).

Two-step domain drift P1 → prose → code vs direct P1 → code. Test whether
qmap adapters compose: W_comp ≈ W_direct on canonical slot read.

  python _stage234_compositional_W.py [--smoke]
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
from _tapelm_ext import DomainAdapter, compose_w_bwd
v0 = v8('results')
v1 = v0 / 'stage234_decision.json'
v2 = v0 / 'stage234_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/external_tinystories_100k_85.txt')
v5 = v8('data/_wikitext103_train.txt')
v6 = 234

def main() -> v7:
    v9 = v88.v47()
    v9.v48('--smoke', action='store_true')
    v10 = v9.v49()
    v11 = v89.v11('cuda' if v89.v118.v109() else 'cpu')
    v12 = 60 if v10.v50 else v90.v51
    v13 = 80 if v10.v50 else v90.v52
    v14 = 60 if v10.v50 else v90.v53
    v15 = 10 if v10.v50 else 55
    v16 = 300 if v10.v50 else 8000
    v17 = v91.v54(v6)
    v55, v56, v57, v58 = v59()
    v18 = v92.v60(v93(v110.v94))
    v19 = v18.v95(v96) or 0
    v20 = v119.v111(v18, v57, v19, v18.v120()).v61(v11)
    with v5.v97('r', encoding='utf-8', errors='ignore') as v62:
        v63 = v62.v98(2000000)
    v21 = v99(v121.v112((v122 for v122 in v132.v130('[A-Za-z][a-z]{2,}', v63) if v133(v122) <= 14)))[:v14]
    v22 = v113(v58, v18.v120()).v61(v11)
    v22.v64(v89.v114(v3, map_location=v11, weights_only=False)['model'])
    v22.v65()
    v23 = v66(v22, v57, v11)
    v24 = v90.v67(v23, v21)
    v68, v69 = v100.v70(v4.v101(encoding='utf-8', errors='ignore'), v18, v19, max_lines=v16)
    v71, v72 = v100.v70(v104.v102(v91.v54(v6 + 1), v10.v50), v18, v19, max_lines=v16, min_line_len=20)
    v25 = v90.v73(v22, v68, v69, v20, v19, v11, v12, v6 + 2)
    v26 = v90.v73(v25, v71, v72, v20, v19, v11, v12, v6 + 3)
    v27 = v90.v73(v22, v71, v72, v20, v19, v11, v12, v6 + 4)
    v28 = v66(v25, v57, v11)
    v29 = v66(v26, v57, v11)
    v30 = v66(v27, v57, v11)
    v31 = v90.v67(v28, v21)
    v32 = v90.v67(v29, v21)
    v33 = v90.v67(v30, v21)
    v74, v75 = v90.v76(v123(256).v61(v11), v31, v24, v17, v13, v11)
    v77, v75 = v90.v76(v123(256).v61(v11), v32, v31, v17, v13, v11)
    v78, v75 = v90.v76(v123(256).v61(v11), v33, v24, v17, v13, v11)
    v34 = v79(v74, v77)
    with v5.v97('r', encoding='utf-8', errors='ignore') as v62:
        v80 = v99(v121.v112((v127.v126(1) for v127 in v134.v131(v62.v98(4000000)) if v133(v127.v126(1)) >= 5)))
    v35 = v103(v115(v80), v17, v15 + 10)[:v15]
    v36 = v80[:v15]
    v81, v82 = v90.v83(v23, v35, v36, v17)
    v37 = v104.v84(v81, v82, v30, v35, v36, v17, query_x=v104.v116(v78))
    v38 = v104.v84(v81, v82, v29, v35, v36, v17, query_x=v104.v116(v34))
    v39 = v104.v84(v81, v82, v30, v35, v36, v17, query_x=v104.v116(v34))
    v40 = v37 - v39
    v41 = v39 >= v37 - 0.1
    v42 = v39 >= 0.7
    v43 = v105(v38 - v39) <= 0.15
    v44 = 'COMPOSITIONAL_W_OK' if v41 and v42 else 'COMPOSITIONAL_W_PARTIAL' if v41 or v39 >= 0.65 else 'COMPOSITIONAL_W_NO'
    v45 = {'stage': 234, 'branch': '228_algebra_compositional_W', 'overall': v44, 'gates': {'G_composed_within_0p10_of_direct': v41, 'G_composed_recall_ge_0p70': v42, 'G_seq_vs_direct_encoder_close': v43}, 'recall_W_direct_on_code_enc': v37, 'recall_W_composed_on_code_enc': v39, 'recall_W_composed_on_seq_enc': v38, 'gap_direct_minus_composed': v40, 'timestamp': v128.v124(v129.v125).v106()}
    v1.v85(v117.v107(v45, indent=2), encoding='utf-8')
    v2.v85(f'# Stage 234 compositional W\n\n**{v44}** direct={v37:.3f} composed={v39:.3f} Δ={v40:.3f}\n', encoding='utf-8')
    v86(v117.v107(v45, indent=2))
    return 0
if v46 == '__main__':
    raise v87(v108())