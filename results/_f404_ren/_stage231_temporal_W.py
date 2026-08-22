"""
Stage 231 — Temporal W: matched qmap vs wrong-era qmap on cross-domain read.

Reuses 227 protocol: canonical bank; code encoder; W_prose_bwd vs W_code_bwd on query.
Gate: matched W recall ≥ wrong W + margin (224-style cross-drop).

  python _stage231_temporal_W.py [--smoke]
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
v1 = v0 / 'stage231_decision.json'
v2 = v0 / 'stage231_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/external_tinystories_100k_85.txt')
v5 = v8('data/_wikitext103_train.txt')
v6 = 231

def main() -> v7:
    v9 = v89.v47()
    v9.v48('--smoke', action='store_true')
    v10 = v9.v49()
    v11 = v90.v11('cuda' if v90.v118.v109() else 'cpu')
    v12 = 80 if v10.v50 else v91.v51
    v13 = 100 if v10.v50 else v91.v52
    v14 = 80 if v10.v50 else v91.v53
    v15 = 12 if v10.v50 else 60
    v16 = 400 if v10.v50 else 8000
    v17 = v92.v54(v6)
    v55, v56, v57, v58 = v59()
    v18 = v93.v60(v94(v110.v95))
    v19 = v18.v96(v97) or 0
    v20 = v119.v111(v18, v57, v19, v18.v120()).v61(v11)
    with v5.v98('r', encoding='utf-8', errors='ignore') as v62:
        v63 = v62.v99(2000000)
    v21 = v100(v121.v112((v122 for v122 in v133.v131('[A-Za-z][a-z]{2,}', v63) if v134(v122) <= 14)))[:v14]
    v22 = v113(v58, v18.v120()).v61(v11)
    v22.v64(v90.v114(v3, map_location=v11, weights_only=False)['model'])
    v22.v65()
    v23 = v66(v22, v57, v11)
    v24 = v91.v67(v23, v21)
    with v5.v98('r', encoding='utf-8', errors='ignore') as v62:
        v68 = v100(v121.v112((v128.v127(1) for v128 in v135.v132(v62.v99(4000000)) if v134(v128.v127(1)) >= 5)))
    v25 = v101(v115(v68), v17, v15 + 10)[:v15]
    v26 = v68[:v15]
    v69, v70 = v91.v71(v23, v25, v26, v17)
    v72, v73 = v102.v74(v4.v103(encoding='utf-8', errors='ignore'), v18, v19, max_lines=v16)
    v75, v76 = v102.v74(v105.v104(v92.v54(v6 + 1), v10.v50), v18, v19, max_lines=v16, min_line_len=20)
    v27 = v91.v77(v22, v72, v73, v20, v19, v11, v12, v6 + 2)
    v28 = v91.v77(v22, v75, v76, v20, v19, v11, v12, v6 + 3)
    v29 = v66(v27, v57, v11)
    v30 = v66(v28, v57, v11)
    v31 = v91.v67(v29, v21)
    v32 = v91.v67(v30, v21)
    v78, v79 = v91.v80(v38(256).v61(v11), v31, v24, v17, v13, v11)
    v81, v79 = v91.v80(v38(256).v61(v11), v32, v24, v17, v13, v11)
    v33 = v105.v82(v69, v70, v30, v25, v26, v17, query_x=v105.v116(v78))
    v34 = v105.v82(v69, v70, v30, v25, v26, v17, query_x=v105.v116(v81))
    v35 = v105.v82(v69, v70, v29, v25, v26, v17, query_x=v105.v116(v78))
    v36 = v83(v23, v29, v21)
    v37 = v83(v23, v30, v21)

    def pick_W(v84: v66) -> v38:
        v85 = v83(v23, v84, v21)
        return v78 if v123(v85 - v36) <= v123(v85 - v37) else v81
    v39 = v105.v82(v69, v70, v30, v25, v26, v17, query_x=v105.v116(v124(v30)))
    v40 = v34 - v33
    v41 = v40 >= 0.08
    v42 = v39 >= v33 + 0.05
    v43 = v35 >= 0.75
    v44 = 'TEMPORAL_W_OK' if v41 and v42 and (v34 >= 0.72) else 'TEMPORAL_W_PARTIAL' if v41 or (v40 >= 0.03 and v34 >= 0.8) else 'TEMPORAL_W_NO'
    v45 = {'stage': 231, 'overall': v44, 'gates': {'G_matched_beats_wrong_W': v41, 'G_era_pick_beats_wrong': v42, 'G_prose_self_qmap': v43}, 'recall_wrong_W_prose_on_code': v33, 'recall_matched_W_code_on_code': v34, 'recall_era_picked_on_code': v39, 'recall_prose_W_on_prose': v35, 'margin_matched_minus_wrong': v40, 'mean_cos_can_prose': v36, 'mean_cos_can_code': v37, 'timestamp': v129.v125(v130.v126).v106()}
    v1.v86(v117.v107(v45, indent=2), encoding='utf-8')
    v2.v86(f'# Stage 231 temporal W\n\n**{v44}** matched={v34:.3f} wrong={v33:.3f} Δ={v40:.3f}\n', encoding='utf-8')
    v87(v117.v107(v45, indent=2))
    return 0
if v46 == '__main__':
    raise v88(v108())