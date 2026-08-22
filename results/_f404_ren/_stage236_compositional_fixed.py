"""
Stage 236 — Fixed-exam compositional W (productize 234 algebra).

Persist a frozen fact list; re-run chained qmap vs direct W on that exam.
Gate: composed recall within 0.10 of direct; both ≥ 0.70.

  python _stage236_compositional_fixed.py [--smoke]
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
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, compose_w_bwd
v0 = v11('results')
v1 = v11('data')
v2 = v0 / 'stage236_decision.json'
v3 = v0 / 'stage236_mini.md'
v4 = v1 / 'stage236_fixed_facts.json'
v5 = v11('checkpoints/stage191_p1_curve.pt')
v6 = v11('data/external_tinystories_100k_85.txt')
v7 = v11('data/_wikitext103_train.txt')
v8 = 236

def ensure_exam(v12: v10, v13: v51) -> v9:
    if v4.v93() and (not v13):
        return v117.v94(v4.v110(encoding='utf-8'))
    v14 = v95.v52(v8)
    with v7.v96('r', encoding='utf-8', errors='ignore') as v53:
        v54 = v97(v9.v118((v135.v134(1) for v135 in v140.v138(v53.v108(4000000)) if v113(v135.v134(1)) >= 5)))
    from _stage192_fp_lexicon import gen_fakes
    v15 = v98(v119(v54), v14, v12 + 10)[:v12]
    v16 = v54[:v12]
    v17 = {'seed': v8, 'subjects': v15, 'values': v16, 'n': v12}
    if not v13:
        v4.v120.v99(parents=True, exist_ok=True)
        v4.v90(v117.v115(v17, indent=2, ensure_ascii=False), encoding='utf-8')
    return v17

def main() -> v10:
    v18 = v100.v55()
    v18.v56('--smoke', action='store_true')
    v19 = v18.v57()
    v20 = v101.v20('cuda' if v101.v127.v121() else 'cpu')
    v21 = 60 if v19.v13 else v102.v58
    v22 = 80 if v19.v13 else v102.v59
    v23 = 60 if v19.v13 else v102.v60
    v12 = 10 if v19.v13 else 55
    v24 = 300 if v19.v13 else 8000
    v14 = v95.v52(v8)
    v25 = v61(v12, v19.v13)
    v15, v16 = (v25['subjects'][:v12], v25['values'][:v12])
    v62, v63, v64, v65 = v66()
    v26 = v103.v67(v104(v122.v105))
    v27 = v26.v106(v107) or 0
    v28 = v128.v123(v26, v64, v27, v26.v129()).v68(v20)
    with v7.v96('r', encoding='utf-8', errors='ignore') as v53:
        v69 = v53.v108(2000000)
    v29 = v97(v9.v118((v130 for v130 in v141.v139('[A-Za-z][a-z]{2,}', v69) if v113(v130) <= 14)))[:v23]
    v30 = v124(v65, v26.v129()).v68(v20)
    v30.v70(v101.v125(v5, map_location=v20, weights_only=False)['model'])
    v30.v71()
    v31 = v72(v30, v64, v20)
    v32 = v102.v73(v31, v29)
    v74, v75 = v102.v76(v31, v15, v16, v14)
    v77, v78 = v109.v79(v6.v110(encoding='utf-8', errors='ignore'), v26, v27, max_lines=v24)
    v80, v81 = v109.v79(v112.v111(v95.v52(v8 + 1), v19.v13), v26, v27, max_lines=v24, min_line_len=20)
    v33 = v102.v82(v30, v77, v78, v28, v27, v20, v21, v8 + 2)
    v34 = v102.v82(v33, v80, v81, v28, v27, v20, v21, v8 + 3)
    v35 = v102.v82(v30, v80, v81, v28, v27, v20, v21, v8 + 4)
    v36 = v72(v33, v64, v20)
    v37 = v72(v34, v64, v20)
    v38 = v72(v35, v64, v20)
    v39 = v102.v73(v36, v29)
    v40 = v102.v73(v37, v29)
    v41 = v102.v73(v38, v29)
    v83, v84 = v102.v85(v131(256).v68(v20), v39, v32, v14, v22, v20)
    v86, v84 = v102.v85(v131(256).v68(v20), v40, v39, v14, v22, v20)
    v87, v84 = v102.v85(v131(256).v68(v20), v41, v32, v14, v22, v20)
    v42 = v88(v83, v86)
    v43 = v112.v89(v74, v75, v38, v15, v16, v14, query_x=v112.v126(v87))
    v44 = v112.v89(v74, v75, v38, v15, v16, v14, query_x=v112.v126(v42))
    v45 = v43 - v44
    v46 = v44 >= v43 - 0.1
    v47 = v44 >= 0.7
    v48 = 'COMPOSITIONAL_FIXED_OK' if v46 and v47 else 'COMPOSITIONAL_FIXED_PARTIAL' if v46 or v44 >= 0.65 else 'COMPOSITIONAL_FIXED_NO'
    v49 = {'stage': 236, 'overall': v48, 'gates': {'G_composed_within_0p10_of_direct': v46, 'G_composed_ge_0p70': v47}, 'exam_path': v104(v4) if v4.v93() else 'ephemeral_smoke', 'n_facts': v113(v15), 'recall_direct': v43, 'recall_composed': v44, 'gap_direct_minus_composed': v45, 'api': 'compose_w_bwd', 'timestamp': v136.v132(v137.v133).v114()}
    v2.v90(v117.v115(v49, indent=2), encoding='utf-8')
    v3.v90(f'# Stage 236 compositional fixed\n\n**{v48}** direct={v43:.3f} composed={v44:.3f}\n', encoding='utf-8')
    v91(v117.v115(v49, indent=2))
    return 0
if v50 == '__main__':
    raise v92(v116())