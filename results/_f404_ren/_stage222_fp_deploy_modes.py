"""
Stage 222 — Which fp pipeline actually needs W after arc_enc shift?

Compares recall modes on the same fact bank after one Stories shift:
  old/old, old keys + new query, W on keys only, W on query only, W both (221), oracle reindex.

  python _stage222_fp_deploy_modes.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
v0 = v8('results')
v1 = v0 / 'stage222_decision.json'
v2 = v0 / 'stage222_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/external_tinystories_100k_85.txt')
v5 = v8('data/_wikitext103_train.txt')
v6 = 222

def recall_new_queries(v9, v10, v11, v12, v13, v14, v15, v16=None, v17=None):
    """bank_q for ctx_fp; keys optionally transformed; queries optionally transformed."""
    v42, v43 = (0, 0)
    for v44, v45 in v46(v13, v14):
        v47 = v11.v95(f'In the report {v44} was linked to the organization.', exclude=v45)
        if v47 is None:
            continue
        v48 = v17(v47.v145(0))[0] if v17 else v47
        v49 = v16(v9) if v16 else v9
        v50 = [v45] + [v14[(v121 + 1) % v153(v14)] for v121 in v146(3)]
        v15.v96(v50)
        v51 = v50.v97(v45)
        v52 = []
        for v53 in v50:
            v98 = [v121 for v121, v143 in v147(v10) if v143 == v53]
            v52.v92(v116((v49[v98] @ v48).v90()) if v98 else -1.0)
        v42 += v7(v154('numpy').v133(v52) == v51)
        v43 += 1
    return (v42 / v90(1, v43), v43)

def main() -> v7:
    v18 = v99.v54()
    v18.v55('--smoke', action='store_true')
    v19 = v18.v56()
    v20 = v100.v20('cuda' if v100.v134.v122() else 'cpu')
    v21 = 80 if v19.v57 else v101.v58
    v22 = 100 if v19.v57 else v101.v59
    v23 = 80 if v19.v57 else v101.v60
    v24 = 12 if v19.v57 else 60
    v15 = v102.v61(v6)
    v62, v63, v64, v65 = v66()
    v25 = v103.v67(v104(v123.v105))
    v26 = v25.v106(v107) or 0
    v27 = v135.v124(v25, v64, v26, v25.v136()).v68(v20)
    with v5.v108('r', encoding='utf-8', errors='ignore') as v69:
        v70 = v69.v109(2000000)
    v28 = v110(v137.v125((v138 for v138 in v157.v155('[A-Za-z][a-z]{2,}', v70) if v153(v138) <= 14)))[:v23]
    v29 = v126(v65, v25.v136()).v68(v20)
    v29.v71(v100.v127(v3, map_location=v20, weights_only=False)['model'])
    v29.v72()
    v30 = v73(v29, v64, v20)
    with v5.v108('r', encoding='utf-8', errors='ignore') as v69:
        v74 = v110(v137.v125((v149.v148(1) for v149 in v158.v156(v69.v109(4000000)) if v153(v149.v148(1)) >= 5)))
    v13 = v111(v128(v74), v15, v24 + 10)[:v24]
    v14 = v74[:v24]
    v75, v10 = v101.v76(v30, v13, v14, v15)
    v31 = v4.v77(encoding='utf-8', errors='ignore')
    v78, v79 = v112.v80(v31, v25, v26, max_lines=500 if v19.v57 else 8000)
    v32 = v101.v81(v29, v78, v79, v27, v26, v20, v21, v6 + 1)
    v33 = v73(v32, v64, v20)
    v82, v83 = v101.v76(v33, v13, v14, v15)
    v34 = v101.v84(v30, v28)
    v35 = v101.v84(v33, v28)
    v85, v86 = v101.v87(v139(256).v68(v20), v34, v35, v15, v22, v20)

    def w_raw(v88):
        return v129.v113(v85.v130(v88), dim=-1)
    v36 = {}
    v36['M1_old_keys_old_query'], v83 = v89(v75, v10, v30, v30, v13, v14, v15)
    v36['M2_old_keys_new_query_no_W'], v83 = v89(v75, v10, v33, v30, v13, v14, v15)
    v36['M3_W_keys_old_query'], v83 = v89(v75, v10, v30, v30, v13, v14, v15, key_x=v114, query_x=None)
    v36['M4_old_keys_W_old_query'], v83 = v89(v75, v10, v30, v30, v13, v14, v15, key_x=None, query_x=v114)
    v36['M5_W_keys_W_old_query_221'], v83 = v89(v75, v10, v30, v30, v13, v14, v15, key_x=v114, query_x=v114)
    v36['M6_W_keys_new_query'], v83 = v89(v75, v10, v33, v30, v13, v14, v15, key_x=v114, query_x=None)
    v36['M7_oracle_new_keys_new_query'], v83 = v89(v82, v10, v33, v33, v13, v14, v15)
    v37 = v90(v36, key=v36.v115)
    v38 = 'FP_DEPLOY_MODES_OK' if v36['M5_W_keys_W_old_query_221'] >= 0.75 else 'FP_DEPLOY_MODES_MIXED'
    v39 = {'stage': 222, 'overall': v38, 'modes': v36, 'best_mode': v37, 'align_W_core': v86, 'mean_cos_word_shift': v116((v34 * v35).v150(-1).v131()), 'interpretation': 'If M2 ~ M7 and >> M5, W is for legacy old-fp extraction; if M6 ~ M7, deploy new encoder on queries + W on keys only.', 'timestamp': v151.v140(v152.v141).v117()}
    v1.v91(v132.v118(v39, indent=2), encoding='utf-8')
    v40 = ['# Stage 222 deploy modes\n'] + [f'- **{v142}**: {v143:.3f}' for v142, v143 in v36.v144()]
    v40.v92(f'\n**{v38}** best={v37}\n')
    v2.v91('\n'.v119(v40), encoding='utf-8')
    v93(v132.v118(v39, indent=2))
    return 0
if v41 == '__main__':
    raise v94(v120())