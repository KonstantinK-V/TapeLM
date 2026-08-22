"""
Stage 223 — Matched vs wrong domain W on the same A-era slot bank.

Train shift_B + W_B (Stories), shift_C + W_C (wiki windows from P1 corpus).
Recall with matched W vs cross W (B keys scenario but C adapter, etc.).

  python _stage223_cross_adapter.py [--smoke]
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
v1 = v0 / 'stage223_decision.json'
v2 = v0 / 'stage223_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/external_tinystories_100k_85.txt')
v5 = v8('data/_wikitext103_train.txt')
v6 = 223

def main() -> v7:
    v9 = v90.v44()
    v9.v45('--smoke', action='store_true')
    v10 = v9.v46()
    v11 = v91.v11('cuda' if v91.v134.v119() else 'cpu')
    v12 = 80 if v10.v47 else v92.v48
    v13 = 100 if v10.v47 else v92.v49
    v14 = 80 if v10.v47 else v92.v50
    v15 = 12 if v10.v47 else 60
    v16 = v93.v51(v6)
    v52, v53, v54, v55 = v56()
    v17 = v94.v57(v95(v120.v96))
    v18 = v17.v97(v98) or 0
    v19 = v135.v121(v17, v54, v18, v17.v136()).v58(v11)
    with v5.v99('r', encoding='utf-8', errors='ignore') as v59:
        v60 = v59.v100(2000000)
    v20 = v61(v137.v122((v138 for v138 in v156.v151('[A-Za-z][a-z]{2,}', v60) if v123(v138) <= 14)))[:v14]
    v21 = v61(v101(v123(v20)))
    v22 = v124(v55, v17.v136()).v58(v11)
    v22.v62(v91.v125(v3, map_location=v11, weights_only=False)['model'])
    v22.v63()
    v23 = v64(v22, v54, v11)
    v24 = v92.v65(v23, v20)
    with v5.v99('r', encoding='utf-8', errors='ignore') as v59:
        v66 = v61(v137.v122((v147.v146(1) for v147 in v157.v152(v59.v100(4000000)) if v123(v147.v146(1)) >= 5)))
    v25 = v102(v126(v66), v16, v15 + 10)[:v15]
    v26 = v66[:v15]
    v67, v68 = v92.v69(v23, v25, v26, v16)
    v27 = v4.v70(encoding='utf-8', errors='ignore')
    v71, v72 = v103.v73(v27, v17, v18, max_lines=500 if v10.v47 else 8000)
    v28 = v92.v74(v22, v71, v72, v19, v18, v11, v12, v6 + 1)
    v29 = v92.v65(v64(v28, v54, v11), v20)
    v75, v76 = v92.v77(v139(256).v58(v11), v24, v29, v16, v13, v11)
    v30 = v92.v74(v22, v52, v53, v19, v18, v11, v12, v6 + 2)
    v31 = v92.v65(v64(v30, v54, v11), v20)
    v78, v76 = v92.v77(v139(256).v58(v11), v24, v31, v16, v13, v11)

    def tr(v79):
        return lambda v80: v140.v127(v79.v141(v80), dim=-1)
    v32 = v64(v28, v54, v11)

    def recall_k(v80, v81, v82):
        v104, v105 = (0, 0)
        for v106, v107 in v108(v25, v26):
            v109 = v81.v128(f'In the report {v106} was linked to the organization.', exclude=v107)
            if v109 is None:
                continue
            v110 = v82(v80)
            v111 = [v107] + [v26[(v142 + 1) % v123(v26)] for v142 in v101(3)]
            v16.v129(v111)
            v112 = v111.v130(v107)
            v113 = []
            for v114 in v111:
                v131 = [v142 for v142, v153 in v154(v68) if v153 == v114]
                v113.v143(v155((v110[v131] @ v109).v132()) if v131 else -1.0)
            v104 += v7(v158('numpy').v148(v113) == v112)
            v105 += 1
        return v104 / v132(1, v105)
    v33 = v83(v67, v32, v115(v75))
    v34 = v83(v67, v32, v115(v78))
    v35 = v64(v30, v54, v11)
    v36 = v83(v67, v35, v115(v78))
    v37 = v83(v67, v35, v115(v75))
    v84, v76 = v92.v85(v67, v68, v23, v25, v26, v16, v115(v75))
    v86, v76 = v92.v85(v67, v68, v23, v25, v26, v16, v115(v78))
    v38 = v33 - v34
    v39 = v36 - v37
    v40 = v33 >= 0.7 and v36 >= 0.7 and (v38 >= 0.03) and (v39 >= 0.03)
    v41 = 'DOMAIN_W_SWITCH_OK' if v40 else 'DOMAIN_W_SWITCH_PARTIAL'
    v42 = {'stage': 223, 'overall': v41, 'recall_B_new_query_W_B_keys': v33, 'recall_B_new_query_W_C_keys_WRONG': v34, 'recall_C_new_query_W_C_keys': v36, 'recall_C_new_query_W_B_keys_WRONG': v37, 'recall_legacy_221_W_B': v84, 'recall_legacy_221_W_C': v86, 'cross_drop_B': v38, 'cross_drop_C': v39, 'note': '221-style: old fp extract + W on keys and queries; cross = wrong adapter', 'timestamp': v149.v144(v150.v145).v116()}
    v1.v87(v133.v117(v42, indent=2), encoding='utf-8')
    v2.v87(f'# Stage 223 cross adapter\n\n**{v41}** B={v33:.3f} wrongWc={v34:.3f} C={v36:.3f} wrongWb={v37:.3f}\n', encoding='utf-8')
    v88(v133.v117(v42, indent=2))
    return 0
if v43 == '__main__':
    raise v89(v118())