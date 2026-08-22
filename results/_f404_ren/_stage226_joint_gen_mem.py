"""
Stage 226 — Joint generation + memory (prose slots while generating in code domain).

Requires 227-style canonical bank. Protocol:
  - Write facts with frozen arc_enc (canonical keys).
  - Code domain: head_code (frozen arc_enc upper); retrieval uses W_code **qmap** (227).
  - Inject retrieved value into code-shaped prompt; 4-way rank of gold among distractors.

Gates:
  G_retrieve  cross-domain recall (canonical + W_code) >= 0.70
  G_joint     gen accuracy with gold inject >= baseline without inject + margin
              OR with retrieved inject >= 0.5 * gold-inject (retrieval useful)

  python _stage226_joint_gen_mem.py [--smoke]
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
from _tapelm_ext import DomainAdapter
v0 = v8('results')
v1 = v0 / 'stage226_decision.json'
v2 = v0 / 'stage226_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/external_tinystories_100k_85.txt')
v5 = v8('data/_wikitext103_train.txt')
v6 = 226

def log(v9: v59) -> None:
    v60(v9, flush=True)

def w_apply(v10):
    return lambda v105: v140.v106(v10.v141(v105), dim=-1)

def retrieve_one(v11, v12, v13, v14):
    v15 = v14(v11)
    v16 = v15 @ v13
    v17 = v7(v16.v107())
    return (v12[v17], v47(v16[v17]))

def main() -> v7:
    v18 = v108.v61()
    v18.v62('--smoke', action='store_true')
    v19 = v18.v63()
    v20 = v109.v20('cuda' if v109.v158.v142() else 'cpu')
    v21 = 80 if v19.v64 else v110.v65
    v22 = 80 if v19.v64 else 600
    v23 = 100 if v19.v64 else v110.v66
    v24 = 80 if v19.v64 else v110.v67
    v25 = 12 if v19.v64 else 40
    v26 = 400 if v19.v64 else 8000
    v27 = v111.v68(v6)
    v69, v70, v71, v72 = v73()
    v28 = v112.v74(v59(v143.v113))
    v29 = v28.v114(v115) or 0
    v30 = v159.v144(v28, v71, v29, v28.v160()).v75(v20)
    with v5.v116('r', encoding='utf-8', errors='ignore') as v76:
        v77 = v76.v117(2000000)
    v31 = v118(v161.v145((v162 for v162 in v179.v176('[A-Za-z][a-z]{2,}', v77) if v149(v162) <= 14)))[:v24]
    v32 = v146(v72, v28.v160()).v75(v20)
    v32.v78(v109.v147(v3, map_location=v20, weights_only=False)['model'])
    v32.v79()
    v33 = v80(v32, v71, v20)
    v34 = v110.v81(v33, v31)
    with v5.v116('r', encoding='utf-8', errors='ignore') as v76:
        v82 = v118(v161.v145((v9.v171(1) for v9 in v180.v177(v76.v117(4000000)) if v149(v9.v171(1)) >= 5)))
    v35 = v119(v148(v82), v27, v25 + 10)[:v25]
    v36 = v82[:v25]
    v83, v12 = v110.v84(v33, v35, v36, v27)
    v37 = v120.v85(v111.v68(v6 + 1), v19.v64)
    v86, v87 = v121.v88(v37, v28, v29, max_lines=v26, min_line_len=20)
    v38 = v110.v89(v32, v86, v87, v30, v29, v20, v21, v6 + 2)
    v39 = v80(v38, v71, v20)
    v40 = v110.v81(v39, v31)
    v90, v91 = v110.v92(v163(256).v75(v20), v34, v40, v27, v23, v20)
    v93, v94 = v110.v92(v163(256).v75(v20), v40, v34, v27, v23, v20)
    v41 = v42 = 0
    v43 = []
    for v95, v96 in v97(v35, v36):
        v13 = v39.v122(f'In the report {v95} was linked to the organization.', exclude=v96)
        if v13 is None:
            continue
        v98 = v164(v93)(v13.v165(0))[0]
        v123, v124 = v125(v83, v12, v98, key_x=lambda v11: v11)
        v41 += v7(v123 == v96)
        v42 += 1
        v43.v126((v95, v96, v123))
    v44 = v41 / v127(1, v42)
    v45 = v120.v99(v32, v86, v87, v30, v29, v20, v22, v6 + 3)
    v46 = v128((1 for v152, v172, v173 in v43 if v173 == v172)) / v127(1, v149(v43))

    def rank_gold_after_inject(v100, v101: v118[v166[v59, v59, v59]]) -> v47:
        """(ctx_prefix, inject_value, gold) → fraction where gold scores highest among {gold}+3 distractors."""
        v41 = v42 = 0
        for v129, v130, v96 in v101:
            v77 = f'# memory: {v130}\ndef label():\n    return '
            v131 = v28.v167(v77).v131
            if not v131:
                continue
            v132 = v109.v150([v131], dtype=v109.v168, device=v20)
            v133 = v132 == v29
            v151, v152, v152 = v100.v153(v30[v132], v133, ids=v132)
            v134 = v151[0, -1]
            v135 = [v96] + [v36[(v17 + 1) % v149(v36)] for v17 in v178(3)]
            v27.v154(v135)
            v16 = []
            for v136 in v135:
                v155 = v28.v167(v136).v131
                v16.v126(v47(v134[v155[0]]) if v155 else -1000000000.0)
            v41 += v7(v135[v7(v181.v107(v16))] == v96)
            v42 += 1
        return v41 / v127(1, v42)
    v48 = [(v95, v96, v96) for v95, v96, v152 in v43]
    v49 = [(v95, v123, v96) for v95, v96, v123 in v43]
    v50 = [(v95, 'UNKNOWN', v96) for v95, v96, v152 in v43]
    v51 = v102(v45, v50)
    v52 = v102(v45, v48)
    v53 = v102(v45, v49)
    v54 = v44 >= 0.7
    v55 = v52 >= v51 + 0.05 or v53 >= v51 + 0.05
    v56 = 'JOINT_GEN_MEM_OK' if v54 and v55 else 'JOINT_GEN_MEM_PARTIAL' if v54 or v55 else 'JOINT_GEN_MEM_NO'
    v57 = {'stage': 226, 'overall': v56, 'gates': {'G_retrieve': v54, 'G_joint': v55}, 'align_W_code_fwd': v91, 'align_W_code_bwd_qmap': v94, 'mean_cos_code_shift': v47((v34 * v40).v128(-1).v156()), 'recall_canonical_W_code_qmap': v44, 'ret_exact_rate': v46, 'gen_rank4': {'no_inject': v51, 'gold_inject': v52, 'retrieved_inject': v53}, 'n_items': v42, 'note': 'Canonical slots + code query via W_code qmap (227) + head_code; joint=4-way rank after inject', 'timestamp': v174.v169(v175.v170).v137()}
    v1.v103(v157.v138(v57, indent=2), encoding='utf-8')
    v2.v103(f'# Stage 226 joint gen+mem\n\n**{v56}** recall={v44:.3f} gen none/gold/ret={v51:.3f}/{v52:.3f}/{v53:.3f}\n', encoding='utf-8')
    v60(v157.v138(v57, indent=2))
    return 0
if v58 == '__main__':
    raise v104(v139())