"""
Stage 228b — fp-guided decoding (zero-train): memory as decoding-time scorer.

At a code return position, pick among 4 org-name candidates using:
  - head_only: LM logit on first BPE (baseline)
  - fp_retrieved: cos(fp(c), fp(retrieved_value))  # from slot via 227 qmap
  - fp_oracle:    cos(fp(c), fp(gold))             # retrieval upper bound
  - hybrid:       zscore(head) + zscore(fp_retrieved)

No text inject; arc_enc frozen; same mechanics as 194 recall extended to decode.

  python _stage228b_fp_guided_decode.py [--smoke]
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
v1 = v0 / 'stage228b_decision.json'
v2 = v0 / 'stage228b_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/_wikitext103_train.txt')
v5 = 2282

def log(v9: v74) -> None:
    v75(v9, flush=True)

def w_apply(v10):
    return lambda v128: v159.v129(v10.v160(v128), dim=-1)

def retrieve_value(v11, v12, v13, v14, v15, v16, v17):
    v18 = v13.v76(f'In the report {v15} was linked to the organization.', exclude=v16)
    if v18 is None:
        return (None, v18)
    v19 = v173(v14)(v18.v174(0))[0] if v14 is not None else v18
    v20 = v11 @ v19
    v21 = v7(v20.v130())
    return (v12[v21], v19)

def zscore(v22: v24[v25]) -> v24[v25]:
    v23 = v131.v77(v22, dtype=v131.v132)
    if v133(v23) < 2:
        return v22
    v9, v78 = (v23.v134(), v23.v135())
    if v78 < 1e-09:
        return [0.0] * v133(v22)
    return ((v23 - v9) / v78).v79()

@v85.v39()
def head_first_token_scores(v26, v27, v28, v29, v30, v31: v74, v32: v24[v74]) -> v24[v25]:
    v33 = v27.v136(v31).v33
    if not v33:
        return [-1000000000.0] * v133(v32)
    v34 = v85.v80([v33], dtype=v85.v137, device=v30)
    v35 = v34 == v28
    v81, v82, v82 = v26.v83(v29[v34], v35, ids=v34)
    v36 = v81[0, -1]
    v37 = []
    for v38 in v32:
        v84 = v27.v136(v38).v33
        v37.v138(v25(v36[v84[0]]) if v84 else -1000000000.0)
    return v37

@v85.v39()
def fp_scores(v40: v86, v41: v74, v32: v24[v74]) -> v24[v25]:
    v42 = v40.v87([v41])[0]
    v43 = v40.v87(v32)
    return [v25((v43[v21] * v42).v161()) for v21 in v162(v133(v32))]

def pick(v20: v24[v25], v32: v24[v74], v16: v74) -> v6:
    return v32[v7(v131.v130(v20))] == v16

def main() -> v7:
    v44 = v139.v88()
    v44.v89('--smoke', action='store_true')
    v45 = v44.v90()
    v30 = v85.v30('cuda' if v85.v175.v163() else 'cpu')
    v46 = 80 if v45.v91 else v140.v92
    v47 = 80 if v45.v91 else 600
    v48 = 100 if v45.v91 else v140.v93
    v49 = 80 if v45.v91 else v140.v94
    v50 = 12 if v45.v91 else 60
    v51 = 400 if v45.v91 else 8000
    v17 = v141.v95(v5)
    v96, v97, v98, v99 = v100()
    v27 = v142.v101(v74(v164.v143))
    v28 = v27.v144(v145) or 0
    v29 = v176.v165(v27, v98, v28, v27.v177()).v102(v30)
    with v4.v146('r', encoding='utf-8', errors='ignore') as v103:
        v104 = v103.v147(2000000)
    v52 = v24(v178.v166((v179 for v179 in v189.v187('[A-Za-z][a-z]{2,}', v104) if v133(v179) <= 14)))[:v49]
    v53 = v167(v99, v27.v177()).v102(v30)
    v53.v105(v85.v168(v3, map_location=v30, weights_only=False)['model'])
    v53.v106()
    v54 = v86(v53, v98, v30)
    v55 = v140.v107(v54, v52)
    v56 = v141.v95(227)
    with v4.v146('r', encoding='utf-8', errors='ignore') as v103:
        v108 = v24(v178.v166((v9.v183(1) for v9 in v190.v188(v103.v147(4000000)) if v133(v9.v183(1)) >= 5)))
    v57 = v148(v169(v108), v56, v50 + 10)[:v50]
    v58 = v108[:v50]
    v109, v12 = v140.v110(v54, v57, v58, v56)
    v59 = v149.v111(v141.v95(v5 + 1), v45.v91)
    v112, v113 = v150.v114(v59, v27, v28, max_lines=v51, min_line_len=20)
    v60 = v140.v115(v53, v112, v113, v29, v28, v30, v46, v5 + 2)
    v61 = v86(v60, v98, v30)
    v62 = v140.v107(v61, v52)
    v14, v116 = v140.v117(v180(256).v102(v30), v62, v55, v56, v48, v30)
    v63 = v149.v118(v53, v112, v113, v29, v28, v30, v47, v5 + 3)
    v64 = {'head_only': 0, 'fp_retrieved': 0, 'fp_oracle': 0, 'hybrid': 0}
    v65 = 0
    v66 = 0
    for v15, v16 in v119(v57, v58):
        v32 = [v16] + [v58[(v21 + 1) % v133(v58)] for v21 in v162(3)]
        v17.v151(v32)
        v31 = f'def org_of_{v15}():\n    return '
        v152, v82 = v153(v109, v12, v61, v14, v15, v16, v17)
        if v152 is None:
            continue
        v65 += v7(v152 == v16)
        v120 = v154(v63, v27, v28, v29, v30, v31, v32)
        v121 = v155(v54, v152, v32)
        v122 = v155(v54, v16, v32)
        v123 = [v23 + v170 for v23, v170 in v119(v184(v120), v184(v121))]
        v64['head_only'] += v7(v171(v120, v32, v16))
        v64['fp_retrieved'] += v7(v171(v121, v32, v16))
        v64['fp_oracle'] += v7(v171(v122, v32, v16))
        v64['hybrid'] += v7(v171(v123, v32, v16))
        v66 += 1
    v66 = v124(1, v66)
    v67 = {v125: v64[v125] / v66 for v125 in v64}
    v68 = v65 / v66
    v69 = v67['fp_retrieved'] - v67['head_only']
    v70 = v67['fp_retrieved'] >= 0.7 and v69 >= 0.08
    v71 = v67['hybrid'] >= v124(v67['head_only'], v67['fp_retrieved']) + 0.02
    v72 = 'FP_GUIDED_DECODE_YES' if v70 else 'FP_GUIDED_DECODE_PARTIAL' if v69 > 0.03 else 'FP_GUIDED_DECODE_NO'
    v37 = {'stage': '228b', 'overall': v72, 'contract': 'memory is a decoding-time scorer (zero-train fp cos vs retrieved value)', 'gates': {'G_fp_ge_0p70': v67['fp_retrieved'] >= 0.7, 'G_lift_vs_head_ge_0p08': v69 >= 0.08, 'G_hybrid': v71}, 'recall_retrieved_exact': v68, 'align_W_bwd': v116, 'mean_cos_code': v25((v55 * v62).v161(-1).v134()), 'accuracy_4way': v67, 'lift_fp_retrieved_minus_head': v69, 'lift_oracle_minus_head': v67['fp_oracle'] - v67['head_only'], 'n_items': v66, 'timestamp': v185.v181(v186.v182).v156()}
    v1.v126(v172.v157(v37, indent=2), encoding='utf-8')
    v2.v126(f"# Stage 228b fp-guided decode\n\n**{v72}** head={v67['head_only']:.3f} fp_ret={v67['fp_retrieved']:.3f} fp_oracle={v67['fp_oracle']:.3f} hybrid={v67['hybrid']:.3f}\n", encoding='utf-8')
    v75(v172.v157(v37, indent=2))
    return 0
if v73 == '__main__':
    raise v127(v158())