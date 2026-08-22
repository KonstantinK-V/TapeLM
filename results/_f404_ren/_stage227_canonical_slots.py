"""
Stage 227 — Canonical slot storage + domain-conditioned read.

Write always with frozen arc_enc (canonical keys).
Read under domain shift X with either:
  P_keylift:  score(W_fwd @ K_can, q_domain)     # W: old→new (221)
  P_qmap:     score(K_can, W_bwd @ q_domain)     # W: new→old

Gate: cross-family (code query on canonical prose-era slots) drop vs same-family
      < 0.10 on best policy → CANONICAL_STORAGE_OK (one bank, disposable W).

  python _stage227_canonical_slots.py [--smoke]
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
v0 = v10('results')
v1 = v0 / 'stage227_decision.json'
v2 = v0 / 'stage227_mini.md'
v3 = v10('checkpoints/stage191_p1_curve.pt')
v4 = v10('data/external_tinystories_100k_85.txt')
v5 = v10('data/_wikitext103_train.txt')
v6 = v10('data/_stage224_code_corpus.txt')
v7 = 227

def log(v11: v8) -> None:
    v57(v11, flush=True)

def ensure_code(v12: v117.v58, v13: v59) -> v8:
    if v6.v118() and v6.v158().v119 > 10000:
        return v6.v120(encoding='utf-8')
    import _stage224_far_shift as s224
    return v121.v60(v12, n_lines=2000 if v13 else 12000)

def recall(v14, v15, v16, v17, v18, v12, v19=None, v20=None):
    v61, v62 = (0, 0)
    for v63, v64 in v65(v17, v18):
        v66 = v16.v122(f'In the report {v63} was linked to the organization.', exclude=v64)
        if v66 is None:
            continue
        v67 = v20(v66.v168(0))[0] if v20 else v66
        v68 = v19(v14) if v19 else v14
        v69 = [v64] + [v18[(v145 + 1) % v176(v18)] for v145 in v169(3)]
        v12.v123(v69)
        v70 = v69.v124(v64)
        v71 = []
        for v72 in v69:
            v125 = [v145 for v145, v170 in v171(v15) if v170 == v72]
            v71.v146(v103((v68[v125] @ v67).v114()) if v125 else -1.0)
        v61 += v9(v172.v159(v71) == v70)
        v62 += 1
    return v61 / v114(1, v62)

def w_apply(v21: v73):
    return lambda v126: v147.v127(v21.v148(v126), dim=-1)

def main() -> v9:
    v22 = v128.v74()
    v22.v75('--smoke', action='store_true')
    v23 = v22.v76()
    v24 = v129.v24('cuda' if v129.v160.v149() else 'cpu')
    v25 = 80 if v23.v13 else v130.v77
    v26 = 100 if v23.v13 else v130.v78
    v27 = 80 if v23.v13 else v130.v79
    v28 = 12 if v23.v13 else 60
    v29 = 400 if v23.v13 else 8000
    v12 = v117.v58(v7)
    v80, v81, v82, v83 = v84()
    v30 = v131.v85(v8(v150.v132))
    v31 = v30.v133(v134) or 0
    v32 = v161.v151(v30, v82, v31, v30.v162()).v86(v24)
    with v5.v135('r', encoding='utf-8', errors='ignore') as v87:
        v88 = v87.v136(2000000)
    v33 = v137(v163.v152((v164 for v164 in v179.v177('[A-Za-z][a-z]{2,}', v88) if v176(v164) <= 14)))[:v27]
    v34 = v153(v83, v30.v162()).v86(v24)
    v34.v89(v129.v154(v3, map_location=v24, weights_only=False)['model'])
    v34.v90()
    v35 = v91(v34, v82, v24)
    v36 = v130.v92(v35, v33)
    with v5.v135('r', encoding='utf-8', errors='ignore') as v87:
        v93 = v137(v163.v152((v11.v173(1) for v11 in v180.v178(v87.v136(4000000)) if v176(v11.v173(1)) >= 5)))
    v17 = v138(v155(v93), v12, v28 + 10)[:v28]
    v18 = v93[:v28]
    v94, v15 = v130.v95(v35, v17, v18, v12)
    v96, v97 = v139.v98(v4.v120(encoding='utf-8', errors='ignore'), v30, v31, max_lines=v29)
    v99, v100 = v139.v98(v140(v117.v58(v7 + 1), v23.v13), v30, v31, max_lines=v29, min_line_len=20)
    v101('arc shift prose(stories)…')
    v37 = v130.v102(v34, v96, v97, v32, v31, v24, v25, v7 + 2)
    v38 = v91(v37, v82, v24)
    v39 = v130.v92(v38, v33)
    v101('arc shift code…')
    v40 = v130.v102(v34, v99, v100, v32, v31, v24, v25, v7 + 3)
    v41 = v91(v40, v82, v24)
    v42 = v130.v92(v41, v33)
    v43 = v103((v36 * v39).v165(-1).v141())
    v44 = v103((v36 * v42).v165(-1).v141())
    v104, v105 = v130.v106(v73(256).v86(v24), v36, v39, v12, v26, v24)
    v107, v108 = v130.v106(v73(256).v86(v24), v36, v42, v12, v26, v24)
    v109, v110 = v130.v106(v73(256).v86(v24), v39, v36, v12, v26, v24)
    v111, v112 = v130.v106(v73(256).v86(v24), v42, v36, v12, v26, v24)
    v45 = v113(v94, v15, v35, v17, v18, v12)
    v46 = {}
    v46['same_prose_no_W'] = v113(v94, v15, v38, v17, v18, v12)
    v46['same_prose_keylift'] = v113(v94, v15, v38, v17, v18, v12, key_x=v156(v104))
    v46['same_prose_qmap'] = v113(v94, v15, v38, v17, v18, v12, query_x=v156(v109))
    v46['cross_code_no_W'] = v113(v94, v15, v41, v17, v18, v12)
    v46['cross_code_keylift'] = v113(v94, v15, v41, v17, v18, v12, key_x=v156(v107))
    v46['cross_code_qmap'] = v113(v94, v15, v41, v17, v18, v12, query_x=v156(v111))
    v47 = v114(v46['same_prose_keylift'], v46['same_prose_qmap'])
    v48 = v114(v46['cross_code_keylift'], v46['cross_code_qmap'])
    v49 = 'keylift' if v46['same_prose_keylift'] >= v46['same_prose_qmap'] else 'qmap'
    v50 = 'keylift' if v46['cross_code_keylift'] >= v46['cross_code_qmap'] else 'qmap'
    v51 = v47 - v48
    v52 = v45 - v48
    v53 = v51 < 0.1 and v48 >= 0.7
    v54 = 'CANONICAL_STORAGE_OK' if v53 else 'CANONICAL_STORAGE_PARTIAL' if v48 > v46['cross_code_no_W'] + 0.05 else 'CANONICAL_STORAGE_NO'
    v55 = {'stage': 227, 'overall': v54, 'gates': {'G_cross_drop_lt_0p10': v51 < 0.1, 'G_cross_recall_ge_0p70': v48 >= 0.7}, 'recall_canonical_baseline': v45, 'mean_cos_shift': {'stories': v43, 'code': v44}, 'align': {'W_s_fwd': v105, 'W_c_fwd': v108, 'W_s_bwd': v110, 'W_c_bwd': v112}, 'modes': v46, 'best_same_policy': v49, 'best_cross_policy': v50, 'best_same_recall': v47, 'best_cross_recall': v48, 'cross_drop_same_minus_cross': v51, 'drop_vs_canonical_baseline': v52, 'note': 'Keys always canonical (frozen P1). W disposable at read: keylift=old→domain, qmap=domain→old.', 'timestamp': v174.v166(v175.v167).v142()}
    v1.v115(v157.v143(v55, indent=2), encoding='utf-8')
    v2.v115(f'# Stage 227 canonical slots\n\n**{v54}** same={v47:.3f} cross={v48:.3f} drop={v51:.3f}\n', encoding='utf-8')
    v57(v157.v143(v55, indent=2))
    return 0
if v56 == '__main__':
    raise v116(v144())