"""
Stage 229 — Adversarial slot injection (contradiction / multi-hit).

Write two slots with same/similar keys, conflicting values.
Query entity; measure top-2 retrieval, score gap, whether both survive.

Contract expectation: fp memory returns candidates (feature); resolution is upper-layer.

  python _stage229_contradiction_slots.py [--smoke]
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
v0 = v7('results')
v1 = v0 / 'stage229_decision.json'
v2 = v0 / 'stage229_mini.md'
v3 = v7('checkpoints/stage191_p1_curve.pt')
v4 = v7('data/_wikitext103_train.txt')
v5 = 229

def main() -> v6:
    v8 = v75.v31()
    v8.v32('--smoke', action='store_true')
    v9 = v8.v33()
    v10 = v76.v10('cuda' if v76.v110.v97() else 'cpu')
    v11 = 8 if v9.v34 else 30
    v12 = v77.v35(v5)
    v36, v37, v38, v39 = v40()
    v13 = v78.v41(v79(v98.v80))
    v14 = v99(v39, v13.v111()).v42(v10)
    v14.v43(v76.v100(v3, map_location=v10, weights_only=False)['model'])
    v14.v44()
    v15 = v45(v14, v38, v10)
    with v4.v81('r', encoding='utf-8', errors='ignore') as v46:
        v47 = v82(v112.v101((v119.v118(1) for v119 in v123.v122(v46.v124(4000000)) if v84(v119.v118(1)) >= 5)))
    v16 = v83(v102(v47), v12, v11 + 5)[:v11]
    v17 = v47[:v11]
    v18 = v47[v11:2 * v11]
    if v84(v18) < v11:
        v18 = v82(v103(v47[:v11]))
    v48, v49, v50 = ([], [], [])
    for v51, v52, v53 in v54(v16, v17, v18):
        v55 = f'Official records state {v51} was director of {v52} in 1987 .'
        v56 = f'Later revision claims {v51} was director of {v53} in 1999 .'
        v57 = v15.v104([v51])[0]
        v58 = v15.v85(v55, exclude=v52)
        v59 = v15.v85(v56, exclude=v53)
        if v58 is None or v59 is None:
            continue
        v48.v86(v113.v105(v57 + v58, dim=-1))
        v49.v86(v52)
        v50.v86('A')
        v48.v86(v113.v105(v57 + v59, dim=-1))
        v49.v86(v53)
        v50.v86('B')
    v19 = v76.v60(v48, 0)
    v20 = 0
    v21 = 0
    v22 = 0
    v23 = []
    v24 = 0
    for v51, v52, v53 in v54(v16, v17, v18):
        v61 = v15.v85(f'In the report {v51} was linked to the organization.', exclude=None)
        if v61 is None:
            continue
        v62 = (v19 @ v61).v87()
        v63 = v88(v106(v84(v62)), key=lambda v89: v62[v89], reverse=True)
        v64 = v63[:4]
        v65 = [v49[v89] for v89 in v64]
        v66 = v52 in v65[:2]
        v67 = v53 in v65[:2]
        v20 += v6(v66 and v67)
        v68 = [v89 for v89, v114 in v115(v49) if v114 in (v52, v53)]
        v69 = v90((v89 for v89, v114 in v115(v49) if v114 == v52), None)
        v70 = v90((v89 for v89, v114 in v115(v49) if v114 == v53), None)
        if v69 is None or v70 is None:
            continue
        v91, v92 = (v62[v69], v62[v70])
        v23.v86(v107(v91 - v92))
        if v91 >= v92:
            v21 += 1
        else:
            v22 += 1
        v24 += 1
    v25 = v20 / v93(1, v24)
    v26 = v71(v108(v23) / v93(1, v84(v23)))
    v27 = v25 >= 0.4 or v26 < 0.08
    v28 = 'CONTRADICTION_RAW_MEMORY_OK' if v27 else 'CONTRADICTION_COLLAPSE'
    v29 = {'stage': 229, 'overall': v28, 'n_queries': v24, 'rate_both_values_in_top2': v25, 'mean_abs_score_gap_A_vs_B': v26, 'A_wins': v21, 'B_wins': v22, 'interpretation': 'fp nearest-neighbor returns conflicting candidates; resolution is not in the slot layer' if v27 else 'one value dominates — possible key collision / ctx dominates', 'timestamp': v120.v116(v121.v117).v94()}
    v1.v72(v109.v95(v29, indent=2), encoding='utf-8')
    v2.v72(f'# Stage 229 contradiction\n\n**{v28}** both_top2={v25:.3f} gap={v26:.4f}\n', encoding='utf-8')
    v73(v109.v95(v29, indent=2))
    return 0
if v30 == '__main__':
    raise v74(v96())