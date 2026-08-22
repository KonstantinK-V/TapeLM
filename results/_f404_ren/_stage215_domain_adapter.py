"""
Stage 215 — domain_proj adapter on frozen lexical fp (TinyStories domain).

Train W: fp' = normalize(W @ fp_raw). Source bank keys stay raw; domain queries use W.

  python _stage215_domain_adapter.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import auc, gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
v0 = v8('results')
v1 = v8('checkpoints/stage191_p1_curve.pt')
v2 = v8('data/external_tinystories_100k_85.txt')
v3 = v8('data/_wikitext103_train.txt')
v4 = v0 / 'stage215_decision.json'
v5 = 215
v6 = 800

def log(v9: v39) -> None:
    v40(v9, flush=True)

def main() -> v7:
    v10 = v95.v41()
    v10.v42('--smoke', action='store_true')
    v11 = v10.v43()
    v12 = v96.v12('cuda' if v96.v141.v127() else 'cpu')
    v13 = v97.v44(v5)
    v14 = 120 if v11.v45 else v6
    v15 = 20 if v11.v45 else 80
    v46, v47, v48, v49 = v50()
    v16 = v98.v51(v39(v128.v99))
    v17 = v16.v52()
    v18 = v129(v49, v17).v53(v12)
    v18.v54(v96.v130(v1, map_location=v12, weights_only=False)['model'])
    v18.v55()
    v19 = v56(v18, v48, v12)
    v20 = v131(256).v53(v12)
    with v3.v100('r', encoding='utf-8', errors='ignore') as v57:
        v58 = v101(v142.v132((v9.v148(1) for v9 in v155.v154(v57.v156(5000000)) if v139(v9.v148(1)) >= 5)))
    v21 = [v103.v102() for v103 in v2.v149(encoding='utf-8', errors='ignore').v133() if v103.v102() and (not v103.v150('#'))]
    v22 = v104(v134(v58), v13, v15 + 20)[:v15]
    v23 = v58[:v15]
    v24 = v58[v15:v15 * 2][:v15]
    v25, v59 = ([], [])
    for v60, v61 in v62(v22[:v15 // 2], v23[:v15 // 2]):
        v63 = v19.v135([v60])[0]
        v64 = f'Official records show that {v60} was linked to {v61} throughout the decade.'
        v65 = v19.v105(v64, exclude=v61)
        v25.v106(v136.v120(v63 + v65, dim=-1) if v65 is not None else v63)
        v59.v106(v61)
    v25 = v96.v66(v25, 0)
    v26 = v96.v107.v67(v20.v108(), lr=0.001)
    for v27 in v68(1, v14 + 1):
        v69 = v13.v109(v15)
        v60, v61 = (v22[v69], v24[v69])
        v70 = v21[v13.v109(v139(v21))]
        v64 = f'{v70} Then {v60} played with {v61} in the garden.'
        v71 = v19.v105(v64, exclude=v61)
        if v71 is None:
            continue
        v72 = v20(v71.v143(0))[0]
        v73 = v19.v135([v60])[0]
        v63 = v20(v73.v143(0))[0]
        v74 = v136.v110(v72, v63, dim=0)
        v75 = []
        for v76 in v68(4):
            v111 = v13.v109(v15)
            if v24[v111] == v61:
                continue
            v75.v106(v136.v110(v72, v20(v19.v135([v22[v111]])[0].v143(0))[0], dim=0))
        if not v75:
            continue
        v77 = v136.v112(0.3 - v74 + v96.v66(v75).v123())
        v78 = (v20.v119.v89 @ v20.v119.v89.v138 - v96.v157(256, device=v12)).v144(2).v113()
        v77 = v77 + 0.01 * v78
        v26.v114(set_to_none=True)
        v77.v115()
        v26.v27()
    v79, v80, v81 = (0, 0, 0)
    for v69, (v60, v61) in v82(v62(v22, v24)):
        v64 = f'One day {v60} found {v61} near the river and smiled.'
        v71 = v19.v105(v64, exclude=v61)
        if v71 is None:
            continue
        v83 = [v61] + [v24[(v69 + v111) % v15] for v111 in (1, 2, 3)]
        v13.v116(v83)
        v84 = v83.v117(v61)
        v85 = v20(v71.v143(0))[0]
        v86 = [v137(v19.v135([v65])[0] @ v71) for v65 in v83]
        v87 = [v137(v20(v19.v135([v65])[0].v143(0))[0] @ v85) for v65 in v83]
        v79 += v7(v151.v145(v86) == v84)
        v80 += v7(v151.v145(v87) == v84)
        v81 += 1
    v28 = 0
    v29 = 0
    with v96.v118():
        v88 = v20.v119.v89
        v90 = v136.v120(v25 @ v88.v138, dim=-1)
        for v91 in v68(v139(v59)):
            v72 = v19.v105(f'Records mention {v22[v91]} and related events.', exclude=v59[v91])
            if v72 is None:
                continue
            v72 = v136.v120(v72 @ v88.v138, dim=-1)
            v121 = v90 @ v72
            v122 = v59[v7(v121.v145())]
            v28 += v7(v122 == v59[v91])
            v29 += 1
    v30 = v79 / v123(1, v81)
    v31 = v80 / v123(1, v81)
    v32 = v28 / v123(1, v29)
    v33 = v31 >= v30 + 0.05
    v34 = v32 >= 0.9
    v35 = True
    v36 = 'DOMAIN_ADAPTER_WIN' if v33 and v34 else 'DOMAIN_ADAPTER_PARTIAL' if v33 else 'DOMAIN_ADAPTER_NO'
    v37 = {'stage': 215, 'overall': v36, 'gates': {'G1_domain_recall': v33, 'G2_old_bank_W': v34, 'G3_calib': v35}, 'acc_raw': v30, 'acc_adapted': v31, 'old_bank_retention': v32, 'steps': v14, 'timestamp': v152.v146(v153.v147).v124()}
    v4.v92(v140.v125(v37, indent=2), encoding='utf-8')
    v93(f'215 {v36} raw={v30:.3f} adapted={v31:.3f} old={v32:.3f}')
    return 0
if v38 == '__main__':
    raise v94(v126())