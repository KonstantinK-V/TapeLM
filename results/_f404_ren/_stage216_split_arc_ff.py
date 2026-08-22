"""
Stage 216 — frozen char emb+pool; train FF (linear vs GELU control).

  python _stage216_split_arc_ff.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage213_arc_enc_freeze_finetune as s213
from _stage191_night import LR, MICRO, PAD, SelfModelXL, W_SELF, load_data, lr_at, sample_windows
from _stage194_fp_fact_memory import FpBank
v0 = v8('results')
v1 = v8('checkpoints/stage191_p1_curve.pt')
v2 = v8('data/external_tinystories_100k_85.txt')
v3 = v0 / 'stage216_decision.json'
v4 = ['the', 'cat', 'London', 'Elizabeth', 'running', 'quantum']
v5 = 600

def fp_drift(v9: v43, v10: v44, v11: v87[v46]) -> v6:
    v12 = v9.v45(v11)
    return v6(v88((1 - v6((v10[v80] * v12[v127]).v131()) for v127, v80 in v128(v11))))

def train_ff_only(v13, v14: v46, v15, v16, v17, v18, v19, v20, v21):
    for v22 in v13.v47():
        v22.v48 = False
    v23 = v13.v24
    v23.v89.v49()
    for v22 in v23.v89.v47():
        v22.v48 = False
    if v14 == 'linear':
        v50 = v23.v52[0].v51
        v23.v52 = v129.v120(v129.v130(v50, v50), v129.v130(v50, v50)).v74(v19)
    for v22 in v23.v52.v47():
        v22.v48 = True
    v23.v52.v53()
    v25 = v99.v90.v54(v23.v52.v47(), lr=v110 * 0.3)
    for v26 in v55(1, v20 + 1):
        for v56 in v25.v57:
            v56['lr'] = v111(v26, v20)
        v58 = v121(v15, v16, v122, v21, v18).v74(v19)
        v59 = v58 == v18
        v91, v92, v93 = v13.v94(v17[v58], v59, ids=v58)
        v60 = v58[:, 1:]
        v61 = ~v59[:, :-1] & ~v59[:, 1:]
        v62 = v112.v95(v91[:, :-1][v61], v60[v61])
        v63 = v62 + v113 * v93[~v59].v123()
        v25.v96(set_to_none=True)
        v63.v97()
        v25.v26()
    v23.v49()

def main() -> v7:
    v27 = v98.v64()
    v27.v65('--smoke', action='store_true')
    v28 = v27.v66()
    v19 = v99.v19('cuda' if v99.v124.v114() else 'cpu')
    v21 = v100.v67(216)
    v20 = 80 if v28.v68 else v5
    v15, v16, v69, v70 = v71()
    v29 = v101.v72(v46(v115.v102))
    v30 = v29.v73()
    v18 = v29.v103(v104) or 0
    import _stage185_tape_read as s185
    v17 = v125.v116(v29, v69, v18, v30).v74(v19)
    v31 = v2.v75(encoding='utf-8', errors='ignore')
    v76, v77 = v105.v78(v31, v29, v18, max_lines=300 if v28.v68 else 5000)
    v32 = v117(v70, v30).v74(v19)
    v32.v79(v99.v118(v1, map_location=v19, weights_only=False)['model'])
    v32.v49()
    v10 = {v80: v43(v32, v69, v19).v45([v80])[0].v106() for v80 in v4}
    v33 = v107.v81(v32)
    v82(v33, 'linear', v76, v77, v17, v18, v19, v20, v21)
    v34 = v83(v43(v33, v69, v19), v10, v4)
    v35 = v107.v81(v32)
    v82(v35, 'gelu', v76, v77, v17, v18, v19, v20, v21)
    v36 = v83(v43(v35, v69, v19), v10, v4)
    v37 = 1 - v34
    v38 = 1 - v36
    v39 = v37 > 0.95 and v38 < v37 - 0.05
    v40 = v37 > 0.8
    v41 = 'SPLIT_FF_LINEAR_WINS' if v39 else 'SPLIT_FF_PARTIAL' if v37 > v38 else 'SPLIT_FF_NO'
    v3.v84(v119.v108({'stage': 216, 'overall': v41, 'gates': {'G1_linear_vs_gelu': v39, 'G2_linear_recall_geom': v40}, 'cos_linear_min': v37, 'cos_gelu_min': v38, 'timestamp': v134.v132(v135.v133).v126()}, indent=2), encoding='utf-8')
    v85(f'216 {v41} cos_lin~{v37:.3f} cos_gelu~{v38:.3f}')
    return 0
if v42 == '__main__':
    raise v86(v109())