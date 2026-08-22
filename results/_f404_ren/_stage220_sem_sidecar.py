"""
Stage 220 — semantic sidecar vs lexical fp on PAWS (frozen P1).

  python _stage220_sem_sidecar.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from datasets import load_dataset
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage202_semantic_paws import SemHead, lexical_overlap
v0 = v5('results')
v1 = v5('checkpoints/stage191_p1_curve.pt')
v2 = v0 / 'stage220_decision.json'
v3 = 220

def main() -> v4:
    v6 = v62.v27()
    v6.v28('--smoke', action='store_true')
    v7 = v6.v29()
    v8 = v63.v8('cuda' if v63.v107.v84() else 'cpu')
    v63.v30(v3)
    v9 = 400 if v7.v31 else 2000
    v10 = 200 if v7.v31 else 800
    v32, v33, v34, v35 = v36()
    v11 = v64.v37(v65(v85.v66))
    v12 = v11.v67(v68) or 0
    v13 = v108.v86(v11, v34, v12, v11.v109()).v38(v8)
    v14 = v87(v35, v11.v109()).v38(v8)
    v14.v39(v63.v88(v1, map_location=v8, weights_only=False)['model'])
    v14.v40()
    for v15 in v14.v41():
        v15.v42 = False
    v16 = v89(256).v38(v8)
    v17 = v63.v69.v43(v16.v41(), lr=0.0005)
    v18 = v44('paws', 'labeled_final', split='train[:8000]' if v7.v31 else 'train[:25000]')
    v19 = [(v70['sentence1'], v70['sentence2'], v4(v70['label'])) for v70 in v18]
    v20 = v90.v71.v45(v3)
    v20.v46(v19)
    v47, v48 = (v19[:v9], v19[v9:v9 + v10])

    def encode_sent(v49):
        v50 = [v91 for v91 in v11.v114(v49).v50 if v91 != v12][-v110:]
        v51 = v63.v72([v50], device=v8)
        v52 = v51 == v12
        v53 = v14.v73(v13[v51], ids=v51)
        v54 = v14.v54(v53, pad_mask=v52)
        return (v54, ~v52)
    for v21 in v55(2 if v7.v31 else 3):
        v20.v46(v47)
        for v74, v75, v76 in v47[:200] if v7.v31 else v47:
            v92, v93 = v94(v74)
            v95, v96 = v94(v75)
            v97, v98 = (v16(v92, v93), v16(v95, v96))
            v77 = v63.v72([v76], device=v8, dtype=v63.v111)
            v78 = v112.v99(v97, v98)
            v79 = v112.v100((v78 + 1) / 2, v77)
            v17.v101(set_to_none=True)
            v79.v102()
            v17.v103()

    def acc_pairs(v56, v57=True):
        v80, v81 = (0, 0)
        for v74, v75, v76 in v56:
            if v57:
                v92, v93 = v94(v74)
                v95, v96 = v94(v75)
                v97, v98 = (v16(v92, v93), v16(v95, v96))
                v104 = v4(v112.v99(v97, v98).v115() > 0.5)
            else:
                v104 = v4(v116(v74, v75) > 0.5)
            v80 += v4(v104 == v76)
            v81 += 1
        return v80 / v105(1, v81)
    v22 = v58(v48, True)
    v23 = v58(v48, False)
    v24 = v22 > v23 + 0.03
    v25 = 'SEM_SIDECAR_WIN' if v24 else 'SEM_SIDECAR_INVALID_METHOD'
    v2.v59(v106.v82({'stage': 220, 'overall': v25, 'gates': {'G1_sem_vs_lexical_baseline': v24}, 'paws_acc_sem': v22, 'paws_acc_lexical_proxy': v23, 'timestamp': v119.v117(v120.v118).v113()}, indent=2), encoding='utf-8')
    v60(f'220 {v25} sem={v22:.3f} lex={v23:.3f}')
    return 0
if v26 == '__main__':
    raise v61(v83())