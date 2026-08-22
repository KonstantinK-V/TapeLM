"""
Stage 218 — explicit lexicon snap on latent hops (subset of 206 protocol).

  python _stage218_snap_hop.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage204_noise_robustness import noisy
v0 = v6('results')
v1 = v6('checkpoints/stage191_p1_curve.pt')
v2 = v6('data/_wikitext103_train.txt')
v3 = v0 / 'stage218_decision.json'
v4 = 218

def main() -> v5:
    v7 = v59.v31()
    v7.v32('--smoke', action='store_true')
    v8 = v7.v33()
    v9 = v60.v9('cuda' if v60.v94.v82() else 'cpu')
    v10 = v61.v34(v4)
    v11 = 20 if v8.v35 else 60
    v12 = 4
    v13 = 0.15
    v36, v37, v38, v39 = v40()
    v14 = v62.v41(v63(v83.v64))
    v15 = v84(v39, v14.v95()).v42(v9)
    v15.v43(v60.v85(v1, map_location=v9, weights_only=False)['model'])
    v15.v44()
    v16 = v45(v15, v38, v9)
    with v2.v65('r', encoding='utf-8', errors='ignore') as v46:
        v47 = v48(v87.v67((v100.v99(1) for v100 in v18.v101(v46.v106(2000000)) if v92(v100.v99(1)) >= 5)))
    v17 = v18
    v19 = v66(v86(v47), v10, v11 * 5)[:v11 * 4]
    v20 = [v19[v52:v52 + 4] for v52 in v68(0, v92(v19) - 3, 4)][:v11]
    v21 = v48(v87.v67((v88 for v24 in v20 for v88 in v24)))
    v22 = v16.v49(v21)
    v23 = v61.v34(v4 + 1)
    v50, v51 = ([], [])
    for v24 in v20:
        for v52 in v68(v92(v24) - 1):
            v50.v89(v16.v49([v102(v24[v52], v13, v23)])[0])
            v51.v89(v16.v49([v102(v24[v52 + 1], v13, v23)])[0])
    v50, v51 = (v60.v69(v50, 0), v60.v69(v51, 0))

    def run(v53: v70) -> v25:
        v54 = 0
        for v24 in v20:
            v71 = v16.v49([v102(v24[0], v13, v23)])[0]
            for v72 in v68(v12):
                v71 = v51[v5((v50 @ v71).v103())]
                if v53:
                    v71 = v22[v5((v22 @ v71).v103())]
            v73 = v22 @ v71
            v74 = v24[v96(v12, v92(v24) - 1)]
            v75 = [v97 for v97 in v21 if v97 != v74][:3]
            v76 = [v74] + v75
            v77 = v48(v68(4))
            v10.v90(v77)
            v76 = [v76[v52] for v52 in v77]
            v78 = v76.v91(v74)
            v79 = [v25(v73[v21.v91(v97)]) for v97 in v76]
            v54 += v5(v5(v107.v103(v79)) == v78)
        return v54 / v92(v20)
    v26 = v55(False)
    v27 = v55(True)
    v28 = v27 >= v26 + 0.02
    v29 = 'SNAP_HOP_WIN' if v28 else 'SNAP_HOP_INVALID_METHOD'
    v3.v56(v93.v80({'stage': 218, 'overall': v29, 'gates': {'G1_snap_noisy': v28}, 'acc_no_snap': v26, 'acc_snap': v27, 'k': v12, 'p_noise': v13, 'timestamp': v108.v104(v109.v105).v98()}, indent=2), encoding='utf-8')
    v57(f'218 {v29} raw={v26:.3f} snap={v27:.3f}')
    return 0
if v30 == '__main__':
    raise v58(v81())