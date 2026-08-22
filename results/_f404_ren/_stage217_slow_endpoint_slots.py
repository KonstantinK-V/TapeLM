"""
Stage 217 — dual keys: ctx_fp (lex) + slow endpoint; noisy recall vs lex-only.

  python _stage217_slow_endpoint_slots.py [--smoke]
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
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, FpBank
from _stage204_noise_robustness import noisy
from _tapelm_ext import slow_endpoint_vec
v0 = v6('results')
v1 = v6('checkpoints/stage191_p1_curve.pt')
v2 = v6('data/_wikitext103_train.txt')
v3 = v0 / 'stage217_decision.json'
v4 = 217

def main() -> v5:
    v7 = v72.v27()
    v7.v28('--smoke', action='store_true')
    v8 = v7.v29()
    v9 = v73.v9('cuda' if v73.v116.v100() else 'cpu')
    v10 = v74.v30(v4)
    v11 = 15 if v8.v31 else 50
    v12 = 0.3
    v32, v33, v34, v35 = v36()
    v13 = v75.v37(v76(v101.v77))
    v14 = v13.v78(v79) or 0
    v15 = v117.v102(v13, v34, v14, v13.v118()).v38(v9)
    v16 = v103(v35, v13.v118()).v38(v9)
    v16.v39(v73.v104(v1, map_location=v9, weights_only=False)['model'])
    v16.v40()
    v17 = v41(v16, v34, v9)
    with v2.v80('r', encoding='utf-8', errors='ignore') as v42:
        v43 = v81(v119.v105((v126.v125(1) for v126 in v135.v132(v42.v82(3000000)) if v109(v126.v125(1)) >= 5)))
    v18 = [v107.v106() for v107 in v42.v82().v120('\n') if v109(v107.v106()) > 200][:200] if False else []
    with v2.v80('r', encoding='utf-8', errors='ignore') as v42:
        v44 = v42.v82(2000000)
    v18 = [v107.v106() for v107 in v44.v120('\n') if v109(v107.v106()) > 200][:300]
    v19 = v83(v108(v43), v10, v11 + 10)[:v11]
    v20 = v43[:v11]
    v45, v46, v47 = ([], [], [])
    for v48, v49 in v50(v19, v20):
        v51 = v18[v10.v121(v109(v18))][:250]
        v52 = f'{v51} {v48} was appointed director of {v49} in 1987 .'
        v84, v85 = (0, v109(v52))
        v53 = [v86 for v86 in v13.v127(v52[v84:v85]).v110 if v86 != v14]
        v54 = v17.v111([v48])[0]
        v55 = v17.v87(v52, exclude=v49)
        if v55 is None:
            continue
        v45.v88(v122.v112(v54 + v55, dim=-1))
        v56 = v89(v16, v15, v14, v53, v9)
        if v56 is None:
            v45.v113()
            continue
        v46.v88(v56)
        v47.v88(v49)
    v45, v46 = (v73.v90(v45, 0), v73.v90(v46, 0))
    v57, v58, v59 = (0, 0, 0)
    v21 = v74.v30(v4 + 3)
    for v48, v49 in v50(v19, v20):
        v52 = f'According to reports {v48} worked closely with {v49} for many years.'
        v60 = v91(v52, v12, v21)
        v61 = v17.v87(v60, exclude=v49)
        v62 = [v86 for v86 in v13.v127(v60).v110 if v86 != v14]
        v63 = v89(v16, v15, v14, v62, v9)
        if v61 is None or v63 is None:
            continue
        v64 = [v49] + [v20[(v86 + 1) % v109(v20)] for v86 in v128(3)]
        v10.v92(v64)
        v65 = v64.v93(v49)
        v66 = []
        v67 = []
        for v68 in v64:
            v94 = [v86 for v86, v129 in v130(v47) if v129 == v68]
            if not v94:
                v66.v88(-1.0)
                v67.v88(-1.0)
                continue
            v95 = v114((v45[v94] @ v61).v97())
            v96 = v114(v97(v114((v45[v94] @ v61).v97()), v114((v46[v94] @ v63).v97())))
            v66.v88(v95)
            v67.v88(v96)
        v57 += v5(v131.v123(v66) == v65)
        v58 += v5(v131.v123(v67) == v65)
        v59 += 1
    v22 = v57 / v97(1, v59)
    v23 = v58 / v97(1, v59)
    v24 = v23 >= v22 + 0.02
    v25 = 'SLOW_ENDPOINT_WIN' if v24 else 'SLOW_ENDPOINT_INVALID_METHOD'
    v3.v69(v115.v98({'stage': 217, 'overall': v25, 'gates': {'G1_noisy_dual': v24}, 'acc_lex': v22, 'acc_dual': v23, 'p_noise': v12, 'n': v59, 'timestamp': v136.v133(v137.v134).v124()}, indent=2), encoding='utf-8')
    v70(f'217 {v25} lex={v22:.3f} dual={v23:.3f} n={v59}')
    return 0
if v26 == '__main__':
    raise v71(v99())