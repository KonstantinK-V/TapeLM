"""
Stage 219 — slot age decay on stream recall (198-style stress).

  python _stage219_stream_decay.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from _stage191_night import load_data
from _stage194_fp_fact_memory import FpBank
from _stage192_fp_lexicon import gen_fakes
v0 = v5('results')
v1 = v5('checkpoints/stage191_p1_curve.pt')
v2 = v0 / 'stage219_decision.json'
v3 = 219

def main() -> v4:
    v6 = v54.v26()
    v6.v27('--smoke', action='store_true')
    v7 = v6.v28()
    v8 = v55.v8('cuda' if v55.v88.v76() else 'cpu')
    v9 = v56.v29(v3)
    v10 = 40 if v7.v30 else 120
    from tokenizers import Tokenizer
    import _stage177_curve_bpe as s177
    from _stage191_night import SelfModelXL, PAD
    v31, v32, v33, v34 = v35()
    v11 = v57.v36(v58(v77.v59))
    v12 = v78(v34, v11.v89()).v37(v8)
    v12.v38(v55.v79(v1, map_location=v8, weights_only=False)['model'])
    v12.v39()
    v13 = v40(v12, v33, v8)
    v14 = [f'Ent{v44}' for v44 in v80(v10 // 2)]
    v15 = [f'Old{v44}' for v44 in v80(v90(v14))]
    v16 = [f'New{v44}' for v44 in v80(v90(v14))]
    v41, v42, v43 = ([], [], [])
    for v44, (v60, v61, v62) in v45(v63(v14, v15, v16)):
        v46 = v13.v81([v60])[0]
        v41.v64(v46)
        v42.v64(v61)
        v43.v64(100 + v44)
        v41.v64(v46)
        v42.v64(v62)
        v43.v64(1 + v44)
    v17 = v55.v47(v41, 0)
    v18 = v48(v43)
    v19 = v18 / 3.0

    def eval_use_decay(v49: v65) -> v20:
        v66, v67 = (0, 0)
        for v44, v60 in v45(v14):
            v68 = v13.v81([v60])[0]
            v69 = v17 @ v68
            if v49:
                v82 = v55.v91([v100.v95(-v96 / v19) for v96 in v43], device=v8, dtype=v69.v93)
                v69 = v69 * v82
            v70 = {}
            for v83, v84 in v45(v42):
                v70[v84] = v48(v70.v94(v84, -1000000000.0), v20(v69[v83]))
            v71 = v16[v44]
            v72 = [v71, v15[v44], v16[(v44 + 1) % v90(v14)], v15[(v44 + 1) % v90(v14)]]
            v9.v85(v72)
            v73 = v72.v86(v71)
            v66 += v4(v4(v101.v97([v70[v104] for v104 in v72])) == v73)
            v67 += 1
        return v66 / v48(1, v67)
    v21 = v50(False)
    v22 = v50(True)
    v23 = v22 >= v21 + 0.05
    v24 = 'STREAM_DECAY_WIN' if v23 else 'STREAM_DECAY_NO'
    v2.v51(v87.v74({'stage': 219, 'overall': v24, 'gates': {'G1_decay_vs_flat': v23}, 'acc_flat': v21, 'acc_decay': v22, 'timestamp': v102.v98(v103.v99).v92()}, indent=2), encoding='utf-8')
    v52(f'219 {v24} flat={v21:.3f} decay={v22:.3f}')
    return 0
if v25 == '__main__':
    raise v53(v75())