"""Audit stage-255 recall metrics on a saved tape (CPU)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import _stage24x_lib as L
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from tokenizers import Tokenizer
v0 = 255 + 9000

def main() -> v1:
    v2 = v43.v18()
    v2.v19('--run', type=v44, default='results/stream255')
    v2.v19('--seeds', type=v1, default=200)
    v3 = v2.v20()
    v4 = v21(v3.v4)
    v5 = v45.v22((v4 / 'state.json').v46(encoding='utf-8'))
    v6 = [v23 for v47 in v5['probe_facts'].v25() for v23 in v47]
    v7 = v48.v24(v4 / 'tape.pt', map_location='cpu', weights_only=False)
    v25, v26 = (v7['values'], v7['K'])
    v8 = v48.v8('cpu')
    v27, v27, v28, v29 = v30()
    v9 = v49.v31(v44(v57.v50))
    v10 = v9.v51(v52) or 0
    v11 = v58(v29, v9.v60()).v32(v8)
    v11.v33(v48.v24('checkpoints/stage191_p1_curve.pt', map_location=v8, weights_only=False)['model'])
    v11.v34()
    v12 = v35(v11, v28, v8)
    v13 = v36(v59.v53([v23['value'] for v23 in v6] + v25))
    v14 = v54.v37(v6, v13, v12, v26, v25, v0)
    v38(f'tape slots={v61(v25)} probe_facts={v61(v6)}')
    v38(f'fixed-seed metrics: {v14}')
    import random
    v15 = []
    for v16 in v39(v3.v40):
        v41 = v54.v37(v6, v13, v12, v26, v25, 1000 + v16)
        v15.v55(v41['four_way'])
    v38(f'4-way over {v3.v40} distractor seeds: mean={v66.v62(v15):.3f} sd={v66.v63(v15):.3f} min={v64(v15):.3f} max={v65(v15):.3f}')
    return 0
if v17 == '__main__':
    raise v42(v56())