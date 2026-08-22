"""
Stage 232 — L3 stream: age decay + W_version mismatch penalty on canonical slots.

When age ties, wrong `w_version` must lose to correct era. Decay-only cannot
disambiguate; version penalty can.

  python _stage232_stream_w_version.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from _stage191_night import load_data
from _stage194_fp_fact_memory import FpBank
from _tapelm_ext import weighted_slot_sims
v0 = v6('results')
v1 = v0 / 'stage232_decision.json'
v2 = v0 / 'stage232_mini.md'
v3 = v6('checkpoints/stage191_p1_curve.pt')
v4 = 232

def main() -> v5:
    v7 = v59.v30()
    v7.v31('--smoke', action='store_true')
    v8 = v7.v32()
    v9 = v60.v9('cuda' if v60.v89.v77() else 'cpu')
    v10 = v61.v33(v4)
    v11 = 24 if v8.v34 else 100
    from tokenizers import Tokenizer
    import _stage177_curve_bpe as s177
    from _stage191_night import SelfModelXL
    v35, v36, v37, v38 = v39()
    v12 = v62.v40(v63(v78.v64))
    v13 = v79(v38, v12.v90()).v41(v9)
    v13.v42(v60.v80(v3, map_location=v9, weights_only=False)['model'])
    v13.v43()
    v14 = v44(v13, v37, v9)
    v15 = [f'Ent{v49}' for v49 in v81(v11)]
    v16 = [f'StaleWrong{v49}' for v49 in v81(v11)]
    v17 = [f'FreshRight{v49}' for v49 in v81(v11)]
    v45, v46, v47, v48 = ([], [], [], [])
    v18 = 'prose_v2'
    v19 = 50.0
    for v49, v50 in v51(v15):
        v52 = v14.v82([v50])[0]
        v45.v65(v52)
        v46.v65(v16[v49])
        v47.v65(40)
        v48.v65('prose_v1')
        v45.v65(v52)
        v46.v65(v17[v49])
        v47.v65(41)
        v48.v65(v18)
        v45.v65(v52)
        v46.v65(f'OldRight{v49}')
        v47.v65(500)
        v48.v65(v18)
    v20 = v60.v53(v45, 0)

    def eval_mode(v54: v63) -> v21:
        v66, v67 = (0, 0)
        for v49, v50 in v51(v15):
            v68 = v14.v82([v50])[0]
            v69 = v20 @ v68
            if v54 == 'flat':
                pass
            elif v54 == 'decay':
                v69 = v93(v69, v47, v48, v18, v19, version_penalty=1.0)
            elif v54 == 'decay_version':
                v69 = v93(v69, v47, v48, v18, v19, version_penalty=0.05)
            v70 = {}
            for v83, v84 in v51(v46):
                v70[v84] = v87(v70.v94(v84, -1000000000.0), v21(v69[v83]))
            v71 = v17[v49]
            v72 = [v71, v16[v49], v17[(v49 + 1) % v11], v16[(v49 + 1) % v11]]
            v10.v85(v72)
            v73 = v72.v86(v71)
            v66 += v5(v5(v98.v97([v70[v99] for v99 in v72])) == v73)
            v67 += 1
        return v66 / v87(1, v67)
    v22 = v55('flat')
    v23 = v55('decay')
    v24 = v55('decay_version')
    v25 = v24 >= v22 + 0.15
    v26 = v24 >= v23 + 0.1
    v27 = 'STREAM_W_VERSION_OK' if v25 and v26 else 'STREAM_W_VERSION_PARTIAL' if v25 else 'STREAM_W_VERSION_NO'
    v28 = {'stage': 232, 'overall': v27, 'gates': {'G_decay_version_beats_flat': v25, 'G_decay_version_beats_decay_only': v26}, 'acc_flat': v22, 'acc_decay_only': v23, 'acc_decay_plus_w_version': v24, 'active_w_version': v18, 'timestamp': v95.v91(v96.v92).v74()}
    v1.v56(v88.v75(v28, indent=2), encoding='utf-8')
    v2.v56(f'# Stage 232 stream + W version\n\n**{v27}** dv={v24:.3f} decay={v23:.3f} flat={v22:.3f}\n', encoding='utf-8')
    v57(v88.v75(v28, indent=2))
    return 0
if v29 == '__main__':
    raise v58(v76())