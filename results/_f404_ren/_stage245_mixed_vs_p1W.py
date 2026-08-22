"""
Stage 245 — Mixed-scratch encoder without W vs product P1 + W after code shift.

Uses checkpoints/stage238_mixed_scratch.pt and stage191_p1_curve.pt.
Same fact strings; each encoder writes its own bank, then code-shifts.
Compare mixed recall(no W) vs P1 recall(W).

  python _stage245_mixed_vs_p1W.py [--smoke]
"""
from __future__ import annotations
import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
import torch
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage194_fp_fact_memory import FpBank
from _tapelm_ext import DomainAdapter
v0 = 245
v1 = v44.v5 / 'stage245_decision.json'
v2 = v44.v5 / 'stage245_mini.md'
v3 = v44.v5 / '_stage245_log.txt'

def arm(v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20: v45):
    v21 = v46(v7, v8, v15)
    v47, v48 = v44.v49(v21, v10)
    v22 = v44.v50(v10, v11, v21, v47, v48, v0)
    v23 = v83.v51(v89.v65(v0 + 1), v16)
    v52, v53 = v84.v54(v23, v12, v13, max_lines=300 if v16 else 8000, min_line_len=20)
    v24 = v85.v55(v21, v9)
    v25 = v85.v56(v7, v52, v53, v14, v13, v15, v17, v0 + 7)
    v26 = v46(v25, v8, v15)
    v57, v58 = v85.v59(v94(256).v86(v15), v85.v55(v26, v9), v24, v19, v18, v15)
    v27 = v44.v50(v10, v11, v26, v47, v48, v0, W_bwd=None)
    v28 = v44.v50(v10, v11, v26, v47, v48, v0, W_bwd=v57)
    v29 = v28 if v20 else v27
    return {'name': v6, 'A0': v22, 'A_raw': v27, 'A_W': v28, 'chosen': v29, 'use_W': v20, 'W_align': v58}

def main() -> v4:
    v30 = v87.v60()
    v30.v61('--smoke', action='store_true')
    v31 = v30.v62()
    v3.v63('', encoding='utf-8')
    v32 = v44.v64(v3)
    v15 = v88.v15('cuda' if v88.v95.v93() else 'cpu')
    v19 = v89.v65(v0)
    if not v44.v75.v90():
        raise v82(f'missing {v44.v75}')
    v17 = 50 if v31.v16 else v85.v66
    v18 = 60 if v31.v16 else v85.v67
    v33 = 60 if v31.v16 else 400
    v34 = 12 if v31.v16 else 50
    v32(f'Stage245 start {v99.v97(v100.v98).v91()}')
    v68, v68, v8, v69, v12, v70, v13, v14, v71, v68 = v44.v72(v15)
    v73, v68 = v44.v74(v44.v75, v69, v70, v8, v15)
    v68, v76, v9, v68 = v44.v77(v31.v16, v33, v19)
    v10, v11 = v44.v78(v34, v76, v19)
    v35 = v79('P1+W', v71, v8, v9, v10, v11, v12, v13, v14, v15, v31.v16, v17, v18, v19, True)
    v36 = v79('mixed_no_W', v73, v8, v9, v10, v11, v12, v13, v14, v15, v31.v16, v17, v18, v19, False)
    v32(f"P1+W chosen={v35['chosen']:.3f} (raw={v35['A_raw']:.3f} W={v35['A_W']:.3f})")
    v32(f"mixed noW chosen={v36['chosen']:.3f} (raw={v36['A_raw']:.3f} W={v36['A_W']:.3f})")
    v37 = v36['chosen'] - v35['chosen']
    v38 = v35['chosen'] >= 0.7
    v39 = v36['chosen'] >= 0.7
    v40 = v37 >= 0.05
    v41 = v36['A_raw'] + 0.02 >= v36['A_W']
    if v38 and v39 and v40:
        v80 = 'MIXED_NO_W_BEATS_P1W'
    elif v38 and v39 and (v96(v37) < 0.05):
        v80 = 'MIXED_NO_W_TIES_P1W'
    elif v38 and (not v40):
        v80 = 'P1W_BEATS_MIXED_NO_W'
    else:
        v80 = 'MIXED_VS_P1W_NO'
    v42 = {'stage': 245, 'overall': v80, 'gates': {'G_p1W_floor_0p70': v38, 'G_mixed_floor_0p70': v39, 'G_mixed_beats_p1W_0p05': v40, 'G_mixed_raw_ge_own_W': v41}, 'p1': v35, 'mixed': v36, 'gap_mixed_minus_p1': v37, 'note': 'Unexpected if mixed-no-W >= P1+W on same fact strings after code shift.', 'timestamp': v99.v97(v100.v98).v91()}
    v44.v81(v1, v2, v42, 'Stage 245 mixed no-W vs P1+W')
    v32(v80)
    return 0
if v43 == '__main__':
    raise v82(v92())