"""
Stage 241 — Harmful W vs no-W: does wrong-family qmap beat raw (no W)?

Canonical bank; code query encoder. Compare matched W_code, wrong W_stories, no W.

  python _stage241_harmful_W.py [--smoke]
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
v0 = 241
v1 = v30.v5 / 'stage241_decision.json'
v2 = v30.v5 / 'stage241_mini.md'
v3 = v30.v5 / '_stage241_log.txt'

def main() -> v4:
    v6 = v75.v31()
    v6.v32('--smoke', action='store_true')
    v7 = v6.v33()
    v3.v34('', encoding='utf-8')
    v8 = v30.v35(v3)
    v9 = v76.v9('cuda' if v76.v88.v86() else 'cpu')
    v10 = v77.v36(v0)
    v11 = 60 if v7.v37 else v78.v38
    v12 = 80 if v7.v37 else v78.v39
    v13 = 80 if v7.v37 else 400
    v14 = 12 if v7.v37 else 60
    v15 = 300 if v7.v37 else 8000
    v8(f'Stage241 start {v92.v90(v93.v91).v84()}')
    v40, v40, v41, v40, v42, v40, v43, v44, v45, v46 = v30.v47(v9)
    v40, v48, v49, v40 = v30.v50(v7.v37, v13, v10)
    v51, v52 = v30.v53(v14, v48, v10)
    v54, v55 = v30.v56(v46, v51)
    v16 = v78.v57(v46, v49)
    v58, v59 = v79.v60(v30.v87.v80(encoding='utf-8', errors='ignore'), v42, v43, max_lines=v15)
    v17 = v81.v61(v77.v36(v0 + 1), v7.v37)
    v62, v63 = v79.v60(v17, v42, v43, max_lines=v15, min_line_len=20)
    v18 = v78.v64(v45, v58, v59, v44, v43, v9, v11, v0 + 2)
    v19 = v78.v64(v45, v62, v63, v44, v43, v9, v11, v0 + 3)
    v65, v66 = (v82(v18, v41, v9), v82(v19, v41, v9))
    v67, v40 = v78.v68(v89(256).v83(v9), v78.v57(v65, v49), v16, v10, v12, v9)
    v69, v70 = v78.v68(v89(256).v83(v9), v78.v57(v66, v49), v16, v10, v12, v9)
    v20 = v30.v71(v51, v52, v66, v54, v55, v0, W_bwd=None)
    v21 = v30.v71(v51, v52, v66, v54, v55, v0, W_bwd=v67)
    v22 = v30.v71(v51, v52, v66, v54, v55, v0, W_bwd=v69)
    v8(f'code query: none={v20:.3f} wrong_stories_W={v21:.3f} matched={v22:.3f}')
    v23 = v20 - v21
    v24 = v22 - v20
    v25 = v23 >= 0.05
    v26 = v24 >= 0.05
    v27 = v22 >= v21 + 0.1
    if v25 and v26 and v27:
        v72 = 'WRONG_W_HURTS_OK'
    elif v25 or (v21 < v20 and v27):
        v72 = 'WRONG_W_HURTS_PARTIAL'
    else:
        v72 = 'WRONG_W_HURTS_NO'
    v28 = {'stage': 241, 'overall': v72, 'gates': {'G_wrong_worse_than_none_by_0p05': v25, 'G_matched_helps_vs_none_0p05': v26, 'G_matched_beats_wrong_0p10': v27}, 'recall': {'no_W': v20, 'wrong_W_stories': v21, 'matched_W_code': v22}, 'deltas': {'none_minus_wrong': v23, 'matched_minus_none': v24}, 'W_align_code': v70, 'note': 'Deploy guard: prefer no-W over wrong-family W when hurt>0.', 'timestamp': v92.v90(v93.v91).v84()}
    v30.v73(v1, v2, v28, 'Stage 241 harmful W vs no-W')
    v8(v72)
    return 0
if v29 == '__main__':
    raise v74(v85())