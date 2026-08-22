"""
Stage 243 — Same domain-B corpus, different memory carrier.

A facts acquired in TapeLM slots and GPT weights; adapt on identical code corpus:
  TapeLM — query arc_enc shift + W (slots untouched)
  GPT    — CE overwrite (parametric carrier)

Reports which carrier retains A. Related to 239; framed as carrier contrast.

  python _stage243_carrier_drift.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import time
from datetime import datetime, timezone
import torch
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage194_fp_fact_memory import FpBank
from _stage196_tapelm import load_gpt
from _tapelm_ext import DomainAdapter
v0 = 243
v1 = v42.v5 / 'stage243_decision.json'
v2 = v42.v5 / 'stage243_mini.md'
v3 = v42.v5 / '_stage243_log.txt'

def main() -> v4:
    v6 = v99.v43()
    v6.v44('--smoke', action='store_true')
    v7 = v6.v45()
    v3.v46('', encoding='utf-8')
    v8 = v42.v47(v3)
    v9 = v100.v9('cuda' if v100.v113.v110() else 'cpu')
    v10 = v111('random').v48(v0)
    v100.v49(v0)
    v11 = v50.v50()
    v12 = 12 if v7.v51 else 40
    v13 = 240 if v7.v51 else 2400
    v14 = 400 if v7.v51 else 1600
    v15 = 60 if v7.v51 else v101.v52
    v16 = 80 if v7.v51 else v101.v53
    v17 = 60 if v7.v51 else 400
    v18 = 40 if v7.v51 else 120
    v54, v55, v56, v57 = (8, 64, 0.0003, 0.0005)
    v19 = 0.72
    v8(f'Stage243 start {v118.v116(v119.v117).v108()}')
    v58, v58, v59, v58, v60, v58, v61, v62, v63, v64 = v42.v65(v9)
    v58, v66, v67, v68 = v42.v69(v7.v51, v17, v10)
    v70, v71 = v42.v72(v12, v66, v10)
    v20 = v42.v73(v18)
    v74, v75 = v42.v76(v64, v70)
    v21 = v42.v77(v70, v71, v64, v74, v75, v0)
    v22 = v42.v78(v63, v62, v61, v20, v9)
    v23 = v102.v79(v103(v9))
    v80, v58, v58 = v42.v81(v23, v60, v61, v70, v71, v68, v9, v0, v13, v54, v55, v56, v19, 40 if v7.v51 else 100, v8)
    v24 = v42.v82(v23, v60, v61, v70, v71, v9, v0)
    v25 = v104.v83(v111('random').v48(v0 + 1), v7.v51)
    v84, v85 = v105.v86(v25, v60, v61, max_lines=300 if v7.v51 else 8000, min_line_len=20)
    v26 = [v87 for v87 in v60.v114(v25[:200000]).v106 if v87 != v61]
    v27 = v101.v88(v64, v67)
    v28 = v101.v89(v63, v84, v85, v62, v61, v9, v15, v0 + 7)
    v29 = v90(v28, v59, v9)
    v91, v92 = v101.v93(v115(256).v107(v9), v101.v88(v29, v67), v27, v10, v16, v9)
    v30 = v42.v77(v70, v71, v29, v74, v75, v0, W_bwd=v91)
    v31 = v42.v78(v63, v62, v61, v20, v9)
    v42.v94(v23, v26, v54, v55, v57, v14, v9, v0, v8)
    v32 = v42.v82(v23, v60, v61, v70, v71, v9, v0)
    v33 = v42.v95(v23, v20, v9)
    v8(f'carriers after same code-B: slots+W={v30:.3f} weights={v32:.3f} ({v50.v50() - v11:.0f}s)')
    v34 = v30 - v32
    v35 = v21 >= 0.7 and v24 >= 0.7
    v36 = v30 >= 0.8
    v37 = v24 - v32 >= 0.15
    v38 = v34 >= 0.2
    v39 = v112(v31 - v22) < 1e-09 or v31 >= v22 - 0.02
    if v35 and v36 and v37 and v38:
        v96 = 'CARRIER_DRIFT_OK'
    elif v35 and v36 and (v37 or v38 >= 0.1):
        v96 = 'CARRIER_DRIFT_PARTIAL'
    else:
        v96 = 'CARRIER_DRIFT_NO'
    v40 = {'stage': 243, 'overall': v96, 'gates': {'G_memorize': v35, 'G_slots_retain_ge_0p80': v36, 'G_weights_drop_ge_0p15': v37, 'G_gap_ge_0p20': v38, 'G_frozen_gen_stable': v39}, 'slots': {'A0': v21, 'A1': v30, 'next_tok_0': v22, 'next_tok_1': v31}, 'weights': {'A0': v24, 'A1': v32, 'drop': v24 - v32, 'next_tok_1': v33}, 'gap_slots_minus_weights': v34, 'W_align': v92, 'memorize_steps': v80, 'note': 'Same B corpus; carrier = slots+W vs parametric weights.', 'timestamp': v118.v116(v119.v117).v108()}
    v42.v97(v1, v2, v40, 'Stage 243 carrier drift')
    v8(v96)
    return 0
if v41 == '__main__':
    raise v98(v109())