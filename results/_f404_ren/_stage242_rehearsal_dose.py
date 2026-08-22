"""
Stage 242 — GPT rehearsal dose during domain-B: how much A mix to match TapeLM retain.

After shared A acquire + TapeLM code+W retain, sweep GPT code CE with rehearsal in
{0, 0.05, 0.15, 0.30, 0.50}. Report minimal dose where GPT A >= tape_A - 0.05.

  python _stage242_rehearsal_dose.py [--smoke]
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
v0 = 242
v1 = v39.v6 / 'stage242_decision.json'
v2 = v39.v6 / 'stage242_mini.md'
v3 = v39.v6 / '_stage242_log.txt'
v4 = [0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0]

def main() -> v5:
    v7 = v98.v40()
    v7.v41('--smoke', action='store_true')
    v7.v41('--rates', type=v99, default='', help='comma list, overrides the grid')
    v8 = v7.v42()
    v3.v43('', encoding='utf-8')
    v9 = v39.v44(v3)
    v10 = v100.v10('cuda' if v100.v115.v111() else 'cpu')
    v11 = v112('random').v45(v0)
    v100.v46(v0)
    v12 = v47.v47()
    v13 = 10 if v8.v48 else 32
    v14 = 200 if v8.v48 else 2000
    v15 = 200 if v8.v48 else 800
    v16 = 50 if v8.v48 else v101.v49
    v17 = 60 if v8.v48 else v101.v50
    v18 = 60 if v8.v48 else 400
    v51, v52, v53, v54 = (8, 64, 0.0003, 0.0005)
    v19 = 0.72
    v20 = [v113(v114) for v114 in v8.v20.v116(',') if v114.v117()] if v8.v20 else [0.0, 0.15, 0.5, 1.0] if v8.v48 else v4
    v9(f'Stage242 start {v122.v120(v123.v121).v109()}')
    v55, v55, v56, v55, v57, v55, v58, v59, v60, v61 = v39.v62(v10)
    v55, v63, v64, v65 = v39.v66(v8.v48, v18, v11)
    v67, v68 = v39.v69(v13, v63, v11)
    v70, v71 = v39.v72(v61, v67)
    v21 = v102.v73(v103(v10))
    v74, v75, v55 = v39.v76(v21, v57, v58, v67, v68, v65, v10, v0, v14, v51, v52, v53, v19, 40 if v8.v48 else 100, v9)
    v22 = v39.v77(v21, v57, v58, v67, v68, v10, v0)
    v23 = v104.v78(v112('random').v45(v0 + 1), v8.v48)
    v79, v80 = v105.v81(v23, v57, v58, max_lines=300 if v8.v48 else 8000, min_line_len=20)
    v24 = [v82 for v82 in v57.v118(v23[:200000]).v106 if v82 != v58]
    v25 = v101.v83(v61, v64)
    v26 = v101.v84(v60, v79, v80, v59, v58, v10, v16, v0 + 7)
    v27 = v85(v26, v56, v10)
    v86, v87 = v101.v88(v119(256).v107(v10), v101.v83(v27, v64), v25, v11, v17, v10)
    v28 = v39.v89(v67, v68, v27, v70, v71, v0, W_bwd=v86)
    v29 = v28 - 0.05
    v9(f'tape retain={v28:.3f} gpt0={v22:.3f} target_gpt>={v29:.3f}')
    v90, v91 = ({}, {})
    v30 = None
    for v31 in v20:
        v92 = v47.v47()
        v93 = v102.v73(v21)
        v39.v108(v93, v24, v51, v52, v54, v15, v10, v0 + v5(v31 * 100), v9, tag=f'reh{v31}', fact_ids=v75, rehearsal=v31)
        v94 = v39.v77(v93, v57, v58, v67, v68, v10, v0)
        v90[v99(v31)] = v94
        v91[v99(v31)] = v47.v47() - v92
        v9(f'  rehearsal={v31:.2f} -> A={v94:.3f} ({v91[v99(v31)]:.0f}s)')
        if v30 is None and v94 >= v29:
            v30 = v31
    v32 = v28 >= 0.8
    v33 = v90[v99(v20[0])] < v29
    v34 = v30 is not None
    v35 = '1.0' in v90
    v36 = v32 and v33 and v35 and (v113(v90['1.0']) < v29)
    if v36:
        v95 = 'REHEARSAL_DOSE_ANTICF_OK'
    elif v32 and v33 and v34:
        v95 = 'REHEARSAL_DOSE_OK'
    elif v32 and (v33 or v34):
        v95 = 'REHEARSAL_DOSE_PARTIAL'
    else:
        v95 = 'REHEARSAL_DOSE_NO'
    v37 = {'stage': 242, 'overall': v95, 'gates': {'G_tape_retain_ge_0p80': v32, 'G_zero_rehearsal_below_target': v33, 'G_found_dose': v34, 'G_anticf_price_at_full_replay': v36}, 'tape_A_after_B': v28, 'gpt_A0': v22, 'target_gpt': v29, 'min_rehearsal_to_match': v30, 'curve': v90, 'dose_wall_s': v91, 'tape_write_cost': 'one slot write, zero gradient steps (259: ~1e-4 s, params bit-identical)', 'W_align': v87, 'memorize_steps': v74, 'note': 'Price of anti-CF in weights = fraction of A tokens mixed into B CE. At rehearsal 1.0, GPT A retain is below target (see curve) while tape stays 1.0 — overall REHEARSAL_DOSE_ANTICF_OK when the grid includes 1.0. G_found_dose (≥0.95) is optional strictness, not the headline. dose_wall_s vs slot write (259).', 'timestamp': v122.v120(v123.v121).v109(), 'wall_s': v47.v47() - v12}
    v39.v96(v1, v2, v37, 'Stage 242 rehearsal dose')
    v9(v95)
    return 0
if v38 == '__main__':
    raise v97(v110())