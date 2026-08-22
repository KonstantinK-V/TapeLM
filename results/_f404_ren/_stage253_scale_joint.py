"""
Stage 253 — Scale joint CE + 0.2*CPC (252 fork SCALE_JOINT_TOKENS).

Single arm lambda=0.2, 16M CE tokens from P1, no early stop (full budget burn).
252 @4M reference: nt=0.850, hold_ce=4.199, gap=+0.137.

  python _stage253_scale_joint.py [--smoke] [--token-budget N]
"""
from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
v0 = v10('results')
v1 = v0 / 'stage253_decision.json'
v2 = v0 / 'stage253_mini.md'
v3 = v0 / '_stage253_log.txt'
v4 = v10('checkpoints/stage191_p1_curve.pt')
v5 = v10('checkpoints/stage253_joint_l02.pt')
v6 = 253
v7 = 0.2
v8 = {'next_tok': 0.85, 'hold_ce': 4.199211339155833, 'gap': 0.1365489284396171}

def log(v11: v43) -> None:
    v12 = v11 if v11.v85('\n') else v11 + '\n'
    try:
        v86(v12, end='', flush=True)
    except v44:
        v86(v12.v119('ascii', 'replace').v111('ascii'), end='', flush=True)
    v3.v87.v45(parents=True, exist_ok=True)
    with v3.v88('a', encoding='utf-8') as v46:
        v46.v89(v12)

def main() -> v9:
    v13 = v90.v47()
    v13.v48('--smoke', action='store_true')
    v13.v48('--token-budget', type=v9, default=0)
    v14 = v13.v49()
    v3.v50('', encoding='utf-8')
    v15 = v91.v15('cuda' if v91.v112.v105() else 'cpu')
    v16 = v92.v51(v6)
    v91.v52(v6)
    v17 = v53.v53()
    v18 = v14.v54 or (200000 if v14.v55 else 16000000)
    v19 = 8 if v14.v55 else 20
    v20 = 40 if v14.v55 else 120
    v21 = 24 if v14.v55 else 60
    v22 = 8 if v14.v55 else 32
    v23 = 8 if v14.v55 else 16
    v56(f'Stage253 scale joint CE+{v7}*CPC start {v117.v114(v118.v115).v101()} budget={v18}')
    v57, v58, v59, v60 = v61()
    v62, v63 = v93.v64(v58)
    v24 = v94.v65(v43(v106.v95))
    v25 = v24.v66()
    v26 = v24.v96(v97) or 0
    v27 = v113.v107(v24, v59, v26, v25).v67(v15)
    v28 = v108(v60, v25).v67(v15)
    v28.v68(v91.v109(v4, map_location=v15, weights_only=False)['model'])
    v28.v69()
    for v29 in v28.v70():
        v29.v98(False)
    v71, v72 = v93.v73(v16, v19, v14.v55)
    v30 = v74(v28, v59, v15)
    v75, v76 = v99.v77(v30, v71)
    v31 = v93.v78(v20)
    v32 = v31[:v21]
    v33 = v100.v79(v57, v58, v63, v26, v22, v6 + 5)
    v34 = v100.v80(v28, v27, v26, v24, v59, v15, v57, v58, v63, v33, v31, v71, v72, v75, v76)
    v56(f"baseline nt={v34['next_tok']:.3f} hold={v34['hold_ce']:.3f} gap={v34['inversion']['gap_hard_minus_para']:+.3f}")
    v11, v81 = v100.v82(v28, v57, v58, v27, v26, v15, v18, v7, v6 + 1, 'scale_l02', v62, v33, v32, early_stop=False, n_probes=v23)
    v35 = v100.v80(v11, v27, v26, v24, v59, v15, v57, v58, v63, v33, v31, v71, v72, v75, v76)
    v56(f"DONE nt={v35['next_tok']:.3f} hold={v35['hold_ce']:.3f} gap={v35['inversion']['gap_hard_minus_para']:+.3f} unif={v35['uniformity']:.3f} mem={v35['slot_mem']:.3f} leak={v35['param_leak']:.3f} wall={v53.v53() - v17:.0f}s")
    v36 = v35['next_tok'] >= v8['next_tok'] - 0.01
    v37 = v35['inversion']['gap_hard_minus_para'] <= v8['gap'] - 0.005
    v38 = v35['hold_ce'] <= v8['hold_ce'] + 0.03
    v39 = v35['slot_mem'] >= 0.75 and v35['param_leak'] <= 0.4
    v40 = v81['tokens_ce'] >= v18 * 0.98
    if v36 and v37 and v38 and v39 and v40:
        v83 = 'SCALE_JOINT_OK'
    elif v39 and v40 and (v36 or v37):
        v83 = 'SCALE_JOINT_PARTIAL'
    else:
        v83 = 'SCALE_JOINT_NO'
    v41 = {'stage': 253, 'overall': v83, 'lambda': v7, 'token_budget': v18, 'tokens_ce': v81['tokens_ce'], 'reference_252_4M': v8, 'gates': {'G_nt_vs_252': v36, 'G_gap_vs_252': v37, 'G_hold_vs_252': v38, 'G_memory_clean': v39, 'G_full_budget': v40}, 'baseline': v34, 'final': v35, 'train_meta': v81, 'timestamp': v117.v114(v118.v115).v101(), 'wall_s': v53.v53() - v17}
    v1.v50(v110.v102(v41, indent=2), encoding='utf-8')
    v2.v50(f"# Stage 253 scale joint (λ={v7})\n\n**{v83}** budget={v18} tokens_ce={v81['tokens_ce']}\nnt {v34['next_tok']:.3f}->{v35['next_tok']:.3f} gap {v34['inversion']['gap_hard_minus_para']:+.3f}->{v35['inversion']['gap_hard_minus_para']:+.3f} hold {v35['hold_ce']:.3f}\n", encoding='utf-8')
    v56(v110.v102({'overall': v83, 'nt': v35['next_tok'], 'gap': v35['inversion']['gap_hard_minus_para']}, indent=2))
    if not v14.v55 and v81['tokens_ce'] >= 500000:
        v5.v87.v45(exist_ok=True)
        v91.v103({'model': v11.v116(), 'stage': 253, 'lambda': v7, 'tokens_ce': v81['tokens_ce']}, v5)
    return 0
if v42 == '__main__':
    raise v84(v104())