"""
Stage 214 — Recency-weighted ctx_fp (zero-train extension of 194).

Sweep lambda on entity recall vs mean-pool baseline (lambda=0).

  python _stage214_recency_ctx.py
  python _stage214_recency_ctx.py --smoke
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import build_memory, score_entity_items
from _tapelm_ext import RecencyFpBank
v0 = v11('results')
v1 = v11('checkpoints/stage191_p1_curve.pt')
v2 = v11('data/_wikitext103_train.txt')
v3 = v11('data/stage191_exam_v3.jsonl')
v4 = v0 / 'stage214_decision.json'
v5 = v0 / 'stage214_mini.md'
v6 = v0 / '_stage214_log.txt'
v7 = 150000000
v8 = 3000000
v9 = [0.0, 0.05, 0.1, 0.2, 0.35]

def log(v12: v36) -> None:
    v37(v12, flush=True)
    v6.v62.v38(parents=True, exist_ok=True)
    with v6.v63('a', encoding='utf-8') as v39:
        v39.v64(v12 + '\n')

def main() -> v10:
    v13 = v65.v40()
    v13.v41('--smoke', action='store_true')
    v14 = v13.v42()
    v0.v38(parents=True, exist_ok=True)
    v6.v43('', encoding='utf-8')
    v44(f'Stage214 start {v96.v92(v97.v93).v79()}')
    v15 = v66.v15('cuda' if v66.v90.v82() else 'cpu')
    v16 = v45.v45()
    v46, v47, v48, v49 = v50()
    v17 = v67.v51(v36(v83.v68))
    v18 = v17.v52()
    v19 = v17.v69(v70) or 0
    v20 = v84(v49, v18).v53(v15)
    v20.v54(v66.v85(v1, map_location=v15, weights_only=False)['model'])
    v20.v55()
    with v2.v63('r', encoding='utf-8', errors='ignore') as v39:
        v56 = v39.v71(v7 if not v14.v57 else 2000000)
    v21 = v56[-v8:]
    v22 = [v87.v86() for v87 in v21.v91('\n') if 120 < v94(v87.v86()) < 1000][:200 if v14.v57 else 1200]
    v23 = [v88.v72(v73) for v73 in v3.v95(encoding='utf-8').v89()]
    v24 = [0.0, 0.1] if v14.v57 else v9
    v25 = {}
    for v26 in v24:
        v58 = v74(v20, v48, v15, lam=v26)
        v75, v76 = v77(v22, v58, f'lam={v26}')
        v59 = v78(v23, v17, v19, v58, v75, v76)
        v25[v36(v26)] = v59
        v44(f"  lam={v26}: acc={v59['acc']:.3f} n={v59['n']}")
    v27 = v25['0.0']['acc']
    v28 = v60(v24, key=lambda v73: v25[v36(v73)]['acc'])
    v29 = v25[v36(v28)]['acc']
    v30 = v29 - v27
    v31 = v30 >= 0.02 and v28 > 0
    v32 = v29 >= 0.5
    v33 = 'RECENCY_CTX_WIN' if v31 and v32 else 'RECENCY_CTX_MARGINAL' if v30 > 0 else 'RECENCY_CTX_NO'
    v34 = {'stage': 214, 'overall': v33, 'gates': {'G1_delta_vs_mean': v31, 'G2_acc': v32}, 'baseline_acc': v27, 'best_lambda': v28, 'best_acc': v29, 'delta': v30, 'sweep': v25, 'timestamp': v96.v92(v97.v93).v79()}
    v4.v43(v88.v80(v34, indent=2), encoding='utf-8')
    v5.v43(f'# Stage214\n\n**{v33}** best_lam={v28} acc={v29:.3f} (base {v27:.3f})\n', encoding='utf-8')
    v44(f'VERDICT {v33} ({v45.v45() - v16:.0f}s)')
    return 0
if v35 == '__main__':
    raise v61(v81())