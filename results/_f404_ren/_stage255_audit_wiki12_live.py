"""Read-only audit of wiki:12 recall metrics (does not touch running process)."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import _stage24x_lib as L
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tapelm_ext import DomainAdapter
from tokenizers import Tokenizer
v0 = 255 + 9000
v1 = v3('results/stream255/wiki12')

def main() -> v2:
    v4 = v49.v20((v1 / 'state.json').v50(encoding='utf-8'))
    v5 = v51.v21(v1 / 'tape.pt', map_location='cpu', weights_only=False)
    v22, v23 = (v5['K'], v5['values'])
    v6 = v24(256)
    v6.v25(v51.v21(v1 / 'query_adapter.pt', map_location='cpu', weights_only=False))
    v6.v26()
    v7 = v51.v7('cpu')
    v27, v27, v28, v29 = v30()
    v8 = v52.v31(v53(v65.v54))
    v9 = v66(v29, v8.v68()).v32(v7)
    v9.v25(v51.v21('checkpoints/stage191_p1_curve.pt', map_location=v7, weights_only=False)['model'])
    v9.v26()
    v10 = v33(v9, v28, v7)
    v11 = v4['probe_facts']['wiki']
    v12 = [v17 for v17 in v11 if not v17['wq_train']]
    v13 = [v17 for v17 in v11 if v17['wq_train']]
    v14 = v34(v39.v55([v17['value'] for v17 in v11] + v23))
    v35(f"slots={v63(v23)} eval={v63(v12)} train={v63(v13)} chunk={v4['chunk_i']}")
    for v36, v37 in [('frozen', None), ('W_q', v6)]:
        v38 = v62.v46(v12, v14, v10, v22, v23, v0, W_bwd=v37)
        v35(f'  {v36} eval held-out: {v38}')
    v15: v39[v53, v34[v2]] = {}
    for v40, v41 in v42(v23):
        v15.v69(v41, []).v56(v40)
    v16 = v22.v43()
    v44, v45 = ([], [])
    for v17 in v12:
        for v37, v57 in [(None, v44), (v6, v45)]:
            v58 = v62.v77(v10, v17, v37).v73().v43()
            v59 = v16 @ v58
            v60 = v17['value']
            v61 = v43(v59[v15[v60]].v70())
            v57.v56(1 + v2((v59 > v61).v74().v76()))
    v35(f'  frozen ranks: min={v71(v44)} max={v70(v44)} mean={v74(v44) / v63(v44):.1f}')
    v35(f'  W_q ranks:    min={v71(v45)} max={v70(v45)} unique={v72(v75(v45))}')
    v18 = v62.v46(v13, v14, v10, v22, v23, v0, W_bwd=v6)
    v35(f'  W_q on TRAIN half (should not be in eval): {v18}')
    for v17 in v12[:3]:
        v47 = v63(v15.v67(v17['value'], []))
        v35(f"  value '{v17['value']}' slots on tape: {v47}")
    return 0
if v19 == '__main__':
    raise v48(v64())