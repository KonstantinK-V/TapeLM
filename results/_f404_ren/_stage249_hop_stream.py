"""
Stage 249 — Online hop-gated slot admission on a long domain stream.

Stream of candidate facts + filler. Admission:
  hop: keep if cos(fp(fact), evolving hop query) in top-k / above median of batch
  surprise: keep if subject is novel fake (all planted are); baseline = first-B budget
Compare recall of hop-relevant gold set vs uniform budget under same B.

  python _stage249_hop_stream.py [--smoke] [--steps N]  (steps ≈ stream length proxy)
"""
from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
v0 = v8('results')
v1 = v0 / 'stage249_decision.json'
v2 = v0 / 'stage249_mini.md'
v3 = v0 / '_stage249_log.txt'
v4 = v8('checkpoints/stage191_p1_curve.pt')
v5 = v8('data/_wikitext103_train.txt')
v6 = 249

def log(v9: v38) -> None:
    v10 = v9 if v9.v94('\n') else v9 + '\n'
    try:
        v95(v10, end='', flush=True)
    except v39:
        v95(v10.v147('ascii', 'replace').v134('ascii'), end='', flush=True)
    v3.v96.v40(parents=True, exist_ok=True)
    with v3.v97('a', encoding='utf-8') as v41:
        v41.v98(v10)

def main() -> v7:
    v11 = v99.v42()
    v11.v43('--smoke', action='store_true')
    v11.v43('--steps', type=v7, default=0, help='unused train steps; scales n_events')
    v12 = v11.v44()
    v3.v45('', encoding='utf-8')
    v13 = v100.v13('cuda' if v100.v135.v119() else 'cpu')
    v14 = v101.v46(v6)
    v15 = v47.v47()
    v16 = 20 if v12.v48 else 80
    v17 = 20 if v12.v48 else 80
    v18 = 12 if v12.v48 else 40
    v19 = 1 if not v12.v102 else v103(1, v12.v102 // 1000)
    v16 *= v49(4, v19)
    v17 *= v49(4, v19)
    v18 = v49(v18 * v49(3, v19), v16)
    v50(f'Stage249 start {v145.v141(v146.v142).v116()} rel={v16} irrel={v17} B={v18}')
    v51, v51, v52, v53 = v54()
    v20 = v104.v55(v38(v120.v105))
    v21 = v121(v53, v20.v136()).v56(v13)
    v21.v57(v100.v122(v4, map_location=v13, weights_only=False)['model'])
    v21.v58()
    v22 = v59(v21, v52, v13)
    with v5.v97('r', encoding='utf-8', errors='ignore') as v41:
        v60 = v41.v106(6000000 if v12.v48 else 25000000)
    v23 = v61(v123.v107((v9.v137(1) for v9 in v148.v143(v60) if v132(v9.v137(1)) >= 5)))
    v14.v62(v23)

    def mk(v63, v64, v65):
        v66 = [v124 for v124 in v144(v149(v23), v101.v46(v6 + v65), v63 + 40) if v132(v124) >= 5][:v63]
        v36 = []
        for v108, v109 in v110(v66):
            v111 = v23[(v65 + v108) % v132(v23)]
            if v64 == 'org':
                v125 = f'{v109} was appointed director of {v111} in the organization chronicle .'
            else:
                v125 = f'{v109} crossed the river near {v111} during the autumn migration .'
            v36.v126({'S': v109, 'value': v111, 'sent': v125, 'theme': v64, 'fid': f'{v64}_{v108}'})
        return v36
    v24 = v67(v16, 'org', 11)
    v25 = v67(v17, 'geo', 77)
    v26 = v24 + v25
    v14.v62(v26)
    v27 = v22.v68('In the report the organization appointed a new director of governance.')
    if v27 is None:
        v27 = v22.v127(['organization', 'director'])[0]

    def score(v41):
        v69 = v22.v127([v41['S']])[0]
        v70 = v22.v68(v41['sent'], exclude=v41['value'])
        v71 = v138.v128(v69 + v70, dim=-1) if v70 is not None else v69
        return v112((v71 * v27).v115())
    v28 = [(v129(v41), v41) for v41 in v26]
    v28.v72(key=lambda v139: -v139[0])
    v29 = [v41 for v51, v41 in v28[:v18]]
    v30 = v26[:v18]

    def bank_of(v73):
        if not v73:
            return (None, None)
        return v130.v113(v22, v73)
    v74, v75 = v76(v29)
    v77, v78 = v76(v30)
    v31 = [v41['value'] for v41 in v26] + v23[:100]

    def recall_theme(v79, v80, v81):
        v82 = [v41 for v41 in v79 if v140((v41['fid'] == v139['fid'] for v139 in (v29 if v80 is v74 else v30) or []))]
        return v130.v131(v79, v31, v22, v80, v81, v6) if v80 is not None else 0.0

    def theme_hit_rate(v83, v84, v80, v81):
        v85 = 0
        for v41 in v83:
            if v41['fid'] not in {v139['fid'] for v139 in v84}:
                continue
            v114 = v130.v131([v41], v31, v22, v80, v81, v6)
            v85 += v7(v114 >= 0.99)
        v86 = v115((1 for v41 in v83 if v41['fid'] in {v139['fid'] for v139 in v84}))
        return (v85 / v103(1, v86), v86)
    v87, v88 = v89(v24, v29, v74, v75)
    v90, v91 = v89(v24, v30, v77, v78)
    v32 = v115((1 for v41 in v29 if v41['theme'] == 'org')) / v103(1, v132(v29))
    v33 = v115((1 for v41 in v30 if v41['theme'] == 'org')) / v103(1, v132(v30))
    v34 = v32 >= v33 + 0.15
    v35 = v87 >= 0.8 and v88 >= v18 // 3
    if v34 and v35:
        v92 = 'HOP_STREAM_OK'
    elif v34 or v35:
        v92 = 'HOP_STREAM_PARTIAL'
    else:
        v92 = 'HOP_STREAM_NO'
    v36 = {'stage': 249, 'overall': v92, 'budget': v18, 'n_rel': v16, 'n_irrel': v17, 'gates': {'G_hop_precision_vs_uniform': v34, 'G_admitted_rel_util': v35}, 'hop': {'rel_frac': v32, 'rel_acc': v87, 'rel_admitted': v88}, 'uniform': {'rel_frac': v33, 'rel_acc': v90, 'rel_admitted': v91}, 'timestamp': v145.v141(v146.v142).v116(), 'wall_s': v47.v47() - v15}
    v1.v45(v133.v117(v36, indent=2), encoding='utf-8')
    v2.v45(f'# Stage 249 hop stream\n\n**{v92}** hop_rel_frac={v32:.2f} uni={v33:.2f}\n', encoding='utf-8')
    v50(v133.v117(v36, indent=2))
    return 0
if v37 == '__main__':
    raise v93(v118())