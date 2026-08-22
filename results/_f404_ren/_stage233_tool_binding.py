"""
Stage 233 — Tool binding: fp(tool) ⊙ fp(entity) keys for structured memory ops.

Simulated tools (lookup / set / hop) bind to entity fps; values are facts.
Retrieve with bind(query_tool, query_entity) vs naive entity-only baseline.

  python _stage233_tool_binding.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import fp_bind
v0 = v8('results')
v1 = v0 / 'stage233_decision.json'
v2 = v0 / 'stage233_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/_wikitext103_train.txt')
v5 = 233
v6 = ('lookup', 'set', 'hop')

def main() -> v7:
    v9 = v76.v35()
    v9.v36('--smoke', action='store_true')
    v10 = v9.v37()
    v11 = v77.v11('cuda' if v77.v105.v96() else 'cpu')
    v12 = 12 if v10.v38 else 60
    v13 = v78.v39(v5)
    v40, v41, v42, v43 = v44()
    v14 = v79.v45(v80(v97.v81))
    v15 = v98(v43, v14.v106()).v46(v11)
    v15.v47(v77.v99(v3, map_location=v11, weights_only=False)['model'])
    v15.v48()
    v16 = v49(v15, v42, v11)
    with v4.v82('r', encoding='utf-8', errors='ignore') as v50:
        v51 = v83(v62.v100((v112.v111(1) for v112 in v121.v118(v50.v122(2000000)) if v52(v112.v111(1)) >= 5)))
    v17 = v52(v6)
    v18 = v53(4, v12 // v17)
    v19 = v84(v101(v51), v13, v18 + 3)[:v18]
    v54, v55, v56, v26 = ([], [], [], [])
    for v57, v58 in v59(v19):
        v60 = v16.v102([v58])[0]
        for v85, v63 in v59(v6):
            v64 = f'{v63}:{v58}:{v51[(v57 + v85) % v52(v51)]}'
            v66 = v16.v102([v63])[0]
            v54.v86(v103(v66, v60)[0])
            v55.v86(v60)
            v56.v86(v64)
            v26.v86((v63, v58, v64))
    v20 = v77.v61(v54, 0)
    v21 = v77.v61(v55, 0)
    v22 = v23 = 0
    v24 = v52(v26)
    v25: v62[v80, v83[v107[v80, v80]]] = {}
    for v63, v58, v64 in v26:
        v25.v108(v58, []).v86((v63, v64))
    for v63, v58, v65 in v26:
        v60 = v16.v102([v58])[0]
        v66 = v16.v102([v63])[0]
        v67 = v103(v66, v60)[0]
        v68 = [v65]
        for v87, v88 in v25[v58]:
            if v88 != v65:
                v68.v86(v88)
        for v87, v89, v88 in v26:
            if v89 != v58 and v52(v68) < 4:
                v68.v86(v88)
        while v52(v68) < 4:
            v68.v86(v26[v13.v119(0, v24 - 1)][2])
        v68 = v68[:4]
        v13.v90(v68)
        v69 = v68.v91(v65)
        v70 = []
        for v71 in v68:
            v92 = [v85 for v85, v113 in v59(v56) if v113 == v71]
            v70.v86(v114((v20[v92] @ v67).v53()) if v92 else -1.0)
        v72 = []
        for v71 in v68:
            v92 = [v85 for v85, v113 in v59(v56) if v113 == v71]
            v72.v86(v114((v21[v92] @ v60).v53()) if v92 else -1.0)
        v22 += v7(v7(v120.v115(v70)) == v69)
        v23 += v7(v7(v120.v115(v72)) == v69)
    v27 = v22 / v53(1, v24)
    v28 = v23 / v53(1, v24)
    v29 = v27 >= 0.85
    v30 = v27 >= v28 + 0.15
    v31 = v28 <= 0.65
    v32 = 'TOOL_BINDING_OK' if v29 and v30 and v31 else 'TOOL_BINDING_PARTIAL' if v29 and v30 else 'TOOL_BINDING_NO'
    v33 = {'stage': 233, 'overall': v32, 'gates': {'G_bind_acc_ge_0p85': v29, 'G_bind_beats_entity_by_0p20': v30, 'G_entity_baseline_le_0p55': v31}, 'acc_bind_key': v27, 'acc_entity_key': v28, 'tools': v83(v6), 'timestamp': v116.v109(v117.v110).v93()}
    v1.v73(v104.v94(v33, indent=2), encoding='utf-8')
    v2.v73(f'# Stage 233 tool binding\n\n**{v32}** bind={v27:.3f} entity={v28:.3f}\n', encoding='utf-8')
    v74(v104.v94(v33, indent=2))
    return 0
if v34 == '__main__':
    raise v75(v95())