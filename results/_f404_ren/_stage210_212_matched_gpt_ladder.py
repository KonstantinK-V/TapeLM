"""
Refresh 210–212 verdict framing + matched-GPT ladder (cheap; no retrain of 210–212).

  python _stage210_212_matched_gpt_ladder.py           # patch decisions + write ladder JSON
  python _stage210_212_matched_gpt_ladder.py --run-gpt # also run 210 GPT parametric chain baseline

Does NOT change P1 or re-run SoftFollow training.
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, load_data
from _stage196_tapelm import gpt_span, load_gpt
from _stage210_softfollow_forward import CHAIN_LEN, encode_query, encode_word
v0 = v7('results')
v1 = v0 / 'internalization_210_212_claim_scope.json'
v2 = v0 / 'stage210_212_matched_gpt_ladder.json'
v3 = 210212
v4 = {210: v0 / 'stage210_decision.json', 211: v0 / 'stage211_decision.json', 212: v0 / 'stage212_decision.json'}

def load_claim() -> v5:
    if v1.v38():
        return v78.v40(v1.v79(encoding='utf-8'))
    return {}

def rename_overall(v8: v5) -> v5:
    v9 = v8.v39('overall')
    if v9 == 'THESIS_NO':
        v8['overall_legacy'] = 'THESIS_NO'
        v8['overall'] = 'THESIS_NO_AT_SCALE'
    return v8

def patch_stage(v10: v7, v11: v5, v12: v5) -> None:
    if not v10.v38():
        return
    v8 = v78.v40(v10.v79(encoding='utf-8'))
    v8 = v41(v8)
    v8['claim_scope'] = v11
    v13 = v6(v10.v102.v81('stage', '').v101('_')[0])
    v8['gpt_matched_ladder'] = v12.v39(f'stage_{v13}', {})
    if v8['overall'] == 'THESIS_NO_AT_SCALE':
        v8['interpretation'] = 'THESIS_NO_AT_SCALE: gates failed on frozen P1 @ d256/6L with controls run in this JSON. Not a permanent falsification; see gpt_matched_ladder and 209 for matched-GPT / scale context.'
    v10.v42(v78.v80(v8, indent=2), encoding='utf-8')
    v14 = v10.v43(v10.v102.v81('_decision.json', '_mini.md'))
    if v14.v38():
        v44 = v14.v79(encoding='utf-8')
        v44 = v44.v81('`THESIS_NO`', '`THESIS_NO_AT_SCALE`')
        if '**Overall:**' in v44 and 'legacy' not in v44:
            v44 = v44.v81('**Overall:** `THESIS_NO_AT_SCALE`', '**Overall:** `THESIS_NO_AT_SCALE` (legacy label: THESIS_NO)')
        v14.v42(v44, encoding='utf-8')

@v64.v29()
def gpt_chain_parametric(v15, v16: v6=80) -> v5:
    """210-style k-hop 4-way token ID — GPT parametric, no tape (matched LM baseline)."""
    from _stage192_fp_lexicon import gen_fakes
    import numpy as np
    v45, v45, v46, v47 = v48()
    v17 = v82.v49(v83(v103.v84))
    v18 = v17.v85(v86) or 0
    v19 = v50(v15)
    v20 = v87.v51(v3)
    v21 = v52(v88(), v20, 200)
    v22 = [v21[v89 * v91:(v89 + 1) * v91] for v89 in v90(50)]
    v22 = [v24 for v24 in v22 if v95(v24) == v91]
    v23 = []
    for v24 in v22:
        for v53 in v90(1, v91):
            v23.v104((v24[0], v53, v24[v53]))
    v20.v54(v23)
    v23 = v23[:v16]
    v25 = v55(v5.v92((v105 for v24 in v22 for v105 in v24)))
    v26 = {1: 0, 2: 0, 3: 0}
    v27 = {1: 0, 2: 0, 3: 0}
    for v56, v53, v57 in v23:
        v27[v53] += 1
        v58 = v93(v17, v18, v56, v53)
        v59 = [v57] + [v25[v20.v113(0, v95(v25) - 1)] for v45 in v90(3)]
        v60 = v55(v90(4))
        v20.v54(v60)
        v61 = [v59[v89] for v89 in v60]
        v62 = v60.v94(0)
        v63 = [v106(v19, v15, v58, v109(v17, v18, v24)) for v24 in v61]
        v26[v53] += v6(v6(v118.v114(v63)) == v62)
    v28 = {v83(v53): v26[v53] / v107(1, v27[v53]) for v53 in (1, 2, 3)}
    return {'protocol': 'gpt_parametric_chain_4way', 'n': v95(v23), 'acc_by_hop': v28, 'chance': 0.25}

def build_ladder(v30: v65) -> v5:
    v12: v5 = {'timestamp': v115.v110(v116.v111).v96(), 'reference': '209_sem_scaling_teacher_209', 'stage_209_summary': {'verdict': 'STRUCTURAL_BLOCK_NO', 'note': 'PAWS tracks matched GPT at d128/d192/d256 — shared scale ceiling, not curve-specific blindness'}}
    v31 = v4[211]
    if v31.v38():
        v66 = v78.v40(v31.v79(encoding='utf-8'))
        v24 = v66.v39('clean', {})
        v12['stage_211'] = {'status': 'gpt_control_in_original_run', 'internal_tape': v24.v39('internal_tape'), 'endpoint_only': v24.v39('endpoint_only'), 'external_slots': v24.v39('external_slots'), 'gpt_incontext': v24.v39('gpt_incontext'), 'parity_read': 'internal≈gpt_ic≈endpoint<<external; matched GPT does not solve task either'}
    v32 = v4[210]
    if v32.v38():
        v67 = v78.v40(v32.v79(encoding='utf-8'))
        v12['stage_210'] = {'status': 'gpt_ladder_pending' if not v30 else 'gpt_baseline_run', 'curve_soft_follow_test': v67.v39('soft_follow_token', {}).v39('test'), 'curve_external_cosine': v67.v39('external_loop_cosine_test')}
    v33 = v4[212]
    if v33.v38():
        v68 = v78.v40(v33.v79(encoding='utf-8'))
        v12['stage_212'] = {'status': 'curve_only_collision_in_original_run', 'collision_4way': v68.v39('t1_collision_4way'), 'para_hard': v68.v39('t2_para_hard'), 'cross_ref': 'For substrate semantic parity vs GPT use 209; 212 tested instance head on curve tape only'}
    if v30:
        v15 = v64.v15('cuda' if v64.v117.v112() else 'cpu')
        v69 = v97(v15)
        v12.v108('stage_210', {})['gpt_parametric_chain'] = v69
        v12['stage_210']['status'] = 'gpt_baseline_run'
        v12['stage_210']['parity_read'] = 'Compare gpt_parametric_chain acc_by_hop to curve soft_follow_token test; external cosine is curve-only affordance'
    return v12

def main() -> v6:
    v34 = v98.v70()
    v34.v71('--run-gpt', action='store_true', help='Run 210 GPT parametric chain baseline (~1 min GPU)')
    v35 = v34.v72()
    v11 = v73()
    v12 = v74(v35.v30)
    v2.v42(v78.v80(v12, indent=2), encoding='utf-8')
    for v36 in v4.v75():
        v99(v36, v11, v12)
    v76(f'Wrote {v2}; patched {[v36.v102 for v36 in v4.v75() if v36.v38()]}')
    return 0
if v37 == '__main__':
    raise v77(v100())