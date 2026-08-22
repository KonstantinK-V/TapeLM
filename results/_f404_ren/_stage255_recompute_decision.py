"""Re-score stage255 decision JSON with fixed-seed bank metrics (no GPU retrain).

  python _stage255_recompute_decision.py [--decision results/stage255_decision.json] [--tape results/stream255/tape.pt]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import _stage24x_lib as L
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from tokenizers import Tokenizer
v0 = 255 + 9000

def main() -> v1:
    v2 = v73.v43()
    v2.v44('--decision', type=v74, default='results/stage255_decision.json')
    v2.v44('--tape', type=v74, default='results/stream255/tape.pt')
    v2.v44('--state', type=v74, default='results/stream255/state.json')
    v3 = v2.v45()
    v4 = v46(v3.v47)
    if not v4.v75():
        raise v72(f'missing {v4}')
    v5 = v76.v48(v4.v77(encoding='utf-8'))
    v6 = v76.v48(v46(v3.v102).v77(encoding='utf-8'))
    v7 = v6['probe_facts']
    v8 = v78.v49(v46(v3.v79), map_location='cpu', weights_only=False)
    v9 = v8['values']
    v10 = v8['K']
    v11 = v78.v11('cpu')
    v50, v50, v51, v52 = v53()
    v12 = v80.v54(v74(v95.v81))
    v13 = v12.v55()
    v14 = v12.v82(v83) or 0
    v15 = v103.v96(v12, v51, v14, v13).v56(v11)
    v16 = v97(v52, v13).v56(v11)
    v16.v57(v78.v49('checkpoints/stage191_p1_curve.pt', map_location=v11, weights_only=False)['model'])
    v16.v58()
    v17 = v59(v16, v51, v11)
    v18 = v5.v60('history', [])
    v19 = v61(v5.v60('summary', {}).v60('baseline_hold_ce') or {})
    v20 = v46(v3.v79).v62 / 'holdouts.pt'
    if v20.v75() and (not v19):
        v63 = v78.v49(v20, map_location='cpu', weights_only=False)
        for v67, v84 in v63.v64():
            v19[v67] = v104.v98(v16, v84, v15, v14, v11)
    if not v19 and v18:
        for v67, v85 in v18[0].v60('hold_ce', {}).v64():
            v19.v99(v67, v85)
    v21 = v5.v60('summary', {}).v60('baseline_exam')
    if v21 is None:
        v64 = v100.v86(120)
        v21 = v100.v87(v16, v15, v14, v64, v11)
    v22 = v65(v61.v88([v106['value'] for v107 in v7.v9() for v106 in v107] + v9))
    for v23 in v18:
        v66 = {v67: v105.v101(v7[v67], v22, v17, v10, v9, v0) for v67 in v7}
        v23['probe_bank'] = v66
        if 'probe_recall' in v23:
            del v23['probe_recall']
    if not v18:
        raise v72('empty history')
    v24 = v18[-1]
    v25 = v65(v24['hold_ce'].v89())
    v26 = {}
    for v27 in v18:
        for v67, v90 in v27['hold_ce'].v64():
            v26.v99(v67, v90)
    v28 = {v67: v24['hold_ce'][v67] - v26[v67] for v67 in v25}
    v29 = {v67: v24['hold_ce'][v67] - v19.v60(v67, v26[v67]) for v67 in v25}
    v30 = [(v27['tape_slots'], v68((v108['top1'] for v108 in v27['probe_bank'].v9()))) for v27 in v18]
    v31 = [(v27['tape_slots'], v68((v108['mrr'] for v108 in v27['probe_bank'].v9()))) for v27 in v18]
    v32 = v68((v90['top1'] for v90 in v24['probe_bank'].v9()))
    v33 = v68((v90['mrr'] for v90 in v24['probe_bank'].v9()))
    v34 = v30[0][1]
    v35 = v91(v18) >= 2
    v36 = v92(v29.v9()) <= 0.15
    v37 = v92(v28.v9()) <= 0.15
    v38 = v24['exam_next_tok'] >= v21 - 0.01
    v39 = v32 >= v92(0.04, v34 - 0.03)
    v40 = v33 >= 0.06
    if v35 and v36 and v38 and v39:
        v41 = 'STREAM_INGEST_OK'
    elif v35 and v36 and (v38 or v40):
        v41 = 'STREAM_INGEST_PARTIAL'
    else:
        v41 = 'STREAM_INGEST_NO'
    v5['overall'] = v41
    v5['gates'] = {'G_streamed': v35, 'G_no_forget_vs_P1': v36, 'G_peak_hold_regress': v37, 'G_understanding_holds': v38, 'G_recall_top1_floor': v39, 'G_recall_mrr_floor': v40, 'G_tape_bounded': v24.v60('tape_mb', 0) < 2000}
    v5['summary'] = {**v5.v60('summary', {}), 'forget_hold_ce_vs_first_chunk': v28, 'forget_hold_ce_vs_P1': v29, 'baseline_hold_ce': v19, 'baseline_exam': v21, 'recall_top1_vs_bank': v30, 'recall_mrr_vs_bank': v31, 'recall_final_top1': v32, 'recall_final_mrr': v33}
    v5['history'] = v18
    v5['note'] = (v5.v60('note', '') + ' Recomputed probe_bank with fixed-seed top1/MRR.').v69()
    v4.v70(v76.v93(v5, indent=2), encoding='utf-8')
    v71(v76.v93({'overall': v41, 'gates': v5['gates'], 'recall_final': (v32, v33)}, indent=2))
    return 0
if v42 == '__main__':
    raise v72(v94())