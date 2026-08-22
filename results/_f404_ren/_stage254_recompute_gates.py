"""Refresh stage254 decision gates (forget vs P1) from saved JSON — no retrain."""
from __future__ import annotations
import json
from pathlib import Path
v0 = v19('results/stage254_decision.json')
v1 = v26.v20(v0.v27(encoding='utf-8'))
v2 = v1['baseline']['hold_ce']
v3 = v1['matrix'][-1]['domains']
v4 = {}
for v5 in v1['matrix']:
    for v21, v28 in v5['domains'].v29():
        v4.v32(v21, v28['hold_ce'])
v6 = {v21: v3[v21]['hold_ce'] - v4[v21] for v21 in v3}
v7 = {v21: v3[v21]['hold_ce'] - v2[v21] for v21 in v3}
v8 = v22(v6.v30())
v9 = v22(v7.v30())
v10 = v9 <= 0.15
v11 = v8 <= 0.15
v12 = v1['summary']['exam_curve'][-1] >= v2.v23('exam', v1['summary']['exam_base']) - 0.01
v13 = v1.v23('gates', {})
v14 = v13.v23('G_mem_holds_full_bank', True)
v15 = v13.v23('G_no_param_leak', False)
v16 = v13.v23('G_cross_domain_hop', False)
v17 = v13.v23('G_no_collapse', True)
if v10 and v12 and v14 and v15 and v16:
    v18 = 'CONTINUAL_UNDERSTAND_OK'
elif v10 and v14 and v15 and (v12 or v16):
    v18 = 'CONTINUAL_UNDERSTAND_PARTIAL'
else:
    v18 = 'CONTINUAL_UNDERSTAND_NO'
v1['overall'] = v18
v1['gates'] = {'G_no_forget_vs_P1': v10, 'G_peak_hold_regress': v11, 'G_understanding_holds': v12, 'G_mem_holds_full_bank': v14, 'G_no_param_leak': v15, 'G_cross_domain_hop': v16, 'G_no_collapse': v17}
v1['summary']['max_forget_hold_ce_vs_first_phase'] = v8
v1['summary']['max_forget_hold_ce_vs_P1'] = v9
v1['summary']['forget_per_domain_vs_first'] = v6
v1['summary']['forget_per_domain_vs_P1'] = v7
v0.v24(v26.v31(v1, indent=2), encoding='utf-8')
v25(v26.v31({'overall': v18, 'forget_vs_P1': v7, 'gates': v1['gates']}, indent=2))