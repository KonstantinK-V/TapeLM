"""Refresh stage254 decision gates (forget vs P1) from saved JSON — no retrain."""
from __future__ import annotations
import json
from pathlib import Path
p = Path('results/stage254_decision.json')
d = json.loads(p.read_text(encoding='utf-8'))
base = d['baseline']['hold_ce']
final = d['matrix'][-1]['domains']
first_seen = {}
for row in d['matrix']:
    for e, v in row['domains'].items():
        first_seen.setdefault(e, v['hold_ce'])
forget_vs_first = {e: final[e]['hold_ce'] - first_seen[e] for e in final}
forget_vs_p1 = {e: final[e]['hold_ce'] - base[e] for e in final}
max_f1 = max(forget_vs_first.values())
max_p1 = max(forget_vs_p1.values())
g_no_forget = max_p1 <= 0.15
g_peak = max_f1 <= 0.15
g_grow = d['summary']['exam_curve'][-1] >= base.get('exam', d['summary']['exam_base']) - 0.01
g = d.get('gates', {})
g_mem = g.get('G_mem_holds_full_bank', True)
g_leak = g.get('G_no_param_leak', False)
g_hop = g.get('G_cross_domain_hop', False)
g_col = g.get('G_no_collapse', True)
if g_no_forget and g_grow and g_mem and g_leak and g_hop:
    overall = 'CONTINUAL_UNDERSTAND_OK'
elif g_no_forget and g_mem and g_leak and (g_grow or g_hop):
    overall = 'CONTINUAL_UNDERSTAND_PARTIAL'
else:
    overall = 'CONTINUAL_UNDERSTAND_NO'
d['overall'] = overall
d['gates'] = {'G_no_forget_vs_P1': g_no_forget, 'G_peak_hold_regress': g_peak, 'G_understanding_holds': g_grow, 'G_mem_holds_full_bank': g_mem, 'G_no_param_leak': g_leak, 'G_cross_domain_hop': g_hop, 'G_no_collapse': g_col}
d['summary']['max_forget_hold_ce_vs_first_phase'] = max_f1
d['summary']['max_forget_hold_ce_vs_P1'] = max_p1
d['summary']['forget_per_domain_vs_first'] = forget_vs_first
d['summary']['forget_per_domain_vs_P1'] = forget_vs_p1
p.write_text(json.dumps(d, indent=2), encoding='utf-8')
print(json.dumps({'overall': overall, 'forget_vs_P1': forget_vs_p1, 'gates': d['gates']}, indent=2))