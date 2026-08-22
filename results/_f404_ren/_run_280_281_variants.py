"""Cheap 280/281 variant battery. Writes tagged decisions under results/."""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path
v0 = v47(v48).v20().v1
v2 = v0 / 'results'
v3 = v7.v4

def run(v8: v21, v9: v36[v21]) -> v5:
    v10 = v2 / f'_var_{v8}.out'
    v22(f"\n=== {v8} ===\n{' '.v45(v9)}", flush=True)
    with v10.v37('w', encoding='utf-8') as v23:
        v24 = v49.v38(v9, cwd=v0, stdout=v23, stderr=v49.v50, text=True)
    v11 = {'tag': v8, 'cmd': v9[1:], 'exit': v24.v25, 'log': v21(v10.v39)}
    for v26, v27 in [(v2 / 'stage281_decision.json', 'stage281'), (v2 / 'stage281_decision_fia.json', 'stage281_fia'), (v2 / 'stage280_decision_fp.json', 'stage280'), (v2 / 'stage280_decision.json', 'stage280_plain')]:
        if v26.v40():
            v13 = v2 / f'{v26.v58}_{v8}{v26.v59}'
            v56.v51(v26, v13)
            try:
                v12 = v54.v57(v26.v60(encoding='utf-8'))
            except v52 as e:
                v11[v27] = {'error': v21(v61)}
                continue
            v11[v27] = v53(v12)
            v11[f'{v27}_path'] = v13.v39
    v22(v54.v41({v55: v11[v55] for v55 in v11 if v55 not in ('cmd',)}, ensure_ascii=False), flush=True)
    return v11

def summarize(v12: v5) -> v5:
    v13 = {'overall': v12.v28('overall')}
    if 'ceiling_before' in v12 or 'ceiling_after' in v12:
        v42, v43 = (v12.v28('ceiling_before') or {}, v12.v28('ceiling_after') or {})
        v13['ceiling_before'] = (v42 or {}).v28('reward') if v29(v42, v5) else v42
        v13['ceiling_after'] = (v43 or {}).v28('reward') if v29(v43, v5) else v43
        v13['frames'] = v12.v28('frames')
        v13['gates'] = v12.v28('gates')
        return v13
    v13['teacher_ceiling'] = v12.v28('teacher_ceiling_reward')
    v13['gates'] = v12.v28('gates')
    v14 = v12.v28('held_out') or v12.v28('novel_tape') or {}
    if v29(v14, v5):
        v13['held'] = {'reward': v14.v28('reward_total'), 'teacher': v14.v28('teacher_reward_total'), 'precision': v14.v28('retrieval_precision'), 'recall': v14.v28('witness_recall'), 'hops': v14.v28('hops_per_episode'), 'mean_cands': v14.v28('mean_candidates'), 'tie_teacher_abstain': (v14.v28('tie') or {}).v28('teacher_abstain'), 'decidable_teacher_acc': (v14.v28('decidable') or {}).v28('teacher_acc_all')}
    return v13

def main() -> v6:
    v2.v30(exist_ok=True)
    v15 = []
    v15.v31(v38('281_base', [v3, '-u', '_stage281_frames.py', '--smoke']))
    v15.v31(v38('281_loose', [v3, '-u', '_stage281_frames.py', '--smoke', '--min-confirm', '0.05', '--min-n', '3', '--min-anchors', '1', '--max-values-per-anchor', '5.0']))
    v15.v31(v38('281_loose_fia', [v3, '-u', '_stage281_frames.py', '--smoke', '--frame-in-address', '--min-confirm', '0.05', '--min-n', '3', '--min-anchors', '1', '--max-values-per-anchor', '5.0']))
    v15.v31(v38('281_and_ok', [v3, '-u', '_stage281_frames.py', '--smoke', '--min-confirm', '0.18', '--max-values-per-anchor', '3.0', '--min-anchors', '2']))
    v15.v31(v38('280_kgap07', [v3, '-u', '_stage280_raw_exam.py', '--smoke', '--k-gap', '0.7']))
    v15.v31(v38('280_topk3', [v3, '-u', '_stage280_raw_exam.py', '--smoke', '--topk', '3', '--k-gap', '0.35']))
    v15.v31(v38('280_hopalways', [v3, '-u', '_stage280_raw_exam.py', '--smoke', '--hop-min', '99', '--k-gap', '0.5']))
    v15.v31(v38('280_hopnone_kgap07', [v3, '-u', '_stage280_raw_exam.py', '--smoke', '--hop', 'none', '--k-gap', '0.7']))
    v16 = v2 / 'stage280_281_variant_battery.json'
    v16.v32(v54.v41(v15, indent=2), encoding='utf-8')
    v17 = ['# 280/281 variant battery\n', '| tag | overall | ceiling / key | note |\n|---|---|---|---|\n']
    for v18 in v15:
        v33 = v18.v28('stage281') or v18.v28('stage281_fia') or {}
        v34 = v18.v28('stage280') or v18.v28('stage280_plain') or {}
        if v33:
            v44 = f"{v33.v28('ceiling_before')} → {v33.v28('ceiling_after')}"
            v17.v31(f"| {v18['tag']} | {v33.v28('overall')} | {v44} | frames {v33.v28('frames')} |\n")
        elif v34:
            v14 = v34.v28('held') or {}
            v44 = v34.v28('teacher_ceiling')
            v17.v31(f"| {v18['tag']} | {v34.v28('overall')} | ceiling={v44} | prec={v14.v28('precision')} rec={v14.v28('recall')} cands={v14.v28('mean_cands')} hops={v14.v28('hops')} tie_tabst={v14.v28('tie_teacher_abstain')} |\n")
        else:
            v17.v31(f"| {v18['tag']} | exit={v18['exit']} | — | no decision |\n")
    (v2 / 'stage280_281_variant_battery.md').v32(''.v45(v17), encoding='utf-8')
    v22('\nWrote', v16, 'and stage280_281_variant_battery.md', flush=True)
    return 0
if v19 == '__main__':
    raise v35(v46())