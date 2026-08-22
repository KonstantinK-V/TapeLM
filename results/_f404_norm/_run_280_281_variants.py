"""Cheap 280/281 variant battery. Writes tagged decisions under results/."""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
PY = sys.executable

def run(tag: str, cmd: list[str]) -> dict:
    log = RES / f'_var_{tag}.out'
    print(f"\n=== {tag} ===\n{' '.join(cmd)}", flush=True)
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    row = {'tag': tag, 'cmd': cmd[1:], 'exit': p.returncode, 'log': str(log.name)}
    for src, dst_key in [(RES / 'stage281_decision.json', 'stage281'), (RES / 'stage281_decision_fia.json', 'stage281_fia'), (RES / 'stage280_decision_fp.json', 'stage280'), (RES / 'stage280_decision.json', 'stage280_plain')]:
        if src.exists():
            out = RES / f'{src.stem}_{tag}{src.suffix}'
            shutil.copy(src, out)
            try:
                d = json.loads(src.read_text(encoding='utf-8'))
            except Exception as e:
                row[dst_key] = {'error': str(e)}
                continue
            row[dst_key] = summarize(d)
            row[f'{dst_key}_path'] = out.name
    print(json.dumps({k: row[k] for k in row if k not in ('cmd',)}, ensure_ascii=False), flush=True)
    return row

def summarize(d: dict) -> dict:
    out = {'overall': d.get('overall')}
    if 'ceiling_before' in d or 'ceiling_after' in d:
        b, a = (d.get('ceiling_before') or {}, d.get('ceiling_after') or {})
        out['ceiling_before'] = (b or {}).get('reward') if isinstance(b, dict) else b
        out['ceiling_after'] = (a or {}).get('reward') if isinstance(a, dict) else a
        out['frames'] = d.get('frames')
        out['gates'] = d.get('gates')
        return out
    out['teacher_ceiling'] = d.get('teacher_ceiling_reward')
    out['gates'] = d.get('gates')
    h = d.get('held_out') or d.get('novel_tape') or {}
    if isinstance(h, dict):
        out['held'] = {'reward': h.get('reward_total'), 'teacher': h.get('teacher_reward_total'), 'precision': h.get('retrieval_precision'), 'recall': h.get('witness_recall'), 'hops': h.get('hops_per_episode'), 'mean_cands': h.get('mean_candidates'), 'tie_teacher_abstain': (h.get('tie') or {}).get('teacher_abstain'), 'decidable_teacher_acc': (h.get('decidable') or {}).get('teacher_acc_all')}
    return out

def main() -> int:
    RES.mkdir(exist_ok=True)
    rows = []
    rows.append(run('281_base', [PY, '-u', '_stage281_frames.py', '--smoke']))
    rows.append(run('281_loose', [PY, '-u', '_stage281_frames.py', '--smoke', '--min-confirm', '0.05', '--min-n', '3', '--min-anchors', '1', '--max-values-per-anchor', '5.0']))
    rows.append(run('281_loose_fia', [PY, '-u', '_stage281_frames.py', '--smoke', '--frame-in-address', '--min-confirm', '0.05', '--min-n', '3', '--min-anchors', '1', '--max-values-per-anchor', '5.0']))
    rows.append(run('281_and_ok', [PY, '-u', '_stage281_frames.py', '--smoke', '--min-confirm', '0.18', '--max-values-per-anchor', '3.0', '--min-anchors', '2']))
    rows.append(run('280_kgap07', [PY, '-u', '_stage280_raw_exam.py', '--smoke', '--k-gap', '0.7']))
    rows.append(run('280_topk3', [PY, '-u', '_stage280_raw_exam.py', '--smoke', '--topk', '3', '--k-gap', '0.35']))
    rows.append(run('280_hopalways', [PY, '-u', '_stage280_raw_exam.py', '--smoke', '--hop-min', '99', '--k-gap', '0.5']))
    rows.append(run('280_hopnone_kgap07', [PY, '-u', '_stage280_raw_exam.py', '--smoke', '--hop', 'none', '--k-gap', '0.7']))
    summary = RES / 'stage280_281_variant_battery.json'
    summary.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    md = ['# 280/281 variant battery\n', '| tag | overall | ceiling / key | note |\n|---|---|---|---|\n']
    for r in rows:
        s281 = r.get('stage281') or r.get('stage281_fia') or {}
        s280 = r.get('stage280') or r.get('stage280_plain') or {}
        if s281:
            ceil = f"{s281.get('ceiling_before')} → {s281.get('ceiling_after')}"
            md.append(f"| {r['tag']} | {s281.get('overall')} | {ceil} | frames {s281.get('frames')} |\n")
        elif s280:
            h = s280.get('held') or {}
            ceil = s280.get('teacher_ceiling')
            md.append(f"| {r['tag']} | {s280.get('overall')} | ceiling={ceil} | prec={h.get('precision')} rec={h.get('recall')} cands={h.get('mean_cands')} hops={h.get('hops')} tie_tabst={h.get('tie_teacher_abstain')} |\n")
        else:
            md.append(f"| {r['tag']} | exit={r['exit']} | — | no decision |\n")
    (RES / 'stage280_281_variant_battery.md').write_text(''.join(md), encoding='utf-8')
    print('\nWrote', summary, 'and stage280_281_variant_battery.md', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())