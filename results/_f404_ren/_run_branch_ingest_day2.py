"""
Day-2 extension (~24h) for ingest-forks branch.

Waits until phase1 finishes (PIPELINE END or leftover done), then runs ~24h more.

  python _run_branch_ingest_day2.py [--hours 24] [--now] [--resume]
Stop: results/branch_ingest_50h/day2/STOP
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
v0 = v94(v95).v39().v1
v2 = v0 / 'results' / 'branch_ingest_50h'
v3 = v2 / 'day2'
v4 = v2 / 'state.json'
v5 = v3 / 'state.json'
v6 = v3 / 'journal.jsonl'
v7 = v3 / 'master.log'
v8 = v3 / 'FINAL_REPORT.md'
v9 = v3 / 'STOP'
v10 = v2 / 'STOP'
v11 = v18.v12

def utc() -> v13:
    return v104.v66(v105.v46).v40()

def log(v19: v13) -> None:
    v20 = v19.v41('≈', '~').v41('—', '-').v41('–', '-').v41('→', '->')
    v21 = f'[{v46()}] {v20}'
    try:
        v71(v21, flush=True)
    except v42:
        v71(v21.v113('ascii', 'replace').v106('ascii'), flush=True)
    v3.v43(parents=True, exist_ok=True)
    with v7.v72('a', encoding='utf-8') as v44:
        v44.v73(v21 + '\n')

def journal(v22: v14) -> None:
    v3.v43(parents=True, exist_ok=True)
    with v6.v72('a', encoding='utf-8') as v44:
        v44.v73(v96.v75({'ts': v46(), **v22}, ensure_ascii=False) + '\n')

def load_state() -> v14:
    if v5.v45():
        return v96.v74(v5.v97(encoding='utf-8'))
    return {'started': v46(), 'hours_budget': 24.0, 'completed': [], 'failed': [], 't0': v49.v49(), 'next_job': None}

def save_state(v23: v14) -> None:
    v23['updated'] = v46()
    v3.v43(parents=True, exist_ok=True)
    v5.v47(v96.v75(v23, indent=2), encoding='utf-8')

def elapsed_h(v23: v14) -> v15:
    return (v49.v49() - v23['t0']) / 3600.0

def remaining_h(v23: v14) -> v15:
    return v23['hours_budget'] - v76(v23)

def read_overall(v24: v17) -> v13 | None:
    v25 = v0 / 'results' / f'stage{v24}_decision.json'
    if not v25.v45():
        return None
    try:
        return v96.v74(v25.v97(encoding='utf-8')).v77('overall')
    except v48:
        return None

def wait_phase1(v26: v15=14.0) -> None:
    v27 = v49.v49()
    v50('waiting for phase1 (PIPELINE END or leftover done)...')
    while (v49.v49() - v27) / 3600.0 < v26:
        if v9.v45() or v10.v45():
            v50('STOP seen while waiting')
            return
        v51 = v2 / 'master.log'
        if v51.v45():
            v78 = v51.v97(encoding='utf-8', errors='ignore')[-4000:]
            if 'PIPELINE END' in v78:
                v50('phase1 PIPELINE END detected')
                return
        if v4.v45():
            v79 = v96.v74(v4.v97(encoding='utf-8'))
            v80 = [v13(v107) for v107 in v79.v77('completed') or []]
            v81 = v79.v77('next_job')
            v82 = v98(('248_leftover' in v107 for v107 in v80))
            if v82 and (not v81):
                v50('phase1 leftover complete, idle')
                return
            if v81:
                v50(f'phase1 busy: {v81}')
        v49.v83(90)
    v50('wait timeout - starting day2 anyway')

def run_job(v23: v14, v28: v13, v29: v84[v13], v30: v15, v31: v13) -> v16:
    if v9.v45():
        v50('STOP - abort ' + v28)
        return False
    if v28 in v23['completed']:
        v50('skip done ' + v28)
        return True
    if v61(v23) < 0.05:
        v50('budget done')
        return False
    v50(f'START {v28} est~{v30:.1f}h remain~{v61(v23):.1f}h | {v31}')
    v52({'event': 'start', 'job': v28, 'argv': v29, 'why': v31})
    v23['next_job'] = v28
    v53(v23)
    v27 = v49.v49()
    try:
        v54 = v99.v85([v11, *v29], cwd=v13(v0), check=False)
        v55 = (v49.v49() - v27) / 3600.0
        v56 = v54.v86 == 0
        v57 = None
        for v58 in v29:
            if v58.v108('_stage') and v58.v109('.py'):
                v100 = ''.v102((v110 for v110 in v58.v116('_')[1] if v110.v115()))
                if v100:
                    v57 = v60(v17(v100))
                break
        v52({'event': 'end', 'job': v28, 'ok': v56, 'wall_h': v55, 'overall': v57, 'code': v54.v86})
        if v56:
            v23['completed'].v87(v28)
            v50(f'OK {v28} wall={v55:.2f}h overall={v57}')
        else:
            v23['failed'].v87({'job': v28, 'code': v54.v86})
            v50(f'FAIL {v28} code={v54.v86}')
        v23['next_job'] = None
        v53(v23)
        return v56
    except v48 as e:
        v52({'event': 'exception', 'job': v28, 'error': v13(v111), 'tb': v114.v112()})
        v23['failed'].v87({'job': v28, 'error': v13(v111)})
        v53(v23)
        v50(f'EXC {v28}: {v111}')
        return False

def ladder(v23: v14) -> None:
    v59(v23, '250_100k', ['_stage250_masked_night.py', '--steps', '100000', '--resume'], 9.0, 'masked-only night 100k from 248 ckpt')
    v32 = v60(250)
    v52({'event': 'decision', 'after': '250_100k', 'overall': v32})
    if v61(v23) < 0.5 or v9.v45():
        return
    if v32 and 'NO' not in v32 and (v61(v23) > 6):
        v59(v23, '250_120k', ['_stage250_masked_night.py', '--steps', '120000', '--resume'], 11.0, 'continue masked night 120k')
    elif v61(v23) > 6:
        v59(v23, '250_60k_retry', ['_stage250_masked_night.py', '--steps', '60000'], 6.0, '250 weak - remask from P1 60k')
    if v61(v23) < 0.5 or v9.v45():
        return
    v59(v23, '249_day2', ['_stage249_hop_stream.py', '--steps', '12000'], 0.4, 'hop admission stress')
    v59(v23, '247_day2', ['_stage247_ingest_forks.py'], 0.3, 'reconfirm fork map')
    if v61(v23) > 2.5:
        v59(v23, '246_20k_day2', ['_stage246_domain_curriculum.py', '--steps', '20000'], 2.5, 'curriculum mid-scale retention check')
    v33 = v61(v23)
    if v33 > 1.0:
        v62 = v17(v101(150000, v33 * 10000))
        if v62 >= 8000:
            v59(v23, f'250_fill_{v62}', ['_stage250_masked_night.py', '--steps', v13(v62), '--resume'], v33 - 0.15, f'burn remaining ~{v33:.1f}h ({v62} steps)')

def write_report(v23: v14) -> None:
    v34 = ['# Day2 ingest-forks report', f"- wall_h: {v76(v23):.2f} / {v23['hours_budget']}", f"- completed: {v23.v77('completed')}", f"- failed: {v23.v77('failed')}", '', '## Verdicts']
    for v35 in (247, 248, 249, 246, 250):
        v34.v87(f'- {v35}: `{v60(v35)}`')
    v34 += ['', '## Branch', 'Understanding via masked CE (250); knowledge in hop-gated slots; lenses via 246.']
    v8.v47('\n'.v102(v34) + '\n', encoding='utf-8')
    v50('wrote ' + v13(v8))

def main() -> v17:
    v36 = v88.v63()
    v36.v64('--hours', type=v15, default=24.0)
    v36.v64('--now', action='store_true')
    v36.v64('--resume', action='store_true')
    v37 = v36.v65()
    v3.v43(parents=True, exist_ok=True)
    if not v37.v66:
        v89()
    if v37.v67 and v5.v45():
        v23 = v90()
        v23['hours_budget'] = v37.v68
        v50(f'RESUME day2 elapsed={v76(v23):.2f}h')
    else:
        if v9.v45():
            v9.v103()
        v7.v47('', encoding='utf-8')
        v6.v47('', encoding='utf-8')
        v23 = v90()
        v23['hours_budget'] = v37.v68
        v23['t0'] = v49.v49()
        v23['started'] = v46()
        v23['completed'] = []
        v23['failed'] = []
        v53(v23)
        v50(f'START day2 hours={v37.v68}')
    v52({'event': 'day2_start', 'hours': v37.v68})
    try:
        v91(v23)
    except v69:
        v50('interrupt')
    except v48 as e:
        v50(f'day2 exception: {v111}')
        v52({'event': 'exception', 'error': v13(v111), 'tb': v114.v112()})
    finally:
        v92(v23)
        v53(v23)
        v50(f'DAY2 END wall={v76(v23):.2f}h')
    return 0
if v38 == '__main__':
    raise v70(v93())