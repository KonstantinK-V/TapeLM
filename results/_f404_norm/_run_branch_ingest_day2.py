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
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'branch_ingest_50h'
DAY2 = OUT / 'day2'
STATE1 = OUT / 'state.json'
STATE = DAY2 / 'state.json'
JOURNAL = DAY2 / 'journal.jsonl'
MASTER = DAY2 / 'master.log'
REPORT = DAY2 / 'FINAL_REPORT.md'
STOP = DAY2 / 'STOP'
STOP1 = OUT / 'STOP'
PY = sys.executable

def utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str) -> None:
    safe = msg.replace('≈', '~').replace('—', '-').replace('–', '-').replace('→', '->')
    line = f'[{utc()}] {safe}'
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), flush=True)
    DAY2.mkdir(parents=True, exist_ok=True)
    with MASTER.open('a', encoding='utf-8') as f:
        f.write(line + '\n')

def journal(ev: dict) -> None:
    DAY2.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'ts': utc(), **ev}, ensure_ascii=False) + '\n')

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'started': utc(), 'hours_budget': 24.0, 'completed': [], 'failed': [], 't0': time.time(), 'next_job': None}

def save_state(st: dict) -> None:
    st['updated'] = utc()
    DAY2.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding='utf-8')

def elapsed_h(st: dict) -> float:
    return (time.time() - st['t0']) / 3600.0

def remaining_h(st: dict) -> float:
    return st['hours_budget'] - elapsed_h(st)

def read_overall(stage: int) -> str | None:
    p = ROOT / 'results' / f'stage{stage}_decision.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8')).get('overall')
    except Exception:
        return None

def wait_phase1(max_wait_h: float=14.0) -> None:
    t0 = time.time()
    log('waiting for phase1 (PIPELINE END or leftover done)...')
    while (time.time() - t0) / 3600.0 < max_wait_h:
        if STOP.exists() or STOP1.exists():
            log('STOP seen while waiting')
            return
        master = OUT / 'master.log'
        if master.exists():
            tail = master.read_text(encoding='utf-8', errors='ignore')[-4000:]
            if 'PIPELINE END' in tail:
                log('phase1 PIPELINE END detected')
                return
        if STATE1.exists():
            st1 = json.loads(STATE1.read_text(encoding='utf-8'))
            done = [str(x) for x in st1.get('completed') or []]
            nxt = st1.get('next_job')
            leftover_done = any(('248_leftover' in x for x in done))
            if leftover_done and (not nxt):
                log('phase1 leftover complete, idle')
                return
            if nxt:
                log(f'phase1 busy: {nxt}')
        time.sleep(90)
    log('wait timeout - starting day2 anyway')

def run_job(st: dict, job_id: str, argv: list[str], est_h: float, why: str) -> bool:
    if STOP.exists():
        log('STOP - abort ' + job_id)
        return False
    if job_id in st['completed']:
        log('skip done ' + job_id)
        return True
    if remaining_h(st) < 0.05:
        log('budget done')
        return False
    log(f'START {job_id} est~{est_h:.1f}h remain~{remaining_h(st):.1f}h | {why}')
    journal({'event': 'start', 'job': job_id, 'argv': argv, 'why': why})
    st['next_job'] = job_id
    save_state(st)
    t0 = time.time()
    try:
        r = subprocess.run([PY, *argv], cwd=str(ROOT), check=False)
        wall = (time.time() - t0) / 3600.0
        ok = r.returncode == 0
        overall = None
        for tok in argv:
            if tok.startswith('_stage') and tok.endswith('.py'):
                digits = ''.join((ch for ch in tok.split('_')[1] if ch.isdigit()))
                if digits:
                    overall = read_overall(int(digits))
                break
        journal({'event': 'end', 'job': job_id, 'ok': ok, 'wall_h': wall, 'overall': overall, 'code': r.returncode})
        if ok:
            st['completed'].append(job_id)
            log(f'OK {job_id} wall={wall:.2f}h overall={overall}')
        else:
            st['failed'].append({'job': job_id, 'code': r.returncode})
            log(f'FAIL {job_id} code={r.returncode}')
        st['next_job'] = None
        save_state(st)
        return ok
    except Exception as e:
        journal({'event': 'exception', 'job': job_id, 'error': str(e), 'tb': traceback.format_exc()})
        st['failed'].append({'job': job_id, 'error': str(e)})
        save_state(st)
        log(f'EXC {job_id}: {e}')
        return False

def ladder(st: dict) -> None:
    run_job(st, '250_100k', ['_stage250_masked_night.py', '--steps', '100000', '--resume'], 9.0, 'masked-only night 100k from 248 ckpt')
    o250 = read_overall(250)
    journal({'event': 'decision', 'after': '250_100k', 'overall': o250})
    if remaining_h(st) < 0.5 or STOP.exists():
        return
    if o250 and 'NO' not in o250 and (remaining_h(st) > 6):
        run_job(st, '250_120k', ['_stage250_masked_night.py', '--steps', '120000', '--resume'], 11.0, 'continue masked night 120k')
    elif remaining_h(st) > 6:
        run_job(st, '250_60k_retry', ['_stage250_masked_night.py', '--steps', '60000'], 6.0, '250 weak - remask from P1 60k')
    if remaining_h(st) < 0.5 or STOP.exists():
        return
    run_job(st, '249_day2', ['_stage249_hop_stream.py', '--steps', '12000'], 0.4, 'hop admission stress')
    run_job(st, '247_day2', ['_stage247_ingest_forks.py'], 0.3, 'reconfirm fork map')
    if remaining_h(st) > 2.5:
        run_job(st, '246_20k_day2', ['_stage246_domain_curriculum.py', '--steps', '20000'], 2.5, 'curriculum mid-scale retention check')
    rem = remaining_h(st)
    if rem > 1.0:
        extra = int(min(150000, rem * 10000))
        if extra >= 8000:
            run_job(st, f'250_fill_{extra}', ['_stage250_masked_night.py', '--steps', str(extra), '--resume'], rem - 0.15, f'burn remaining ~{rem:.1f}h ({extra} steps)')

def write_report(st: dict) -> None:
    lines = ['# Day2 ingest-forks report', f"- wall_h: {elapsed_h(st):.2f} / {st['hours_budget']}", f"- completed: {st.get('completed')}", f"- failed: {st.get('failed')}", '', '## Verdicts']
    for s in (247, 248, 249, 246, 250):
        lines.append(f'- {s}: `{read_overall(s)}`')
    lines += ['', '## Branch', 'Understanding via masked CE (250); knowledge in hop-gated slots; lenses via 246.']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log('wrote ' + str(REPORT))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=24.0)
    ap.add_argument('--now', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    DAY2.mkdir(parents=True, exist_ok=True)
    if not args.now:
        wait_phase1()
    if args.resume and STATE.exists():
        st = load_state()
        st['hours_budget'] = args.hours
        log(f'RESUME day2 elapsed={elapsed_h(st):.2f}h')
    else:
        if STOP.exists():
            STOP.unlink()
        MASTER.write_text('', encoding='utf-8')
        JOURNAL.write_text('', encoding='utf-8')
        st = load_state()
        st['hours_budget'] = args.hours
        st['t0'] = time.time()
        st['started'] = utc()
        st['completed'] = []
        st['failed'] = []
        save_state(st)
        log(f'START day2 hours={args.hours}')
    journal({'event': 'day2_start', 'hours': args.hours})
    try:
        ladder(st)
    except KeyboardInterrupt:
        log('interrupt')
    except Exception as e:
        log(f'day2 exception: {e}')
        journal({'event': 'exception', 'error': str(e), 'tb': traceback.format_exc()})
    finally:
        write_report(st)
        save_state(st)
        log(f'DAY2 END wall={elapsed_h(st):.2f}h')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())