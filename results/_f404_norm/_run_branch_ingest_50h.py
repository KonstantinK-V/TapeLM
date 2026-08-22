"""
Autonomous ~45h pipeline for the TapeLM ingest-forks branch (247+).

Runs without an interactive agent: executes jobs, reads verdicts, chooses next
steps, writes journals under results/branch_ingest_50h/.

  python _run_branch_ingest_50h.py [--hours 45] [--resume]

Stop: write results/branch_ingest_50h/STOP (or Ctrl+C — state is saved).
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
STATE = OUT / 'state.json'
JOURNAL = OUT / 'journal.jsonl'
MASTER = OUT / 'master.log'
PLAN = OUT / 'PLAN.md'
REPORT = OUT / 'FINAL_REPORT.md'
STOP = OUT / 'STOP'
PY = sys.executable

def utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str) -> None:
    safe = msg.replace('≈', '~').replace('—', '-').replace('–', '-').replace('→', '->').replace('…', '...')
    line = f'[{utc()}] {safe}'
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with MASTER.open('a', encoding='utf-8') as f:
        f.write(f'[{utc()}] {msg}\n')

def journal(event: dict) -> None:
    event = {'ts': utc(), **event}
    with JOURNAL.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'started': utc(), 'hours_budget': 45.0, 'completed': [], 'failed': [], 'skipped': [], 'decisions': [], 'next_job': None, 't0': time.time()}

def save_state(st: dict) -> None:
    st['updated'] = utc()
    STATE.write_text(json.dumps(st, indent=2), encoding='utf-8')

def elapsed_h(st: dict) -> float:
    return (time.time() - st['t0']) / 3600.0

def remaining_h(st: dict) -> float:
    return st['hours_budget'] - elapsed_h(st)

def read_overall(stage: int | str) -> str | None:
    p = ROOT / 'results' / f'stage{stage}_decision.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8')).get('overall')
    except Exception:
        return None

def read_decision(stage: int | str) -> dict | None:
    p = ROOT / 'results' / f'stage{stage}_decision.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def run_job(st: dict, job_id: str, argv: list[str], est_h: float, why: str) -> bool:
    if STOP.exists():
        log('STOP file present — abort before job ' + job_id)
        return False
    if remaining_h(st) < 0.05:
        log('budget exhausted — skip ' + job_id)
        st['skipped'].append(job_id)
        save_state(st)
        return False
    if job_id in st['completed']:
        log(f'already done {job_id} — skip')
        return True
    log(f'START {job_id} est~{est_h:.1f}h remain~{remaining_h(st):.1f}h | {why}')
    journal({'event': 'start', 'job': job_id, 'argv': argv, 'est_h': est_h, 'why': why})
    st['next_job'] = job_id
    save_state(st)
    t0 = time.time()
    try:
        r = subprocess.run([PY, *argv], cwd=str(ROOT), check=False)
        wall = (time.time() - t0) / 3600.0
        ok = r.returncode == 0
        entry = {'event': 'end', 'job': job_id, 'ok': ok, 'returncode': r.returncode, 'wall_h': wall}
        for tok in argv:
            if tok.startswith('_stage') and tok.endswith('.py'):
                tag = tok.replace('_stage', '').split('_')[0]
                if tag.isdigit():
                    entry['overall'] = read_overall(int(tag))
                    entry['decision'] = read_decision(int(tag))
                break
        journal(entry)
        if ok:
            st['completed'].append(job_id)
            log(f"OK {job_id} wall={wall:.2f}h overall={entry.get('overall')}")
        else:
            st['failed'].append({'job': job_id, 'code': r.returncode, 'wall_h': wall})
            log(f'FAIL {job_id} code={r.returncode} wall={wall:.2f}h')
        st['next_job'] = None
        save_state(st)
        return ok
    except Exception as e:
        journal({'event': 'exception', 'job': job_id, 'error': str(e), 'tb': traceback.format_exc()})
        st['failed'].append({'job': job_id, 'error': str(e)})
        save_state(st)
        log(f'EXCEPTION {job_id}: {e}')
        return False

def write_plan(hours: float) -> None:
    PLAN.write_text(f'# Branch ingest-forks ~{hours:.0f}h autonomous plan\n\nStarted from 247 `INGEST_FORK_SLOTS_AND_HOP`: bindings→tape, CE without bindings, hop as admission.\n\n## Job ladder (adaptive)\n\n1. **Confirm 247** — second seed / re-run if needed (~0.2h)\n2. **249 hop stream** — precision of hop admission (~0.2–1h)\n3. **248 masked understand** — 4k then 12k then 20k steps if gates hold (~3–12h)\n4. **246 curriculum** — 10k/domain; if wiki mem holds + GPT PPL rises → **30k/domain** (~8h + ~20h)\n5. **248 mega** — remaining budget into masked CE night (up to 25k)\n6. **Multi-seed 247** — seeds 2471/2472 if time (~0.5h)\n7. **FINAL_REPORT.md** — synthesis for the branch\n\n## Stop\n\n`echo stop > results/branch_ingest_50h/STOP`\n\n## Logs\n\n- `master.log` — human timeline\n- `journal.jsonl` — machine events + verdicts\n- `state.json` — resume pointer\n', encoding='utf-8')

def decide_and_run(st: dict) -> None:
    """Main adaptive loop — decisions encoded here."""
    hours = st['hours_budget']
    o247 = read_overall(247)
    if o247 is None or 'SLOTS' not in (o247 or ''):
        run_job(st, '247_rerun', ['_stage247_ingest_forks.py'], 0.25, '247 missing/weak — re-run full')
        o247 = read_overall(247)
    else:
        log(f'247 already {o247} — keep')
        journal({'event': 'decision', 'decision': 'keep_247', 'overall': o247})
    if STOP.exists() or remaining_h(st) < 0.1:
        return
    run_job(st, '249_hop', ['_stage249_hop_stream.py'], 0.3, 'map hop admission precision')
    o249 = read_overall(249)
    journal({'event': 'decision', 'after': '249', 'overall': o249, 'next': '248_masked_4k'})
    if STOP.exists() or remaining_h(st) < 0.2:
        return
    run_job(st, '248_4k', ['_stage248_masked_understand.py', '--steps', '4000'], 2.5, '247 fork: grow understanding without binding CE')
    d248 = read_decision(248) or {}
    o248 = d248.get('overall')
    mem_cf = (d248.get('masked') or {}).get('mem_after_code_W', 0)
    nt = (d248.get('masked') or {}).get('next_tok', 0)
    if o248 and 'NO' not in o248 and (remaining_h(st) > 4):
        steps_mid = 12000
        run_job(st, '248_12k', ['_stage248_masked_understand.py', '--steps', str(steps_mid)], 6.0, f'248 {o248} mem_cf={mem_cf} nt={nt} — scale masked CE')
        d248 = read_decision(248) or d248
        o248 = d248.get('overall')
    else:
        journal({'event': 'decision', 'skip': '248_12k', 'reason': o248})
    if remaining_h(st) > 8 and o248 and ('OK' in (o248 or '')):
        run_job(st, '248_20k', ['_stage248_masked_understand.py', '--steps', '20000'], 10.0, 'masked CE night — deepen understanding arm')
    if STOP.exists() or remaining_h(st) < 1:
        return
    run_job(st, '246_10k', ['_stage246_domain_curriculum.py', '--steps', '10000'], 8.0, 'multi-domain retention matrix at 10k/domain')
    d246 = read_decision(246) or {}
    drops = d246.get('drops_first_to_final') or {}
    wiki = drops.get('wiki') or {}
    tape_ok = (d246.get('matrix') or [{}])[-1].get('domains', {}).get('wiki', {}).get('tape_mem', 0) >= 0.7
    ppl_rise = wiki.get('gpt_ppl_rise', 0) >= 1.0
    journal({'event': 'decision', 'after': '246_10k', 'overall': d246.get('overall'), 'tape_wiki_ok': tape_ok, 'gpt_ppl_rise': wiki.get('gpt_ppl_rise'), 'escalate_30k': bool(tape_ok and ppl_rise and (remaining_h(st) > 12))})
    if tape_ok and ppl_rise and (remaining_h(st) > 12):
        run_job(st, '246_30k', ['_stage246_domain_curriculum.py', '--steps', '30000'], 22.0, 'escalate: wiki mem holds & GPT PPL rises at 10k → paper-scale 30k')
    elif remaining_h(st) > 12:
        run_job(st, '248_fill', ['_stage248_masked_understand.py', '--steps', '15000'], min(12.0, remaining_h(st) - 1), '246 did not clear escalate gates — burn budget on masked understand')
    if STOP.exists() or remaining_h(st) < 0.5:
        return
    run_job(st, '249_scaled', ['_stage249_hop_stream.py', '--steps', '8000'], 0.5, 'larger hop/uniform admission stress')
    if remaining_h(st) > 1.0:
        run_job(st, '247_confirm', ['_stage247_ingest_forks.py'], 0.3, 'confirm fork map still holds')
    rem = remaining_h(st)
    if rem > 2.0:
        extra = int(min(25000, rem * 3600 / 2.5))
        if extra >= 3000:
            run_job(st, f'248_leftover_{extra}', ['_stage248_masked_understand.py', '--steps', str(extra)], rem - 0.3, f'burn remaining ~{rem:.1f}h into masked CE ({extra} steps)')

def write_report(st: dict) -> None:
    lines = [f'# Branch ingest-forks final report', f'', f"- started: {st.get('started')}", f'- finished: {utc()}', f"- wall_h: {elapsed_h(st):.2f} / budget {st['hours_budget']}", f"- completed: {st.get('completed')}", f"- failed: {st.get('failed')}", f"- skipped: {st.get('skipped')}", f'', f'## Verdicts']
    for s in (247, 248, 249, 246):
        d = read_decision(s)
        if d:
            lines.append(f"- **{s}**: `{d.get('overall')}`")
        else:
            lines.append(f'- **{s}**: (missing)')
    lines += ['', '## Branch recommendation', '1. Keep unknown bindings out of CE (247/248).', '2. Use hop-similarity as slot admission (247/249), not as CE curriculum.', '3. Domain lenses (246) for eras; backbone understanding via masked CE.', '4. Escalate curriculum steps only when wiki mem holds and GPT PPL rises.', '', 'See journal.jsonl for decision trail.']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log('wrote ' + str(REPORT))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=45.0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.resume and STATE.exists():
        st = load_state()
        st['hours_budget'] = args.hours
        log(f"RESUME elapsed={elapsed_h(st):.2f}h completed={st.get('completed')}")
    else:
        if STOP.exists():
            STOP.unlink()
        st = load_state()
        st['hours_budget'] = args.hours
        st['t0'] = time.time()
        st['started'] = utc()
        st['completed'] = []
        st['failed'] = []
        st['skipped'] = []
        MASTER.write_text('', encoding='utf-8')
        JOURNAL.write_text('', encoding='utf-8')
        write_plan(args.hours)
        save_state(st)
        log(f'START pipeline hours={args.hours}')
    journal({'event': 'pipeline_start', 'hours': args.hours, 'resume': args.resume})
    try:
        decide_and_run(st)
    except KeyboardInterrupt:
        log('KeyboardInterrupt — saving state')
        journal({'event': 'interrupt'})
    except Exception as e:
        log(f'pipeline exception: {e}')
        journal({'event': 'pipeline_exception', 'error': str(e), 'tb': traceback.format_exc()})
    finally:
        write_report(st)
        save_state(st)
        log(f'PIPELINE END wall={elapsed_h(st):.2f}h')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())