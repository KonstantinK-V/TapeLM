"""Attribution queue: 280 casefold control -> 282 two-witness full -> --no-probe."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'

def log(msg: str) -> None:
    print(msg, flush=True)

def run(cmd: list[str], out_path: Path) -> int:
    log(f"RUN {' '.join(cmd)}")
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    with out_path.open('w', encoding='utf-8', errors='replace') as f:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        return p.wait()

def snap_decision(glob_pat: str, dest: Path) -> None:
    files = sorted(RES.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    if not files:
        log(f'no decision matching {glob_pat}')
        return
    src = files[-1]
    dest.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    d = json.loads(dest.read_text(encoding='utf-8'))
    h = d.get('held_out') or {}
    log(f"saved {dest.name}: overall={d.get('overall')} rew={h.get('reward_total')} teach={h.get('teacher_reward_total') or d.get('teacher_ceiling_reward')} tie_abs={(h.get('tie') or {}).get('abstain')} probe={h.get('probe_hit_rate')}")

def main() -> int:
    log('=== 1/3 control 280 with casefold (same knobs as m2b) ===')
    code = run([sys.executable, '_stage280_raw_exam.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--hop', 'fp', '--min-mentions', '2', '--min-per-family', '8', '--run-tag', 'casefold'], RES / '_stage280_full_fp_casefold.out')
    snap_decision('stage280_decision_fp_casefold.json', RES / 'stage280_decision_fp_casefold.json')
    if code != 0:
        log(f'280 casefold exit={code} — continue to 282 anyway? stopping')
        return code
    log('=== 2/3 282 full two-witness ===')
    code = run([sys.executable, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2'], RES / '_stage282_full_m2_twowitness.out')
    snap_decision('stage282_decision.json', RES / 'stage282_decision_full_m2_twowitness.json')
    if code != 0:
        return code
    log('=== 3/3 282 --no-probe ===')
    code = run([sys.executable, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2', '--no-probe'], RES / '_stage282_full_m2_noprobe.out')
    nop = RES / 'stage282_decision_noprobe.json'
    if nop.exists():
        snap_decision('stage282_decision_noprobe.json', RES / 'stage282_decision_full_m2_noprobe_twowitness.json')
    else:
        files = sorted(RES.glob('stage282_decision*.json'), key=lambda p: p.stat().st_mtime)
        if files:
            (RES / 'stage282_decision_full_m2_noprobe_twowitness.json').write_text(files[-1].read_text(encoding='utf-8'), encoding='utf-8')
            log(f'saved noprobe from {files[-1].name}')
    log('=== attribution queue done ===')
    return code
if __name__ == '__main__':
    raise SystemExit(main())