"""Case-fold fix: smoke first; full+noprobe only if probe_hit_rate > 0."""
from __future__ import annotations
import json
import math
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

def latest_decision() -> Path | None:
    files = sorted(RES.glob('stage282_decision*.json'), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

def main() -> int:
    log('=== 282 case-fold restart ===')
    code = run([sys.executable, '_stage282_mind.py', '--smoke'], RES / '_stage282_smoke.out')
    dec = latest_decision()
    if code != 0 or dec is None:
        log(f'smoke hard fail exit={code} dec={dec}')
        return code or 2
    d = json.loads(dec.read_text(encoding='utf-8'))
    h = d.get('held_out') or {}
    phr = h.get('probe_hit_rate')
    log(f"smoke overall={d.get('overall')} probe_hit_rate={phr} mean_probes_tie={(h.get('tie') or {}).get('mean_probes')} conflict_when_tie={h.get('conflict_when_tie')}")
    log(f"gates={json.dumps(d.get('gates'), ensure_ascii=True)}")
    (RES / 'stage282_decision_smoke_casefold.json').write_text(json.dumps(d, indent=2), encoding='utf-8')
    if phr is None or (isinstance(phr, float) and (math.isnan(phr) or phr <= 0.0)):
        log('probe_hit_rate still 0/NaN — STOP before full')
        return 3
    log('probe_hit_rate > 0 — launching full')
    run([sys.executable, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2'], RES / '_stage282_full_m2.out')
    dec = latest_decision()
    if dec:
        (RES / 'stage282_decision_full_m2_casefold.json').write_text(dec.read_text(encoding='utf-8'), encoding='utf-8')
    log('launching --no-probe')
    run([sys.executable, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2', '--no-probe'], RES / '_stage282_full_m2_noprobe.out')
    dec = latest_decision()
    if dec:
        (RES / 'stage282_decision_full_m2_noprobe_casefold.json').write_text(dec.read_text(encoding='utf-8'), encoding='utf-8')
    log('=== done ===')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())