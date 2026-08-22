"""
After stage 254 finishes, run stage 255 single-domain wiki stream (12 chunks) on GPU.

  python _run_queue_wiki12_after_254.py

Watches results/_stage254_full.out for completion or stage254 matrix with 4 phases.
Then: python _stage255_stream_ingest.py --schedule wiki:12 --chunk-lines 25000
Logs to results/_stage255_wiki12_full.out
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
LOG254 = ROOT / 'results' / '_stage254_full.out'
DEC254 = ROOT / 'results' / 'stage254_decision.json'
OUT255 = ROOT / 'results' / '_stage255_wiki12_full.out'
POLL_S = 120

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    qlog = ROOT / 'results' / '_run_queue_wiki12_runner.log'
    try:
        with qlog.open('a', encoding='utf-8') as f:
            f.write(line)
    except OSError:
        pass

def stage254_done() -> bool:
    if DEC254.exists():
        import json
        try:
            d = json.loads(DEC254.read_text(encoding='utf-8'))
            if d.get('stage') == 254 and len(d.get('matrix') or []) >= 4:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    if LOG254.exists():
        tail = LOG254.read_text(encoding='utf-8', errors='ignore')[-12000:]
        if 'after news:' in tail and 'CONTINUAL' in tail:
            return True
    return False

def main() -> int:
    log(f'queue wiki:12 waiting for stage 254 (poll {POLL_S}s)')
    while not stage254_done():
        if LOG254.exists():
            lines = [ln for ln in LOG254.read_text(encoding='utf-8', errors='ignore').strip().splitlines() if ln.strip()]
            tail = (lines[-1][:120] if lines else '...').encode('ascii', 'replace').decode('ascii')
            log(f'  254 tail: {tail}')
        time.sleep(POLL_S)
    log('254 done — starting stage 255 wiki:12 on GPU')
    if OUT255.exists() and time.time() - OUT255.stat().st_mtime < 3600:
        log(f'skip launch: {OUT255.name} recently updated (another run may be active)')
        return 0
    cmd = [sys.executable, str(ROOT / '_stage255_stream_ingest.py'), '--schedule', 'wiki:12', '--chunk-lines', '25000', '--epochs-per-chunk', '1.0', '--replay-frac', '0.2', '--ckpt-every', '2', '--run-tag', 'wiki12']
    try:
        with OUT255.open('w', encoding='utf-8') as out:
            rc = subprocess.call(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    except PermissionError:
        log(f'skip launch: cannot write {OUT255.name} (file locked by another process)')
        return 0
    log(f'255 wiki:12 exit={rc}')
    return rc
if __name__ == '__main__':
    raise SystemExit(main())