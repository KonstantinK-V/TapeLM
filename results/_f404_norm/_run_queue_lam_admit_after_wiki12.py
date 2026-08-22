"""
After wiki:12 baseline stream finishes, run the same schedule with dynamic lambda-admit.

  python _run_queue_lam_admit_after_wiki12.py

Baseline is assumed to log to results/_stage255_wiki12_full.out (fixed lambda 0.2).
Comparison run: --run-tag wiki12_lam --lambda-admit -> results/_stage255_wiki12_lam_full.out
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
LOG_W12 = ROOT / 'results' / '_stage255_wiki12_full.out'
OUT_LAM = ROOT / 'results' / '_stage255_wiki12_lam_full.out'
POLL_S = 180

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    print(line, end='', flush=True)
    with (ROOT / 'results' / '_run_queue_lam_admit.log').open('a', encoding='utf-8') as f:
        f.write(line)

def wiki12_done() -> bool:
    if LOG_W12.exists():
        tail = LOG_W12.read_text(encoding='utf-8', errors='ignore')[-15000:]
        if 'schedule exhausted' in tail:
            return True
        if '"overall":' in tail and 'STREAM_INGEST' in tail:
            return True
    for name in ('stage255_decision.json', 'stage255_decision_wiki12.json'):
        p = ROOT / 'results' / name
        if p.exists():
            import json
            d = json.loads(p.read_text(encoding='utf-8'))
            if d.get('summary', {}).get('chunks', 0) >= 12:
                return True
    return False

def main() -> int:
    log('waiting for wiki:12 baseline (fixed lambda) to finish')
    while not wiki12_done():
        if LOG_W12.exists():
            lines = [ln for ln in LOG_W12.read_text(encoding='utf-8', errors='ignore').splitlines() if ln.strip()]
            log(f"  wiki12 tail: {(lines[-1][:100] if lines else '...')}")
        time.sleep(POLL_S)
    if OUT_LAM.exists() and time.time() - OUT_LAM.stat().st_mtime < 7200:
        log('skip: wiki12_lam log recently updated')
        return 0
    log('wiki12 done — lambda-admit A/B paused (re-run baseline with fixed gates first)')
    log('  manual: python _stage255_stream_ingest.py --schedule wiki:12 --chunk-lines 25000 --lambda-admit --run-tag wiki12_lam')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())