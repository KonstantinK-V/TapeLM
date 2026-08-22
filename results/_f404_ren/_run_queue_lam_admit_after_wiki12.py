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
v0 = v28(v29).v11().v1
v2 = v0 / 'results' / '_stage255_wiki12_full.out'
v3 = v0 / 'results' / '_stage255_wiki12_lam_full.out'
v4 = 180

def log(v7: v12) -> None:
    v8 = v7 if v7.v20('\n') else v7 + '\n'
    v13(v8, end='', flush=True)
    with (v0 / 'results' / '_run_queue_lam_admit.log').v21('a', encoding='utf-8') as v14:
        v14.v22(v8)

def wiki12_done() -> v5:
    if v2.v15():
        v16 = v2.v30(encoding='utf-8', errors='ignore')[-15000:]
        if 'schedule exhausted' in v16:
            return True
        if '"overall":' in v16 and 'STREAM_INGEST' in v16:
            return True
    for v9 in ('stage255_decision.json', 'stage255_decision_wiki12.json'):
        v17 = v0 / 'results' / v9
        if v17.v15():
            import json
            v23 = v35.v31(v17.v30(encoding='utf-8'))
            if v23.v36('summary', {}).v36('chunks', 0) >= 12:
                return True
    return False

def main() -> v6:
    v18('waiting for wiki:12 baseline (fixed lambda) to finish')
    while not v24():
        if v2.v15():
            v25 = [v32 for v32 in v2.v30(encoding='utf-8', errors='ignore').v37() if v32.v38()]
            v18(f"  wiki12 tail: {(v25[-1][:100] if v25 else '...')}")
        v33.v26(v4)
    if v3.v15() and v33.v33() - v3.v39().v34 < 7200:
        v18('skip: wiki12_lam log recently updated')
        return 0
    v18('wiki12 done — lambda-admit A/B paused (re-run baseline with fixed gates first)')
    v18('  manual: python _stage255_stream_ingest.py --schedule wiki:12 --chunk-lines 25000 --lambda-admit --run-tag wiki12_lam')
    return 0
if v10 == '__main__':
    raise v19(v27())