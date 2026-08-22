"""Run unexpected-comparison stages 240–245 in order.

  python _run_stages_240_245.py [--smoke] [--from N]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
STAGES = [(240, '_stage240_cf_vs_rag.py'), (241, '_stage241_harmful_W.py'), (242, '_stage242_rehearsal_dose.py'), (243, '_stage243_carrier_drift.py'), (244, '_stage244_forget_clean.py'), (245, '_stage245_mixed_vs_p1W.py')]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--from', dest='start', type=int, default=240)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    for num, script in STAGES:
        if num < args.start:
            continue
        cmd = [sys.executable, str(root / script)]
        if args.smoke:
            cmd.append('--smoke')
        print(f"\n=== RUN {num} {' '.join(cmd)} ===\n", flush=True)
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            print(f'FAILED stage {num} code={r.returncode}', flush=True)
            return r.returncode
    print('\nAll requested stages finished.', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())