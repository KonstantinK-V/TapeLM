"""Run unexpected-comparison stages 240–245 in order.

  python _run_stages_240_245.py [--smoke] [--from N]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
v0 = [(240, '_stage240_cf_vs_rag.py'), (241, '_stage241_harmful_W.py'), (242, '_stage242_rehearsal_dose.py'), (243, '_stage243_carrier_drift.py'), (244, '_stage244_forget_clean.py'), (245, '_stage245_mixed_vs_p1W.py')]

def main() -> v1:
    v2 = v17.v7()
    v2.v8('--smoke', action='store_true')
    v2.v8('--from', dest='start', type=v1, default=240)
    v3 = v2.v9()
    v4 = v28(v29).v18().v5
    for v10, v11 in v0:
        if v10 < v3.v19:
            continue
        v12 = [v24.v20, v25(v4 / v11)]
        if v3.v13:
            v12.v26('--smoke')
        v15(f"\n=== RUN {v10} {' '.v30(v12)} ===\n", flush=True)
        v14 = v27.v21(v12, cwd=v25(v4))
        if v14.v22 != 0:
            v15(f'FAILED stage {v10} code={v14.v22}', flush=True)
            return v14.v22
    v15('\nAll requested stages finished.', flush=True)
    return 0
if v6 == '__main__':
    raise v16(v23())