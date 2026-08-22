"""How the tape's numbers move with its size - the dimension this project has never varied.

WHY. Everything measured here sits at ONE point: 30 MB, 1500 addresses, a 400-line window. Every
ceiling, every rate, every sentence that begins "the binding constraint is". We have never asked
which of those are asymptotic and which are artefacts of that single point, and three standing
conclusions are exactly the kind that must move with size:

  310  "composition failed and the corpus is at fault"
  324  "memory cannot fill"
  327  "0.807 of the truth is present and we reach 0.217 of it"

THE CURVE THAT DECIDES is the gap `anywhere - union`. If it NARROWS as the tape grows, 0.217 is a
small-scale illness and the cure is more corpus. If it WIDENS, retrieval falls behind knowledge
the faster the more knowledge there is, and an offer of eight enumerated candidates is the wrong
operation no matter how it is tuned - which puts the constraint interface back on the table by
asymptotics rather than by the trigger it was parked on.

Two sweeps, because the tape has two sizes and they are not the same knob:
  --addresses    how many places are kept - the tape's WIDTH at a fixed slice of corpus
  --window-lines how much text the region is drawn from - the tape's DEPTH in the corpus

Both are run through _audit327_presence unchanged, so every point is the same measurement the
single-point number came from. Torch-free; minutes per point.

    python _sweep335_scale.py
    python _sweep335_scale.py --quick        # three points per sweep instead of five
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
v0 = v4('results/_stage335_scale.json')
v1 = v4('results/_stage327_presence.json')
v2 = ('own', 'walk_k', 'walk_2k', 'shared', 'union_k_shared', 'anywhere', 'present_unreached', 'reach_over_presence', 'places', 'questions')

def one(v5, v6, v7):
    """One point of the sweep: run the audit, read what it wrote."""
    v8 = [v60.v43, '_audit327_presence.py', '--addresses', v53(v5), '--window-lines', v53(v6)] + v7
    v9 = v44.v25(v8, capture_output=True, text=True)
    if v9.v45 != 0 or not v1.v61():
        v28(f'  FAILED addresses={v5} window={v6}\n{v9.v62[-400:]}{v9.v63[-400:]}')
        return None
    v10 = v46.v26(v1.v47(encoding='utf-8'))
    return {v27: v10.v48(v27) for v27 in v2}

def show(v11, v12, v13):
    v28(f'\n{v11}')
    v28(f"  {v12:>8}  {'own':>7} {'walkK':>7} {'union':>7} {'anywhere':>9} {'GAP':>7} {'reached':>8}")
    for v29, v10 in v13:
        if v10 is None:
            continue
        v30 = v10['anywhere'] - v10['union_k_shared']
        v28(f"  {v29:>8}  {v10['own']:7.4f} {v10['walk_k']:7.4f} {v10['union_k_shared']:7.4f} {v10['anywhere']:9.4f} {v30:7.4f} {v10['reach_over_presence']:8.4f}")
    v14 = [(v29, v10['anywhere'] - v10['union_k_shared']) for v29, v10 in v13 if v10]
    if v49(v14) >= 2:
        v50, v51 = (v14[0], v14[-1])
        v31 = v51[1] - v50[1]
        v32 = 'NARROWS - the offer catches up as the tape grows' if v31 < -0.01 else 'WIDENS - retrieval falls behind knowledge' if v31 > 0.01 else 'FLAT - the gap is scale-invariant'
        v28(f'  gap {v50[1]:.4f} at {v50[0]} -> {v51[1]:.4f} at {v51[0]}   {v32}')

def main() -> v3:
    v15 = v52.v33()
    v15.v34('--quick', action='store_true')
    v15.v34('--bytes', type=v3, default=30000000)
    v15.v34('--frame-max', type=v3, default=3)
    v15.v34('--seed', type=v3, default=1337)
    v16 = v15.v35()
    v7 = ['--bytes', v53(v16.v37), '--frame-max', v53(v16.v54), '--sample', 'region', '--seed', v53(v16.v38)]
    v17 = [750, 1500, 3000] if v16.v36 else [375, 750, 1500, 3000, 6000]
    v18 = [200, 400, 800] if v16.v36 else [100, 200, 400, 800, 1600]
    v19 = {'bytes': v16.v37, 'seed': v16.v38, 'addresses': {}, 'window_lines': {}}
    v20 = []
    for v21 in v17:
        v28(f'addresses={v21} ...', flush=True)
        v10 = v55(v21, 400, v7)
        v19['addresses'][v53(v21)] = v10
        v20.v56((v21, v10))
    v22 = []
    for v23 in v18:
        v28(f'window={v23} ...', flush=True)
        v10 = v55(1500, v23, v7)
        v19['window_lines'][v53(v23)] = v10
        v22.v56((v23, v10))
    v0.v57.v39(parents=True, exist_ok=True)
    v0.v40(v46.v58(v19, indent=1), encoding='utf-8')
    v41('WIDTH - how many places are kept (window 400)', 'addresses', v20)
    v41('DEPTH - how much text the region is drawn from (1500 places)', 'window', v22)
    v28(f'\nGAP = anywhere - union: what is on the tape and not in front of the mind.')
    v28(f'written to {v0}')
    return 0
if v24 == '__main__':
    raise v42(v59())