"""CAN A PLACE BE MADE THICK? The two knobs that move mentions-per-place, swept together.

WHY. 346 measured a place at 4.24 MENTIONS. One is hidden, so the mind sees three values and can
form three lens pairs, of which 1.7 are non-empty. 84% of co-occurring pairs are seen EXACTLY
ONCE. Every second-order operation this project has tried - composition (310), enumeration at
scale (335), the constraint (345) - has been computing a statistic over about four samples. No
statistic works on four samples, and that is one sentence for three separate failures.

AND THE CORPUS TEST FOR IT DID NOT HAPPEN. 30 MB gave 4.24 mentions per place; 120 MB gave 3.99.
The tape is built from a --window-lines region, so the window is 400 lines whatever the corpus
is: more bytes only move where the window can land. The knob never moved the quantity - the same
shape of fault as the dead --addresses flag in 335, caught this time by printing the quantity
next to the knob.

THE TWO KNOBS THAT DO MOVE IT, and they are different in kind:

  --window-lines   thickness bought with MORE TEXT. A place accumulates mentions from a wider
                   region, so recurrence rises without changing what a place IS.
  --frame-max      thickness bought with a LOOSER DEFINITION of a place. A frame matching on 3
                   tokens each side recurs rarely by construction; 2 or 1 recurs far more often,
                   but the places mean less. This is the write path (342d) - counting, so the
                   invariant is untouched - and it is the first time it has been swept.

WHAT TO READ, in this order:
  mentions_per_place   did the knob move the quantity at all. If not, stop reading.
  support_2plus        the share of co-occurring pairs seen more than once. A distribution of
                       singletons has no peak, and no rule for taking a maximum will invent one.
  one_present_topm     what a lens could reach at a matched offer - the payoff of thickness.
  one_count_right      what it actually resolves to. The gap between these two is the illness.

A LOOSER FRAME BUYS THICKNESS AT THE COST OF MEANING, so the two knobs must be read apart: if
--window-lines raises support and --frame-max does not, thickness is real; if only --frame-max
does, we have made places bigger and emptier and the numbers will say so through
one_present_topm failing to follow.

    python _sweep347_thick.py
    python _sweep347_thick.py --quick
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
v0 = v4('results/_stage347_thick.json')
v1 = v4('results/_stage346_lens.json')
v2 = ('places', 'mentions_per_place', 'support_1', 'support_2plus', 'support_3plus', 'one_present_topm', 'one_count_right', 'pair_present_topm', 'mean_lens_offer', 'in_own', 'questions')

def one(v5):
    v6 = v46.v23([v63.v61, '_audit346_lens.py'] + v5, capture_output=True, text=True)
    if v6.v47 != 0 or not v1.v62():
        v26(f"  FAILED {' '.v64(v5)}\n{v6.v65[-300:]}{v6.v66[-300:]}")
        return None
    v7 = v48.v24(v1.v49(encoding='utf-8'))
    return {v25: v7.v50(v25) for v25 in v2}

def show(v8, v9, v10):
    v26(f'\n{v8}')
    v26(f"  {v9:>8} {'places':>7} {'ment/pl':>8} {'sup2+':>7} {'sup3+':>7} {'offer':>7} {'present@8':>10} {'argmax':>8} {'pair@8':>8}")
    for v27, v7 in v10:
        if v7 is None:
            continue
        v26(f"  {v27:>8} {v7['places']:>7} {v7['mentions_per_place']:8.2f} {v7['support_2plus']:7.4f} {v7['support_3plus']:7.4f} {v7['mean_lens_offer']:7.1f} {v7['one_present_topm']:10.4f} {v7['one_count_right']:8.4f} {v7['pair_present_topm']:8.4f}")
    v11 = [(v27, v7) for v27, v7 in v10 if v7]
    if v51(v11) < 2:
        return
    v28, v29 = (v11[0][1]['mentions_per_place'], v11[-1][1]['mentions_per_place'])
    if v52(v29 - v28) < 0.25:
        v26(f'  DEAD KNOB: mentions per place {v28:.2f} -> {v29:.2f}. This knob does not thicken a place, so nothing below it is a test of thickness.')
        return
    v30, v31 = (v11[0][1]['support_2plus'], v11[-1][1]['support_2plus'])
    v32, v33 = (v11[0][1]['one_present_topm'], v11[-1][1]['one_present_topm'])
    v34, v35 = (v11[0][1]['one_count_right'], v11[-1][1]['one_count_right'])
    v26(f'  thickness {v28:.2f} -> {v29:.2f}   support2+ {v30:.4f} -> {v31:.4f}   present@8 {v32:.4f} -> {v33:.4f}   argmax {v34:.4f} -> {v35:.4f}')
    if v31 > v30 + 0.02 and v35 > v34 + 0.01:
        v26('  THICKNESS PAYS: the distribution stops being singletons AND the resolved answer follows. The substrate was the constraint, and the write path is the lever this project has been looking for.')
    elif v31 > v30 + 0.02:
        v26('  THICKER BUT NOT BETTER: support rises and the resolved answer does not. The places got bigger without getting more meaningful - which is what a looser frame would do, and it is not thickness in the sense that matters.')
    else:
        v26('  NO: the knob moves mentions per place and the distribution stays singletons.')

def main() -> v3:
    v12 = v53.v36()
    v12.v37('--quick', action='store_true')
    v12.v37('--bytes', type=v3, default=30000000)
    v12.v37('--addresses', type=v3, default=1500)
    v12.v37('--seed', type=v3, default=1337)
    v13 = v12.v38()
    v14 = ['--bytes', v54(v13.v40), '--addresses', v54(v13.v41), '--seed', v54(v13.v55)]
    v15 = [400, 1600] if v13.v39 else [400, 800, 1600, 3200, 6400]
    v16 = [3, 2] if v13.v39 else [3, 2, 1]
    v17 = {'bytes': v13.v40, 'addresses': v13.v41, 'window': {}, 'frame_max': {}}
    v18 = []
    for v19 in v15:
        v26(f'window={v19} ...', flush=True)
        v7 = v56(v14 + ['--window-lines', v54(v19), '--frame-max', '3'])
        v17['window'][v54(v19)] = v7
        v18.v57((v19, v7))
    v20 = []
    for v21 in v16:
        v26(f'frame_max={v21} ...', flush=True)
        v7 = v56(v14 + ['--window-lines', '400', '--frame-max', v54(v21)])
        v17['frame_max'][v54(v21)] = v7
        v20.v57((v21, v7))
    v0.v58.v42(parents=True, exist_ok=True)
    v0.v43(v48.v59(v17, indent=1), encoding='utf-8')
    v44('MORE TEXT - a wider region, the place unchanged', 'window', v18)
    v44('A LOOSER PLACE - the write path, same text', 'frame_max', v20)
    v26(f'\nwritten to {v0}')
    return 0
if v22 == '__main__':
    raise v45(v60())