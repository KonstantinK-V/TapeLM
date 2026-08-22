"""Static and behavioural check of 379's channel feature. No torch, no corpus.

Seven properties. Every one of them would be a silent wrong number rather than a crash, and
three of them would be a LEAK - a feature that tells Phi which candidate is the answer for a
bookkeeping reason rather than an evidential one.

  1. OFF IS OFF. With REACH_CHANNEL false the tail is empty, so an arm without the lever has a
     bit-for-bit identical node vector to the arm before it.
  2. The walk is the all-zero baseline; connect, home and copy are one-hot and distinct.
  3. Only the ANSWERED row carries the indicators; every other row carries zeros of the same
     width, or the two builders disagree and the graph crashes mid-run.
  4. Both builders append the tail, in the same place, after `confirm`.
  5. The declared width grows by exactly three when the lever is on, in BOTH places that
     construct a Deriver.
  6. A value the offer never proposed reads as the walk baseline, not as a crash.
  7. reach_candidates exports from_place, or the feature reads nothing at all.

    python _check379_channel.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
v0 = v2('_stage289_derivation.py')

def load(v3):
    v4 = v0.v15(encoding='utf-8')
    v5 = {}
    for v6 in v3:
        v16 = v32.v28(f'^def {v6}\\(.*?(?=\\n(?:def |@|\\w))', v4, v32.v34 | v32.v35)
        if not v16:
            v26(f'FAIL: {v6} not found in {v0}')
            v39.v36(1)
        v29(v37(v16.v24(0), v6, 'exec'), v5)
    return (v5, v4)

def main() -> v1:
    v5, v4 = v17(('reach_channel', 'channel_feat'))
    v18, v19 = (v5['reach_channel'], v5['channel_feat'])
    v7 = []
    v8 = {'_reach_c': {'from_place': {'w': 7, 'c': -1, 'h': -2, 'k': -3}}}
    v5['REACH_CHANNEL'] = False
    if v18(None, v8, 'k') != (0.0, 0.0, 0.0):
        v7.v30('1. the feature is computed while the lever is off')
    if v19({'channel': (0.0, 0.0, 1.0)}, 0, 0) != []:
        v7.v30('1. the node tail is non-empty while the lever is off')
    v5['REACH_CHANNEL'] = True
    v9 = {v10: v18(None, v8, v10) for v10 in ('w', 'c', 'h', 'k')}
    if v9['w'] != (0.0, 0.0, 0.0):
        v7.v30(f"2. the walk is not the zero baseline: {v9['w']}")
    for v10 in ('c', 'h', 'k'):
        if v38(v9[v10]) != 1.0:
            v7.v30(f'2. `{v10}` is not one-hot: {v9[v10]}')
    if v31({v9[v10] for v10 in ('c', 'h', 'k')}) != 3:
        v7.v30(f'2. two channels collided onto the same indicator: {v9}')
    v11 = {'channel': v9['k']}
    v20, v21 = (v19(v11, 2, 2), v19(v11, 1, 2))
    if v20 != [0.0, 0.0, 1.0]:
        v7.v30(f'3. the answered row does not carry the channel: {v20}')
    if v22(v21):
        v7.v30(f'3. a non-answered row carries the channel: {v21}')
    if v31(v20) != v31(v21):
        v7.v30(f'3. width differs by row: {v31(v20)} vs {v31(v21)}')
    if v18(None, v8, 'absent') != (0.0, 0.0, 0.0):
        v7.v30('6. an unoffered value does not read as the walk baseline')
    if v18(None, {}, 'k') != (0.0, 0.0, 0.0):
        v7.v30('6. a question with no walk raises or answers non-zero')
    if v19({}, 0, 0) != [0.0, 0.0, 0.0]:
        v7.v30('6. a world with no channel key does not fall back to zeros')
    v12 = v32.v23('REACH_CONFIRM else \\[\\]\\)\\n\\s*\\+ channel_feat\\(q, i, qrow\\)', v4)
    if v31(v12) != 2:
        v7.v30(f'4. the tail is appended in {v31(v12)} builders after confirm, want 2')
    if v31(v32.v23('\\(3 if REACH_CHANNEL else 0\\)', v4)) != 2:
        v7.v30('5. the node width does not grow by three in both Deriver constructions')
    v13 = v32.v28('^def reach_candidates\\(.*?(?=\\n(?:def |@|\\w))', v4, v32.v34 | v32.v35).v24(0)
    if '"from_place": {c: from_place[c] for c in cands}' not in v13:
        v7.v30('7. reach_candidates does not export from_place')
    if v7:
        v26('FAIL')
        for v25 in v7:
            v26('  ' + v25)
        return 1
    v26('PASS  walk (0,0,0)  connect (1,0,0)  home (0,1,0)  copy (0,0,1)')
    v26('  off is off, one-hot and distinct, answered row only, constant width,')
    v26('  both builders and both widths agree, unoffered values fall back to the walk.')
    return 0
if v14 == '__main__':
    raise v27(v33())