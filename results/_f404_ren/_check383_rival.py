"""Is the counting rival blind anywhere the mind can see? Torch-free.

THE TRAP, walked into three times now and named in reach_count_rival's own docstring: a rival
that cannot NAME a candidate is not losing, it is mute, and its zero is a definition rather than
a defeat. It was closed for depth (`_dp` places are appended) and left open for the channels.

A value contributed by connect or copy stands at NO WALKED PLACE - if it did, the walk lane
would have offered it first and the interleave would have deduped it. The old rival iterated
PLACES and filtered by membership, so those candidates could never be scored. `--connect` has
been in the standing arm since 365.

Seven properties (5-7 are 383, the tie-break):

  1. A candidate standing at a walked place is scored by its share there.
  2. A CHANNEL candidate - at no walked place - is scored too, at the place 381 resolves for it.
  3. The rival can WIN with a channel candidate. If it cannot, the fix is cosmetic.
  4. A candidate with no resolvable place is skipped, not crashed on and not scored as zero-
     denominator.
  5. The walked pass still runs first, and the walk's own order remains the LAST tie-break.
  6. (383) EQUAL SHARE IS BROKEN BY THE RAW COUNT. `top_share` reads 0.999-1.000 on every seed
     of every arm ever run, so the share rule saturates on values that OWN their place - and
     with --min-fillers 1 that is a single-filler frame, where the share is 1.0 whether the
     value stands there nine times or twice. The walk's order was the tie-break most favourable
     to counting in ORDERING and the least favourable in STRENGTH.
  7. (383) The number of candidates tying at the winning share is COUNTED, so "the rule
     saturates" is a measurement rather than an inference from top_share.

    python _check382_rival.py
"""
from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path
v0 = v2('_stage289_derivation.py')

def main() -> v1:
    v3 = v0.v15(encoding='utf-8')
    v4 = v34.v16('^def reach_count_rival\\(.*?(?=\\n(?:def |@|\\w))', v3, v34.v35 | v34.v36)
    if not v4:
        v32(f'FAIL: reach_count_rival not found in {v0}')
        return 1
    v5 = {0: [('w', [], 1), ('x', [], 3)], 1: [('k', [], 3), ('y', [], 1)], 2: [('z', [], 5)]}
    v6 = {}

    def fake_rc(v17, v18):
        return v6['rc']
    v7 = {'reach_candidates': v19, 'Counter': v20, 'reach_index': lambda v17: {'fills': v5}, 'REACH_DEPTH': 1}
    v21(v37(v4.v30(0), 'reach_count_rival', 'exec'), v7)
    v8 = v7['reach_count_rival']
    v9 = []
    v6['rc'] = {'cands': ['w', 'k'], 'places': [(0, None, 1.0)], 'real_place': {'w': 0, 'k': 1}}
    v22, v23 = v8(None, {})
    if v22 != 'k':
        v9.v38(f'3. the rival cannot win with a channel candidate: got {v22!r} {v23:.3f}')
    elif v42(v23 - 0.75) > 1e-09:
        v9.v38(f'2. the channel candidate scored {v23:.3f}, want 0.750 at its own place')
    v6['rc'] = {'cands': ['w', 'x'], 'places': [(0, None, 1.0)], 'real_place': {'w': 0, 'x': 0}}
    if v8(None, {})[0] != 'x':
        v9.v38('1. a walked candidate is no longer scored by its share at its place')
    v6['rc'] = {'cands': ['w', 'ghost'], 'places': [(0, None, 1.0)], 'real_place': {'w': 0, 'ghost': None}}
    try:
        v25, v39 = v8(None, {})
    except v24 as e:
        v9.v38(f'4. a placeless candidate raised {v45(v44).v14}: {v44}')
        v25 = None
    if v25 not in (None, 'w', 'x'):
        v9.v38(f'4. a placeless candidate was scored anyway: {v25!r}')
    v5[3] = [('t', [], 1), ('u', [], 3)]
    v10 = {}
    v6['rc'] = {'cands': ['w', 't'], 'places': [(0, None, 1.0)], 'real_place': {'w': 0, 't': 3}}
    v26, v27 = v8(None, v10)
    if v42(v27 - 0.25) > 1e-09 or v26 != 'w':
        v9.v38(f"5. an exact tie did not go to the walk's own order: {v26!r} {v27:.3f}")
    v5[4] = [('h', [], 9), ('g', [], 27)]
    v6['rc'] = {'cands': ['w', 'h'], 'places': [(0, None, 1.0)], 'real_place': {'w': 0, 'h': 4}}
    v28, v29 = v8(None, {})
    if v28 != 'h':
        v9.v38(f'6. equal share is still decided by walk order, not by the count: {v28!r}')
    elif v42(v29 - 0.25) > 1e-09:
        v9.v38(f'6. the winning share changed: {v29:.3f}')
    v5[5], v5[6] = ([('p', [], 2)], [('r', [], 9)])
    v6['rc'] = {'cands': ['p', 'r'], 'places': [(5, None, 1.0)], 'real_place': {'p': 5, 'r': 6}}
    if v8(None, {})[0] != 'r':
        v9.v38('6. with both candidates at share 1.0 the rule still ignores the count - this is exactly the saturation the fix exists for')
    v11 = {}
    v6['rc'] = {'cands': ['p', 'r'], 'places': [(5, None, 1.0)], 'real_place': {'p': 5, 'r': 6}}
    v8(None, v11)
    if v11.v40('_cr_ties') != 2:
        v9.v38(f"7. two candidates tied at share 1.0 and ties reads {v11.v40('_cr_ties')}")
    v6['rc'] = {'cands': ['w', 'x'], 'places': [(0, None, 1.0)], 'real_place': {'w': 0, 'x': 0}}
    v12 = {}
    v8(None, v12)
    if v12.v40('_cr_ties') != 1:
        v9.v38(f"7. a determinate winner reads ties {v12.v40('_cr_ties')}, want 1")
    v13 = v4.v30(0)
    if 'for v in cands:' not in v13:
        v9.v38('2. the channel pass is gone - candidates at no walked place are mute')
    elif v13.v43('for j, _it, _sim in places:') > v13.v43('for v in cands:'):
        v9.v38('5. the channel pass runs before the walked pass')
    if 'rc.get("real_place"' not in v13:
        v9.v38("2. the rival does not use 381's resolved place")
    if v9:
        v32('FAIL')
        for v31 in v9:
            v32('  ' + v31)
        return 1
    v32('PASS  walked .25 vs channel .75 - the rival names the channel one; placeless')
    v32('  candidates skipped; equal share now breaks by the raw count and only then by the')
    v32("  walk's order; ties at the winning share are counted.")
    return 0
if v14 == '__main__':
    raise v33(v41())