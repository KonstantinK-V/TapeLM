"""Static and behavioural check of 380's deep root. No torch, no corpus.

The bug being fixed was silent for four steps: reach_deep rebuilt a value->place map from the
WALKED places only, so a candidate offered by connect (since 365) or copy (since 377) fell back
to `places[0][0]` - the walk's first place, unrelated to where that value stands. Nothing
crashed; the second read simply started in the wrong neighbourhood, and `reachable_rate` counts
deep candidates, so reach moved with an identical offer.

Seven properties. Every one is a wrong number rather than an exception:

  1. A WALKED candidate keeps its own walked place.
  2. A NON-WALK candidate roots where it lives - the place of its first row - and NOT at
     places[0][0]. Checked for all three channels: connect (-1), home (-2), copy (-3).
  3. Rows at the question's OWN address are skipped, so the second read never starts at home.
  4. A candidate with no usable row at all falls back to places[0][0] rather than raising.
  5. The old walk-only map is GONE from reach_deep - if it is rebuilt anywhere the fix is dead
     even though this file passes.
  6. The root reads rc["rows_of"], not `ev`: the offer's rows do not depend on --reach-import,
     and a root that moved with the import policy would make depth and evidence one knob.
  7. (381) `cand_places` counts PLACES, not the -1/-2/-3 channel markers. from_place stores the
     channel for non-walk candidates, so every connect candidate used to collapse into one
     pseudo-place and every copy candidate into another - and the 377 argument that copy evicts
     walked candidates from distinct places was drawn from that collapse.

    python _check380_deeproot.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
v0 = v2('_stage289_derivation.py')

def main() -> v1:
    v3 = v0.v18(encoding='utf-8')
    v4 = {'reach_index': lambda v8: {'of': {'A_home': 0, 'A_far': 5, 'A_near': 9}}}
    v5 = None
    for v6 in ('place_of_rows', 'deep_root_of'):
        v19 = v30.v23(f'^def {v6}\\(.*?(?=\\n(?:def |@|\\w))', v3, v30.v31 | v30.v32)
        if not v19:
            v25(f'FAIL: {v6} not found in {v0}')
            return 1
        v27(v34(v19.v22(0), v6, 'exec'), v4)
        v5 = v19
    v7 = v4['deep_root_of']
    v8 = {'straddr': ['A_home', 'A_home', 'A_far', 'A_near']}
    v9 = {'address': 'A_home'}
    v10 = [(7, None, 0.0), (3, None, 0.0)]
    v11 = {'from_place': {'walked': 7, 'conn': -1, 'home': -2, 'copy': -3, 'orphan': -3, 'homeonly': -3}, 'rows_of': {'walked': [2], 'conn': [2], 'home': [3], 'copy': [3, 2], 'orphan': [], 'homeonly': [0, 1]}}
    v12 = []
    if v7(v8, v9, v11, 'walked', v10) != 7:
        v12.v28(f"1. a walked candidate did not keep its place: {v7(v8, v9, v11, 'walked', v10)}")
    for v6, v20 in (('conn', 5), ('home', 9), ('copy', 9)):
        v21 = v7(v8, v9, v11, v6, v10)
        if v21 == 7:
            v12.v28(f'2. `{v6}` fell back to places[0][0] instead of its own place')
        elif v21 != v20:
            v12.v28(f'2. `{v6}` rooted at {v21}, want {v20}')
    if v7(v8, v9, v11, 'homeonly', v10) != 7:
        v12.v28("3. a candidate whose only rows are at the question's own place did not fall back - the second read would start at home")
    if v7(v8, v9, v11, 'orphan', v10) != 7:
        v12.v28('4. a candidate with no rows did not fall back to places[0][0]')
    if v7(v8, v9, v11, 'absent', v10) != 7:
        v12.v28('4. an unknown value did not fall back to places[0][0]')
    v13 = v30.v23('^def reach_deep\\(.*?(?=\\n(?:def |@|\\w))', v3, v30.v31 | v30.v32).v22(0)
    if 'from_place.setdefault(v, j)' in v13 or 'from_place = {}' in v13:
        v12.v28('5. reach_deep still rebuilds a walk-only value->place map')
    if 'deep_root_of(p, q, rc, cands[best], places)' not in v13:
        v12.v28("5. reach_deep does not call deep_root_of for the mind's root")
    v14 = v5.v22(0).v29('"""')[-1]
    if 'rc["rows_of"]' not in v14:
        v12.v28("6. the root does not read the offer's own rows")
    if v30.v23('\\bev\\b', v14):
        v12.v28('6. the root reads `ev` - it would move with --reach-import')
    v15 = v30.v23('^def reach_candidates\\(.*?(?=\\n(?:def |@|\\w))', v3, v30.v31 | v30.v32)
    v16 = v15.v22(0)
    if 'len({from_place[c] for c in cands})' in v16:
        v12.v28('7. n_places still counts channel markers as places')
    if 'real_place[c] for c in cands' not in v16:
        v12.v28('7. n_places does not count resolved places')
    if 'place_of_rows(p, q, rows_of[c])' not in v16:
        v12.v28('7. reach_candidates does not resolve a real place per candidate')
    if v12:
        v25('FAIL')
        for v24 in v12:
            v25('  ' + v24)
        return 1
    v25('PASS  walked->7 (its own place)   connect->5   home->9   copy->9')
    v25('  own-place rows skipped, rowless and unknown values fall back to places[0][0],')
    v25("  the walk-only map is gone, the root reads the offer's rows not the import,")
    v25('  and cand_places counts places rather than the -1/-2/-3 channel markers.')
    return 0
if v17 == '__main__':
    raise v26(v33())