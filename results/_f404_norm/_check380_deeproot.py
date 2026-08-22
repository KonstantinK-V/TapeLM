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
SRC = Path('_stage289_derivation.py')

def main() -> int:
    src = SRC.read_text(encoding='utf-8')
    ns = {'reach_index': lambda p: {'of': {'A_home': 0, 'A_far': 5, 'A_near': 9}}}
    m = None
    for name in ('place_of_rows', 'deep_root_of'):
        mm = re.search(f'^def {name}\\(.*?(?=\\n(?:def |@|\\w))', src, re.S | re.M)
        if not mm:
            print(f'FAIL: {name} not found in {SRC}')
            return 1
        exec(compile(mm.group(0), name, 'exec'), ns)
        m = mm
    root = ns['deep_root_of']
    p = {'straddr': ['A_home', 'A_home', 'A_far', 'A_near']}
    q = {'address': 'A_home'}
    places = [(7, None, 0.0), (3, None, 0.0)]
    rc = {'from_place': {'walked': 7, 'conn': -1, 'home': -2, 'copy': -3, 'orphan': -3, 'homeonly': -3}, 'rows_of': {'walked': [2], 'conn': [2], 'home': [3], 'copy': [3, 2], 'orphan': [], 'homeonly': [0, 1]}}
    fails = []
    if root(p, q, rc, 'walked', places) != 7:
        fails.append(f"1. a walked candidate did not keep its place: {root(p, q, rc, 'walked', places)}")
    for name, want in (('conn', 5), ('home', 9), ('copy', 9)):
        got = root(p, q, rc, name, places)
        if got == 7:
            fails.append(f'2. `{name}` fell back to places[0][0] instead of its own place')
        elif got != want:
            fails.append(f'2. `{name}` rooted at {got}, want {want}')
    if root(p, q, rc, 'homeonly', places) != 7:
        fails.append("3. a candidate whose only rows are at the question's own place did not fall back - the second read would start at home")
    if root(p, q, rc, 'orphan', places) != 7:
        fails.append('4. a candidate with no rows did not fall back to places[0][0]')
    if root(p, q, rc, 'absent', places) != 7:
        fails.append('4. an unknown value did not fall back to places[0][0]')
    body = re.search('^def reach_deep\\(.*?(?=\\n(?:def |@|\\w))', src, re.S | re.M).group(0)
    if 'from_place.setdefault(v, j)' in body or 'from_place = {}' in body:
        fails.append('5. reach_deep still rebuilds a walk-only value->place map')
    if 'deep_root_of(p, q, rc, cands[best], places)' not in body:
        fails.append("5. reach_deep does not call deep_root_of for the mind's root")
    code = m.group(0).split('"""')[-1]
    if 'rc["rows_of"]' not in code:
        fails.append("6. the root does not read the offer's own rows")
    if re.search('\\bev\\b', code):
        fails.append('6. the root reads `ev` - it would move with --reach-import')
    rcands = re.search('^def reach_candidates\\(.*?(?=\\n(?:def |@|\\w))', src, re.S | re.M)
    rbody = rcands.group(0)
    if 'len({from_place[c] for c in cands})' in rbody:
        fails.append('7. n_places still counts channel markers as places')
    if 'real_place[c] for c in cands' not in rbody:
        fails.append('7. n_places does not count resolved places')
    if 'place_of_rows(p, q, rows_of[c])' not in rbody:
        fails.append('7. reach_candidates does not resolve a real place per candidate')
    if fails:
        print('FAIL')
        for f in fails:
            print('  ' + f)
        return 1
    print('PASS  walked->7 (its own place)   connect->5   home->9   copy->9')
    print('  own-place rows skipped, rowless and unknown values fall back to places[0][0],')
    print("  the walk-only map is gone, the root reads the offer's rows not the import,")
    print('  and cand_places counts places rather than the -1/-2/-3 channel markers.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())