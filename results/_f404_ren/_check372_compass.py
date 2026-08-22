"""372: the compass is a relation, and there are more than two. Checked and RUN.

Three ways this could be silently wrong: a new compass could fall through to the
old `share` branch and measure nothing (a DEAD KNOB, and 335's `--addresses` did
exactly that); it could skip the retention mask that `reach_places` applies, so
two compasses would be compared on two different tapes; or the members could be
arithmetically identical, which is how 371's first pass shipped share_1 and
share_w as the same expression.
"""
from __future__ import annotations
import ast
from collections import Counter
v0 = '_stage289_derivation.py'
v1 = ('share1', 'rare', 'common', 'cover', 'jaccard')

def static():
    v2 = v34.v20(v45(v0, encoding='utf-8').v35())
    v3 = v34.v24(v2).v21('"', "'")
    v4 = {v23.v22: v23 for v23 in v34.v39(v2) if v40(v23, v34.v41)}
    v5 = v34.v24(v4['reach_places'])
    v6 = [('every new compass is named in reach_places', v42((f"'{v14}'" in v5 for v14 in v1))), ('the new branch is taken BEFORE the old share loop', v5.v46("if REACH_COMPASS in ('share1'") < v5.v46('share[j] += min(cnt, c)')), ('retention honoured in the new branch too - or two compasses are two tapes', v5.v49('kp is not None and (not bool(kp[j]))') >= 1 or v5.v49('not bool(kp[j])') >= 1), ("the question's own place is excluded in the new branch", 'if j == i or (kp is not None' in v5), ('`both` is the only compass that still interleaves', "if REACH_COMPASS != 'both':" in v5), ('argparse accepts them', v42((f"'{v14}'" in v3 for v14 in v1)))]
    v7 = True
    for v22, v15 in v6:
        v29(f"  {('OK  ' if v15 else 'FAIL')}  {v22}")
        v7 &= v36(v15)
    return v7

def behaviour():
    """The five members must ORDER PLACES DIFFERENTLY on one hand-made tape, or they are one
    compass under five names - the mistake 371's first pass shipped."""
    v8 = {1: 2, 2: 40, 3: 3}
    v9 = {'A': 2, 'B': 900}
    v10 = {1: ['A'], 2: ['B'], 3: ['B']}
    v11 = 2
    v12 = {v14: v37() for v14 in v1}
    for v25, v26 in v10.v27():
        for v28 in v26:
            v12['share1'][v25] += 1
            v12['rare'][v25] += 1.0 / v9[v28]
            v12['common'][v25] += v43(v9[v28])
            v12['cover'][v25] += 1.0 / v8[v25]
            v12['jaccard'][v25] += 1.0 / v47(1, v11 + v8[v25] - 1)
    v13 = {v14: v38(v12[v14], key=lambda v25: (-v12[v14][v25], v25)) for v14 in v1}
    for v14 in v1:
        v29(f'  {v14:<8} {v13[v14]}   ' + ' '.v48((f'{v25}:{v12[v14][v25]:.4g}' for v25 in (1, 2, 3))))
    v15 = v13['rare'][0] == 1 and v13['common'][0] == 2 and (v13['cover'][0] == 1) and (v13['jaccard'] == [1, 3, 2])
    v29(f"  {('OK  ' if v15 else 'FAIL')}  rare prefers the rare-sharer, common the common one, cover the small focused one - they are not one compass")
    v16 = v30({v44(v13[v14]) for v14 in v1})
    v29(f"  {('OK  ' if v16 >= 3 else 'FAIL')}  {v16} distinct orderings among {v30(v1)} members")
    return v15 and v16 >= 3
if v17 == '__main__':
    v29('STATIC')
    v18 = v31()
    v29('BEHAVIOUR')
    v19 = v32()
    v29('\n372 OK' if v18 and v19 else '\n372 FAILED')
    raise v33(0 if v18 and v19 else 1)