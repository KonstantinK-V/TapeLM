"""365/367: the extra channels, checked statically and RUN on a hand-made tape.

Three ways this could be wrong and none of them would show up in a loss curve:
the channel could hand the question its own answer back, the interleave could
quietly widen the offer so the arm wins on budget, or the strict form 363
closed could creep back in as a threshold.
"""
from __future__ import annotations
import ast
from collections import Counter
v0 = '_stage289_derivation.py'

def static():
    v1 = v44(v0, encoding='utf-8').v20()
    v2 = v41.v21(v1)
    v3 = v41.v22(v2)
    v4 = v3.v23('"', "'")
    v5 = {v25.v24: v25 for v25 in v41.v45(v2) if v46(v25, v41.v47)}
    v6 = v41.v22(v5['reach_connect'])
    v7 = v41.v22(v5['reach_candidates'])
    v8 = [("the question's own place is excluded", 'j != i' in v6), ('its own values are excluded (recall covers them)', 'if v in own:' in v6), ('retention honoured, as reach_places honours it', 'retain_keep(p)' in v6 and 'bool(kp[j])' in v6), ('overlap-weighted', 'overlap[j] += 1' in v6 and 'score[v] += ov' in v6), ('no threshold - 363 closed the strict form', '>= 2' not in v6), ('367: home lane exists, round-robin, tagged apart from the connect lane', 'if OWN_IN_OFFER:' in v7 and 'from_place[v] = -2' in v7 and ('for tup in zip_longest(*lanes)' in v7)), ('367: home values imported from the same source as every candidate', 'outside_mentions(p, q, v)' in v7), ('interleaved BEFORE the cap, so the offer does not grow', v7.v49('cands = mixed') < v7.v49('cands = cands[:REACH_CANDS]')), ('exactly one cap', v7.v50('cands = cands[:REACH_CANDS]') == 1), ('flag exists and defaults OFF', "add_argument('--connect', action='store_true'" in v4), ('assigned from args', 'CONNECT, CONNECT_MAX = (args.connect, args.connect_max)' in v3), ('written into the report', "'connect': bool(CONNECT)" in v4)]
    v9 = True
    for v24, v14 in v8:
        v31(f"  {('OK  ' if v14 else 'FAIL')}  {v24}")
        v9 &= v42(v14)
    return v9

def behaviour():
    """The ranking itself, on a tape where the weighted and plain orders DIFFER - otherwise the
    test would pass against the plain count that 363 measured as worse."""
    v10 = {'A', 'B', 'C'}
    v11 = {1: {'A', 'B', 'C', 'X'}, 2: {'A', 'Y'}, 3: {'A', 'Y'}, 4: {'A', 'Y'}, 5: {'A', 'Y'}}
    v12 = v26({v27: v48(v51 & v10) for v27, v51 in v11.v52()})
    v13 = v26()
    for v27, v28 in v12.v29(4000):
        for v30 in v11[v27]:
            if v30 in v10:
                continue
            v13[v30] += v28
    v31(f'  overlap {v53(v12)}')
    v31(f'  score   {v53(v13)}   (X from one place sharing 3, Y from four sharing 1)')
    v14 = v13['X'] == 3 and v13['Y'] == 4 and ('A' not in v13)
    v31(f"  {('OK  ' if v14 else 'FAIL')}  weight counts relatedness, own values never scored")
    from itertools import zip_longest
    v32, v33 = ([['w1', 'w2', 'w3'], ['c1', 'w2', 'c2'], ['o1', 'o2']], 6)
    v34, v35 = (v43(), [])
    for v15 in v36(*v32):
        for v37 in v15:
            if v37 is not None and v37 not in v34:
                v34.v54(v37)
                v35.v55(v37)
    v16 = v35[:v33]
    v31(f'  three lanes -> {v16}')
    v14 &= v16 == ['w1', 'c1', 'o1', 'w2', 'o2', 'w3'] and v48(v43(v16)) == v33
    v31(f"  {('OK  ' if v14 else 'FAIL')}  round-robin, deduped on first appearance, walk first, capped at {v33}")
    return v14
if v17 == '__main__':
    v31('STATIC')
    v18 = v38()
    v31('BEHAVIOUR')
    v19 = v39()
    v31('\n365 OK' if v18 and v19 else '\n365 FAILED')
    raise v40(0 if v18 and v19 else 1)