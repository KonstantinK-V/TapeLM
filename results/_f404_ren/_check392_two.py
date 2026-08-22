"""Check of 392 on a designed tape. No torch, no corpus.

A = the|sat, B = one|ran, gold C = the|ran (unique filler zebra). C is in addr-N of both
and in the walk of neither. Same 390 twelve-line tape, plus a second document so the null
has somewhere to go.

Every property is a wrong number. Failures 1-8 are re-introduced by monkeypatch.
"""
from __future__ import annotations
import argparse
import random
import re
from collections import Counter
from pathlib import Path
import _audit390_address as A
import _audit392_two as B
v0 = v4('_audit392_two.py')
v1 = v34.v5(topm=8, max_questions=200, places=8)
v2 = ['aa the cat sat bb', 'ii the dog sat jj', 'oo xx cat yy pp', 'ee the zebra ran ff', 'gg the zebra ran hh', 's1 one elk ran s2', 's3 one ibex ran s4', 'kk one pig sat ll', '= other =', 'zz no overlap here 11', 'yy still other 22']

def designed():
    v6 = v50.v35(v2, frame_max=1, min_fillers=1)
    v7 = v6['of_addr'][1, ('the',), ('sat',)]
    v8 = v6['of_addr'][1, ('one',), ('ran',)]
    v9 = v6['of_addr'][1, ('the',), ('ran',)]
    return (v6, v7, v8, v9)

def props():
    v10 = []
    v6, v7, v8, v9 = v36()
    v11 = v42.v23(v6, v7, v8, v1)
    if v11 is None:
        return ['0. designed pair produced no gold']
    if v6['toks'][v6['places'][v9][0]] != 'zebra' and 'zebra' not in v6['prof'][v9]:
        v10.v51('0. gold place is not the zebra hole')
    v12 = v37(v6['on_line'][v6['owner'][v6['places'][v7][0]]])
    v12.v38(v7)
    v13 = v37(v50.v52(v6, v7, v12))
    v14 = v37(v6['on_line'][v6['owner'][v6['places'][v8][0]]])
    v14.v38(v8)
    v15 = v37(v50.v52(v6, v8, v14))
    if v9 not in v13 or v9 not in v15:
        v10.v51('1. gold is not in addr-N(A) ∩ addr-N(B) on the designed tape')
    if v11['in_a_uncap'] != 1 or v11['in_b_uncap'] != 1:
        v10.v51('1. measure_pair did not see C in both address neighbourhoods')
    if v42.v21(v6, v9) != 'zebra':
        v10.v51(f'2. unique filler of C is {v42.v21(v6, v9)}, not zebra')
    if v11['hit_a'] != 0 or v11['hit_b'] != 0:
        v10.v51(f"3. walk hit A={v11['hit_a']} B={v11['hit_b']}, expected 0/0 — cloze")
    if v11['hit_both'] != 1:
        v10.v51('4. address intersection missed C, which is the whole claim')
    if v11['hit_concat'] != 0:
        v10.v51('5. concat of two empty walks found C')
    if v7 == v8:
        v10.v51('6. A and B collapsed')
    if v9 in v12:
        v10.v51("7. gold is on A's line — window artefact")
    v16 = v0.v39(encoding='utf-8')
    if 'prof[pid]' in v16.v58('def walk8', 1)[-1].v58('def addr_set', 1)[0]:
        pass
    if 'measure(' in v16 and 'hidden' in v16:
        v10.v51('8. 392 reintroduced a cloze hidden-token question')
    v17 = v42.v29(v2)
    if v53(v17) < 2:
        v10.v51(f'9. documents() returned {v53(v17)} docs, expected 2')
    v18 = [v40 for v40 in v59(v53(v6['places'])) if v6['owner'][v6['places'][v40][0]] >= 7]
    if v18:
        v41 = v42.v23(v6, v7, v18[0], v1)
        if v41 is not None and v41['hit_both'] == 1 and (v41['hit_a'] == 0):
            v10.v51('10. null document still hits both — gold is global, not joint')
    return v10

def mutations():
    """Re-introduce failures. A check that never fires is a comment."""
    v19 = []
    v6, v7, v8, v9 = v36()
    v20 = v42.v21
    v42.v21 = lambda v6, v54, v55=(): 'zebra' if v54 == v9 else v20(v6, v54, v55)
    v42.v21 = v20
    v22 = v42.v23

    def leak_walk(v6, v7, v8, v43):
        v11 = v22(v6, v7, v8, v43)
        if v11:
            v11['hit_a'] = 1
        return v11
    v42.v23 = v24
    v11 = v42.v23(v6, v7, v8, v1)
    if v11 and v11['hit_a'] == 1:
        v19.v51(3)
    v42.v23 = v22

    def miss_both(v6, v7, v8, v43):
        v11 = v22(v6, v7, v8, v43)
        if v11:
            v11['hit_both'] = 0
        return v11
    v42.v23 = v25
    v26 = v44()
    if v45((v46.v60('4.') for v46 in v26)):
        v19.v51(4)
    v42.v23 = v22

    def concat_hit(v6, v7, v8, v43):
        v11 = v22(v6, v7, v8, v43)
        if v11:
            v11['hit_concat'] = 1
        return v11
    v42.v23 = v27
    v26 = v44()
    if v45((v46.v60('5.') for v46 in v26)):
        v19.v51(5)
    v42.v23 = v22
    v28 = v42.v29
    v42.v29 = lambda v56: [v61(v59(v53(v56)))]
    v26 = v44()
    if v45((v46.v60('9.') for v46 in v26)):
        v19.v51(9)
    v42.v29 = v28
    return v19

def main() -> v3:
    v30 = v44()
    if v30:
        v48('FAIL designed tape')
        for v46 in v30:
            v48(' ', v46)
        return 1
    v19 = v47()
    v31 = {3, 4, 5, 9}
    v32 = v31 - v37(v19)
    if v32:
        v48(f'checker is a comment; did not catch re-introduced {v62(v32)}')
        return 1
    v48(f'all properties hold, and re-introduced failures {v62(v19)} were caught')
    return 0
if v33 == '__main__':
    raise v49(v57())