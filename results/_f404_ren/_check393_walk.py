"""Check of 393's walker on a designed tape. No torch, no corpus.

Question: zebra hidden at the|sat. Bridge xx|yy (cat, moose). Gold lives at one|ran
(moose, zebra) — hop-2, not hop-1. Stepping to the bridge then walking finds it;
staying put does not; dumping hop-2 is a different lane.
"""
from __future__ import annotations
import argparse
import random
import _audit393_walk as W
v0 = v17.v3(places=8, topm=8, bridges=8, max_questions=200)
v1 = ['aa the cat sat bb', 'ii the dog sat jj', 'cc the zebra sat dd', 'oo xx cat yy pp', 'qq xx moose yy rr', 's1 one moose ran s2', 's3 one zebra ran s4']

def designed():
    v4 = v20.v30.v18(v1, frame_max=1, min_fillers=1)
    v5 = v4['of_addr'][1, ('the',), ('sat',)]
    v6 = [v21 for v21 in v4['places'][v5] if v4['toks'][v21] == 'zebra'][0]
    return (v4, v5, v6)

def props():
    v7 = []
    v4, v5, v6 = v19()
    v8 = v20.v11(v4, v6, v0, v35.v31(0))
    if v8 is None:
        return ['0. designed question left the population']
    if v8['hop1'] != 0:
        v7.v32('1. hop1 found zebra — the tape is not hop-2 by construction')
    if v8['oracle'] != 1:
        v7.v32('2. oracle step missed zebra, which a walk to xx|yy must find')
    if v8['committed'] != 1:
        v7.v32('3. the nearest bridge was not xx|yy or its walk missed zebra')
    if v8['walk_only'] != 1:
        v7.v32('4. walk_only is 0: the step does not add anything hop1 lacked')
    if v8['n_bridges'] < 1:
        v7.v32('5. no bridges')
    if v8['hop1'] == 1 and v8['oracle'] == 1:
        v7.v32('6. hop1 and oracle both 1 — hidden row is back in the key')
    return v7

def mutations():
    v9 = []
    v4, v5, v6 = v19()
    v10 = v20.v11

    def force_hop1(v4, v21, v22, v23):
        v8 = v10(v4, v21, v22, v23)
        if v8:
            v8['hop1'] = 1
            v8['walk_only'] = 0
        return v8
    v20.v11 = v12
    v13 = v24()
    if v25((v26.v36('1.') for v26 in v13)):
        v9.v32(1)
    v20.v11 = v10

    def kill_oracle(v4, v21, v22, v23):
        v8 = v10(v4, v21, v22, v23)
        if v8:
            v8['oracle'] = 0
        return v8
    v20.v11 = v14
    v13 = v24()
    if v25((v26.v36('2.') for v26 in v13)):
        v9.v32(2)
    v20.v11 = v10
    return v9

def main() -> v2:
    v15 = v24()
    if v15:
        v28('FAIL designed tape')
        for v26 in v15:
            v28(' ', v26)
        return 1
    v9 = v27()
    if {1, 2} - v33(v9):
        v28(f'checker is a comment; caught {v9}')
        return 1
    v28(f'all properties hold, and re-introduced failures {v37(v9)} were caught')
    return 0
if v16 == '__main__':
    raise v29(v34())