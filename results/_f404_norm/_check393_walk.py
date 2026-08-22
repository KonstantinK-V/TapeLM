"""Check of 393's walker on a designed tape. No torch, no corpus.

Question: zebra hidden at the|sat. Bridge xx|yy (cat, moose). Gold lives at one|ran
(moose, zebra) — hop-2, not hop-1. Stepping to the bridge then walking finds it;
staying put does not; dumping hop-2 is a different lane.
"""
from __future__ import annotations
import argparse
import random
import _audit393_walk as W
ARGS = argparse.Namespace(places=8, topm=8, bridges=8, max_questions=200)
LINES = ['aa the cat sat bb', 'ii the dog sat jj', 'cc the zebra sat dd', 'oo xx cat yy pp', 'qq xx moose yy rr', 's1 one moose ran s2', 's3 one zebra ran s4']

def designed():
    T = W.A.build_tape(LINES, frame_max=1, min_fillers=1)
    a = T['of_addr'][1, ('the',), ('sat',)]
    slot = [s for s in T['places'][a] if T['toks'][s] == 'zebra'][0]
    return (T, a, slot)

def props():
    f = []
    T, a, slot = designed()
    m = W.measure(T, slot, ARGS, random.Random(0))
    if m is None:
        return ['0. designed question left the population']
    if m['hop1'] != 0:
        f.append('1. hop1 found zebra — the tape is not hop-2 by construction')
    if m['oracle'] != 1:
        f.append('2. oracle step missed zebra, which a walk to xx|yy must find')
    if m['committed'] != 1:
        f.append('3. the nearest bridge was not xx|yy or its walk missed zebra')
    if m['walk_only'] != 1:
        f.append('4. walk_only is 0: the step does not add anything hop1 lacked')
    if m['n_bridges'] < 1:
        f.append('5. no bridges')
    if m['hop1'] == 1 and m['oracle'] == 1:
        f.append('6. hop1 and oracle both 1 — hidden row is back in the key')
    return f

def mutations():
    caught = []
    T, a, slot = designed()
    real = W.measure

    def force_hop1(T, s, args, rng):
        m = real(T, s, args, rng)
        if m:
            m['hop1'] = 1
            m['walk_only'] = 0
        return m
    W.measure = force_hop1
    fails = props()
    if any((x.startswith('1.') for x in fails)):
        caught.append(1)
    W.measure = real

    def kill_oracle(T, s, args, rng):
        m = real(T, s, args, rng)
        if m:
            m['oracle'] = 0
        return m
    W.measure = kill_oracle
    fails = props()
    if any((x.startswith('2.') for x in fails)):
        caught.append(2)
    W.measure = real
    return caught

def main() -> int:
    bad = props()
    if bad:
        print('FAIL designed tape')
        for x in bad:
            print(' ', x)
        return 1
    caught = mutations()
    if {1, 2} - set(caught):
        print(f'checker is a comment; caught {caught}')
        return 1
    print(f'all properties hold, and re-introduced failures {sorted(caught)} were caught')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())