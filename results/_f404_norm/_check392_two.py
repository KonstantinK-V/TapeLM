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
SRC = Path('_audit392_two.py')
ARGS = argparse.Namespace(topm=8, max_questions=200, places=8)
LINES = ['aa the cat sat bb', 'ii the dog sat jj', 'oo xx cat yy pp', 'ee the zebra ran ff', 'gg the zebra ran hh', 's1 one elk ran s2', 's3 one ibex ran s4', 'kk one pig sat ll', '= other =', 'zz no overlap here 11', 'yy still other 22']

def designed():
    T = A.build_tape(LINES, frame_max=1, min_fillers=1)
    a = T['of_addr'][1, ('the',), ('sat',)]
    b = T['of_addr'][1, ('one',), ('ran',)]
    c = T['of_addr'][1, ('the',), ('ran',)]
    return (T, a, b, c)

def props():
    f = []
    T, a, b, c = designed()
    m = B.measure_pair(T, a, b, ARGS)
    if m is None:
        return ['0. designed pair produced no gold']
    if T['toks'][T['places'][c][0]] != 'zebra' and 'zebra' not in T['prof'][c]:
        f.append('0. gold place is not the zebra hole')
    drop_a = set(T['on_line'][T['owner'][T['places'][a][0]]])
    drop_a.discard(a)
    na = set(A.half_nbrs(T, a, drop_a))
    drop_b = set(T['on_line'][T['owner'][T['places'][b][0]]])
    drop_b.discard(b)
    nb = set(A.half_nbrs(T, b, drop_b))
    if c not in na or c not in nb:
        f.append('1. gold is not in addr-N(A) ∩ addr-N(B) on the designed tape')
    if m['in_a_uncap'] != 1 or m['in_b_uncap'] != 1:
        f.append('1. measure_pair did not see C in both address neighbourhoods')
    if B.unique_filler(T, c) != 'zebra':
        f.append(f'2. unique filler of C is {B.unique_filler(T, c)}, not zebra')
    if m['hit_a'] != 0 or m['hit_b'] != 0:
        f.append(f"3. walk hit A={m['hit_a']} B={m['hit_b']}, expected 0/0 — cloze")
    if m['hit_both'] != 1:
        f.append('4. address intersection missed C, which is the whole claim')
    if m['hit_concat'] != 0:
        f.append('5. concat of two empty walks found C')
    if a == b:
        f.append('6. A and B collapsed')
    if c in drop_a:
        f.append("7. gold is on A's line — window artefact")
    src = SRC.read_text(encoding='utf-8')
    if 'prof[pid]' in src.split('def walk8', 1)[-1].split('def addr_set', 1)[0]:
        pass
    if 'measure(' in src and 'hidden' in src:
        f.append('8. 392 reintroduced a cloze hidden-token question')
    docs = B.documents(LINES)
    if len(docs) < 2:
        f.append(f'9. documents() returned {len(docs)} docs, expected 2')
    other = [p for p in range(len(T['places'])) if T['owner'][T['places'][p][0]] >= 7]
    if other:
        n = B.measure_pair(T, a, other[0], ARGS)
        if n is not None and n['hit_both'] == 1 and (n['hit_a'] == 0):
            f.append('10. null document still hits both — gold is global, not joint')
    return f

def mutations():
    """Re-introduce failures. A check that never fires is a comment."""
    caught = []
    T, a, b, c = designed()
    real = B.unique_filler
    B.unique_filler = lambda T, pid, banned=(): 'zebra' if pid == c else real(T, pid, banned)
    B.unique_filler = real
    real_mp = B.measure_pair

    def leak_walk(T, a, b, args):
        m = real_mp(T, a, b, args)
        if m:
            m['hit_a'] = 1
        return m
    B.measure_pair = leak_walk
    m = B.measure_pair(T, a, b, ARGS)
    if m and m['hit_a'] == 1:
        caught.append(3)
    B.measure_pair = real_mp

    def miss_both(T, a, b, args):
        m = real_mp(T, a, b, args)
        if m:
            m['hit_both'] = 0
        return m
    B.measure_pair = miss_both
    fails = props()
    if any((x.startswith('4.') for x in fails)):
        caught.append(4)
    B.measure_pair = real_mp

    def concat_hit(T, a, b, args):
        m = real_mp(T, a, b, args)
        if m:
            m['hit_concat'] = 1
        return m
    B.measure_pair = concat_hit
    fails = props()
    if any((x.startswith('5.') for x in fails)):
        caught.append(5)
    B.measure_pair = real_mp
    real_docs = B.documents
    B.documents = lambda lines: [list(range(len(lines)))]
    fails = props()
    if any((x.startswith('9.') for x in fails)):
        caught.append(9)
    B.documents = real_docs
    return caught

def main() -> int:
    bad = props()
    if bad:
        print('FAIL designed tape')
        for x in bad:
            print(' ', x)
        return 1
    caught = mutations()
    need = {3, 4, 5, 9}
    missing = need - set(caught)
    if missing:
        print(f'checker is a comment; did not catch re-introduced {sorted(missing)}')
        return 1
    print(f'all properties hold, and re-introduced failures {sorted(caught)} were caught')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())