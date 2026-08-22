"""Torch-free checks for ladder step 1: the constraint interface.

THE ONE FAULT THAT WOULD FAKE A RESULT. The hidden truth stands at the question's OWN PLACE. If
that place is not subtracted out of the co-occurrence count, then every value standing beside
the lens HERE is counted as if it had been seen somewhere else - and the tape resolves straight
to the answer it was supposed to have to find. The arm would read beautifully and mean nothing.
This is the same subtraction reach_places does to the query fingerprint, for the same reason,
and it is checked here by RUNNING cons_resolve on a hand-made tape rather than by reading it.

Also asserted:
  - Phi's output space is the question's own rows and nothing else. That is what makes a fact
    unencodable in it, so it is the invariant's structural half and must not quietly widen.
  - the lens itself cannot be the answer.
  - the teacher is exact: the payoff of stage two comes from what the TAPE resolves, not from
    the lens, or the loss would be teaching the mind to like rows rather than to be right.
  - the three counting rivals exist and are distinct rules.
  - the walk is measured on the SAME question, or gate (b) compares two question sets.

    python _check345_cons.py
"""
from __future__ import annotations
import ast
import math
from collections import Counter
from pathlib import Path
SRC = Path('_stage289_derivation.py')

def fn(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None

class FakeTape:

    def __init__(self, values):
        self.values = values

def build(tree, names, ns):
    for nm in names:
        f = fn(tree, nm)
        if f is None:
            raise SystemExit(f'BROKEN: {nm} is gone')
        exec(compile(ast.Module(body=[f], type_ignores=[]), nm, 'exec'), ns)

def main() -> int:
    src = SRC.read_text(encoding='utf-8')
    tree = ast.parse(src)
    bad = []
    vals = ['lens', 'TRUTH', 'lens', 'TRUTH', 'lens', 'DECOY', 'DECOY']
    fills = [[('lens', [0], 1), ('TRUTH', [1], 1)], [('lens', [2], 1), ('TRUTH', [3], 1)], [('lens', [4], 1), ('DECOY', [5, 6], 2)]]
    by_val = {'lens': [(0, 1), (1, 1), (2, 1)], 'TRUTH': [(0, 1), (1, 1)], 'DECOY': [(2, 2)]}
    ix = {'fills': fills, 'by_val': by_val, 'of': {'A0': 0, 'A1': 1, 'A2': 2}}
    p = {'tape': FakeTape(vals), '_reach': ix, '_cons_cooc': {}}
    q = {'address': 'A0', 'slots': [0, 1], 'query_row': 1, 'truth_value': 'TRUTH'}
    ns = {'Counter': Counter, 'math': math, 'CONS_TOPM': 8, 'CONS_LENSES': 6, 'CONS_RESOLVE': 'count', 'reach_index': lambda pp: pp['_reach']}
    build(tree, ['cons_cooc', 'cons_resolve', 'cons_lenses', 'cons_place'], ns)
    best, bn, tot, top = ns['cons_resolve'](p, q, 'lens')
    print(f"cons_resolve through 'lens' -> {best} (count {bn} of {tot}), top {top}")
    if best == 'TRUTH':
        bad.append("THE HIDDEN TRUTH LEAKS: the question's own place was not subtracted, so the tape resolved to the very value it was supposed to have to find")
    elif best != 'DECOY':
        bad.append(f"cons_resolve answered {best}, expected DECOY (2 votes against TRUTH's 1)")
    if 'lens' in (top or []) or best == 'lens':
        bad.append('the lens can be its own answer')
    q2 = {'address': 'A1', 'slots': [2, 3], 'query_row': 1, 'truth_value': 'TRUTH'}
    b2, n2, t2, _ = ns['cons_resolve'](p, q2, 'lens')
    print(f'asked from place 1 -> {b2} (count {n2} of {t2})')
    if b2 != 'DECOY' or n2 != 2:
        bad.append(f'resolving from another place gave {b2}/{n2}, expected DECOY/2')
    ns['CONS_RESOLVE'] = 'share'
    bs, ns_, _t, _tp = ns['cons_resolve'](p, q, 'lens')
    print(f'share rule -> {bs} ({ns_:.3f})')
    if bs == 'TRUTH':
        bad.append('the share rule leaks the hidden truth where the count rule does not')
    vals2 = ['lens', 'TRUTH', 'lens', 'RARE', 'lens', 'COMMON', 'COMMON', 'COMMON', 'COMMON', 'COMMON']
    fills2 = [[('lens', [0], 1), ('TRUTH', [1], 1)], [('lens', [2], 1), ('RARE', [3], 1)], [('lens', [4], 1), ('COMMON', [5], 1)], [('COMMON', [6, 7, 8, 9], 4)]]
    by_val2 = {'lens': [(0, 1), (1, 1), (2, 1)], 'TRUTH': [(0, 1)], 'RARE': [(1, 1)], 'COMMON': [(2, 1), (3, 4)]}
    p2 = {'tape': FakeTape(vals2), '_cons_cooc': {}, '_reach': {'fills': fills2, 'by_val': by_val2, 'of': {'A0': 0, 'A1': 1, 'A2': 2, 'A3': 3}}}
    q3 = {'address': 'A0', 'slots': [0, 1], 'query_row': 1, 'truth_value': 'TRUTH'}
    ns['CONS_RESOLVE'] = 'count'
    bc, _n, _t, _tp = ns['cons_resolve'](p2, q3, 'lens')
    p2['_cons_cooc'] = {}
    ns['CONS_RESOLVE'] = 'share'
    bsh, _n2, _t2, _tp2 = ns['cons_resolve'](p2, q3, 'lens')
    print(f'frequency pull-apart: count -> {bc}   share -> {bsh}   (share must prefer RARE)')
    if bsh != 'RARE':
        bad.append(f'the share rule chose {bsh}; it exists to prefer the value whose whole presence is here (RARE 1/1) over a frequent one (COMMON 1/5)')
    ns['CONS_RESOLVE'] = 'count'
    ns['CONS_RESOLVE'] = 'place'
    jp = ns['cons_place'](p, q, 'lens')
    print(f'cons_place(lens) from place 0 -> {jp}')
    if jp == 0:
        bad.append("384: cons_place returned the question's OWN place - the hidden truth is standing there and the lens would read the answer out of the question")
    elif jp != 1:
        bad.append(f'384: cons_place chose {jp}, expected 1 (equal counts, larger share of its own hole: 1 of 2 against 1 of 3)')
    bp, np_, tp_, top_p = ns['cons_resolve'](p, q, 'lens')
    print(f'place rule -> {bp} (count {np_} of {tp_}), top {top_p}')
    if bp == 'DECOY':
        bad.append("384: the place rule reproduced the SUM's answer (DECOY, pooled from place 2) - no selection took place")
    elif bp != 'TRUTH':
        bad.append(f'384: answering from place 1 alone must give TRUTH, its only other filler; got {bp}')
    if 'lens' in (top_p or []):
        bad.append('384: the lens is its own answer under the place rule')
    q4 = {'address': 'A2', 'slots': [4, 5], 'query_row': 1, 'truth_value': 'DECOY'}
    if ns['cons_place'](p, q4, 'DECOY') is not None:
        bad.append("384: a lens standing only at the question's own place still got a place")
    if ns['cons_resolve'](p, q4, 'DECOY')[0] is not None:
        bad.append('384: a lens with no other place still resolved to something')
    ns['CONS_RESOLVE'] = 'count'
    lens = ns['cons_lenses'](p, q)
    print(f'cons_lenses -> {lens}')
    if lens != ['lens']:
        bad.append(f"the lens set is {lens}; it must be exactly the question's own visible rows - that is what makes a fact unencodable in Phi's output")
    cl = fn(tree, 'cons_lenses')
    if cl is None or "q['slots'][:q['query_row']]" not in ast.unparse(cl):
        bad.append("cons_lenses does not read the question's own visible rows")
    if cl and ('by_val' in ast.unparse(cl) or 'cands' in ast.unparse(cl)):
        bad.append("cons_lenses reaches outside the question's own rows: the output space of Phi has widened and the invariant's structural half is gone")
    cls = ast.unparse(fn(tree, 'cons_loss') or ast.parse(''))
    for needle, why in (('cons_answers', 'stage two must be priced on what the TAPE resolves'), ('REACH_GAMMA', 'one read is still paid at gamma, like every other read'), ('torch.softmax(l2, 0)', 'the lens choice is a policy, not a label')):
        if needle not in cls:
            bad.append(f'cons_loss: {why}')
    if 'reach_reward(q, [x if x is not None else REFUSE_LABEL for x in names2]' not in cls:
        bad.append('cons_loss does not turn an unresolvable lens into a refusal, so a lens that answers nothing would be scored as a wrong answer rather than as silence')
    crs = ast.unparse(fn(tree, 'cons_rivals') or ast.parse(''))
    for nm in ('rare', 'frequent', 'decisive'):
        if f"'{nm}'" not in crs:
            bad.append(f'the counting rival `{nm}` is gone')
    if 'min(lens' not in crs or 'max(lens' not in crs:
        bad.append('the rivals are not opposite rules, so a direction that happens to suit the tape could win the comparison on its own')
    if 'walk_answerable' not in src or 'reach_candidates(p, q)' not in src:
        bad.append('the walk is not measured on the same question, so gate (b) would compare two question sets rather than two interfaces')
    print()
    if bad:
        for b in bad:
            print(f'BROKEN: {b}')
        return 1
    print('CONS OK')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())