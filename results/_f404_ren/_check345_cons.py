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
v0 = v2('_stage289_derivation.py')

def fn(v3, v4):
    for v5 in v68.v30(v3):
        if v73(v5, v68.v74) and v5.v4 == v4:
            return v5
    return None

class FakeTape:

    def __init__(v31, v32):
        v31.v32 = v32

def build(v3, v6, v7):
    for v8 in v6:
        v33 = v62(v3, v8)
        if v33 is None:
            raise v67(f'BROKEN: {v8} is gone')
        v69(v75(v68.v77(body=[v33], type_ignores=[]), v8, 'exec'), v7)

def main() -> v1:
    v9 = v0.v34(encoding='utf-8')
    v3 = v68.v35(v9)
    v10 = []
    v11 = ['lens', 'TRUTH', 'lens', 'TRUTH', 'lens', 'DECOY', 'DECOY']
    v12 = [[('lens', [0], 1), ('TRUTH', [1], 1)], [('lens', [2], 1), ('TRUTH', [3], 1)], [('lens', [4], 1), ('DECOY', [5, 6], 2)]]
    v13 = {'lens': [(0, 1), (1, 1), (2, 1)], 'TRUTH': [(0, 1), (1, 1)], 'DECOY': [(2, 2)]}
    v14 = {'fills': v12, 'by_val': v13, 'of': {'A0': 0, 'A1': 1, 'A2': 2}}
    v15 = {'tape': v70(v11), '_reach': v14, '_cons_cooc': {}}
    v16 = {'address': 'A0', 'slots': [0, 1], 'query_row': 1, 'truth_value': 'TRUTH'}
    v7 = {'Counter': v36, 'math': v37, 'CONS_TOPM': 8, 'CONS_LENSES': 6, 'CONS_RESOLVE': 'count', 'reach_index': lambda v76: v76['_reach']}
    v38(v3, ['cons_cooc', 'cons_resolve', 'cons_lenses', 'cons_place'], v7)
    v39, v40, v41, v42 = v7['cons_resolve'](v15, v16, 'lens')
    v43(f"cons_resolve through 'lens' -> {v39} (count {v40} of {v41}), top {v42}")
    if v39 == 'TRUTH':
        v10.v71("THE HIDDEN TRUTH LEAKS: the question's own place was not subtracted, so the tape resolved to the very value it was supposed to have to find")
    elif v39 != 'DECOY':
        v10.v71(f"cons_resolve answered {v39}, expected DECOY (2 votes against TRUTH's 1)")
    if 'lens' in (v42 or []) or v39 == 'lens':
        v10.v71('the lens can be its own answer')
    v17 = {'address': 'A1', 'slots': [2, 3], 'query_row': 1, 'truth_value': 'TRUTH'}
    v44, v45, v46, v47 = v7['cons_resolve'](v15, v17, 'lens')
    v43(f'asked from place 1 -> {v44} (count {v45} of {v46})')
    if v44 != 'DECOY' or v45 != 2:
        v10.v71(f'resolving from another place gave {v44}/{v45}, expected DECOY/2')
    v7['CONS_RESOLVE'] = 'share'
    v48, v49, v50, v51 = v7['cons_resolve'](v15, v16, 'lens')
    v43(f'share rule -> {v48} ({v49:.3f})')
    if v48 == 'TRUTH':
        v10.v71('the share rule leaks the hidden truth where the count rule does not')
    v18 = ['lens', 'TRUTH', 'lens', 'RARE', 'lens', 'COMMON', 'COMMON', 'COMMON', 'COMMON', 'COMMON']
    v19 = [[('lens', [0], 1), ('TRUTH', [1], 1)], [('lens', [2], 1), ('RARE', [3], 1)], [('lens', [4], 1), ('COMMON', [5], 1)], [('COMMON', [6, 7, 8, 9], 4)]]
    v20 = {'lens': [(0, 1), (1, 1), (2, 1)], 'TRUTH': [(0, 1)], 'RARE': [(1, 1)], 'COMMON': [(2, 1), (3, 4)]}
    v21 = {'tape': v70(v18), '_cons_cooc': {}, '_reach': {'fills': v19, 'by_val': v20, 'of': {'A0': 0, 'A1': 1, 'A2': 2, 'A3': 3}}}
    v22 = {'address': 'A0', 'slots': [0, 1], 'query_row': 1, 'truth_value': 'TRUTH'}
    v7['CONS_RESOLVE'] = 'count'
    v52, v53, v50, v51 = v7['cons_resolve'](v21, v22, 'lens')
    v21['_cons_cooc'] = {}
    v7['CONS_RESOLVE'] = 'share'
    v54, v55, v56, v57 = v7['cons_resolve'](v21, v22, 'lens')
    v43(f'frequency pull-apart: count -> {v52}   share -> {v54}   (share must prefer RARE)')
    if v54 != 'RARE':
        v10.v71(f'the share rule chose {v54}; it exists to prefer the value whose whole presence is here (RARE 1/1) over a frequent one (COMMON 1/5)')
    v7['CONS_RESOLVE'] = 'count'
    v7['CONS_RESOLVE'] = 'place'
    v23 = v7['cons_place'](v15, v16, 'lens')
    v43(f'cons_place(lens) from place 0 -> {v23}')
    if v23 == 0:
        v10.v71("384: cons_place returned the question's OWN place - the hidden truth is standing there and the lens would read the answer out of the question")
    elif v23 != 1:
        v10.v71(f'384: cons_place chose {v23}, expected 1 (equal counts, larger share of its own hole: 1 of 2 against 1 of 3)')
    v58, v59, v60, v61 = v7['cons_resolve'](v15, v16, 'lens')
    v43(f'place rule -> {v58} (count {v59} of {v60}), top {v61}')
    if v58 == 'DECOY':
        v10.v71("384: the place rule reproduced the SUM's answer (DECOY, pooled from place 2) - no selection took place")
    elif v58 != 'TRUTH':
        v10.v71(f'384: answering from place 1 alone must give TRUTH, its only other filler; got {v58}')
    if 'lens' in (v61 or []):
        v10.v71('384: the lens is its own answer under the place rule')
    v24 = {'address': 'A2', 'slots': [4, 5], 'query_row': 1, 'truth_value': 'DECOY'}
    if v7['cons_place'](v15, v24, 'DECOY') is not None:
        v10.v71("384: a lens standing only at the question's own place still got a place")
    if v7['cons_resolve'](v15, v24, 'DECOY')[0] is not None:
        v10.v71('384: a lens with no other place still resolved to something')
    v7['CONS_RESOLVE'] = 'count'
    v25 = v7['cons_lenses'](v15, v16)
    v43(f'cons_lenses -> {v25}')
    if v25 != ['lens']:
        v10.v71(f"the lens set is {v25}; it must be exactly the question's own visible rows - that is what makes a fact unencodable in Phi's output")
    v26 = v62(v3, 'cons_lenses')
    if v26 is None or "q['slots'][:q['query_row']]" not in v68.v63(v26):
        v10.v71("cons_lenses does not read the question's own visible rows")
    if v26 and ('by_val' in v68.v63(v26) or 'cands' in v68.v63(v26)):
        v10.v71("cons_lenses reaches outside the question's own rows: the output space of Phi has widened and the invariant's structural half is gone")
    v27 = v68.v63(v62(v3, 'cons_loss') or v68.v35(''))
    for v64, v65 in (('cons_answers', 'stage two must be priced on what the TAPE resolves'), ('REACH_GAMMA', 'one read is still paid at gamma, like every other read'), ('torch.softmax(l2, 0)', 'the lens choice is a policy, not a label')):
        if v64 not in v27:
            v10.v71(f'cons_loss: {v65}')
    if 'reach_reward(q, [x if x is not None else REFUSE_LABEL for x in names2]' not in v27:
        v10.v71('cons_loss does not turn an unresolvable lens into a refusal, so a lens that answers nothing would be scored as a wrong answer rather than as silence')
    v28 = v68.v63(v62(v3, 'cons_rivals') or v68.v35(''))
    for v8 in ('rare', 'frequent', 'decisive'):
        if f"'{v8}'" not in v28:
            v10.v71(f'the counting rival `{v8}` is gone')
    if 'min(lens' not in v28 or 'max(lens' not in v28:
        v10.v71('the rivals are not opposite rules, so a direction that happens to suit the tape could win the comparison on its own')
    if 'walk_answerable' not in v9 or 'reach_candidates(p, q)' not in v9:
        v10.v71('the walk is not measured on the same question, so gate (b) would compare two question sets rather than two interfaces')
    v43()
    if v10:
        for v66 in v10:
            v43(f'BROKEN: {v66}')
        return 1
    v43('CONS OK')
    return 0
if v29 == '__main__':
    raise v67(v72())