"""Prove 296 before spending an hour on it. Seconds, no corpus and no model.

296 stops grading halves. One exam, one payoff: find the value the corpus put somewhere, or say
there is none. Five things have to hold, and each is a way the run could print a number that
means nothing:

  1 THE SPLIT IS BALANCED. 291 died because refusal was correct 93% of the time, so any mark of
    the refusal option was a mark of the answer. At 50/50 the mark carries nothing, which is the
    only reason the refusal world may keep its natural row count without becoming a tell.
  2 UNANSWERABLE IS A LIST WITHOUT THE ANSWER, not a smaller world. The truth is present among
    the candidates exactly when the question is answerable, and refusal is on every list.
  3 THE CANDIDATE WORLDS ARE ALL THE SAME SIZE, in both kinds of question, so no row count
    distinguishes an answerable question from an unanswerable one.
  4 THE PAYOFF IS 280's, EXACTLY, and it prices both degenerate policies: always answering loses
    a point on every unanswerable question, always refusing gives up every point it could have
    found. If either beat a competent policy the exam would be measuring nothing.
  5 REFUSAL DOES NOT ZERO THE IMPORT BUDGET. outside_mentions of a sentinel is empty, so a naive
    minimum over the candidate list would collapse every world to the bare evidence and make the
    question contentless - the failure 292 spent a run learning about.

    python _check296_mixed.py
"""
from __future__ import annotations
import random
from collections import Counter
import torch
import _stage289_derivation as s289
from _check293_identity import FakeBank
from _check294_open import pack294

def main() -> v0:
    v1 = True
    v2 = v70.v28('cpu')
    v38.v29, v38.v30, v38.v31 = (False, 0, False)
    v38.v32, v38.v33, v38.v34 = (True, True, 2)
    v38.v35, v38.v36, v38.v37 = ('anchor', 'uniform', 4)
    v38.v3 = 3
    v38.v4 = v39(v38.v40)
    v41, v42 = (v71(), v72())
    v5 = v43(v41['texts'])
    v6 = [('omega held a summit in geneva that spring', 'Geneva'), ('omega opened a studio in oslo after the tour', 'Oslo'), ('omega signed a pact in cairo before dawn', 'Cairo'), ('omega moved reserves to dublin for winter', 'Dublin'), ('omega founded a chapter in prague overnight', 'Prague'), ('omega shipped crates to lisbon that month', 'Lisbon')]
    for v44, (v73, v14) in v45(v6):
        v41['texts'].v50(v73)
        v41['tape'].v53.v50(v14)
        v41['straddr'].v50(f'omega|{v73.v74()[1]}')
        for v46 in v73.v74():
            v41['postings'].v97(v46, []).v50(v5 + v44)
    v7 = v43(v41['texts'])
    v41['texts_lc'] = [v73.v75() for v73 in v41['texts']]
    v41['n_slots'] = v7
    v8 = v70.v87().v47(296)
    v41['ctx_keys'] = v70.v88.v76.v48(v70.v77(v7, 16, generator=v8), dim=-1)
    v41['anc_keys'] = v70.v88.v76.v48(v70.v77(v7, 16, generator=v8), dim=-1)
    v41['slot_keys_slot'] = v49(v56(v7))
    v41['items'].v50({'S': 'omega', 'address': 'fp9:omega|held', 'slots': v49(v56(v5, v5 + v43(v6))), 'kind': 'clean'})
    v41.v51('_ident', None)
    v9 = v52((v78 for v78 in v38.v91(v41) if v78['S'] == 'kostya'))
    v10 = v49(v41['tape'].v53)
    v54, v16 = (v79(), [])
    v11 = v80.v55(0)
    for v12 in v56(400):
        v15 = v38.v81(v41, v9, v11, 2, v10)
        if v15 is None:
            continue
        v54[v15['answerable']] += 1
        v16.v50(v15)
    v7 = v57(1, v43(v16))
    v13 = v54[True] / v7
    v14 = 0.4 < v13 < 0.6
    v1 &= v14
    v58(f'1 answerable rate {v13:.3f} over {v7} draws: {v14}')
    v14 = True
    for v15 in v16:
        v59 = v15['truth_value']
        v14 &= v38.v82 in v15['cands']
        v14 &= (v59 in v15['cands']) == v15['answerable']
        v14 &= v15['cands'][v15['label']] == (v59 if v15['answerable'] else v38.v82)
        v14 &= v43(v15['cands']) == v38.v37 + 1
        v60 = [v41['tape'].v53[v89] for v89 in v15['slots'][:v15['query_row']]]
        v14 &= v67((v65 not in v60 for v65 in v15['cands']))
    v1 &= v61(v14)
    v17 = v52((v15 for v15 in v16 if v15['answerable']))
    v18 = v52((v15 for v15 in v16 if not v15['answerable']))
    v58(f'2 truth on the list iff answerable, refusal always on it, nothing on a row: {v61(v14)}')
    v58(f"  answerable {v17['cands']}\n  unanswerable {v18['cands']} truth {v18['truth_value']!r}")
    v19 = {}
    for v62, v15 in (('answerable', v17), ('unanswerable', v18)):
        v63 = v38.v68(v41, v15, v49(v15['cands']))
        v64 = {}
        for v65 in v15['cands']:
            v15.v51('_base', None)
            v64[v65] = v43(v38.v98(v41, v15, v42, v2, query_value=v65, import_k=v63)[2])
        v19[v62] = v64
        v66 = {v83 for v65, v83 in v64.v92() if v65 != v38.v82}
        v14 = v43(v66) == 1
        v1 &= v14
        v58(f'3 {v62}: candidate worlds {v99(v66)}, refusal world {v64[v38.v82]}, budget {v63}: {v14}')
    v14 = {v84 for v84 in v19['answerable'].v53()} == {v84 for v84 in v19['unanswerable'].v53()}
    v1 &= v14
    v58(f'  and the two kinds are indistinguishable by size: {v14}')
    v20 = {(True, False, True): 1.0, (True, True, False): 0.75, (True, False, False): -1.0, (False, True, False): 1.0, (False, False, False): -1.0}
    v14 = v67((v38.v93(v94, v95, v96) == v90 for (v96, v94, v95), v90 in v20.v92()))
    v1 &= v14
    v58(f'4 payoff cells exact (+1 found / -1 wrong / +1 correct silence / +0.75 hedge): {v14}')
    v21 = [(1, 0, 1), (1, 0, 0), (0, 1, 0), (0, 1, 0)]
    v22 = v85((v38.v93(False, v61(v100), v61(v101)) for v101, v102, v100 in v21)) / 4
    v23 = v85((v38.v93(True, False, v61(v101)) for v101, v102, v100 in v21)) / 4
    v24 = v85((v38.v93(not v101, True, v61(v101)) for v101, v102, v100 in v21)) / 4
    v14 = v24 > v22 and v24 > v23
    v1 &= v14
    v58(f'  always-answer {v22:+.3f}  always-silent {v23:+.3f}  competent {v24:+.3f} - both degenerates lose: {v14}')
    v25 = v38.v68(v41, v17, v49(v17['cands']))
    v26 = v38.v68(v41, v17, [v65 for v65 in v17['cands'] if v65 != v38.v82])
    v14 = v25 == v26 >= 1
    v1 &= v14
    v58(f'5 budget with refusal on the list {v25} == without it {v26}, and nonzero: {v14}')
    v58('\nMIXED OK' if v1 else '\nMIXED FAILED')
    return 0 if v1 else 1
if v27 == '__main__':
    raise v69(v86())