"""Prove 294 before spending an hour on it. Seconds, no corpus and no model.

294 is 292's question with 293's two disciplines applied and the addressing heuristic removed:
an address is one exact anchor string, and the three wrong answers are drawn rather than built.
Five things have to hold, and each is a way a plausible number could mean nothing:

  1 AN ADDRESS IS AN EXACT STRING. Every row of an anchor address was written under that anchor
    and no other, so nothing about which rows count as evidence went through a cosine, a tau or
    a word overlap. That is the whole point: the invariant forbids an approximation at the place
    where something is decided, and this is that place.
  2 THE TARGET IS FOREIGN. The hidden value is on none of the evidence rows, so retrieval inside
    the address cannot reach it - 292's property, kept, because it is what makes the comparison
    against the mind worth running.
  3 EVERY WORLD CARRIES THE SAME ROWS. Equal imports through the shared budget and an identical
    evidence set, or Phi can read a row count as the answer, which is the bookkeeping tell that
    made 289's ladder unreadable.
  4 NO CANDIDATE IS BUILT FROM ITS DISTANCE. Distractors are any value the address does not
    carry; `bucket_of` records where each one happens to live and is used only in the report.
    292's rungs were built BY relatedness and `mean_phi` read that construction back as an
    inverted landscape on six seeds out of six - this is the arm that tells the two apart.
  5 THE ROW CAP IS A BUDGET, NOT A DECISION. An anchor carries dozens of mentions and the graph
    is quadratic; the cap keeps the nearest in tape order, in tape order, identically in all
    four worlds.

    python _check294_open.py
"""
from __future__ import annotations
import random
from collections import Counter
import torch
import _stage289_derivation as s289
from _check293_identity import FakeBank, fake_pack

def pack294():
    """293's toy tape plus a second anchor, so the values hidden at `kostya` have a mention
    somewhere else and the shared import budget is not zero."""
    v1 = v25()
    v2 = ['dynamo signed spartak players before the tournament', 'dynamo travelled to moscow for the away leg', 'dynamo recruited gorky graduates that summer', 'dynamo opened a berlin office after the merger', 'dynamo sent reserves to paris for the friendly', 'dynamo loaned players to lyon that autumn']
    v3 = ['Spartak', 'Moscow', 'Gorky', 'Berlin', 'Paris', 'Lyon']
    v4 = ['dynamo|signed', 'dynamo|travelled to', 'dynamo|recruited', 'dynamo|opened', 'dynamo|sent', 'dynamo|loaned']
    v5 = v26(v1['texts'])
    v1['texts'] += v2
    v1['texts_lc'] = [v29.v57() for v29 in v1['texts']]
    v1['tape'].v6 += v3
    v1['straddr'] += v4
    v1['items'].v27({'S': 'dynamo', 'address': 'fp9:dynamo|signed', 'slots': v34(v54(v5, v5 + v26(v2))), 'kind': 'clean'})
    for v28, v29 in v30(v2):
        for v31 in v29.v58():
            v1['postings'].v80(v31, []).v27(v5 + v28)
    v7 = v26(v1['texts'])
    v1['n_slots'] = v7
    v8 = v61.v69().v32(294)
    v1['ctx_keys'] = v61.v70.v59.v33(v61.v60(v7, 16, generator=v8), dim=-1)
    v1['anc_keys'] = v61.v70.v59.v33(v61.v60(v7, 16, generator=v8), dim=-1)
    v1['slot_keys_slot'] = v34(v54(v7))
    v1.v35('_ident', None)
    return v1

def main() -> v0:
    v9 = True
    v10 = v61.v36('cpu')
    v44.v37, v44.v38 = (False, 0)
    v44.v39, v44.v40 = (True, 2)
    v44.v41, v44.v42, v44.v43 = ('anchor', 'uniform', 4)
    v44.v11 = 3
    v44.v12 = v45(v44.v46)
    v1, v47 = (v62(), v63())
    v13 = v44.v48(v1)
    v14 = {v49['S']: v49 for v49 in v13}
    v15 = v71(v14) == ['dynamo', 'kostya'] and v64((v44.v81(v1['straddr'][v65])[0] == v49['S'] for v49 in v13 for v65 in v49['slots']))
    v9 &= v50(v15)
    v51(f"1 anchor addresses {[(v49['S'], v49['slots']) for v49 in v13]}")
    v51(f'  every row written under that exact anchor, no grouping rule involved: {v50(v15)}')
    v16 = [v17 for v17 in v44.v72(v1, v77.v74(0)) if v17.v73('uniform')]
    v51(f'  open questions built: {v26(v16)}')
    v9 &= v26(v16) > 0
    if not v16:
        v51('\nOPEN294 FAILED (no questions on the toy tape)')
        return 1
    v17 = v16[0]
    v18 = v17['cands'][v17['label']]
    v19 = [v1['tape'].v6[v65] for v65 in v17['slots'][:v17['query_row']]]
    v15 = v18 not in v19 and v64((v66 not in v19 for v66 in v17['cands'] if v66 == v18))
    v9 &= v50(v15)
    v51(f'2 truth {v18!r} vs evidence {v19} - foreign to every row: {v50(v15)}')
    v20 = v44.v52(v1, v17, v34(v17['cands']))
    v21 = {v26(v44.v75(v1, v17, v47, v10, query_value=v66, import_k=v20)[2]) for v66 in v17['cands']}
    v15 = v20 >= 1 and v26(v21) == 1
    v9 &= v50(v15)
    v51(f'3 shared budget {v20}, world sizes {v21} - one size for four worlds: {v50(v15)}')
    v15 = v45(v17['bucket_of']) == {v66 for v66 in v17['cands'] if v66 != v18} and v64((v76 in ('same_anchor', 'elsewhere') for v76 in v17['bucket_of'].v6())) and (v18 not in v17['bucket_of'])
    v9 &= v50(v15)
    v51(f"4 buckets {v17['bucket_of']} recorded for the distractors only, truth excluded: {v50(v15)}")
    v22 = v53()
    for v23 in v54(300):
        v55 = v44.v67(v1, v14['kostya'], v77.v74(v23), 0, v34(v1['tape'].v6))
        if v55 is not None:
            v22[v55['label']] += 1
    v51(f'  label position over {v78(v22.v6())} draws: {v79(v71(v22.v13()))}')
    v15 = v17['query_row'] == v26(v17['slots']) - 1 and v17['query_row'] <= v44.v11 and (v17['slots'][:v17['query_row']] == v71(v17['slots'][:v17['query_row']]))
    v9 &= v50(v15)
    v51(f"5 rows {v17['slots']} capped at {v44.v11}, in tape order, query row last: {v50(v15)}")
    v51('\nOPEN294 OK' if v9 else '\nOPEN294 FAILED')
    return 0 if v9 else 1
if v24 == '__main__':
    raise v56(v68())