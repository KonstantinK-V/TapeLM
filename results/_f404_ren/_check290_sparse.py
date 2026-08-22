"""Prove 290 and 291 before spending hours on them. Seconds, no corpus and no model.

Everything checked here is a property of the construction rather than of the tape, so a fake
pack tests it honestly and instantly. Seven things have to hold, and each one is a way the run
could produce a plausible number that means nothing:

  1 the neighbourhood is DETERMINISTIC and does not include the query address. A neighbourhood
    that resamples makes every question a different question at each evaluation - the defect
    the fixed probe tape exists to prevent - and one that includes its own address hands the
    hidden row's siblings back as evidence.
  2 the graph base is invalidated by every helper that derives a question from another. A view
    that reuses the full question's base scores the wrong rows while every printed number still
    looks reasonable, which is the exact shape of failure this project keeps catching late.
  3 the base cache changes nothing: base-built graphs equal freshly-built ones bit for bit
    (torch.equal), so the speedup is a speedup and not a second implementation.
  4 the sparse verb keeps the dense conventions - query row last, sentinel value, rows in tape
    order - because drop_rows, view_of, region_views_of and both rivals all assume them.
  5 the refusal world is the question with the query row left unknown, and it is scored by the
    same Phi as every candidate. If it were anything else, refusal would be a second head.
  6 §19.7's growth: adding the two edge channels with zero-initialised input columns leaves the
    function bit-identical, so a narrower mind can be widened without being retrained.
  7 292's target is FOREIGN to the evidence and all four completed worlds carry the same number
    of rows. Unequal counts are the bookkeeping tell that made the ladder unreadable in 289, and
    a zero shared budget would make the four worlds one graph scored four times.

    python _check290_sparse.py
"""
from __future__ import annotations
import random
import torch
import _stage289_derivation as s289

class FakeTape:

    def __init__(v33, v26):
        v33.v26 = v26

def fake_pack():
    """Four addresses over eight slots. Two share an anchor, two share a relation, and one pair
    shares a rare word - one witness for each of N(a)'s three routes."""
    v1 = ['kostya was born in the year nineteen eighty five in kaluga', 'kostya was born in kaluga according to the parish register', 'kostya played for spartak during the winter season', 'the parish register of kaluga records several births', 'sweden defeated canada in the final match of the tournament', 'canada defeated sweden in the final match of the tournament', 'leipzig and weimar were connected by the same railway line', 'weimar and leipzig were connected by the same railway line']
    v2 = ['1985', 'Kaluga', 'Spartak', 'Kaluga', 'Canada', 'Sweden', 'Weimar', 'Leipzig']
    v3 = [{'S': 'kostya', 'address': 'fp0:kostya|born in', 'slots': [0, 1], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp1:kostya|played for', 'slots': [2], 'kind': 'clean'}, {'S': 'register', 'address': 'fp2:register|born in', 'slots': [3], 'kind': 'clean'}, {'S': 'match', 'address': 'fp3:match|defeated', 'slots': [4, 5], 'kind': 'clean'}, {'S': 'line', 'address': 'fp4:line|connected by', 'slots': [6, 7], 'kind': 'clean'}]
    v4 = {}
    for v27, v34 in v35(v1):
        for v36 in v34.v86():
            v4.v123(v36, []).v75(v27)
    return {'tape': v87(v2), 'texts': v1, 'texts_lc': [v34.v98() for v34 in v1], 'items': v3, 'postings': v4, 'n_slots': v76(v2)}

class FakeBank:

    def ctx_fp(v33, v37, v38=None):
        v39 = v89.v115().v66(v116(v124(v37)) % 2 ** 31)
        return v89.v117.v103.v88(v89.v104(16, generator=v39), dim=-1)

def main() -> v0:
    v5 = True
    v6 = v89.v40('cpu')
    v43.v41, v43.v42 = (2, True)
    v43.v7 = v72(v43.v73) | v72(v43.v90)
    v43.v8 = 0
    v44, v45 = (v74(), v91())
    v9 = 'fp0:kostya|born in'
    v10 = v43.v46(v44, v9, 2)
    v44.v47('_nb')
    v11 = v43.v46(v44, v9, 2)
    v12 = v10 == v11 and v9 not in v10 and v10
    v5 &= v48(v12)
    v49(f'1 N(a) = {v10}')
    v49(f'  deterministic, excludes itself, non-empty: {v48(v12)}')
    v13 = ('fp1:kostya|played for' in v10, 'fp2:register|born in' in v10)
    v49(f'  anchor route hit: {v13[0]}   relation route hit: {v13[1]}')
    v5 &= v50(v13)
    v14 = v43.v51(v44, v44['items'][0], v105.v92(0), 0, 2)
    v15 = v14['query_row']
    v12 = v15 == v76(v14['slots']) - 1 and (not v106(v14['vals'][v15], v107)) and (v14['slots'][:v15] == v108(v14['slots'][:v15])) and (v14['slots'][v15] == v44['items'][0]['slots'][0]) and (v14['own_rows'] == {1})
    v5 &= v48(v12)
    v49(f"4 sparse question: rows {v14['slots']} query_row {v15} own {v108(v14['own_rows'])}  conventions kept: {v48(v12)}")
    v49(f"  candidates {v14['cands']}  answerable {v14['answerable']}  label {v14['label']}")
    v43.v52(v44, v14, v45, v6)
    v16 = [('view_of', v43.v109(v14, v105.v92(1), 0.6)), ('drop_rows', v43.v110(v14, v105.v92(1), 0.6)), ('region_views_of', v43.v118(v14, 3)[0])]
    for v53, v54 in v16:
        if v54 is None:
            continue
        v12 = '_base' not in v54
        v5 &= v12
        v49(f'2 {v53:16s} drops the cached base: {v12}')
    v17 = True
    for v18 in v14['cands']:
        v14.v47('_base', None)
        v55 = v43.v64(v44, v14, v45, v6, None if v18 == v43.v93 else v18)
        v56 = v43.v64(v44, v14, v45, v6, None if v18 == v43.v93 else v18)
        v17 &= v50((v89.v70(v119, v120) for v119, v120 in v125(v55, v56)))
    v5 &= v17
    v49(f'3 cached graph == freshly built graph, every candidate (torch.equal): {v17}')
    v14.v47('_base', None)
    v57, v58, v59 = v43.v60(v44, v14, v45, v6, query_value=v43.v93, import_k=0)
    v61, v62, v63 = v43.v64(v44, v14, v45, v6, None)
    v12 = v89.v70(v57, v61) and v89.v70(v58, v62) and v89.v70(v59, v63)
    v5 &= v12
    v19 = v14['query_row']
    v20 = v111(v59[v19][0]) == 0.0 and v111(v58[v19].v121()) == 0.0
    v5 &= v20
    v49(f'5 refusal world == unknown-query-row world: {v12}   query row has no value share and no same-value edge: {v20}')
    v21 = v43.v65(v6, d=8, n_edge=5, n_node=9)
    v49(f'  Phi scores it like any candidate: {v111(v21.v97(v57, v58, v59)):+.4f}')
    v89.v66(0)
    v22 = v43.v65(v6, d=8, n_edge=3, n_node=9)
    v89.v66(0)
    v23 = v43.v65(v6, d=8, n_edge=5, n_node=9, grown=2)
    with v89.v94():
        v23.v112[0].v67[:, :3] = v22.v112[0].v67
        v23.v112[0].v96.v95(v22.v112[0].v96)
        v23.v122[0].v67.v95(v22.v122[0].v67)
        v23.v122[0].v96.v95(v22.v122[0].v96)
        for v27 in (0, 2):
            v23.v126[v27].v67.v95(v22.v126[v27].v67)
            v23.v126[v27].v96.v95(v22.v126[v27].v96)
    with v89.v94():
        v68 = v22.v97(v57[..., :3], v58, v59)
        v69 = v23.v97(v57, v58, v59)
    v12 = v89.v70(v68, v69)
    v5 &= v12
    v49(f'6 widening 3 -> 5 edge channels is function-identical: {v12}  ({v111(v68):+.6f} vs {v111(v69):+.6f})')
    v43.v41, v43.v71, v43.v8 = (0, True, 2)
    v43.v7 = v72(v43.v73)
    v43.v24 = True
    v25 = v74()
    v25['items'].v75({'S': 'elsewhere', 'address': 'fp9:elsewhere|noted', 'slots': [8, 9, 10], 'kind': 'clean'})
    v25['texts'] += ['kaluga was noted in the register of that province', 'spartak was noted in the register of that province', 'weimar was noted in the register of that province']
    v25['texts_lc'] = [v34.v98() for v34 in v25['texts']]
    v25['tape'].v26 += ['Kaluga', 'Spartak', 'Weimar']
    v25['n_slots'] = v76(v25['tape'].v26)
    for v27 in (8, 9, 10):
        for v36 in v25['texts'][v27].v86():
            v25['postings'].v123(v36, []).v75(v27)
    from collections import defaultdict as _dd
    v28 = v77(v78)
    for v29 in v25['items']:
        v28[v29['address'].v86(':', 1)[-1].v86('|')[0]].v75(v29)
    v30 = v78(v25['tape'].v26)
    v31 = []
    for v29 in v25['items']:
        for v79 in v99(v76(v29['slots'])):
            v100 = v43.v113(v25, v29, v105.v92(3), v79, v28, v30)
            if v100 is not None:
                v31.v75(v100)
    v49(f'\n-- 292 --\n7 open questions built on the toy tape: {v76(v31)}')
    if v31:
        v80 = v31[0]
        v81 = v80['cands'][v80['label']]
        v82 = [v25['tape'].v26[v114] for v114 in v80['slots'][:v80['query_row']]]
        v12 = v81 not in v82
        v5 &= v12
        v49(f'  truth {v81!r} vs evidence {v82} - foreign to every row: {v12}')
        v12 = v76(v80['cands']) == 4 and v80['cands'] == v108(v80['cands'])
        v5 &= v12
        v49(f"  four candidates, sorted so position says nothing: {v12}  {v80['cands']}")
        v83 = v43.v101(v25, v80, v78(v80['cands']))
        v5 &= v83 >= 1
        v49(f'  shared import budget {v83} - equal for the truth and all three rungs: {v83 >= 1}')
        v84 = {v18: v76(v43.v60(v25, v80, v45, v6, query_value=v18, import_k=v83)[2]) for v18 in v80['cands']}
        v12 = v76(v72(v84.v26())) == 1
        v5 &= v12
        v49(f'  every completed world has the same row count {v108(v72(v84.v26()))}: {v12}  (unequal counts are the bookkeeping tell that killed the ladder in 289)')
    else:
        v49('  (toy tape supplied none - the real check is `open.n` in the run report)')
    v49('\nSPARSE OK' if v5 else '\nSPARSE FAILED')
    return 0 if v5 else 1
if v32 == '__main__':
    raise v85(v102())