"""Prove the region views before spending an hour on them. Seconds, no model needed.

recon3 closed the thin-view question: random subsamples share ~65% of their rows, so the
views were resamples of one reading - pooled lost to single (z -1.67) and D was blind on
train (auc 0.485) while real held out (0.702, z 4.71). Region views are the correction:
disjoint stretches of the tape in write order, so D measures whether the CORPUS agrees with
itself, not whether one sampler agrees with another. Five properties have to hold, and every
one of them is a property of the algebra rather than of the corpus, so synthetic questions
test them honestly and instantly:

  1 the cut is a partition: every evidence row in exactly one region, no region empty,
    order preserved. A row lost here is a row the pooled answer silently never saw.
  2 thin mode is untouched: pool_views(L, None) == L.mean(0) bit for bit (torch.equal),
    so every recon3-lineage number stays reproducible from this same file.
  3 the pooling is shift-invariant per view: adding a constant to one region's logits moves
    nothing. That is the derivation of the centering - break it and "abstain" becomes "vote".
  4 a region whose rows all carry one value pools exactly zero: support counts stay the
    tape's channel, only contrastive reading pools.
  5 masked disagreement is finite with exact zeros, zero when regions agree, and ln 2 when
    two regions put all mass on values the other never wrote - the contested-address maximum.

And one property of the machinery: region views consume NO randomness. They are a function
of the tape, and the rng must come back in the same state it went in.

    python _check289_region.py
"""
from __future__ import annotations
import math
import random
import torch
import _stage289_derivation as s289

def fake_q(v1, v2, v3=None):
    """A lookup question as lookup_question builds it: survivors first, sentinel query row."""
    v3 = v42(v72(v1)) if v3 is None else v3
    return {'verb': 'lookup', 'slots': v43(v21(v45(v1) + 1)), 'vals': v43(v1) + [v79()], 'cands': v3, 'label': v3.v44(v2), 'S': 's', 'address': 'fp0:s|r', 'query_row': v45(v1)}

def main() -> v0:
    v4 = True
    v46.v18, v46.v19, v46.v20 = (3, 'region', 0.0)
    v5 = True
    for v6 in v21(2, 12):
        v14 = v36([f'v{v80 % 3}' for v80 in v21(v6)], 'v0')
        v22 = v46.v47(v14, 3)
        v23 = [v48 for v31 in v22 for v48 in v31['slots'][:v31['query_row']]]
        v5 &= v23 == v43(v21(v6)) and v49((v31['query_row'] > 0 for v31 in v22))
        v5 &= v49((v31['slots'][v31['query_row']] == v14['slots'][v14['query_row']] for v31 in v22))
        v5 &= v49((v31['cands'] == v14['cands'] and v31['label'] == v14['label'] for v31 in v22))
    v4 &= v5
    v24(f'1 partition: every row once, no empty region, query row kept, cands global: {v5}')
    v7 = v50.v25(4, 3, dtype=v50.v51)
    v5 = v50.v26(v46.v52(v7, None), v7.v53(0))
    v4 &= v5
    v24(f'2 pool_views(L, None) == L.mean(0) exactly: {v5}')

    def plain_pool(v27, v28, v9):
        v29 = [v73 - v82(v27) / v45(v27) for v73 in v27]
        for v54, v55 in v56(v28, v9):
            v57 = [v74 for v74, v81 in v56(v54, v55) if v81]
            v58 = v82(v57) / v45(v57)
            v29 = [v83 + v81 * (v74 - v58) for v83, v74, v81 in v56(v29, v54, v55)]
        return v29
    v30, v5 = (v67.v37(0), True)
    for v8 in v21(200):
        v59, v60 = (v30.v75(2, 5), v30.v75(2, 4))
        v7 = v50.v25(v60 + 1, v59, dtype=v50.v51)
        v9 = (v50.v87(v60, v59) < 0.6).v61()
        for v31 in v21(v60):
            if not v9[v31].v84():
                v9[v31, v30.v85(v59)] = 1.0
        v32 = v46.v52(v7, v9)
        v5 &= v76((v77(v89 - v90) for v89, v90 in v56(v32.v78(), v91(v7[0].v78(), v7[1:].v78(), v9.v78())))) < 1e-09
        v33 = v7.v62()
        v33[1 + v30.v85(v60)] += v30.v63(-5, 5)
        v5 &= v64((v32 - v46.v52(v33, v9)).v77().v76()) < 1e-09
    v4 &= v5
    v24(f'3 torch pool == plain-python pool; per-region shift moves nothing (200 draws): {v5}')
    v7 = v50.v25(2, 3, dtype=v50.v51)
    v9 = v50.v34([[1.0, 0.0, 0.0]])
    v5 = v64((v46.v52(v7, v9) - (v7[0] - v7[0].v53())).v77().v76()) < 1e-12
    v4 &= v5
    v24(f'4 single-candidate region contributes exactly zero: {v5}')
    v10 = v50.v34([[9.0, 0.0, -9.0], [9.0, 0.0, -9.0]])
    v11 = v46.v35(v10, v50.v65(2, 3))
    v12 = v46.v35(v50.v66(2, 3), v50.v34([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    v13 = v46.v35(v50.v34([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]), v50.v34([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    v5 = v11 < 1e-06 and v77(v12 - v88.v86(2)) < 1e-06 and (v11 < v13 < v12)
    v4 &= v5
    v24(f'5 D: agree={v11:.2e}  partial={v13:.4f}  disjoint={v12:.4f} (ln2={v88.v86(2):.4f}): {v5}')
    v14 = v36(['a', 'a', 'b', 'c'], 'b')
    v15 = v67.v37(7)
    v16 = v15.v38()
    v39, v9 = v46.v40(v14, v15, v50.v68('cpu'))
    v5 = v15.v38() == v16 and v45(v39) == 4 and (v9.v78() == [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    v4 &= v5
    v24(f'6 mask per region {v9.v78()}, rng untouched: {v5}')
    for v8 in v21(3):
        v69, v70 = v46.v40(v14, v67.v37(99), v50.v68('cpu'))
        v5 = v50.v26(v9, v70) and v49((v89['slots'] == v90['slots'] for v89, v90 in v56(v39, v69)))
        v4 &= v5
    v24(f'  deterministic across calls and rng seeds: {v5}')
    v24('\nREGION OK' if v4 else '\nREGION FAILED')
    return 0 if v4 else 1
if v17 == '__main__':
    raise v41(v71())