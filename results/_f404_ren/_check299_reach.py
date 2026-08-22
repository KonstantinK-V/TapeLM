"""Prove 299 before spending an hour on it. Seconds, no corpus and no model.

299 removes the offered candidate list. A hole is hidden, the mind walks to the nearest places
by frame fingerprint, and it may say only a filler it reached - or nothing. Five things:

  1 NOBODY OFFERS ANYTHING. The candidate set is whatever the walk reached, in the walk's own
    order, never by frequency - frequency is exactly the rival's rule and handing it to the
    construction would decide the comparison in advance.
  2 THE TRUTH IS NOT PLANTED. If no reached place carries it, silence is the correct answer and
    the question is unanswerable without anyone rigging a 50/50 split.
  3 EVERY CANDIDATE WORLD IS THE SAME SIZE - one shared budget - so no row count is a tell.
  4 THE TWO STAGES ARE WHAT THE WALK BUYS. Before it, only the values already at this address
    are sayable; after it, the tape is. Expanding is the difference between having options and
    not having them, and it is priced.
  5 THE PAYOFF IS 280's, with silence correct exactly when the truth was out of reach.

    python _check299_reach.py
"""
from __future__ import annotations
import random
import torch
import torch.nn.functional as F
import _stage289_derivation as s289
from _check293_identity import FakeBank
from _check294_open import pack294

def main() -> v0:
    v1 = True
    v2 = v75.v25('cpu')
    v26.v3 = v26.v4 = v26.v5 = False
    v26.v27, v26.v28, v26.v29 = (0, 2, 0.05)
    v26.v30, v26.v31, v26.v32 = (True, 8, 8)
    v26.v6 = v33(v26.v34)
    v35, v36 = (v76(), v77())
    v7 = v75.v88().v37(299)
    v35['frame_fps'] = v38(v89.v78(v75.v90(v35['n_slots'], 16, generator=v7), dim=-1))
    v35['frame_nfill'] = [2] * v35['n_slots']
    v35['frame_nfill_max'] = 2
    v35['frame_mode'] = True
    v8 = [v9 for v9 in v26.v91(v35, v100.v98(0)) if v9.v92('reach')]
    v39(f'0 questions {v79(v8)}')
    v1 &= v79(v8) > 0
    v9 = v8[0]
    v10 = v26.v40(v35, v9)
    v39(f"1 walked to {v79(v10['places'])} places -> candidates {v10['cands']}")
    v11 = {v35['tape'].v80[v81] for v93, v94, v95 in v10['places'] for v81 in v94['slots']}
    v12 = v33(v10['cands']) <= v11 and v79(v10['cands']) <= v26.v32
    v1 &= v41(v12)
    v39(f'  every candidate came from a reached place, cap respected: {v41(v12)}')
    v12 = v9['address'] not in {v94['address'] for v93, v94, v95 in v10['places']}
    v1 &= v41(v12)
    v39(f'  the walk does not return to its own place: {v41(v12)}')
    v12 = v79(v9['slots']) <= v26.v42
    v1 &= v41(v12)
    v39(f"  own place capped at {v26.v42} rows (got {v79(v9['slots'])}): {v41(v12)}")
    v13 = v26.v43(v35, v9)
    v39(f"2 truth {v9['truth_value']!r} reachable: {v13} (nobody planted it)")
    v75.v37(0)
    v14 = v26.v44(v2, d=8, n_node=9)
    v45, v46 = (v10['rows_of'], v10['cands'])
    v15 = v82([v26.v28] + [v79(v45[v83]) for v83 in v46]) if v46 else 0
    v16 = {v79(v26.v99(v35, v9, v36, v2, v83, v45[v83], v15)[2]) for v83 in v46}
    v12 = v79(v16) <= 1
    v1 &= v41(v12)
    v39(f'3 candidate worlds all {v16} rows, budget {v15}: {v41(v12)}')
    v47, v48, v49, v50, v51, v52 = v26.v53(v14, v35, v9, v2, v36)
    v12 = v79(v47) == v79(v49) + 2 and v79(v48) == v79(v50) + 1
    v1 &= v41(v12)
    v39(f'4 stage 1 = own {v49} + refuse + expand ({v79(v47)}), stage 2 = reached + refuse ({v79(v48)}): {v41(v12)}')
    v17 = v26.v54(v9, v50 + [v26.v96], v13, v2)
    v18 = 1.0 if not v13 else 0.75
    v12 = v84(v17[-1]) == v18
    v1 &= v41(v12)
    v39(f'5 silence pays {v84(v17[-1])} (reachable={v13}, so {v18}): {v41(v12)}')
    v19 = v26.v55(v14, v35, v9, v2, v36)
    v56, v57 = (v75.v85(v47, 0), v75.v85(v48, 0))
    v20 = v26.v54(v9, v49 + [v26.v96], v13, v2)
    v21 = -((v56[:-1] * v20).v97() + v56[-1] * (v26.v101 * (v57 * v17).v97() - v26.v29))
    v12 = v86(v84(v19) - v84(v21)) < 1e-06
    v1 &= v41(v12)
    v39(f'  loss == expected payoff of the walk: {v41(v12)} ({v84(v19):+.6f} vs {v84(v21):+.6f})')
    v26.v22 = True
    v9.v58('_reach_g', None)
    v59, v60, v61, v62, v63, v64 = v26.v53(v14, v35, v9, v2, v36)
    v65, v66 = v26.v67(v61, v62)
    v12 = v79(v59) == v79(v61) + 1 and v65 == v61 and (v79(v60) == v79(v62) if v62 else v79(v60) == 1) and (v66 == (v62 or [v26.v96]))
    v1 &= v41(v12)
    v39(f'6 no-refuse: stage1={v38(v65)}+expand ({v79(v59)}), stage2={v38(v66)} ({v79(v60)}): {v41(v12)}')
    v26.v23 = True
    v9.v58('_reach_g', None)
    v68, v69, v70, v71, v72, v73 = v26.v53(v14, v35, v9, v2, v36)
    v12 = v79(v68) == v79(v70) + 1 and v86(v84(v68[-1]) - v84(v69.v102())) < 1e-06 and (v79(v68) == v79(v70) + 1)
    v1 &= v41(v12)
    v39(f'7 lookahead: step logit == max(l2) ({v84(v68[-1]):+.4f}): {v41(v12)}')
    v26.v23 = False
    v26.v22 = False
    v39('\nREACH OK' if v1 else '\nREACH FAILED')
    return 0 if v1 else 1
if v24 == '__main__':
    raise v74(v87())