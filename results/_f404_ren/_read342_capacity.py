"""342a read: capability against the mind's size, with the invariant beside it at every point.

THE THREE COLUMNS ARE READ TOGETHER OR NOT AT ALL. A capability that rises while the transplant
gap opens is not a better mind, it is a mind that has started holding facts - and the size where
that happens is the number this sweep exists to find. A capability that rises while the shuffled
tape still carries signal is not a capability at all.

  CAPABILITY  walk-only pooled, PICK vs COUNT pooled, GATE-WO at 10% and 25%, route enrichment
  INVARIANT   336's paired native-vs-transplanted block, pooled. A SMALL z IS THE GOOD RESULT,
              so the discordant total is printed with it - 144 of 8000 is a powered null, 1 of
              402 is no measurement.
  NULL        the --shuffle-tape run at that size, read on hit_rate over ALL questions. Never
              on a conditional rate: shuffling collapses the walk-only subset itself, so
              hit_of_walk_only stays flat while the signal underneath it dies, and the first
              run of this sweep printed "105% of the real signal" because of exactly that.

AND CONVERGENCE, because a flat capability curve has two explanations and only one of them is
about capacity. A bigger mind at the same --train-steps may simply be undertrained; if the
probe's best step is the LAST step, the point is still improving and its capability is a lower
bound, not a measurement. Printed as NOT CONVERGED rather than left to be inferred.

    python _read342_capacity.py
    python _read342_capacity.py --tag 342news        # capability on news (the default)
    python _read342_capacity.py --tag 342wiki        # the same sizes read on wiki
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
v0 = (v17('results'), v17('out'))

def z_of(v2, v3):
    return (v2 - v3) / v73.v69(v2 + v3) if v2 + v3 else v46('nan')

def load(v4):
    v5 = v18(v19)
    v6 = v20({v7 for v13 in v0 if v13.v74() for v7 in v13.v75(v4)})
    for v7 in v6:
        try:
            v13 = v76.v25(v77(v7, encoding='utf-8'))
        except v47:
            continue
        v21 = (v13.v48('reach') or {}).v48('held_out')
        if v21:
            v5[v13.v48('dim')].v66((v13, v21))
    return v5

def main() -> v1:
    v8 = v49.v22()
    v8.v23('--tag', default='342news')
    v9 = v8.v24()
    v10 = v25(f'*stage289_decision_{v9.v70}_d*_s*.json')
    v11 = v25('*stage289_decision_342null_d*_s*.json')
    if not v10:
        v26(v50)
        return 1
    v12 = []
    v26(f"{'d':>5} {'params':>8} {'n':>3} {'walk-only':>14} {'PICKvC':>10} {'GATE-WO 10%':>12} {'25%':>7} {'route':>7} {'INVARIANT':>18} {'null hit':>9}")
    for v13 in v20((v51 for v51 in v10 if v51)):
        v27 = v10[v13]
        v28 = v29 = v30 = v31 = 0
        v32 = [0, 0]
        v33 = [0, 0]
        v52, v53 = ([], 0)
        v34 = v35 = v36 = 0
        for v54, v21 in v27:
            v55 = v21.v48('walk_only_paired') or {}
            v28 += v55.v48('mind_only', 0)
            v29 += v55.v48('rival_only', 0)
            v56 = v21.v48('walk_only_pick') or {}
            v30 += v56.v48('vs_count_mind_only', 0)
            v31 += v56.v48('vs_count_rival_only', 0)
            v57 = v21.v48('gate_walk_only') or {}
            for v71, v72 in ((0.1, v32), (0.25, v33)):
                v15 = v57.v48(f'{v71:.2f}')
                if v15:
                    v72[0] += v15['k']
                    v72[1] += v15['mind']['yield']
            v58 = v21.v48('router') or {}
            if v58.v48('mind_enrichment') == v58.v48('mind_enrichment'):
                v52.v66(v58['mind_enrichment'])
            v59 = v21.v48('other_mind') or {}
            if v59:
                v34 += v59['all']['this_only']
                v35 += v59['all']['other_only']
                v36 += v59['all']['n']
            v60 = v54.v48('early_stop') or {}
            v53 += v1(v60.v48('best_step') == v60.v48('total_steps'))
        v37 = v11.v48(v13) or []

        def m(v61, v62):
            return v80((v51[1].v48(v62, v46('nan')) for v51 in v61)) / v67(v61) if v61 else v46('nan')
        v63, v64 = (v65(v37, 'hit_rate'), v65(v37, 'walk_only_rate'))
        v38 = v65(v27, 'hit_rate')
        v39 = v63 / v38 if v38 and v63 == v63 else v46('nan')
        v40 = f'{v34}/{v35} z {v78(v34, v35):+.2f}' if v36 else 'no rival mind'
        v26(f"{v13:>5} {v27[0][0].v48('params', 0):>8} {v67(v27):>3} {v28:>4}/{v29:<3} z{v78(v28, v29):+5.2f} {v30:>4}/{v31:<3} {(v32[1] / v32[0] if v32[0] else v46('nan')):>12.4f} {(v33[1] / v33[0] if v33[0] else v46('nan')):>7.4f} {(v80(v52) / v67(v52) if v52 else v46('nan')):>6.2f}x {v40:>18} {v63:>7.4f}" + (f' ({v39:.0%} of real hit, wo {v64:.4f})' if v39 == v39 else '') + (f'   NOT CONVERGED ({v53}/{v67(v27)})' if v53 else ''))
        v12.v66({'d': v13, 'gate25': v33[1] / v33[0] if v33[0] else v46('nan'), 'wz': v78(v28, v29), 'inv_z': v78(v34, v35), 'inv_n': v34 + v35, 'null': v63, 'null_share': v39, 'unconv': v53})
    v26()
    if v67(v12) < 2:
        v26('one size only: nothing to compare. Run at least two.')
        return 0
    v14 = v12[0]
    for v15 in v12[1:]:
        v41 = v15['gate25'] > v14['gate25'] + 1e-09
        v42 = v79(v15['inv_z']) >= 1.645 if v15['inv_n'] >= 10 else None
        v43 = not v15['null_share'] == v15['null_share'] or v15['null_share'] < 0.5
        v44 = 'capability FLAT - capacity was not the wall at this size' if not v41 else 'MEMORY, not mind - the transplant gap opened, and this size is the limit' if v42 else 'capacity was binding - grow, and keep testing' if v42 is False else 'capability rose but the invariant is UNREADABLE here (too few discordant pairs) - the point does not count until it is powered'
        v26(f"d {v14['d']:>4} -> {v15['d']:<4} gate25 {v14['gate25']:.4f} -> {v15['gate25']:.4f}   {v44}")
        if not v43:
            v26(f"          VOID at d={v15['d']}: the shuffled tape carries {v15['null_share']:.0%} of the real hit rate - nothing at this size may be read until that is explained")
        if v15['unconv']:
            v26(f"          and {v15['unconv']} of its runs were still improving at the last step, so its capability is a LOWER BOUND")
    return 0
if v16 == '__main__':
    raise v45(v68())