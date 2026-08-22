"""THE CEILING OF 34.4, READ OFF DUMPS ALREADY IN HAND. No run, no torch, no corpus.

33-VOID cost this project a step because a ceiling that was already measured went unread. So the
walk_only lever starts here: every quantity that can void it is ALREADY REPORTED by the stage, and
this file is the reading, not a new measurement.

WHAT walk_only IS. `answerable and not truth_in_own` - the truth is NOT among the values already
standing at this hole, and it IS among the ones the walk reached. On that population staying is
arithmetically wrong, so it is the one population where the merge cannot be the answer and the
route has to decide. 34.4 is the proposal to train stay/go THERE and nowhere else.

WHAT THE STAGE ALREADY PRINTS, and where each number lands:

    walk_only_arrive     of the walk_only questions, how many got a step at all   THE ROUTER
    walk_only_pick.n     how many of them were stepped on                         THE SIZE
    walk_only_pick       mind / rival / count_rival on those                      THE PICK
    step_rate            how often the arm steps at all
    deep_only_rate       the truth reachable ONLY by the second read              W1'S CEILING
    hit_of_deep_only     and what the arm does there
    hit_of_own, ceiling  the other two halves of the population

THE FOUR VOID CHECKS, DECLARED HERE BEFORE ANY DUMP IS OPENED. Each one, if it fires, closes the
lever WITHOUT a training run - which is the entire point of reading first.

  V1  walk_only_arrive >= 0.95  ->  VOID. The router ALREADY steps where staying is wrong, so a
      term that teaches it to step there has nothing to teach. Whatever is lost on that
      population is lost in the PICK, and the route is not the lever.
  V2  the walk_only share of the exam < 0.02  ->  VOID. The masked gradient would be under 2% of
      the training signal: the arm is a slower control, and a null on it would say nothing about
      routing. (34.4 cites ~0.036, so this is expected to pass - it is here because expecting is
      not measuring.)
  V3  deep_only_rate <= 0.05  ->  W1 IS VOID, which is Kostya's own condition: depth as a
      decision needs a population where the second read UNIQUELY holds the truth.
  V4  count_rival_rate >= the mind's rate on walk_only  ->  the pick there is a counting problem,
      not a routing one, and the route lever is aimed at the wrong half.

POOLED AS COUNTS, NEVER AS MEANS OF MEANS. `walk_only_pick` carries integers, so the seeds pool
exactly; `walk_only_arrive` is a rate, so its population is recovered as pick.n / arrive and the
pooled rate is the sum of stepped over the sum of the population. A mean of four rates is a
different number and this project has been bitten by that shape before.

    python _read394_walkonly.py results/stage289_decision*391ctl*.json
    python _read394_walkonly.py results/stage289_decision*.json --held
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
v0 = ('held_out', 'train_control')

def num(v2):
    """a reported nan or a missing key must READ as missing, not as a zero that pools"""
    if v2 is None:
        return None
    try:
        v16 = v38(v2)
    except (v39, v40):
        return None
    return None if v52.v41(v16) else v16

def pull(v3, v4):
    v5 = v42.v17(v43(v3, encoding='utf-8'))
    v6 = (v5.v18('reach') or {}).v18(v4)
    if not v6:
        return None
    v7 = v6.v18('walk_only_pick') or {}
    v8 = v19(v6.v18('walk_only_arrive'))
    v9 = v7.v18('n')
    v10 = v9 / v8 if v8 and v9 is not None and (v8 > 0) else None
    return {'file': v53(v3).v20, 'seed': v5.v18('seed'), 'arm': v4, 'arrive': v8, 'stepped': v9, 'pop': v10, 'mind': v7.v18('mind'), 'rival': v7.v18('rival'), 'count_rival': v7.v18('count_rival'), 'step_rate': v19(v6.v18('step_rate')), 'deep_only': v19(v6.v18('deep_only_rate')), 'hit_of_deep_only': v19(v6.v18('hit_of_deep_only')), 'hit_of_own': v19(v6.v18('hit_of_own')), 'ceiling': v19(v6.v18('ceiling')), 'n_rows': v6.v18('n') or v6.v18('n_rows'), 'moves': (v5.v18('reach') or {}).v18('moves'), 'move_teach': (v5.v18('reach') or {}).v18('move_teach')}

def main(v11) -> v1:
    v12 = [v21 for v21 in v11 if not v21.v58('--')]
    if not v12:
        v44(v45)
        return 1
    v22, v23 = (v46(), [])
    for v13 in v12:
        v24 = v53(v13).v20.v47('_')
        if v24 not in v22:
            v22.v54(v24)
            v23.v55(v13)
    if v48(v23) != v48(v12):
        v44(f'note: dropped {v48(v12) - v48(v23)} duplicate report name(s)')
    v14 = ('held_out',) if '--held' in v11 else v0
    for v4 in v14:
        v25 = [v27 for v27 in (v60(v13, v4) for v13 in v23) if v27]
        if not v25:
            continue
        v44(f'\n=== {v4}  ({v48(v25)} run(s)) ===')
        v26 = ('seed', 'walk_only', 'arrive', 'stepped', 'mind', 'count', 'step_rate', 'deep_only')
        v44(''.v56((v61.v57(11) for v61 in v26)))

        def cell(v16, v49=4):
            """digits=None prints the value as it is; 0 is ZERO DECIMALS, not "no format" -
            the falsy-zero version of this line printed a recovered population as
            392.857142857."""
            if v16 is None:
                return '-'.v57(11)
            return (v63(v16) if v49 is None else f'{v16:.{v49}f}').v57(11)
        for v27 in v25:
            v44(v62(v27['seed'], None) + v62(v27['pop'], 0) + v62(v27['arrive']) + v62(v27['stepped'], None) + v62(v27['mind'], None) + v62(v27['count_rival'], None) + v62(v27['step_rate']) + v62(v27['deep_only']))
        v10 = v50((v27['pop'] for v27 in v25 if v27['pop']))
        v9 = v50((v27['stepped'] for v27 in v25 if v27['stepped'] is not None))
        v28 = v50((v27['mind'] for v27 in v25 if v27['mind'] is not None))
        v29 = v50((v27['count_rival'] for v27 in v25 if v27['count_rival'] is not None))
        v30 = v50((v27['n_rows'] for v27 in v25 if v27['n_rows']))
        v8 = v9 / v10 if v10 else v38('nan')
        v31 = v10 / v30 if v30 else v38('nan')
        v32 = [v27['deep_only'] for v27 in v25 if v27['deep_only'] is not None]
        v33 = v50(v32) / v48(v32) if v32 else v38('nan')
        v34 = v28 / v9 if v9 else v38('nan')
        v35 = v29 / v9 if v9 else v38('nan')
        v44(f'\nPOOLED  walk_only {v10:.0f} of {v30} rows ({v31:.4f})  arrive {v8:.4f}  on the stepped: mind {v34:.4f}  count {v35:.4f}  deep_only {v33:.4f}')
        v16 = []
        if not v52.v41(v8) and v8 >= 0.95:
            v16.v55(f"V1 FIRED: arrive {v8:.4f} >= 0.95 - the router already steps where staying is wrong. 34.4's lever has nothing to teach; the loss is in the pick.")
        if not v52.v41(v31) and v31 < 0.02:
            v16.v55(f'V2 FIRED: walk_only is {v31:.4f} of the exam - the masked gradient is under 2% of the signal and the arm is a slower control.')
        if not v52.v41(v33) and v33 <= 0.05:
            v16.v55(f"V3 FIRED: deep_only {v33:.4f} <= 0.05 - depth as a decision (W1) is void on this tape, by Kostya's own condition.")
        if not (v52.v41(v34) or v52.v41(v35)) and v35 >= v34:
            v16.v55(f"V4 FIRED: on walk_only the counting rival is {v35:.4f} against the mind's {v34:.4f} - the pick there is a counting problem, and the route lever is aimed at the wrong half.")
        for v36 in v16:
            v44('  ' + v36)
        if not v16:
            v44("  no void check fired: 34.4's lever may be run, under the gate in section 35.")
    return 0
if v15 == '__main__':
    raise v37(v51(v59.v11[1:]))