"""338: WHAT SHOULD THE TAPE KEEP - four rules at one budget, read side by side.

WHY RETENTION AND NOT MORE CORPUS. 335 measured the tape holding 0.807 of the hidden truths
while the offer showed the mind 0.217 of them, and swept both of the tape's sizes. Across WIDTH
the gap opened monotonically, 0.369 -> 0.590, with the share shown FALLING as the tape grew;
across depth at fixed width it was non-monotone and saturated. Growing the corpus makes the
enumeration worse, not better. So the remaining lever is which places are kept at all - and
that is a decision, which belongs to the mind's half of the split.

WHAT IS COMPARED. Every run draws its questions from the WHOLE tape and differs only in which
places the walk may visit and offer from, capped at the same N. Four rules:

    --retain-by random   what the tape does today, and the control the others must beat
    --retain-by own      the places with the most mentions
    --retain-by share    the places whose single most frequent filler dominates them
    --retain-by mind     the places where Phi answers with the widest margin

THE NUMBER THAT DECIDES is `reachable_rate` at matched N: of the same questions, how many have
their truth somewhere in the walk's offer. It is a property of the tape and the walk, with no
sampling noise once the seed is fixed - the only draw is the tape itself. So the statistic is
SIGN CONSISTENCY ACROSS SEEDS, not a z: a rule wins if it is ahead of `random` on at least 3
of 4 seeds. `hit_rate` is printed beside it because reaching more is worthless if the mind
cannot then pick, and a retention rule that raises reach while lowering the pick has moved the
problem rather than solved it.

THE GUARD, written before the first run and not negotiable afterwards. `mind` requires a frozen
transplanted Phi (the stage refuses otherwise). The test is the transplant: choose what a NEWS
tape keeps using a mind fitted only to wiki. If the foreign tape comes out no worse than the
counting rules build it, the retention policy is not corpus-specific and the invariant holds.
If it comes out worse, the idea is dropped whole. There is no rescuing patch for this one.

    python _read338_retention.py out/_stage289_decision_338*_s1337.json
    python _read338_retention.py out/_stage289_decision_338*.json --held
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
v0 = ('reachable_rate', 'ceiling', 'own_hit_rate', 'walk_only_rate', 'hit_rate', 'hit_of_walk_only', 'step_rate')

def main(v2) -> v1:
    v3 = [v9 for v9 in v2 if not v9.v41('--')]
    if not v3:
        v19(v23)
        return 1
    v4 = ['held_out'] if '--held' in v2 else ['held_out', 'train_control']
    v5 = v11(v12)
    v13, v14 = (v24(), v24())
    for v6 in v3:
        v15 = v37.v25(v38(v6, encoding='utf-8'))
        v16 = v15.v26('reach') or {}
        v17 = v15.v26('retain_by') if v15.v26('retain') else 'whole tape'
        if v15.v26('retain'):
            v13.v27(v15['retain'])
        v14.v27(((v15.v26('tape_shape') or {}).v26('held_out') or {}).v26('addresses'))
        for v9 in v4:
            v28 = v16.v26(v9)
            if v28:
                v5[v9, v17][v15['seed']] = v28
    if v29(v13) > 1:
        v19(f'NOT COMPARED: {v18(v13)} different --retain budgets among these files. Retention rules are compared at ONE N; anything else is comparing tape sizes.')
        return 1
    v7 = (v18(v13) or ['all'])[0]
    v8 = v18((v30 for v30 in v14 if v30))
    v19(f'tape   {v8} addresses built, {v7} retained ({v7 / v8[0]:.2f} of the tape)' if v8 and v42(v7, v1) else f'tape   {v7} retained')
    v19(f'       {v29(v3)} files')
    for v9 in v4:
        v20 = v18({v28 for v45, v28 in v5 if v45 == v9})
        if not v20:
            continue
        v19(f"\n[{v9}]  {'rule':<12}" + ''.v43((f"{v39.v48('_rate', ''):>15}" for v39 in v0)))
        for v17 in v20:
            v31 = v5[v9, v17]
            v32 = {v39: v40((v47.v26(v39, v49('nan')) for v47 in v31.v50())) / v29(v31) for v39 in v0}
            v19(f'       {v17:<12}' + ''.v43((f'{v32[v39]:15.4f}' for v39 in v0)) + f'   ({v29(v31)} seeds)')
        v21 = v5[v9, 'random'] or v5[v9, 'whole tape']
        if not v21:
            v19('       no `random` control among these files: nothing to compare against')
            continue
        for v17 in v20:
            if v17 in ('random', 'whole tape'):
                continue
            v31 = v5[v9, v17]
            v33 = v18(v24(v31) & v24(v21))
            if not v33:
                continue
            v34 = v40((1 for v46 in v33 if v31[v46]['reachable_rate'] > v21[v46]['reachable_rate']))
            v35 = v40((1 for v46 in v33 if v31[v46]['hit_rate'] > v21[v46]['hit_rate']))
            v19(f"       {v17:<12} vs random: reach ahead on {v34}/{v29(v33)} seeds, hit ahead on {v35}/{v29(v33)}{('   PASSES (>=3 of 4)' if v34 >= 3 and v29(v33) >= 4 else '')}")
    v19('\nreachable = truth somewhere in the offer; hit = the mind said it. A rule that raises the first and lowers the second has moved the problem.')
    return 0
if v10 == '__main__':
    raise v22(v36(v44.v2[1:]))