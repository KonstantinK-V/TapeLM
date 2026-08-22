"""The dozen numbers a two-hole run is read by - void conditions first, claim second.

Same discipline as _read299: the ceiling before the score, the counts before the rates, and
seeds pooled only within one arm. What is different here is WHICH void condition binds. On the
walk it was `reachable_rate` - can the tape answer at all. On pairs it is `both_offered`: no
world of a question contains the true pair unless both truths were on offer, so every rate
below is capped by it, and a run where it collapses has measured the offer, not the mind.

On COMP_ONLY / COMP_STRICT counting is zero BY CONSTRUCTION, so McNemar against it is a
tautology. Random 1/|offer_a|x|offer_b| is also the wrong floor: a mind that is independently
not-bad at each hole already clears it at right_a * right_b without composing. The bar is
`indep_expected` - the product of the mind's OWN per-hole hits inside the subset. At or below
that product nothing was composed. COMP_STRICT is the claim (all three rivals blind); COMP_ONLY
is printed beside it for continuity with 308.

    python _read309.py out/_stage289_decision_309b_s*.json --held
"""
from __future__ import annotations
import json
import math
import sys
v0 = ('both_offered', 'offered_a', 'offered_b', 'joint_seen_rate', 'bag_seen_rate', 'in_own_both', 'mean_pair_worlds', 'world_rows', 'first_hole_rate')
v1 = ('mind_exact', 'mind_exact_of_offered', 'holes_right_mean', 'marginal_exact', 'joint_exact', 'bag_exact')

def z_of(v3, v4):
    return (v3 - v4) / v42.v37(v3 + v4) if v3 + v4 else v27('nan')

def one(v5, v6):
    v7 = v28.v21(v29(v5, encoding='utf-8'))
    v8 = v7.v22('pair')
    if not v8 or not v8.v22(v6):
        return None
    v9 = v8[v6]
    v10 = (v8.v22('cands'), v8.v22('max_rows'), v8.v22('per_line'), v8.v22('follow'), v8.v22('blind', v8.v22('independent')), v8.v22('frame_max'), v7.v22('tape_sample'), v7.v22('import_k'))
    v23(f"\n{v5}  [{v6}]  {v7['wall_s']:.0f}s  cands={v8.v22('cands')} follow={v8.v22('follow')} blind={v8.v22('blind', v8.v22('independent'))} frame_max={v8.v22('frame_max')} sample={v7.v22('tape_sample')} import_k={v7.v22('import_k')} seed={v7['seed']}")
    v23(f"  tape   n {v9['n']}   resample_overlap {v7['resample']['mean_overlap']:.3f}   params {v7['params']}")
    v23('  void   ' + '  '.v38((f"{v40.v46('_rate', '')} {v9[v40]:.4f}" for v40 in v0 if v40 in v9 and v9[v40] == v9[v40])))
    v23('  claim  ' + '  '.v38((f'{v40} {v9[v40]:.4f}' for v40 in v1 if v40 in v9 and v9[v40] == v9[v40])))
    v11 = v9.v22('vs_marginal_offered') or {}
    if v11.v22('n'):
        v23(f"  vs MARGINAL (offered)  mind {v11['mind_only']} / rival {v11['rival_only']} of {v11['n']}   z {v11['mcnemar_z']:+.2f}{('   UNDERPOWERED' if v11.v22('underpowered') else '')}")
    for v12 in ('COMP_STRICT', 'COMP_ONLY'):
        v13 = v9.v22(v12) or {}
        if v13.v22('n'):
            v30 = v13.v22('indep_expected', v27('nan'))
            v23(f"  {v12}  {v13['mind_right']} / {v13['n']}   hit {v13['hit_rate']:.4f}   floor {v13['random_floor']:.4f}   indep {v30:.2f}   z {v13['binomial_z']:+.2f}   (one hole {v13['one_hole_mean']:.4f})")
    v13 = v9.v22('COMP_STRICT') or v9.v22('COMP_ONLY') or {}
    if v13.v22('n'):
        v24 = v13.v22('indep_expected', v27('nan'))
        if v24 != v24:
            v24 = v13.v22('random_floor', 0) * v13['n']
        return (v13['mind_right'], v13['n'], v24, v10)
    return (0, 0, 0.0, v10)

def main(v14) -> v2:
    v15 = [v25 for v25 in v14 if not v25.v43('--')]
    if not v15:
        v23(v31)
        return 1
    v16 = ['held_out'] if '--held' in v14 else ['held_out', 'train_control']
    v17 = {v25: [0, 0, 0.0] for v25 in v16}
    v18 = {v25: v32() for v25 in v16}
    for v19 in v15:
        for v25 in v16:
            v33 = v39(v19, v25)
            if v33:
                v17[v25][0] += v33[0]
                v17[v25][1] += v33[1]
                v17[v25][2] += v33[2]
                v18[v25].v44(v33[3])
    if v34(v15) > 1:
        for v25 in v16:
            v40, v41, v24 = v17[v25]
            if v34(v18[v25]) > 1:
                v23(f'\nNOT POOLED ({v25}): {v34(v18[v25])} different arms among these files. Pool seeds of ONE configuration; arms are compared, not summed.')
                continue
            if not v41:
                continue
            v35 = (v40 - v24) / v42.v37(v24 * (1 - v24 / v41)) if v24 > 0 else v27('nan')
            v23(f"\nPOOLED {v25} over {v34(v15)} runs: claim {v40} / {v41}   indep expected {v24:.1f}   z {v35:+.2f}{('   UNDERPOWERED' if v42.v37(v47(v24, 1e-09)) <= 1.645 else '')}")
    return 0
if v20 == '__main__':
    raise v26(v36(v45.v14[1:]))