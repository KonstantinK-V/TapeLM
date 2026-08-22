"""Ladder step 1: the mind emits a constraint, the tape resolves it. Both gates, pooled.

WHAT IS BEING DECIDED. Today Phi ranks eight candidates the walk enumerated; here it chooses
WHICH OF ITS OWN ROWS to look through, and the tape answers by counting what stands with that
value over every place that holds it. Phi's output stops being a value and becomes a query.

GATE (a) - IS CHOOSING THE QUERY A DECISION, OR A COUNT?
  The mind's lens against three exact counting rules for choosing the same row: `rare` (fewest
  homes), `frequent` (most), `decisive` (the lens whose top co-occurrence takes the largest
  share of its own total - the strongest of the three, and the one the gate is set against).
  Paired, on the questions where the truth is NOT among the question's own rows, because those
  are the ones an index on the place cannot answer at all. Pass: z >= +1.645 pooled AND ahead on
  >= 3 of 4 seeds.

GATE (b) - DOES IT REACH MORE THAN THE ENUMERATION IT REPLACES?
  `answerable` is "the truth is what the tape says through SOME lens". `walk_answerable` is the
  incumbent measured ON THE SAME QUESTION: the truth among eight candidates from eight nearby
  places. Both are properties of the tape and the operation, never of the mind. Pass: ahead on
  >= 3 of 4 seeds. Reads are printed beside it - a lens costs ONE read against the walk's eight,
  so reaching more at a fifth of the cost is 335's asymptotic argument becoming a measurement.

A step is decided by these two numbers and closed. No third gate, no polish.

    python _read345_cons.py results/stage289_decision_345cons_s*.json
"""
from __future__ import annotations
import json
import math
import sys

def z_of(v1, v2):
    return (v1 - v2) / v30.v27(v1 + v2) if v1 + v2 else v18('nan')

def main(v3) -> v0:
    v4 = [v8 for v8 in v3 if not v8.v31('--')]
    if not v4:
        v12(v19)
        return 1
    v5 = ['held_out'] if '--held' in v3 else ['held_out', 'train_control']
    v6 = {v8: {'b': 0, 'c': 0, 'n': 0, 'up_a': 0, 'up_b': 0, 'seen': 0, 'void': 0, 'cov': 0} for v8 in v5}
    for v7 in v4:
        v10 = v28.v20(v29(v7, encoding='utf-8'))
        v11 = v10.v21('cons')
        if not v11:
            v12(f'{v7}: no cons block - was --constrain set?')
            continue
        for v8 in v5:
            v22 = v11.v21(v8)
            if not v22:
                continue
            v23 = v22['beyond_own']
            v24 = v23['vs_decisive']
            v12(f"\n{v7}  [{v8}]  seed={v10['seed']}  lenses {v11['lenses']}  mean {v22['mean_lenses']:.2f}")
            v12(f"  void    n {v22['n']}  beyond_own {v22['n_beyond_own']}  constrain_rate {v22['constrain_rate']:.4f}  own_hit {v22['own_hit_rate']:.4f}")
            v12(f"  GATE b  answerable {v22['answerable']:.4f} vs walk {v22['walk_answerable']:.4f}   present@topm {v22['present_topm']:.4f}   reads {v22['reads_constraint']:.0f} vs {v22['reads_walk']:.0f}   {('AHEAD' if v22['answerable'] > v22['walk_answerable'] else 'behind')}")
            v12(f"  GATE a  beyond_own hit {v23['hit']:.4f}   decisive {v23['decisive']:.4f}   rare {v23['rare']:.4f}   frequent {v23['frequent']:.4f}")
            v12(f"          vs decisive  {v24['mind_only']}/{v24['rival_only']} of {v24['n']}   z {v24['mcnemar_z']:+.2f}   |   vs walk rival {v23['vs_walk_rival']['mind_only']}/{v23['vs_walk_rival']['rival_only']} z {v23['vs_walk_rival']['mcnemar_z']:+.2f}")
            v12(f"  hit     {v22['hit_rate']:.4f}   chosen lens share when constrained {v22['chosen_share_when_constrained']:.4f}")
            if v22['constrain_rate'] < 0.02:
                v12(f"  VOID    constrain_rate {v22['constrain_rate']:.4f}: the lens is taken on {v22['constrain_rate'] * v22['n']:.0f} of {v22['n']} questions, so gate (a) is not measurable here. Gate (b) still is - it is a property of the tape and the operation, not of the mind.")
                v6[v8]['void'] += 1
            v6[v8]['b'] += v24['mind_only']
            v6[v8]['c'] += v24['rival_only']
            v6[v8]['n'] += v24['n']
            v6[v8]['up_a'] += v0(v24['mind_only'] > v24['rival_only'])
            v6[v8]['up_b'] += v0(v22['answerable'] > v22['walk_answerable'])
            v6[v8]['cov'] += v0(v22['present_topm'] > v22['walk_answerable'])
            v6[v8]['seen'] += 1
    v12()
    for v8 in v5:
        v13 = v6[v8]
        if not v13['seen']:
            continue
        v14 = v25(v13['b'], v13['c'])
        v15 = v14 >= 1.645 and v13['up_a'] >= 3 and (v13['void'] == 0)
        v16 = v13['up_b'] >= 3
        v12(f"POOLED {v8} over {v13['seen']} runs")
        v12(f"  GATE a  mind {v13['b']} / decisive {v13['c']} of {v13['n']}   z {v14:+.2f}   ahead on {v13['up_a']}/{v13['seen']} seeds   -> {('PASS' if v15 else 'VOID' if v13['void'] else 'FAIL')}" + (f" ({v13['void']}/{v13['seen']} runs never took the lens)" if v13['void'] else ''))
        v12(f"  GATE b  reaches more than the walk on {v13['up_b']}/{v13['seen']} seeds   -> {('PASS' if v16 else 'FAIL')}")
        v12(f"          coverage at a MATCHED offer (top{{m}} vs the walk's eight): ahead on {v13['cov']}/{v13['seen']} seeds")
        if v15 and v16:
            v12('  STEP 1 PASSES - the mind chooses the query better than any count AND the constraint reaches further than the enumeration. Go to step 2 (composition, retried as an intersection of two constraints).')
        elif v16 and (not v15):
            v12('  THE INTERFACE WORKS AND THE CHOICE IS A COUNT. Keep the constraint, drop Phi from the lens choice, and step 2 is built on the counted lens - the tape took another decision back, as it did in 338.')
        elif v15 and (not v16):
            v12("  THE CHOICE IS A DECISION AND THE INTERFACE REACHES NO FURTHER. Phi ranks queries better than counting, but the operation does not beat enumeration, so 335's wall stands and step 2 has nothing new to stand on.")
        elif v13['cov'] >= 3:
            v12("  THE LENS REACHES, THE TAPE'S ARGMAX LOSES IT. Coverage at a matched offer is ahead of the walk while the resolved answer is behind: the operation finds the truth and the resolution rule discards it. That is 317 exactly - raw counts favour globally frequent fillers while these truths are rare by construction. Re-run with --cons-resolve share. This REPLACES the capacity retry; it does not add a cycle.")
        else:
            v12('  STEP 1 FAILS both gates. One retry, then the step is closed and the constraint interface is not the way past 335.')
    return 0
if v9 == '__main__':
    raise v17(v26(v32.v3[1:]))