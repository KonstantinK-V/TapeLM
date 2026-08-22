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

def z_of(b, c):
    return (b - c) / math.sqrt(b + c) if b + c else float('nan')

def main(argv) -> int:
    files = [a for a in argv if not a.startswith('--')]
    if not files:
        print(__doc__)
        return 1
    arms = ['held_out'] if '--held' in argv else ['held_out', 'train_control']
    pool = {a: {'b': 0, 'c': 0, 'n': 0, 'up_a': 0, 'up_b': 0, 'seen': 0, 'void': 0, 'cov': 0} for a in arms}
    for f in files:
        d = json.load(open(f, encoding='utf-8'))
        cb = d.get('cons')
        if not cb:
            print(f'{f}: no cons block - was --constrain set?')
            continue
        for a in arms:
            r = cb.get(a)
            if not r:
                continue
            bo = r['beyond_own']
            vd = bo['vs_decisive']
            print(f"\n{f}  [{a}]  seed={d['seed']}  lenses {cb['lenses']}  mean {r['mean_lenses']:.2f}")
            print(f"  void    n {r['n']}  beyond_own {r['n_beyond_own']}  constrain_rate {r['constrain_rate']:.4f}  own_hit {r['own_hit_rate']:.4f}")
            print(f"  GATE b  answerable {r['answerable']:.4f} vs walk {r['walk_answerable']:.4f}   present@topm {r['present_topm']:.4f}   reads {r['reads_constraint']:.0f} vs {r['reads_walk']:.0f}   {('AHEAD' if r['answerable'] > r['walk_answerable'] else 'behind')}")
            print(f"  GATE a  beyond_own hit {bo['hit']:.4f}   decisive {bo['decisive']:.4f}   rare {bo['rare']:.4f}   frequent {bo['frequent']:.4f}")
            print(f"          vs decisive  {vd['mind_only']}/{vd['rival_only']} of {vd['n']}   z {vd['mcnemar_z']:+.2f}   |   vs walk rival {bo['vs_walk_rival']['mind_only']}/{bo['vs_walk_rival']['rival_only']} z {bo['vs_walk_rival']['mcnemar_z']:+.2f}")
            print(f"  hit     {r['hit_rate']:.4f}   chosen lens share when constrained {r['chosen_share_when_constrained']:.4f}")
            if r['constrain_rate'] < 0.02:
                print(f"  VOID    constrain_rate {r['constrain_rate']:.4f}: the lens is taken on {r['constrain_rate'] * r['n']:.0f} of {r['n']} questions, so gate (a) is not measurable here. Gate (b) still is - it is a property of the tape and the operation, not of the mind.")
                pool[a]['void'] += 1
            pool[a]['b'] += vd['mind_only']
            pool[a]['c'] += vd['rival_only']
            pool[a]['n'] += vd['n']
            pool[a]['up_a'] += int(vd['mind_only'] > vd['rival_only'])
            pool[a]['up_b'] += int(r['answerable'] > r['walk_answerable'])
            pool[a]['cov'] += int(r['present_topm'] > r['walk_answerable'])
            pool[a]['seen'] += 1
    print()
    for a in arms:
        e = pool[a]
        if not e['seen']:
            continue
        z = z_of(e['b'], e['c'])
        ga = z >= 1.645 and e['up_a'] >= 3 and (e['void'] == 0)
        gb = e['up_b'] >= 3
        print(f"POOLED {a} over {e['seen']} runs")
        print(f"  GATE a  mind {e['b']} / decisive {e['c']} of {e['n']}   z {z:+.2f}   ahead on {e['up_a']}/{e['seen']} seeds   -> {('PASS' if ga else 'VOID' if e['void'] else 'FAIL')}" + (f" ({e['void']}/{e['seen']} runs never took the lens)" if e['void'] else ''))
        print(f"  GATE b  reaches more than the walk on {e['up_b']}/{e['seen']} seeds   -> {('PASS' if gb else 'FAIL')}")
        print(f"          coverage at a MATCHED offer (top{{m}} vs the walk's eight): ahead on {e['cov']}/{e['seen']} seeds")
        if ga and gb:
            print('  STEP 1 PASSES - the mind chooses the query better than any count AND the constraint reaches further than the enumeration. Go to step 2 (composition, retried as an intersection of two constraints).')
        elif gb and (not ga):
            print('  THE INTERFACE WORKS AND THE CHOICE IS A COUNT. Keep the constraint, drop Phi from the lens choice, and step 2 is built on the counted lens - the tape took another decision back, as it did in 338.')
        elif ga and (not gb):
            print("  THE CHOICE IS A DECISION AND THE INTERFACE REACHES NO FURTHER. Phi ranks queries better than counting, but the operation does not beat enumeration, so 335's wall stands and step 2 has nothing new to stand on.")
        elif e['cov'] >= 3:
            print("  THE LENS REACHES, THE TAPE'S ARGMAX LOSES IT. Coverage at a matched offer is ahead of the walk while the resolved answer is behind: the operation finds the truth and the resolution rule discards it. That is 317 exactly - raw counts favour globally frequent fillers while these truths are rare by construction. Re-run with --cons-resolve share. This REPLACES the capacity retry; it does not add a cycle.")
        else:
            print('  STEP 1 FAILS both gates. One retry, then the step is closed and the constraint interface is not the way past 335.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))