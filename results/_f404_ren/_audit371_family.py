"""IS THE SPACE OF RELATIONS BIGGER THAN THE THREE WE WROTE BY HAND?

THE QUESTION, AND IT IS THE LAST STRUCTURAL ONE THIS DESIGN HAS. The tape has three relations
and a human wrote all three: substitution (places with a similar filler fingerprint), recall
(this place itself), connection (places sharing a filler, overlap-weighted). Phi chooses among
their OUTPUTS and has never chosen WHAT TO COUNT. Scaling the corpus gives a better index, not
a different mind, for exactly that reason.

But all three are the SAME SHAPE: a rule that scores OTHER PLACES given THIS place, after which
their fillers are read off. That shape is enumerable. So before anything is made learnable, the
question is whether there is a space to learn IN:

    if a perfect chooser over a family of relations reaches far more than the best single one,
    there is something to learn and the relation set is a real degree of freedom.
    If the oracle sits on top of the best single relation, the three are one relation in three
    costumes, there is nothing to hand over, and this design is finished.

THE FAMILY, every member a COUNT with no fitted constant. Each returns a scored list of OTHER
places; the offer is the fillers of the top ones, at the same top-m for every member:

    own            this place's other rows                          (recall, channel 2)
    share_k        places sharing >= k fillers, k = 1, 2, 3         (connection, generalised)
    share_w        sharing, weighted by how many fillers            (365's winner)
    rare_w         sharing, weighted by 1/frequency of the shared filler - a shared RARE word
                   says more than a shared common one, and that is a count, not a hyperparameter
    common_w       the opposite weighting, kept because a family with only the plausible
                   members in it is a family chosen by me
    cos_k          the k nearest by frame fingerprint               (substitution, channel 1)
    two_hop        places sharing a filler with a place that shares a filler with this one
    same_line      places whose mentions sit on lines this place's mentions sit on (304 closed
                   the line CHANNEL; here it is one member of a family and costs nothing to ask)
    len_match      places with the same number of distinct fillers - a pure structure relation
                   that knows nothing about content, in as the family's own null

  GATE  on the own-fails cut: oracle_when_own_fails - null_when_own_fails > 0.05.
        Whole-population oracle - oracle_null is still printed (it failed: twelve shots, not
        twelve relations). The cut is defined by recall's outcome, identically for both arms;
        the absolute numbers are conditional, same discipline as 363's AUC|defined.
  Also reported: how MANY members are ever the sole one that reaches (a family where one member
  is nearly always the only winner is a family with one useful relation and a tail).

    python _audit371_family.py
    python _audit371_family.py --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage371_family.json')

def main() -> v2:
    v4 = v83.v33()
    v4.v34('--bytes', type=v2, default=30000000)
    v4.v34('--frame-max', type=v2, default=3)
    v4.v34('--min-fillers', type=v2, default=1)
    v4.v34('--lines', type=v2, default=25000)
    v4.v34('--window-lines', type=v2, default=400)
    v4.v34('--topm', type=v2, default=8, help='values offered, the SAME for every member - a relation may not win on budget')
    v4.v34('--places-k', type=v2, default=8, help='places a relation may return')
    v4.v34('--neigh-cap', type=v2, default=4000)
    v4.v34('--max-questions', type=v2, default=1500)
    v4.v34('--seed', type=v2, default=1337)
    v4.v34('--corpus', default=v107(v0))
    v5 = v4.v35()
    v6 = v3(v5.v123).v108('r', encoding='utf-8', errors='ignore').v36(v5.v37)
    v7 = [v85.v84() for v85 in v6.v109('\n') if v48(v85.v84()) >= 80]
    v8 = v7[:v2(0.7 * v48(v7))][:v5.v8]
    v9 = v86.v38(v5.v39)
    if v5.v40 and v5.v40 < v48(v8):
        v41 = v9.v87(v48(v8) - v5.v40)
        v8 = v8[v41:v41 + v5.v40]
    v42, v43, v44 = v88.v45(v8, v5.v46, v5.v47)
    if not v42:
        v79('no tape')
        return 1
    v10 = [v59(v50) for v110, v50 in v42]
    v11 = v48(v10)
    v12 = {}
    for v49, v50 in v51(v10):
        for v25 in v50:
            v12[v25] = v49
    v13 = [v53((v43[v25] for v25 in v50)) for v50 in v10]
    v14 = [v56((v43[v25] for v25 in v50)) for v50 in v10]
    v15 = v52(v53)
    for v49, v54 in v51(v13):
        for v55 in v54:
            v15[v55].v111(v49)
    v16 = v56((v43[v25] for v50 in v10 for v25 in v50))
    v17 = [v53((v44[v25] for v25 in v50)) for v50 in v10]
    v18 = v52(v53)
    for v49, v57 in v51(v17):
        for v58 in v57:
            v18[v58].v111(v49)
    v19 = v52(v59)
    for v49, v54 in v51(v13):
        v19[v48(v54)].v89(v49)

    def neighbours(v49, v60):
        """every place sharing at least one filler, with HOW MANY it shares and which."""
        v61 = v52(v53)
        for v55 in v60:
            for v64 in v15[v55]:
                if v64 != v49:
                    v61[v64].v111(v55)
        return v61

    def members(v49, v60, v61):
        v62 = {}
        v62['share_1'] = {v64: v48(v112) for v64, v112 in v61.v119()}
        v62['share_2'] = {v64: v48(v112) for v64, v112 in v61.v119() if v48(v112) >= 2}
        v62['share_3'] = {v64: v48(v112) for v64, v112 in v61.v119() if v48(v112) >= 3}
        v62['share_w'] = {v64: v48(v112) / v48(v13[v64]) for v64, v112 in v61.v119()}
        v62['rare_w'] = {v64: v113((1.0 / v16[v55] for v55 in v112)) for v64, v112 in v61.v119()}
        v62['common_w'] = {v64: v113((v124(v16[v55]) for v55 in v112)) for v64, v112 in v61.v119()}
        v62['mention_w'] = {v64: v113((v125(v14[v49][v55], v14[v64][v55]) for v55 in v112)) for v64, v112 in v61.v119()}
        v62['cos_k'] = {v64: v48(v112) / v74(1, v48(v60 | v13[v64]) - v48(v112)) for v64, v112 in v61.v119()}
        v63 = v56()
        for v64 in v59(v61)[:64]:
            for v55 in v13[v64]:
                for v114 in v15[v55]:
                    if v114 != v49 and v114 not in v61:
                        v63[v114] += 1
        v62['two_hop'] = v90(v63)
        v65 = v56()
        for v58 in v17[v49]:
            for v64 in v18[v58]:
                if v64 != v49:
                    v65[v64] += 1
        v62['same_line'] = v90(v65)
        v66 = v19.v91(v48(v60), ())
        v62['len_match'] = {v64: 1 for v64 in v59(v66)[:v5.v76 * 4] if v64 != v49}
        return v62
    v20 = [v25 for v50 in v10 for v25 in v50]
    v9.v67(v20)
    v21 = ['own', 'share_1', 'share_2', 'share_3', 'share_w', 'rare_w', 'common_w', 'mention_w', 'cos_k', 'two_hop', 'same_line', 'len_match']
    v22 = v56()
    v23 = v56()
    v24 = v56()
    for v25 in v20:
        if v24['n'] >= v5.v92:
            break
        v49 = v12[v25]
        v68 = v43[v25]
        v60 = {v43[v115] for v115 in v10[v49] if v115 != v25}
        if not v60:
            continue
        v24['n'] += 1
        v69 = {'own': v68 in {v120 for v120, v127 in v56((v43[v115] for v115 in v10[v49] if v115 != v25)).v128(v5.v75)}}
        v61 = v93(v49, v60)
        if v48(v61) > v5.v94:
            v95 = v80(v61, key=lambda v64: -v48(v61[v64]))[:v5.v94]
            v61 = {v64: v61[v64] for v64 in v95}
        v70 = v96(v49, v60, v61)
        for v31 in v21:
            if v31 == 'own':
                continue
            v97 = v70.v91(v31, {})
            v98 = v80(v97, key=lambda v64: (-v97[v64], v64))[:v5.v76]
            v99 = v56()
            for v64 in v98:
                for v55, v26 in v14[v64].v119():
                    if v55 not in v60:
                        v99[v55] += v26
            v69[v31] = v68 in {v120 for v120, v127 in v99.v128(v5.v75)}
        v71 = {'own': v69['own']}
        for v72 in v100(v48(v21) - 1):
            v101 = [v9.v87(v11) for v121 in v100(v5.v76)]
            v99 = v56()
            for v64 in v101:
                if v64 == v49:
                    continue
                for v55, v122 in v14[v64].v119():
                    if v55 not in v60:
                        v99[v55] += v122
            v71[f'null_{v72}'] = v68 in {v120 for v120, v127 in v99.v128(v5.v75)}
        v24['oracle_null'] += v102(v71.v116())
        if not v69['own']:
            v24['nf'] += 1
            v24['oracle_nf'] += v102((v69[v31] for v31 in v21))
            v24['null_nf'] += v102((v71[v126] for v126 in v71 if v126 != 'own'))
        for v31 in v21:
            v22[v31] += v69[v31]
        v73 = [v31 for v31 in v21 if v69[v31]]
        v24['oracle'] += v103(v73)
        if v48(v73) == 1:
            v23[v73[0]] += 1
            v24['sole'] += 1
    v26 = v74(1, v24['n'])
    v27 = {v31: v22[v31] / v26 for v31 in v21}
    v28 = v74(v27, key=v27.v91)
    v29 = {'lines': v48(v8), 'places': v11, 'questions': v24['n'], 'topm': v5.v75, 'places_k': v5.v76, 'min_fillers': v5.v47, 'per_relation': v27, 'best': v28, 'best_reach': v27[v28], 'oracle': v24['oracle'] / v26, 'oracle_null': v24['oracle_null'] / v26, 'sole_winner_rate': v24['sole'] / v26, 'sole_by': {v31: v23[v31] / v26 for v31 in v21 if v23[v31]}}
    v29['over_best'] = v29['oracle'] - v29['best_reach']
    v29['over_null_family'] = v29['oracle'] - v29['oracle_null']
    v30 = v74(1, v24['nf'])
    v29['own_fails'] = v24['nf'] / v26
    v29['oracle_when_own_fails'] = v24['oracle_nf'] / v30
    v29['null_when_own_fails'] = v24['null_nf'] / v30
    v29['over_null_when_own_fails'] = v29['oracle_when_own_fails'] - v29['null_when_own_fails']
    v1.v104.v77(parents=True, exist_ok=True)
    v1.v78(v117.v105(v29, indent=1), encoding='utf-8')
    v79(f"tape     {v11} places, {v24['n']} questions, topm {v5.v75}, places_k {v5.v76}")
    for v31 in v80(v21, key=lambda v115: -v27[v115]):
        v81 = '  <- null' if v31 == 'len_match' else ''
        v79(f'   {v31:<11} {v27[v31]:.4f}{v81}')
    v79(f"BEST     {v28} {v29['best_reach']:.4f}")
    v79(f"ORACLE   {v29['oracle']:.4f}   over best {v29['over_best']:+.4f}")
    v79(f"NULL FAM {v29['oracle_null']:.4f}   over it {v29['over_null_family']:+.4f}   ({v48(v21) - 1} random place-sets + own, the SAME number of shots)")
    v79(f"OWN FAILS on {v29['own_fails']:.4f} of questions - THE ONLY ONES A PLACE RELATION COULD MATTER ON:")
    v79(f"   relations {v29['oracle_when_own_fails']:.4f}   random family {v29['null_when_own_fails']:.4f}   over it {v29['over_null_when_own_fails']:+.4f}")
    v79(f"SOLE     {v29['sole_winner_rate']:.4f} of questions are reached by EXACTLY ONE member: " + '  '.v118((f'{v126} {v55:.3f}' for v126, v55 in v80(v29['sole_by'].v119(), key=lambda v129: -v129[1])[:6])))
    if v29['over_null_when_own_fails'] > 0.05:
        v79(f"\nTHE RELATIONS CARRY, AND THE WHOLE-POPULATION GATE WAS DILUTED BY RECALL. On the {v29['own_fails']:.0%} of questions recall cannot answer, a perfect chooser over the family reaches {v29['oracle_when_own_fails']:.4f} against {v29['null_when_own_fails']:.4f} for the same number of RANDOM place-sets: {v29['over_null_when_own_fails']:+.4f}. That is a real degree of freedom on the only questions where a relation could ever show, and 372 is worth building.")
    elif v29['over_best'] > 0.05 and v29['over_null_family'] > 0.05:
        v79(f"\nTHERE IS A SPACE TO LEARN IN. A perfect chooser over the family reaches {v29['oracle']:.4f} against the best single relation's {v29['best_reach']:.4f}. The relation set is a real degree of freedom, the three we wrote by hand are not the whole of it, and 372 is worth building: the relation arrives as EVIDENCE ON THE CANDIDATE, not as a second head - the only shape that avoids the 4x law and the lane problem 367 measured.")
    elif v29['over_best'] > 0.05:
        v79(f"\nSHOTS ON GOAL, NOT RELATIONS. The oracle beats the best single member by {v29['over_best']:+.4f}, but a family of RANDOM place-sets of the same size reaches {v29['oracle_null']:.4f} - within {v29['over_null_family']:+.4f}. The gain is twelve tries at top-8, not twelve relations. Nothing to hand over.")
    else:
        v79(f"\nNO SPACE. The oracle over twelve relations sits {v29['over_best']:+.4f} above the best single one. They are one relation in twelve costumes, there is nothing to hand over to Phi, and letting go of the hand-written set would change nothing. This design is finished at what it already measures.")
    v79(f'\nwritten to {v1}')
    return 0
if v32 == '__main__':
    raise v82(v106())