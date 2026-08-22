"""A TRUTH SIGNAL THAT IS NOT "THE TOKEN THAT STOOD THERE". The missing criterion.

WHY THIS IS THE BLOCKER. Four traits of a real mind were named at step 0. Three were tested and
closed. The fourth - PRODUCE WHAT WAS NEVER OFFERED - was never tested at all, and the reason is
not the substrate: IT IS THE EXAM. Every reward in this project is "say the token that was in
the hole". A produced value that is not that token scores zero whether it is brilliant or
gibberish, so generation cannot fail here - IT CANNOT BE MEASURED. No architecture fixes that.

WHAT A CRITERION WOULD HAVE TO DO. Score a candidate WITHOUT knowing the answer, and keep
working for candidates THE OFFER NEVER PROPOSED - otherwise it is just the offer's own count
under another name, and a count cannot score a value it has never seen at such a place.

THE CANDIDATE CRITERION, and it is a count, not a heuristic: A VALUE IS WORTH SAYING IF SAYING
IT CONNECTS. Write c into this hole and the tape gains a mention of c at this place; c is then
offered wherever this place's fillers are a lens. So

    U(c) = how many positions elsewhere hold c AND sit at a place that already shares a filler
           with this one.

Nothing about the truth enters it. It is defined for ANY value in the vocabulary, including
values the offer never proposed - which is the whole point.

WHERE IT IS TESTED, and this is chosen to be the hard place: THE HOLES NEITHER CHANNEL CAN
REACH. Not in the walk's offer, not among the place's own fillers. That is precisely the subset
where a mind would have to PRODUCE, and where every number in this project is currently zero.
The truth is dropped into a pool of random vocabulary and U has to find it.

    AUC_U      the truth against `--pool` random values, scored by U
    AUC_FREQ   the same by raw corpus frequency        THE RIVAL - the dumb label-free score
    AUC_NULL   U computed against a RANDOM OTHER PLACE's fillers - same shape, wrong place.
               If this is not near 0.5 the signal is about the vocabulary, not about the hole.

  GATE  AUC_U - AUC_FREQ > 0.05 AND AUC_U - AUC_NULL > 0.05.
        Then there is a label-free criterion that reaches past the offer, generation becomes a
        measurable task for the first time, and the fourth trait is open.
        If AUC_U ~ AUC_FREQ, "useful" means "common" and we have learned that the criterion is
        the same count we already had. If AUC_U ~ 0.5, the fourth trait is closed for want of a
        criterion, not for want of a mind - and that is worth knowing exactly as much.

    python _audit363_useful.py
    python _audit363_useful.py --min-fillers 1 --pool 128
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage363_useful.json')

def auc(v4, v5):
    """one positive against many: the fraction of distractors it outranks, ties half."""
    if not v5:
        return 0.5
    v6 = v108((1.0 for v149 in v5 if v4 > v149)) + 0.5 * v108((1.0 for v149 in v5 if v4 == v149))
    return v6 / v109(v5)

def main() -> v2:
    v7 = v110.v40()
    v7.v41('--bytes', type=v2, default=30000000)
    v7.v41('--frame-max', type=v2, default=3)
    v7.v41('--min-fillers', type=v2, default=1)
    v7.v41('--lines', type=v2, default=25000)
    v7.v41('--window-lines', type=v2, default=400)
    v7.v41('--topm', type=v2, default=8)
    v7.v41('--pool', type=v2, default=64, help='random distractors per question')
    v7.v41('--match-freq', action='store_true', help="draw distractors from the truth's own frequency band (same power of two, widened until the band holds enough). The first run showed the uniform pool hands FREQ an AUC of 0.70 for free - random vocabulary is mostly hapaxes, so 'commoner than noise' already finds the truth. Matching deletes that channel BY CONSTRUCTION: what survives is the part of U that is about THIS hole, which is the only part that could ever score generation")
    v7.v41('--cap', type=v2, default=500, help='positions read per value, bounded so one frequent token cannot cost a minute. Reported.')
    v7.v41('--neigh-cap', type=v2, default=4000, help='neighbourhood places scanned when U is used FORWARD as a third channel. Reported; raise it if the top quartile looks truncated')
    v7.v41('--max-questions', type=v2, default=1500)
    v7.v41('--seed', type=v2, default=1337)
    v7.v41('--corpus', default=v143(v0))
    v8 = v7.v42()
    v9 = v3(v8.v159).v144('r', encoding='utf-8', errors='ignore').v43(v8.v44)
    v10 = [v112.v111() for v112 in v9.v102('\n') if v109(v112.v111()) >= 80]
    v11 = v10[:v2(0.7 * v109(v10))][:v8.v11]
    v12 = v113.v45(v8.v46)
    if v8.v47 and v8.v47 < v109(v11):
        v48 = v12.v114(v109(v11) - v8.v47)
        v11 = v11[v48:v48 + v8.v47]
    v49, v50, v51 = v115.v52(v11, v8.v53, v8.v54)
    if not v49:
        v105('no tape')
        return 1
    v13 = [v64(v58) for v145, v58 in v49]
    v55, v56 = ({}, v60(v64))
    for v57, v58 in v59(v13):
        for v25 in v58:
            v55[v25] = v57
            v56[v50[v25]].v117(v25)
    v14 = [v61((v50[v25] for v25 in v58)) for v58 in v13]
    v15 = v60(v61)
    for v57, v62 in v59(v14):
        for v20 in v62:
            v15[v20].v146(v57)
    v16 = v63(v56)
    v17 = {v20: v109(v56[v20]) for v20 in v16}
    v18 = {v20: v17[v20].v116() for v20 in v16}
    v19 = v60(v64)
    for v20 in v16:
        v19[v18[v20]].v117(v20)
    v21 = {}

    def offer_of(v57, v65):
        """today's substitution offer, exactly as every audit builds it"""
        v66 = v77((v50[v149] for v149 in v13[v57]))
        v66[v65] -= 1
        if v66[v65] <= 0:
            del v66[v65]
        v67 = v64(v66)[:6]
        if not v67:
            return v61()
        v68 = v77((v50[v149] for v149 in v13[v57]))
        v69 = v77()
        for v20 in v67:
            v23 = v21.v147(v20)
            if v23 is None:
                v23 = v77()
                for v72 in v56[v20]:
                    for v154 in v13[v55[v72]]:
                        if v50[v154] != v20:
                            v23[v50[v154]] += 1
                v21[v20] = v23
            for v118, v26 in v23.v148():
                v26 -= v68.v147(v118, 0)
                if v26 > 0 and v118 != v20:
                    v69[v118] += v26
        return {v118 for v118, v155 in v69.v156(v8.v106)}

    def useful(v23, v70, v71):
        """positions holding c that sit at a place already sharing a filler with this one.
        The question's OWN place never counts - writing an answer cannot corroborate itself.

        Returns BOTH counts. Positions is the raw one; DISTINCT PLACES is the same measure with
        the frequency channel largely removed by construction - a value repeated nine times at
        one place connects to ONE place, and connecting broadly is what "useful" was supposed
        to mean. The matched run showed FREQ still at 0.56-0.58, so the coarse bit_length bands
        do not delete frequency entirely; this deletes what they left."""
        v26, v119 = (0, v61())
        for v72 in v56[v23][:v8.v95]:
            v90 = v55[v72]
            if v90 != v71 and v90 in v70:
                v26 += 1
                v119.v146(v90)
        return (v26, v109(v119))

    def draw_pool(v65, v73):
        if v8.v74:
            v120, v118 = (v64(v19[v18[v65]]), 0)
            while v109(v120) < v8.v88 + 8 and v118 < 20:
                v118 += 1
                v120 += v19.v147(v18[v65] - v118, []) + v19.v147(v18[v65] + v118, [])
        else:
            v120 = v16
        v75 = []
        while v109(v75) < v8.v88:
            v20 = v120[v12.v114(v109(v120))]
            if v20 != v65 and v20 not in v73:
                v75.v117(v20)
        return v75
    v22 = [v25 for v58 in v13 for v25 in v58]
    v12.v76(v22)
    v23 = v77()
    v78, v79, v80, v81, v82, v83 = ([], [], [], [], [], [])
    v24 = []
    for v25 in v22:
        if v23['n'] >= v8.v121:
            break
        v57 = v55[v25]
        v65 = v50[v25]
        v73 = {v50[v149] for v149 in v13[v57] if v149 != v25}
        v23['seen'] += 1
        v84 = v122(v57, v65)
        if v65 in v73 or v65 in v84:
            continue
        v23['n'] += 1
        v70 = v61()
        for v20 in v73:
            v70 |= v15[v20]
        v70.v123(v57)
        if not v70:
            v23['no_neighbourhood'] += 1
            continue
        v85 = v12.v114(v109(v13))
        for v86 in v124(4):
            if v85 != v57 and v109(v14[v85]) == v109(v73):
                break
            v85 = v12.v114(v109(v13))
        v87 = v61()
        for v20 in v14[v85]:
            v87 |= v15[v20]
        v87.v123(v57)
        v88 = v125(v65, v73)
        v89 = v63(v70)
        if v109(v89) > v8.v34:
            v89 = [v89[v152] for v152 in v124(0, v109(v89), v94(1, v109(v89) // v8.v34))]
            v23['neigh_capped'] += 1
        v126, v127, v128 = (v77(), v77(), v77())
        for v90 in v89:
            v129 = v109(v14[v90] & v73)
            for v20 in v14[v90]:
                v126[v20] += 1
                v127[v20] += v129
                if v129 >= 2:
                    v128[v20] += 1
        for v20 in v64(v73) + v64(v84):
            v126.v150(v20, None)
            v127.v150(v20, None)
            v128.v150(v20, None)
        v23['u_reach'] += v65 in {v118 for v118, v155 in v126.v156(v8.v106)}
        v23['uw_reach'] += v65 in {v118 for v118, v155 in v127.v156(v8.v106)}
        v23['us_reach'] += v65 in {v118 for v118, v155 in v128.v156(v8.v106)}
        v23['us_offer'] += v109(v128)
        v23['u_offer'] += v109(v126)
        v83.v117(v151(v127.v147(v65, 0), [v127.v147(v20, 0) for v20 in v88]))
        if v109(v70) <= v8.v34:
            v130 = v108((1 for v90 in v70 if v65 in v14[v90]))
            if v126[v65] != v130:
                raise v107('forward U != neighbourhood place-count of the truth')
        v131, v132 = v133(v65, v70, v57)
        v91 = [v133(v20, v70, v57) for v20 in v88]
        v134, v135 = v133(v65, v87, v57)
        v92 = [v133(v20, v87, v57) for v20 in v88]
        v78.v117(v151(v131, [v149[0] for v149 in v91]))
        v81.v117(v151(v132, [v149[1] for v149 in v91]))
        v79.v117(v151(v17[v65], [v17[v20] for v20 in v88]))
        v80.v117(v151(v134, [v149[0] for v149 in v92]))
        v82.v117(v151(v135, [v149[1] for v149 in v92]))
        v23['truth_nonzero'] += v131 > 0
        v23['pool_all_zero'] += v136((v149[0] == 0 for v149 in v91))
        v23['null_all_zero'] += v134 == 0 and v136((v149[0] == 0 for v149 in v92))
        v23['p1'] += v136((v131 > v149[0] for v149 in v91))
        v93 = 1 + v108((1 for v149 in v91 if v149[0] >= v131))
        v23['p5'] += v93 <= 5
        v24.v117((v109(v70), v81[-1], 1.0 if v93 <= 5 else 0.0, 1.0 if v132 > 0 else 0.0))
    v26 = v94(1, v109(v78))
    v27 = {'lines': v109(v11), 'places': v109(v13), 'vocab': v109(v16), 'seen': v23['seen'], 'unreachable_questions': v23['n'], 'scored': v109(v78), 'no_neighbourhood': v23['no_neighbourhood'], 'pool': v8.v88, 'cap': v8.v95, 'min_fillers': v8.v54, 'match_freq': v137(v8.v74), 'auc_useful': v108(v78) / v26, 'auc_freq': v108(v79) / v26, 'auc_null': v108(v80) / v26, 'auc_places': v108(v81) / v26, 'auc_places_null': v108(v82) / v26, 'truth_nonzero': v23['truth_nonzero'] / v26, 'pool_all_zero': v23['pool_all_zero'] / v26, 'null_all_zero': v23['null_all_zero'] / v26, 'p_at_1': v23['p1'] / v26, 'p_at_5': v23['p5'] / v26, 'chance_p_at_5': 5.0 / (v8.v88 + 1), 'chance_p_at_1': 1.0 / (v8.v88 + 1), 'window_lines': v8.v47}
    v24.v96(key=lambda v138: v138[0])
    v28 = v94(1, v109(v24) // 4)

    def split(v97):
        v98 = [v138 for v138 in v97 if v138[3] > 0]
        v99 = [v138 for v138 in v97 if v138[3] == 0]
        return (v108((v138[1] for v138 in v98)) / v109(v98) if v98 else v157('nan'), v108((v138[1] for v138 in v99)) / v109(v99) if v99 else v157('nan'))
    v29 = [(0, 0.9, 0.0, 1.0), (0, 0.9, 0.0, 1.0), (0, 0.4, 0.0, 0.0), (0, 0.4, 0.0, 0.0)]
    v100, v101 = v102(v29)
    v30 = v108((v138[3] for v138 in v29)) / v109(v29)
    v31 = v108((v138[1] for v138 in v29)) / v109(v29)
    if v139(v30 * v100 + (1.0 - v30) * v101 - v31) > 1e-12:
        raise v107('recombination identity failed on the synthetic case')
    v27['by_neighbourhood'] = [{'quartile': v152 + 1, 'n': v109(v97), 'neigh_lo': v97[0][0], 'neigh_hi': v97[-1][0], 'auc_places': v108((v138[1] for v138 in v97)) / v109(v97), 'p_at_5': v108((v138[2] for v138 in v97)) / v109(v97), 'truth_nonzero': v108((v138[3] for v138 in v97)) / v109(v97), 'auc_when_defined': v102(v97)[0], 'auc_when_silent': v102(v97)[1]} for v152, v97 in v59([v24[0:v28], v24[v28:2 * v28], v24[2 * v28:3 * v28], v24[3 * v28:]]) if v97]
    v32, v33 = v102(v24)
    v27['auc_when_defined'] = v32
    v27['auc_when_silent'] = v33
    v27['u_reach'] = v23['u_reach'] / v26
    v27['uw_reach'] = v23['uw_reach'] / v26
    v27['us_reach'] = v23['us_reach'] / v26
    v27['us_offer'] = v23['us_offer'] / v26
    v27['auc_weighted'] = v108(v83) / v94(1, v109(v83))
    v27['u_offer'] = v23['u_offer'] / v26
    v27['neigh_capped'] = v23['neigh_capped'] / v26
    v27['neigh_cap'] = v8.v34
    v27['over_freq'] = v27['auc_useful'] - v27['auc_freq']
    v27['over_null'] = v27['auc_useful'] - v27['auc_null']
    v27['places_over_null'] = v27['auc_places'] - v27['auc_places_null']
    v1.v140.v103(parents=True, exist_ok=True)
    v1.v104(v153.v141(v27, indent=1), encoding='utf-8')
    v105(f'tape     {v109(v13)} places, {v109(v16)} values, min_fillers {v8.v54}   window {v109(v11)}   cap {v8.v95}')
    v105(f"SUBSET   {v23['n']} of {v23['seen']} questions reach NEITHER channel; {v109(v78)} scored, {v23['no_neighbourhood']} had no neighbourhood")
    v105(f"USEFUL   AUC {v27['auc_useful']:.4f}   (the truth against {v8.v88} random values)")
    v105(f"FREQ     AUC {v27['auc_freq']:.4f}   over freq {v27['over_freq']:+.4f}")
    v105(f"NULL     AUC {v27['auc_null']:.4f}   over null {v27['over_null']:+.4f}   (a random other place's neighbourhood)")
    v105(f"PLACES   AUC {v27['auc_places']:.4f}   null {v27['auc_places_null']:.4f}   over null {v27['places_over_null']:+.4f}   (distinct places, not positions)")
    v105(f"ALIVE?   truth scores nonzero {v27['truth_nonzero']:.4f}   whole pool zero {v27['pool_all_zero']:.4f}   null all zero {v27['null_all_zero']:.4f}")
    v105(f"USABLE   p@1 {v27['p_at_1']:.4f} (chance {v27['chance_p_at_1']:.4f})   p@5 {v27['p_at_5']:.4f} (chance {v27['chance_p_at_5']:.4f})   - what a REWARD would have to live on")
    v105(f"DEFINED  AUC|defined {v27['auc_when_defined']:.4f}   AUC|silent {v27['auc_when_silent']:.4f}   (label-conditioned diagnostic, never a gate)")
    v35 = v8.v106 / v94(1.0, v27['u_offer'])
    v105(f"CHANNEL  U forward: reach {v27['u_reach']:.4f}   overlap-weighted {v27['uw_reach']:.4f}   of the UNREACHABLE holes at top-{v8.v106}")
    v105(f"         out of {v27['u_offer']:.0f} candidate values - chance {v35:.4f}, so {v27['u_reach'] / v94(v35, 1e-09):.0f}x and {v27['uw_reach'] / v94(v35, 1e-09):.0f}x   ({v27['neigh_capped']:.2f} capped at {v8.v34})")
    v36 = v8.v106 / v94(1.0, v27['us_offer'])
    v105(f"         STRICT (neighbours sharing >= 2 fillers): reach {v27['us_reach']:.4f} out of {v27['us_offer']:.0f} values - chance {v36:.4f}, {v27['us_reach'] / v94(v36, 1e-09):.0f}x")
    v105(f"         AUC by overlap weight {v27['auc_weighted']:.4f} against plain {v27['auc_places']:.4f}")
    v105('BY NEIGHBOURHOOD (split on a number known before the answer):')
    for v37 in v27['by_neighbourhood']:
        v105(f"   q{v37['quartile']}  places {v37['neigh_lo']:>5}-{v37['neigh_hi']:<5} n {v37['n']:>4}   AUC {v37['auc_places']:.4f}   p@5 {v37['p_at_5']:.4f}   defined {v37['truth_nonzero']:.4f}   AUC|defined {v37['auc_when_defined']:.4f}   AUC|silent {v37['auc_when_silent']:.4f}")
    v38 = [v37['auc_when_defined'] for v37 in v27['by_neighbourhood']]
    if v38 and v94(v38) - v158(v38) < 0.08:
        v105('   -> AUC|defined is FLAT across quartiles: the trend is COVERAGE, not discrimination. The criterion is as sharp on a thin hole as on a thick one, it is simply SILENT more often - and silence is fixable with density, sharpness would not be. (label-conditioned diagnostic, never a gate)')
    if v8.v74 and v139(v27['auc_freq'] - 0.5) > 0.08:
        v105('\nMATCHING FAILED - FREQ is still ' + f"{v27['auc_freq']:.4f}" + ' on a pool meant to erase it. Widen the bands or shrink the pool before reading anything.')
    elif v8.v74:
        v105('\nRead BY NEIGHBOURHOOD: rich-and-sharp can be a task on that share; flat 2x-over-chance closes the fourth trait for want of a measure.')
    elif v27['over_freq'] > 0.05 and v27['over_null'] > 0.05:
        v105(f"\nTHERE IS A CRITERION. On the holes NEITHER channel reaches - where every number this project has printed is zero - a label-free count finds the truth in a pool of {v8.v88} at AUC {v27['auc_useful']:.4f}, beating raw frequency and its own wrong-place null. Generation can be SCORED, which is the thing it has always lacked, and the fourth trait becomes a task instead of a wish.")
    elif v27['over_null'] > 0.05:
        v105(f"\n'USEFUL' MEANS 'COMMON'. The signal is real against its null but does not beat raw frequency ({v27['auc_freq']:.4f}). It is the count we already had wearing a new name, and it cannot tell a good novel answer from a frequent one.")
    else:
        v105('\nNO CRITERION. Connecting to the neighbourhood says nothing about being right. The fourth trait is closed FOR WANT OF A MEASURE, not for want of a mind: on these holes there is no label-free way to tell a produced answer from noise, so no reward can be written and no architecture can be blamed.')
    v105(f'\nwritten to {v1}')
    return 0
if v39 == '__main__':
    raise v107(v142())