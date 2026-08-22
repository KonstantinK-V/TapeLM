"""THE CEILING OF CHOOSING A MOVE. Measured before a better chooser is built.

385 and 386 both failed their gate, and both failed the same way: the more the mind used a move
other than `step`, the worse the arm. 386 added `move_hit` and it named the fault precisely - on
three seeds of four the hit of `share` was BELOW the hit of `step`, so the mind was picking the
second move on the questions where that move is worse. The probe is one row, and one row is
evidently not enough to tell which lane will pay.

BUT THE LOSS IS NOT THE PRICE OF COMMITTING. The merged offer is the two lanes interleaved and
cut at eight, so `merged` is a SUBSET of `step_top8 | share_top8`. A perfect move chooser
therefore reaches at least as much as the interleave, always, by construction - while the
measured reach FELL on three seeds of four (.6525 -> .5943). Every point of that gap is chooser
error, not the cost of choosing. So there is headroom above the interleave, and the question is
how much.

    reach_step      the truth in the fingerprint walk's eight
    reach_share     the truth in the connect lane's eight
    reach_merged    the truth in the interleaved eight            TODAY
    reach_oracle    the truth in EITHER lane                      A PERFECT CHOOSER
    only_share      the truth in the share lane and NOT in step   WHERE THE MOVE DECIDES
    overlap         mean Jaccard of the two eights                ARE THEY EVEN DIFFERENT
    reach_random    K random places                               the floor

THE SUSPECT, DECLARED BEFORE THE RUN. The arm runs with `fp=fillers`, so a place's fingerprint
IS its bag of fillers - and the cosine between two filler profiles is ZERO unless they share a
filler. `connect` walks places that share a filler, weighted by how many. So both lanes draw
from THE SAME NEIGHBOUR SET and differ only in how they RANK it: cosine of the profile against
count of the overlap. If that is right, `only_share` is small, the two moves are one move with
two orderings, and no chooser can pay - the fix would be a move that reaches somewhere the walk
cannot, not a better way of picking between two views of the same neighbourhood.

  GATE  oracle - merged > 0.05 AND only_share > 0.05.
        Then a perfect choice is worth having and the next lever is the CHOOSER - the one-row
        probe is the thing to replace. If only_share is small the lanes are redundant, the move
        as an output space is closed on this tape, and what is needed is a genuinely different
        move.

  Measured on the population where reach matters: the truth is NOT among the question's own
  values. Own values are excluded from both lanes, the question's own place from both
  neighbourhoods, and the hidden position from everything.

    python _audit387_moves.py
    python _audit387_moves.py --window-lines 1600 --places 16
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import _tape_frames as tframes
v0 = v4('data/_wikitext103_train.txt')
v1 = v4('results/_stage387_moves.json')
v2 = (8, 16, 32, 64, 128)

def main() -> v3:
    v5 = v85.v27()
    v5.v28('--bytes', type=v3, default=30000000)
    v5.v28('--frame-max', type=v3, default=3)
    v5.v28('--min-fillers', type=v3, default=1)
    v5.v28('--lines', type=v3, default=25000)
    v5.v28('--window-lines', type=v3, default=400)
    v5.v28('--places', type=v3, default=8)
    v5.v28('--topm', type=v3, default=8)
    v5.v28('--max-questions', type=v3, default=3000)
    v5.v28('--seed', type=v3, default=1337)
    v5.v28('--corpus', default=v124(v0))
    v6 = v5.v29()
    v7 = v4(v6.v75).v125('r', encoding='utf-8', errors='ignore').v30(v6.v31)
    v8 = [v87.v86() for v87 in v7.v126('\n') if v99(v87.v86()) >= 80]
    v9 = v8[:v3(0.7 * v99(v8))][:v6.v9]
    v10 = v88.v32(v6.v33)
    if v6.v34 and v6.v34 < v99(v9):
        v35 = v10.v89(v99(v9) - v6.v34)
        v9 = v9[v35:v35 + v6.v34]
    v36, v37, v38 = v90.v39(v9, v6.v40, v6.v41)
    if not v36:
        v79('no tape')
        return 1
    v11 = [v45(v17) for v127, v17 in v36]
    v12 = {}
    for v42, v17 in v43(v11):
        for v20 in v17:
            v12[v20] = v42
    v13 = [v59((v37[v20] for v20 in v17)) for v17 in v11]
    v14 = [v137.v128(v135((v65 * v65 for v65 in v46.v144()))) or 1.0 for v46 in v13]
    v15 = v44(v45)
    for v42, v46 in v43(v13):
        for v47 in v46:
            v15[v47].v101(v42)

    def neighbours(v42, v48):
        """every place sharing at least one filler, with the shared-filler COUNT.

        The cosine of two filler profiles is zero unless they share a filler, so this is not a
        shortcut for the walk - it is the walk's entire non-zero support, and the connect
        channel's neighbourhood is the same set. That identity is the thing being measured.

        THE QUERY'S PROFILE IS PASSED IN, AND THE FIRST VERSION OF THIS FUNCTION READ
        `prof[pid]` - WHICH STILL CONTAINS THE HIDDEN TOKEN. The truth was therefore one of the
        keys its own search was run from: every place holding it got overlap credit, and the
        lane then accumulated score for it out of those places. The stage never did this -
        `reach_connect` builds its lens from `slots[:query_row]`, the hidden row excluded - so
        the audit was measuring a channel the arm does not have. Every share number printed
        before this fix is inflated by it.
        """
        v91, v92 = (v59(), v59())
        for v47, v65 in v48.v61():
            for v52 in v15[v47]:
                if v52 != v42:
                    v91[v52] += 1
                    v92[v52] += v65 * v13[v52][v47]
        return (v91, v92)

    def lane_step(v42, v49, v48):
        """the walk's lane IN FULL - every filler of the K nearest places, in place order.

        Returned uncut so the RANK of the truth can be read. The eight the arm actually offers
        is out[:topm]; anything past that is what the cap is throwing away.
        """
        v93, v92 = v94(v42, v48)
        v50 = v137.v128(v135((v65 * v65 for v65 in v48.v144()))) or 1.0
        v51 = v98(v92, key=lambda v52: (-(v92[v52] / (v50 * v14[v52])), v52))[:v6.v11]
        v53, v95 = ([], v129(v49))
        for v52 in v51:
            for v47, v130 in v13[v52].v131():
                if v47 not in v95:
                    v95.v139(v47)
                    v53.v101(v47)
        return v53

    def lane_share(v42, v49, v48, v54=False):
        """connect's lane in full, ranked two ways.

        Note the ASYMMETRY, which is not a bug and is worth having written down: `step` sees the
        fillers of K places, while `share` scores every place that shares a filler at all. The
        two moves were never the same size of read.

        SUM AGAINST SHARE, AND THIS PROJECT HAS MADE THE SAME MISTAKE TWICE ALREADY. 365's rule
        is score(v) = the SUM of the overlaps of every neighbouring place holding v, so a value
        standing at many places accumulates score for being COMMON. The depth block says what
        that costs: the truth is in this lane .78-.88 of the time and sits at mean rank 67 (315
        at w1600), and on this population the truth is RARE by construction - the question is
        here precisely because its answer is not among its own values. 317 found exactly this in
        cons_resolve, measured the raw-count rival at 2/69 = 0.029 against a one-place rule's
        0.222, and fixed it by dividing; 383 found it again in the count rival. Third appearance.

        Dividing by HOW MANY PLACES hold v turns the sum into the MEAN OVERLAP of the places
        where v stands - "on average, how related to my place are the places this value lives
        at". Each place contributes exactly once to both halves, so the ratio is two exact
        counts and nothing is fitted.

        THE CONTROL THIS NEEDS, and it is not optional: dividing by frequency PREFERS RARE
        VALUES, and the truth here is rare. A frequency-matched decoy is ranked the same two ways
        so that "the truth moved up" can be told apart from "everything rare moved up".
        """
        v91, v96 = v94(v42, v48)
        v55 = v59()
        for v52, v97 in v91.v61():
            for v47 in v13[v52]:
                if v47 not in v49:
                    v55[v47] += v97
        if not v54:
            return [v47 for v47, v140 in v55.v131()]
        return v98(v55, key=lambda v47: (-(v55[v47] / v99(v15[v47])), -v55[v47], v47))

    def lane_random(v42, v49, v56):
        v53, v95 = ([], v129(v49))
        v57 = v99(v11)
        for v58 in v100(v6.v11):
            v52 = v10.v89(v57)
            if v52 == v42:
                continue
            for v47, v130 in v13[v52].v131():
                if v47 not in v95:
                    v95.v139(v47)
                    v53.v101(v47)
                    if v99(v53) >= v56:
                        return v53
        return v53
    v16 = v59()
    for v17 in v11:
        for v20 in v17:
            v16[v37[v20]] += 1
    v18 = v44(v45)
    for v47, v60 in v16.v61():
        v18[v60.v141()].v101(v47)

    def band_draw(v62, v63):
        v102, v103 = (v45(v18[v16[v62].v141()]), 0)
        while v99(v102) < 16 and v103 < 20:
            v103 += 1
            v102 += v18.v132(v16[v62].v141() - v103, [])
            v102 += v18.v132(v16[v62].v141() + v103, [])
        for v58 in v100(64):
            v47 = v102[v10.v89(v99(v102))]
            if v47 != v62 and v47 not in v63:
                return v47
        return None
    v19 = [v20 for v17 in v11 for v20 in v17]
    v10.v64(v19)
    v65, v66 = (v59(), [])
    for v20 in v19:
        if v65['n'] >= v6.v104:
            break
        v42 = v12[v20]
        v62 = v37[v20]
        v49 = {v37[v133] for v133 in v11[v42] if v133 != v20}
        if not v49 or v62 in v49:
            continue
        v65['n'] += 1
        v48 = v59((v37[v133] for v133 in v11[v42] if v133 != v20))
        v67 = v105(v42, v49, v48)
        v68 = v106(v42, v49, v48)
        v69 = v106(v42, v49, v48, norm_by_places=True)
        v70 = v107(v62, v49)
        v65['len_step'] += v99(v67)
        v65['len_share'] += v99(v68)
        for v80, v108 in (('st', v67), ('sh', v68), ('sn', v69)):
            v109 = v108.v142(v62) + 1 if v62 in v108 else 0
            if v109:
                v65[f'in_{v80}'] += 1
                v65[f'rank_{v80}'] += v109
                for v25 in v2:
                    v65[f'{v80}@{v25}'] += v109 <= v25
        if v70 is not None:
            v65['dn'] += 1
            for v80, v108 in (('dsh', v68), ('dsn', v69)):
                v109 = v108.v142(v70) + 1 if v70 in v108 else 0
                if v109:
                    for v25 in v2:
                        v65[f'{v80}@{v25}'] += v109 <= v25
        v71 = {v47: v110 for v110, v47 in v43(v67)}
        for v110, v47 in v43(v68):
            v71[v47] = v134(v71.v132(v47, v110), v110)
        if v62 in v71:
            for v25 in v2:
                v65[f'un@{v25}'] += v71[v62] + 1 <= v25
        v111, v112 = (v67[:v6.v76], v68[:v6.v76])
        v73, v113 = ([], v129())
        for v72 in v114(v111, v112):
            for v115 in v72:
                if v115 is not None and v115 not in v113:
                    v113.v139(v115)
                    v73.v101(v115)
        v73 = v73[:v6.v76]
        v116, v117, v118 = (v129(v111), v129(v112), v129(v73))
        v65['step'] += v62 in v116
        v65['share'] += v62 in v117
        v65['merged'] += v62 in v118
        v65['oracle'] += v62 in v116 | v117
        v65['only_step'] += v62 in v116 and v62 not in v117
        v65['only_share'] += v62 in v117 and v62 not in v116
        v65['both'] += v62 in v116 and v62 in v117
        v65['random'] += v62 in v129(v138(v42, v49, v6.v76))
        if v116 or v117:
            v66.v101(v99(v116 & v117) / v74(1, v99(v116 | v117)))
    v21 = v74(1, v65['n'])
    v22 = {v56: v65[v56] / v21 for v56 in ('step', 'share', 'merged', 'oracle', 'only_step', 'only_share', 'both', 'random')}
    v23 = {'corpus': v6.v75, 'lines': v99(v9), 'places': v99(v11), 'questions': v65['n'], 'reach_k': v6.v11, 'topm': v6.v76, 'min_fillers': v6.v41, **{f'reach_{v56}': v47 for v56, v47 in v22.v61()}, 'lane_overlap': v135(v66) / v74(1, v99(v66))}
    v23['headroom'] = v22['oracle'] - v22['merged']
    v1.v119.v77(parents=True, exist_ok=True)
    v1.v78(v136.v120(v23, indent=1), encoding='utf-8')
    v79(f"tape     {v99(v11)} places, {v65['n']} questions where the truth is NOT among the question's own values, {v6.v11} places walked, top-{v6.v76}")
    v79(f"LANES    step {v22['step']:.4f}   share {v22['share']:.4f}   random {v22['random']:.4f}   mean Jaccard of the two eights {v23['lane_overlap']:.4f}")
    v79(f"WHO      both {v22['both']:.4f}   only step {v22['only_step']:.4f}   only share {v22['only_share']:.4f}")
    v79(f"CEILING  merged {v22['merged']:.4f} (today)   oracle {v22['oracle']:.4f} (a perfect move)   headroom {v23['headroom']:+.4f}")
    v79(f"DEPTH    lane sizes: step {v65['len_step'] / v21:.1f}   share {v65['len_share'] / v21:.1f}   (step sees {v6.v11} places, share every place sharing a filler)")
    for v80, v81 in (('st', 'step '), ('sh', 'share'), ('sn', 'sh/pl'), ('un', 'union')):
        v82 = '  '.v121((f"@{v25} {v65[f'{v80}@{v25}'] / v21:.4f}" for v25 in v2))
        v83 = ''
        if v80 != 'un':
            v122 = v74(1, v65[f'in_{v80}'])
            v83 = f"   present {v65[f'in_{v80}'] / v21:.4f}   mean rank when present {v65[f'rank_{v80}'] / v122:.1f}"
        v79(f'  {v81}  {v82}{v83}')
        for v25 in v2:
            v23[f'{v80}_at_{v25}'] = v65[f'{v80}@{v25}'] / v21
    v23['present_step'], v23['present_share'] = (v65['in_st'] / v21, v65['in_sh'] / v21)
    v23['cut_cost'] = v23['un_at_128'] - v22['merged']
    v24 = v74(1, v65['dn'])
    for v25 in v2:
        v23[f'decoy_sum_at_{v25}'] = v65[f'dsh@{v25}'] / v24
        v23[f'decoy_norm_at_{v25}'] = v65[f'dsn@{v25}'] / v24
    v23['rerank_truth'] = v23['sn_at_8'] - v23['sh_at_8']
    v23['rerank_decoy'] = v23['decoy_norm_at_8'] - v23['decoy_sum_at_8']
    v23['rerank_net'] = v23['rerank_truth'] - v23['rerank_decoy']
    v79(f'  decoy  ' + '  '.v121((f"@{v25} {v23[f'decoy_sum_at_{v25}']:.4f}->{v23[f'decoy_norm_at_{v25}']:.4f}" for v25 in v2)) + f"   ({v65['dn']} frequency twins, sum -> per-place)")
    v79(f"RERANK   share@8 {v23['sh_at_8']:.4f} -> {v23['sn_at_8']:.4f} ({v23['rerank_truth']:+.4f})   decoy {v23['rerank_decoy']:+.4f}   net {v23['rerank_net']:+.4f}")
    if v65['n'] < 300:
        v79(f"\nVOID, NOT A RESULT. Only {v65['n']} questions had the truth outside their own values - too few for a 0.05 gate. Widen --window-lines and read it again.")
    elif v23['headroom'] > 0.05 and v22['only_share'] > 0.05:
        v79(f"\nCHOOSING IS WORTH SOMETHING AND THE CHOOSER IS THE FAULT. A perfect move reaches {v22['oracle']:.4f} against the interleave's {v22['merged']:.4f}, and {v22['only_share']:.4f} of these questions are reachable ONLY through the second lane. 386's mind gave that back and more by picking on one probe row, so the next lever is the CHOOSER, not the move set: something that reads more of a lane than its first row before committing to it.")
    elif v22['only_share'] > 0.05:
        v79(f"\nTHE LANES DIFFER AND THE INTERLEAVE ALREADY TAKES BOTH. {v22['only_share']:.4f} of these questions are reachable through share alone, so the second move is a real route - but the interleave already reaches {v22['merged']:.4f} against a perfect move's {v22['oracle']:.4f}, headroom {v23['headroom']:+.4f}. CHOOSING CANNOT ADD WHAT POOLING ALREADY HAS: committing to one lane can only lose the other, which is precisely what 385 and 386 measured. A move pays only where the cap forces the interleave to drop something the chooser would have kept.")
    elif v23['headroom'] > 0.05:
        v79(f"\nHEADROOM WITHOUT A REASON TO MOVE. The oracle beats the interleave by {v23['headroom']:+.4f}, but only {v22['only_share']:.4f} of questions need the second lane - the gain is the CAP, not the channel: both lanes hold the truth deeper than eight and the interleave cuts one of them short. That is an offer question, and 347 has answered offer questions four times.")
    else:
        v79(f"\nTHE TWO LANES ARE ONE LANE WITH TWO ORDERINGS. A perfect move choice is worth {v23['headroom']:+.4f} over the interleave, only {v22['only_share']:.4f} of questions are reachable through share alone, at a mean overlap of {v23['lane_overlap']:.4f}. With `fp=fillers` the walk's cosine is non-zero only between places that SHARE A FILLER, which is exactly connect's neighbourhood - so the two moves read the same places in a different order. No chooser can pay here. What is needed is a move that reaches where the walk cannot, and neither of these two is one.")
    if v23['rerank_net'] > 0.05:
        v79(f"\nTHE LANE WAS BADLY ORDERED, AND THAT IS A FREE WIN. Ranking connect by the MEAN overlap of the places a value stands at, instead of the SUM, moves the truth into the offered eight {v23['rerank_truth']:+.4f} of the time while lifting a frequency twin only {v23['rerank_decoy']:+.4f} - so this is finding answers, not preferring rare words. Same eight candidates, same cap, same channel: 365's rule has been summing where it should have been dividing, the third time this project has made that exact mistake after 317 and 383.")
    elif v23['rerank_truth'] > 0.05:
        v79(f"\nRARITY, NOT RELATEDNESS. The reranking lifts the truth {v23['rerank_truth']:+.4f} into the eight and lifts a frequency-matched decoy {v23['rerank_decoy']:+.4f} - within {v143(v23['rerank_net']):.4f} of the same. Dividing by how many places hold a value simply prefers rare values, and on this population the truth is rare by construction. The gain is the null's, not the channel's.")
    else:
        v79(f"\nTHE ORDERING IS NOT THE FAULT. Mean overlap instead of sum moves the truth {v23['rerank_truth']:+.4f} into the eight. The truth sits deep in this lane for some other reason than frequency, and the next question is what the top of the lane is actually full of.")
    if v23['cut_cost'] > 0.05:
        v79(f"\nTHE CUT BINDS, NOT THE MATERIAL. Read to depth 128 the two lanes together hold the truth {v23['un_at_128']:.4f} of the time against the offered {v22['merged']:.4f} - {v23['cut_cost']:+.4f} is sitting below rank eight. The question stops being WHERE TO LOOK and becomes HOW MANY the mind can weigh, which 368 tested on a merged offer and never on the depth of one lane.")
    else:
        v79(f"\nTHE MATERIAL BINDS, NOT THE CUT. Reading to depth 128 adds only {v23['cut_cost']:+.4f} over the offered eight, so the truths the arm misses are not sitting further down the lane - they are not in it at any depth. That is the same wall 373, 375 and 376 each reached from a different side, and no arrangement of the offer moves it.")
    v79(f'\nwritten to {v1}')
    return 0
if v26 == '__main__':
    raise v84(v123())