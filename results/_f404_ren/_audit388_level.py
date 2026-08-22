"""IS THERE A LEVEL ABOVE A PLACE? Kostya's question, made into a count.

WHERE THIS COMES FROM. Asked for e=mc2 a person does not scan facts - something like
science -> physics -> relativity happens first, and only then the answer, with a logic linking
each narrowing to the question. THIS PROJECT HAS NO SUCH LEVEL. Every operation it has ever run
is place -> place: the fingerprint walk, connect, copy, the deep read, the moves. There is no
"physics" on the tape, only individual holes.

WHAT THE LAST THREE STEPS LEFT, so this is not another lap of the same track:
  387 depth   `step` is exhausted (lane of 24, truth at rank 5). `share` HOLDS THE TRUTH .78-.88
              of the time and puts it at rank 67 - 315 at w1600 - out of 444 - 2466 candidates.
              The material is there and the ORDER is hopeless.
  387 rerank  ranking that lane by MEAN overlap instead of the SUM made it far worse
              (share@8 .516 -> .166, and the frequency decoy fell too, so it was not a rarity
              artifact). That is a real finding in the opposite direction: connect's signal is
              ACCUMULATED WEAK EVIDENCE - many loosely related places agreeing - not one strong
              link. The 317/383 "divide, do not sum" precedent does not transfer here.
  387 gate    the move as an output space is closed: pooling beats choosing, headroom +0.02.

CLOSED, AND NOT TO BE PROPOSED AGAIN: strict two-filler connect (365, lost badly), intersecting
two lenses (346), value-lenses at all (384), a wider offer (347, four times).

WHAT HAS NEVER EXISTED IS TRANSITIVITY. `connect` is ONE hop of sharing a filler. Places A and C
can belong to one region through B without sharing anything directly, and a region is not a
place. This measures whether such regions exist on the tape at all, before anything is built to
use them.

    THE LEVEL, BY COUNTING AND NOTHING ELSE. The tape is a bipartite graph of places and values.
    Every place starts as its own label; then each value takes the commonest label among the
    places holding it, and each place takes the commonest label among the values standing in it,
    ties to the smaller label. R rounds, swept. Linear in the incidences - no pairwise blow-up,
    no threshold, no fitted constant. Values come out labelled too, so every candidate has a
    region and the question's place has one.

  same_label      the truth's region is the question's region ...
  decoy_label     ... against a FREQUENCY-MATCHED twin, because big regions match by size alone
  null_label      ... and against labels permuted between values, size distribution preserved
  in_label@8      the connect lane RESTRICTED to the question's region, same sum ranking
  narrow          how much of the lane the region keeps - the compression the level buys

  GATE  in_label@8 - share@8 > 0.05 AND same_label - decoy_label > 0.05.
        Then the tape HAS regions, narrowing works, and the "linking logic" of the metaphor
        becomes a count: a candidate is admissible if it is in the question's region. If not,
        there is no level above a place here, and one would have to be BUILT rather than found.

    python _audit388_level.py
    python _audit388_level.py --rounds 8 --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v4('data/_wikitext103_train.txt')
v1 = v4('results/_stage388_level.json')
v2 = (8, 16, 32)

def main() -> v3:
    v5 = v99.v36()
    v5.v37('--bytes', type=v3, default=30000000)
    v5.v37('--frame-max', type=v3, default=3)
    v5.v37('--min-fillers', type=v3, default=1)
    v5.v37('--lines', type=v3, default=25000)
    v5.v37('--window-lines', type=v3, default=400)
    v5.v37('--rounds', type=v3, default=4)
    v5.v37('--weight', choices=('plain', 'inv'), default='inv', help="how a place weighs a value's vote for a region. `plain` is one vote per standing; `inv` divides by how many places hold that value. HUB VALUES BRIDGE EVERYTHING - function words stand in holes too - and with `plain` the propagation collapses to one giant region, which would print as 'the tape has no level' when it means 'this construction collapsed'. Swept, not tuned: run both")
    v5.v37('--bridges', type=v3, default=32, help='how many hop-1 places the two-hop block expands through, best overlap first. A cost bound, swept, not a threshold on the graph')
    v5.v37('--topm', type=v3, default=8)
    v5.v37('--max-questions', type=v3, default=3000)
    v5.v37('--seed', type=v3, default=1337)
    v5.v37('--corpus', default=v129(v0))
    v6 = v5.v38()
    v7 = v4(v6.v93).v130('r', encoding='utf-8', errors='ignore').v39(v6.v40)
    v8 = [v101.v100() for v101 in v7.v131('\n') if v121(v101.v100()) >= 80]
    v9 = v8[:v3(0.7 * v121(v8))][:v6.v9]
    v10 = v102.v41(v6.v42)
    if v6.v43 and v6.v43 < v121(v9):
        v44 = v10.v103(v121(v9) - v6.v43)
        v9 = v9[v44:v44 + v6.v43]
    v45, v46, v47 = v104.v48(v9, v6.v49, v6.v50)
    if not v45:
        v97('no tape')
        return 1
    v11 = [v57(v22) for v132, v22 in v45]
    v12 = {}
    for v51, v22 in v52(v11):
        for v17 in v22:
            v12[v17] = v51
    v13 = v102.v41(v6.v42)
    v14 = [v17 for v22 in v11 for v17 in v22]
    v13.v53(v14)
    v15 = [v60((v46[v17] for v17 in v22)) for v22 in v11]
    v16 = v12
    v32, v54 = ([], v63())
    for v17 in v14:
        if v121(v32) >= v6.v105:
            break
        v51 = v16[v17]
        v55 = {v46[v133] for v133 in v11[v51] if v133 != v17}
        if not v55 or v46[v17] in v55:
            continue
        v32.v106(v17)
        v54.v107(v17)
    v18 = [v60((v46[v133] for v133 in v22 if v133 not in v54)) for v22 in v11]
    v19 = v56(v57)
    for v51, v58 in v52(v18):
        for v31 in v58:
            v19[v31].v106(v51)
    v20 = v59(v19)
    v21 = v60()
    for v22 in v11:
        for v17 in v22:
            v21[v46[v17]] += 1
    v23 = v57(v61(v121(v11)))
    v24 = {}
    for v25 in v61(v6.v62):
        for v31 in v20:
            v108 = v60((v23[v114] for v114 in v19[v31]))
            if v108:
                v24[v31] = v143(v108, key=lambda v64: (-v108[v64], v64))
        for v51, v58 in v52(v18):
            v108 = v60()
            for v31, v134 in v58.v65():
                v108[v24[v31]] += v134 / v121(v19[v31]) if v6.v94 == 'inv' else v134
            if v108:
                v23[v51] = v143(v108, key=lambda v64: (-v108[v64], v64))
    v26 = v60(v24.v109())
    v27 = v56(v63)
    for v31, v64 in v24.v65():
        v27[v64].v107(v31)
    v28 = v57(v24.v109())
    v10.v53(v28)
    v29 = v66(v110(v59(v24), v28))
    v30 = v56(v57)
    for v31 in v20:
        v30[v21[v31].v111()].v106(v31)

    def band_draw(v67, v68):
        v69 = v21[v67].v111()
        v112, v113 = (v57(v30[v69]), 0)
        while v121(v112) < 16 and v113 < 20:
            v113 += 1
            v112 += v30.v122(v69 - v113, []) + v30.v122(v69 + v113, [])
        for v70 in v61(64):
            v31 = v112[v10.v103(v121(v112))]
            if v31 != v67 and v31 not in v68:
                return v31
        return None

    def lane(v51, v71, v72, v73=None):
        """connect's lane, ranked by the SUM of overlaps - 387 measured that summing is right
        here - optionally restricted to one region."""
        v74 = v60()
        for v31 in v72:
            for v114 in v19[v31]:
                if v114 != v51:
                    v74[v114] += 1
        v75 = v60()
        for v114, v115 in v74.v65():
            for v31 in v18[v114]:
                if v31 in v71:
                    continue
                if v73 is not None and v24[v31] != v73:
                    continue
                v75[v31] += v115
        return [v31 for v31, v144 in v75.v138()]

    def lane_hop2(v51, v71, v76):
        """TRANSITIVITY, LOCALLY, WITH NOTHING GLOBAL TO COLLAPSE.

        The propagation above is one instrument and it can degenerate into a single region; this
        block does not depend on it. A place two hops away shares nothing with the question -
        it is related THROUGH a bridge - and that is the relation this project has never had.
        `connect` is one hop, `--reach-depth 2` is a second sequential READ rooted at a
        candidate, and neither is this.

        The score is a count: for a value v standing at a hop-2 place j, sum over the bridges b
        of min(overlap(q,b), overlap(b,j)) - the weakest link of the path, added over paths,
        because 387 measured that this channel's signal is ACCUMULATED weak evidence.
        """
        v77 = v63(v76) | {v51}
        v75 = v60()
        for v69, v116 in v76.v65():
            v117 = v60()
            for v31 in v18[v69]:
                for v114 in v19[v31]:
                    if v114 not in v77:
                        v117[v114] += 1
            for v114, v135 in v117.v65():
                v136 = v143(v116, v135)
                for v137 in v18[v114]:
                    if v137 not in v71:
                        v75[v137] += v136
        return [v31 for v31, v144 in v75.v138()]
    v78, v79 = (v60(), [])
    for v17 in v32:
        v51 = v12[v17]
        v67 = v46[v17]
        v71 = {v46[v133] for v133 in v11[v51] if v133 != v17}
        v78['n'] += 1
        v80 = v23[v51]
        v78['same'] += v24.v122(v67) == v80
        v78['null'] += v29.v122(v67) == v80
        v81 = v118(v67, v71)
        if v81 is not None:
            v78['dn'] += 1
            v78['decoy'] += v24.v122(v81) == v80
        v72 = v60((v46[v133] for v133 in v11[v51] if v133 != v17))
        v82 = v119(v51, v71, v72)
        v83 = v119(v51, v71, v72, only_label=v80)
        v84 = v60()
        for v31 in v72:
            for v114 in v19[v31]:
                if v114 != v51:
                    v84[v114] += 1
        v85 = v66(v84.v138(v6.v139))
        v86 = v120(v51, v71, v85)
        v78['h2_len'] += v121(v86)
        v87 = v86.v145(v67) + 1 if v67 in v86 else 0
        if v87:
            v78['h2_in'] += 1
            for v34 in v2:
                v78[f'h2@{v34}'] += v87 <= v34
        v88 = v82.v145(v67) + 1 if v67 in v82 else 0
        v78['h2_only'] += v140(v87) and (not v88)
        v78['h1_only'] += v140(v88) and (not v87)
        if v82:
            v79.v106(v121(v83) / v121(v82))
        v89 = v27.v122(v80, ())
        v78['reg_n'] += v121(v89)
        v90 = v67 in v89
        v78['reg_hit'] += v90
        v78['reg_only'] += v90 and v67 not in v63(v82[:v6.v147])
        v78['reg_union'] += v90 or v67 in v63(v82[:v6.v147])
        for v123, v64 in (('all', v82), ('lab', v83)):
            v124 = v64.v145(v67) + 1 if v67 in v64 else 0
            if v124:
                for v34 in v2:
                    v78[f'{v123}@{v34}'] += v124 <= v34
    v91, v92 = (v125(1, v78['n']), v125(1, v78['dn']))
    v33 = {'corpus': v6.v93, 'lines': v121(v9), 'places': v121(v11), 'values': v121(v20), 'questions': v78['n'], 'rounds': v6.v62, 'weight': v6.v94, 'regions': v121(v26), 'region_max': v125(v26.v109()) if v26 else 0, 'region_max_frac': v125(v26.v109()) / v121(v20) if v26 else 0.0, 'region_mean': v141(v26.v109()) / v121(v26) if v26 else 0.0, 'same_label': v78['same'] / v91, 'decoy_label': v78['decoy'] / v92, 'null_label': v78['null'] / v91, 'narrow': v141(v79) / v125(1, v121(v79))}
    for v34 in v2:
        v33[f'all_at_{v34}'] = v78[f'all@{v34}'] / v91
        v33[f'lab_at_{v34}'] = v78[f'lab@{v34}'] / v91
    for v34 in v2:
        v33[f'hop2_at_{v34}'] = v78[f'h2@{v34}'] / v91
    v33['hop2_present'] = v78['h2_in'] / v91
    v33['hop2_only'] = v78['h2_only'] / v91
    v33['hop1_only'] = v78['h1_only'] / v91
    v33['hop2_len'] = v78['h2_len'] / v91
    v33['region_size'] = v78['reg_n'] / v91
    v33['region_reach'] = v78['reg_hit'] / v91
    v33['region_only'] = v78['reg_only'] / v91
    v33['region_union'] = v78['reg_union'] / v91
    v33['union_gain'] = v33['region_union'] - v33['all_at_8']
    v33['label_gain'] = v33['same_label'] - v33['decoy_label']
    v33['reach_gain'] = v33['lab_at_8'] - v33['all_at_8']
    v1.v126.v95(parents=True, exist_ok=True)
    v1.v96(v142.v127(v33, indent=1), encoding='utf-8')
    v97(f"tape     {v121(v11)} places, {v121(v20)} values, {v78['n']} questions where the truth is not among the question's own values, {v6.v62} rounds")
    v97(f"LEVEL    {v33['regions']} regions over the values ({v6.v94})   mean {v33['region_mean']:.1f}   largest {v33['region_max']} ({v33['region_max_frac']:.1%} of the values)   the lane keeps {v33['narrow']:.4f} of itself")
    v97(f"BELONGS  truth {v33['same_label']:.4f}   frequency twin {v33['decoy_label']:.4f}   permuted labels {v33['null_label']:.4f}   gain {v33['label_gain']:+.4f}")
    v97('REACH    ' + '  '.v146((f"@{v34} {v33[f'all_at_{v34}']:.4f}->{v33[f'lab_at_{v34}']:.4f}" for v34 in v2)) + '   (whole lane -> its region)')
    v97(f"SOURCE   region holds {v33['region_size']:.1f} values   reach {v33['region_reach']:.4f}   only region {v33['region_only']:.4f}   region+lane@8 {v33['region_union']:.4f}   over lane@8 {v33['union_gain']:+.4f}   cost {v33['region_size'] + v6.v147:.0f}")
    v97(f'HOP2     ' + '  '.v146((f"@{v34} {v33[f'hop2_at_{v34}']:.4f}" for v34 in v2)) + f"   present {v33['hop2_present']:.4f}   lane {v33['hop2_len']:.0f}   only hop2 {v33['hop2_only']:.4f}   only hop1 {v33['hop1_only']:.4f}")
    if v78['n'] < 300:
        v97(f"\nVOID, NOT A RESULT. Only {v78['n']} questions; widen --window-lines.")
    elif v33['region_max_frac'] > 0.5:
        v97(f"\nTHE PROPAGATION COLLAPSED - NOT A RESULT ABOUT THE TAPE. One region holds {v33['region_max_frac']:.1%} of the values, so belonging to it says nothing: truth {v33['same_label']:.4f} and twin {v33['decoy_label']:.4f} are both near one by construction. HUB VALUES BRIDGE EVERYTHING - function words stand in holes too - and they merge every region into one. Try --weight {('plain' if v6.v94 == 'inv' else 'inv')} and fewer --rounds; if the largest region stays over half the values at every setting, label propagation is the wrong instrument here and the question is still open.")
    elif v33['reach_gain'] > 0.05 and v33['label_gain'] > 0.05:
        v97(f"\nTHE TAPE HAS REGIONS AND NARROWING WORKS. The truth shares the question's region {v33['same_label']:.4f} of the time against {v33['decoy_label']:.4f} for a frequency twin, and keeping only that region lifts the offered eight {v33['reach_gain']:+.4f} while throwing away {1 - v33['narrow']:.0%} of the lane. The linking logic of the metaphor is now a count: a candidate is admissible if it is in the question's region. That is the first level above a place this project has had.")
    elif v33['label_gain'] > 0.05:
        v97(f"\nTHE REGIONS ARE REAL AND DO NOT NARROW USEFULLY. The truth belongs to the question's region {v33['same_label']:.4f} against a twin's {v33['decoy_label']:.4f}, so the level EXISTS - but restricting the lane to it moves the offered eight {v33['reach_gain']:+.4f}. The region is not the bottleneck: what buries the truth at rank 67 is inside the region too.")
    else:
        v97(f"\nTHIS CONSTRUCTION FINDS NO USABLE LEVEL. The truth shares the question's region {v33['same_label']:.4f} against {v33['decoy_label']:.4f} for a frequency twin and {v33['null_label']:.4f} for permuted labels - so what looks like a region is mostly its own size. Propagating labels over the tape finds communities of places, and they are not the communities a question needs. Sweep --weight and --rounds before concluding: the claim earned here is about LABEL PROPAGATION on this tape, not about every possible level, and a level may still have to be BUILT from something the tape does not already say.")
    if v33['hop2_only'] > 0.05:
        v97(f"\nTRANSITIVITY REACHES WHERE ONE HOP CANNOT. {v33['hop2_only']:.4f} of these questions have the truth at a place two hops away and NOWHERE one hop away - related through a bridge, sharing nothing with the question directly. That is a relation this project has never had: connect is one hop and --reach-depth 2 is a second sequential read rooted at a candidate, not this. Whether it PAYS is the @8 column, and whether it is affordable is the lane size.")
    else:
        v97(f"\nTWO HOPS ADD NOTHING THE ONE HOP DID NOT HAVE. Only {v33['hop2_only']:.4f} of questions are reachable at two hops and not at one, against {v33['hop1_only']:.4f} the other way. The sharing graph closes on itself: a place two hops out holds the same values as the places one hop out, so transitivity is not a new route here, and a level above a place cannot be built by walking further.")
    v97(f'\nwritten to {v1}')
    return 0
if v35 == '__main__':
    raise v98(v128())