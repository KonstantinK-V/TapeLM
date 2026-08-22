"""ONE POSITION, ONE ADDRESS - the fiat 356 exposed, and the only part of it worth anything.

WHAT 356 SETTLED, INCLUDING AGAINST MY OWN PREDICTION:

    coverage      0.5594 of ALL interior positions are on the tape (0.5937 at frame_max 1).
                  Not a sliver. Everything measured for 350 steps sat inside a MAJORITY of the
                  text, and the honest end-to-end figure is coverage x hit = 0.559 x 0.445
                  = 0.25 of every token position in wikitext, recovered by counts plus 5633
                  parameters.
    rarity        A ONCE-SEEN token reaches the tape 0.6836 of the time; a token seen 101+
                  times reaches it 0.5260. I PREDICTED THE OPPOSITE. The tape is not a lattice
                  of categories - a frequent token fails because its hole has ONE filler and is
                  dropped, which is predictability, not rarity. Facts are present.
    the fiat      costs 3.4 points of coverage (0.5937 -> 0.5594) and touches only the 16.2% of
                  positions that reach width 2 or 3. AS A COVERAGE LEVER IT IS NOT WORTH A RUN,
                  and the plan of "give the width to Phi" dies here on its own numbers.

WHAT SURVIVES, AND IT IS A DIFFERENT MECHANISM. `frame_keep` writes `best_key[i] = the widest
recurring frame` - ONE address per position, the narrower paradigms thrown away. But a width-3
position is simultaneously a member of its width-2 and width-1 paradigms, and THOSE ARE THE
COARSE ONES: more fillers, more bridges, the very places a walk needs to leave a narrow
paradigm and land in another. The tape currently deletes, for 16% of its positions, the only
edges that connect them widely.

THIS IS NOT A SECOND RELATION. 349 (same line) and 350 (directed triples) added a DIFFERENT
kind of edge and both lost. This is the SAME substitution relation read at more than one
granularity - the one thing the write path has always thrown away.

    reach_wide    the truth in the offer, one address per position          (today)
    reach_multi   the same with every recurring width kept as an address
    multi_only    reached with the coarse layer and not without it          THE NUMBER
    reach_null    THE NULL: the coarse addresses PERMUTED among the positions that have one -
                  same layer, same sizes, wrong owners. A gain that survives this is the
                  position's own coarse paradigm; a gain that does not is offer-widening again,
                  which is how 354 died.

  GATE  reach_multi - reach_null > 0.05 at matched top-m.

    python _audit357_resolution.py
    python _audit357_resolution.py --window-lines 3200      # thicker, as 347 swept it
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage357_resolution.json')

def build(v4, v5, v6):
    """every recurring width kept, not just the widest. Returns the token list and, per width,
    the address -> positions map. The widest-only tape is recovered from the same tables, so
    the two arms cannot differ by anything except which addresses are kept."""
    v7 = [v24 for v79 in v4 for v24 in v79.v113()]
    v8 = v25(v7)
    v9 = {}
    v10 = v26(v27(1, v8 - 1))
    for v11 in v27(1, v5 + 1):
        v28 = v50(v26)
        for v29 in v10:
            if v29 - v11 < 0 or v29 + 1 + v11 > v8:
                continue
            v28[v128(v7[v29 - v11:v29]), v128(v7[v29 + 1:v29 + 1 + v11])].v86(v29)
        v30 = {v80: v48 for v80, v48 in v28.v51() if v25(v48) >= 2}
        if not v30:
            break
        v9[v11] = v30
        v10 = [v29 for v48 in v30.v123() for v29 in v48]
    return (v7, v9)

def main() -> v2:
    v12 = v81.v31()
    v12.v32('--bytes', type=v2, default=30000000)
    v12.v32('--frame-max', type=v2, default=3)
    v12.v32('--min-fillers', type=v2, default=2)
    v12.v32('--lines', type=v2, default=25000)
    v12.v32('--window-lines', type=v2, default=400)
    v12.v32('--topm', type=v2, default=8)
    v12.v32('--max-questions', type=v2, default=2000)
    v12.v32('--seed', type=v2, default=1337)
    v12.v32('--corpus', default=v114(v0))
    v13 = v12.v33()
    v14 = v3(v13.v124).v115('r', encoding='utf-8', errors='ignore').v34(v13.v35)
    v15 = [v83.v82() for v83 in v14.v113('\n') if v25(v83.v82()) >= 80]
    v4 = v15[:v2(0.7 * v25(v15))][:v13.v4]
    v16 = v84.v36(v13.v37)
    if v13.v38 and v13.v38 < v25(v4):
        v39 = v16.v85(v25(v4) - v13.v38)
        v4 = v4[v39:v39 + v13.v38]
    v7, v9 = v40(v4, v13.v5, v13.v6)
    if not v9:
        v77('no tape')
        return 1
    v17 = v41(v9)
    v18 = {}
    for v11 in v17:
        for v80, v48 in v9[v11].v51():
            for v29 in v48:
                v18[v29] = (v11, v80)
    v42, v43 = ([], [])
    v44, v45 = (v50(v26), v50(v26))

    def add(v46, v47, v48):
        if v25({v7[v29] for v29 in v48}) < v13.v6:
            return
        v49 = v25(v46)
        v46.v86(v26(v48))
        for v29 in v48:
            v47[v29].v86(v49)
    v19 = v50(v26)
    for v29, (v11, v80) in v18.v51():
        v19[v11, v80].v86(v29)
    for v52, v48 in v19.v51():
        v87(v42, v44, v41(v48))
    for v11 in v17:
        for v88, v48 in v9[v11].v51():
            v87(v43, v45, v41(v48))
    v53, v54 = ([], v50(v26))
    for v11 in v17:
        v56, v89 = ([], [])
        for v88, v48 in v9[v11].v51():
            if v25({v7[v29] for v29 in v48}) < v13.v6:
                continue
            v90 = [v29 for v29 in v48 if v18[v29][0] == v11]
            v91 = [v29 for v29 in v48 if v18[v29][0] != v11]
            v56.v86((v90, v25(v91)))
            v89.v116(v91)
        v16.v63(v89)
        v55 = 0
        for v90, v92 in v56:
            v93 = v90 + v89[v55:v55 + v92]
            v55 += v92
            v49 = v25(v53)
            v53.v86(v93)
            for v29 in v93:
                v54[v29].v86(v49)

    def offers(v46, v47):
        v57 = v50(v26)
        for v49, v48 in v94(v46):
            for v29 in v48:
                v57[v7[v29]].v86(v29)
        v58 = {}

        def co(v95, v96):
            v21 = v58.v117(v95)
            if v21 is None:
                v21 = v64()
                for v29 in v57[v95]:
                    for v49 in v47[v29]:
                        for v125 in v46[v49]:
                            if v7[v125] != v95:
                                v21[v7[v125]] += 1
                v58[v95] = v21
            v97 = v64()
            v98 = v64((v7[v29] for v29 in v96))
            for v118, v119 in v21.v51():
                v119 -= v98.v117(v118, 0)
                if v119 > 0:
                    v97[v118] = v119
            return v97
        return v59
    v60, v61, v62 = (v99(v42, v44), v99(v43, v45), v99(v53, v54))
    v20 = [(v49, v29) for v49, v48 in v94(v42) for v29 in v48]
    v16.v63(v20)
    v20 = v20[:v13.v100]
    v21 = v64()
    v65, v66 = ([], [])
    for v49, v29 in v20:
        v67 = v7[v29]
        v68 = v64((v7[v101] for v101 in v42[v49]))
        v68[v67] -= 1
        if v68[v67] <= 0:
            del v68[v67]
        v69 = v26(v68)[:6]
        if not v69:
            continue
        v21['n'] += 1
        v70 = [v101 for v120 in v44[v29] for v101 in v42[v120]]
        v71 = [v101 for v120 in v45[v29] for v101 in v43[v120]]
        v72 = [v101 for v120 in v54[v29] for v101 in v53[v120]]

        def top(v59, v102):
            v39 = v64()
            for v95 in v69:
                v39 += v59(v95, v102)
            return ({v11 for v11, v126 in v39.v127(v13.v74)}, v25(v39))
        v103, v104 = v105(v60, v70)
        v106, v107 = v105(v61, v71)
        v108, v109 = v105(v62, v72)
        v21['reach_wide'] += v67 in v103
        v21['reach_multi'] += v67 in v106
        v21['reach_null'] += v67 in v108
        v21['multi_only'] += v67 in v106 and v67 not in v103
        v21['lost'] += v67 in v103 and v67 not in v106
        v65.v86(v104)
        v66.v86(v107)
    v8 = v73(1, v21['n'])
    v22 = {'lines': v25(v4), 'questions': v21['n'], 'topm': v13.v74, 'wide_places': v25(v42), 'multi_places': v25(v43), 'reach_wide': v21['reach_wide'] / v8, 'reach_multi': v21['reach_multi'] / v8, 'reach_null': v21['reach_null'] / v8, 'multi_only': v21['multi_only'] / v8, 'lost': v21['lost'] / v8, 'offer_wide': v121(v65) / v73(1, v25(v65)), 'offer_multi': v121(v66) / v73(1, v25(v66))}
    v22['gain'] = v22['reach_multi'] - v22['reach_wide']
    v22['gain_over_null'] = v22['reach_multi'] - v22['reach_null']
    v1.v110.v75(parents=True, exist_ok=True)
    v1.v76(v122.v111(v22, indent=1), encoding='utf-8')
    v77(f"tape     {v25(v42)} widest-only places, {v25(v43)} multi-resolution, {v21['n']} questions, topm {v13.v74}")
    v77(f"WIDE     reach {v22['reach_wide']:.4f}   offer {v22['offer_wide']:.0f}")
    v77(f"MULTI    reach {v22['reach_multi']:.4f}   offer {v22['offer_multi']:.0f}   gain {v22['gain']:+.4f}")
    v77(f"NULL     reach {v22['reach_null']:.4f}   (coarse addresses permuted)   gain over null {v22['gain_over_null']:+.4f}")
    v77(f"APART    only with the coarse layer {v22['multi_only']:.4f}   lost to the crowd {v22['lost']:.4f}")
    if v22['gain_over_null'] > 0.05:
        v77(f"\nRESOLUTION IS A LEVER. Keeping every recurring width as an address reaches {v22['gain_over_null']:+.4f} beyond a coarse layer of the same shape with the wrong owners. The write path has been deleting the edges that connect narrow paradigms, and it is the same substitution relation - nothing new is assumed.")
    else:
        v77('\nRESOLUTION IS NOT A LEVER: the coarse layer buys no more than a permuted one of the same size. One position, one address costs nothing, the fiat is vindicated, and the substitution lattice is as connected as it is going to get.')
    v77(f'\nwritten to {v1}')
    return 0
if v23 == '__main__':
    raise v78(v112())