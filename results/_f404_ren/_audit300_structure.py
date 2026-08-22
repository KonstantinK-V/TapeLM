"""How much structure is there to compose over - measured before anything is built for it.

WHY THIS COMES FIRST. 299 settled that the mind ties counting nearly everywhere and beats it in
one corner, and the corner turned out to be the boring one: ceiling 23-27%, of which own_hit is
18-22% and walk_only about 5%. So roughly four fifths of every answerable question is CONFIRM -
the hole repeats a filler its own frame already showed - and the walk, the whole point of 299,
can affect a twentieth of the outcome. Polishing a reader against that is polishing a lookup.

The reason is structural, not statistical. THE FRAME TAPE IS A FLAT BAG OF INDEPENDENT PLACES.
The parsed tape's address had two halves, so places were tied together by a shared anchor and
one could ask what a subject's OTHER relations say. That tie went out with the parser - rightly,
its relation half was a function word - and what replaced it, filler-bag similarity, is not a
tie between places at all. It is a measure of resemblance, and resemblance lets you find
something similar; it does not let you put two records together.

So this asks whether the corpus carries a tie we have not used, and it asks in counts only - no
ink, no training, no model. Three candidate ties, each exact:

  OWN      the truth already stands at this very frame (the CONFIRM case, today's 18-22%)
  SHARE    it stands at a frame that shares at least one filler with this one. This is the exact
           ceiling of the filler-bag walk - no fingerprint can reach a place with nothing in
           common, so whatever the cosine does, it cannot beat this.
  LINE     it stands at ANOTHER FRAME IN THE SAME LINE. Two frames written in one sentence are
           about one thing. Counted, no grammar, no stopwords - the same kind of fact the cutter
           itself is built from.
  NEAR     the same, one line either side.

What decides whether to build anything is not LINE's size but LINE MINUS (OWN or SHARE): the
questions only adjacency can answer. And beside it, the price - how many candidates a step along
the line offers, against the eight the walk offers now. A tie that answers a third of the
questions by handing over two hundred candidates is not an answer, it is a bigger haystack.

    python _audit300_structure.py                       600 KB, frame_max 3, as 299 ran
    python _audit300_structure.py --bytes 4000000 --frame-max 3 --addresses 3000
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage300_structure.json')

def main() -> v2:
    v4 = v102.v30()
    v4.v31('--bytes', type=v2, default=4000000)
    v4.v31('--frame-max', type=v2, default=3)
    v4.v31('--addresses', type=v2, default=3000)
    v4.v31('--lines', type=v2, default=25000)
    v4.v31('--seed', type=v2, default=1337)
    v4.v31('--max-questions', type=v2, default=20000)
    (v4.v31('--window-lines', type=v2, default=0, help='grow the region to EXACTLY this many lines instead of stopping at an address budget. Without it, changing --min-fillers changes the window too: at 1 every position qualifies at once, the window closes after about fifty lines and mentions per address falls to 1.2, so mf=1 and mf=2 are never compared on the same text. Fix the window and only the rule moves'),)
    (v4.v31('--min-fillers', type=v2, default=2, help='how many DIFFERENT values a hole must have taken for its frame to be kept. 2 since 297, never questioned, and it deletes exactly the deterministic records: a frame whose surroundings FIX the answer has one filler and is discarded as a fixed phrase. `capital of france is ___` can never be on the tape. That is why an address is made of recurring glue and almost never names an entity, and therefore why there are no pointers to compose over - we delete them when writing'),)
    v4.v31('--sample', choices=('uniform', 'region'), default='uniform', help="how the tape's addresses are chosen. `uniform` is what every run has done: 3000 drawn from the whole corpus's qualifying frames, 1.2% of 250195, so two frames of one sentence are on the tape together about one time in a hundred. `region` takes EVERY frame of a contiguous stretch of lines until the budget is full, which is the same tape size over a connected object")
    v5 = v4.v32()
    v6 = v0.v135('r', encoding='utf-8', errors='ignore').v33(v5.v34)
    v7 = [v104.v103() for v104 in v6.v136('\n') if v42(v104.v103()) >= 80]
    v8 = v7[:v2(0.7 * v42(v7))][:v5.v8]
    v9 = v105.v35(v5.v36)
    v11, v37, v38 = v106.v39(v8, v5.v40, v5.v41)
    v10 = v42(v11)
    if v5.v43 == 'region':
        if v5.v44:
            v107 = v106.v137(v11, v38)
            v108 = v9.v138(v140(1, v42(v8)))
            v109 = v52(v53)
            for v85 in v55(v5.v44):
                for v97, v51 in v107.v143((v108 + v85) % v42(v8), ()):
                    v109[v97].v113(v51)
            v11 = [(v97, v117(v57)) for v97, v57 in v109.v59() if v42({v37[v51] for v51 in v57}) >= v5.v41]
        else:
            v11 = v106.v139(v11, v37, v38, v42(v8), v5.v110, v9, v5.v41)
    elif v5.v110 and v10 > v5.v110:
        v11 = v9.v43(v11, v5.v110)
    if not v11:
        v100('no tape')
        return 1
    v45, v46, v47, v48 = ([], [], [], [])
    for (v62, v111, v112), v49 in v11:
        v50 = f"{' '.v142(v111)}|{' '.v142(v112)}"
        v48.v113(v50)
        for v51 in v49:
            v45.v113(v50)
            v46.v113(v37[v51])
            v47.v113(v38[v51])
    v12 = v42(v45)
    v13 = v52(v53)
    v14 = v52(v54)
    v15 = v52(v54)
    v16 = v52(v53)
    for v17 in v55(v12):
        v13[v45[v17]].v113(v17)
        v14[v45[v17]].v114(v46[v17])
        v15[v46[v17]].v114(v45[v17])
        if v47[v17] >= 0:
            v16[v47[v17]].v113(v17)
    v18 = v53(v55(v12))
    if v5.v56 and v12 > v5.v56:
        v18 = v9.v43(v18, v5.v56)
    v19 = {}
    for v20 in v13:
        v111, v115, v112 = v20.v116('|')
        v19[v20] = [v62 for v62 in v111.v136() + v112.v136() if v62]
    v21 = v52(v54)
    for v57, v58 in v15.v59():
        v21[v57] = v58

    def ref_places_of(v20):
        v60 = v117(v54(v19.v143(v20, ())), key=lambda v62: v42(v21.v143(v62, ())))
        v61 = []
        for v62 in v60:
            for v88 in v117(v21.v143(v62, ()))[:8]:
                if v88 != v20 and v88 not in v61:
                    v61.v113(v88)
                if v42(v61) >= 8:
                    return v61
        return v61
    v22 = v63()
    for v64, v65 in v66(v8):
        for v62 in v54(v65.v136()):
            v22[v62] += 1
    v23 = 3

    def rare_shared(v67, v68):
        if v67 < 0 or v68 < 0 or v67 == v68:
            return 0
        v69 = {v62 for v62 in v8[v67].v136() if v22[v62] <= v23}
        return v118((1 for v62 in v54(v8[v68].v136()) if v62 in v69))
    v24 = (8, 16, 32, 64, 128, 256, 512, 10 ** 9)
    v25 = v63()
    v26 = {}

    def near8_of(v20, v70):
        v71 = (v20, v144(v70)) if v42(v70) < 6 else v20
        if v71 in v26:
            return v26[v71]
        v72 = {v88 for v145 in v70 for v88 in v15[v145]} - {v20}
        v61 = v117(v72, key=lambda v88: -v42(v14[v88] & v70))[:8]
        v26[v71] = v61
        return v61
    v27 = v63()
    v73, v74, v75, v76, v77 = ([], [], [], [], [])
    v78, v79, v80 = ([], [], [])
    for v17 in v18:
        v119, v57, v64 = (v45[v17], v46[v17], v47[v17])
        v81 = v120((v46[v124] == v57 for v124 in v13[v119] if v124 != v17))
        v70 = {v46[v124] for v124 in v13[v119] if v124 != v17}
        v82 = v120((v14[v88] & v70 for v88 in v15[v57] if v88 != v119))
        v83 = [v124 for v124 in v16.v143(v64, ()) if v45[v124] != v119] if v64 >= 0 else []
        v65 = v120((v46[v124] == v57 for v124 in v83))
        v84 = v65
        for v85 in (-1, 1):
            if not v84 and v64 >= 0:
                v84 = v120((v46[v124] == v57 and v45[v124] != v119 for v124 in v16.v143(v64 + v85, ())))
        v86 = v42(v15[v57] - {v119}) > 0
        v27['n'] += 1
        v27['own'] += v81
        v27['share'] += v82
        v27['line'] += v65
        v27['near'] += v84
        v27['anywhere'] += v86
        v27['own_or_share'] += v81 or v82
        v27['line_only'] += v65 and (not (v81 or v82))
        v27['near_only'] += v84 and (not (v81 or v82))
        v27['nothing'] += not (v81 or v82 or v65 or v84)
        v73.v113(v42({v46[v124] for v124 in v83}))
        v75.v113(v42({v45[v124] for v124 in v83}))
        v87 = v121(v119, v70)
        v74.v113(v118((v42(v14[v88]) for v88 in v87)))
        v122, v123 = (v54(), None)
        for v88 in v87:
            for v124 in v13[v88]:
                v62 = v46[v124]
                if v62 not in v122:
                    v122.v114(v62)
                    if v62 == v57 and v123 is None:
                        v123 = v42(v122)
        for v89 in v24:
            v25[v89] += v2(v123 is not None and v123 <= v89)
        if v123 is not None:
            v76.v113(v123)
        v90 = v123 is not None and v123 <= 8
        v91 = v54(v87) | {v119}
        v125, v96 = (v54(), False)
        v92 = 0
        for v88 in v87:
            for v126 in v121(v88, v14[v88]):
                if v126 in v91:
                    continue
                v91.v114(v126)
                for v124 in v13[v126]:
                    v62 = v46[v124]
                    if v62 not in v125:
                        v125.v114(v62)
                        v92 += 1
                        if v62 == v57 and v92 <= 8:
                            v96 = True
        v93 = v127(v119)
        v128, v129, v94 = (v54(), 0, False)
        for v88 in v93:
            for v124 in v13[v88]:
                v62 = v46[v124]
                if v62 not in v128:
                    v128.v114(v62)
                    v129 += 1
                    if v62 == v57 and v129 <= 8:
                        v94 = True
        v27['ref'] += v94
        v27['ref_only'] += v94 and (not (v81 or v82 or v65))
        v78.v113(v129)
        v95 = v130((v62 for v62 in (v46[v124] for v88 in v87 for v124 in v13[v88]) if v62 != v57), None)
        if v64 >= 0 and v95 is not None:
            v79.v113(v140((v146(v64, v47[v124]) for v124 in v13[v119] if v46[v124] == v57 and v124 != v17), default=0) or v140((v146(v64, v47[v124]) for v88 in v87 for v124 in v13[v88] if v46[v124] == v57), default=0))
            v80.v113(v140((v146(v64, v47[v124]) for v88 in v87 for v124 in v13[v88] if v46[v124] == v95), default=0))
        v27['hop2'] += v96
        v27['hop2_only'] += v96 and (not (v81 or v90))
        v77.v113(v92)

    def pct(v97):
        return v27[v97] / v140(1, v27['n'])
    v28 = {'bytes': v5.v34, 'frame_max': v5.v40, 'sample': v5.v43, 'min_fillers': v5.v41, 'window_lines': v5.v44, 'addresses': v42(v48), 'frame_pool': v10, 'slots': v12, 'questions': v27['n'], 'lines': v42(v8), 'mentions_per_address': v12 / v140(1, v42(v48)), 'own': v131('own'), 'share': v131('share'), 'line': v131('line'), 'near': v131('near'), 'anywhere': v131('anywhere'), 'own_or_share': v131('own_or_share'), 'line_only': v131('line_only'), 'near_only': v131('near_only'), 'nothing': v131('nothing'), 'line_candidates_mean': v118(v73) / v140(1, v42(v73)), 'line_places_mean': v118(v75) / v140(1, v42(v75)), 'share_candidates_mean_at_8': v118(v74) / v140(1, v42(v74)), 'reachable_at_cap': {'all' if v97 > 10 ** 8 else v97: v25[v97] / v140(1, v27['n']) for v97 in v24}, 'truth_rank_median': v117(v76)[v42(v76) // 2] if v76 else None, 'truth_found_share': v42(v76) / v140(1, v27['n']), 'hop2': v131('hop2'), 'hop2_only': v131('hop2_only'), 'hop2_candidates_mean': v118(v77) / v140(1, v42(v77)), 'ref': v131('ref'), 'ref_only': v131('ref_only'), 'ref_candidates_mean': v118(v78) / v140(1, v42(v78)), 'confirm_rare_true': v118(v79) / v140(1, v42(v79)), 'confirm_rare_wrong': v118(v80) / v140(1, v42(v80)), 'confirm_separates': v118((1 for v147, v148 in v149(v79, v80) if v147 > v148)) / v140(1, v118((1 for v147, v148 in v149(v79, v80) if v147 != v148)))}
    v1.v132.v98(parents=True, exist_ok=True)
    v1.v99(v141.v133(v28, indent=1), encoding='utf-8')
    v100(f"tape   [{v5.v43} min_fillers={v5.v41}] {v28['addresses']} addresses, {v12} slots, {v28['mentions_per_address']:.2f} mentions/address, pool {v10}")
    v100(f"reach  own {v28['own']:.3f}  share {v28['share']:.3f} (the walk's exact ceiling)  own_or_share {v28['own_or_share']:.3f}")
    v100(f"ties   line {v28['line']:.3f}  near {v28['near']:.3f}  anywhere {v28['anywhere']:.3f}")
    v100(f"NEW    line_only {v28['line_only']:.3f}   near_only {v28['near_only']:.3f}   unanswerable by anything {v28['nothing']:.3f}")
    v100('APERTURE  reachable when the walk may name the top C fillers of its eight places:')
    v100('          ' + '  '.v142((f'C={v97}:{v57:.3f}' for v97, v57 in v28['reachable_at_cap'].v59())))
    v100(f"          the truth sits at rank {v28['truth_rank_median']} of that list (median, over the {v28['truth_found_share']:.3f} where it is in the list at all)")
    v100(f"REF    through a pointer {v28['ref']:.4f}   and ONLY so {v28['ref_only']:.4f}   (offers {v28['ref_candidates_mean']:.0f} fillers)")
    v100(f"CONF   rare words shared with this line: truth {v28['confirm_rare_true']:.2f} vs wrong {v28['confirm_rare_wrong']:.2f}, truth ahead in {v28['confirm_separates']:.3f} of the pairs that differ")
    v100(f"HOP2   reachable in two hops {v28['hop2']:.4f}   and ONLY in two {v28['hop2_only']:.4f}   (the second ring offers {v28['hop2_candidates_mean']:.0f} fillers)")
    v100(f"price  a line step offers {v28['line_candidates_mean']:.1f} fillers from {v28['line_places_mean']:.1f} places; the filler walk offers {v28['share_candidates_mean_at_8']:.1f} from 8")
    v100(f'\nwritten to {v1}')
    return 0
if v29 == '__main__':
    raise v101(v134())