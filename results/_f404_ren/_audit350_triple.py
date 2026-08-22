"""The last cheap question: can a DIRECTED relation be counted out of the same raw text?

WHERE THIS SITS. 347 and 349 together closed the substrate:

    the tape records (frame) -> fillers. Two fillers relate when they can fill THE SAME HOLE.
    That is ALTERNATION, not association - a paradigm table, which is a lexicon.
    A lexicon ranks alternatives. It cannot produce content.

349 asked whether the same tape holds a second relation - "stood on the same line" instead of
"stands in the same hole" - and the answer was: distinguishable but strictly worse. 29% of what
it reaches is unique, and it reaches half as much (0.103 vs 0.193), resolves five times worse
(argmax 0.008 vs 0.035) and is less supported. Unioning the two is sixteen candidates instead of
eight, which is WIDER ENUMERATION - the operation 335 closed as asymptotically wrong. So it adds
nothing that is not already known to be the wrong direction.

WHAT HAS NEVER BEEN COUNTED. Both relations are UNDIRECTED and both are about co-membership: same
hole, or same line. Neither records WHAT STOOD BETWEEN. A directed triple does:

    (A, the text between, B)        A stood, then this, then B

That is still pure counting off raw text - no parsing, no model, no threshold - and it is the
first relation on the table that CAN GENERATE: given A and a pattern, it produces a B that A's
own paradigm never contained. Substitution can only ever hand back things that already stood
where A stands.

THIS AUDIT DOES NOT BUILD IT. It asks whether such triples RECUR at all in the same 30 MB, in the
same columns as 346 and 349 so all three relations are comparable on one page:

    present@8    the truth among the top eight the relation offers
    argmax       what it resolves to
    support2+    the share of (A, pattern) -> B triples seen more than once
    tri_only     truths reached by the triple and NOT by substitution - the number that decides

If triples do not recur, then at this corpus size there is nothing directed to count, and the
project's result is the separation proof, stated as such. If they do, the write path has a shape
worth rebuilding for, and that is a new phase rather than a lever.

THE QUESTION'S OWN LINES ARE EXCLUDED WHOLE, as in 349: a triple drawn from the very sentence the
hole sits in is the sentence, not evidence from elsewhere.

    python _audit350_triple.py
    python _audit350_triple.py --gap 2        # tighter patterns, fewer and sharper
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v4('data/_wikitext103_train.txt')
v1 = v4('results/_stage350_triple.json')
v2 = 8

def main() -> v3:
    v5 = v84.v28()
    v5.v29('--bytes', type=v3, default=30000000)
    v5.v29('--frame-max', type=v3, default=3)
    v5.v29('--min-fillers', type=v3, default=2)
    v5.v29('--addresses', type=v3, default=1500)
    v5.v29('--lines', type=v3, default=25000)
    v5.v29('--window-lines', type=v3, default=400)
    v5.v29('--gap', type=v3, default=4, help='most tokens allowed BETWEEN the two ends of a triple. The pattern is what constrains, so a wider gap means more patterns and each one rarer - the same crowding 347 measured, in a different place')
    v5.v29('--seed', type=v3, default=1337)
    v5.v29('--max-questions', type=v3, default=3000)
    v5.v29('--corpus', default=v101(v0))
    v6 = v5.v30()
    v7 = v4(v6.v121).v102('r', encoding='utf-8', errors='ignore').v31(v6.v32)
    v8 = [v86.v85() for v86 in v7.v103('\n') if v47(v86.v85()) >= 80]
    v9 = v8[:v3(0.7 * v47(v8))][:v6.v9]
    v10 = v87.v33(v6.v34)
    v35, v36, v37 = v88.v38(v9, v6.v39, v6.v40)
    if v6.v11:
        v41 = v88.v89(v35, v37)
        v42 = v10.v90(v76(1, v47(v9)))
        v43 = v53(v54)
        for v44 in v56(v6.v11):
            for v104, v65 in v41.v105((v42 + v44) % v47(v9), ()):
                v43[v104].v95(v65)
        v45 = {(v42 + v44) % v47(v9) for v44 in v56(v6.v11)}
        v35 = [(v104, v116(v73)) for v104, v73 in v43.v60() if v47({v36[v65] for v65 in v73}) >= v6.v40]
    else:
        v45 = v55(v56(v47(v9)))
    if v6.v46 and v47(v35) > v6.v46:
        v35 = v10.v91(v35, v6.v46)
    if not v35:
        v82('no tape')
        return 1
    v12 = [[v36[v65] for v65 in v51] for v106, v51 in v35]
    v13 = [v54(v51) for v106, v51 in v35]
    v14 = v47(v12)
    v48, v49 = ({}, v53(v54))
    for v50, v51 in v52(v13):
        for v18 in v51:
            v48[v18] = v50
            v49[v37[v18]].v95(v18)
    v15 = v53(v54)
    for v50, v51 in v52(v13):
        for v18 in v51:
            v15[v36[v18]].v95(v18)
    v16 = v55(v15)
    v17 = v53(v54)
    for v18 in v56(v47(v36)):
        if v37[v18] in v45 and v36[v18] in v16:
            v17[v37[v18]].v95(v18)
    v19 = v53(v57)
    v20 = v53(v55)
    for v58, v59 in v17.v60():
        for v61 in v56(v47(v59)):
            for v92 in v56(v61 + 1, v47(v59)):
                v65, v117 = (v59[v61], v59[v92])
                v77 = v117 - v65 - 1
                if v77 > v6.v77:
                    break
                v107 = v118(v36[v65 + 1:v117])
                v19[v36[v65], v107][v36[v117]] += 1
                v20[v36[v65], v107, v36[v117]].v119(v58)
    v21 = [(v50, v65) for v50 in v56(v14) for v65 in v56(v47(v12[v50])) if v47(v12[v50]) >= 2]
    v10.v62(v21)
    v21 = v21[:v6.v93]
    v22 = v57()
    v63, v64 = ([], [])
    v23 = v57()
    for v50, v65 in v21:
        v66 = v12[v50][v65]
        v67 = v57(v12[v50])
        v67[v66] -= 1
        if v67[v66] <= 0:
            del v67[v66]
        v68 = v54(v67)[:6]
        if not v68:
            continue
        v22['n'] += 1
        v69 = v55(v13[v50])
        v70 = {v37[v18] for v18 in v69}

        def paradigm(v73):
            v94 = v57()
            for v18 in v15[v73]:
                if v18 in v69:
                    continue
                for v108 in v13[v48[v18]]:
                    if v108 not in v69 and v36[v108] != v73:
                        v94[v36[v108]] += 1
            return v94

        def triple(v73):
            """Everything the directed relation offers from this end, with the question's own
            LINES dropped whole - a triple drawn from the sentence the hole sits in is the
            sentence. Both directions, because "A then B" and "B then A" are different
            relations and a tape that recorded only one would be choosing a reading order."""
            v94 = v57()
            for (v61, v107), v109 in v19.v60():
                if v61 != v73:
                    continue
                for v92, v24 in v109.v60():
                    v120 = v20[v61, v107, v92] - v70
                    if v120 and v92 != v73:
                        v94[v92] += v47(v120)
            return v94
        v71 = v57()
        v72 = v57()
        for v73 in v68:
            v71 += v110(v73)
            v72 += v111(v73)
        v63.v95(v47(v71))
        v64.v95(v47(v72))
        for v96, v24 in v72.v60():
            v23[v24] += 1
        v74 = v66 in {v112 for v112, v122 in v71.v123(v2)}
        v75 = v66 in {v112 for v112, v122 in v72.v123(v2)}
        v22['par_present'] += v74
        v22['tri_present'] += v75
        v22['tri_right'] += v113(v72) and v72.v123(1)[0][0] == v66
        v22['par_right'] += v113(v71) and v71.v123(1)[0][0] == v66
        v22['tri_only'] += v75 and (not v74)
        v22['par_only'] += v74 and (not v75)
        v22['tri_empty'] += not v72
    v24 = v76(1, v22['n'])
    v25 = v76(1, v97(v23.v114()))
    v26 = {'bytes': v6.v32, 'window_lines': v6.v11, 'gap': v6.v77, 'places': v14, 'questions': v22['n'], 'n_patterns': v47(v19), 'paradigm': {'present_topm': v22['par_present'] / v24, 'argmax_right': v22['par_right'] / v24, 'offer': v97(v63) / v76(1, v47(v63))}, 'triple': {'present_topm': v22['tri_present'] / v24, 'argmax_right': v22['tri_right'] / v24, 'offer': v97(v64) / v76(1, v47(v64)), 'support_2plus': v97((v73 for v104, v73 in v23.v60() if v104 >= 2)) / v25, 'empty': v22['tri_empty'] / v24}, 'tri_only': v22['tri_only'] / v24, 'par_only': v22['par_only'] / v24}
    v1.v98.v78(parents=True, exist_ok=True)
    v1.v79(v115.v99(v26, indent=1), encoding='utf-8')
    v80, v81 = (v26['paradigm'], v26['triple'])
    v82(f"tape    {v14} places, {v22['n']} questions, {v47(v19)} (A,pattern) keys, gap {v6.v77}")
    v82(f"SUBST   present@{v2} {v80['present_topm']:.4f}  argmax {v80['argmax_right']:.4f}  offer {v80['offer']:.1f}")
    v82(f"TRIPLE  present@{v2} {v81['present_topm']:.4f}  argmax {v81['argmax_right']:.4f}  offer {v81['offer']:.1f}  support2+ {v81['support_2plus']:.4f}  empty {v81['empty']:.4f}")
    v82(f"APART   only by TRIPLE {v26['tri_only']:.4f}   only by SUBST {v26['par_only']:.4f}")
    if v81['empty'] > 0.5:
        v82('\nNOTHING DIRECTED TO COUNT at this corpus size: over half the questions have no triple from any of their own rows. The relation is not in 30 MB.')
    elif v26['tri_only'] > 0.05 and v81['argmax_right'] >= v80['argmax_right']:
        v82('\nA GENERATIVE RELATION EXISTS: the triple reaches truths substitution cannot AND resolves at least as well. The write path has a shape worth rebuilding for - a new phase, not a lever.')
    else:
        v82("\nNOT BETTER THAN SUBSTITUTION: the directed relation adds reach only by widening the offer, which 335 already closed. The substrate is the ceiling and the project's result is the separation proof.")
    v82(f'\nwritten to {v1}')
    return 0
if v27 == '__main__':
    raise v83(v100())