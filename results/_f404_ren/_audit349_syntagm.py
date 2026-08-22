"""Does the tape hold ANY relation other than substitutability? Measured before anything is built.

WHAT 347 SETTLED, AND IT IS NOT WHAT IT WAS BUILT TO SETTLE. Thickening a place works exactly as
intended and does not help:

    window   ment/pl   support2+    offer   present@8   argmax   in_own
       400      4.24      0.1600     37.8      0.1933   0.0373   0.2610
      6400      8.69      0.3512    202.9      0.1600   0.0397   0.5120

Thickness x2.05. Support x2.20 - the singleton problem GENUINELY IMPROVES, exactly as predicted.
And the lens's answer set grows x5.37, so the top eight of two hundred catches LESS than the top
eight of thirty-eight. The distribution gets more supported and more crowded at the same time,
and the ratio does not move. in_own nearly doubles: as a place thickens, the answer is more and
more often ALREADY THERE. Thickness helps the INDEX, not the constraint.

THE DIAGNOSIS THAT FOLLOWS, AND IT IS ABOUT THE TAPE AND NOT THE RULE. Two fillers "co-occur" on
this tape when they stand at THE SAME PLACE - the same frame, at different times. That does not
mean they are associated. It means they are ALTERNATIVES: things that can fill the same hole.
The tape's only relation is SUBSTITUTABILITY, which is a paradigmatic relation, and the walk's
fingerprint over filler bags is already an approximation of exactly it. So the constraint was
never a new relation - it was the walk's relation computed exactly instead of by hash, which is
why sharpening it, thickening it and re-resolving it all failed the same way.

A SUBSTITUTION RELATION CAN RANK ALTERNATIVES. IT CANNOT PRODUCE CONTENT. 343 said Phi is a
chooser and not a generator; this says the TAPE is a chooser's tape, and no operation over it
will generate.

WHAT THIS AUDIT ASKS. Is there a SYNTAGMATIC relation in the same tape - not "w can stand where v
stands" but "w stood ALONGSIDE v", in the same line, at a different place? That is a different
relation with a different shape: it links places rather than substituting within one, and it is
the only kind that can put a value in front of the mind that its own paradigm does not contain.

Everything is measured in 346's columns so the two relations are directly comparable, on the
same tape and the same questions:

    present@8   the truth in the top eight the relation offers
    argmax      what it resolves to
    support2+   the share of related pairs seen more than once
    offer       how many values it has to choose between

THE QUESTION'S OWN PLACE IS EXCLUDED, and so is any LINE it stands on - otherwise the value
sitting next to the hole in the very same sentence counts as evidence from elsewhere, which is
the same leak cons_resolve subtracts and 304's line channel was closed for.

    python _audit349_syntagm.py
    python _audit349_syntagm.py --window-lines 3200      # and on a thick tape
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v4('data/_wikitext103_train.txt')
v1 = v4('results/_stage349_syntagm.json')
v2 = 8

def main() -> v3:
    v5 = v73.v23()
    v5.v24('--bytes', type=v3, default=30000000)
    v5.v24('--frame-max', type=v3, default=3)
    v5.v24('--min-fillers', type=v3, default=2)
    v5.v24('--addresses', type=v3, default=1500)
    v5.v24('--lines', type=v3, default=25000)
    v5.v24('--window-lines', type=v3, default=400)
    v5.v24('--seed', type=v3, default=1337)
    v5.v24('--max-questions', type=v3, default=3000)
    v5.v24('--corpus', default=v90(v0))
    v6 = v5.v25()
    v7 = v4(v6.v110).v91('r', encoding='utf-8', errors='ignore').v26(v6.v27)
    v8 = [v75.v74() for v75 in v7.v92('\n') if v41(v75.v74()) >= 80]
    v9 = v8[:v3(0.7 * v41(v8))][:v6.v9]
    v10 = v76.v28(v6.v29)
    v30, v31, v32 = v77.v33(v9, v6.v34, v6.v35)
    if v6.v11:
        v36 = v77.v78(v30, v32)
        v37 = v10.v79(v64(1, v41(v9)))
        v38 = v42(v43)
        for v39 in v80(v6.v11):
            for v93, v54 in v36.v94((v37 + v39) % v41(v9), ()):
                v38[v93].v96(v54)
        v30 = [(v93, v106(v63)) for v93, v63 in v38.v102() if v41({v31[v54] for v54 in v63}) >= v6.v35]
    if v6.v40 and v41(v30) > v6.v40:
        v30 = v10.v81(v30, v6.v40)
    if not v30:
        v71('no tape')
        return 1
    v12 = [[v31[v54] for v54 in v45] for v95, v45 in v30]
    v13 = [v43(v45) for v95, v45 in v30]
    v14 = v41(v12)
    v15 = {}
    v16 = v42(v43)
    for v44, v45 in v46(v13):
        for v47 in v45:
            v15[v47] = v44
            v16[v32[v47]].v96(v47)
    v17 = v42(v43)
    for v44, v45 in v46(v13):
        for v47 in v45:
            v17[v31[v47]].v96(v47)
    v18 = [(v44, v54) for v44 in v80(v14) for v54 in v80(v41(v12[v44])) if v41(v12[v44]) >= 2]
    v10.v48(v18)
    v18 = v18[:v6.v82]
    v19 = v49()
    v50, v51 = ([], [])
    v52, v53 = (v49(), v49())
    for v44, v54 in v18:
        v55 = v12[v44][v54]
        v56 = v49(v12[v44])
        v56[v55] -= 1
        if v56[v55] <= 0:
            del v56[v55]
        v57 = v43(v56)[:6]
        if not v57:
            continue
        v19['n'] += 1
        v19['in_own'] += v55 in v56
        v58 = v83(v13[v44])

        def paradigm(v63):
            """346's relation: values at the PLACES that hold v. Substitutability."""
            v84 = v49()
            for v47 in v17[v63]:
                if v47 in v58:
                    continue
                for v97 in v13[v15[v47]]:
                    if v97 not in v58 and v31[v97] != v63:
                        v84[v31[v97]] += 1
            return v84

        def syntagm(v63):
            """The other relation: values that stood ON THE SAME LINE as v, at other places.

            A line where v stands AT THIS QUESTION'S OWN PLACE is dropped whole - not just the
            hidden slot. The value beside the hole in the same sentence is the sentence, not
            evidence from elsewhere, and counting it is the leak 304 was closed for.
            """
            v84 = v49()
            for v47 in v17[v63]:
                if v47 in v58:
                    continue
                v98 = v16.v94(v32[v47], ())
                if v103((v97 in v58 for v97 in v98)):
                    continue
                for v97 in v98:
                    if v15[v97] != v15[v47] and v31[v97] != v63:
                        v84[v31[v97]] += 1
            return v84
        v85, v86 = (None, None)
        v59 = v60 = -1
        v61 = v62 = False
        for v63 in v57:
            v99, v100 = (v107(v63), v108(v63))
            v50.v96(v41(v99))
            v51.v96(v41(v100))
            for v101, v20 in v99.v102():
                v52[v20] += 1
                if v20 > v59:
                    v59, v85 = (v20, v101)
            for v101, v20 in v100.v102():
                v53[v20] += 1
                if v20 > v60:
                    v60, v86 = (v20, v101)
            if v55 in {v101 for v101, v111 in v99.v112(v2)}:
                v61 = True
            if v55 in {v101 for v101, v111 in v100.v112(v2)}:
                v62 = True
        v19['par_present'] += v61
        v19['par_right'] += v85 == v55
        v19['syn_present'] += v62
        v19['syn_right'] += v86 == v55
        v19['syn_only'] += v62 and (not v61)
        v19['par_only'] += v61 and (not v62)
        v19['syn_empty'] += not v103((v108(v63) for v63 in v57))
    v20 = v64(1, v19['n'])
    v65, v66 = (v64(1, v104(v52.v109())), v64(1, v104(v53.v109())))
    v21 = {'bytes': v6.v27, 'window_lines': v6.v11, 'places': v14, 'questions': v19['n'], 'in_own': v19['in_own'] / v20, 'paradigm': {'present_topm': v19['par_present'] / v20, 'argmax_right': v19['par_right'] / v20, 'offer': v104(v50) / v64(1, v41(v50)), 'support_2plus': v104((v63 for v93, v63 in v52.v102() if v93 >= 2)) / v65}, 'syntagm': {'present_topm': v19['syn_present'] / v20, 'argmax_right': v19['syn_right'] / v20, 'offer': v104(v51) / v64(1, v41(v51)), 'support_2plus': v104((v63 for v93, v63 in v53.v102() if v93 >= 2)) / v66, 'empty': v19['syn_empty'] / v20}, 'syn_only': v19['syn_only'] / v20, 'par_only': v19['par_only'] / v20}
    v1.v87.v67(parents=True, exist_ok=True)
    v1.v68(v105.v88(v21, indent=1), encoding='utf-8')
    v69, v70 = (v21['paradigm'], v21['syntagm'])
    v71(f"tape    {v14} places, {v19['n']} questions, in_own {v21['in_own']:.4f}, window {v6.v11}")
    v71(f"SUBST   present@{v2} {v69['present_topm']:.4f}  argmax {v69['argmax_right']:.4f}  offer {v69['offer']:.1f}  support2+ {v69['support_2plus']:.4f}")
    v71(f"ALONG   present@{v2} {v70['present_topm']:.4f}  argmax {v70['argmax_right']:.4f}  offer {v70['offer']:.1f}  support2+ {v70['support_2plus']:.4f}  empty {v70['empty']:.4f}")
    v71(f"APART   reached only by ALONG {v21['syn_only']:.4f}   only by SUBST {v21['par_only']:.4f}")
    if v70['empty'] > 0.5:
        v71('\nTHE RELATION IS NOT THERE: over half the questions have no same-line partner at all. A frame tape records holes, not neighbours, and this is what that costs.')
    elif v21['syn_only'] > 0.03:
        v71('\nA SECOND RELATION EXISTS: it reaches truths substitutability does not. That is the first thing on this tape that could GENERATE rather than rank, and the write path is where it would have to be recorded properly.')
    else:
        v71('\nSAME RELATION IN DIFFERENT CLOTHES: standing alongside reaches nothing that standing-in-place does not. The tape holds one relation and it is substitution.')
    v71(f'\nwritten to {v1}')
    return 0
if v22 == '__main__':
    raise v72(v89())