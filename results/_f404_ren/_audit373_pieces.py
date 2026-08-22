"""IS THE ANSWER MADE OF MATERIAL THAT IS ALREADY REACHABLE? The atom, never once varied.

THE LAST UNTOUCHED PART OF THE CONSTRUCTION. A value has been a WHOLE WHITESPACE TOKEN from the
first frame commit. Every closure since - the decision (five rearrangements), capacity (two
tests), the relation set (371, 372a, 372b) - was measured with the answer drawn from a finite
bag of whole tokens. While the answer is CHOSEN FROM A BAG, "produce what was never offered" is
not hard, it is ARITHMETICALLY IMPOSSIBLE: you can only pick what is in the bag.

If values were PIECES, a hole could be filled by an assembly that never stood there whole. That
is production, and it is the one route to step 0's fourth trait that the substrate has never
been given. But before any of that is built, one number decides whether the idea is empty:

    ON THE HOLES NOTHING REACHES, IS THE TRUTH MADE OF MATERIAL THAT IS ALREADY IN FRONT OF US?

WHAT IS MEASURED, torch-free, on exactly the subset 363 used - truth not among the place's own
fillers and not in the walk's offer, where every number this project prints is zero:

    material   the values actually available at that hole: its own fillers plus the top-m offer
    COVER      is EVERY piece of the truth present among the pieces of that material
    ASSEMBLE   stricter and far more interpretable: can the truth be formed by CONCATENATING
               whole available values - "wikitext" out of "wiki" and "text"
    NULL       the same two, for a token drawn from the TRUTH'S OWN FREQUENCY BAND. Pieces of
               common words are everywhere, so an unmatched null would hand cover a win for
               free - 363 learned that the hard way and this reuses its bands.
    job        how many available values an assembler would have to consider. The size of the
               problem, as 351 reported `paths`

  GATE  cover - cover_null > 0.05 AND assemble - assemble_null > 0.05.
        Then the material for the unreachable answers is already present and only the ASSEMBLY
        is missing - a piece-level tape has something real to do, and the fourth trait becomes
        a task.
        If cover is high but the null matches it, "made of available pieces" is true of any
        word and means nothing. If assemble is ~0 for both, the answers are not compounds of
        what is reachable, the atom was never the obstacle, and the design is finished.

  `--ngram` is swept, not tuned: 3, 4, 5. A conclusion that only holds at one size is a
  conclusion about that size.

    python _audit373_pieces.py
    python _audit373_pieces.py --ngram 4 --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage373_pieces.json')

def pieces(v4, v5):
    """character k-grams with boundaries marked, so a prefix and an infix are different
    material. No fitted constant beyond k, which is swept."""
    v6 = '^' + v4 + '$'
    if v30(v6) <= v5:
        return {v6}
    return {v6[v33:v33 + v5] for v33 in v31(v30(v6) - v5 + 1)}

def assemble(v4, v7, v8=3):
    """can `tok` be spelled by concatenating whole available values? Exact, no scoring, and
    bounded at three parts because a four-part spelling of a wikitext token is a coincidence
    rather than a compound - the bound is reported."""
    v9 = v30(v4)
    v10 = {0}
    for v11 in v31(v8):
        v32 = v81()
        for v33 in v10:
            for v23 in v7:
                if v23 and v4.v114(v23, v33):
                    v110 = v33 + v30(v23)
                    if v110 == v9:
                        return True
                    v32.v115(v110)
        if not v32:
            return False
        v10 = v32
    return False

def main() -> v2:
    v12 = v82.v34()
    v12.v35('--bytes', type=v2, default=30000000)
    v12.v35('--frame-max', type=v2, default=3)
    v12.v35('--min-fillers', type=v2, default=1)
    v12.v35('--lines', type=v2, default=25000)
    v12.v35('--window-lines', type=v2, default=400)
    v12.v35('--topm', type=v2, default=8)
    v12.v35('--ngram', type=v2, default=3)
    v12.v35('--max-parts', type=v2, default=3)
    v12.v35('--max-questions', type=v2, default=1500)
    v12.v35('--seed', type=v2, default=1337)
    v12.v35('--corpus', default=v100(v0))
    v13 = v12.v36()
    v14 = v3(v13.v116).v101('r', encoding='utf-8', errors='ignore').v37(v13.v38)
    v15 = [v84.v83() for v84 in v14.v102('\n') if v30(v84.v83()) >= 80]
    v16 = v15[:v2(0.7 * v30(v15))][:v13.v16]
    v17 = v85.v39(v13.v40)
    if v13.v41 and v13.v41 < v30(v16):
        v42 = v17.v86(v30(v16) - v13.v41)
        v16 = v16[v42:v42 + v13.v41]
    v43, v44, v45 = v87.v46(v16, v13.v47, v13.v48)
    if not v43:
        v79('no tape')
        return 1
    v18 = [v56(v52) for v103, v52 in v43]
    v49, v50 = ({}, v55(v56))
    for v51, v52 in v53(v18):
        for v6 in v52:
            v49[v6] = v51
            v50[v44[v6]].v89(v6)
    v19 = v54(v50)
    v20 = {v23: v30(v50[v23]) for v23 in v19}
    v21 = {v23: v20[v23].v88() for v23 in v19}
    v22 = v55(v56)
    for v23 in v19:
        v22[v21[v23]].v89(v23)
    v24 = {}

    def offer_of(v51, v57):
        v58 = v65((v44[v107] for v107 in v18[v51]))
        v58[v57] -= 1
        if v58[v57] <= 0:
            del v58[v57]
        v59 = v56(v58)[:6]
        if not v59:
            return v81()
        v60 = v65((v44[v107] for v107 in v18[v51]))
        v61 = v65()
        for v23 in v59:
            v26 = v24.v104(v23)
            if v26 is None:
                v26 = v65()
                for v105 in v50[v23]:
                    for v111 in v18[v49[v105]]:
                        if v44[v111] != v23:
                            v26[v44[v111]] += 1
                v24[v23] = v26
            for v90, v9 in v26.v106():
                v9 -= v60.v104(v90, 0)
                if v9 > 0 and v90 != v23:
                    v61[v90] += v9
        return {v90 for v90, v112 in v61.v113(v13.v76)}

    def band_draw(v57, v62):
        v91, v90 = (v56(v22[v21[v57]]), 0)
        while v30(v91) < 16 and v90 < 20:
            v90 += 1
            v91 += v22.v104(v21[v57] - v90, []) + v22.v104(v21[v57] + v90, [])
        for v63 in v31(64):
            v23 = v91[v17.v86(v30(v91))]
            if v23 != v57 and v23 not in v62:
                return v23
        return None
    v25 = [v6 for v52 in v18 for v6 in v52]
    v17.v64(v25)
    v26 = v65()
    v66, v67, v68 = ([], [], [])
    for v6 in v25:
        if v26['n'] >= v13.v92:
            break
        v51 = v49[v6]
        v57 = v44[v6]
        v58 = {v44[v107] for v107 in v18[v51] if v107 != v6}
        if not v58:
            continue
        v26['seen'] += 1
        v61 = v93(v51, v57)
        if v57 in v58 or v57 in v61:
            continue
        v26['n'] += 1
        v69 = (v58 | v61) - {v57}
        if not v69:
            continue
        v70 = v81()
        for v23 in v69:
            v70 |= v94(v23, v13.v75)
        v68.v89(v30(v69))
        v71 = v94(v57, v13.v75)
        v66.v89(v30(v71 & v70) / v30(v71))
        v26['cover'] += v71 <= v70
        v26['assemble'] += v95(v57, v69, v13.v8)
        v72 = v96(v57, v69)
        if v72 is None:
            continue
        v26['nn'] += 1
        v73 = v94(v72, v13.v75)
        v67.v89(v30(v73 & v70) / v30(v73))
        v26['cover_null'] += v73 <= v70
        v26['assemble_null'] += v95(v72, v69, v13.v8)
    v9 = v74(1, v26['n'])
    v27 = v74(1, v26['nn'])
    v28 = {'lines': v30(v16), 'places': v30(v18), 'vocab': v30(v19), 'ngram': v13.v75, 'max_parts': v13.v8, 'topm': v13.v76, 'min_fillers': v13.v48, 'seen': v26['seen'], 'unreachable': v26['n'], 'cover': v26['cover'] / v9, 'cover_null': v26['cover_null'] / v27, 'assemble': v26['assemble'] / v9, 'assemble_null': v26['assemble_null'] / v27, 'piece_share_truth': v108(v66) / v74(1, v30(v66)), 'piece_share_null': v108(v67) / v74(1, v30(v67)), 'job': v108(v68) / v74(1, v30(v68))}
    v28['cover_gain'] = v28['cover'] - v28['cover_null']
    v28['assemble_gain'] = v28['assemble'] - v28['assemble_null']
    v28['share_gain'] = v28['piece_share_truth'] - v28['piece_share_null']
    v1.v97.v77(parents=True, exist_ok=True)
    v1.v78(v109.v98(v28, indent=1), encoding='utf-8')
    v79(f'tape     {v30(v18)} places, {v30(v19)} values, {v13.v75}-grams, parts <= {v13.v8}')
    v79(f"SUBSET   {v26['n']} of {v26['seen']} questions reach NEITHER channel; the material at each is {v28['job']:.1f} values")
    v79(f"COVER    every piece present: truth {v28['cover']:.4f}   null {v28['cover_null']:.4f}   gain {v28['cover_gain']:+.4f}")
    v79(f"SHARE    fraction of pieces present: truth {v28['piece_share_truth']:.4f}   null {v28['piece_share_null']:.4f}   gain {v28['share_gain']:+.4f}")
    v79(f"ASSEMBLE spelled by whole available values: truth {v28['assemble']:.4f}   null {v28['assemble_null']:.4f}   gain {v28['assemble_gain']:+.4f}")
    if v28['cover_gain'] > 0.05 and v28['assemble_gain'] > 0.05:
        v79(f"\nTHE MATERIAL IS ALREADY THERE AND ONLY THE ASSEMBLY IS MISSING. On the holes nothing reaches, the truth is spelled out of what is in front of the mind {v28['assemble']:.4f} of the time against {v28['assemble_null']:.4f} for a word of the same frequency. The atom was the obstacle: while a value is a whole token the answer must be CHOSEN, and pieces let it be BUILT. The fourth trait becomes a task over {v28['job']:.0f} available values.")
    elif v28['cover_gain'] > 0.05:
        v79(f"\nCOVERED BUT NOT SPELLABLE. Every piece of the truth is present ({v28['cover']:.4f} against {v28['cover_null']:.4f}) yet it cannot be built by concatenating whole available values ({v28['assemble']:.4f} vs {v28['assemble_null']:.4f}). The pieces are there in the way the letters of any word are there - real material, no compositional route from it to the answer.")
    else:
        v79("\nTHE ATOM WAS NOT THE OBSTACLE. The unreachable truths are made of no more available material than a random word of the same frequency. Pieces would give the mind nothing to build with, and 'produce what was never offered' does not become possible by making the values smaller.")
    v79(f'\nwritten to {v1}')
    return 0
if v29 == '__main__':
    raise v80(v99())