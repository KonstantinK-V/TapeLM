"""Is the ink a lossy copy of something the tape already counts exactly?

THE QUESTION. The walk goes to places whose FINGERPRINTS are near - a hash over the bag of
fillers, so an approximation. But the tape holds an exact relation the ink can only imitate:
two places are linked when the SAME FILLER was written in both holes, and the tape knows how
many times. That is a count, so by this project's own invariant it belongs in the write path,
and interpolation belongs only where the mind reads.

If an exact shared-filler walk reaches the truth as often as the cosine walk, the ink is a
lossy copy of a countable relation and should be replaced by it. If the cosine reaches MORE,
the ink is generalising past what is written - which is a finding of the opposite sign and
just as useful, because it says the approximation earns its place.

This is 310's discipline: measure the substrate before building anything that stands on it.

    python _audit323_shared.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v6('data/_wikitext103_train.txt')
v1 = v6('results/_stage323_shared.json')
v2 = 8
v3 = 8
v4 = 64

def fp(v7: v30, v8: v5=v4):
    """The ink, standing in for bank.fp: blake2b over character trigrams, nothing trained.

    Not the stage's exact encoder, and it does not need to be - what is being compared is
    A HASH OF THE FILLER BAG against EXACT FILLER SHARING. Any pure hash of the same bag has
    the same character: it collides, it cannot count, and it cannot be inspected. If the
    conclusion depended on which hash, it would not be a conclusion about approximation.
    """
    v9 = [0.0] * v8
    v10 = f'  {v7}  '
    for v11 in v31(v50(v10) - 2):
        v32 = v121.v114(v10[v11:v11 + 3].v122('utf-8'), digest_size=8).v73()
        v9[v5.v115(v32[:4], 'big') % v8] += 1.0
    v12 = v74((v101 * v101 for v101 in v9)) ** 0.5
    return [v101 / v12 for v101 in v9] if v12 else v9

def main() -> v5:
    v13 = v75.v33()
    v13.v34('--bytes', type=v5, default=30000000)
    v13.v34('--frame-max', type=v5, default=3)
    v13.v34('--min-fillers', type=v5, default=2)
    v13.v34('--addresses', type=v5, default=1500)
    v13.v34('--lines', type=v5, default=25000)
    v13.v34('--window-lines', type=v5, default=400)
    v13.v34('--sample', choices=('uniform', 'region'), default='region')
    v13.v34('--seed', type=v5, default=1337)
    v13.v34('--max-questions', type=v5, default=3000)
    v14 = v13.v35()
    v15 = v0.v102('r', encoding='utf-8', errors='ignore').v36(v14.v37)
    v16 = [v77.v76() for v77 in v15.v103('\n') if v50(v77.v76()) >= 80]
    v17 = v16[:v5(0.7 * v50(v16))][:v14.v17]
    v18 = v78.v38(v14.v39)
    v19, v40, v41 = v79.v42(v17, v14.v43, v14.v44)
    if v14.v45 == 'region':
        if v14.v46:
            v80 = v79.v104(v19, v41)
            v81 = v18.v105(v68(1, v50(v17)))
            v51 = v52(v106)
            for v27 in v31(v14.v46):
                for v116, v11 in v80.v117((v81 + v27) % v50(v17), ()):
                    v51[v116].v86(v11)
            v19 = [(v116, v123(v9)) for v116, v9 in v51.v87() if v50({v40[v11] for v11 in v9}) >= v14.v44]
            if v14.v82 and v50(v19) > v14.v82:
                v19 = v18.v45(v19, v14.v82)
        else:
            v19 = v79.v107(v19, v40, v41, v50(v17), v14.v82, v18, v14.v44)
    elif v14.v82 and v50(v19) > v14.v82:
        v19 = v18.v45(v19, v14.v82)
    if not v19:
        v71('no tape')
        return 1
    v47, v48 = ([], [])
    for (v83, v84, v85), v49 in v19:
        v47.v86(f"{' '.v124(v84)}|{' '.v124(v85)}")
        v48.v86([v40[v11] for v11 in v49])
    v20 = v50(v47)
    v21 = [v57(v9) for v9 in v48]
    v22 = []
    for v23 in v21:
        v51 = [0.0] * v4
        for v9, v26 in v23.v87():
            v88 = v108(v9)
            for v11 in v31(v4):
                v51[v11] += v26 * v88[v11]
        v12 = v74((v101 * v101 for v101 in v51)) ** 0.5
        v22.v86([v101 / v12 for v101 in v51] if v12 else v51)
    v24 = v52(v53)
    for v54, v23 in v55(v21):
        for v9 in v23:
            v24[v9].v109(v54)
    v25 = [(v54, v11) for v54 in v31(v20) for v11 in v31(v50(v48[v54])) if v50(v48[v54]) >= 2]
    v18.v56(v25)
    v25 = v25[:v14.v89]
    v26 = v57()
    v58, v59 = ([], [])
    for v54, v11 in v25:
        v60 = v48[v54][v11]
        v61 = v57(v48[v54])
        v61[v60] -= 1
        if v61[v60] <= 0:
            del v61[v60]
        v26['n'] += 1
        v26['in_own'] += v60 in v61
        v51 = [0.0] * v4
        for v9, v90 in v61.v87():
            v88 = v108(v9)
            for v27 in v31(v4):
                v51[v27] += v90 * v88[v27]
        v62 = v74((v101 * v101 for v101 in v51)) ** 0.5
        v63 = [v101 / v62 for v101 in v51] if v62 else v51
        v64 = [(v74((v126 * v23 for v126, v23 in v127(v63, v22[v92]))), v92) for v92 in v31(v20) if v92 != v54]
        v64.v91(reverse=True)
        v65 = [v92 for v118, v92 in v64[:v2]]
        v66 = v57()
        for v9, v90 in v61.v87():
            for v92 in v24[v9]:
                if v92 != v54:
                    v66[v92] += v90
        v67 = [v92 for v92, v119 in v66.v120(v2)]

        def offer(v93):
            v110, v111 = (v53(), [])
            for v92 in v93:
                for v9 in v48[v92]:
                    if v9 not in v110:
                        v110.v109(v9)
                        v111.v86(v9)
            return v111[:v3]
        v94, v95 = (v112(v65), v112(v67))
        v26['cos_reach'] += v60 in v53(v94)
        v26['share_reach'] += v60 in v53(v95)
        v26['both'] += v60 in v53(v94) and v60 in v53(v95)
        v26['cos_only'] += v60 in v53(v94) and v60 not in v53(v95)
        v26['share_only'] += v60 in v53(v95) and v60 not in v53(v94)
        v96, v97 = (v112(v65 + []), v112(v67 + []))
        v58.v86(v96.v125(v60) + 1 if v60 in v96 else 0)
        v59.v86(v97.v125(v60) + 1 if v60 in v97 else 0)
        v26['overlap'] += v50(v53(v65) & v53(v67))
    v12 = v68(1, v26['n'])
    v27 = v68(1, v26['cos_only'] + v26['share_only'])
    v28 = {'bytes': v14.v37, 'sample': v14.v45, 'window_lines': v14.v46, 'places': v20, 'questions': v26['n'], 'own_hit': v26['in_own'] / v12, 'cos_reach': v26['cos_reach'] / v12, 'share_reach': v26['share_reach'] / v12, 'both': v26['both'] / v12, 'cos_only': v26['cos_only'], 'share_only': v26['share_only'], 'paired_z': (v26['cos_only'] - v26['share_only']) / v27 ** 0.5, 'places_overlap_mean': v26['overlap'] / v12 / v2, 'rank_cos_mean': v74(v58) / v12, 'rank_share_mean': v74(v59) / v12}
    v1.v98.v69(parents=True, exist_ok=True)
    v1.v70(v113.v99(v28, indent=1), encoding='utf-8')
    v71(f"tape    {v20} places, {v26['n']} questions, own_hit {v28['own_hit']:.4f}")
    v71(f"reach   cosine {v28['cos_reach']:.4f}   exact-shared {v28['share_reach']:.4f}   both {v28['both']:.4f}")
    v71(f"PAIRED  cosine-only {v26['cos_only']}  shared-only {v26['share_only']}   z {v28['paired_z']:+.2f}   (positive = the ink beats the count)")
    v71(f"places  the two walks pick the same place {v28['places_overlap_mean']:.4f} of the time")
    v71(f'\nwritten to {v1}')
    return 0
if v29 == '__main__':
    raise v72(v100())