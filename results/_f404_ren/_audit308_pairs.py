"""Is there composition on this tape at all - measured before anything is built for it.

WHAT WE HAVE ESTABLISHED, and why this is the question left. The mind beats exact counting when
deciding WHERE to look and when answering where no lookup on the place can answer; that survives
two corpus transplants and does not survive removing the message passing. What it has never done
is assemble an answer from more than one record, and three candidate ways to chain records were
measured and refuted: two hops (2.0% at 138 fillers a step), sentence adjacency (which turned out
to be a word repeating inside one line) and the pointer relation (0.16%). All three looked for
composition IN THE DATA - a chain from record to record.

THIS ASKS THE OTHER FORM: composition IN THE ANSWER. Hide TWO fillers on one line, at two
different places. The answer is a PAIR and it has to be consistent.

    the capital of ___ is ___          hidden: france, paris

Why that is composition rather than a lookup in disguise:

  THE RIVAL IS STRUCTURALLY BLIND on a nameable subset. Counting takes each hole on its own - the
  product of two marginals - and cannot use the joint constraint. If the true pair ALSO never
  stood together anywhere else on the tape, counting has no joint statistic either. Those
  questions are `comp_only`, and they are built the same way walk_only was: not "beat the lookup
  where it is right" but "answer where it has nothing".

  THE MIND NEEDS NOTHING NEW. Phi already scores a COMPLETED world. Filling two holes instead of
  one is the same operation, and how well the completed world hangs together IS the consistency
  of the pair - which is the definition of Phi rather than an addition to it.

  THE TEACHER IS EXACT. The record knows both values.

THE LEAK THAT HAD TO BE CLOSED FIRST. In `the capital of france is paris` the address of the
second hole is `france is|...` - it CONTAINS the first hidden token, so one hole's surroundings
hand over the other's answer. Two holes are only taken when their positions are further apart
than frame_max, so no window can cover the other's token. Exact, checkable, no judgement.

    python _audit308_pairs.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage308_pairs.json')

def main() -> v2:
    v4 = v72.v23()
    v4.v24('--bytes', type=v2, default=30000000)
    v4.v24('--frame-max', type=v2, default=3)
    v4.v24('--min-fillers', type=v2, default=2)
    v4.v24('--addresses', type=v2, default=1500)
    v4.v24('--lines', type=v2, default=25000)
    v4.v24('--window-lines', type=v2, default=0)
    v4.v24('--sample', choices=('uniform', 'region'), default='region')
    v4.v24('--seed', type=v2, default=1337)
    v4.v24('--pairs-per-line', type=v2, default=2)
    v4.v24('--max-questions', type=v2, default=20000)
    v4.v24('--joint-lines', type=v2, default=400, help='how many co-occurring lines are read when asking whether the true pair ever stood together. A cost bound; raising it can only turn comp_only DOWN, so the number this prints is an upper bound on the subset')
    v5 = v4.v25()
    v6 = v0.v97('r', encoding='utf-8', errors='ignore').v26(v5.v27)
    v7 = [v74.v73() for v74 in v6.v98('\n') if v44(v74.v73()) >= 80]
    v8 = v7[:v2(0.7 * v44(v7))][:v5.v8]
    v9 = v75.v28(v5.v29)
    v10, v30, v31 = v76.v32(v8, v5.v33, v5.v34)
    if v5.v35 == 'region':
        if v5.v36:
            v77 = v76.v99(v10, v31)
            v78 = v9.v100(v102(1, v44(v8)))
            v79 = v45(v46)
            for v66 in v48(v5.v36):
                for v65, v43 in v77.v105((v78 + v66) % v44(v8), ()):
                    v79[v65].v83(v43)
            v10 = [(v65, v110(v111)) for v65, v111 in v79.v52() if v44({v30[v43] for v43 in v111}) >= v5.v34]
            if v5.v80 and v44(v10) > v5.v80:
                v10 = v9.v35(v10, v5.v80)
        else:
            v10 = v76.v101(v10, v30, v31, v44(v8), v5.v80, v9, v5.v34)
    elif v5.v80 and v44(v10) > v5.v80:
        v10 = v9.v35(v10, v5.v80)
    if not v10:
        v70('no tape')
        return 1
    v37, v38, v39, v40 = ([], [], [], [])
    for (v81, v82, v62), v41 in v10:
        v42 = f"{' '.v106(v82)}|{' '.v106(v62)}"
        for v43 in v41:
            v37.v83(v42)
            v38.v83(v30[v43])
            v39.v83(v31[v43])
            v40.v83(v43)
    v11 = v44(v37)
    v12 = v45(v46)
    v13 = v45(v46)
    v14 = v45(v47)
    for v15 in v48(v11):
        v12[v37[v15]].v83(v15)
        v13[v39[v15]].v83(v15)
        v14[v37[v15]].v84(v39[v15])
    v16 = v45(v49)
    for v15 in v48(v11):
        v16[v39[v15]][v37[v15]] = v38[v15]
    v17 = []
    for v50, v51 in v13.v52():
        if v44(v51) < 2:
            continue
        v53 = [(v56, v57) for v107, v56 in v91(v51) for v57 in v51[v107 + 1:] if v37[v56] != v37[v57] and v113(v40[v56] - v40[v57]) > v5.v33]
        if not v53:
            continue
        v9.v54(v53)
        v17.v85(v53[:v5.v67])
    v9.v54(v17)
    if v5.v18:
        v17 = v17[:v5.v18]
    v19 = v55()
    v20 = []
    for v56, v57 in v17:
        v86, v87, v88, v89 = (v37[v56], v37[v57], v38[v56], v38[v57])
        v58 = v55((v38[v108] for v108 in v12[v86] if v108 != v56))
        v59 = v55((v38[v108] for v108 in v12[v87] if v108 != v57))
        v19['n'] += 1
        v20.v83(v44(v58) * v44(v59))
        v60 = v88 in v58 and v89 in v59
        v19['both_offered'] += v60
        if not v60:
            continue
        v61 = (v58.v112(1)[0][0], v59.v112(1)[0][0])
        v62 = v61 == (v88, v89)
        v19['marginal_right'] += v62
        v63 = False
        v64 = v14[v86] & v14[v87]
        for v65, v90 in v91(v64):
            if v90 == v39[v56] or v65 >= v5.v109:
                continue
            v92 = v16[v90]
            if v92.v105(v86) == v88 and v92.v105(v87) == v89:
                v63 = True
                break
        v19['joint_seen'] += v63
        v19['comp_only'] += not v62 and (not v63)

    def pct(v65, v66='n'):
        return v19[v65] / v102(1, v19[v66])
    v21 = {'bytes': v5.v27, 'frame_max': v5.v33, 'sample': v5.v35, 'min_fillers': v5.v34, 'window_lines': v5.v36, 'slots': v11, 'questions': v19['n'], 'pairs_per_line': v5.v67, 'both_offered': v93('both_offered'), 'marginal_right': v93('marginal_right', 'both_offered'), 'joint_seen': v93('joint_seen', 'both_offered'), 'comp_only': v93('comp_only'), 'comp_only_of_offered': v93('comp_only', 'both_offered'), 'pair_price_mean': v103(v20) / v102(1, v44(v20))}
    v1.v94.v68(parents=True, exist_ok=True)
    v1.v69(v104.v95(v21, indent=1), encoding='utf-8')
    v70(f"tape   {v11} slots, {v19['n']} two-hole questions (holes further apart than frame_max, so neither window covers the other)")
    v70(f"reach  both truths offered {v21['both_offered']:.4f}   pairs to consider {v21['pair_price_mean']:.0f}")
    v70(f"rival  marginals right {v21['marginal_right']:.4f}   pair seen together elsewhere {v21['joint_seen']:.4f}   (of those offered)")
    v70(f"COMP   counting has nothing either way: {v21['comp_only']:.4f} of all, {v21['comp_only_of_offered']:.4f} of the answerable")
    v70(f'\nwritten to {v1}')
    return 0
if v22 == '__main__':
    raise v71(v96())