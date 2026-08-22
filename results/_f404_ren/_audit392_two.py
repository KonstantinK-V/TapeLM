"""B2: TWO PLACES IN, ONE PLACE OUT. Understanding as a count. The cloze eight cannot sit here.

A and B are places from the SAME wiki document. Gold is a place C in the ADDRESS neighbourhood
of both (N_addr(A) ∩ N_addr(B)) that has a unique filler. Filler-neighbourhood cannot be N:
a unique-filler C in filler-N(A) means A already holds that filler, which is cloze.

Rivals, all cap 8, all PLACES (346's concat form, named new reason: places not values):
  walk A, walk B, concat N_walk(A) ∪ N_walk(B) cut at 8.
`both` is the address intersection, ranked the way 390 ranks address neighbours, cut at 8.

Null: B drawn from another document. Must be flat.

VOID, read first: if gold is already in walk(A)'s eight on more than half the questions, this
is cloze in a costume — do not read the gate.

GATE, declared before the run: hit(both) > max(hit A, hit B) + 0.05 on 3/4 seeds, AND the
null is below half of that gain. Four seeds 1337, 8642, 2890, 4711.

Not appointed-director. Not U. Not a vocab softmax.

    python _check392_two.py
    python _audit392_two.py --seed 1337
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
import _audit390_address as A
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage392_two.json')

def documents(v4):
    """Wiki articles: a `= title =` line starts a new doc. Fallback: one doc."""
    v29, v5 = ([], [])
    for v48, v49 in v50(v4):
        if v49.v112('=') and v49.v113('=') and v5:
            v29.v93(v5)
            v5 = [v48]
        else:
            v5.v93(v48)
    if v5:
        v29.v93(v5)
    return [v51 for v51 in v29 if v97(v51) >= 2]

def pids_on_lines(v6, v7):
    v8 = []
    for v9, v52 in v50(v6['places']):
        if v94((v6['owner'][v120] in v7 for v120 in v52)):
            v8.v93(v9)
    return v8

def unique_filler(v6, v9, v10=()):
    v11 = [v53 for v53 in v6['prof'][v9] if v53 not in v10]
    return v11[0] if v97(v11) == 1 else None

def walk8(v6, v9, v12, v13):
    v14 = v6['prof'][v9]
    return v95.v54(v6, v9, v14, v13, v12)

def addr_set(v6, v9, v12):
    return v55(v95.v58(v6, v9, v12))

def rank_addr(v6, v9, v15, v13):
    return v95.v56(v6, v9, v15, v13)

def measure_pair(v6, v16, v17, v18):
    """One (A,B). Gold = unique-filler place in addr-N(A) ∩ addr-N(B). None if empty."""
    v19 = v55(v6['on_line'][v6['owner'][v6['places'][v16][0]]])
    v20 = v55(v6['on_line'][v6['owner'][v6['places'][v17][0]]])
    v19.v57(v16)
    v20.v57(v17)
    v21 = v95.v58(v6, v16, v19)
    v22 = v95.v58(v6, v17, v20)
    v23 = {v59: v21[v59] + v22[v59] for v59 in v21 if v59 in v22 and v59 != v16 and (v59 != v17)}
    v24 = []
    for v59, v60 in v23.v61():
        v53 = v96(v6, v59)
        if v53 is not None:
            v24.v93((v59, v53, v60))
    if not v24:
        return None
    v24.v62(key=lambda v121: (-v121[2], v121[0]))
    v36, v63, v64 = v24[0]
    v25 = v65(v6, v16, v19, v18.v66)
    v26 = v65(v6, v17, v20, v18.v66)
    v27 = v95.v67(v25, v26, cap=v18.v66)
    v28 = v68(v6, v16, {v59: v23[v59] for v59 in v23}, v18.v66)
    return {'n': 1, 'hit_a': v2(v36 in v25), 'hit_b': v2(v36 in v26), 'hit_concat': v2(v36 in v27), 'hit_both': v2(v36 in v28), 'in_a_uncap': v2(v36 in v21), 'in_b_uncap': v2(v36 in v22), 'n_inter': v97(v23)}

def run(v6, v29, v18, v30, v31=False):
    v32 = {}
    for v69, v4 in v50(v29):
        for v70 in v4:
            v32[v70] = v69
    v33 = [v98(v6, v55(v51)) for v51 in v29]
    v36, v35 = (v99(), 0)
    v71, v34 = (0, 0)
    while v34 < v18.v100 and v71 < v18.v100 * 40:
        v71 += 1
        v69 = v30.v101(v97(v29))
        v72 = v33[v69]
        if v97(v72) < 2:
            continue
        v16, v17 = v30.v102(v72, 2)
        if v6['owner'][v6['places'][v16][0]] == v6['owner'][v6['places'][v17][0]]:
            continue
        if v31:
            v103 = [v59 for v59, v123 in v50(v33) if v59 != v69 and v97(v123) >= 1]
            if not v103:
                continue
            v17 = v30.v114(v33[v30.v114(v103)])
        v73 = v104(v6, v16, v17, v18)
        if v73 is None:
            continue
        v34 += 1
        for v13, v53 in v73.v61():
            v36[v13] += v53
        v35 += v73['hit_a']
    v36['seen'] = v34
    v36['void_a'] = v35
    return v36

def main() -> v2:
    v37 = v105.v74()
    v37.v75('--bytes', type=v2, default=30000000)
    v37.v75('--frame-max', type=v2, default=3)
    v37.v75('--min-fillers', type=v2, default=1)
    v37.v75('--lines', type=v2, default=25000)
    v37.v75('--window-lines', type=v2, default=400)
    v37.v75('--topm', type=v2, default=8)
    v37.v75('--max-questions', type=v2, default=1500)
    v37.v75('--seed', type=v2, default=1337)
    v37.v75('--corpus', default=v115(v0))
    v37.v75('--out', default=v115(v1))
    v18 = v37.v76()
    v38 = v3(v18.v124).v116('r', encoding='utf-8', errors='ignore').v77(v18.v78)
    v39 = [v107.v106() for v107 in v38.v117('\n') if v97(v107.v106()) >= 80]
    v4 = v39[:v2(0.7 * v97(v39))][:v18.v4]
    v30 = v108.v79(v18.v80)
    if v18.v81 and v18.v81 < v97(v4):
        v82 = v30.v101(v97(v4) - v18.v81)
        v4 = v4[v82:v82 + v18.v81]
    v6 = v95.v83(v4, v18.v84, v18.v85)
    v29 = v86(v4)
    if v97(v29) < 2:
        v41 = v97(v4)
        v29 = [v118(v122(v48, v125(v48 + 8, v41))) for v48 in v122(0, v41, 8)]
        v29 = [v51 for v51 in v29 if v97(v51) >= 2]
        v89(f'no article headers — {v97(v29)} windows of 8 lines')
    v36 = v87(v6, v29, v18, v30, null=False)
    v40 = v87(v6, v29, v18, v108.v79(v18.v80 + 1), null=True)
    v41 = v88(1, v36['seen'])
    v42 = v88(1, v40['seen'])
    v43 = {'seed': v18.v80, 'places': v97(v6['places']), 'docs': v97(v29), 'n': v36['seen'], 'n_null': v40['seen'], 'hit_a': v36['hit_a'] / v41, 'hit_b': v36['hit_b'] / v41, 'hit_concat': v36['hit_concat'] / v41, 'hit_both': v36['hit_both'] / v41, 'null_both': v40['hit_both'] / v42, 'void_frac': v36['void_a'] / v41, 'n_inter': v36['n_inter'] / v41}
    v44 = v88(v43['hit_a'], v43['hit_b'])
    v45 = v43['hit_both'] - v44
    v46 = v43['void_frac'] > 0.5
    v89(f"places {v43['places']}  docs {v43['docs']}  pairs {v43['n']}  null {v43['n_null']}")
    v89(f"VOID CHECK  gold already in walk(A)@8  {v43['void_frac']:.3f}" + ('  CLOZE — do not read' if v46 else '  ok'))
    v89(f"HIT        A {v43['hit_a']:.4f}  B {v43['hit_b']:.4f}  concat {v43['hit_concat']:.4f}  both {v43['hit_both']:.4f}  vs max(A,B) {v45:+.4f}")
    v89(f"NULL       both {v43['null_both']:.4f}")
    v3(v18.v8).v109.v90(parents=True, exist_ok=True)
    v3(v18.v8).v91(v119.v110(v43, indent=2), encoding='utf-8')
    v89(f'wrote {v18.v8}')
    return 0
if v47 == '__main__':
    raise v92(v111())