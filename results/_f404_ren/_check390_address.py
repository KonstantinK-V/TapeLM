"""Check of 390's address audit ON A DESIGNED TAPE. No torch, no corpus.

Twelve properties, each a WRONG NUMBER rather than an exception - a checker that only catches
crashes catches nothing, because every fault this project has actually made printed a plausible
number. And every property here is verified the only way a property can be: the failure it exists
to catch is RE-INTRODUCED at the bottom of this file, and the check must fire on it. A check that
has never fired is a comment.

THE TAPE. Twelve lines, `frame_max=1`, so an address is exactly (previous word | next word) and
nothing is approximate. The question is `zebra` in `the zebra sat`, and the tape is built so that:

    the | sat   cat  zebra  dog        THE QUESTION. own = {cat, dog}, hidden = zebra
    the | ran   zebra  moose           holds THE TRUTH, shares MY LEFT, and shares NO FILLER
                                       with the question - invisible to the walk, at any cap
    one | sat   pig  cow               shares MY RIGHT
    the | saw   fox  wolf              shares my left, but ONE OF ITS SLOTS IS ON THE HIDDEN
                                       SLOT'S LINE - a window artefact, dropped from every lane
    one | ran   elk  ibex              COMPOSED: my right-sharer's left x my left-sharer's right
    xx  | yy    cat  bird              shares the filler `cat` - the only place the walk reaches

So the standing arm CANNOT reach the answer on this question and the address CAN, by
construction. That is the claim the audit exists to measure at scale, written small enough to be
checked by hand.

  1. THE SECTION 27 LEAK. The walk is run from the profile WITHOUT the hidden row. Reading
     `prof[pid]` puts the truth in its own search key: `the|ran` holds `zebra`, so the walk would
     find it through the answer and then score the answer out of it - the exact fault that voided
     387 and 388.
  2. SAME-LINE PLACES ARE DROPPED FROM EVERY LANE. `the|saw` has a slot on the hidden slot's
     line; frames overlap, so that is the same words seen twice, not a second record.
  3. OWN VALUES ARE IN NO LANE. What already stands at this hole is not a candidate for it.
  4. THE ADDRESS IS BLIND TO FILLERS - both address neighbours are unreachable by the walk. If
     this reads zero the whole step is closed, which is what the audit's void check is for.
  5. THE QUESTION'S OWN PLACE IS NEVER A SOURCE, in any lane.
  6. COMPOSITION EXCLUDES WHAT WAS ALREADY IN HAND. `the|ran` and `one|sat` are both products of
     the pool's halves; counting them would let `comp_only` be won by re-offering the walk.
  7. BOTH HALVES CANNOT MATCH. The key is (w, left, right) with |left| = |right| = w, so sharing
     both halves is being the same place. Non-zero here means the key stopped being a bijection.
  8. AN ABSENT TRUTH RANKS 0, and a present one is 1-BASED - the reading discipline that a null
     must be readable on an absolute quantity.
  9. THE DECOY IS FREQUENCY MATCHED and is never the truth or an own value.
 10. THE MERGE IS ROUND-ROBIN AND CAPPED, never appended (347).
 11. THE AGGREGATOR ADDS UP THE PER-QUESTION NUMBERS AND NOTHING ELSE.
 12. NO LANE READS THE ASKING PLACE'S PROFILE - checked on the SOURCE as well, because 1 can only
     see the fault where the designed tape happens to expose it.

    python _check390_address.py
"""
from __future__ import annotations
import argparse
import random
import re
from collections import Counter
from pathlib import Path
import _audit390_address as A
v0 = v5('_audit390_address.py')
v1 = ['aa the cat sat bb', 'cc the zebra sat dd the fox saw ee2', 'ee the zebra ran ff', 'gg the moose ran hh', 'ii the dog sat jj', 'kk one pig sat ll', 'mm one cow sat nn', 'oo xx cat yy pp', 'qq xx bird yy rr', 's1 one elk ran s2', 's3 one ibex ran s4', 't1 the wolf saw t2']
v2 = v42.v6(places=8, topm=8, max_questions=10000)

def designed():
    v7 = v83.v43(v1, frame_max=1, min_fillers=1)
    v8 = v7['of_addr'][1, ('the',), ('sat',)]
    v9 = [v21 for v21 in v7['places'][v8] if v7['toks'][v21] == 'zebra'][0]
    return (v7, v8, v9)

def props():
    """Every property as a number read off the designed question. Returns the failures."""
    v10 = []
    v7, v8, v9 = v44()
    if v84(v7['places']) != 6:
        return [f"0. the designed tape is not the designed tape: {v84(v7['places'])} places"]
    v11 = v83.v45(v7, v9, v2, v101.v85(0))
    if v11 is None:
        return ['0. the designed question left the population']
    v12 = v11['_lanes']
    v13 = [v32 for v34 in v12.v102() for v32 in v34]
    if v11['n_fill'] != 1:
        v10.v86(f"1. the walk's neighbourhood is {v11['n_fill']} places, not 1 - the hidden row is back in the search key (section 27)")
    if v11['std8'] != 0:
        v10.v86('1. the standing arm reached the truth on a question built so it cannot - the walk is running from a profile that still holds the answer')
    if v11['dropped'] != 1:
        v10.v86(f"2. {v11['dropped']} same-line places dropped, not 1")
    if v87((1 for v32 in v13 if v32 in ('fox', 'wolf'))) != 0:
        v10.v86("2. a place on the hidden slot's own line is in a lane - frames overlap, so that is the same words twice and not a second record")
    if v87((1 for v32 in v13 if v32 in ('cat', 'dog'))) != 0:
        v10.v86('3. a value already standing at this hole is offered as a candidate for it')
    if (v11['n_half'], v11['n_half_new']) != (2, 2):
        v10.v86(f"4. address neighbours {v11['n_half']} of which unseen by the walk {v11['n_half_new']}, expected 2 and 2 - the address lane is not blind to fillers, so it is not a channel the walk lacks")
    if (v11['half8'], v11['half_only'], v11['half_any']) != (1, 1, 1):
        v10.v86(f"4. the truth is reachable by address and the lane missed it: half8={v11['half8']} half_only={v11['half_only']} half_any={v11['half_any']}")
    v14 = {v46 for v46 in v7['on_line'][v7['owner'][v9]] if v46 != v8}
    if v8 in v83.v88(v7, v8, v14):
        v10.v86('5. the asking place is its own address neighbour')
    v15 = v47((v7['toks'][v40] for v40 in v7['places'][v8] if v40 != v9))
    if v8 in v83.v103(v7, v8, v15, v14)[0]:
        v10.v86('5. the asking place is its own filler neighbour')
    if v8 in v83.v89(v7, v8, {v8}, v14):
        v10.v86('5. the asking place is returned as a composition of its own halves')
    if (v11['n_comp'], v11['n_comp_new']) != (1, 1):
        v10.v86(f"6. composition returned {v11['n_comp']} places ({v11['n_comp_new']} new), expected 1 and 1 - places already in hand are being counted as composed, so comp_only could be won by re-offering the walk")
    if v11['n_pool'] != 4:
        v10.v86(f"6. the composing pool is {v11['n_pool']} places, not 4")
    if v87((1 for v32 in v12['comp'] if v32 in ('elk', 'ibex'))) != 2:
        v10.v86(f"6. the composed lane is {v12['comp']}, not the fillers of `one|ran`")
    if v11['both_halves'] != 0:
        v10.v86(f"7. {v11['both_halves']} places share BOTH halves with the question - the address key is no longer (w, left, right) with |left| = |right| = w")
    if v11['_rank_std'] != 0:
        v10.v86(f"8. an absent truth ranks {v11['_rank_std']}, not 0")
    if v11['_rank_half'] != 4:
        v10.v86(f"8. the truth ranks {v11['_rank_half']} in the full address lane, expected 4 (1-based)")
    v37, v18 = (v101.v85(1), 0)
    v16 = {v7['toks'][v40] for v40 in v7['places'][v8] if v40 != v9}
    for v17 in v48(200):
        v49 = v83.v90(v7, 'zebra', v16 | {'zebra'}, v37)
        if v49 is None:
            continue
        if v49 == 'zebra' or v49 in v16:
            v18 += 1
    if v18:
        v10.v86(f'9. the decoy was the truth or an own value {v18} times in 200')
    v19 = v83.v50([1, 2, 3], [4, 5], cap=4)
    if v19 != [1, 4, 2, 5]:
        v10.v86(f'10. the merge is {v19}, not round-robin cut at the cap - an appended lane makes the arm win on budget (347)')
    if v84(v12['std']) > v2.v51:
        v10.v86(f"10. the standing offer is {v84(v12['std'])} long, past the cap")
    v52, v53 = v83.v54(v7, v2, v101.v85(0))
    v20 = v47()
    for v21 in [v40 for v91 in v7['places'] for v40 in v91]:
        v55 = v83.v45(v7, v21, v2, v101.v85(0))
        if v55:
            for v56, v32 in v55.v104():
                if not v56.v112('_'):
                    v20[v56] += v32
    v22 = [v56 for v56 in v20 if v56 not in ('rand8', 'rand_only', 'd_n', 'd_std8', 'd_half8', 'd_half_only', 'd_comp_only') and v52[v56] != v20[v56]]
    if v22:
        v10.v86(f'11. the aggregator disagrees with the per-question numbers on {v22}')
    v23 = v0.v57(encoding='utf-8')
    v24 = v23.v92('# ------', 2)[-1]
    for v25 in ('filler_nbrs', 'lane_share', 'lane_addr_full'):
        v58 = v105.v93(f'^def {v25}\\(.*?(?=\\n(?:def |# ---))', v23, v105.v106 | v105.v107)
        v59 = v105.v108('"""(?:.|\\n)*?"""', '', v58.v110(0)) if v58 else ''
        if 'prof"][pid]' in v59:
            v10.v86(f"12. {v25} reads the asking place's own profile - the hidden row is in the search key again (section 27)")
    if 'qprof' not in v24:
        v10.v86('12. the query profile is gone from the lanes')
    return v10

def _leak(v7, v26, v15, v14=()):
    return v83.v94['_orig_filler_nbrs'](v7, v26, v7['prof'][v26], v14)

def _no_drop(v7, v26, v14=()):
    return v83.v94['_orig_half_nbrs'](v7, v26, ())

def _keep_own(v7, v27, v16):
    return v83.v94['_orig_fillers_of'](v7, v27, ())

def _fillers_too(v7, v26, v14=()):
    """the address made to require a shared filler - the thing it exists NOT to require"""
    v28 = v83.v94['_orig_half_nbrs'](v7, v26, v14)
    v29 = {v46 for v46 in v28 if v95(v83.v94['_T_prof'][v46]) & v95(v83.v94['_T_prof'][v26])}
    return v47({v46: v28[v46] for v46 in v29})

def _no_pool(v7, v26, v30, v14=()):
    return v83.v94['_orig_compose_nbrs'](v7, v26, {v26}, v14)

def _rank0(v31, v32):
    for v60, v40 in v61(v31):
        if v40 == v32:
            return v60
    return 0

def _append(*v12, v33=None):
    v62, v63 = ([], v95())
    for v34 in v12:
        for v32 in v34:
            if v32 not in v63:
                v63.v111(v32)
                v62.v86(v32)
    return v62 if v33 is None else v62[:v33]

def _self_nbr(v7, v26, v14=()):
    v28 = v83.v94['_orig_half_nbrs'](v7, v26, v14)
    v28[v26] = 1
    return v28

def _both(v7, v26, v14=()):
    v28 = v83.v94['_orig_half_nbrs'](v7, v26, v14)
    return v47({v46: 2 for v46 in v28})

def _truth_decoy(v7, v35, v36, v37):
    return v35

def _double(v7, v38, v37):
    v52, v64 = v83.v94['_orig_run'](v7, v38, v37)
    v52['half8'] += 1
    return (v52, v64)
v3 = (('filler_nbrs', v65, '1.'), ('half_nbrs', v66, '2.'), ('fillers_of', v67, '3.'), ('half_nbrs', v68, '4.'), ('compose_nbrs', v69, '6.'), ('rank_of', v70, '8.'), ('interleave', v71, '10.'), ('half_nbrs', v72, '5.'), ('half_nbrs', v73, '7.'), ('band_draw', v74, '9.'), ('run', v75, '11.'))

def main() -> v4:
    v39 = v76()
    for v25, v77, v78 in v3:
        v79 = v96(v83, v25)
        v97(v83, f'_orig_{v25}', v79)
        v7, v98, v99 = v44()
        v83.v80 = v7['prof']
        v97(v83, v25, v77)
        try:
            v19 = v76()
        finally:
            v97(v83, v25, v79)
        if not v109((v113.v112(v78) for v113 in v19)):
            v39.v86(f'MUTATION {v78} ({v25}): the failure was re-introduced and check {v78} did not fire - it is a comment, not a check')
    for v40 in v39:
        v81('FAIL ' + v40)
    v81(f'{v84(v39)} failures, {v84(v3)} mutations' if v39 else f'all properties hold, and all {v84(v3)} re-introduced failures were caught')
    return 1 if v39 else 0
if v41 == '__main__':
    raise v82(v100())