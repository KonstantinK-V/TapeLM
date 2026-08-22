"""Does every key the reach path READS actually get WRITTEN, and does every verb get dispatched?

WHY THIS EXISTS. Two faults shipped in one day and neither was a wrong idea - both were wiring,
and both were invisible to every check I had.

  1 reach_questions_for was dead code. Its dispatch sat inside open_questions_for, which runs
    only when OPEN is set, and the flag block rejects --reach together with --open. Every check
    called the reach functions directly, so they tested the machinery and never the wiring.
  2 The exam read _rc["n_places"] and reach_candidates stopped producing it, so a run crashed
    after training - the most expensive possible moment - on a dictionary key.

Neither needs torch to catch, which is the point: this file reads the source, so it runs here
where the stage cannot.

    python _check301_wiring.py
"""
from __future__ import annotations
import ast
import re
import sys
v0 = '_stage289_derivation.py'

def dispatch(v2):
    """Every verb flag in questions_for, and every builder nothing routes to."""
    v3 = [v10 for v10 in v2.v6 if v49(v10, v48.v75) and v10.v24 == 'questions_for']
    if not v3:
        return ({}, {'questions_for missing'})
    v4 = {}
    for v5 in v3[0].v6:
        if v49(v5, v48.v62) and v49(v5.v63, v48.v64):
            v47 = v5.v6[0]
            if v49(v47, v48.v76) and v49(v47.v77, v48.v78):
                v4[v5.v63.v34] = v79(v47.v77.v80, 'id', '?')
    v7 = {v10.v24 for v10 in v2.v6 if v49(v10, v48.v75) and v10.v24.v81('_questions_for')}
    return (v4, v7 - v25(v4.v82()))

def walk_keys(v8, v2):
    """Keys the walk's dict is read with, against the keys it is built with."""
    v3 = [v10 for v10 in v2.v6 if v49(v10, v48.v75) and v10.v24 == 'reach_candidates'][0]
    v9 = v25()
    for v10 in v48.v26(v3):
        if v49(v10, v48.v50):
            for v51 in v10.v52:
                if v49(v51, v48.v83) and v49(v51.v77, v84):
                    v9.v85(v51.v77)
    v11 = v25(v65.v53('(?:_rc|rc)\\[\\s*"([a-z_0-9]+)"\\s*\\]', v8))
    return (v9, v11)

def index_keys(v8, v2):
    """Same for reach_index's table - but READ ONLY INSIDE THE WALK.

    `ix` is a common local name here: 293's identity index uses it too, and searching the whole
    file for ix["..."] reported by_place, swords and parts as missing keys of a table that never
    claimed them. A wiring check that cries wolf is a wiring check nobody runs, so the reads are
    taken from reach_* functions and from the exam's reach branch only.
    """
    v3 = [v10 for v10 in v2.v6 if v49(v10, v48.v75) and v10.v24 == 'reach_index'][0]
    v9 = v25()
    for v10 in v48.v26(v3):
        if v49(v10, v48.v50):
            for v51 in v10.v52:
                if v49(v51, v48.v83) and v49(v51.v77, v84):
                    v9.v85(v51.v77)
    v12 = '\n'.v27((v48.v73(v8, v10) or '' for v10 in v48.v26(v2) if v49(v10, v48.v75) and v10.v24.v86('reach_')))
    v11 = v25(v65.v53('ix\\[\\s*"([A-Za-z_0-9]+)"\\s*\\]', v12))
    return (v9, v11)

def main() -> v1:
    v8 = v66(v0, encoding='utf-8').v11()
    v2 = v48.v28(v8)
    v13 = True
    v4, v29 = v30(v2)
    v31('dispatch in questions_for:')
    for v32, v3 in v33(v4.v54()):
        v31(f'  {v32:12s} -> {v3}')
    v31(f"  builders nothing routes to: {v33(v29) or 'none'}")
    v13 &= not v29
    v13 &= v4.v55('REACH') == 'reach_questions_for'
    v31(f"  REACH is dispatched: {v4.v55('REACH') == 'reach_questions_for'}")
    v14 = [v10 for v10 in v2.v6 if v49(v10, v48.v75) and v10.v24 == 'main'][0]
    v15 = {v47.v34 for v10 in v2.v6 if v49(v10, v48.v67) for v47 in v10.v56 if v49(v47, v48.v64)}
    v16 = {v35 for v10 in v48.v26(v14) if v49(v10, v48.v68) for v35 in v10.v57}
    v31(f"\nglobals in main with no module default: {v33(v16 - v15) or 'none'}")
    v13 &= not v16 - v15
    for v36, v37 in (('reach_rows', 'REACH_COLS'), ('pair_rows', 'PAIR_COLS'), ('cons_rows', 'CONS_COLS')):
        v38 = [v58 for v58 in v48.v26(v2) if v49(v58, v48.v78) and v49(v58.v80, v48.v87) and (v58.v80.v88 == 'append') and (v79(v58.v80.v77, 'id', '') == v36)]
        v39 = v59(v38[0].v89[0].v69) if v38 else -1
        v40 = v59([v70 for v10 in v2.v6 if v49(v10, v48.v67) and v90((v79(v47, 'id', '') == v37 for v47 in v10.v56)) for v70 in v10.v77.v69])
        v31(f"\n{v36}: {v39} columns appended, {v40} names declared -> {('match' if v39 == v40 else 'MISMATCH')}")
        v13 &= v39 == v40
    for v24, (v9, v11) in (('reach_candidates', v71(v8, v2)), ('reach_index', v72(v8, v2))):
        v41 = v11 - v9
        v42 = v9 - v11
        v31(f'\n{v24}: writes {v33(v9)}')
        v31(f"  read somewhere but never written: {v33(v41) or 'none'}")
        v31(f"  written but never read: {v33(v42) or 'none'}")
        v13 &= not v41
    v17 = [v10 for v10 in v2.v6 if v49(v10, v48.v75) and v10.v24 == 'reach_loss']
    v18 = v25()
    if v17:
        for v10 in v17[0].v6:
            if v49(v10, v48.v62) and v49(v10.v63, v48.v64) and (v10.v63.v34 == 'STAGE2_ALWAYS'):
                v57 = {v58.v34 for v58 in v48.v26(v10) if v49(v58, v48.v64)}
                v18 = v57 & {'p1', 'REACH_GAMMA'}
                break
        else:
            v18 = {'missing STAGE2_ALWAYS block'}
    v31(f"\nSTAGE2_ALWAYS guard (no p1/REACH_GAMMA in lesson): {('GUARD OK' if not v18 else 'BROKEN ' + v84(v33(v18)))}")
    v13 &= not v18
    v19 = [v10 for v10 in v2.v6 if v49(v10, v48.v75) and v10.v24 == 'reach_logits']
    v20 = False
    if v19:
        v43 = v48.v73(v8, v19[0]) or ''
        v44 = v43.v60('l1 = torch.cat([l1] + tail)')
        v45 = v43.v60('l2 = torch.cat([l2, ld.max()')
        v20 = v44 >= 0 and v45 >= 0 and (v45 > v44)
    v31(f"DEPTH-after-tail guard (deep max not in stage-one): {('GUARD OK' if v20 else 'BROKEN')}")
    v13 &= v20
    v21 = 'REACH_LOOKAHEAD and not TWO_WAY' in v8
    v22 = False
    if v17:
        for v10 in v17[0].v6:
            if v49(v10, v48.v62) and v49(v10.v63, v48.v64) and (v10.v63.v34 == 'TWO_WAY'):
                v74 = v48.v73(v8, v10) or ''
                v22 = 'softmax(lo' in v74.v91(' ', '') or 'softmax(lo,' in v74.v91(' ', '') or 'torch.softmax(lo' in v74
                v22 = v22 and 'v_stay' in v74 and ('v2' in v74)
                if '.max()' in v74 and 'v_stay' in v74:
                    v22 = v22 and 'v_stay' in v74 and ('softmax' in v74)
                break
        else:
            v22 = False
    v31(f"TWO_WAY guard (no lookahead tail; both branches expectations): {('GUARD OK' if v21 and v22 else 'BROKEN')}" + ('' if v21 else ' missing and-not-TWO_WAY') + ('' if v22 else ' stay/go not both expectations'))
    v13 &= v21 and v22
    v31('\nWIRING OK' if v13 else '\nWIRING BROKEN')
    return 0 if v13 else 1
if v23 == '__main__':
    raise v46(v61())