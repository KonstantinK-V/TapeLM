"""Check of 34.4: the walk_only route mask, and the reader that decides whether to run it.

The stage half is read off the source; the reader half is RUN, on dumps this file writes, because
the reader is pure python and a checker that can execute the thing it checks should.

  1. OFF IS OFF. `all` is the default and the mask lives inside its own branch, so every earlier
     run is bit for bit.
  2. REFUSED WITHOUT --two-way. Without it stage one's logits ARE the own worlds, so detaching
     them would freeze the home pick too and the arm would differ from its control in two
     decisions instead of one.
  3. THE MASK IS THE TAPE'S: `answerable and not truth_in_own`. A mask on the mind's own
     correctness would be a moving target - the fault calib_term's docstring names.
  4. ONLY THE ROUTER IS CUT. The detach is on `p1`, after the softmax. Detaching `l1` would cut
     the summaries' path to the worlds and take both picks with it.
  5. THE MASK IS IN THE LOSS ONLY. If ROUTE_ON reached reach_logits or reach_pick, the EXAM would
     see the truth - which is a leak, not a lever.
  6. THE DENOMINATOR IS EVERY QUESTION. `n` is counted before the branch, `live` inside it, or
     `route_on_live` reads 1.0 on an arm that trained the router on 4% of its questions.
  7. REPORTED, and 8. IN THE ARM SIGNATURE - a mind whose router was taught on a different
     population is not the same mind.
  9. THE READER'S FOUR VOID CHECKS EXIST WITH THEIR THRESHOLDS, and they are what decides whether
     the arm is run at all.
 10. THE READER POOLS AS COUNTS. A mean of four rates is a different number, and the population is
     recovered exactly as pick.n / arrive.
 11. A REPORTED nan READS AS MISSING, never as a zero that pools - the reading discipline that a
     null must be read on an absolute quantity.

Every property is a wrong number, and every one is verified by re-introducing its own failure.

    python _check394_route.py
"""
from __future__ import annotations
import io
import json
import re
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import _read394_walkonly as R
v0 = v6('_stage289_derivation.py')
v1 = v6('_read394_walkonly.py')

def code_of(v7):
    return v54.v27('"""(?:.|\\n)*?"""', '', v7)

def body(v8, v9):
    v10 = v54.v28(f'^def {v9}\\(.*?(?=\\n(?:def |@|# ---))', v8, v54.v55 | v54.v56)
    return v10.v57(0) if v10 else ''
v2 = ((1337, 0.3, 120, 44, 31, 0.061), (8642, 0.28, 110, 39, 30, 0.058))

def write_dumps(v11, v12=None):
    for v29, v30, v31, v32, v33, v34 in v2:
        v35 = {'seed': v29, 'wall_s': 1.0, 'reach': {'held_out': {'walk_only_arrive': v30 if v12 is None else v12, 'walk_only_pick': {'n': v31, 'mind': v32, 'rival': 20, 'count_rival': v33}, 'step_rate': 0.21, 'deep_only_rate': v34, 'hit_of_deep_only': v87('nan'), 'hit_of_own': 0.7, 'ceiling': 0.4, 'n': 9000}}}
        (v11 / f'stage289_decision_s{v29}.json').v58(v85.v76(v35))

def read_out(v11):
    v13 = v59.v36()
    with v60(v13):
        v77.v61([v89(v90) for v90 in v94(v11.v95('*.json'))] + ['--held'])
    return v13.v37()

def props(v8=None, v14=None):
    v8 = v0.v62(encoding='utf-8') if v8 is None else v8
    v14 = v1.v62(encoding='utf-8') if v14 is None else v14
    v15 = []
    v16 = v38(v63(v8, 'reach_loss'))
    if not v54.v28('^ROUTE_ON = "all"', v8, v54.v56):
        v15.v64('1. ROUTE_ON does not default to `all`, so earlier runs are not bit for bit')
    if 'if ROUTE_ON == "walk_only":' not in v16:
        v15.v64('1. the mask is not inside its own branch in reach_loss')
    if 'if ROUTE_ON != "all" and not args.two_way:' not in v8:
        v15.v64('2. --route-on is accepted without --two-way, where cutting the route cuts the home pick with it')
    v17 = v16.v39('if ROUTE_ON == "walk_only":')
    v18 = v16.v39('p2 = torch.softmax(l2, 0)', v65(0, v17))
    v19 = v16[v17:v18] if 0 <= v17 < v18 else ''
    if not v19:
        v15.v64('0. the mask block was not found between its branch and the stage-two softmax')
    if 'if ans and q["truth_value"] not in set(own):' not in v16:
        v15.v64('3. the mask is not `answerable and not truth_in_own`')
    for v20 in ('mind_right', '_said ==', 'argmax'):
        if v20 in v19:
            v15.v64(f"3. the mask reads {v20!r} - the mind's own correctness is a moving target")
    if 'p1 = p1.detach()' not in v16:
        v15.v64("4. the route's probability is not the thing detached")
    for v20 in ('l1 = l1.detach()', 'l2 = l2.detach()', 'v2 = v2.detach()', 'v_stay = v_stay.detach()', 'lo = lo.detach()'):
        if v20 in v16:
            v15.v64(f'4. {v20} - that cuts a PICK, not the route, and the arm would differ from its control in two decisions')
    for v9 in ('reach_logits', 'reach_pick', 'reach_candidates', 'reach_move_pick'):
        if 'ROUTE_ON' in v38(v63(v8, v9)):
            v15.v64(f'5. ROUTE_ON reaches {v9} - the mask would be visible to the exam')
    v40, v41, v42 = (v19.v39('_ROUTE_LIVE["n"]'), v19.v39('if ans and'), v19.v39('_ROUTE_LIVE["live"]'))
    if not 0 <= v40 < v41 < v42:
        v15.v64(f'6. the live count is not n-before-branch, live-inside (n={v40} if={v41} live={v42}) - route_on_live would read 1.0 on an arm that trained the router on a few percent of its questions')
    for v21 in ('route_on_live', 'route_on_seen'):
        if f'"{v21}"' not in v8:
            v15.v64(f'7. {v21} is not reported')
    v22 = v8[v8.v39('# 341 IS IN THE SIGNATURE'):][:900]
    if '"route_on": ROUTE_ON' not in v22:
        v15.v64('8. route_on is not in the arm signature - a mind whose router was taught on another population would transplant onto one that was not')
    for v23 in ('V1 FIRED', 'V2 FIRED', 'V3 FIRED', 'V4 FIRED', 'arrive >= 0.95', 'share < 0.02', 'deep_r <= 0.05', 'c_rate >= m_rate'):
        if v23 not in v14:
            v15.v64(f'9. the reader is missing {v23!r} - a void check that is not in the file is not a void check')
    with v78.v66() as v43:
        v11 = v6(v43)
        v67(v11)
        v44 = v68(v11)
        for v23 in ('walk_only 793 of 18000 rows (0.0440)', 'arrive 0.2901', 'mind 0.3609', 'count 0.2652'):
            if v23 not in v44:
                v15.v64(f'10. the pooled reading is wrong: expected {v23!r} in\n{v44}')
        if 'no void check fired' not in v44:
            v15.v64(f'10. a void check fired on a dump built not to fire one:\n{v44}')
        if 'deep_only 0.0595' not in v44:
            v15.v64('11. deep_only did not pool to 0.0595 - a nan or a missing key is being read as a number')
        v67(v11, arrive_override=0.97)
        v45 = v68(v11)
        if 'V1 FIRED' not in v45:
            v15.v64('9. V1 did not fire on arrive 0.97')
    with v78.v66() as v43:
        v11 = v6(v43)
        v67(v11)
        for v69, (v29, *v86) in v70(v2):
            v71 = v11 / f'stage289_decision_s{v29}.json'
            v35 = v85.v79(v71.v62())
            if v69:
                v35['reach']['held_out']['deep_only_rate'] = v87('nan')
            v71.v58(v85.v76(v35))
        v46 = v68(v11)
        if 'deep_only 0.0610' not in v46:
            v15.v64(f'11. a nan deep_only_rate is being pooled instead of skipped:\n{v46}')
    return v15
v3 = (('the mask runs by default', 'ROUTE_ON = "all"', 'ROUTE_ON = "walk_only"', '1.'), ('accepted without --two-way', 'if ROUTE_ON != "all" and not args.two_way:', 'if False:', '2.'), ("the mask reads the mind's own correctness", 'if ans and q["truth_value"] not in set(own):', 'if ans and int(l1.argmax()) == 1:', '3.'), ('the logits are detached, taking both picks with them', '            p1 = p1.detach()', '            l1 = l1.detach()', '4.'), ('the mask is visible to the offer', '    if which is None and "_reach_c" in q:', '    if ROUTE_ON == "walk_only" or (which is None and "_reach_c" in q):', '5.'), ('the denominator counts only the live questions', '        _ROUTE_LIVE["n"] += 1\n        if ans and q["truth_value"] not in set(own):\n            _ROUTE_LIVE["live"] += 1', '        if ans and q["truth_value"] not in set(own):\n            _ROUTE_LIVE["n"] += 1\n            _ROUTE_LIVE["live"] += 1', '6.'), ('not reported', '"route_on_live": (_ROUTE_LIVE["live"]', '"unused_live": (_ROUTE_LIVE["live"]', '7.'), ('not in the arm signature', '\n                "move_teach": MOVE_TEACH, "route_on": ROUTE_ON,', '\n                "move_teach": MOVE_TEACH,', '8.'))
v4 = (('a void check is gone', 'if not math.isnan(arrive) and arrive >= 0.95:', 'if False:', '9.'), ('the population is not recovered from the two reported numbers', '    pop = (stepped / arrive) if (arrive and stepped is not None and arrive > 0) else None', '    pop = stepped', '10.'), ('a nan pools as a zero', '    return None if math.isnan(v) else v', '    return v', '11.'))

def main() -> v5:
    v8, v14 = (v0.v62(encoding='utf-8'), v1.v62(encoding='utf-8'))
    v24 = v47()
    for v9, v48, v49, v50 in v3:
        if v8.v80(v48) != 1:
            v24.v64(f'MUTATION {v50} ({v9}): its anchor occurs {v8.v80(v48)} times')
            continue
        if not v81((v92.v91(v50) for v92 in v47(src=v8.v88(v48, v49, 1)))):
            v24.v64(f'MUTATION {v50} ({v9}): the failure was re-introduced and check {v50} did not fire - it is a comment, not a check')
    for v9, v48, v49, v50 in v4:
        if v14.v80(v48) != 1:
            v24.v64(f'MUTATION {v50} ({v9}): its anchor occurs {v14.v80(v48)} times')
            continue
        v51 = v72(v77.v73)
        v74(v82(v14.v88(v48, v49, 1), '<mutant>', 'exec'), v77.v73)
        try:
            v75 = v47(rdr=v14.v88(v48, v49, 1))
        finally:
            v77.v73.v83()
            v77.v73.v84(v51)
        if not v81((v92.v91(v50) for v92 in v75)):
            v24.v64(f'MUTATION {v50} ({v9}): the failure was re-introduced and check {v50} did not fire - it is a comment, not a check')
    for v25 in v24:
        v52('FAIL ' + v25)
    v52(f'{v93(v24)} failures' if v24 else f'all properties hold, and all {v93(v3) + v93(v4)} re-introduced failures were caught')
    return 1 if v24 else 0
if v26 == '__main__':
    raise v53(v61())