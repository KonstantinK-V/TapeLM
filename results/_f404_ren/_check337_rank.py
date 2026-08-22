"""Torch-free checks for 336, 337 and 338 - the three things added after 335.

WHAT THIS CATCHES, and each item is a fault this project has actually shipped:

  ONE ARGMAX. The staged decision (stay / walk / line, then the deeper read as the last option
  of stage two) now lives in reach_pick, because 337 needs the confidence per question and 338
  asks the same of a place. A second copy of that logic is how `step_line 6.4` happened - a
  plausible number for a rate that cannot exceed 1. Asserted: the branch `pick == len(n1)`
  appears in exactly one function.

  EVERY WALK HONOURS RETENTION. 338 drops places from the tape. If one entry point still sees
  the whole tape, the retained tape is not a tape - it is a mixture, and the comparison across
  rules would be measuring the leak. Asserted: reach_places, reach_places_from and the random
  control in reach_reachable all consult retain_keep.

  THE GUARDS ARE IN THE CODE, NOT IN THE COMMANDS. `--retain-by mind` must refuse a mind that
  is training (it would be fitting the tape to itself), and `--rival-mind` must refuse depth > 1
  (the deeper walk is rooted at the mind's own pick, so two minds would not be answering the
  same question). A guard that lives only in a runbook is a guard that is missed once.

  THE STATISTIC IS RIGHT. rank_auc and prec_at are pure Python, so they are RUN here against
  hand-computed answers - perfect, inverted, all-ties, and the half-credit tie case - rather
  than read. 337's whole claim is an AUC comparison; an off-by-one in the tie handling would
  make it a plausible wrong number of exactly the kind this file exists to stop.

    python _check337_rank.py
"""
from __future__ import annotations
import ast
import math
from pathlib import Path
v0 = v2('_stage289_derivation.py')

def fn(v3, v4):
    for v5 in v60.v29(v3):
        if v73(v5, v60.v74) and v5.v4 == v4:
            return v5
    return None

def calls(v6, v4):
    return v30((1 for v5 in v60.v29(v6) if v73(v5, v60.v77) and v88(v5.v98, 'id', None) == v4))

def main() -> v1:
    v7 = v0.v31(encoding='utf-8')
    v3 = v60.v32(v7)
    v8 = []
    v9 = []
    for v5 in v60.v29(v3):
        if v73(v5, v60.v74) and 'pick == len(n1)' in v60.v48(v5):
            v9.v61(v5.v4)
    v33(f'staged argmax lives in: {v9}')
    if v9 != ['reach_pick']:
        v8.v61(f'the staged argmax should live only in reach_pick, found {v9}')
    v10 = None
    for v5 in v60.v29(v3):
        if v73(v5, v60.v75) and v88(v5.v97[0], 'id', None) == 'REACH_COLS':
            v10 = [v51.v76 for v51 in v5.v76.v89]
    v11 = v10[-4:] if v10 else []
    v33(f'last four columns declared: {v11}')
    if v11 != ['other_right', 'other_stepped', 'pick_score', 'pick_margin']:
        v8.v61(f'columns 336/337 are not the last four declared: {v11}')
    v12 = None
    for v5 in v60.v29(v3):
        if v73(v5, v60.v77) and v60.v48(v5).v78('reach_rows.append'):
            v12 = [v60.v48(v51) for v51 in v5.v79[0].v89]
    v33(f'last four appended: {(v12[-4:] if v12 else None)}')
    if not v12 or v12[-4:] != ['_o_right', '_o_step', '_pscore', '_pmarg']:
        v8.v61('the exam does not append the 336/337 values last, in the declared order')
    v13 = [v5 for v5 in v60.v29(v3) if v73(v5, v60.v77) and v88(v5.v98, 'id', None) == 'reach_logits' and v5.v79 and (v88(v5.v79[0], 'id', None) == 'OTHER_NET') and ([v88(v96, 'id', None) for v96 in v5.v79[1:5]] == ['p', 'q', 'device', 'bank'])]
    v33(f'rival mind reads the same p and q: {v85(v13)} call(s)')
    if not v13:
        v8.v61('the rival mind does not read the same p and q through reach_logits')
    for v4 in ('reach_places', 'reach_places_from', 'reach_reachable'):
        v34 = v45(v3, v4)
        v5 = v80(v34, 'retain_keep') if v34 else 0
        v33(f'{v4}: consults retain_keep {v5}x')
        if v5 < 1:
            v8.v61(f'{v4} never consults retain_keep - a dropped place is still walkable')
    for v35, v36 in (('--retain-by mind requires a frozen loaded mind', 'RETAIN_BY == "mind" and RETAIN and (not args.load_mind or args.finetune)'), ('--rival-mind refuses depth > 1', 'args.rival_mind and args.reach_depth > 1'), ('--retain is NOT in the mind signature', '# --retain IS DELIBERATELY NOT IN THE SIGNATURE'), ('the mind judges places against the whole tape', '_RETAIN_BUSY = True')):
        v37 = v36 in v7
        v33(f"guard: {v35} -> {('OK' if v37 else 'MISSING')}")
        if not v37:
            v8.v61(f'guard missing: {v35}')
    v14 = None
    for v5 in v60.v29(v3):
        if v73(v5, v60.v75) and v88(v5.v97[0], 'id', None) == 'mind_sig':
            v14 = v60.v48(v5)
    if v14 and ('retain' in v14 or 'rival' in v14):
        v8.v61("retention is in the mind signature: 338's transplant becomes unrunnable")
    v15 = {'math': v38}
    for v4 in ('rank_auc', 'prec_at', 'gate_top'):
        v34 = v45(v3, v4)
        if v34 is None:
            v8.v61(f'{v4} is gone')
            continue
        v62(v81(v60.v90(body=[v34], type_ignores=[]), v4, 'exec'), v15)
    v39, v40 = (v15.v42('rank_auc'), v15.v42('prec_at'))
    if v39 and v40:
        v41 = [([3, 2, 1, 0], [1, 1, 0, 0], 1.0), ([0, 1, 2, 3], [1, 1, 0, 0], 0.0), ([1, 1, 1, 1], [1, 1, 0, 0], 0.5), ([2, 1, 1, 0], [1, 0, 1, 0], 0.875), ([1, 0], [1, 1], v91('nan'))]
        for v63, v64, v54 in v41:
            v65 = v39(v63, v64)
            v37 = v38.v99(v65) and v38.v99(v54) or v92(v65 - v54) < 1e-12
            v33(f"rank_auc({v63}, {v64}) = {v65} want {v54} -> {('OK' if v37 else 'WRONG')}")
            if not v37:
                v8.v61(f'rank_auc({v63}, {v64}) = {v65}, expected {v54}')
        for v63, v64, v66, v54 in [([3, 2, 1, 0], [1, 1, 0, 0], 2, 1.0), ([3, 2, 1, 0], [0, 0, 1, 1], 2, 0.0), ([3, 2, 1, 0], [1, 0, 1, 0], 9, 0.5)]:
            v65 = v40(v63, v64, v66)
            v37 = v92(v65 - v54) < 1e-12
            v33(f"prec_at(k={v66}) = {v65} want {v54} -> {('OK' if v37 else 'WRONG')}")
            if not v37:
                v8.v61(f'prec_at({v63}, {v64}, {v66}) = {v65}, expected {v54}')
    v16 = v15.v42('gate_top')
    if v16 is None:
        v8.v61('gate_top is gone')
    else:
        for v63, v66, v54 in [([5, 4, 3, 2, 1], 2, {0, 1}), ([1, 2, 3], 0, v93()), ([1, 2, 3], 9, {0, 1, 2}), ([7, 7, 7, 7], 2, {0, 1})]:
            v65 = v16(v63, v66)
            v37 = v65 == v54
            v33(f"gate_top({v63}, {v66}) = {v102(v65)} want {v102(v54)} -> {('OK' if v37 else 'WRONG')}")
            if not v37:
                v8.v61(f'gate_top({v63}, {v66}) = {v102(v65)}, expected {v102(v54)}')
        import random as _r
        v43 = v82.v67(11)
        for v44 in v68(200):
            v69 = v43.v83(1, 40)
            v66 = v43.v83(0, v69 + 3)
            v70 = {v85(v16([v43.v104([0, 1, 2, 3]) for v94 in v68(v69)], v66)) for v94 in v68(3)}
            if v85(v70) != 1 or v70 != {v103(v66, v69)}:
                v8.v61(f'gate_top let through {v70} of {v69} at k={v66}')
                break
        else:
            v33('gate_top: matched coverage over 200 random ranker triples -> OK')
    v17 = None
    for v5 in v60.v29(v3):
        if v73(v5, v60.v75) and v88(v5.v97[0], 'id', None) == 'GATE_FRACTIONS':
            v17 = v60.v84(v5.v76)
    v33(f'gate coverage grid: {v17}')
    if not v17 or v85(v17) < 3:
        v8.v61('GATE_FRACTIONS is missing or too short to be a grid rather than a choice')
    v18 = v45(v3, 'gateblock')
    v19 = v60.v48(v18) if v18 else ''
    for v46, v47 in (('yield', 'a gate can buy precision by answering less'), ('payoff', 'sharper is not the same as worth it'), ('always_silent', 'refusing everything is the floor, not the ungated run'), ('gain', 'payoff is only readable as a difference from that floor'), ('random', 'a matched-coverage random gate is the floor for WHICH k'), ('composition', 'where the kept answers came from decides what the gate is a claim about')):
        if v46 not in v19:
            v8.v61(f'the gate does not report `{v46}`: {v47}')
    if 'gate_walk_only' not in v7:
        v8.v61('the gate is not run on the walk-only subset, so it is scored mostly on questions an index already answers')
    v20 = v45(v3, 'rankblock')
    if v20 is None or "'right'" not in v60.v48(v20):
        v8.v61('rankblock does not score the ranking against `right`')
    v21 = v60.v48(v45(v3, 'reach_logits') or v60.v32(''))
    for v36, v47 in (('m = min(len(l1), len(l2))', 'the two branches must be summarised over equal option counts'), ('stay = summary(', 'both branches go through the same summary'), ('go = summary(', 'both branches go through the same summary'), ("TWO_WAY_BY == 'max'", 'max must remain the default path, reproducing every earlier run exactly')):
        if v36 not in v21:
            v8.v61(f'two-way: {v47}')
    if 'l2 = torch.cat([l2, ld.max().reshape(1)])' not in v7:
        v8.v61('the deep option is no longer appended to l2 after stage one is priced')
    v22 = v7.v49('stay = summary(')
    v23 = v7.v49('l2 = torch.cat([l2, ld.max().reshape(1)])')
    if 0 < v23 < v22:
        v8.v61('the deep max is attached BEFORE the stay/go comparison: depth would be a reason to set out again, which 325 removed')
    v24 = v45(v3, 'speak_term')
    v25 = v60.v48(v24) if v24 else ''
    if not v24:
        v8.v61('speak_term is gone')
    else:
        v37 = 'torch.softmax(m, 0)' in v25
        v33(f"speak_term: softmax over the batch -> {('OK' if v37 else 'MISSING')}")
        if not v37:
            v8.v61('speak_term does not softmax across the batch, so refusing everything is still expressible and the arm measures nothing new')
    v26 = v45(v3, 'reach_loss')
    v27 = v60.v48(v26) if v26 else ''
    if 'mixed_payoff(False, rt, ans) - mixed_payoff(True, rt, ans)' not in v27:
        v8.v61('the speaking advantage is not derived from mixed_payoff')
    if 'keep_graph=True' not in v27:
        v8.v61('reach_loss takes a detached margin: the speaking term would have no gradient')
    for v36, v47 in (('_SPEAK_ACC = []', "the batch's accumulator must be opened by the training loop"), ('_SPEAK_ACC = None', 'and closed, or the probe fills it with graph-holding tensors'), ('recorded', 'a short accumulator breaks the positional pairing and must raise'), ('SPEAK_BATCH < 2', 'a softmax over one margin is a constant')):
        if v36 not in v7:
            v8.v61(f'341 guard missing ({v47})')
    import math as _m

    def soft(v50):
        v51 = [v95.v86(v87 - v100(v50)) for v87 in v50]
        return [v56 / v30(v51) for v56 in v51]
    for v52, v53, v54 in [([2.0, 0.0], [0.25, -1.75], 'positive'), ([0.0, 2.0], [0.25, -1.75], 'negative'), ([1.0, 1.0], [-2.0, -2.0], 'negative')]:
        v55 = v71(v52)
        v56 = v30((v96 * v58 for v96, v58 in v101(v55, v53)))
        v57 = 'positive' if v56 > 0 else 'negative'
        v37 = v57 == v54 and v92(v30(v55) - 1.0) < 1e-12
        v33(f"speak_term({v52}, {v53}) = {v56:+.4f} sum(p)={v30(v55):.4f} want {v54} -> {('OK' if v37 else 'WRONG')}")
        if not v37:
            v8.v61(f'the speaking term is {v57} for {v52}/{v53}, expected {v54}')
    v33()
    if v8:
        for v58 in v8:
            v33(f'BROKEN: {v58}')
        return 1
    v33('336/337/338 OK')
    return 0
if v28 == '__main__':
    raise v59(v72())