"""389: does the calibration term actually remove the gauge? No torch needed for most of it, but
torch is used where it exists - the term is eight lines and every one of them can be wrong in a way
that still trains.

WHAT IS BEING CHECKED, and why each property is a wrong NUMBER rather than an exception:

  1. SHIFT INVARIANCE, ACROSS the batch. Adding one constant to every question's score must not
     change the term. If it did, the term would be a level control and could be won by pushing
     everything down - which is exactly how the three per-question refusal attempts were won by
     "always refuse".
  2. SHIFT SENSITIVITY, WITHIN the batch. Raising ONE question's score must change the term. This
     is the whole content: a per-question offset must no longer be free.
  3. DIRECTION. Raising an ANSWERABLE question's score must increase the term; raising an
     UNANSWERABLE one must decrease it. A sign error here trains the mind to be least confident
     exactly where the tape can answer, and nothing would crash.
  4. NO TARGET, NO TERM. All-positive and all-negative batches return exactly zero - not a
     fabricated uniform target, and not a NaN from dividing by npos = 0.
  5. THE OPTIMUM IS NOT A CONSTANT. A flat score vector scores strictly worse than one that
     separates the answerable questions, at the same mean. If a constant were optimal the term
     would teach nothing.
  6. THE LABEL IS THE TAPE'S, NOT THE MIND'S. reach_loss must record `ans` (truth in cands) and
     must NOT record `rt` (the mind's own correctness): training on `right` would be a moving
     target and would leave no held-out target to read the result on.
  7. ONE PICK FEEDS BOTH TERMS. reach_pick is called ONCE when either accumulator is live. Two
     calls would build the staged argmax twice with no guarantee the two terms agree about which
     world was settled on.
  8. THE ACCUMULATOR IS SCOPED. _CALIB_ACC is None outside the batch and is cleared in a
     `finally`, so a raising question cannot leave graph-holding tensors alive across steps.
  9. THE PAIRING IS POSITIONAL. A short accumulator raises rather than silently pairing question
     i's score with question j's label.
 10. B = 1 IS REFUSED. A softmax over one score is 1.0 whatever Phi says, and the gauge it exists
     to remove would still be per-question - the arm would be its own control wearing a flag.
 11. THE ARM IS IN THE TRANSPLANT SIGNATURE. A calibrated and an uncalibrated mind no longer
     measure on the same ruler.
 12. THE BATCH IS NOT QUIETLY SHARED. With both flags on, B is the larger of the two, and each
     term is added only if its own flag asked for it.

    python _check389_calib.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
v0 = v2('_stage289_derivation.py')

def _mini_torch():
    """The four operations calib_term uses, on plain floats. Forward only.

    This exists so the arithmetic of the term is checked in an environment without torch rather
    than taken on trust until the run. It deliberately implements the ops NAIVELY - if the stub
    and the real library ever disagreed, the properties below are about a formula that holds
    either way (shift invariance, sign, the zero case), so a stub that is wrong in the same
    direction as torch is not a way to pass.
    """
    import math

    class V:
        v9 = ('x',)

        def __init__(v38, v39):
            v38.v39 = [v52(v46) for v46 in v39]
        v10 = None
        v11 = None

        def __mul__(v38, v40):
            return v60([v75 * v76 for v75, v76 in v79(v38.v39, v40.v39)])

        def __truediv__(v38, v41):
            return v60([v75 / v52(v41) for v75 in v38.v39])

        def sum(v38):
            return v61(v38.v39)

        def __len__(v38):
            return v62(v38.v39)

    class _T:

        @v12
        def stack(v42):
            return v60([v52(v46) for v46 in v42])

        @v12
        def tensor(v39, v11=None, v10=None, v43=False):
            return v60(v39) if v70(v39, (v77, v78)) else v52(v39)

        @v12
        def zeros(v44, v11=None, v10=None):
            return 0.0

        @v12
        def device(v45):
            return v45

        @v12
        def log_softmax(v46, v47):
            v48 = v63(v46.v39)
            v49 = v71.v64(v61((v71.v80(v75 - v48) for v75 in v46.v39)))
            return v60([v75 - v48 - v49 for v75 in v46.v39])
    return v13()

def main() -> v1:
    v3 = v0.v14(encoding='utf-8')
    v4 = []
    try:
        import torch
        v15 = True
    except v16:
        v50, v15 = (v72(), False)
        v36('NOTE: torch missing - properties 1-5 run against a numeric stub of the four ops calib_term uses (stack, tensor, zeros, log_softmax). The FORMULA is checked; the gradient sub-check needs the real thing and is skipped.')
    if True:
        v17 = {'torch': v50}
        v18 = v57.v32('^def calib_term\\(.*?(?=\\n(?:def |@|\\w))', v3, v57.v33 | v57.v65)
        if not v18:
            v36(f'FAIL: calib_term not found in {v0}')
            return 1
        v51(v66(v18.v31(0), 'calib_term', 'exec'), v17)
        v19 = v17['calib_term']
        v20 = v50.v11('cpu')

        def T(v42):
            return [v50.v73(v52(v39), requires_grad=True) for v39 in v42]
        v21 = [0.3, -1.2, 2.0, 0.1]
        v22 = [1.0, 0.0, 1.0, 0.0]
        v23 = v52(v19(v68(v21), v22, v20))
        v24 = v52(v19(v68([v39 + 4.7 for v39 in v21]), v22, v20))
        if v67(v24 - v23) > 1e-05:
            v4.v56(f'1. the term is not shift invariant across the batch: {v23:.6f} -> {v24:.6f}. It can be won by a level, like `always refuse` was')
        v25 = v21[:]
        v25[0] += 1.0
        v26 = v21[:]
        v26[1] += 1.0
        v27 = v52(v19(v68(v25), v22, v20))
        v28 = v52(v19(v68(v26), v22, v20))
        if v67(v27 - v23) < 1e-06:
            v4.v56("2. raising one question's score changed nothing - the per-question offset is still free and the term does not tie the gauge")
        if not v27 > v23:
            v4.v56(f"3. raising an ANSWERABLE question's score did not raise the term ({v23:.6f} -> {v27:.6f}) - the sign is inverted")
        if not v28 < v23:
            v4.v56(f"3. raising an UNANSWERABLE question's score did not lower the term ({v23:.6f} -> {v28:.6f})")
        for v53, v54 in (('all-negative', [0.0, 0.0, 0.0, 0.0]), ('all-positive', [1.0, 1.0, 1.0, 1.0])):
            try:
                v49 = v52(v19(v68(v21), v54, v20))
            except v16 as e:
                v4.v56(f'4. a {v53} batch raised {v82(v81).v8}: {v81} - the term has no guard for a batch with no target')
                continue
            if v49 != 0.0 or v49 != v49:
                v4.v56(f'4. a {v53} batch returned {v49} instead of an exact zero')
        v29 = v52(v19(v68([0.5, 0.5, 0.5, 0.5]), v22, v20))
        v30 = v52(v19(v68([1.5, -0.5, 1.5, -0.5]), v22, v20))
        if not v30 > v29:
            v4.v56(f'5. a constant score is not worse than a separating one ({v29:.6f} vs {v30:.6f}) - the term teaches nothing')
        if v15:
            v55 = v68(v21)
            v19(v55, v22, v20).v69()
            if v55[0].v74 is None or v52(v55[0].v74) == 0.0:
                v4.v56('5. no gradient reaches the scores - the term is decorative')
    v5 = v57.v32('^def reach_loss\\(.*?(?=\\n(?:def |@|\\w))', v3, v57.v33 | v57.v65).v31(0)
    if '_CALIB_ACC.append((_sc, 1.0 if ans else 0.0))' not in v5:
        v4.v56("6. reach_loss does not record (raw score, answerable) - check the label: `rt` is the mind's own correctness and must NOT be the teacher")
    if v57.v32('_CALIB_ACC\\.append\\(\\(_sc, 1\\.0 if rt', v5):
        v4.v56('6. the calibration term is trained on `right`, its own moving correctness, and `right` is then no longer a held-out target')
    if v5.v58('reach_pick(q, l1, l2, own, cands, l3, lcands, keep_graph=True)') != 1:
        v4.v56('7. reach_pick is not called exactly once in reach_loss - the two terms have no guarantee of agreeing about which world was settled on')
    if 'if _SPEAK_ACC is not None or _CALIB_ACC is not None:' not in v5:
        v4.v56('7. the pick is not guarded by BOTH accumulators')
    v6 = v57.v32('if SPEAK_BATCH or CALIB_BATCH:.*?(?=\\n        elif VIEWS)', v3, v57.v33)
    if not v6:
        v4.v56('12. the batch block is not driven by both flags - --calib-batch alone would never form a batch and the term would silently never fire')
    else:
        v34 = v6.v31(0)
        if 'bn = max(SPEAK_BATCH, CALIB_BATCH)' not in v34:
            v4.v56("12. B is not the larger of the two batches - one term would be given the other's size without saying so")
        if '_CALIB_ACC = [] if CALIB_BATCH else None' not in v34:
            v4.v56('8. the calibration accumulator is opened even when the flag is off')
        if '_SPEAK_ACC = _CALIB_ACC = None' not in v34:
            v4.v56('8. the accumulators are not both cleared - a raising question would leave graph-holding tensors alive across steps')
        if 'finally:' not in v34:
            v4.v56('8. the accumulators are not cleared in a `finally`')
        if 'calibration batch recorded' not in v34:
            v4.v56('9. a short calibration accumulator does not raise - the pairing would stop being positional silently')
        if 'if CALIB_BATCH:' not in v34 or 'if SPEAK_BATCH:' not in v34:
            v4.v56('12. a term is added without checking its own flag')
    if '--calib-batch needs at least 2 questions' not in v3:
        v4.v56('10. --calib-batch 1 is not refused, and it is a constant term over a free per-question gauge - the arm would be its own control')
    v7 = v57.v32('"speak_batch": SPEAK_BATCH.*?"constrain": CONSTRAIN', v3, v57.v33)
    if not v7 or '"calib_batch": CALIB_BATCH' not in v7.v31(0):
        v4.v56('11. the calibration arm is missing from the transplant signature - a calibrated mind could be dropped into an uncalibrated run in silence')
    if v4:
        v36('FAIL')
        for v35 in v4:
            v36('  ' + v35)
        return 1
    v36('PASS  the term is invariant to a batch-wide shift and sensitive to a per-question one,')
    v36('  it rises on answerable questions and falls on unanswerable ones, a batch with no')
    v36('  target returns an exact zero, a constant score is strictly worse than a separating')
    v36("  one, the teacher is the tape's `answerable` and never the mind's `right`, one pick")
    v36('  feeds both terms, the accumulator is scoped and positional, B=1 is refused, and the')
    v36('  arm is in the transplant signature.')
    return 0
if v8 == '__main__':
    raise v37(v59())