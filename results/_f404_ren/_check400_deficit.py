"""Check of 400's deficit feature and of the artifact check that decided its one bit.

400 asks the dual of 399: does the line CLOSE what the rest of its scope lacks. The ways it can
print a number that means something else are specific, and one of them fired on the real run - the
`Return` bit, which turned out to be detecting the cut rather than the line.

  1. EQUIVARIANCE. Renaming every identifier must not move a single number. The feature is
     `type(node).__name__` and nothing else.
  2. THE LEAK, IN BOTH CURRENCIES. The pooled line is out of its own scope's TYPES and out of its
     LINE COUNT. Leaving it in either place lets the true scope be recognised by its own content.
  3. THE TWIN excludes the candidate itself, takes the nearest remaining-line count, and breaks
     ties by index - deterministic, so no number moves with a draw.
  4. THE DIRECTION IS THE ARGMAX OF THE DECLARED SCORE. Reversing a declared direction on seeing
     the number is the after-the-fact rescue this project has refused four times, so the code must
     contain no flip.
  5. THE POPULATION IS `amb_live` WITH THE TRUTH AMONG THE TIED - never the full corpus, where
     counting already answers.
  6. THE ARTIFACT CHECK READS THE FULL BODY. `bit_rivals_noret` must be computed with NOTHING
     dropped: if it read the remainder it would be the same quantity as the bit itself and could
     never fire. This is the check that voided a +0.34.
  7. THE BIT IS REPORTED BESIDE THE SCORE, NEVER BLENDED INTO IT.
  8. EXPECTED ACCURACY under uniform tie-breaking, exact and seed-free.

    python _check400_deficit.py
"""
from __future__ import annotations
import re
from pathlib import Path
import _audit398_scope as S
import _audit399_shape as H
import _audit400_deficit as A
v0 = v5('_audit400_deficit.py')
v1 = '\ndef alpha(one):\n    two = one + 1\n    for three in range(two):\n        two = two + three\n    return two\n\n\ndef beta(four):\n    five = four\n    return five\n\n\ndef gamma(six):\n    seven = six\n    eight = seven\n    nine = eight\n    return nine\n'
v2 = v1
for v6, v7 in (('alpha', 'zulu'), ('beta', 'victor'), ('gamma', 'whisky'), ('one', 'kilo'), ('two', 'lima'), ('three', 'mike'), ('four', 'november'), ('five', 'oscar'), ('six', 'papa'), ('seven', 'quebec'), ('eight', 'romeo'), ('nine', 'sierra')):
    v2 = v2.v26(v6, v7)

def deficit_vec(v8):
    v9 = v53.v27(v8)
    v28, v29 = v54.v30(v8, v9['owner'])
    v10 = v31((v55 for v55, v81 in v28.v82() if 'For' in v81))
    v11 = v32(v9['funcs'])
    v12 = v9['owner'][v10]
    from collections import Counter as C
    v13 = v33(v9['owner'].v56())
    v14 = [v13[v19] - (1 if v19 == v12 else 0) for v19 in v35(v11)]
    v15 = {v19: v54.v57(v29, v19, v10) for v19 in v35(v11)}
    v16 = v28.v34(v10, v58())
    v17 = [v32([v71 for v71 in v16 if v71 not in v15[v19]]) for v19 in v35(v11)]
    v18 = []
    for v19 in v35(v11):
        v59, v60 = v72.v61(v14, v19)
        v18.v62(v73(v17[v19] - (v17[v59] if v59 is not None else 0)))
    return (v18, v63(v16), v14)

def props(v8=None):
    v8 = v0.v43(encoding='utf-8') if v8 is None else v8
    v20 = []
    v36, v37, v38 = v39(v1)
    v40, v41, v42 = v39(v2)
    if (v36, v37, v38) != (v40, v41, v42):
        v20.v62(f'1. renaming moved the feature: {v36} vs {v40}, types {v37} vs {v41} - the deficit is reading identity')
    v9 = v53.v27(v1)
    v28, v29 = v54.v30(v1, v9['owner'])
    v21 = v31((v55 for v55, v81 in v28.v82() if 'For' in v81))
    v19 = v9['owner'][v21]
    if 'For' in v54.v57(v29, v19, v21) or 'For' not in v54.v57(v29, v19, -1):
        v20.v62("2. the pooled line's types are not taken out of its own scope")
    if 'rem = [n_body[i] - (1 if i == true_i else 0) for i in range(nf)]' not in v8:
        v20.v62("2. the pooled line is not taken out of its own scope's LINE COUNT, so the size-twin is matched against a size that still contains the answer")
    if v72.v61([5, 5, 9], 0)[0] != 1 or v72.v61([5, 9, 6], 0)[0] != 2:
        v20.v62(f'3. the twin is not the nearest by size: {v72.v61([5, 9, 6], 0)}')
    if v72.v61([7, 7, 7], 1)[0] == 1:
        v20.v62('3. a scope is its own twin, so the subtraction is identically zero')
    if v72.v61([5, 5, 9], 0)[1] != 0:
        v20.v62('3. the reported gap is not |size difference|')
    if v74.v64('expected_acc\\(\\[-', v8) or 's_def = [-' in v8 or '-x for x in s_def' in v8:
        v20.v62('4. the declared direction is negated somewhere - the argmax of the score is the attachment, and a sign flipped after the fact is a rescue')
    if 'c["deficit"] += H.expected_acc(s_def, true_i, tied)' not in v8:
        v20.v62('4. the score entering the accuracy is not `s_def` as declared')
    if 'if len(tied) < 2 or top <= 0.0 or true_i not in tied:' not in v8:
        v20.v62('5. the population is not amb_live with the truth among the tied')
    if 'full = {i: H.types_wo(body_t, i, -1) for i in tied}' not in v8:
        v20.v62('6. the artifact check does not read the FULL body - reading the remainder would make it the same quantity as the bit, and it could never fire')
    if 'c["bit_true_noret_full"] += int("Return" not in full[true_i])' not in v8:
        v20.v62("6. the true scope's full body is not checked, so 'the bit is the cut' cannot be told from 'the true scope simply has no Return'")
    v22 = v8[v8.v75('if "Return" in lt:'):]
    if 's_def' in v22[:600]:
        v20.v62('7. the bit is mixed into the general score')
    if v54.v65([1.0, 1.0, 0.0], 0, [0, 1, 2]) != 0.5:
        v20.v62('8. expected accuracy is not 1/|argmax|')
    return v20
v3 = (('the twin may be the candidate itself', '        if j == i:\n            continue', '        if False:\n            continue', '3.'), ('the twin ignores size', '        d = abs(rem[j] - rem[i])', '        d = float(j)', '3.'), ('the pooled line stays in the size the twin matches', '            rem = [n_body[i] - (1 if i == true_i else 0) for i in range(nf)]', '            rem = [n_body[i] for i in range(nf)]', '2.'), ('the declared direction is flipped', 'c["deficit"] += H.expected_acc(s_def, true_i, tied)', 'c["deficit"] += H.expected_acc([-x for x in s_def], true_i, tied)', '4.'), ('the population widens to every scored line', '            if len(tied) < 2 or top <= 0.0 or true_i not in tied:', '            if False:', '5.'), ('the artifact check reads the remainder, so it can never fire', '                full = {i: H.types_wo(body_t, i, -1) for i in tied}', '                full = {i: H.types_wo(body_t, i, ln) for i in tied}', '6.'))

def main() -> v4:
    v8 = v0.v43(encoding='utf-8')
    v23 = v44()
    for v45, v46, v47, v48 in v3:
        if v8.v76(v46) != 1:
            v23.v62(f'MUTATION {v48} ({v45}): its anchor occurs {v8.v76(v46)} times')
            continue
        v49 = v66(v72.v67)
        v50 = v8.v26(v46, v47, 1)
        try:
            v77(v83(v50, '<mutant>', 'exec'), v72.v67)
            v68 = v44(src=v50)
        except v69 as e:
            v68 = [f'{v48} the mutant raised {v86(v87).v25}']
        finally:
            v72.v67.v78()
            v72.v67.v79(v49)
        if not v80((v85.v84(v48) for v85 in v68)):
            v23.v62(f'MUTATION {v48} ({v45}): the failure was re-introduced and check {v48} did not fire - it is a comment, not a check')
    for v24 in v23:
        v51('FAIL ' + v24)
    v51(f'{v32(v23)} failures' if v23 else f'all properties hold, and all {v32(v3)} re-introduced failures were caught')
    return 1 if v23 else 0
if v25 == '__main__':
    raise v52(v70())