"""Prove 298 - the frame tape and the route - before spending an hour on either.

298 is the whole construction rather than another arm: the write path becomes counting, the
mind may read on before answering, and one payoff prices finding, stepping and silence together.
Five things have to hold:

  1 A FRAME IS A RECURRENCE, NOT A RULE. The address is a hole whose surroundings the corpus
    wrote at least twice, its width is the widest the corpus supports, and it exists only if the
    hole took at least two different values. No tau, no stopwords, no grammar, no chosen length.
  2 THE VALUE IS WHAT VARIES. Every assertion's value is the token that stood in the hole, and
    its context is the line, so the ink has a sentence to work with and ctx_fp excludes the value.
  3 WITHOUT IMPORTS THE CANDIDATES ARE INDISTINGUISHABLE - and that is why the step exists. An
    absent value gives a bit-identical graph, which wrecked 289's ladder; here it is the honest
    reason to read more instead of guessing.
  4 WITH IMPORTS THEY SEPARATE, so reading on actually buys something.
  5 THE ROUTE IS ONE SOFTMAX. `expand` is one more world scored by the same Phi, the loss is the
    closed-form expected payoff of the two-step decision, and reading is priced.

    python _check298_route.py
"""
from __future__ import annotations
import random
import torch
import _stage289_derivation as s289
import _tape_frames as tframes
from _check293_identity import FakeBank
from _check294_open import pack294

def main() -> v0:
    v1 = True
    v2 = v66.v23('cpu')
    v3 = ['the cat sat on the mat', 'the dog sat on the rug', 'the cat sat on the rug', 'a bird flew over the mat']
    v6, v24, v25 = v67.v26(v3, frame_max=6, min_fillers=2)
    v4 = {}
    for v5 in v6:
        v4.v92(v5['address'], []).v68(v5['value'])
    v27(f'1 addresses {v69(v24)}: {v93(v75(v4.v100())[:6])}')
    v7 = v29((v69(v46(v94)) >= 2 for v94 in v4.v85())) and v69(v24) > 0
    v1 &= v28(v7)
    v27(f'  every address took at least two different fillers: {v28(v7)}')
    v8 = [v15 for v15 in v4 if v15.v84('the|sat')]
    v7 = v28(v8) and v46(v4[v8[0]]) == {'cat', 'dog'}
    v1 &= v28(v7)
    v27(f"  'the ___ sat' -> {(v97(v46(v4[v8[0]])) if v8 else None)}: {v28(v7)}")
    v7 = v29((v5['value'] in v5['ctx'].v95() for v5 in v6))
    v1 &= v28(v7)
    v27(f'2 every value is a token of its own line (context is the sentence): {v28(v7)}')
    v30, v31, v32 = v67.v26(['x q y', 'x w y'], frame_max=6, min_fillers=2)
    v7 = v69({v5['address'] for v5 in v30}) == 1
    v1 &= v28(v7)
    v27(f'  a frame that cannot widen stays at width 1: {v28(v7)}')
    v45.v33, v45.v34, v45.v35 = (False, 0, False)
    v45.v36, v45.v37, v45.v38, v45.v39 = (True, True, True, 2)
    v45.v40, v45.v41, v45.v42 = ('anchor', 'uniform', 4)
    v45.v43, v45.v44 = (3, 0.05)
    v45.v9 = v46(v45.v47)
    v48, v49 = (v70(), v71())
    v10 = v50((v72 for v72 in v45.v96(v48) if v72['S'] == 'kostya'))
    v11 = v73.v51(0)
    v12 = None
    for v13 in v52(50):
        v12 = v45.v74(v48, v10, v11, 2, v75(v48['tape'].v85))
        if v12 is not None:
            break
    if v12 is None:
        v27('\nROUTE FAILED (no question on the toy tape)')
        return 1
    v66.v53(0)
    v14 = v45.v54(v2, d=8)
    v15 = v45.v55(v48, v12, v75(v12['cands']))
    v56, v57 = v45.v58(v14, v48, v12, v2, v49)
    v7 = v69(v56) == v69(v12['cands']) + 1 and v69(v57) == v69(v12['cands'])
    v1 &= v7
    v27(f'\n3 stage-1 logits {v69(v56)} (candidates + expand), stage-2 {v69(v57)}: {v7}')
    v16 = v59(v56[:-1].v86() - v56[:-1].v87())
    v7 = v16 < 1e-06
    v1 &= v7
    v27(f'  without imports every candidate world is the same graph (spread {v16:.2e}): {v7} - this is why the step exists')
    v17 = []
    for v18 in v12['cands']:
        if v18 == v45.v76:
            continue
        v77, v78, v79 = v45.v80(v48, v12, v49, v2, query_value=v18, import_k=v15)
        v17.v68(v79.v88())
    v7 = v60((v59((v17[v89] - v17[v90]).v82().v91()) > 1e-06 for v89 in v52(v69(v17)) for v90 in v52(v89 + 1, v69(v17))))
    v1 &= v7
    v27(f'4 with imports the completed worlds differ as graphs: {v7}')
    v19 = v45.v61(v12, v2)
    v62, v63 = (v66.v81(v56, 0), v66.v81(v57, 0))
    v20 = -((v62[:-1] * v19).v91() + v62[-1] * ((v63 * v19).v91() - v45.v44))
    v21 = v45.v64(v14, v48, v12, v2, v49)
    v7 = v82(v59(v21) - v59(v20)) < 1e-06
    v1 &= v7
    v27(f'5 loss == closed-form expected payoff of the route: {v7} ({v59(v21):+.6f} vs {v59(v20):+.6f})')
    v7 = v59(v19[v12['label']]) == 1.0 and v59(v19.v91()) < v59(v69(v19))
    v1 &= v7
    v27(f"  reward vector {[v98(v59(v99), 2) for v99 in v19]} (label {v12['label']}, answerable {v12['answerable']}): {v7}")
    v27(f'  a step costs {v45.v44}, so always expanding pays that on every question')
    v27('\nROUTE OK' if v1 else '\nROUTE FAILED')
    return 0 if v1 else 1
if v22 == '__main__':
    raise v65(v83())