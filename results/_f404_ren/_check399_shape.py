"""Check of 399's shape feature. No torch, no corpus - the corpus is two designed sources.

399 exists to answer one question: on the 23% where names tie, is there evidence that is NOT the
name count? So the feature's defining property is not accuracy, it is BLINDNESS TO IDENTITY - and
that is checkable exactly, by renaming every symbol and requiring every number to be unchanged.

  1. EQUIVARIANCE. Two sources identical in structure and different in every identifier must give
     the SAME per-line type sets and the SAME shape scores. If one identifier reaches the
     feature, "shape" is names in disguise and the whole comparison is void.
  2. THE LEAK, ON SHAPE. A scope's types are read with the pooled line REMOVED: a line's own node
     types are otherwise evidence for its own scope, which is 398's `bound_wo` fault in the other
     currency.
  3. EXPECTED ACCURACY, NOT A COIN. A rival that ties at the top scores 1/|argmax|, exactly, so
     no number moves with the seed and a tying rival is priced honestly rather than by a draw.
  4. THE DECISION POPULATION IS ONLY `amb_live`. A line whose name argmax is unique must not
     enter the tie-break numbers at all, or the 23% would be diluted by the 61% counting already
     answers.
  5. THE SIZE RIVAL EXCLUDES THE POOLED LINE from the true scope's body count - conservative, and
     against the rival that turned out to carry the population.
  6. THE CONTROL IS ON THE FULL POPULATION, every scored line, or "the catalogue was not paid
     for" is a claim about the 23% alone.
  7. THE GATE NEEDS ALL THREE RIVALS BEATEN. Beating a coin while size does better is not
     evidence about the line.

    python _check399_shape.py
"""
from __future__ import annotations
import re
from argparse import Namespace
from pathlib import Path
import _audit398_scope as S
import _audit399_shape as A
v0 = v5('_audit399_shape.py')
v1 = '\ndef alpha(one):\n    two = one + 1\n    for three in range(two):\n        two = two + three\n    return two\n\n\ndef beta(four):\n    five = four\n    return five\n'
v2 = v1.v6('alpha', 'zulu').v6('one', 'whisky').v6('two', 'xray').v6('three', 'yankee').v6('beta', 'victor').v6('four', 'uniform').v6('five', 'tango')

def shape_vec(v7, v8):
    v9 = v54.v26(v7)
    v16, v27 = v55.v28(v7, v9['owner'])
    v10 = v8(v9, v16)
    v11 = v29(v9['funcs'])
    v12 = {v14: v55.v56(v27, v14, v10) for v14 in v30(v11)}
    v13 = {}
    for v14 in v30(v11):
        for v31 in v12[v14]:
            v13[v31] = v13.v32(v31, 0) + 1
    v15 = v16.v32(v10, v57())
    return ([v67((1.0 / v79(1, v13[v31]) for v31 in v15 if v31 in v12[v14])) for v14 in v30(v11)], v15, v10)

def first_for(v9, v16):
    return v33((v58 for v58, v72 in v16.v73() if 'For' in v72))

def props(v7=None):
    v7 = v0.v44(encoding='utf-8') if v7 is None else v7
    v17 = []
    v34, v35, v36 = v37(v1, v38)
    v39, v40, v41 = v37(v2, v38)
    if v35 != v40 or v34 != v39 or v36 != v41:
        v17.v59(f'1. renaming every identifier moved the feature: line {v36} vs {v41}, types {v77(v35)} vs {v77(v40)}, scores {v34} vs {v39} - `shape` is reading identity, so it is the name count in disguise')
    if v42((v24 in v74(v77(v35)) for v24 in ('alpha', 'one', 'two', 'three'))):
        v17.v59(f'1. an identifier is inside the type set: {v77(v35)}')
    v9 = v54.v26(v1)
    v16, v27 = v55.v28(v1, v9['owner'])
    v18 = v38(v9, v16)
    v14 = v9['owner'][v18]
    if 'For' not in v55.v56(v27, v14, -1):
        v17.v59('2. the designed case is not designed: the loop line is not in its scope')
    if 'For' in v55.v56(v27, v14, v18):
        v17.v59("2. a scope keeps the node types of the pooled line - the line is evidence for its own scope (398's bound_wo fault, in the other currency)")
    v19 = (v55.v60([1.0, 1.0, 0.0], 0, [0, 1, 2]), v55.v60([1.0, 1.0, 0.0], 2, [0, 1, 2]), v55.v60([2.0, 1.0, 0.0], 0, [0, 1, 2]))
    if v19 != (0.5, 0.0, 1.0):
        v17.v59(f'3. expected accuracy reads {v19}, expected (0.5, 0.0, 1.0) - a rival that ties must be priced at 1/|argmax|, not by a coin')
    if 'if len(tied) < 2 or top <= 0.0:' not in v7 or 'c["amb"] += 1' not in v7:
        v17.v59('4. the decision population is not restricted to a live tie')
    if 'c["n"] += 1' not in v7.v75('if true_i not in tied:')[-1][:200]:
        v17.v59('4. lines whose true scope is not among the tied still enter the numbers')
    if 's_size = [float(n_body[i] - (1 if i == true_i else 0))' not in v7:
        v17.v59('5. the size rival does not take the pooled line out of the true scope, so the confound is given a free point')
    if 'c["full_n"] += 1' not in v7 or 'c["full_name"] +=' not in v7:
        v17.v59('6. the control is not counted on every scored line')
    v20 = v61.v43('gate = \\((?:.|\\n)*?\\)\\n', v7)
    v21 = v20.v62(0) if v20 else ''
    for v22 in ('shape_minus_random', 'shape_minus_rawname', 'shape_minus_size'):
        if v22 not in v21:
            v17.v59(f'7. {v22} is not in the gate - beating a coin while another rival does better is not evidence about the line')
    return v17
v3 = (('the feature reads identifiers', '        if ln is not None:\n            line_t[ln].add(type(node).__name__)', '        if ln is not None:\n            line_t[ln].add(type(node).__name__)\n            line_t[ln].add(getattr(node, "id", ""))', '1.'), ("a scope keeps the pooled line's types", '    return {t for t, lns in body_t.get(i, {}).items() if lns - {drop_line}}', '    return set(body_t.get(i, {}))', '2.'), ('a tying rival is priced by a coin', '    return (1.0 / len(best)) if truth_i in best else 0.0', '    return 1.0 if truth_i in best else 0.0', '3.'), ('unique-argmax lines enter the tie-break', '            if len(tied) < 2 or top <= 0.0:', '            if False:', '4.'), ('the size rival keeps its free point', '            s_size = [float(n_body[i] - (1 if i == true_i else 0)) for i in range(nf)]', '            s_size = [float(n_body[i]) for i in range(nf)]', '5.'), ('the control counts only the ambiguous lines', '            c["full_n"] += 1', '            c["full_n"] += 0', '6.'), ('the gate drops the size rival', '    gate = (rep["shape_minus_random"] > 0.05 and rep["shape_minus_rawname"] > 0.05\n            and rep["shape_minus_size"] > 0.05)', '    gate = (rep["shape_minus_random"] > 0.05 and rep["shape_minus_rawname"] > 0.05)', '7.'))

def main() -> v4:
    v7 = v0.v44(encoding='utf-8')
    v23 = v45()
    for v46, v47, v48, v49 in v3:
        if v7.v68(v47) != 1:
            v23.v59(f'MUTATION {v49} ({v46}): its anchor occurs {v7.v68(v47)} times')
            continue
        v50 = v63(v55.v64)
        v51 = v7.v6(v47, v48, 1)
        try:
            v69(v76(v51, '<mutant>', 'exec'), v55.v64)
            v19 = v45(src=v51)
        except v65 as e:
            v19 = [f'{v49} the mutant raised {v80(v81).v25}']
        finally:
            v55.v64.v70()
            v55.v64.v71(v50)
        if not v42((v20.v78(v49) for v20 in v19)):
            v23.v59(f'MUTATION {v49} ({v46}): the failure was re-introduced and check {v49} did not fire - it is a comment, not a check')
    for v24 in v23:
        v52('FAIL ' + v24)
    v52(f'{v29(v23)} failures' if v23 else f'all properties hold, and all {v29(v3)} re-introduced failures were caught')
    return 1 if v23 else 0
if v25 == '__main__':
    raise v53(v66())