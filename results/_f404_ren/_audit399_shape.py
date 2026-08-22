"""THE CEILING OF SHAPE, ON THE 23% WHERE NAMES TIE. Torch-free, before any Phi.

398 measured the object and gave the law for what comes next: on this corpus the counting rival
puts the true scope at the top 96% of the time and picks it uniquely 61% - so a mechanism trained
on the whole file would be that rival with parameters, which is 359's trap entered from the other
side. THE POPULATION IS ONLY `amb_live`: lines where two or more scopes TIE at a non-zero name
score. There, by construction, name overlap cannot decide.

SO THE EVIDENCE MAY NOT BE NAME OVERLAP, and one alternative is declared here BEFORE the run:
SHAPE. The multiset of AST NODE TYPES on the line, against the node types of the candidate's body.
No identifier is read - only `type(node).__name__` - so the feature is equivariant to renaming by
construction, not by an argument. `_check399_shape.py` asserts that no source identifier can reach
it.

    score_shape(scope) = sum over the node TYPES on the line of
                         [type occurs in that scope's body] / (how many scopes of this file have it)

which is the same form as 398's name count with types in place of names, so the two are comparable
and neither is given a shape the other lacks. The undivided version is the declared control
reading, exactly as `--raw` was there.

THE LEAK, INHERITED AND EXTENDED. 398's `bound_wo` already removes the pooled line from the scope
it came from; here the same removal is applied to the scope's TYPES, because a line's own node
types would otherwise be evidence for its own scope. Both are mutation-tested.

THE RIVALS, ALL THREE DECLARED BEFORE THE RUN. Every one is scored by EXPECTED accuracy under
uniform tie-breaking - `1/|argmax| if the truth is in the argmax else 0` - so no rival is helped
or hurt by a coin, and the numbers do not move with the seed.

    random     1/|tied|                     the floor: what a coin gets
    raw name   the UNDIVIDED name overlap   the nearest thing to a name-based tie-break, and the
                                            guard against "shape is names in disguise"
    size       the larger candidate body    THE CONFOUND: a bigger scope contains more node types
                                            for the same reason it contains more of everything

  VOID CHECK, READ FIRST
      `true_in_tie` - the share of ambiguous lines whose true scope is even among the tied. What
      is below that is not a decision anyone can win, and the ceiling would be measured on a
      population that has no answer in it.

  GATE
      shape - random > 0.05, AND shape - raw_name > 0.05, AND shape - size > 0.05.
      PAIRED CONTROL on the FULL population: name-argmax where it is unique, shape where it ties,
      must not fall below the pure name accuracy. The catalogue is not to be paid for the 23%.

  If the gate fails, Phi is not built here: there is no evidence on that population that is not
  the count, and 398's ceiling stands as "a counter of scopes", which is knowledge and not a mind.

    python _audit399_shape.py
    python _audit399_shape.py --raw          # the declared control reading
"""
from __future__ import annotations
import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _audit398_scope as S
v0 = v2('results/_stage399_shape.json')

def types_of(v3, v4):
    """lineno -> the set of AST node TYPE NAMES on it, and scope -> the types of its body lines.

    Only `type(node).__name__` is read. No identifier, no attribute string, no constant value
    reaches this, so renaming every symbol in the file cannot move a single number.
    """
    try:
        v27 = v58.v57(v3)
    except v28:
        return (None, None)
    v5 = v29(v30)
    for v6 in v58.v31(v27):
        v32 = v59(v6, 'lineno', None)
        if v32 is not None:
            v5[v32].v88(v106(v6).v26)
    v7 = v29(lambda: v29(v30))
    for v32, v33 in v5.v34():
        v8 = v4.v60(v32)
        if v8 is None:
            continue
        for v35 in v33:
            v7[v8][v35].v88(v32)
    return (v5, v7)

def types_wo(v7, v8, v9):
    """the types of scope i's body with the pooled line taken out - 398's discipline, on shape"""
    return {v35 for v35, v89 in v7.v60(v8, {}).v34() if v89 - {v9}}

def expected_acc(v10, v11, v12):
    """EXPECTED accuracy under uniform tie-breaking: exact, seed-free, and it prices a rival that
    ties at the top honestly instead of letting a coin decide the number."""
    if not v12:
        return 0.0
    v13 = v36((v10[v18] for v18 in v12))
    v14 = [v18 for v18 in v12 if v10[v18] == v13]
    return 1.0 / v64(v14) if v11 in v14 else 0.0

def run(v15, v16, v17):
    v18 = v37()
    for v19 in v15:
        try:
            v3 = v2(v19).v90(encoding='utf-8', errors='ignore')
        except v61:
            continue
        v38 = v91.v62(v3)
        if not v38 or v64(v38['funcs']) < v16.v92:
            continue
        v5, v7 = v63(v3, v38['owner'])
        if v5 is None:
            continue
        v39 = v64(v38['funcs'])
        v40 = v37(v38['owner'].v93())
        v41 = [v32 for v32 in v48(v38['owner']) if v32 not in v38['heads']]
        v17.v65(v41)
        v42 = 0
        for v32 in v41:
            if v42 >= v16.v94:
                break
            v66 = v38['used'].v60(v32, v30())
            if not v66:
                continue
            v42 += 1
            v67 = v38['owner'][v32]
            v68 = v91.v95(v38, v32)
            v69 = [v91.v100(v38, v8, v32) for v8 in v97(v39)]
            v70 = [v91.v101(v66, v69[v8], v68, True) for v8 in v97(v39)]
            v71 = [v91.v101(v66, v69[v8], v68, False) for v8 in v97(v39)]
            v13 = v36(v70)
            v72 = [v8 for v8 in v97(v39) if v70[v8] == v13]
            v18['full_n'] += 1
            v18['full_name'] += v96(v70, v67, v102(v97(v39)))
            if v64(v72) < 2 or v13 <= 0.0:
                v18['full_combined'] += v96(v70, v67, v102(v97(v39)))
                continue
            v18['amb'] += 1
            v18['true_in_tie'] += v1(v67 in v72)
            v73 = {v8: v103(v7, v8, v32) for v8 in v97(v39)}
            v74 = v37()
            for v8 in v97(v39):
                for v35 in v73[v8]:
                    v74[v35] += 1
            v75 = v5.v60(v32, v30())
            v76 = [v104((1.0 / v36(1, v74[v35]) if v16.v82 else 1.0 for v35 in v75 if v35 in v73[v8])) for v8 in v97(v39)]
            v77 = [v105(v40[v8] - (1 if v8 == v67 else 0)) for v8 in v97(v39)]
            v18['full_combined'] += v96(v76, v67, v72)
            if v67 not in v72:
                continue
            v18['n'] += 1
            v18['tied_size'] += v64(v72)
            v18['random'] += 1.0 / v64(v72)
            v18['shape'] += v96(v76, v67, v72)
            v18['rawname'] += v96(v71, v67, v72)
            v18['size'] += v96(v77, v67, v72)
            v18['line_types'] += v64(v75)
    return v18

def main() -> v1:
    v20 = v78.v43()
    v20.v44('--files', default='_*.py')
    v20.v44('--per-file', type=v1, default=40)
    v20.v44('--min-scopes', type=v1, default=8)
    v20.v44('--raw', dest='normalise', action='store_false', help='the declared control reading: the UNDIVIDED type count')
    v20.v44('--seed', type=v1, default=1337)
    v20.v44('--out', default=v98(v0))
    v16 = v20.v45()
    v17 = v79.v46(v16.v47)
    v15 = v48(v2('.').v80(v16.v15))
    v18 = v49(v15, v16, v17)
    v50, v51, v52 = (v36(1, v18['n']), v36(1, v18['amb']), v36(1, v18['full_n']))
    v21 = {'seed': v16.v47, 'normalise': v81(v16.v82), 'files': v64(v15), 'scored': v18['full_n'], 'amb_live': v18['amb'] / v52, 'decided_on': v18['n'], 'true_in_tie': v18['true_in_tie'] / v51, 'tied_size': v18['tied_size'] / v50, 'line_types': v18['line_types'] / v50, 'random': v18['random'] / v50, 'shape': v18['shape'] / v50, 'rawname': v18['rawname'] / v50, 'size': v18['size'] / v50, 'full_name': v18['full_name'] / v52, 'full_combined': v18['full_combined'] / v52}
    v21['shape_minus_random'] = v21['shape'] - v21['random']
    v21['shape_minus_rawname'] = v21['shape'] - v21['rawname']
    v21['shape_minus_size'] = v21['shape'] - v21['size']
    v21['control_delta'] = v21['full_combined'] - v21['full_name']
    v53(f"{v21['files']} files, {v21['scored']} lines scored, amb_live {v21['amb_live']:.4f}, {('normalised' if v16.v82 else 'RAW')} type count")
    v53(f"VOID CHECK  true_in_tie {v21['true_in_tie']:.4f}  <- read first: the share of ambiguous lines whose true scope is even among the tied")
    v53(f"POPULATION  {v21['decided_on']} lines, {v21['tied_size']:.2f} tied scopes each, {v21['line_types']:.2f} node types per line")
    v53(f"TIE-BREAK   shape {v21['shape']:.4f}   random {v21['random']:.4f}   raw name {v21['rawname']:.4f}   size {v21['size']:.4f}")
    v53(f"            shape-random {v21['shape_minus_random']:+.4f}   shape-rawname {v21['shape_minus_rawname']:+.4f}   shape-size {v21['shape_minus_size']:+.4f}")
    v53(f"CONTROL     full population: name {v21['full_name']:.4f}  name+shape {v21['full_combined']:.4f}  delta {v21['control_delta']:+.4f}")
    v22 = v21['shape_minus_random'] > 0.05 and v21['shape_minus_rawname'] > 0.05 and (v21['shape_minus_size'] > 0.05)
    v23 = v21['control_delta'] >= 0.0
    v21['gate'], v21['control_ok'] = (v81(v22), v81(v23))
    if v22 and v23:
        v53('\nSHAPE CARRIES THE TIE. There is evidence on the decision population that is NOT the name count, it beats the coin, the undivided name overlap and the size confound, and the catalogue is not paid for it. A mind built here would have something to learn that counting does not already hold.')
    elif not v22:
        v53('\nSHAPE DOES NOT CARRY THE TIE: ' + ('it does not beat a coin. ' if v21['shape_minus_random'] <= 0.05 else '') + ('the undivided NAME overlap does as well - it is names in disguise. ' if v21['shape_minus_rawname'] <= 0.05 else '') + ('SIZE does as well - a bigger scope holds more node types for the same reason it holds more of everything. ' if v21['shape_minus_size'] <= 0.05 else '') + 'Do not build Phi here on this feature.')
    else:
        v53('\nTHE CATALOGUE WAS PAID FOR IT: the combined rule is below the name rule on the full population, so the 23% was bought with the 96%.')
    v24 = v2(v16.v24)
    v24.v83.v54(parents=True, exist_ok=True)
    v25 = v99.v85(v24.v90()) if v24.v84() else {}
    v25[f"{v16.v47}{('' if v16.v82 else '_raw')}"] = v21
    v24.v55(v99.v86(v25, indent=1))
    v53(f'wrote {v24}')
    return 0
if v26 == '__main__':
    raise v56(v87())