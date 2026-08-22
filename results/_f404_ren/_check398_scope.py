"""Check of 398's scope ceiling. No torch, no corpus - the corpus is a designed source string.

398 measures a relation this project has never had on its tape: DEFINITION -> USE, directed and
bound by name identity. The ways it can print a plausible wrong number are specific:

  1. THE SECTION 27 LEAK. What a scope binds must be computed WITH THE POOLED LINE REMOVED. A
     line `omega = omega * alpha` binds `omega`; leaving it in lets the true scope bind a name
     that only that line binds, and the attachment recognises its own answer.
  2. THE COUNT IS NORMALISED. `self`, `q`, `p` are bound by nearly every scope; a raw sum ranks
     by idiom. 317, 383 and 387 each found that fault after the fact - here it is divided from
     the start, with the raw sum available only as a declared control reading.
  3. INNERMOST OWNERSHIP. A line inside a nested def belongs to the NESTED def; only the nested
     def's NAME is bound in the enclosing one. Otherwise a closure's body would be evidence for
     its parent.
  4. A FUNCTION'S HEADER IS NOT A BODY LINE - attaching a def line to its own scope is
     degenerate.
  5. ARGUMENTS ARE BOUND, `self` included, or every method looks like it binds nothing.
  6. A TIE AT ZERO IS NOT A DECISION. Lines where no scope scores anything tie trivially; the
     void check must be read on the LIVE ambiguous share, or a no-signal population would be
     reported as the population where a mind is needed.
  7. THE FOREIGN NULL IS MATCHED IN CARDINALITY (347). A stranger offered fewer scopes loses by
     arithmetic.
  8. FREE NAMES ARE A SHARE OF THE LINE'S OWN USED NAMES, and `free_argmax` is THE OBJECTIVE THE
     COUNT MAXIMISES - it may be printed and must not appear in a gate.

Every property is a number on the designed source, and each has its own failure re-introduced.

    python _check398_scope.py
"""
from __future__ import annotations
import random
import re
from argparse import Namespace
from pathlib import Path
import _audit398_scope as A
v0 = v4('_audit398_scope.py')
v1 = '\ndef outer(alpha):\n    omega = omega * alpha\n    return omega\n\n\ndef other(alpha, omega):\n    return omega + alpha\n\n\nclass Thing:\n    def method(self, beta):\n        def inner(gamma):\n            zeta = gamma + beta\n            return zeta\n        return inner(beta)\n'

def props(v5=None):
    v5 = v0.v31(encoding='utf-8') if v5 is None else v5
    v6 = []
    v7 = v40.v22(v1)
    if v7 is None or v52(v7['funcs']) != 4:
        return [f"0. the designed source gives {v7 and v52(v7['funcs'])} scopes, not 4"]
    v8 = {v27.v23: v24 for v24, (v27, v60, v61) in v53(v7['funcs'])}
    if v41(v8) != {'outer', 'other', 'method', 'inner'}:
        return [f'0. the scopes are {v66(v8)}']
    v9 = v8['outer']
    v10 = v25((v42 for v42, v62 in v7['used'].v63() if 'omega' in v62 and 'alpha' in v62))
    v11 = v40.v26(v7, v9, v10)
    v12 = {v27 for v27 in v7['binds'][v9]}
    if 'omega' in v11:
        v6.v43('1. the pooled line is still binding `omega` in its own scope - the attachment recognises a name only that line binds (section 27)')
    if 'omega' not in v12:
        v6.v43('1. `omega` is not bound in outer at all - the designed case is not designed')
    if 'alpha' not in v11:
        v6.v43('1. the argument stopped being bound when a body line was removed')
    v13 = v40.v28(v7, v10)
    if v13.v44('alpha') != 2:
        v6.v43(f"2. `alpha` is bound in {v13.v44('alpha')} scopes, expected 2 (outer, other)")
    v14 = v40.v29({'alpha', 'omega'}, v40.v26(v7, v8['other'], -1), v13, True)
    v15 = v40.v29({'alpha', 'omega'}, v40.v26(v7, v8['other'], -1), v13, False)
    if not (v64(v14 - 1.5) < 1e-09 and v64(v15 - 2.0) < 1e-09):
        v6.v43(f'2. the score is {v14} normalised / {v15} raw, expected 1.5 and 2.0 - a name bound by two scopes must count half')
    v16 = v25((v42 for v42, v62 in v7['used'].v63() if 'gamma' in v62))
    if v7['owner'][v16] != v8['inner']:
        v6.v43("3. a line of the nested def is owned by the enclosing scope - a closure's body would be evidence for its parent")
    if 'inner' not in v7['binds'][v8['method']]:
        v6.v43("3. the nested def's NAME is not bound in the scope its def line sits in")
    if 'zeta' in v7['binds'][v8['method']]:
        v6.v43("3. the nested def's local is bound in the enclosing scope")
    for v17 in ('outer', 'other', 'method', 'inner'):
        if v7['funcs'][v8[v17]][1] not in v7['heads']:
            v6.v43(f"4. {v17}'s header line is not marked as a header, so it can be pooled")
    if 'self' not in v7['binds'][v8['method']] or 'beta' not in v7['binds'][v8['method']]:
        v6.v43("5. a method's arguments (self included) are not bound")
    if 'void = rep["amb_live"] <= 0.05' not in v5:
        v6.v43('6. the void check is not read on the LIVE ambiguous share - a tie at zero is no signal, and counting it would report a decision population that is empty')
    if 'c["amb_live"] += int(ties > 1 and top > 0.0)' not in v5:
        v6.v43('6. the live ambiguous share is not computed as `ties > 1 and top > 0`')
    v18 = v45.v30('gate_c = \\(([^)]|\\n)*?\\)\\n', v5)
    if v18 and 'free_argmax' in v18.v54(0):
        v6.v43('8. free_argmax is inside the gate - it is the quantity the count MAXIMISES, so gating on it is gating on the objective')
    if 'free(bounds[true_i])' not in v5 or '/ len(names)' not in v5:
        v6.v43("8. free names are not a share of the line's own used names")
    if 'fidx = fidx[:nf]' not in v5:
        v6.v43('7. the foreign null is not cut to the same number of scopes - a stranger offered fewer loses by arithmetic (347)')
    return v6
v2 = (('the pooled line keeps binding its own names', '    return {n for n, lns in sc["binds"][i].items() if lns - {drop_line}}', '    return set(sc["binds"][i])', '1.'), ('the count is a raw sum', '        return sum(1.0 / max(1, counts.get(n, 1)) for n in names if n in bound)', '        return sum(1.0 for n in names if n in bound)', '2.'), ('a nested body belongs to its parent', '    funcs.sort(key=lambda f: (f[2] - f[1]))', '    funcs.sort(key=lambda f: -(f[2] - f[1]))', '3.'), ('headers are pooled like body lines', '    heads = {a for _n, a, _b in funcs}', '    heads = set()', '4.'), ('arguments are not bound', '            if x is not None:\n                binds[i][x.arg].add(a)', '            if x is None:\n                binds[i][x.arg].add(a)', '5.'), ('the void check counts ties at zero', 'void = rep["amb_live"] <= 0.05', 'void = rep["ambiguous"] <= 0.05', '6.'), ('the foreign null gets fewer scopes', '        fidx = fidx[:nf]', '        fidx = fidx[:1]', '7.'), ('the gate reads the objective the count maximises', '    gate_c = (rep["free_true"] < rep["free_random"] - 0.05', '    gate_c = (rep["free_argmax"] < rep["free_random"] - 0.05', '8.'))

def main() -> v3:
    v5 = v0.v31(encoding='utf-8')
    v19 = v32()
    for v23, v33, v34, v35 in v2:
        if v5.v55(v33) != 1:
            v19.v43(f'MUTATION {v35} ({v23}): its anchor occurs {v5.v55(v33)} times')
            continue
        v36 = v46(v40.v47)
        v37 = v5.v48(v33, v34, 1)
        try:
            v56(v65(v37, '<mutant>', 'exec'), v40.v47)
            v49 = v32(src=v37)
        except v50 as e:
            v49 = [f'{v35} the mutant raised {v69(v70).v21}']
        finally:
            v40.v47.v57()
            v40.v47.v58(v36)
        if not v59((v68.v67(v35) for v68 in v49)):
            v19.v43(f'MUTATION {v35} ({v23}): the failure was re-introduced and check {v35} did not fire - it is a comment, not a check')
    for v20 in v19:
        v38('FAIL ' + v20)
    v38(f'{v52(v19)} failures' if v19 else f'all properties hold, and all {v52(v2)} re-introduced failures were caught')
    return 1 if v19 else 0
if v21 == '__main__':
    raise v39(v51())