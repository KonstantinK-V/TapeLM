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
SRC = Path('_audit398_scope.py')
DESIGNED = '\ndef outer(alpha):\n    omega = omega * alpha\n    return omega\n\n\ndef other(alpha, omega):\n    return omega + alpha\n\n\nclass Thing:\n    def method(self, beta):\n        def inner(gamma):\n            zeta = gamma + beta\n            return zeta\n        return inner(beta)\n'

def props(src=None):
    src = SRC.read_text(encoding='utf-8') if src is None else src
    f = []
    sc = A.scopes_of(DESIGNED)
    if sc is None or len(sc['funcs']) != 4:
        return [f"0. the designed source gives {sc and len(sc['funcs'])} scopes, not 4"]
    by_name = {n.name: i for i, (n, _a, _b) in enumerate(sc['funcs'])}
    if set(by_name) != {'outer', 'other', 'method', 'inner'}:
        return [f'0. the scopes are {sorted(by_name)}']
    o = by_name['outer']
    ln = next((l for l, u in sc['used'].items() if 'omega' in u and 'alpha' in u))
    b_wo = A.bound_wo(sc, o, ln)
    b_with = {n for n in sc['binds'][o]}
    if 'omega' in b_wo:
        f.append('1. the pooled line is still binding `omega` in its own scope - the attachment recognises a name only that line binds (section 27)')
    if 'omega' not in b_with:
        f.append('1. `omega` is not bound in outer at all - the designed case is not designed')
    if 'alpha' not in b_wo:
        f.append('1. the argument stopped being bound when a body line was removed')
    counts = A.bind_counts(sc, ln)
    if counts.get('alpha') != 2:
        f.append(f"2. `alpha` is bound in {counts.get('alpha')} scopes, expected 2 (outer, other)")
    s_norm = A.score({'alpha', 'omega'}, A.bound_wo(sc, by_name['other'], -1), counts, True)
    s_raw = A.score({'alpha', 'omega'}, A.bound_wo(sc, by_name['other'], -1), counts, False)
    if not (abs(s_norm - 1.5) < 1e-09 and abs(s_raw - 2.0) < 1e-09):
        f.append(f'2. the score is {s_norm} normalised / {s_raw} raw, expected 1.5 and 2.0 - a name bound by two scopes must count half')
    zl = next((l for l, u in sc['used'].items() if 'gamma' in u))
    if sc['owner'][zl] != by_name['inner']:
        f.append("3. a line of the nested def is owned by the enclosing scope - a closure's body would be evidence for its parent")
    if 'inner' not in sc['binds'][by_name['method']]:
        f.append("3. the nested def's NAME is not bound in the scope its def line sits in")
    if 'zeta' in sc['binds'][by_name['method']]:
        f.append("3. the nested def's local is bound in the enclosing scope")
    for nm in ('outer', 'other', 'method', 'inner'):
        if sc['funcs'][by_name[nm]][1] not in sc['heads']:
            f.append(f"4. {nm}'s header line is not marked as a header, so it can be pooled")
    if 'self' not in sc['binds'][by_name['method']] or 'beta' not in sc['binds'][by_name['method']]:
        f.append("5. a method's arguments (self included) are not bound")
    if 'void = rep["amb_live"] <= 0.05' not in src:
        f.append('6. the void check is not read on the LIVE ambiguous share - a tie at zero is no signal, and counting it would report a decision population that is empty')
    if 'c["amb_live"] += int(ties > 1 and top > 0.0)' not in src:
        f.append('6. the live ambiguous share is not computed as `ties > 1 and top > 0`')
    gate_c = re.search('gate_c = \\(([^)]|\\n)*?\\)\\n', src)
    if gate_c and 'free_argmax' in gate_c.group(0):
        f.append('8. free_argmax is inside the gate - it is the quantity the count MAXIMISES, so gating on it is gating on the objective')
    if 'free(bounds[true_i])' not in src or '/ len(names)' not in src:
        f.append("8. free names are not a share of the line's own used names")
    if 'fidx = fidx[:nf]' not in src:
        f.append('7. the foreign null is not cut to the same number of scopes - a stranger offered fewer loses by arithmetic (347)')
    return f
MUTANTS = (('the pooled line keeps binding its own names', '    return {n for n, lns in sc["binds"][i].items() if lns - {drop_line}}', '    return set(sc["binds"][i])', '1.'), ('the count is a raw sum', '        return sum(1.0 / max(1, counts.get(n, 1)) for n in names if n in bound)', '        return sum(1.0 for n in names if n in bound)', '2.'), ('a nested body belongs to its parent', '    funcs.sort(key=lambda f: (f[2] - f[1]))', '    funcs.sort(key=lambda f: -(f[2] - f[1]))', '3.'), ('headers are pooled like body lines', '    heads = {a for _n, a, _b in funcs}', '    heads = set()', '4.'), ('arguments are not bound', '            if x is not None:\n                binds[i][x.arg].add(a)', '            if x is None:\n                binds[i][x.arg].add(a)', '5.'), ('the void check counts ties at zero', 'void = rep["amb_live"] <= 0.05', 'void = rep["ambiguous"] <= 0.05', '6.'), ('the foreign null gets fewer scopes', '        fidx = fidx[:nf]', '        fidx = fidx[:1]', '7.'), ('the gate reads the objective the count maximises', '    gate_c = (rep["free_true"] < rep["free_random"] - 0.05', '    gate_c = (rep["free_argmax"] < rep["free_random"] - 0.05', '8.'))

def main() -> int:
    src = SRC.read_text(encoding='utf-8')
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f'MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times')
            continue
        saved = dict(A.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, '<mutant>', 'exec'), A.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f'{tag} the mutant raised {type(e).__name__}']
        finally:
            A.__dict__.clear()
            A.__dict__.update(saved)
        if not any((g.startswith(tag) for g in got)):
            fails.append(f'MUTATION {tag} ({name}): the failure was re-introduced and check {tag} did not fire - it is a comment, not a check')
    for x in fails:
        print('FAIL ' + x)
    print(f'{len(fails)} failures' if fails else f'all properties hold, and all {len(MUTANTS)} re-introduced failures were caught')
    return 1 if fails else 0
if __name__ == '__main__':
    raise SystemExit(main())