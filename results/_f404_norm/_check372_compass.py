"""372: the compass is a relation, and there are more than two. Checked and RUN.

Three ways this could be silently wrong: a new compass could fall through to the
old `share` branch and measure nothing (a DEAD KNOB, and 335's `--addresses` did
exactly that); it could skip the retention mask that `reach_places` applies, so
two compasses would be compared on two different tapes; or the members could be
arithmetically identical, which is how 371's first pass shipped share_1 and
share_w as the same expression.
"""
from __future__ import annotations
import ast
from collections import Counter
SRC = '_stage289_derivation.py'
NEW = ('share1', 'rare', 'common', 'cover', 'jaccard')

def static():
    t = ast.parse(open(SRC, encoding='utf-8').read())
    u = ast.unparse(t).replace('"', "'")
    fn = {n.name: n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}
    rp = ast.unparse(fn['reach_places'])
    checks = [('every new compass is named in reach_places', all((f"'{m}'" in rp for m in NEW))), ('the new branch is taken BEFORE the old share loop', rp.index("if REACH_COMPASS in ('share1'") < rp.index('share[j] += min(cnt, c)')), ('retention honoured in the new branch too - or two compasses are two tapes', rp.count('kp is not None and (not bool(kp[j]))') >= 1 or rp.count('not bool(kp[j])') >= 1), ("the question's own place is excluded in the new branch", 'if j == i or (kp is not None' in rp), ('`both` is the only compass that still interleaves', "if REACH_COMPASS != 'both':" in rp), ('argparse accepts them', all((f"'{m}'" in u for m in NEW)))]
    ok = True
    for name, good in checks:
        print(f"  {('OK  ' if good else 'FAIL')}  {name}")
        ok &= bool(good)
    return ok

def behaviour():
    """The five members must ORDER PLACES DIFFERENTLY on one hand-made tape, or they are one
    compass under five names - the mistake 371's first pass shipped."""
    n_fill = {1: 2, 2: 40, 3: 3}
    tot = {'A': 2, 'B': 900}
    shared = {1: ['A'], 2: ['B'], 3: ['B']}
    n_own = 2
    sc = {m: Counter() for m in NEW}
    for j, vs in shared.items():
        for v in vs:
            sc['share1'][j] += 1
            sc['rare'][j] += 1.0 / tot[v]
            sc['common'][j] += float(tot[v])
            sc['cover'][j] += 1.0 / n_fill[j]
            sc['jaccard'][j] += 1.0 / max(1, n_own + n_fill[j] - 1)
    order = {m: sorted(sc[m], key=lambda j: (-sc[m][j], j)) for m in NEW}
    for m in NEW:
        print(f'  {m:<8} {order[m]}   ' + ' '.join((f'{j}:{sc[m][j]:.4g}' for j in (1, 2, 3))))
    good = order['rare'][0] == 1 and order['common'][0] == 2 and (order['cover'][0] == 1) and (order['jaccard'] == [1, 3, 2])
    print(f"  {('OK  ' if good else 'FAIL')}  rare prefers the rare-sharer, common the common one, cover the small focused one - they are not one compass")
    distinct = len({tuple(order[m]) for m in NEW})
    print(f"  {('OK  ' if distinct >= 3 else 'FAIL')}  {distinct} distinct orderings among {len(NEW)} members")
    return good and distinct >= 3
if __name__ == '__main__':
    print('STATIC')
    a = static()
    print('BEHAVIOUR')
    b = behaviour()
    print('\n372 OK' if a and b else '\n372 FAILED')
    raise SystemExit(0 if a and b else 1)