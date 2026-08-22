"""366: own worlds get the same import as the step. Eight static properties.

The claim is equal size, so the check is that the own branch actually builds
`world(v, walk_rows, budget)` and that shortfalls are counted rather than asserted.
Default stays OFF; `--connect` wiring is untouched.
"""
from __future__ import annotations
import ast
SRC = '_stage289_derivation.py'

def static():
    src = open(SRC, encoding='utf-8').read()
    t = ast.parse(src)
    u = ast.unparse(t)
    u1 = u.replace('"', "'")
    fn = {n.name: n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}
    rl = ast.unparse(fn['reach_logits'])
    checks = [('flag exists and defaults OFF', "add_argument('--own-import', action='store_true'" in u1 and 'OWN_IMPORT = False' in src), ('assigned from args', 'OWN_IMPORT = args.own_import' in u), ('written into the report', "'own_import': bool(OWN_IMPORT)" in u1 and "'own_import_full'" in u1), ("own branch pads with the walk's rows, same budget", 'g1 = [world(v, walk_rows, budget) for v in own]' in rl and 'outside_mentions' not in rl), ('default path preserved: own worlds still empty', 'g1 = [world(v, [], 0) for v in own]' in rl), ('shortfalls counted, not asserted', '_OWN_IMPORT_N[0] += 1' in rl and 'len(walk_rows) >= budget' in rl), ('CONNECT wiring unchanged', 'CONNECT, CONNECT_MAX = (args.connect, args.connect_max)' in u and "'connect': bool(CONNECT)" in u1), ('_check365 still has a channel to read', 'def reach_connect' in src and 'def reach_candidates' in src)]
    ok = True
    for name, good in checks:
        print(f"  {('OK  ' if good else 'FAIL')}  {name}")
        ok &= bool(good)
    return ok
if __name__ == '__main__':
    print('STATIC')
    a = static()
    print()
    print('366 OK' if a else '366 BROKEN')
    raise SystemExit(0 if a else 1)