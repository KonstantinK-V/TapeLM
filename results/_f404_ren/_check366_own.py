"""366: own worlds get the same import as the step. Eight static properties.

The claim is equal size, so the check is that the own branch actually builds
`world(v, walk_rows, budget)` and that shortfalls are counted rather than asserted.
Default stays OFF; `--connect` wiring is untouched.
"""
from __future__ import annotations
import ast
v0 = '_stage289_derivation.py'

def static():
    v1 = v23(v0, encoding='utf-8').v11()
    v2 = v21.v12(v1)
    v3 = v21.v13(v2)
    v4 = v3.v14('"', "'")
    v5 = {v16.v15: v16 for v16 in v21.v24(v2) if v25(v16, v21.v26)}
    v6 = v21.v13(v5['reach_logits'])
    v7 = [('flag exists and defaults OFF', "add_argument('--own-import', action='store_true'" in v4 and 'OWN_IMPORT = False' in v1), ('assigned from args', 'OWN_IMPORT = args.own_import' in v3), ('written into the report', "'own_import': bool(OWN_IMPORT)" in v4 and "'own_import_full'" in v4), ("own branch pads with the walk's rows, same budget", 'g1 = [world(v, walk_rows, budget) for v in own]' in v6 and 'outside_mentions' not in v6), ('default path preserved: own worlds still empty', 'g1 = [world(v, [], 0) for v in own]' in v6), ('shortfalls counted, not asserted', '_OWN_IMPORT_N[0] += 1' in v6 and 'len(walk_rows) >= budget' in v6), ('CONNECT wiring unchanged', 'CONNECT, CONNECT_MAX = (args.connect, args.connect_max)' in v3 and "'connect': bool(CONNECT)" in v4), ('_check365 still has a channel to read', 'def reach_connect' in v1 and 'def reach_candidates' in v1)]
    v8 = True
    for v15, v17 in v7:
        v18(f"  {('OK  ' if v17 else 'FAIL')}  {v15}")
        v8 &= v22(v17)
    return v8
if v9 == '__main__':
    v18('STATIC')
    v10 = v19()
    v18()
    v18('366 OK' if v10 else '366 BROKEN')
    raise v20(0 if v10 else 1)