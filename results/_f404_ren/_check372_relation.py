"""372b: relation as evidence on the candidate. Static, and it RUNS.

Seven ways this could be silently the wrong 372 again: the offer could change; the walk
could leave cos; an unwitnessed candidate could become an empty world (291's size
marker); the own place could witness itself; retention could be skipped; a threshold
could sneak back in; argparse could not accept `relation` (335's dead knob).
"""
from __future__ import annotations
import ast
v0 = '_stage289_derivation.py'

def static():
    v1 = v19.v11(v26(v0, encoding='utf-8').v20())
    v2 = v19.v15(v1).v12('"', "'")
    v3 = {v14.v13: v14 for v14 in v19.v22(v1) if v23(v14, v19.v24)}
    v4 = v19.v15(v3['reach_relation_rows'])
    v5 = v19.v15(v3['reach_rows_for'])
    v6 = v19.v15(v3['reach_candidates'])
    v7 = v19.v15(v3['reach_places'])
    v8 = [('walk still returns on cos before any share scoring', "if REACH_COMPASS == 'cos':" in v7 and v7.v29("if REACH_COMPASS == 'cos':") < v7.v29('ownc = Counter')), ('reach_candidates does not branch on import=relation - the offer is identical', "REACH_IMPORT == 'relation'" not in v6 and 'if REACH_IMPORT' not in v6), ('unwitnessed candidate falls back to walk rows, not empty', 'return rel if rel else rows' in v5), ("the question's own place is excluded", 'if j != i and (kp is None' in v4 or 'j != i' in v4), ('retention honoured, as reach_places honours it', 'retain_keep(p)' in v4 and 'bool(kp[j])' in v4), ('overlap is a count, no threshold', 'overlap[j] += 1' in v4 and '>= 2' not in v4), ('argparse accepts relation', "'relation'" in v2 and "add_argument('--reach-import'" in v2)]
    v9 = True
    for v13, v16 in v8:
        v17(f"  {('OK  ' if v16 else 'FAIL')}  {v13}")
        v9 &= v21(v16)
    v17(f"\n{('372b OK  7/7' if v9 else '372b FAILED')}  ({v27((1 for v31, v30 in v8 if v30))}/{v28(v8)})")
    return v9
if v10 == '__main__':
    raise v18(0 if v25() else 1)