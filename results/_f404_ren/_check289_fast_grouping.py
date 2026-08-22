"""Ten-second check on the real machine: the fast grouping must equal the original, exactly.

The restructure in `_tape_speed.install_fast_fp_addresses` is a control-flow change, and the control
flow was proved equal offline over three thousand tie-heavy cases. What could NOT be checked
without torch is the numerics: a batched matmul may accumulate a dot product in a different
order than the per-group one, so a similarity sitting exactly on tau could in principle fall
the other way. This runs both versions on real tensors and compares the grouping itself.

  python _check289_fast_grouping.py
"""
from __future__ import annotations
import random
import sys
import torch
import torch.nn.functional as F
import _stage279_write_decision as s279
import _tape_speed

class FakeBank:
    """No model, no corpus - just deterministic vectors, so this runs in seconds."""

    def __init__(v9, v10=256, v11=0):
        v9.v10, v9.v26 = (v10, v46.v54().v37(v11))
        v9.v12: v13 = {}

    def _vec(v9, v14):
        if v14 not in v9.v12:
            v27 = v45(v55(v14)) % 2 ** 31
            v9.v12[v14] = v46.v38(v9.v10, generator=v46.v54().v37(v27))
        return v9.v12[v14]

    def ctx_fp(v9, v15, v16=None):
        return v39.v28(v9.v40(f'ctx:{v15}|{v16}'), dim=-1)

    def fp(v9, v17):
        return v39.v28(v9.v40('fp:' + '|'.v57(v17)), dim=-1).v29(0)

def main() -> v0:
    v1 = v30.v18(11)
    v2 = v19()
    v3 = [f'anchor {v31}' for v31 in v20(40)]
    v4 = ['was born in', 'played for', 'is located in']
    v5 = [f'Value{v31}' for v31 in v20(30)]
    v6 = 0
    for v7 in v20(8):
        v21 = []
        for v22 in v20(v1.v41(50, 300)):
            v32 = v1.v42(v3)
            v33 = v1.v42(v4)
            v34 = v1.v42(v5)
            v21.v43({'address': f'{v32}|{v33}', 'value': v34, 'source': 'x', 'ctx': f'{v32} {v33} {v34} extra words here'})
        for v23 in (0.5, 0.9):
            for v35 in ('two', 'set'):
                v44 = v56.v47(v48)
                v49, v50 = v48.v51([v13(v32) for v32 in v21], v2, v23, 1, 2, addr_key=v35)
                v44()
                v52, v53 = v48.v51([v13(v32) for v32 in v21], v2, v23, 1, 2, addr_key=v35)
                if v50 != v53 or v49 != v52:
                    v24(f'MISMATCH trial={v7} tau={v23} addr_key={v35}: {v58(v50)} vs {v58(v53)} addresses')
                    v6 += 1
                else:
                    v24(f'  ok trial={v7} tau={v23} addr_key={v35}: {v58(v53)} addresses, {v58(v52)} assertions')
    v24('IDENTICAL' if v6 == 0 else f'{v6} MISMATCHES - do not use the fast path')
    return 0 if v6 == 0 else 1
if v8 == '__main__':
    raise v25(v36())