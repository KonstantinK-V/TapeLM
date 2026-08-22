"""191b — fairness probe: curve meaning measured like GPT's (mean-pooled states).

P4 compared GPT mean-pooled hidden vs curve slow ENDPOINT. Re-measure curve P1
with mean-pooled FAST channel (and fast+slow concat) — same pooling as GPT.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
v0 = v1('checkpoints/stage191_p1_curve.pt')

def main():
    v2 = v40.v2('cuda' if v40.v91.v65() else 'cpu')
    v9, v10, v11, v12 = v13()
    v3 = v41.v14(v42(v66.v43))
    v4 = v3.v15()
    v5 = v3.v44(v45) or 0
    v6 = v92.v67(v3, v11, v5, v4).v16(v2)
    v7 = v68(v12, v4).v16(v2)
    v7.v17(v40.v69(v0, map_location=v2, weights_only=False)['model'])
    v7.v18()

    @v40.v25()
    def states(v19):
        v20 = v40.v46([v19[-v98:]], dtype=v40.v70, device=v2)
        v21 = v20 == v5
        v22 = v7.v47(v6[v20], v20)
        v23 = v7.v23(v22, pad_mask=v21)
        v48, v34, v34 = v7.v48(v22, v21)
        v24 = (~v21)[0]
        return (v23[0][v24], v48[0][v24])

    def z_fast(v19):
        v49, v34 = v50(v19)
        return v49.v51(0)

    def z_cat(v19):
        v49, v52 = v50(v19)
        return v40.v53([v49.v51(0), v52.v51(0)])

    def gate_B(v26):

        def zt(v54):
            return v26([v93 for v93 in v3.v102(v54).v99 if v93 != v5])
        v27 = lambda v58, v59: v71(v100.v94(v58, v59, dim=-1))
        v28 = [v27(v95(v58), v95(v59)) for v58, v59 in v96.v72]
        v29 = [v27(v95(v58), v95(v59)) for v58, v59 in v96.v73]
        return {'para': v71(v101.v51(v28)), 'hard': v71(v101.v51(v29)), 'gap': v71(v101.v51(v29) - v101.v51(v28))}

    def doclink(v26, v30=80):
        v31 = v74.v55(7)
        v32 = v75(v10) - 1
        v33 = 0
        for v34 in v56(v30):
            v76, v77 = (v31.v97(0, v32 - 1), v31.v97(0, v32 - 1))
            v78, v79 = (v10[v76], v10[v76 + 1])
            v80, v81 = (v10[v77], v10[v77 + 1])
            if v79 - v78 < v98 + 16 or v81 - v80 < v98:
                continue
            v57 = (v78 + v79) // 2
            v58 = v9[v78:v103(v78 + v98, v57)].v82()
            v59 = v9[v57:v57 + v98].v82()
            v60 = v9[v80:v80 + v98].v82()
            v83, v84, v85 = (v26(v58), v26(v59), v26(v60))
            v33 += v86(v71(v100.v94(v83, v84, dim=-1)) > v71(v100.v94(v83, v85, dim=-1)))
        return v33 / v87(1, v30)
    for v35, v36 in (('fast_meanpool', v61), ('fast+slow_meanpool', v62)):
        v37 = {'gateB': v88(v36), 'doclink': v89(v36)}
        v63(v35, v90.v64(v37))
    v1('results/stage191_p4b_fair.json').v38(v90.v64({v30: {'gateB': v88(v36), 'doclink': v89(v36)} for v30, v36 in (('fast_meanpool', v61), ('fast+slow_meanpool', v62))}, indent=2), encoding='utf-8')
if v8 == '__main__':
    v39()