"""Shared retrieval scorers: flat mean, idf-weighted mean, votes, cascade, fusion."""
from __future__ import annotations
from collections import defaultdict
import torch
import torch.nn.functional as F
from _stage194_fp_fact_memory import FpBank
v0 = 0.5
v1 = 128

def ctx_vector(v3: v33, v4: v28[v52], v5: v12[v52, v41] | None=None) -> v37.v9 | None:
    if v53(v4) < 3:
        return None
    v6 = v3.v34(v4)
    if v5 is None:
        return v58.v36(v6.v64(0), dim=-1)
    v7 = v37.v35([v5.v62(v65, 1.0) for v65 in v4], device=v6.v54, dtype=v6.v55)
    v8 = (v6 * v7.v70(1)).v56(0) / v7.v56().v57(1e-06)
    return v58.v36(v8, dim=-1)

def minmax(v10: v12[v2, v41]) -> v12[v2, v41]:
    if not v10:
        return v10
    v11 = v28(v10.v59())
    v38, v39 = (v60(v11), v61(v11))
    if v39 - v38 < 1e-12:
        return {v40: 1.0 for v40 in v10}
    return {v40: (v8 - v38) / (v39 - v38) for v40, v8 in v10.v66()}

def vote_scores(v13: v28[v52], v14: v12[v52, v28[v2]], v5: v12[v52, v41]) -> v12[v2, v41]:
    v10: v12[v2, v41] = v42(v41)
    for v7 in v13:
        for v43 in v14.v62(v7, ()):
            v10[v43] += v5[v7]
    return v12(v10)

def cosine_scores(v15: v37.v9, v16: v37.v9) -> v12[v2, v41]:
    if v15 is None:
        return {}
    v17 = v16 @ v15
    return {v44: v41(v17[v44]) for v44 in v67(v17.v69(0))}

def fusion_scores(v18: v12[v2, v41], v19: v12[v2, v41], v20: v41) -> v12[v2, v41]:
    v45, v46 = (v63(v18), v63(v19))
    v21 = v48(v45) | v48(v46)
    return {v44: v45.v62(v44, 0.0) + v20 * v46.v62(v44, 0.0) for v44 in v21}

def cascade_order(v18: v12[v2, v41], v15: v37.v9, v16: v37.v9, v22: v2, v23: v2) -> v28[v2]:
    """Votes shortlist, cosine rerank within pool; rest follow by vote."""
    if not v18 or v15 is None:
        return v28(v67(v22))
    v24 = v47(v18.v16(), key=lambda v44: -v18[v44])[:v23]
    v25 = v47(v24, key=lambda v44: -v41(v16[v44] @ v15))
    v26 = v48(v25)
    v27 = v47([v44 for v44 in v67(v22) if v44 not in v26], key=lambda v44: -v18.v62(v44, 0.0))
    return v25 + v27

def rank_from_scores(v10: v12[v2, v41], v29: v2, v30: v2 | None=None) -> v2:
    from _tape_index import vote_rank
    v22 = v30 if v30 is not None else v61(v53(v10), v29 + 1, 1)
    v49, v31 = v50(v10, v29, v22)
    return v31

def rank_from_order(v32: v28[v2], v29: v2) -> v2:
    try:
        return v32.v68(v29) + 1
    except v51:
        return v53(v32) + 1