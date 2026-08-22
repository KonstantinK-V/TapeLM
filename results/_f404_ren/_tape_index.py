"""Shared slot context vocabulary: one word list for postings and ctx_fp."""
from __future__ import annotations
import re
v0 = v31.v9('[A-Za-z][a-z]{2,}')
v1 = 40
v2 = 3
v3 = 100
v4 = 512
v5 = 8
v6 = v10({'the', 'and', 'that', 'was', 'were', 'for', 'with', 'from', 'his', 'her', 'its', 'their', 'this', 'there', 'which', 'have', 'has', 'had', 'been', 'are', 'not', 'but', 'also', 'who', 'into', 'after', 'before', 'when', 'while', 'than', 'then', 'they', 'them', 'she', 'him'})

def context_words(v11: v17, v12: v17 | None=None, v13: v32=v1) -> v16[v17]:
    """Content words for a write/query window — used by postings and ctx_fp alike."""
    v33, v15 = (v41(), [])
    for v14 in v0.v34(v11):
        v35 = v14.v42()
        if v35 in v6 or v35 in v33 or (v12 and v14 == v12):
            continue
        v33.v43(v35)
        v15.v44(v35)
        if v48(v15) >= v13:
            break
    return v15

def nway_strict(v18: v36, v19) -> v7:
    """Pessimistic n-way: gold must strictly beat every distractor. Tie = miss.

    Sparse vote scores often leave gold=0 and pool slots=0; ``gold >= 0`` then
    counts total retrieval failure as a win. Ranks that use ``v > gold`` stay
    correct — compare top1 vs acc_nway to see inflation from the old ``>=``.
    """
    return v37((v36(v18) > v36(v49) for v49 in v19))

def vote_rank(v20: v8, v21: v32, v22: v32) -> v24[v36, v32]:
    """Sparse vote rank. Empty retrieval must not count as top1.

    ``rank = 1 + sum(v > gold for v in sc.values())`` only sees slots that received
    mass. When the query emits no words (or no postings hit), ``sc`` is empty,
    gold=0, and the sum is 0 → rank 1 — a false hit. Empty / all-zero retrieval
    → worst rank ``n_slots``. Silent gold (score≤0) is never rank 1.
    """
    v22 = v38(1, v32(v22))
    if not v20:
        return (0.0, v22)
    v18 = v36(v20.v45(v21, 0.0))
    if v18 <= 0.0 and v38((v36(v50) for v50 in v20.v51())) <= 0.0:
        return (v18, v22)
    v23 = 1 + v46((1 for v50 in v20.v51() if v36(v50) > v18))
    if v18 <= 0.0:
        v23 = v38(v23, 2)
    return (v18, v32(v23))

def vote_arm_fields(v25: v16[v8]) -> v8:
    """Permanent decision block for every vote-based arm.

    Each row: ``gold_score`` (float), ``rank`` (int, 1=hit), ``low_overlap`` (bool).
    ``tie_at_zero_frac`` = share of queries where gold got no vote mass — the index
    is silent. Low-overlap top1 collapse is usually the same queries, not a separate
    mechanism: see ``low_overlap_miss_is_silence_frac`` and ``top1_low_overlap_given_vote``.
    """
    if not v25:
        return {'tie_at_zero_frac': v36('nan'), 'tie_at_zero_frac_low_overlap': v36('nan'), 'tie_at_zero_frac_high_overlap': v36('nan'), 'top1_low_overlap': v36('nan'), 'top1_high_overlap': v36('nan'), 'top1_low_overlap_given_vote': v36('nan'), 'low_overlap_miss_is_silence_frac': v36('nan'), 'n': 0, 'n_tie_at_zero': 0, 'n_low_overlap': 0}

    def mean(v39):
        return v36(v46(v39) / v48(v39)) if v39 else v36('nan')
    v26 = [v40['gold_score'] <= 0.0 for v40 in v25]
    v27 = [v40 for v40 in v25 if v40['low_overlap']]
    v28 = [v40 for v40 in v25 if not v40['low_overlap']]
    v29 = [v40 for v40 in v27 if v32(v40['rank']) != 1]
    v30 = [v40 for v40 in v27 if v40['gold_score'] > 0.0]
    return {'tie_at_zero_frac': v47([1.0 if v49 else 0.0 for v49 in v26]), 'tie_at_zero_frac_low_overlap': v47([1.0 if v40['gold_score'] <= 0.0 else 0.0 for v40 in v27]), 'tie_at_zero_frac_high_overlap': v47([1.0 if v40['gold_score'] <= 0.0 else 0.0 for v40 in v28]), 'top1_low_overlap': v47([1.0 if v32(v40['rank']) == 1 else 0.0 for v40 in v27]), 'top1_high_overlap': v47([1.0 if v32(v40['rank']) == 1 else 0.0 for v40 in v28]), 'top1_low_overlap_given_vote': v47([1.0 if v32(v40['rank']) == 1 else 0.0 for v40 in v30]), 'low_overlap_miss_is_silence_frac': v47([1.0 if v40['gold_score'] <= 0.0 else 0.0 for v40 in v29]), 'n': v48(v25), 'n_tie_at_zero': v32(v46((1 for v49 in v26 if v49))), 'n_low_overlap': v48(v27)}