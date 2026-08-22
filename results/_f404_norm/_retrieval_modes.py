"""Shared retrieval scorers: flat mean, idf-weighted mean, votes, cascade, fusion."""
from __future__ import annotations
from collections import defaultdict
import torch
import torch.nn.functional as F
from _stage194_fp_fact_memory import FpBank
FUSION_LAM = 0.5
CASCADE_POOL = 128

def ctx_vector(bank: FpBank, words: list[str], idf: dict[str, float] | None=None) -> torch.Tensor | None:
    if len(words) < 3:
        return None
    fps = bank.fp(words)
    if idf is None:
        return F.normalize(fps.mean(0), dim=-1)
    w = torch.tensor([idf.get(x, 1.0) for x in words], device=fps.device, dtype=fps.dtype)
    v = (fps * w.unsqueeze(1)).sum(0) / w.sum().clamp_min(1e-06)
    return F.normalize(v, dim=-1)

def minmax(sc: dict[int, float]) -> dict[int, float]:
    if not sc:
        return sc
    vals = list(sc.values())
    lo, hi = (min(vals), max(vals))
    if hi - lo < 1e-12:
        return {k: 1.0 for k in sc}
    return {k: (v - lo) / (hi - lo) for k, v in sc.items()}

def vote_scores(qwords: list[str], postings: dict[str, list[int]], idf: dict[str, float]) -> dict[int, float]:
    sc: dict[int, float] = defaultdict(float)
    for w in qwords:
        for cid in postings.get(w, ()):
            sc[cid] += idf[w]
    return dict(sc)

def cosine_scores(q: torch.Tensor, keys: torch.Tensor) -> dict[int, float]:
    if q is None:
        return {}
    sims = keys @ q
    return {j: float(sims[j]) for j in range(sims.size(0))}

def fusion_scores(vote_sc: dict[int, float], cos_sc: dict[int, float], lam: float) -> dict[int, float]:
    vn, cn = (minmax(vote_sc), minmax(cos_sc))
    ids = set(vn) | set(cn)
    return {j: vn.get(j, 0.0) + lam * cn.get(j, 0.0) for j in ids}

def cascade_order(vote_sc: dict[int, float], q: torch.Tensor, keys: torch.Tensor, n: int, pool_size: int) -> list[int]:
    """Votes shortlist, cosine rerank within pool; rest follow by vote."""
    if not vote_sc or q is None:
        return list(range(n))
    by_vote = sorted(vote_sc.keys(), key=lambda j: -vote_sc[j])[:pool_size]
    pool = sorted(by_vote, key=lambda j: -float(keys[j] @ q))
    pool_set = set(pool)
    rest = sorted([j for j in range(n) if j not in pool_set], key=lambda j: -vote_sc.get(j, 0.0))
    return pool + rest

def rank_from_scores(sc: dict[int, float], gold: int, n_slots: int | None=None) -> int:
    from _tape_index import vote_rank
    n = n_slots if n_slots is not None else max(len(sc), gold + 1, 1)
    _, rank = vote_rank(sc, gold, n)
    return rank

def rank_from_order(order: list[int], gold: int) -> int:
    try:
        return order.index(gold) + 1
    except ValueError:
        return len(order) + 1