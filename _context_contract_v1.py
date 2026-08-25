"""Executable SOTE context contract v1.

The policy sees only name-free summaries and emits a constraint from ACTIONS.
The tape applies that constraint to all exact cards of CURRENT.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path

from _audit628_ctxswap import address_kernel, unique_best

LAW = (
    Path("_CONTEXT_CONTRACT_V1.txt").read_text(encoding="utf-8")
    if Path("_CONTEXT_CONTRACT_V1.txt").exists()
    else ""
)

ACTIONS = (
    "REFUSE",
    "QUERY",
    "VIA_SUM",
    "VIA_MAX",
    "QUERY_VIA",
    "QUERY_AND_VIA",
)
C_READ = 0.05


def _query_score(pg, row, query_pi, current, high_set, co, df, n_use):
    return address_kernel(
        pg, row["pi"], query_pi, current, high_set, co, df, n_use,
    )


def _via_scores(pg, row, current, via_pis, high_set, co, df, n_use):
    return [
        address_kernel(
            pg, row["pi"], hpi, current, high_set, co, df, n_use,
        )
        for hpi in via_pis
    ]


def constraint_scores(
    pg, rows, query_pi, current, via_pis, high_set, co, df, n_use,
):
    """Held-blind score of every exact CURRENT card under each constraint."""
    out = {action: [] for action in ACTIONS}
    for row in rows:
        query = _query_score(
            pg, row, query_pi, current, high_set, co, df, n_use,
        )
        vias = _via_scores(
            pg, row, current, via_pis, high_set, co, df, n_use,
        )
        via_sum = sum(vias) / math.sqrt(len(vias)) if vias else 0.0
        via_max = max(vias, default=0.0)
        out["QUERY"].append((query, row))
        out["VIA_SUM"].append((via_sum, row))
        out["VIA_MAX"].append((via_max, row))
        out["QUERY_VIA"].append((via_sum * (1.0 + query), row))
        out["QUERY_AND_VIA"].append((query * via_sum, row))
    return out


def resolve_constraints(
    pg, rows, query_pi, current, via_pis, high_set, co, df, n_use,
):
    """Constraint -> one exact address, or None for REFUSE/tie/no evidence."""
    scores = constraint_scores(
        pg, rows, query_pi, current, via_pis, high_set, co, df, n_use,
    )
    resolved = {"REFUSE": None}
    for action in ACTIONS[1:]:
        resolved[action] = unique_best(scores[action])
    return resolved, scores


def _score_stats(scored, chosen):
    vals = [score for score, _row in scored]
    positives = [score for score in vals if score > 0.0]
    order = sorted(vals, reverse=True)
    top = order[0] if order else 0.0
    second = order[1] if len(order) > 1 else 0.0
    margin = max(top - second, 0.0)
    n = max(len(vals), 1)
    mean = statistics.fmean(vals) if vals else 0.0
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    if chosen is None:
        maj = rows_n = keys_n = vote = 0.0
    else:
        maj, neg_rows, neg_keys = chosen["count_key"]
        rows_n = -neg_rows
        keys_n = -neg_keys
        tok_counts = Counter(row["tok"] for _score, row in scored)
        vote = tok_counts[chosen["tok"]] / n
    return [
        float(chosen is not None),
        math.tanh(top),
        math.tanh(margin),
        len(positives) / n,
        math.tanh(mean),
        math.tanh(std),
        float(maj),
        math.log1p(rows_n) / 4.0,
        math.log1p(keys_n) / 3.0,
        vote,
    ]


def state_features(
    pg, rows, query_pi, current, via_pis, high_set, co, df, n_use,
):
    """Fixed-width, name-free policy input plus tape-resolved actions."""
    resolved, scores = resolve_constraints(
        pg, rows, query_pi, current, via_pis, high_set, co, df, n_use,
    )
    current_degree = len(pg["by_place"].get(current, ()))
    query_via = [
        address_kernel(
            pg, query_pi, hpi, current, high_set, co, df, n_use,
        )
        for hpi in via_pis
    ]
    feats = [
        math.log1p(len(rows)) / 4.0,
        math.log1p(len(via_pis)) / 3.0,
        math.log1p(current_degree) / 5.0,
        math.tanh(max(query_via, default=0.0)),
        math.tanh(
            sum(query_via) / math.sqrt(len(query_via))
            if query_via else 0.0
        ),
    ]
    for action in ACTIONS[1:]:
        feats.extend(_score_stats(scores[action], resolved[action]))
    chosen = {
        row["pi"]
        for action, row in resolved.items()
        if action != "REFUSE" and row is not None
    }
    feats.append(len(chosen) / max(len(ACTIONS) - 1, 1))
    return feats, resolved


def full_feedback(resolved, held):
    """Tape teacher: reward every constraint, never expose held to policy."""
    hits, reads, rewards = [], [], []
    for action in ACTIONS:
        row = resolved[action]
        read = int(row is not None)
        hit = int(read and row["tok"] == held)
        reward = (1.0 if hit else -C_READ) if read else 0.0
        hits.append(hit)
        reads.append(read)
        rewards.append(reward)
    return hits, reads, rewards
