"""Executable integrated contract: frozen 618, learned REFUSE branch only."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path

from _audit606_bridge import extract_at, place_offer
from _audit628_ctxswap import address_kernel

LAW = (
    Path("_INTEGRATED_CONTRACT_V1.txt").read_text(encoding="utf-8")
    if Path("_INTEGRATED_CONTRACT_V1.txt").exists()
    else ""
)

ACTIONS = ("SEARCH_ONE", "COMMIT_RESOLVED", "REFUSE")
K = 3
CAP = 6
C_SEARCH = 0.05
C_COMMIT = 0.05


def leftover_records(pg, pin, skip, env_m, mid_set, high_set, forbid):
    """Held-blind leftover doors with their exact exposing Petya address."""
    records = []
    seen = set()
    for pi in pg["by_place"].get(pin, ()):
        if pi == skip:
            continue
        _bag, uniq = place_offer(pg["places"][pi], pin, env_m, mid_set)
        for door in uniq:
            if (
                door in seen
                or door in forbid
                or door not in mid_set
                or door in high_set
            ):
                continue
            seen.add(door)
            records.append(dict(door=door, door_support_pi=pi))
    return records


def open_record(
    pg, record, pin, skip, env_m, mid_set, high_set, co, df, n_use,
):
    """One priced door read -> exact hop1 observations appended to W."""
    door = record["door"]
    observations = []
    for pi in pg["by_place"].get(door, ()):
        if pi == skip:
            continue
        place = pg["places"][pi]
        tok, _bag, uniq = extract_at(
            place, door, env_m, mid_set, co, df, n_use,
        )
        if not uniq or tok is None:
            continue
        if tok in {pin, door} or tok not in mid_set or tok in high_set:
            continue
        observations.append(dict(
            tok=tok,
            hop_pi=pi,
            door_support_pi=record["door_support_pi"],
            majority=place["majority"],
            count_key=place["count_key"],
        ))
        if len(observations) >= K:
            break
    return dict(
        door_support_pi=record["door_support_pi"],
        observations=observations,
    )


def commit_resolved(opened):
    """Frozen 618-style peak over hop1 observations; held is unavailable."""
    observations = [
        obs
        for opened_door in opened
        for obs in opened_door["observations"]
    ]
    if not observations:
        return None
    cnt = Counter(obs["tok"] for obs in observations)
    if len(cnt) == 1:
        top = next(iter(cnt))
    else:
        (top, n1), (_second, n2) = cnt.most_common(2)
        if n1 < 2 or n1 <= n2:
            return None
    support = next(obs for obs in observations if obs["tok"] == top)
    return dict(
        tok=top,
        hop_pi=support["hop_pi"],
        majority=support["majority"],
        count_key=support["count_key"],
        votes=cnt[top],
        n_obs=len(observations),
        n_distinct=len(cnt),
    )


def _stats(values):
    if not values:
        return [0.0, 0.0, 0.0]
    return [
        math.tanh(statistics.fmean(values)),
        math.tanh(max(values)),
        math.tanh(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    ]


def state_features(
    pg, query_pi, pin, opened, total_doors, high_set, co, df, n_use,
):
    """Name-free W summary. Content equality may count; identities never enter."""
    observations = [
        obs
        for opened_door in opened
        for obs in opened_door["observations"]
    ]
    cnt = Counter(obs["tok"] for obs in observations)
    counts = sorted(cnt.values(), reverse=True)
    top = counts[0] if counts else 0
    second = counts[1] if len(counts) > 1 else 0
    commit = commit_resolved(opened)
    n_open = len(opened)
    n_obs = len(observations)

    q_door = []
    q_hop = []
    door_hop = []
    for opened_door in opened:
        dpi = opened_door["door_support_pi"]
        q_door.append(address_kernel(
            pg, query_pi, dpi, pin, high_set, co, df, n_use,
        ))
        for obs in opened_door["observations"]:
            hpi = obs["hop_pi"]
            q_hop.append(address_kernel(
                pg, query_pi, hpi, pin, high_set, co, df, n_use,
            ))
            door_hop.append(address_kernel(
                pg, dpi, hpi, pin, high_set, co, df, n_use,
            ))

    hop_chain = []
    hop_pis = [obs["hop_pi"] for obs in observations]
    for left, right in zip(hop_pis, hop_pis[1:]):
        hop_chain.append(address_kernel(
            pg, left, right, pin, high_set, co, df, n_use,
        ))

    if commit is None:
        maj = rows_n = keys_n = commit_votes = 0.0
    else:
        maj, neg_rows, neg_keys = commit["count_key"]
        rows_n = -neg_rows
        keys_n = -neg_keys
        commit_votes = commit["votes"]

    feats = [
        n_open / max(CAP, 1),
        max(total_doors - n_open, 0) / max(CAP, 1),
        math.log1p(total_doors) / 3.0,
        math.log1p(n_obs) / 4.0,
        len(cnt) / max(n_obs, 1),
        top / max(n_obs, 1),
        second / max(n_obs, 1),
        (top - second) / max(top, 1),
        float(commit is not None),
        commit_votes / max(n_obs, 1),
        float(maj),
        math.log1p(rows_n) / 4.0,
        math.log1p(keys_n) / 3.0,
        n_open * C_SEARCH,
    ]
    feats.extend(_stats(q_door))
    feats.extend(_stats(q_hop))
    feats.extend(_stats(door_hop))
    feats.extend(_stats(hop_chain))
    return feats, commit


def valid_actions(state_index, last_state, commit):
    """Held-blind action mask."""
    return (
        state_index < last_state,
        commit is not None,
        True,
    )


def full_feedback_q(commits, held):
    """Teacher-only dynamic Q targets for every operation at every W state."""
    targets = [None] * len(commits)
    future = 0.0
    last = len(commits) - 1
    for state_index in range(last, -1, -1):
        commit = commits[state_index]
        q_search = (
            -C_SEARCH + future if state_index < last else -1.0
        )
        q_commit = (
            (1.0 if commit["tok"] == held else -C_COMMIT)
            if commit is not None else -1.0
        )
        q_refuse = 0.0
        q = [q_search, q_commit, q_refuse]
        targets[state_index] = q
        future = max(q)
    return targets
