"""Shared honest holdout helpers for post-624 exams.

Two fillers of one address are two records. Never invent a joint ctx-ask row.
Held never filters pin/leftover candidates (pass forbid={pin} only).
"""
from __future__ import annotations

from _audit589_hop3 import adjust_frame_stats


def hide_two(co, df, keys, held_ctx, held_ask, delta):
    """Add/remove the two real records of (keys,ctx) and (keys,ask)."""
    ctx_row = set(keys) | {held_ctx}
    ask_row = set(keys) | {held_ask}
    adjust_frame_stats(co, df, ctx_row, delta)
    adjust_frame_stats(co, df, ask_row, delta)
