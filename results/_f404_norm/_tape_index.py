"""Shared slot context vocabulary: one word list for postings and ctx_fp."""
from __future__ import annotations
import re
WORD_RE = re.compile('[A-Za-z][a-z]{2,}')
CONTEXT_WORD_CAP = 40
CONTEXT_WORD_MIN = 3
CTX_WIN = 100
VOTES_AUTO_MIN_SLOTS = 512
DEFAULT_RETRIEVE_TOPK = 8
STOP = frozenset({'the', 'and', 'that', 'was', 'were', 'for', 'with', 'from', 'his', 'her', 'its', 'their', 'this', 'there', 'which', 'have', 'has', 'had', 'been', 'are', 'not', 'but', 'also', 'who', 'into', 'after', 'before', 'when', 'while', 'than', 'then', 'they', 'them', 'she', 'him'})

def context_words(text: str, exclude: str | None=None, cap: int=CONTEXT_WORD_CAP) -> list[str]:
    """Content words for a write/query window — used by postings and ctx_fp alike."""
    seen, out = (set(), [])
    for w in WORD_RE.findall(text):
        lw = w.lower()
        if lw in STOP or lw in seen or (exclude and w == exclude):
            continue
        seen.add(lw)
        out.append(lw)
        if len(out) >= cap:
            break
    return out

def nway_strict(gold: float, distractor_scores) -> bool:
    """Pessimistic n-way: gold must strictly beat every distractor. Tie = miss.

    Sparse vote scores often leave gold=0 and pool slots=0; ``gold >= 0`` then
    counts total retrieval failure as a win. Ranks that use ``v > gold`` stay
    correct — compare top1 vs acc_nway to see inflation from the old ``>=``.
    """
    return all((float(gold) > float(s) for s in distractor_scores))

def vote_rank(sc: dict, gold_slot: int, n_slots: int) -> tuple[float, int]:
    """Sparse vote rank. Empty retrieval must not count as top1.

    ``rank = 1 + sum(v > gold for v in sc.values())`` only sees slots that received
    mass. When the query emits no words (or no postings hit), ``sc`` is empty,
    gold=0, and the sum is 0 → rank 1 — a false hit. Empty / all-zero retrieval
    → worst rank ``n_slots``. Silent gold (score≤0) is never rank 1.
    """
    n_slots = max(1, int(n_slots))
    if not sc:
        return (0.0, n_slots)
    gold = float(sc.get(gold_slot, 0.0))
    if gold <= 0.0 and max((float(v) for v in sc.values())) <= 0.0:
        return (gold, n_slots)
    rank = 1 + sum((1 for v in sc.values() if float(v) > gold))
    if gold <= 0.0:
        rank = max(rank, 2)
    return (gold, int(rank))

def vote_arm_fields(rows: list[dict]) -> dict:
    """Permanent decision block for every vote-based arm.

    Each row: ``gold_score`` (float), ``rank`` (int, 1=hit), ``low_overlap`` (bool).
    ``tie_at_zero_frac`` = share of queries where gold got no vote mass — the index
    is silent. Low-overlap top1 collapse is usually the same queries, not a separate
    mechanism: see ``low_overlap_miss_is_silence_frac`` and ``top1_low_overlap_given_vote``.
    """
    if not rows:
        return {'tie_at_zero_frac': float('nan'), 'tie_at_zero_frac_low_overlap': float('nan'), 'tie_at_zero_frac_high_overlap': float('nan'), 'top1_low_overlap': float('nan'), 'top1_high_overlap': float('nan'), 'top1_low_overlap_given_vote': float('nan'), 'low_overlap_miss_is_silence_frac': float('nan'), 'n': 0, 'n_tie_at_zero': 0, 'n_low_overlap': 0}

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float('nan')
    silent = [r['gold_score'] <= 0.0 for r in rows]
    low = [r for r in rows if r['low_overlap']]
    high = [r for r in rows if not r['low_overlap']]
    low_miss = [r for r in low if int(r['rank']) != 1]
    low_hit = [r for r in low if r['gold_score'] > 0.0]
    return {'tie_at_zero_frac': mean([1.0 if s else 0.0 for s in silent]), 'tie_at_zero_frac_low_overlap': mean([1.0 if r['gold_score'] <= 0.0 else 0.0 for r in low]), 'tie_at_zero_frac_high_overlap': mean([1.0 if r['gold_score'] <= 0.0 else 0.0 for r in high]), 'top1_low_overlap': mean([1.0 if int(r['rank']) == 1 else 0.0 for r in low]), 'top1_high_overlap': mean([1.0 if int(r['rank']) == 1 else 0.0 for r in high]), 'top1_low_overlap_given_vote': mean([1.0 if int(r['rank']) == 1 else 0.0 for r in low_hit]), 'low_overlap_miss_is_silence_frac': mean([1.0 if r['gold_score'] <= 0.0 else 0.0 for r in low_miss]), 'n': len(rows), 'n_tie_at_zero': int(sum((1 for s in silent if s))), 'n_low_overlap': len(low)}