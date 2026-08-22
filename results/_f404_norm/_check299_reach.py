"""Prove 299 before spending an hour on it. Seconds, no corpus and no model.

299 removes the offered candidate list. A hole is hidden, the mind walks to the nearest places
by frame fingerprint, and it may say only a filler it reached - or nothing. Five things:

  1 NOBODY OFFERS ANYTHING. The candidate set is whatever the walk reached, in the walk's own
    order, never by frequency - frequency is exactly the rival's rule and handing it to the
    construction would decide the comparison in advance.
  2 THE TRUTH IS NOT PLANTED. If no reached place carries it, silence is the correct answer and
    the question is unanswerable without anyone rigging a 50/50 split.
  3 EVERY CANDIDATE WORLD IS THE SAME SIZE - one shared budget - so no row count is a tell.
  4 THE TWO STAGES ARE WHAT THE WALK BUYS. Before it, only the values already at this address
    are sayable; after it, the tape is. Expanding is the difference between having options and
    not having them, and it is priced.
  5 THE PAYOFF IS 280's, with silence correct exactly when the truth was out of reach.

    python _check299_reach.py
"""
from __future__ import annotations
import random
import torch
import torch.nn.functional as F
import _stage289_derivation as s289
from _check293_identity import FakeBank
from _check294_open import pack294

def main() -> int:
    ok = True
    dev = torch.device('cpu')
    s289.IDENTITY = s289.MIXED = s289.OPEN = False
    s289.NEIGHBOURS, s289.IMPORT_K, s289.STEP_COST = (0, 2, 0.05)
    s289.REACH, s289.REACH_K, s289.REACH_CANDS = (True, 8, 8)
    s289.EDGES_ON = set(s289.EDGES)
    p, bank = (pack294(), FakeBank())
    g = torch.Generator().manual_seed(299)
    p['frame_fps'] = list(F.normalize(torch.randn(p['n_slots'], 16, generator=g), dim=-1))
    p['frame_nfill'] = [2] * p['n_slots']
    p['frame_nfill_max'] = 2
    p['frame_mode'] = True
    qs = [q for q in s289.reach_questions_for(p, random.Random(0)) if q.get('reach')]
    print(f'0 questions {len(qs)}')
    ok &= len(qs) > 0
    q = qs[0]
    rc = s289.reach_candidates(p, q)
    print(f"1 walked to {len(rc['places'])} places -> candidates {rc['cands']}")
    reached = {p['tape'].values[s] for _j, it, _ in rc['places'] for s in it['slots']}
    v = set(rc['cands']) <= reached and len(rc['cands']) <= s289.REACH_CANDS
    ok &= bool(v)
    print(f'  every candidate came from a reached place, cap respected: {bool(v)}')
    v = q['address'] not in {it['address'] for _j, it, _ in rc['places']}
    ok &= bool(v)
    print(f'  the walk does not return to its own place: {bool(v)}')
    v = len(q['slots']) <= s289.REACH_MAX_ROWS
    ok &= bool(v)
    print(f"  own place capped at {s289.REACH_MAX_ROWS} rows (got {len(q['slots'])}): {bool(v)}")
    ansb = s289.reach_answerable(p, q)
    print(f"2 truth {q['truth_value']!r} reachable: {ansb} (nobody planted it)")
    torch.manual_seed(0)
    net = s289.Deriver(dev, d=8, n_node=9)
    rows_of, cands = (rc['rows_of'], rc['cands'])
    budget = min([s289.IMPORT_K] + [len(rows_of[c]) for c in cands]) if cands else 0
    sizes = {len(s289.reach_world(p, q, bank, dev, c, rows_of[c], budget)[2]) for c in cands}
    v = len(sizes) <= 1
    ok &= bool(v)
    print(f'3 candidate worlds all {sizes} rows, budget {budget}: {bool(v)}')
    l1, l2, own, cs, _l3, _lc = s289.reach_logits(net, p, q, dev, bank)
    v = len(l1) == len(own) + 2 and len(l2) == len(cs) + 1
    ok &= bool(v)
    print(f'4 stage 1 = own {own} + refuse + expand ({len(l1)}), stage 2 = reached + refuse ({len(l2)}): {bool(v)}')
    R2 = s289.reach_reward(q, cs + [s289.REFUSE_LABEL], ansb, dev)
    want = 1.0 if not ansb else 0.75
    v = float(R2[-1]) == want
    ok &= bool(v)
    print(f'5 silence pays {float(R2[-1])} (reachable={ansb}, so {want}): {bool(v)}')
    loss = s289.reach_loss(net, p, q, dev, bank)
    p1, p2 = (torch.softmax(l1, 0), torch.softmax(l2, 0))
    R1 = s289.reach_reward(q, own + [s289.REFUSE_LABEL], ansb, dev)
    manual = -((p1[:-1] * R1).sum() + p1[-1] * (s289.REACH_GAMMA * (p2 * R2).sum() - s289.STEP_COST))
    v = abs(float(loss) - float(manual)) < 1e-06
    ok &= bool(v)
    print(f'  loss == expected payoff of the walk: {bool(v)} ({float(loss):+.6f} vs {float(manual):+.6f})')
    s289.REACH_NO_REFUSE = True
    q.pop('_reach_g', None)
    l1n, l2n, own_n, cs_n, _l3n, _lcn = s289.reach_logits(net, p, q, dev, bank)
    n1, n2 = s289.reach_names(own_n, cs_n)
    v = len(l1n) == len(own_n) + 1 and n1 == own_n and (len(l2n) == len(cs_n) if cs_n else len(l2n) == 1) and (n2 == (cs_n or [s289.REFUSE_LABEL]))
    ok &= bool(v)
    print(f'6 no-refuse: stage1={list(n1)}+expand ({len(l1n)}), stage2={list(n2)} ({len(l2n)}): {bool(v)}')
    s289.REACH_LOOKAHEAD = True
    q.pop('_reach_g', None)
    l1l, l2l, own_l, cs_l, _l3l, _lcl = s289.reach_logits(net, p, q, dev, bank)
    v = len(l1l) == len(own_l) + 1 and abs(float(l1l[-1]) - float(l2l.max())) < 1e-06 and (len(l1l) == len(own_l) + 1)
    ok &= bool(v)
    print(f'7 lookahead: step logit == max(l2) ({float(l1l[-1]):+.4f}): {bool(v)}')
    s289.REACH_LOOKAHEAD = False
    s289.REACH_NO_REFUSE = False
    print('\nREACH OK' if ok else '\nREACH FAILED')
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())