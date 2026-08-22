"""Prove the region views before spending an hour on them. Seconds, no model needed.

recon3 closed the thin-view question: random subsamples share ~65% of their rows, so the
views were resamples of one reading - pooled lost to single (z -1.67) and D was blind on
train (auc 0.485) while real held out (0.702, z 4.71). Region views are the correction:
disjoint stretches of the tape in write order, so D measures whether the CORPUS agrees with
itself, not whether one sampler agrees with another. Five properties have to hold, and every
one of them is a property of the algebra rather than of the corpus, so synthetic questions
test them honestly and instantly:

  1 the cut is a partition: every evidence row in exactly one region, no region empty,
    order preserved. A row lost here is a row the pooled answer silently never saw.
  2 thin mode is untouched: pool_views(L, None) == L.mean(0) bit for bit (torch.equal),
    so every recon3-lineage number stays reproducible from this same file.
  3 the pooling is shift-invariant per view: adding a constant to one region's logits moves
    nothing. That is the derivation of the centering - break it and "abstain" becomes "vote".
  4 a region whose rows all carry one value pools exactly zero: support counts stay the
    tape's channel, only contrastive reading pools.
  5 masked disagreement is finite with exact zeros, zero when regions agree, and ln 2 when
    two regions put all mass on values the other never wrote - the contested-address maximum.

And one property of the machinery: region views consume NO randomness. They are a function
of the tape, and the rng must come back in the same state it went in.

    python _check289_region.py
"""
from __future__ import annotations
import math
import random
import torch
import _stage289_derivation as s289

def fake_q(vals, hid_val, cands=None):
    """A lookup question as lookup_question builds it: survivors first, sentinel query row."""
    cands = sorted(set(vals)) if cands is None else cands
    return {'verb': 'lookup', 'slots': list(range(len(vals) + 1)), 'vals': list(vals) + [object()], 'cands': cands, 'label': cands.index(hid_val), 'S': 's', 'address': 'fp0:s|r', 'query_row': len(vals)}

def main() -> int:
    ok = True
    s289.VIEWS, s289.VIEW_MODE, s289.ROW_DROPOUT = (3, 'region', 0.0)
    v = True
    for qr in range(2, 12):
        q = fake_q([f'v{i % 3}' for i in range(qr)], 'v0')
        regs = s289.region_views_of(q, 3)
        flat = [s for r in regs for s in r['slots'][:r['query_row']]]
        v &= flat == list(range(qr)) and all((r['query_row'] > 0 for r in regs))
        v &= all((r['slots'][r['query_row']] == q['slots'][q['query_row']] for r in regs))
        v &= all((r['cands'] == q['cands'] and r['label'] == q['label'] for r in regs))
    ok &= v
    print(f'1 partition: every row once, no empty region, query row kept, cands global: {v}')
    L = torch.randn(4, 3, dtype=torch.float64)
    v = torch.equal(s289.pool_views(L, None), L.mean(0))
    ok &= v
    print(f'2 pool_views(L, None) == L.mean(0) exactly: {v}')

    def plain_pool(L0, Lr, M):
        out = [x - sum(L0) / len(L0) for x in L0]
        for lv, mv in zip(Lr, M):
            pres = [l for l, m in zip(lv, mv) if m]
            mu = sum(pres) / len(pres)
            out = [o + m * (l - mu) for o, l, m in zip(out, lv, mv)]
        return out
    rnd, v = (random.Random(0), True)
    for _ in range(200):
        C, V = (rnd.randint(2, 5), rnd.randint(2, 4))
        L = torch.randn(V + 1, C, dtype=torch.float64)
        M = (torch.rand(V, C) < 0.6).double()
        for r in range(V):
            if not M[r].any():
                M[r, rnd.randrange(C)] = 1.0
        base = s289.pool_views(L, M)
        v &= max((abs(a - b) for a, b in zip(base.tolist(), plain_pool(L[0].tolist(), L[1:].tolist(), M.tolist())))) < 1e-09
        Ls = L.clone()
        Ls[1 + rnd.randrange(V)] += rnd.uniform(-5, 5)
        v &= float((base - s289.pool_views(Ls, M)).abs().max()) < 1e-09
    ok &= v
    print(f'3 torch pool == plain-python pool; per-region shift moves nothing (200 draws): {v}')
    L = torch.randn(2, 3, dtype=torch.float64)
    M = torch.tensor([[1.0, 0.0, 0.0]])
    v = float((s289.pool_views(L, M) - (L[0] - L[0].mean())).abs().max()) < 1e-12
    ok &= v
    print(f'4 single-candidate region contributes exactly zero: {v}')
    big = torch.tensor([[9.0, 0.0, -9.0], [9.0, 0.0, -9.0]])
    agree = s289.disagreement(big, torch.ones(2, 3))
    disj = s289.disagreement(torch.zeros(2, 3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    part = s289.disagreement(torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]), torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    v = agree < 1e-06 and abs(disj - math.log(2)) < 1e-06 and (agree < part < disj)
    ok &= v
    print(f'5 D: agree={agree:.2e}  partial={part:.4f}  disjoint={disj:.4f} (ln2={math.log(2):.4f}): {v}')
    q = fake_q(['a', 'a', 'b', 'c'], 'b')
    rng = random.Random(7)
    state = rng.getstate()
    qvs, M = s289.views_and_mask(q, rng, torch.device('cpu'))
    v = rng.getstate() == state and len(qvs) == 4 and (M.tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ok &= v
    print(f'6 mask per region {M.tolist()}, rng untouched: {v}')
    for _ in range(3):
        qvs2, M2 = s289.views_and_mask(q, random.Random(99), torch.device('cpu'))
        v = torch.equal(M, M2) and all((a['slots'] == b['slots'] for a, b in zip(qvs, qvs2)))
        ok &= v
    print(f'  deterministic across calls and rng seeds: {v}')
    print('\nREGION OK' if ok else '\nREGION FAILED')
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())