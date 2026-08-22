"""Prove 296 before spending an hour on it. Seconds, no corpus and no model.

296 stops grading halves. One exam, one payoff: find the value the corpus put somewhere, or say
there is none. Five things have to hold, and each is a way the run could print a number that
means nothing:

  1 THE SPLIT IS BALANCED. 291 died because refusal was correct 93% of the time, so any mark of
    the refusal option was a mark of the answer. At 50/50 the mark carries nothing, which is the
    only reason the refusal world may keep its natural row count without becoming a tell.
  2 UNANSWERABLE IS A LIST WITHOUT THE ANSWER, not a smaller world. The truth is present among
    the candidates exactly when the question is answerable, and refusal is on every list.
  3 THE CANDIDATE WORLDS ARE ALL THE SAME SIZE, in both kinds of question, so no row count
    distinguishes an answerable question from an unanswerable one.
  4 THE PAYOFF IS 280's, EXACTLY, and it prices both degenerate policies: always answering loses
    a point on every unanswerable question, always refusing gives up every point it could have
    found. If either beat a competent policy the exam would be measuring nothing.
  5 REFUSAL DOES NOT ZERO THE IMPORT BUDGET. outside_mentions of a sentinel is empty, so a naive
    minimum over the candidate list would collapse every world to the bare evidence and make the
    question contentless - the failure 292 spent a run learning about.

    python _check296_mixed.py
"""
from __future__ import annotations
import random
from collections import Counter
import torch
import _stage289_derivation as s289
from _check293_identity import FakeBank
from _check294_open import pack294

def main() -> int:
    ok = True
    dev = torch.device('cpu')
    s289.IDENTITY, s289.NEIGHBOURS, s289.REFUSE = (False, 0, False)
    s289.OPEN, s289.MIXED, s289.IMPORT_K = (True, True, 2)
    s289.ADDRESS_FROM, s289.OPEN_CANDS, s289.OPEN_N_CANDS = ('anchor', 'uniform', 4)
    s289.ANCHOR_MAX_ROWS = 3
    s289.EDGES_ON = set(s289.EDGES)
    p, bank = (pack294(), FakeBank())
    base = len(p['texts'])
    fillers = [('omega held a summit in geneva that spring', 'Geneva'), ('omega opened a studio in oslo after the tour', 'Oslo'), ('omega signed a pact in cairo before dawn', 'Cairo'), ('omega moved reserves to dublin for winter', 'Dublin'), ('omega founded a chapter in prague overnight', 'Prague'), ('omega shipped crates to lisbon that month', 'Lisbon')]
    for i, (t, v) in enumerate(fillers):
        p['texts'].append(t)
        p['tape'].values.append(v)
        p['straddr'].append(f'omega|{t.split()[1]}')
        for w in t.split():
            p['postings'].setdefault(w, []).append(base + i)
    n = len(p['texts'])
    p['texts_lc'] = [t.lower() for t in p['texts']]
    p['n_slots'] = n
    g = torch.Generator().manual_seed(296)
    p['ctx_keys'] = torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=-1)
    p['anc_keys'] = torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=-1)
    p['slot_keys_slot'] = list(range(n))
    p['items'].append({'S': 'omega', 'address': 'fp9:omega|held', 'slots': list(range(base, base + len(fillers))), 'kind': 'clean'})
    p.pop('_ident', None)
    item = next((it for it in s289.anchor_items(p) if it['S'] == 'kostya'))
    allv = list(p['tape'].values)
    kinds, made = (Counter(), [])
    rng = random.Random(0)
    for seed in range(400):
        q = s289.lookup_mixed_question(p, item, rng, 2, allv)
        if q is None:
            continue
        kinds[q['answerable']] += 1
        made.append(q)
    n = max(1, len(made))
    rate = kinds[True] / n
    v = 0.4 < rate < 0.6
    ok &= v
    print(f'1 answerable rate {rate:.3f} over {n} draws: {v}')
    v = True
    for q in made:
        truth = q['truth_value']
        v &= s289.REFUSE_LABEL in q['cands']
        v &= (truth in q['cands']) == q['answerable']
        v &= q['cands'][q['label']] == (truth if q['answerable'] else s289.REFUSE_LABEL)
        v &= len(q['cands']) == s289.OPEN_N_CANDS + 1
        ev = [p['tape'].values[s] for s in q['slots'][:q['query_row']]]
        v &= all((c not in ev for c in q['cands']))
    ok &= bool(v)
    qa = next((q for q in made if q['answerable']))
    qu = next((q for q in made if not q['answerable']))
    print(f'2 truth on the list iff answerable, refusal always on it, nothing on a row: {bool(v)}')
    print(f"  answerable {qa['cands']}\n  unanswerable {qu['cands']} truth {qu['truth_value']!r}")
    sizes = {}
    for name, q in (('answerable', qa), ('unanswerable', qu)):
        k = s289.shared_import_budget(p, q, list(q['cands']))
        got = {}
        for c in q['cands']:
            q.pop('_base', None)
            got[c] = len(s289.build_graph(p, q, bank, dev, query_value=c, import_k=k)[2])
        sizes[name] = got
        cand = {v_ for c, v_ in got.items() if c != s289.REFUSE_LABEL}
        v = len(cand) == 1
        ok &= v
        print(f'3 {name}: candidate worlds {sorted(cand)}, refusal world {got[s289.REFUSE_LABEL]}, budget {k}: {v}')
    v = {x for x in sizes['answerable'].values()} == {x for x in sizes['unanswerable'].values()}
    ok &= v
    print(f'  and the two kinds are indistinguishable by size: {v}')
    cells = {(True, False, True): 1.0, (True, True, False): 0.75, (True, False, False): -1.0, (False, True, False): 1.0, (False, False, False): -1.0}
    v = all((s289.mixed_payoff(sil, right, ans) == want for (ans, sil, right), want in cells.items()))
    ok &= v
    print(f'4 payoff cells exact (+1 found / -1 wrong / +1 correct silence / +0.75 hedge): {v}')
    rows = [(1, 0, 1), (1, 0, 0), (0, 1, 0), (0, 1, 0)]
    always_a = sum((s289.mixed_payoff(False, bool(r), bool(a)) for a, _, r in rows)) / 4
    always_s = sum((s289.mixed_payoff(True, False, bool(a)) for a, _, r in rows)) / 4
    perfect = sum((s289.mixed_payoff(not a, True, bool(a)) for a, _, r in rows)) / 4
    v = perfect > always_a and perfect > always_s
    ok &= v
    print(f'  always-answer {always_a:+.3f}  always-silent {always_s:+.3f}  competent {perfect:+.3f} - both degenerates lose: {v}')
    k_with = s289.shared_import_budget(p, qa, list(qa['cands']))
    k_without = s289.shared_import_budget(p, qa, [c for c in qa['cands'] if c != s289.REFUSE_LABEL])
    v = k_with == k_without >= 1
    ok &= v
    print(f'5 budget with refusal on the list {k_with} == without it {k_without}, and nonzero: {v}')
    print('\nMIXED OK' if ok else '\nMIXED FAILED')
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())