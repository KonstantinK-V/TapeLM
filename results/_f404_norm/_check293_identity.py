"""Prove 293 before spending an hour on it. Seconds, no corpus and no model.

293 moves the mind to where fp_addresses' threshold currently stands: the verb is no longer
"what value goes in this slot" but "do these mentions name the same place". That is the one
decision in this project still made by a hand-set rule, and putting Phi there is worth nothing
if the question can be answered without understanding anything. Seven things have to hold, and
each is a way the run could print a plausible number that means nothing:

  1 THE LABEL IS NOT CIRCULAR AND NOT TRIVIAL. The truth is the (anchor, value) pair - two
    strings the encoder never touches - and every candidate, the truth included, has relation
    words DISJOINT from the core's. A label taken from fp_addresses' grouping would make the
    rival the labeller; a label taken from string-address identity would be free to the shared
    word channel, because the same string address means the same relation words.
  2 THE LABEL CANNOT BE SEEN. The value is what the label is made of and it is excluded from
    every channel: hidden as a sentinel, so the same-value edge is exactly zero in every world,
    and already excluded from the ink and the word sets by ctx_fp/context_words. --ident-values
    show is the sanity bolt in the other direction: with values visible the label IS an input.
  3 EVERY WORLD CARRIES THE SAME NUMBER OF ROWS, and they differ in exactly one row. Unequal
    counts are the bookkeeping tell that made 289's ladder unreadable and 291's refusal unfair.
  4 THE CANDIDATE ROW IS MARKED, IDENTICALLY IN ALL FOUR WORLDS. A proposed member may not be
    mistaken for an observed one, and the mark may not carry the label.
  5 POSITION SAYS NOTHING. 292's rungs were built by relatedness and Phi learned to read the
    construction rather than the fit - an inverted landscape on three seeds out of three. Here
    the four candidates are shuffled, so the label's position is uniform.
  6 THE QUESTION IS DETERMINISTIC given its rng, or every evaluation is a different question.
  7 THE HEURISTIC RIVAL IS THE WRITING RULE, not a description of it: at an impossible tau it
    declines to link and answers nothing, at a permissive one it links and answers.

    python _check293_identity.py
"""
from __future__ import annotations
import random
from collections import Counter
import torch
import _stage289_derivation as s289

class FakeTape:

    def __init__(self, values):
        self.values = values

def fake_pack():
    """One anchor. Kaluga is written twice in words that share nothing - one place said two ways,
    which is the fragmentation case - and four same-anchor mentions carry other values."""
    texts = ['kostya was born in kaluga in the spring of that year', 'kostya reportedly hails from kaluga according to the parish register', 'kostya played for spartak during the winter season', 'kostya died in moscow after a long illness', 'kostya captained the reserve side for two seasons', 'kostya studied at gorky university before the war']
    vals = ['Kaluga', 'Kaluga', 'Spartak', 'Moscow', 'Reserve', 'Gorky']
    straddr = ['kostya|born in', 'kostya|hails from', 'kostya|played for', 'kostya|died in', 'kostya|captained', 'kostya|studied at']
    items = [{'S': 'kostya', 'address': 'fp0:kostya|born in', 'slots': [0, 1], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp1:kostya|played for', 'slots': [2], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp2:kostya|died in', 'slots': [3], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp3:kostya|captained', 'slots': [4], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp4:kostya|studied at', 'slots': [5], 'kind': 'clean'}]
    postings = {}
    for i, t in enumerate(texts):
        for w in t.split():
            postings.setdefault(w, []).append(i)
    n = len(texts)
    g = torch.Generator().manual_seed(293)
    ctx = torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=-1)
    anc = torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=-1)
    return {'tape': FakeTape(vals), 'texts': texts, 'texts_lc': [t.lower() for t in texts], 'items': items, 'postings': postings, 'n_slots': n, 'straddr': straddr, 'slot_keys_slot': list(range(n)), 'ctx_keys': ctx, 'anc_keys': anc}

class FakeBank:

    def ctx_fp(self, text, exclude=None):
        g = torch.Generator().manual_seed(abs(hash(text)) % 2 ** 31)
        return torch.nn.functional.normalize(torch.randn(16, generator=g), dim=-1)

def main() -> int:
    ok = True
    dev = torch.device('cpu')
    s289.IDENTITY, s289.IDENT_CANDS, s289.IDENT_CORE = (True, 4, 3)
    s289.IDENT_VALUES, s289.IDENT_TAU, s289.IDENT_OVERLAP = ('hide', 0.9, 2)
    s289.EDGES_ON = set(s289.EDGES)
    s289.NEIGHBOURS, s289.OPEN, s289.IMPORT_K = (0, False, 0)
    p, bank = (fake_pack(), FakeBank())
    s289.IDENT_SUPPLY.clear()
    qs = [q for q in s289.identity_questions_for(p, random.Random(0)) if q.get('ident')]
    print(f'0 questions built: {len(qs)}  supply {dict(s289.IDENT_SUPPLY)}')
    ok &= len(qs) == 2
    q = qs[0]
    truth = q['cand_slots'][q['label']]
    core_rel = set()
    for s in q['slots']:
        core_rel |= s289.str_parts(p['straddr'][s])[1]
    by_value = [s for s in q['cand_slots'] if p['tape'].values[s] == q['place'][1]]
    v = by_value == [truth] and all((s289.str_parts(p['straddr'][s])[0] == q['S'] for s in q['cand_slots'])) and all((not s289.str_parts(p['straddr'][s])[1] & core_rel for s in q['cand_slots'])) and (p['straddr'][truth] not in {p['straddr'][s] for s in q['slots']}) and (truth not in q['slots'])
    ok &= bool(v)
    print(f"1 place {q['place']}, core {q['slots']} rel {sorted(core_rel)}, cands {q['cand_slots']} truth {truth}")
    print(f'  value identifies the truth uniquely, no candidate shares a relation word, the truth is phrased differently: {bool(v)}')
    worlds = [s289.build_graph(p, s289.identity_world(p, q, s), bank, dev, query_value=None, import_k=0) for s in q['cand_slots']]
    v = all((float(same.abs().sum()) == 0.0 for _, same, _ in worlds))
    ok &= v
    print(f'2 same-value edge is exactly zero in every world (values hidden): {v}')
    s289.IDENT_VALUES = 'show'
    shown = s289.build_graph(p, s289.identity_world(p, q, q['cand_slots'][0]), bank, dev, query_value=None, import_k=0)
    s289.IDENT_VALUES = 'hide'
    print(f'  --ident-values show restores them (nonzero same edges: {int((shown[1] != 0).sum())}), so the shortcut stays measurable')
    ns = {tuple(nf.shape) for _, _, nf in worlds}
    v = len(ns) == 1 and len(q['slots']) + 1 == worlds[0][2].shape[0]
    ok &= v
    print(f'3 every world has the same shape {ns}, core+1 rows: {v}')
    qcol = [nf[:, 4].tolist() for _, _, nf in worlds]
    v = all((c == qcol[0] for c in qcol)) and qcol[0][-1] == 1.0 and (sum(qcol[0]) == 1.0)
    ok &= v
    print(f'4 the candidate row is the query row in all four, identically {qcol[0]}: {v}')
    pos = Counter()
    for seed in range(400):
        qq = s289.identity_question(p, 'kostya', 'Kaluga', 0, random.Random(seed))
        pos[qq['label']] += 1
    v = len(pos) == 4 and max(pos.values()) - min(pos.values()) < 80
    ok &= v
    print(f'5 label position over 400 draws {dict(sorted(pos.items()))}: {v}')
    a = s289.identity_question(p, 'kostya', 'Kaluga', 0, random.Random(11))
    b = s289.identity_question(p, 'kostya', 'Kaluga', 0, random.Random(11))
    v = a['cand_slots'] == b['cand_slots'] and a['label'] == b['label'] and (a['slots'] == b['slots'])
    ok &= v
    print(f'6 same rng, same question: {v}')
    s289.IDENT_TAU, s289.IDENT_OVERLAP = (2.0, 2)
    strict = s289.ident_rivals(p, q)
    s289.IDENT_TAU, s289.IDENT_OVERLAP = (-1.0, 0)
    loose = s289.ident_rivals(p, q)
    s289.IDENT_TAU, s289.IDENT_OVERLAP = (0.9, 2)
    v = strict['heur'] is None and loose['heur'] is not None and (loose['_heur_accepted'] == 4)
    ok &= v
    print(f"7 tau 2.0 -> declines to link ({strict['heur']}), tau -1 -> links all four ({loose['_heur_accepted']}): {v}")
    print(f"  cos1nn {loose['cos1nn']}  rare {loose['rare']}  truth s{truth}")
    s289.IDENT_IMPORT = 2
    q.pop('_ibudget', None)
    b = s289.ident_budget(p, q)
    ns2 = {s289.build_graph(p, s289.identity_world(p, q, s), bank, dev, query_value=None, import_k=0)[2].shape[0] for s in q['cand_slots']}
    s289.IDENT_IMPORT = 0
    v = b == 0 and len(ns2) == 1
    ok &= v
    print(f'8 --ident-import 2 on a tape with no siblings: budget {b}, world sizes {ns2}: {v}')
    au = s289.identity_audit(p, random.Random(1))
    print(f'\naudit on the toy tape: {au}')
    print('\nIDENT OK' if ok else '\nIDENT FAILED')
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())