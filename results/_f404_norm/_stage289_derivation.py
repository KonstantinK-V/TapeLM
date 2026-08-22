"""
Stage 289 — the derivation moves into exact algebra, and the mind keeps only the judgment.

The project's invariant is one line: whatever DECIDES may not be approximate; whatever is
TRAINED may not hold facts. Two runs of this stage violated it from the other side - the
weights were being asked to approximate something exactly computable from their own input.

COUNT and COMPARE are functions of the same-value relation s_ij alone:

  new_i   = 1 - max_{j<i} s_ij      (is row i the first of its group; row order is tape order)
  count   = sum_i new_i             (number of groups)
  compare = sign(count_A - count_B)

Today s_ij is string equality, so both answers are EXACT - union-find arithmetic, zero
parameters. The 5+ cap existed because a classifier needs a closed answer set; the arithmetic
has none - exact_count returns 40 on forty distinct values - and count_label now caps only the
string printed in the confusion table. The earlier gate measured how well
7.9k parameters approximate this - count 0.965 falling to 0.903 as lookup grew into a real
task, because a description and a preference ordering were fighting over one pooled vector.
That interference is not fixed here; it is REMOVED: there is one trained task left.

What stays trained is the only place with real uncertainty: LOOKUP. The hidden mention's
context is the query; for each candidate the query row is COMPLETED with that value and the
resulting world is pooled and scored by one scalar Phi. The candidate whose world hangs
together best wins. This is 288's repair loop turned inward, and the query-row indicator stays
set so a completed world is never mistaken for an observed one.

The two trained surfaces this architecture leaves are both judgments, never arithmetic:
Phi (world coherence, examined here), and s_ij itself once values stop being exact strings
("USA" / "United States") - which is 286's evidence-agreement task, so the loop closes.

The examiner for lookup is paired IN-RUN: the majority rival answers the same questions, so
McNemar on the discordant items replaces two marginals. The rival over survivors is
Bayes-optimal when the query context carries nothing, so beating it - significantly, paired -
is exactly the claim that the context channel carries information counts do not have.

  python _stage289_derivation.py --smoke
  python _stage289_derivation.py --train-steps 6000
  python _stage289_derivation.py --train-steps 6000 --holdout address
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from itertools import zip_longest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage271_controller as s271
import _stage279_write_decision as s279
import _stage280_raw_exam as s280
import _stage286_evidence as s286
import _stage289a_presupposition as s289a
import _tape_frames as tframes
from _tape_speed import INK_DEGENERATE, WORD_RULES, BigramBank, CachedBank, HashFp, install_assertion_cache, install_fast_fp_addresses, verify_hash_ink, verify_word_rule
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 2890
COUNT_MAX = 5
COUNT_LABELS = tuple((str(i) for i in range(1, COUNT_MAX))) + (f'{COUNT_MAX}+',)
COMPARE_LABELS = ('first', 'second', 'equal')
LOG_PATH = RES / '_stage289_log.txt'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line)

def count_label(k: int) -> str:
    return COUNT_LABELS[min(k, COUNT_MAX) - 1]

def soft_new(vals, sim=None):
    """new_i = 1 - max_{j<i} s_ij : is row i the first of its group.

    THE FORMULA IS THE GENERAL CASE, and that is the whole reason it is safe to take counting
    out of the weights. With s_ij in {0,1} - today's string equality - this is union-find and
    the answer is exact. With s_ij in [0,1] it is a differentiable soft count: new_i is a real
    number, the sum is fractional, and the gradient flows straight into whatever produced s_ij.
    The day sameness becomes a trained judgment ("USA" / "United States"), the mind trains
    THROUGH this arithmetic without one character of it changing.

    So the exact algebra is not a wall replacing a learnable part. It is the degenerate case of
    a learnable one, with today's degenerate sameness substituted in. The rule that put it here
    is not "whatever is computable, compute it" - that rule is a ratchet that ends with no mind
    at all, since every answer about a finite tape is computable in principle. The rule is:
    whatever the input DETERMINES is arithmetic; whatever the input UNDERDETERMINES is judgment.
    count given s_ij is determined. s_ij is not. Which value fills a gap is not. Whether a
    question should be answered at all is not. Those stay with the mind.
    """
    sim = sim or (lambda a, b: 1.0 if a == b else 0.0)
    return [1.0 - (max((sim(vals[j], vals[i]) for j in range(i))) if i else 0.0) for i in range(len(vals))]

def soft_count(vals, sim=None) -> float:
    return sum(soft_new(vals, sim))

def exact_new(vals, same=None):
    """The hard special case, kept because integers are what the examiner compares."""
    sim = None if same is None else lambda a, b: 1.0 if same(a, b) else 0.0
    return [int(round(x)) for x in soft_new(vals, sim)]

def exact_count(vals, same=None) -> int:
    return sum(exact_new(vals, same))

def exact_answer(q):
    """The exact verdict for the exact verbs; raises on lookup, which is judged, not computed."""
    if q['verb'] == 'count':
        return count_label(exact_count(q['vals']))
    if q['verb'] == 'compare':
        ka = exact_count(q['vals'][:q['n_first']])
        kb = exact_count(q['vals'][q['n_first']:])
        return 'first' if ka > kb else 'second' if kb > ka else 'equal'
    raise ValueError('lookup is not exact: it is the judgment the mind is for')
TRAIN_VERBS = ('lookup',)

def count_question(pack, item):
    """How many distinct values does this address carry? The truth is a property of the tape."""
    vals = [pack['tape'].values[s] for s in item['slots']]
    if len(vals) < 2:
        return None
    return {'verb': 'count', 'slots': list(item['slots']), 'vals': vals, 'label': count_label(len(set(vals))), 'S': item['S'], 'address': item['address']}

def compare_question(pack, a, b):
    """Which of two addresses carries more distinct values?

    Both address's mentions go into ONE graph, with a side indicator per row. Nothing tells the
    mind how many rows each side has beyond what it can see, and the answer is not a count but
    an ordering, so the two verbs cannot share a shortcut: a mind that memorised "this many rows
    means this label" for COUNT gets nothing here, where both sides sit in the same graph.
    """
    va = [pack['tape'].values[s] for s in a['slots']]
    vb = [pack['tape'].values[s] for s in b['slots']]
    if len(va) < 2 or len(vb) < 2:
        return None
    ka, kb = (len(set(va)), len(set(vb)))
    lab = 'first' if ka > kb else 'second' if kb > ka else 'equal'
    return {'verb': 'compare', 'slots': list(a['slots']) + list(b['slots']), 'vals': va + vb, 'n_first': len(va), 'label': lab, 'S': a['S'], 'S2': b['S'], 'address': a['address'], 'address2': b['address']}

def lookup_question(pack, item, rng, hid=None):
    """286's question: hide the VALUE of one mention, keep the mention.

    The first version of this dropped the hidden slot from the graph entirely and still called
    itself "286's question, unchanged". It was not, and the difference is the whole task. With
    the row gone, the only evidence is the surviving counts, and under uniform hiding the
    posterior over the hidden value is proportional to (m_j + 1) - so argmax over remaining
    counts is BAYES-OPTIMAL and the majority rival is an unbeatable ceiling, not an opponent.
    Both arms of the depth ablation were then noise around that ceiling, one above it and one
    below, which is exactly what 0.365 and 0.279 against a 0.330 floor look like.

    Keeping the row restores the question 286 actually asks: the hidden mention's CONTEXT is
    the query, and the mind decides which candidate's mentions that context belongs with. That
    is the same query-versus-evidence rank channel 289a's blind pair runs on, and it is the
    only channel that carries information a counter does not already have.
    """
    slots = list(item['slots'])
    if len(slots) < 3:
        return None
    hid = rng.randrange(len(slots)) if hid is None else hid
    vals = [pack['tape'].values[s] for s in slots]
    cands = sorted(set(vals[:hid] + vals[hid + 1:]))
    if len(cands) < 2:
        return None
    if vals[hid] not in cands:
        return None
    keep = slots[:hid] + slots[hid + 1:]
    return {'verb': 'lookup', 'slots': keep + [slots[hid]], 'vals': [vals[i] for i in range(len(slots)) if i != hid] + [object()], 'cands': cands, 'label': cands.index(vals[hid]), 'S': item['S'], 'address': item['address'], 'hid': hid, 'query_row': len(keep)}
NEIGHBOURS = 0
REFUSE = False
GRAPH_CACHE = True
REFUSE_LABEL = '\x00REFUSE'

def addr_parts(address):
    """(anchor, relation) of an fp address, split exactly as pack_from_corpus splits it."""
    tail = address.split(':', 1)[-1]
    a, r = (tail.split('|', 1) + [''])[:2]
    return (a, r)

def neighbourhood(p, addr, k, routes=('anchor', 'rel', 'word')):
    """N(a): up to k addresses by shared anchor, k by shared relation, k by shared rare words.

    Three routes, unioned, no weighting - because a weighted blend would be three constants
    chosen by hand, and which route matters is the mind's decision, not the rule's. Each route
    is a discrete match on something the tape already wrote; nothing is a similarity someone
    picked.

    Deterministic: candidates arrive in tape order (address index) and the first k are taken, so
    the neighbourhood is a property of the tape and not of a draw. That matters for the same
    reason region views are deterministic - a neighbourhood that resamples would make every
    question a different question at every evaluation.

    The rare-word route reuses `postings`, the index build_graph already computes rarity from,
    and counts an address once per shared rare word, so an address that shares several beats one
    that shares one. Ties break on address index, again for determinism.
    """
    nb = p.setdefault('_nb', {})
    if (addr, k, routes) in nb:
        return nb[addr, k, routes]
    idx = p.get('_addr_index')
    if idx is None:
        by_anchor, by_rel, by_word, order = (defaultdict(list), defaultdict(list), defaultdict(list), {})
        med = p.get('_median')
        if med is None:
            lens = sorted((len(v) for v in p['postings'].values()))
            med = lens[len(lens) // 2] if lens else 1
            p['_median'] = med
        for i, it in enumerate(p['items']):
            order[it['address']] = i
            an, rl = addr_parts(it['address'])
            by_anchor[an].append(it['address'])
            if rl:
                by_rel[rl].append(it['address'])
            seen = set()
            for sl in it['slots']:
                for w in context_words(p['texts'][sl], exclude=p['tape'].values[sl]):
                    if len(p['postings'].get(w, ())) < med and w not in seen:
                        seen.add(w)
                        by_word[w].append(it['address'])
        idx = p['_addr_index'] = {'anchor': by_anchor, 'rel': by_rel, 'word': by_word, 'order': order, 'slots': {it['address']: it['slots'] for it in p['items']}}
    an, rl = addr_parts(addr)
    out, seen = ([], {addr})
    here = idx['order'].get(addr, 0)

    def take(cands, n):
        for c in cands[:n]:
            if c not in seen:
                seen.add(c)
                out.append(c)

    def near(cands):
        return sorted((a for a in cands if a != addr), key=lambda a: (abs(idx['order'][a] - here), idx['order'][a]))
    if 'anchor' in routes:
        take(near(idx['anchor'].get(an, ())), k)
    if 'rel' in routes:
        take(near(idx['rel'].get(rl, ())), k)
    hits = Counter()
    for sl in idx['slots'].get(addr, ()):
        for w in context_words(p['texts'][sl], exclude=p['tape'].values[sl]):
            if len(p['postings'].get(w, ())) < p['_median']:
                for a in idx['word'].get(w, ()):
                    if a != addr:
                        hits[a] += 1
    if 'word' in routes:
        take([a for a, _ in sorted(hits.items(), key=lambda kv: (-kv[1], idx['order'][kv[0]]))], k)
    nb[addr, k, routes] = out
    return out

def lookup_sparse_question(p, item, rng, hid, k, routes=('anchor', 'rel', 'word')):
    """The verb the current one throws away, and the one 1-NN cannot always answer.

    `lookup_question` needs three mentions, so at 2.85 mentions per address most of the
    distribution is discarded - 804 train addresses produced 265 questions. Addresses with one
    or two mentions are exactly where the neighbourhood has to do the work, and with ONE mention,
    hiding it leaves the address with no row of its own: 1-NN within the address is not worse,
    it is **undefined**. The answer can only come from N(a).

    Rows are the address's surviving mentions plus every row of N(a), kept in TAPE ORDER so a
    region view still cuts stretches of the corpus (§21) rather than "own rows" against
    "neighbour rows". The query row goes last, carrying the sentinel, exactly as the dense verb
    leaves it.

    Two outcomes, and which one a question gets is a property of the tape:

      the hidden value IS among N(a)'s values   answerable, label = its index
      it is NOT                                 unanswerable - the input genuinely does not
                                                determine the answer, and the only correct
                                                output is to say so (291, --refuse)

    The dense verb DROPS the second case ("target not a function of input", 286 failure mode
    12), and for a closed-set verb that is right. Here it is the signal: refuse IS a function of
    the input, and the tape supplies the label for free. Without --refuse those questions are
    dropped as before, so 290 and 291 differ in exactly one thing.
    """
    own = list(item['slots'])
    if not 1 <= len(own) <= 2:
        return None
    hid = hid % len(own)
    hidden_slot = own[hid]
    truth = p['tape'].values[hidden_slot]
    rows = [s for s in own if s != hidden_slot]
    for b in neighbourhood(p, item['address'], k, routes):
        rows += list(p['_addr_index']['slots'].get(b, ()))[:k]
    rows = sorted(set(rows) - {hidden_slot})
    if not rows:
        return None
    cands = sorted({p['tape'].values[s] for s in rows})
    if len(cands) < 2:
        return None
    answerable = truth in cands
    if not answerable and (not REFUSE):
        return None
    if REFUSE:
        cands = cands + [REFUSE_LABEL]
    label = cands.index(truth) if answerable else cands.index(REFUSE_LABEL) if REFUSE else None
    if label is None:
        return None
    return {'verb': 'lookup', 'sparse': True, 'answerable': answerable, 'slots': rows + [hidden_slot], 'vals': [p['tape'].values[s] for s in rows] + [object()], 'cands': cands, 'label': label, 'S': item['S'], 'address': item['address'], 'hid': hid, 'own_rows': {s for s in own if s != hidden_slot}, 'query_row': len(rows)}
OPEN = False
OBJECTIVE = 'ce'
OPEN_NEAR = [0, 0]

def lookup_open_question(p, item, rng, hid, by_anchor, all_values):
    """292. The hidden value is FOREIGN to the evidence, so retrieval inside the address cannot
    reach it and the mind has to say which world the corpus would have written.

    §19.6 is where the mind/retrieval question stops being statistical: 1-NN cannot produce a
    value that no row carries - not badly, at all. The dense verb never tests that, because its
    candidates are by construction the values already lying on the address's rows.

    The construction is the LADDER, and the ladder is finally legitimate here. In 289 it could
    not be settled: the true value lived at the address so it had nothing to import, every rung
    came from elsewhere and had everything, and Phi could read "imported rows present" as
    "wrong" - a bookkeeping tell, not distance. The fix was never a compensation, it was this
    question. Require the hidden value to occur EXACTLY ONCE at the address, so hiding it removes
    it from the evidence entirely; then the truth is as foreign as every rung, all four import
    the same number of rows through shared_import_budget, and the comparison is symmetric by
    construction rather than by adjustment.

    Two things are measured, and the second is the one that has never passed:

      accuracy over {true, near, middle, far}     floor 0.25, and no rival over the address's
                                                  own rows can score above it by construction
      the landscape, Phi(true) > near > mid > far the ordering a generator needs a direction in

    Candidates are sorted, so nothing about position says which is which.
    """
    own = list(item['slots'])
    if len(own) < 2:
        return None
    hid = hid % len(own)
    hidden_slot = own[hid]
    truth = p['tape'].values[hidden_slot]
    rows = [s for s in own if s != hidden_slot]
    if any((p['tape'].values[s] == truth for s in rows)):
        return None
    stub = {'cands': [truth], 'address': item['address'], 'slots': own, 'S': item['S'], 'query_row': len(rows)}
    attach_ladder(p, stub, by_anchor, all_values, rng)
    rungs = dict(stub.get('ladder') or {})
    if 'near' not in rungs:
        used = {truth} | set(rungs.values())
        for b in neighbourhood(p, item['address'], 3):
            cand = [p['tape'].values[sl] for sl in p['_addr_index']['slots'].get(b, ())]
            cand = [x for x in cand if x not in used]
            if cand:
                rungs['near'] = cand[0]
                break
    if len(rungs) != 3:
        return None
    OPEN_NEAR[0 if stub.get('ladder', {}).get('near') == rungs['near'] else 1] += 1
    cands = sorted([truth] + [rungs[r] for r in LADDER])
    if len(cands) != 4:
        return None
    q = {'verb': 'lookup', 'open': True, 'slots': rows + [hidden_slot], 'vals': [p['tape'].values[s] for s in rows] + [object()], 'cands': cands, 'label': cands.index(truth), 'rung_of': {rungs[r]: r for r in LADDER}, 'S': item['S'], 'address': item['address'], 'hid': hid, 'query_row': len(rows)}
    if shared_import_budget(p, q, list(q['cands'])) < 1:
        return None
    return q
ADDRESS_FROM = 'fp'
OPEN_CANDS = 'ladder'
OPEN_N_CANDS = 4
ANCHOR_MAX_ROWS = 8

def anchor_items(p):
    """Addresses made of one exact string. The grouping rule is not loosened here, it is absent:
    two mentions share an address when the corpus wrote the same anchor, and never otherwise."""
    ix = ident_index(p)
    if ix is None:
        return []
    return [{'S': anc, 'address': f'anc:{anc}', 'slots': list(slots), 'kind': 'clean'} for anc, slots in sorted(ix['by_anc'].items()) if len(slots) >= 2]

def lookup_open_uniform(p, item, rng, hid, all_values):
    """292's question, with the wrong answers drawn instead of constructed.

    Everything that made 292 legitimate is kept: the hidden value occurs exactly once at the
    address so it is foreign to every evidence row, and shared_import_budget gives all four
    worlds the same number of imported rows. What goes is the ladder - the three distractors are
    any values the address does not carry, so nothing about a candidate's relation to this
    address says whether it is the answer. The distance is recorded per candidate and read off
    the same logits afterwards, which is the only way to ask whether the inversion is real.
    """
    own = list(item['slots'])
    if len(own) < 2:
        return None
    hid = hid % len(own)
    hidden = own[hid]
    truth = p['tape'].values[hidden]
    rows = [s for s in own if s != hidden]
    if any((p['tape'].values[s] == truth for s in rows)):
        return None
    if len(rows) > ANCHOR_MAX_ROWS:
        rows = sorted(sorted(rows, key=lambda s: (abs(s - hidden), s))[:ANCHOR_MAX_ROWS])
    here = {p['tape'].values[s] for s in own}
    cands = [truth]
    for _ in range(64 * OPEN_N_CANDS):
        if len(cands) == OPEN_N_CANDS:
            break
        v = all_values[rng.randrange(len(all_values))]
        if v not in here and v not in cands:
            cands.append(v)
    if len(cands) != OPEN_N_CANDS:
        return None
    cands = sorted(cands)
    anc = item['S']
    ix = ident_index(p)
    at_anchor = {p['tape'].values[s] for s in ix['by_anc'].get(anc, ())} if ix else set()
    q = {'verb': 'lookup', 'open': True, 'uniform': True, 'slots': rows + [hidden], 'vals': [p['tape'].values[s] for s in rows] + [object()], 'cands': cands, 'label': cands.index(truth), 'bucket_of': {c: 'same_anchor' if c in at_anchor else 'elsewhere' for c in cands if c != truth}, 'S': anc, 'address': item['address'], 'hid': hid, 'query_row': len(rows)}
    if shared_import_budget(p, q, list(q['cands'])) < 1:
        return None
    return q
PATTERNS = False
PAT_W = 2

def pattern_stats(p, keep=None):
    """Exact co-occurrence counts over anchors. `keep` restricts to an anchor subset (the
    internal split), so train-side labels never touch the anchors the worlds are built from."""
    ix = ident_index(p)
    if ix is None:
        return None
    av = {a: {p['tape'].values[s] for s in sl} for a, sl in ix['by_anc'].items() if keep is None or keep(a)}
    cnt, pair = (Counter(), Counter())
    for a, vs in av.items():
        for v in vs:
            cnt[v] += 1
        for x in vs:
            for y in vs:
                if x < y:
                    pair[x, y] += 1
    return {'av': av, 'cnt': cnt, 'pair': pair, 'N': max(1, len(av)), 'ix': ix}

def pattern_lift(st, x, y):
    """N * P(xy) / (P(x) P(y)). None when either value never occurs on this side - no label."""
    if st['cnt'][x] == 0 or st['cnt'][y] == 0:
        return None
    return st['pair'].get((x, y), 0) * st['N'] / (st['cnt'][x] * st['cnt'][y])

def pattern_rules(st):
    """Every pair witnessed by at least PAT_W anchors: enough rows for a fixed-size world."""
    return [xy for xy, nab in sorted(st['pair'].items()) if nab >= PAT_W]

def pattern_world(p, st, x, y):
    """The rule as rows: from the first PAT_W witness anchors (lexicographic - independent of
    any statistic), one x-row and one y-row each. 2*PAT_W rows for every rule, so the row count
    cannot carry the label; no query row; values visible - the same-value edges ARE the shape
    the rule claims. Phi sees this world and nothing else: no support, no confidence, no lift."""
    wit = sorted((a for a, vs in st['av'].items() if x in vs and y in vs))[:PAT_W]
    slots = []
    for a in wit:
        sl = st['ix']['by_anc'][a]
        slots.append(min((s for s in sl if p['tape'].values[s] == x)))
        slots.append(min((s for s in sl if p['tape'].values[s] == y)))
    slots = sorted(set(slots))
    if len(slots) != 2 * PAT_W:
        return None
    return {'verb': 'lookup', 'slots': slots, 'vals': [p['tape'].values[s] for s in slots], 'S': x, 'query_row': -1, 'cands': [x, y], 'label': 0}

def run_patterns(p, held, bank, device, args):
    """Mine on train, judge with Phi, label on held, race the rule's own train statistics."""
    import torch.nn.functional as Fn
    log(f'  295 patterns: witnesses {PAT_W}')

    def side(a):
        return int.from_bytes(hashlib.blake2b(a.encode(), digest_size=2).digest(), 'big') % 2
    stA = pattern_stats(p, keep=lambda a: side(a) == 0)
    stB = pattern_stats(p, keep=lambda a: side(a) == 1)
    st_all, st_held = (pattern_stats(p), pattern_stats(held))
    if not (stA and stB and st_all and st_held):
        log('  295 needs straddr packs')
        return 1

    def labelled(st_mine, st_lab, src):
        out = []
        for x, y in pattern_rules(st_mine):
            lf = pattern_lift(st_lab, x, y)
            if lf is None:
                continue
            w = pattern_world(src, st_mine, x, y)
            if w is not None:
                out.append((w, int(lf > 1.0), pattern_lift(st_mine, x, y)))
        return out
    train_set = labelled(stA, stB, p)
    eval_set = labelled(st_all, st_held, p)
    log(f'  rules: train {len(train_set)} eval {len(eval_set)} | base survive: train {sum((l for _, l, _ in train_set)) / max(1, len(train_set)):.3f} eval {sum((l for _, l, _ in eval_set)) / max(1, len(eval_set)):.3f}')
    if not train_set or not eval_set:
        log('  295: empty rule set - need more addresses or lower co-occurrence bar')
        return 1
    net = Deriver(device, d=args.dim, n_node=8 + (1 if TAPE == 'frames' else 0) + (1 if REACH_HOME_COS else 0) + (3 if REACH_CHANNEL else 0))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(SEED)
    steps = args.train_steps or 3000
    for step in range(steps):
        w, lab, _ = train_set[rng.randrange(len(train_set))]
        E, sm, nf = build_graph(p, w, bank, device)
        loss = Fn.binary_cross_entropy_with_logits(net.phi(E, sm, nf), torch.tensor(float(lab), device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    cands = sorted({lf for _, _, lf in train_set})
    thr = max(cands, key=lambda t: sum((int(lf >= t) == l for _, l, lf in train_set))) if cands else 1.0
    b10 = b01 = ok_m = ok_r = 0
    with torch.no_grad():
        for w, lab, lf in eval_set:
            E, sm, nf = build_graph(p, w, bank, device)
            m = int(float(net.phi(E, sm, nf)) > 0.0)
            r = int(lf >= thr)
            ok_m += int(m == lab)
            ok_r += int(r == lab)
            b10 += int(m == lab and r != lab)
            b01 += int(r == lab and m != lab)
    d = b10 + b01
    out = {'stage': '295', 'witnesses': PAT_W, 'rules_train': len(train_set), 'rules_eval': len(eval_set), 'rival_threshold': thr, 'base_survive': sum((l for _, l, _ in eval_set)) / max(1, len(eval_set)), 'phi_accuracy': ok_m / max(1, len(eval_set)), 'rival_accuracy': ok_r / max(1, len(eval_set)), 'paired': {'model_only_right': b10, 'rival_only_right': b01, 'discordant': d, 'mcnemar_z': (b10 - b01) / math.sqrt(d) if d else float('nan'), 'max_achievable_z': math.sqrt(d) if d else 0.0}, 'seed': SEED, 'timestamp': datetime.now(timezone.utc).isoformat()}
    log(f'  295 {json.dumps(out)}')
    tag = '_' + args.run_tag if args.run_tag else ''
    (RES / f'stage289_patterns{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    return 0
MIXED = False
TAPE = 'parser'
ROUTE = False
STEP_COST = 0.05
REACH_GAMMA = 1.0
SHUFFLE_TAPE = False
CONSTRAIN = False
CONS_LENSES = 6
CONS_TOPM = 8
TWO_WAY_BY = 'max'
CONS_RESOLVE = 'count'
SPEAK_BATCH = 0
SPEAK_WEIGHT = 1.0
_SPEAK_ACC = None
CALIB_BATCH = 0
CALIB_WEIGHT = 1.0
_CALIB_ACC = None
OTHER_NET = None
RETAIN = 0
RETAIN_BY = 'random'
RETAIN_CTX = None
_RETAIN_BUSY = False
COHERENCE = 0
DEEP_ROOT = 'mind'
REACH_COMPASS = 'cos'
REACH_DEPTH = 1
BISECT = False
FINETUNE = False
STAGE2_ALWAYS = 0.0
TWO_WAY = False
EQUAL_TAILS = False
FRAME_MAX = 12
OWN_IN_OFFER = False
OWN_IMPORT = False
_OWN_IMPORT_N = [0, 0]
CONNECT = False
CONNECT_MAX = 4000
COPY = False
COPY_D = 4
COPY_BACKFILL = False
REACH_CHANNEL = False
MOVES_ON = False
MOVES = ('step', 'share', 'lines')
MOVE_TEACH = 0.0
_MOVE_LIVE = Counter()
ROUTE_ON = 'all'
_ROUTE_LIVE = Counter()
MOVE_ALL = ('step', 'share', 'lines')
MIN_FILLERS = 2

def expanded_world(p, q, bank, device, k):
    """What the mind would be looking at if it read on: the question's rows plus everything the
    tape offers about every option. Candidate-independent on purpose - it is ONE world, so Phi
    can score `read more` with the same scalar it scores an answer with, and the route needs no
    second head and no policy network."""
    extra = []
    for c in q['cands']:
        if c == REFUSE_LABEL:
            continue
        extra += outside_mentions(p, q, c)[:k]
    w = dict(q)
    w.pop('_base', None)
    w.pop('_ibudget', None)
    w['slots'] = list(q['slots']) + extra
    w['vals'] = list(q['vals']) + [p['tape'].values[s] for s in extra]
    return build_graph(p, w, bank, device, query_value=None, import_k=0)

def route_logits(net, p, q, device, bank):
    """The route, unrolled once, as one softmax over worlds.

    Stage 1 imports NOTHING, and that is the point rather than an economy: with no imports an
    absent value gives a bit-identical graph, so the candidates are genuinely indistinguishable
    and the honest move is to read more. The degeneracy that wrecked 289's ladder becomes the
    reason the step exists. Stage 2 is the same question with the tape's own word on each
    option brought in.

    `expand` is not an action of a different kind - it is one more world, scored by the same
    Phi. So there is no policy head, no sampling, and the whole two-step decision stays
    differentiable through the scalar the project already has.
    """
    k = shared_import_budget(p, q, list(q['cands']))
    l1 = torch.stack([net.phi(*build_graph(p, q, bank, device, query_value=c, import_k=0)) for c in q['cands']])
    q.pop('_base', None)
    l_exp = net.phi(*expanded_world(p, q, bank, device, k))
    q.pop('_base', None)
    l2 = torch.stack([net.phi(*build_graph(p, q, bank, device, query_value=c, import_k=k)) for c in q['cands']])
    return (torch.cat([l1, l_exp.reshape(1)]), l2)

def route_reward(q, device):
    R = torch.full((len(q['cands']),), -1.0, device=device)
    R[q['label']] = 1.0
    if q.get('answerable') and REFUSE_LABEL in q['cands']:
        R[q['cands'].index(REFUSE_LABEL)] = 0.75
    return R

def route_loss(net, p, q, device, bank):
    """Expected payoff of the whole route, in closed form: answer now, or pay the step price and
    answer from what reading brought. No baseline, no sampling, no RL machinery."""
    l1, l2 = route_logits(net, p, q, device, bank)
    R = route_reward(q, device)
    p1, p2 = (torch.softmax(l1, 0), torch.softmax(l2, 0))
    v2 = (p2 * R).sum() - STEP_COST
    return -((p1[:-1] * R).sum() + p1[-1] * v2)
FRAME_POOL = [0]
REACH = False
REACH_K = 8
REACH_CANDS = 8
REACH_MAX_Q = 2000
REACH_MAX_ROWS = 12
HOME_COS_STAGE = 'both'
REACH_COLS = ('answerable', 'silent', 'mind_right', 'rival_margin', 'rival_right', 'stepped', 'n_cands', 'rows_candidate', 'rows_expand', 'reachable_wide', 'reachable_random', 'truth_in_own', 'own_rival_right', 'n_own', 'max_own_count', 'n_places', 'line_reach', 'line_rival', 'step_line', 'world_rows_own', 'count_rival_right', 'top_share', 'bisect_right', 'bisect_splits', 'depth_reached', 'deep_only', 'other_right', 'other_stepped', 'pick_score', 'pick_margin', 'cr_ties', 'move_id')
RIX = {n: i for i, n in enumerate(REACH_COLS)}
REACH_CONFIRM = False
CONF_DF = 3
CONF_HOMES = 8
CONF_WINDOW = 0
REACH_LINE = False
REACH_HOME_COS = False
REACH_IMPORT = 'walk'
TAPE_SAMPLE = 'uniform'
FRAME_FP = 'address'
REACH_LOOKAHEAD = False
REACH_NO_REFUSE = False
REACH_ROWS_PER_VALUE = 4

def reach_index(p):
    """Every address as one frame fingerprint, stacked once per pack."""
    ix = p.get('_reach')
    if ix is not None:
        return ix
    fps = p.get('frame_fps')
    items = [it for it in p['items'] if it['slots']]
    if fps is None or not items:
        return None
    M = torch.stack([fps[it['slots'][0]] for it in items])
    fills = []
    for it in items:
        seen, order = ({}, [])
        for sl in it['slots']:
            v = p['tape'].values[sl]
            if v not in seen:
                seen[v] = [[], 0]
                order.append(v)
            e = seen[v]
            e[1] += 1
            if len(e[0]) < REACH_ROWS_PER_VALUE:
                e[0].append(sl)
        fills.append([(v, seen[v][0], seen[v][1]) for v in order])
    hs, hn = ({}, Counter())
    if REACH_HOME_COS:
        for sl, ad in enumerate(p['straddr']):
            v = p['tape'].values[sl]
            f = fps[sl]
            hs[v] = f.clone() if v not in hs else hs[v] + f
            hn[v] += 1
    by_val = defaultdict(list)
    for j, fl in enumerate(fills):
        for v, _rows, c in fl:
            by_val[v].append((j, c))
    ix = {'items': items, 'M': M, 'fills': fills, 'home_sum': hs, 'home_n': hn, 'by_val': dict(by_val), 'of': {it['address']: i for i, it in enumerate(items)}}
    p['_reach'] = ix
    return ix

def reach_question(p, item, rng, hid):
    """A hidden filler and the rows of its own place. No candidates: those are walked to."""
    own = list(item['slots'])
    if len(own) < 2:
        return None
    hid %= len(own)
    qs = own[hid]
    rows = [s for s in own if s != qs]
    if len(rows) > REACH_MAX_ROWS - 1:
        rows = sorted(rng.sample(rows, REACH_MAX_ROWS - 1))
    return {'verb': 'reach', 'reach': True, 'S': item['S'], 'address': item['address'], 'hid': hid, 'slots': rows + [qs], 'vals': [p['tape'].values[s] for s in rows] + [object()], 'query_row': len(rows), 'truth_value': p['tape'].values[qs]}

def retain_keep(p):
    """338: WHICH PLACES THE TAPE KEEPS - chosen once per pack, by counting or by the mind.

    WHY THIS EXISTS. 335 measured the tape holding 0.807 of the truths and the offer showing
    0.217 of them, and the width sweep showed the gap OPENING as the tape grows: 0.369 -> 0.590
    over a 16x range, with the share shown falling. More corpus is not the lever. What is left
    is retention - if only some places can be kept, which ones - and that is a decision, which
    is the mind's half of the split rather than the tape's.

    WHAT IS MEASURED, and it is deliberately not "does the retained tape answer its own
    questions". Retaining the fattest places would trivially win that, because it would also
    change WHICH QUESTIONS EXIST. So: questions are drawn from the WHOLE tape, unchanged, and
    retention decides only which places the WALK MAY VISIT AND OFFER FROM. The question's own
    rows are untouched. Every rule is then compared at the same number of retained places -
    matched budget, as everywhere else in this project.

    THE RULES:
      random  what the tape does today, and the control every other rule must beat
      own     the places with the most mentions. The obvious count.
      share   the places whose single most frequent filler dominates them, as a share of the
              place's mentions. The other obvious count, and not the same one: `own` prefers
              big places, `share` prefers decided ones.
      mind    the places where Phi, asked a question there, answers with the widest margin.

    THE INVARIANT IS AT RISK HERE AND THE GUARD IS IN THE CODE. A retention policy that has to
    be fitted to a corpus would be knowledge leaking into the deciding half. So `mind` REFUSES
    to run on a mind that is training: it requires --load-mind and a frozen Phi, and the test
    that decides the idea is the transplant - choose the places on news with a mind fitted only
    to wiki. If the foreign tape comes out no worse than the counting rules build it, the
    policy is not corpus-specific. If it is worse, the idea is dropped whole. There is no
    rescuing patch for this one; that was written down before the first run.
    """
    global _RETAIN_BUSY
    if not RETAIN or _RETAIN_BUSY:
        return None
    m = p.get('_retain')
    if m is not None:
        return m
    ix = reach_index(p)
    if ix is None:
        return None
    items = ix['items']
    P = len(items)
    if RETAIN >= P:
        p['_retain'] = torch.ones(P, dtype=torch.bool)
        return p['_retain']
    rng = random.Random(SEED + 9338)
    if RETAIN_BY == 'random':
        sc = [rng.random() for _ in range(P)]
    elif RETAIN_BY == 'own':
        sc = [float(len(it['slots'])) for it in items]
    elif RETAIN_BY == 'share':
        sc = []
        for fl in ix['fills']:
            tot = sum((c for _v, _r, c in fl))
            sc.append(max((c for _v, _r, c in fl), default=0) / max(1, tot))
    else:
        if RETAIN_CTX is None:
            raise RuntimeError('--retain-by mind: the walk ran before the mind existed')
        net, device, bank = RETAIN_CTX
        _RETAIN_BUSY = True
        try:
            sc = []
            qr = random.Random(SEED + 9339)
            with torch.no_grad():
                for it in items:
                    q = reach_question(p, it, qr, 0)
                    if q is None:
                        sc.append(float('-inf'))
                        continue
                    l1, l2, own, cands, l3, lcands = reach_logits(net, p, q, device, bank)
                    _s, _st, _d, _score, mg = reach_pick(q, l1, l2, own, cands, l3, lcands)
                    sc.append(mg)
        finally:
            _RETAIN_BUSY = False
    order = sorted(range(P), key=lambda i: (-sc[i], i))[:RETAIN]
    m = torch.zeros(P, dtype=torch.bool)
    m[torch.tensor(order, dtype=torch.long)] = True
    p['_retain'] = m
    return m

def reach_places(p, q, k):
    """The k nearest places by frame fingerprint, in order, this one excluded."""
    ix = reach_index(p)
    i = ix['of'].get(q['address']) if ix else None
    if i is None:
        return []
    qv = ix['M'][i]
    if FRAME_FP == 'fillers' and p.get('frame_sum') is not None:
        ad = q['address']
        n_ad = p['frame_cnt'].get(ad, 0)
        vf = p['val_fp'].get(q['truth_value'])
        if n_ad > 1 and vf is not None:
            qv = F.normalize(p['frame_sum'][ad] - vf, dim=-1)
    q['_qv'] = qv
    sims = (ix['M'] @ qv).clone()
    sims[i] = -2.0
    kp = retain_keep(p)
    if kp is not None:
        sims = sims.masked_fill(~kp.to(sims.device), -2.0)
    order = [j for j in sims.argsort(descending=True)[:k].tolist() if sims[j] > -2.0]
    out = [(j, ix['items'][j], float(sims[j])) for j in order]
    if REACH_COMPASS == 'cos':
        return out
    ownc = Counter((p['tape'].values[s] for s in q['slots'][:q['query_row']]))
    share = Counter()
    if REACH_COMPASS in ('share1', 'rare', 'common', 'cover', 'jaccard'):
        n_own = len(ownc)
        for v, cnt in ownc.items():
            hits = ix['by_val'].get(v, ())
            tot = sum((c for _j, c in hits)) or 1
            for j, c in hits:
                if j == i or (kp is not None and (not bool(kp[j]))):
                    continue
                if REACH_COMPASS == 'share1':
                    share[j] += 1
                elif REACH_COMPASS == 'rare':
                    share[j] += 1.0 / tot
                elif REACH_COMPASS == 'common':
                    share[j] += float(tot)
                elif REACH_COMPASS == 'cover':
                    share[j] += 1.0 / max(1, len(ix['fills'][j]))
                else:
                    share[j] += 1.0 / max(1, n_own + len(ix['fills'][j]) - 1)
    else:
        for v, cnt in ownc.items():
            for j, c in ix['by_val'].get(v, ()):
                if j != i and (kp is None or bool(kp[j])):
                    share[j] += min(cnt, c)
    sh = [(j, ix['items'][j], float(n)) for j, n in share.most_common(k)]
    if REACH_COMPASS != 'both':
        return sh
    seen, mixed = (set(), [])
    for a, b in zip_longest(out, sh):
        for e in (a, b):
            if e is not None and e[0] not in seen:
                seen.add(e[0])
                mixed.append(e)
    return mixed[:2 * k]

def reach_connect(p, q, k):
    """365: THE THIRD CHANNEL. Values that stand at places RELATED TO THIS ONE, ranked by how
    related those places are.

    The tape has had two channels since the reach verb existed - `own` (what stood at this hole
    before: recall) and the walk's `cands` (what stands at holes with a similar fingerprint:
    substitution). 363 measured a third on holes NEITHER reaches, which are 48% of questions at
    --min-fillers 1 and where every number this project has printed is zero:

        neighbourhood   places that share at least one filler with this place
        overlap         HOW MANY fillers each shares - a count, and the weight
        score(v)        sum of the overlaps of the neighbourhood places holding v

    It reached 0.0975 of those holes at top-8 out of 2425 candidates, ~30x chance, against a
    wrong-place null and against frequency (AUC 0.692, over-null +0.152, over-freq +0.092).
    The overlap WEIGHT beat the plain count at both window sizes; the STRICT form - a neighbour
    must share two fillers - lost badly (5.6x chance against 28x), because single-filler
    neighbours carry the signal. That closure is why this is weighted and not thresholded.

    THE QUESTION'S OWN PLACE IS EXCLUDED, and so are its own values: recall already covers them
    and a channel that hands the question its own row back is not a channel. Retention is
    honoured exactly as `reach_places` honours it - a dropped place is not there.
    """
    ix = reach_index(p)
    i = ix['of'].get(q['address']) if ix else None
    if i is None:
        return []
    own = Counter((p['tape'].values[s] for s in q['slots'][:q['query_row']]))
    kp = retain_keep(p)
    overlap = Counter()
    for v in own:
        for j, _c in ix['by_val'].get(v, ()):
            if j != i and (kp is None or bool(kp[j])):
                overlap[j] += 1
    if not overlap:
        return []
    score, rows_of = (Counter(), {})
    for j, ov in overlap.most_common(CONNECT_MAX):
        for v, rows, _c in ix['fills'][j]:
            if v in own:
                continue
            score[v] += ov
            if len(rows_of.setdefault(v, [])) < REACH_ROWS_PER_VALUE:
                rows_of[v].extend(rows[:REACH_ROWS_PER_VALUE - len(rows_of[v])])
    return [(v, rows_of.get(v, []), n) for v, n in score.most_common(k)]

def reach_reachable(p, q, k, rand_rng=None):
    """Would the true filler be sayable if the walk went `k` places - or `k` random ones?

    Set membership only, no graphs, so the three variants cost nothing next to the run. The
    random one is the decisive one: if a cosine walk reaches the truth no more often than an
    arbitrary handful of places does, there is no direction in this ink and the verb is
    measuring a lottery. That has to be known before any payoff is read, not after.
    """
    ix = reach_index(p)
    if ix is None:
        return False
    if rand_rng is None:
        js = [j for j, _it, _s in reach_places(p, q, k)]
    else:
        kp = retain_keep(p)
        pool = [j for j, it in enumerate(ix['items']) if it['address'] != q['address'] and (kp is None or bool(kp[j]))]
        js = rand_rng.sample(pool, min(k, len(pool)))
    seen, out = (set(), [])
    for j in js:
        for v, _rows, _c in ix['fills'][j]:
            if v not in seen:
                seen.add(v)
                out.append(v)
    return q['truth_value'] in set(out[:REACH_CANDS])

def reach_candidates(p, q, k=None, which=None):
    """What the walk makes sayable: fillers in place order, deduped, capped by the cost bound.

    Cached on the question so the walk is one object and the rival, the mind and the report all
    grade the SAME traversal rather than three re-derivations of it.

    385: `which` names ONE move's lane instead of the merged offer, and is passed only while the
    mind is choosing between moves - those calls are not cached, because they are proposals and
    not the question's offer. Once the move is chosen it is written to q["_move"] and the
    ordinary cached call builds that lane and only it.
    """
    if which is None and '_reach_c' in q:
        return q['_reach_c']
    if which is None:
        which = q.get('_move', 'all')
        if MOVES_ON and '_move' not in q:
            raise RuntimeError('385: reach_candidates called before the move was chosen')
    ix = reach_index(p)
    places = reach_places(p, q, REACH_K if k is None else k)
    seen, cands, rows_of, from_place = (set(), [], {}, {})
    for j, _it, _sim in places:
        for v, rows, _c in ix['fills'][j]:
            rows_of.setdefault(v, []).extend(rows)
            if v not in seen:
                seen.add(v)
                cands.append(v)
                from_place[v] = j
    if which != 'all':
        if which == 'step':
            conn, cop = ([], [])
        elif which == 'share':
            conn, cop = (reach_connect(p, q, REACH_CANDS), [])
            cands = [v for v, _r, _n in conn]
        elif which == 'lines':
            conn, cop = ([], reach_copy(p, q, REACH_CANDS))
            cands = [v for v, _r, _n in cop]
        else:
            raise RuntimeError(f'385: unknown move {which!r}')
        for v, rows, _n in conn:
            rows_of[v], from_place[v] = (list(rows), -1)
        for v, rows, _n in cop:
            rows_of[v], from_place[v] = (list(rows), -3)
    elif CONNECT or OWN_IN_OFFER or COPY:
        lanes = [cands]
        if CONNECT:
            conn = reach_connect(p, q, REACH_CANDS)
            lanes.append([v for v, _r, _n in conn])
        else:
            conn = []
        cop = reach_copy(p, q, REACH_CANDS) if COPY else []
        if cop and (not COPY_BACKFILL):
            lanes.append([v for v, _r, _n in cop])
        own_vals = sorted({p['tape'].values[s] for s in q['slots'][:q['query_row']]})
        if OWN_IN_OFFER:
            lanes.append(own_vals)
        seen2, mixed = (set(), [])
        for tup in zip_longest(*lanes):
            for e in tup:
                if e is not None and e not in seen2:
                    seen2.add(e)
                    mixed.append(e)
        if cop and COPY_BACKFILL:
            for v, _rows, _n in cop:
                if len(mixed) >= REACH_CANDS:
                    break
                if v not in seen2:
                    seen2.add(v)
                    mixed.append(v)
        for v, rows, _n in conn:
            if v in seen2 and v not in rows_of:
                rows_of[v] = list(rows)
                from_place[v] = -1
        for v, rows, _n in cop:
            if v in seen2 and v not in rows_of:
                rows_of[v] = list(rows)
                from_place[v] = -3
        if OWN_IN_OFFER:
            for v in own_vals:
                if v in seen2 and v not in rows_of:
                    rows_of[v] = list(outside_mentions(p, q, v))
                    from_place[v] = -2
        cands = mixed
    cands = cands[:REACH_CANDS]
    rows_of = {c: rows_of[c] for c in cands}
    real_place = {c: from_place[c] if from_place[c] >= 0 else place_of_rows(p, q, rows_of[c]) for c in cands}
    out = {'cands': cands, 'rows_of': rows_of, 'places': places, 'n_places': len({j for j in (real_place[c] for c in cands) if j is not None}), 'from_place': {c: from_place[c] for c in cands}, 'real_place': real_place, 'own': sorted({p['tape'].values[s] for s in q['slots'][:q['query_row']]})}
    if which == q.get('_move', 'all'):
        q['_reach_c'] = out
    return out

def reach_channel(p, q, value):
    """379: WHICH CHANNEL OFFERED THIS CANDIDATE - three indicators on the answered row.

    377 and 378 are an A/B on the merge rule and together they say the rule is the wrong object
    to be tuning. Interleaving the copy lane bought reach +0.060 on 4/4 seeds and hit +0.021 on
    3/4; making it a backfill - so that it can never displace a walked candidate - collapsed the
    supply to +0.006 and lost hit on 3/4. The copy candidates are therefore WORTH MORE than the
    walked ones they evicted, and the eviction was not the channel's cost but the channel
    itself. Neither allocation is a decision: round-robin and backfill are both CONSTANTS, and
    which one a hole wants plainly differs by hole.

    Phi cannot make that decision today for a simple reason - IT CANNOT SEE WHERE A CANDIDATE
    CAME FROM. `from_place` has recorded the channel since 365 (-1 connect, -2 home, -3 copy)
    and only ever fed `n_places`. So the offer stays exactly as 377 built it, the head stays
    one, the budget stays one, and the single new thing is that each candidate now carries its
    provenance into the same softmax that already ranks it. One objective, no second term - 321,
    341 and 352 each measured that a second objective costs about 4x the route.

    THREE INDICATORS AND NOT ONE NUMBER, because a channel is a category: a single scalar would
    impose an ordering (connect < home < copy) that nothing measured. The walk is the all-zero
    baseline, so an arm without the lever is bit-for-bit the arm before it.

    NO LEAK: every candidate in the question carries a channel and the value hidden at the hole
    is not one of the inputs. In stage one the flag is all zeros for every world, but the reason
    is narrower than "they are all home worlds" and worth writing down: `reach_connect` and
    `reach_copy` both drop own values by construction (`if v in own: continue`), so a home value
    cannot carry a negative from_place. Under --own-in-offer it can (-2), and then stage one is
    no longer constant - which is a property of that arm, not of this feature, and it is not in
    the standing arm.
    """
    if not REACH_CHANNEL:
        return (0.0, 0.0, 0.0)
    rc = q.get('_reach_c')
    j = rc['from_place'].get(value) if rc else None
    if j is None or j >= 0:
        return (0.0, 0.0, 0.0)
    return {-1: (1.0, 0.0, 0.0), -2: (0.0, 1.0, 0.0), -3: (0.0, 0.0, 1.0)}[j]

def channel_feat(q, i, qrow):
    """the three indicators as a node-vector tail, carried only by the row the world answers -
    the channel is a property of what the world asserts, not of the evidence around it."""
    if not REACH_CHANNEL:
        return []
    return list(q.get('channel') or (0.0, 0.0, 0.0)) if i == qrow else [0.0, 0.0, 0.0]

def reach_home_cos(p, q, value, qv):
    """cos(where this value usually stands, this place) - with this place taken out of the
    average for EVERY candidate, so the subtraction carries no news about which was hidden."""
    ix = p.get('_reach')
    if not REACH_HOME_COS or ix is None or (not ix['home_sum']) or (qv is None):
        return 0.0
    hs = ix['home_sum'].get(value)
    if hs is None:
        return 0.0
    here = ix['of'].get(q['address'])
    if here is None:
        return 0.0
    at_place = sum((c for v, _r, c in ix['fills'][here] if v == value))
    n = ix['home_n'][value] - at_place
    if n <= 0:
        return 0.0
    vec = hs - at_place * ix['M'][here]
    return float(F.normalize(vec, dim=-1) @ qv)

def confirm_index(p):
    """Rare words per line, counted once per pack. Rarity is a document frequency - a count."""
    ix = p.get('_conf')
    if ix is not None:
        return ix
    texts = {}
    for sl, li in enumerate(p.get('line') or ()):
        if li >= 0 and li not in texts:
            texts[li] = p['texts'][sl]
    df = Counter()
    for t in texts.values():
        for w in set(t.split()):
            df[w] += 1
    ix = {li: frozenset((w for w in set(t.split()) if df[w] <= CONF_DF)) for li, t in texts.items()}
    p['_conf'] = ix
    return ix

def reach_confirm(p, q, value):
    """How many RARE words of this question's line also stand around this value elsewhere.

    Kostya's channel, and 305 says it is there: the truth's home lines share 1.67x more rare
    words with the question's line than a wrong candidate's do, separating 65% of the differing
    pairs against 50% by chance. The `rare` EDGE channel was meant to carry this and never did -
    rare_nonzero_rate 0.003 in every run - because it compares rows INSIDE one world, and the
    signal lives between the question's line and the candidate's homes ELSEWHERE.

    THE LEAK, closed: the hidden mention sits on the question's own line, and a line shares every
    rare word with itself. Home lines equal to the question's are dropped for EVERY candidate
    alike, so nothing here needs to know which value was hidden.
    """
    if not REACH_CONFIRM or not p.get('line'):
        return 0.0
    ix = confirm_index(p)
    qli = p['line'][q['slots'][q['query_row']]]
    here = ix.get(qli)
    if not here:
        return 0.0
    best = 0
    for sl in outside_mentions(p, q, value)[:CONF_HOMES]:
        li = p['line'][sl]
        if li < 0 or li == qli:
            continue
        for d in range(-CONF_WINDOW, CONF_WINDOW + 1):
            other = ix.get(li + d)
            if other:
                best = max(best, len(here & other))
    return math.log1p(best) / math.log1p(8.0)

def reach_line_index(p):
    """Slots by line, once per pack."""
    ix = p.get('_lineix')
    if ix is None:
        ix = defaultdict(list)
        for sl, li in enumerate(p.get('line') or ()):
            if li >= 0:
                ix[li].append(sl)
        ix = dict(ix)
        p['_lineix'] = ix
    return ix

def reach_line_candidates(p, q):
    """What the OTHER frames of this hole's sentence offer, in line order, deduped and capped.

    The hidden row's own place is excluded, so nothing here is the question's own evidence, and
    positions partition the corpus so a sibling is always another word.
    """
    if '_line_c' in q:
        return q['_line_c']
    cands, rows_of = ([], {})
    lines = p.get('line')
    if lines:
        qs = q['slots'][q['query_row']]
        for sl in reach_line_index(p).get(lines[qs], ()):
            if p['straddr'][sl] == q['address'] or sl in set(q['slots']):
                continue
            v = p['tape'].values[sl]
            if v not in rows_of:
                if len(cands) >= REACH_CANDS:
                    continue
                rows_of[v] = []
                cands.append(v)
            if len(rows_of[v]) < REACH_ROWS_PER_VALUE:
                rows_of[v].append(sl)
    out = {'cands': cands, 'rows_of': rows_of}
    q['_line_c'] = out
    return out

def copy_index(p):
    """line -> its text, once per pack. `texts` is the whole line, so this reaches every token
    of a neighbouring sentence and not only the ones the frame cutter kept."""
    ix = p.get('_copyix')
    if ix is None:
        ix = {}
        for sl, li in enumerate(p.get('line') or ()):
            if li >= 0 and li not in ix:
                ix[li] = p['texts'][sl]
        p['_copyix'] = ix
    return ix

def reach_copy(p, q, k):
    """376: THE FOURTH CHANNEL - the truth is often STANDING NEXT DOOR, and reading never looked.

    375 closed the address atom: spelling frames in form-classes raised coverage by +0.5..0.77
    and moved the unreachable share not at all, so the wall is material - the missing truths
    barely repeat inside the window and a count cannot reach what was counted once. But "counted
    once at a place" is not "absent from the page". An article repeats its subject in running
    text constantly, and 376 measured it on exactly the holes where every number this project
    prints is zero:

        en w400   D=4    copy .538  null .186  +0.352
        en w1600  D=16   copy .679  null .293  +0.386
        de w8000  D=4    copy .120  null .051  +0.069

    The german row is the channel's own limit written down: news does not repeat its subject the
    way an encyclopedia article does, so this pays on DOCUMENTS, not on any text.

    TWO THINGS 376 DID NOT SAY, and they bound what to expect here:
      - it tested PRESENCE, not choice. A +-4 line window holds a few hundred tokens against an
        offer of eight, so presence .538 is not hit .538. What arrives here is a ranked lane.
      - it tested ALL tokens of the neighbourhood. A token that stands on no place has no rows,
        so it cannot be a world; the lane keeps only tape values, which is fewer.

    THE QUESTION'S OWN LINE IS DROPPED WHOLE. 376 subtracted the question's own standing by one
    count, which is enough for a presence audit and NOT enough here: the hidden value sits on
    that line and a lane reading its text would recover the answer from the sentence it was
    hidden in. Dropping the line removes the leak by construction rather than by bookkeeping,
    and costs only the D=0 term, which was never the best D anywhere.

    Rank is a count: how many times the value stands in the neighbourhood, nearest line first
    on a tie. Values already at this hole are the recall channel's and are excluded.
    """
    if not COPY or not p.get('line'):
        return []
    ix = copy_index(p)
    qli = p['line'][q['slots'][q['query_row']]]
    if qli < 0:
        return []
    bv = by_value(p)
    own = {p['tape'].values[s] for s in q['slots'][:q['query_row']]}
    score, dist = (Counter(), {})
    for delta in range(-COPY_D, COPY_D + 1):
        if delta == 0:
            continue
        text = ix.get(qli + delta)
        if not text:
            continue
        for w in text.split():
            if w in own or w not in bv:
                continue
            score[w] += 1
            if abs(delta) < dist.get(w, COPY_D + 1):
                dist[w] = abs(delta)
    if not score:
        return []
    out = []
    for w in sorted(score, key=lambda v: (-score[v], dist[v], v)):
        rows = outside_mentions(p, q, w)
        if not rows:
            continue
        out.append((w, rows, score[w]))
        if len(out) >= k:
            break
    return out

def reach_line_rival(p, q):
    """The same sentence read by counting: its most frequent sibling filler."""
    lc = reach_line_candidates(p, q)
    if not lc['cands']:
        return None
    return max(lc['cands'], key=lambda v: (len(lc['rows_of'][v]), -lc['cands'].index(v)))

def reach_relation_rows(p, q, value):
    """372b: THE RELATION AS EVIDENCE ON THE CANDIDATE, which is what 371 actually measured.

    371 said: on the questions recall cannot answer, choosing among relations beats random
    places (0.1725 vs 0.0901). It did NOT say to stop walking by the fingerprint. 372a read it
    as "replace the compass" and lost on every member - reach fell WITH hit and cand_places
    collapsed from 4.35 to about 1.6, so the sharing graph is simply thinner than the
    fingerprint, not a different-but-equal route.

    So the walk stays `cos` and nothing about the offer changes. What changes is WHICH MENTION
    a candidate brings as its evidence: instead of the first rows the walk happened to pass, or
    its homes anywhere on the tape, the candidate brings its mentions AT PLACES RELATED TO THIS
    ONE - best overlap first. Same candidate list, same budget, same single head. The relation
    is a property of the evidence, not a second decision.
    """
    ix = reach_index(p)
    if ix is None:
        return []
    i = ix['of'].get(q['address'])
    own = {p['tape'].values[s] for s in q['slots'][:q['query_row']]}
    kp = retain_keep(p)
    overlap = Counter()
    for v in own:
        for j, _c in ix['by_val'].get(v, ()):
            if j != i and (kp is None or bool(kp[j])):
                overlap[j] += 1
    out = []
    for j, _ov in overlap.most_common(CONNECT_MAX):
        for v, rws, _c in ix['fills'][j]:
            if v == value:
                out.extend(rws)
                break
        if len(out) >= REACH_ROWS_PER_VALUE:
            break
    return out

def reach_rows_for(p, q, value, rows):
    """The evidence a candidate brings: its walked rows, its homes elsewhere on the tape, or
    the mentions it has at places RELATED to this one (372b)."""
    if REACH_IMPORT == 'homes':
        return outside_mentions(p, q, value)
    if REACH_IMPORT == 'relation':
        rel = reach_relation_rows(p, q, value)
        return rel if rel else rows
    return rows

def reach_world(p, q, bank, device, value, rows, budget):
    """The question's rows, the query row filled in, and up to `budget` rows the walk found for
    that value. One budget for every candidate, so no world is larger than another."""
    slots = list(q['slots']) + [s for s in rows[:budget] if s not in q['slots']]
    vals = [p['tape'].values[s] for s in q['slots'][:q['query_row']]]
    vals += [value] + [p['tape'].values[s] for s in slots[len(q['slots']):]]
    w = {'verb': 'lookup', 'S': q['S'], 'slots': slots, 'vals': vals, 'query_row': q['query_row'], 'n_first': len(slots), 'home_cos': 0.0 if HOME_COS_STAGE == 'stage2' and (not q.get('_stage2')) else reach_home_cos(p, q, value, q.get('_qv')), 'confirm': reach_confirm(p, q, value), 'channel': reach_channel(p, q, value)}
    return build_graph(p, w, bank, device, query_value=None, import_k=0)

def reach_move_pick(net, p, q, device, bank):
    """385: THE MIND EMITS A MOVE AND THE TAPE EXECUTES IT.

    WHAT IS ACTUALLY NEW. Four channels exist - the fingerprint walk, connect (365), copy (376),
    and the home rows - and every one of them has always been MERGED BY A FIXED RULE into a
    single offer, with Phi choosing a NAME out of eight. Its only search action in the whole
    system is the root of hop 2. So the thing that decides WHERE TO LOOK has never been the
    mind; it has been a constant, and four attempts to tune that constant each cost more than
    they brought (347 thickness, 365 connect, 369 addresses, 377 copy).

    Here the output space is MOVES. `step` is the fingerprint walk, `share` steps to places that
    share a filler, `lines` reads the neighbouring lines. Each is offered at the UNCHANGED cap,
    so a move is not a thinner offer - it is a different one. A fact still cannot be encoded in
    an output space of three names, so the invariant is untouched, and it is one head, one
    softmax, one objective: 321, 341 and 352 each measured a second objective at about 4x.

    THE CHOICE IS MADE BEFORE THE CANDIDATES ARE SCORED, and that is the whole point. Enumerating
    a lane is cheap; scoring worlds is not. Each move is judged on ONE PROBE WORLD - its first
    candidate carried by a single row - so the mind spends |MOVES| small worlds to decide, then
    the chosen move alone is executed at full cap. Under lookahead the mind would instead score
    every lane in full and take the best, which is today's argmax wearing a different name and
    decides nothing.

    EQUAL SIZE AMONG THE PROBES, because this project has twice been undone by a row-count
    marker (291, 296): every probe is one row, and a move whose first candidate has no row is
    not probed at all rather than probed smaller.

    A move that offers nothing is not on the ballot. If none of them offers anything the first
    move is taken, so the question still has an offer and the arm never silently becomes a
    different population.
    """
    if not MOVES_ON:
        return (None, None, [], [])
    props = []
    for m in MOVES:
        rc = reach_candidates(p, q, which=m)
        cs = rc['cands']
        if not cs:
            continue
        rows = rc['rows_of'].get(cs[0], [])
        if rows:
            props.append((m, cs[0], rows[:1], q['truth_value'] in set(cs)))
    if not props:
        q['_move'] = MOVES[0]
        return (MOVES[0], None, [], [])
    gs = []
    for _m, v, rw, _a in props:
        q.pop('_base', None)
        q['_stage2'] = True
        gs.append(reach_world(p, q, bank, device, v, rw, 1))
    l0 = torch.stack([net.phi(*x) for x in gs])
    if '_move' not in q:
        q['_move'] = props[int(l0.argmax())][0]
    q.pop('_base', None)
    return (q['_move'], l0, [m for m, _v, _r, _a in props], [a for _m, _v, _r, a in props])

def reach_logits(net, p, q, device, bank):
    """Stage 1: say one of the values already here, refuse, or walk. Stage 2: say one of the
    values the walk reached, or refuse. Both are worlds scored by the same Phi."""
    if MOVES_ON and ('_move' not in q or MOVE_TEACH):
        _mv, _l0, _mnames, _mans = reach_move_pick(net, p, q, device, bank)
        q['_move_l0'] = (_mv, _mnames)
        q['_move_ballot'] = None if _l0 is None else (_l0, _mans)
    rc = reach_candidates(p, q)
    own, cands, rows_of = (rc['own'], rc['cands'], rc['rows_of'])
    ev = {c: reach_rows_for(p, q, c, rows_of[c]) for c in cands}
    budget = min([IMPORT_K] + [len(ev[c]) for c in cands]) if cands else 0

    def world(value, rows, b, stage2=False):
        q.pop('_base', None)
        q['_stage2'] = stage2
        return reach_world(p, q, bank, device, value, rows, b)
    lc = reach_line_candidates(p, q) if REACH_LINE else {'cands': [], 'rows_of': {}}
    lcands, lrows = (lc['cands'], lc['rows_of'])
    lbudget = min([IMPORT_K] + [len(lrows[c]) for c in lcands]) if lcands else 0
    g = q.get('_reach_g')
    if g is None:
        walk_rows = [s for c in cands for s in ev[c]]
        if OWN_IMPORT:
            ovr = {v: outside_mentions(p, q, v) for v in own}
            for v in own:
                _OWN_IMPORT_N[0] += 1
                _OWN_IMPORT_N[1] += 1 if len(ovr[v]) >= budget else 0
            g1 = [world(v, ovr[v], budget) for v in own]
        else:
            g1 = [world(v, [], 0) for v in own]
        if not REACH_NO_REFUSE:
            g1.append(world(REFUSE_LABEL, [], 0))
        if not REACH_LOOKAHEAD:
            g1.append(world(REFUSE_LABEL, [s for c in cands for s in ev[c][:budget]], 10 ** 6))
        g2 = [world(c, ev[c], budget, True) for c in cands]
        if not REACH_NO_REFUSE or not cands:
            g2.append(world(REFUSE_LABEL, walk_rows, budget, True))
        g3 = [world(c, lrows[c], lbudget) for c in lcands] if lcands else []
        g = (g1, g2, g3)
        if q.get('_keep_g'):
            q['_reach_g'] = g
    g1, g2, g3 = g
    l1 = torch.stack([net.phi(*x) for x in g1])
    l2 = torch.stack([net.phi(*x) for x in g2])
    l3 = torch.stack([net.phi(*x) for x in g3]) if g3 else None
    if REACH_DEPTH > 1:
        ld, dcands, dplaces = reach_deep(net, p, q, device, bank, cands, ev, budget, l2)
        q['_deep'] = (ld, dcands, dplaces)
    if TWO_WAY:
        q['_own_l'] = l1
        m = min(len(l1), len(l2)) if len(l1) and len(l2) else 0

        def summary(x):
            """WHAT A BRANCH IS WORTH, in one number. `max` is the best world behind it - the
            lookahead rule, unchanged. `margin` is the GAP between its best and second-best,
            which is a different claim: not "how good is the best thing here" but "how sure am
            I which of these it is".

            The margin is the quantity 337 measured at AUC 0.866 and 352 at 0.969 - the mind
            knows when it is right better than it knows anything else - and it currently decides
            NOTHING. This is the smallest form in which it decides something. It is a declared
            rule and not a fitted one, and `max` reproduces every earlier run exactly.
            """
            if not len(x):
                return torch.full((1,), -1000000000.0, device=l1.device)
            if TWO_WAY_BY == 'max' or len(x) < 2:
                return x.max().reshape(1)
            sv = x.sort(descending=True).values
            return (sv[0] - sv[1]).reshape(1)
        stay = summary(l1[:m] if m else l1)
        go = summary(l2[:m] if m else l2) if len(l2) else torch.full((1,), -1000000000.0, device=l1.device)
        l1 = torch.cat([stay, go])
    if REACH_LOOKAHEAD and (not TWO_WAY):
        m = min(len(l2), len(l3)) if EQUAL_TAILS and REACH_LINE and (l3 is not None) and len(l2) and len(l3) else 0
        tail = [(l2[:m] if m else l2).max().reshape(1)]
        if REACH_LINE:
            tail.append((l3[:m] if m else l3).max().reshape(1) if l3 is not None else torch.full((1,), -1000000000.0, device=l1.device))
        l1 = torch.cat([l1] + tail)
    if REACH_DEPTH > 1:
        ld, _dc, _dp = q.get('_deep', (None, [], []))
        if ld is not None:
            l2 = torch.cat([l2, ld.max().reshape(1)])
    return (l1, l2, own, cands, l3, lcands)

def coherence_block(net, p, bank, device, n_pairs, rng):
    """Does Phi rank a REAL fragment of tape above a corrupted one - no hole, no teacher?

    THE CLAIM THIS PROJECT MAKES IN ITS OWN WORDS is that Phi is the coherence of a completed
    world. Every world it has ever been shown has a hole and a demand for an answer, so what has
    actually been measured is a DISCRIMINATOR OVER CANDIDATES. Those are not the same thing, and
    if Phi cannot tell an intact fragment from a tampered one, the honest name is the narrower.

    The pair differs in exactly one value. Same slots, same size, same place fingerprints, same
    rare-word overlaps - those are read from the slot's own text, which is untouched - so what
    moves is the `same` matrix and the count share, which is precisely what moves between two
    candidates in an ordinary world. Nothing here is easier than what the mind already does.

    No teacher, no rival, no label at exam time: this is Phi against itself, and a coin means
    the word coherence has to be given back.
    """
    ix = reach_index(p)
    if ix is None or not ix['items']:
        return None
    items = [it for it in ix['items'] if len(it['slots']) >= 3]
    if not items:
        return None
    vals = p['tape'].values
    win = tie = 0
    gaps = []
    with torch.no_grad():
        for _ in range(n_pairs):
            it = items[rng.randrange(len(items))]
            rows = list(it['slots'])
            if len(rows) > REACH_MAX_ROWS:
                rows = rng.sample(rows, REACH_MAX_ROWS)
            here = {vals[s] for s in rows}
            for _try in range(8):
                o = ix['items'][rng.randrange(len(ix['items']))]
                if o['address'] == it['address'] and len(ix['items']) > 1:
                    continue
                alien = vals[o['slots'][rng.randrange(len(o['slots']))]]
                if alien not in here:
                    break
            else:
                continue
            real = {'verb': 'lookup', 'S': it['S'], 'slots': rows, 'vals': [vals[s] for s in rows], 'query_row': -1, 'n_first': len(rows)}
            bad = dict(real)
            bv = list(real['vals'])
            bv[rng.randrange(len(bv))] = alien
            bad['vals'] = bv
            a = float(net.phi(*build_graph(p, real, bank, device, query_value=None, import_k=0)))
            b = float(net.phi(*build_graph(p, bad, bank, device, query_value=None, import_k=0)))
            win += a > b
            tie += a == b
            gaps.append(a - b)
    n = len(gaps)
    if not n:
        return None
    return {'n': n, 'real_higher': win / n, 'ties': tie / n, 'mean_gap': sum(gaps) / n, 'binomial_z': (win - 0.5 * n) / (0.25 * n) ** 0.5 if n else float('nan')}

def reach_places_from(p, j0, k, seen_places):
    """The k places nearest to PLACE j0, rather than to the question's own place.

    The deeper walk is the same operation one read further on: the same fingerprints, the same
    cosine, the same cap. Places already visited are excluded so a chain cannot pay twice for
    standing still, which would make depth free money.
    """
    ix = reach_index(p)
    if ix is None:
        return []
    sims = (ix['M'] @ ix['M'][j0]).clone()
    for j in seen_places:
        sims[j] = -2.0
    kp = retain_keep(p)
    if kp is not None:
        sims = sims.masked_fill(~kp.to(sims.device), -2.0)
    order = sims.argsort(descending=True)[:k].tolist()
    return [(j, ix['items'][j], float(sims[j])) for j in order if j not in seen_places and sims[j] > -2.0]

def place_of_rows(p, q, rows):
    """381: THE PLACE A CANDIDATE ACTUALLY STANDS AT - its first row that is not at home.

    A row is a slot, a slot has an address, an address is a place. Rows at the question's own
    address are skipped, the same exclusion the walk and the connect channel already apply.
    Returns None when the candidate has no row anywhere else, which is a real state and not an
    error: `outside_mentions` can be empty for a value whose only standing is the hidden one.
    """
    ix = reach_index(p)
    for sl in rows:
        if p['straddr'][sl] == q['address']:
            continue
        jj = ix['of'].get(p['straddr'][sl])
        if jj is not None:
            return jj
    return None

def deep_root_of(p, q, rc, value, places):
    """380: WHERE THE SECOND READ STARTS - the place the chosen candidate ACTUALLY LIVES.

    THE BUG THIS REPLACES HAS BEEN LIVE SINCE 365. `reach_deep` rebuilt its own value->place map
    by scanning only the WALKED places, so a candidate offered by any other channel was not in
    it and the root silently fell back to `places[0][0]` - the walk's first place, which has
    nothing to do with where that value stands. Connect candidates have hit that fallback since
    `--connect` existed; copy candidates since 377; and 379, by letting Phi SEE the channel,
    made it pick non-walk candidates more often and so fired the fallback more often. The
    measured cost is not small, because `reachable_rate` counts the deeper candidates too
    (`ansble = truth in set(cands) | set(_dc_all)`): the offer can be identical and reach still
    moves. On 379 held, hit_of_deep fell .980->.830, .949->.793, .919->.796 on the three seeds
    where the mind stepped to copy, and rose on the one where the root was still a walk place.

    CONSEQUENCE FOR WHAT IS ALREADY MEASURED: 365conn, 377copy, 378bf and 379 are all partly
    confounded by this, in proportion to how often the mind picked a non-walk candidate. It also
    explains 378's s4711 collapse better than "the channel teaches the router not to step" did:
    backfill feeds copy candidates ONLY onto thin questions, the mind picks them, the second
    read roots at an unrelated walk place, depth returns garbage, and the router learns to stay.
    The mechanism was right, the reason was wrong.

    THE FIX IS ONE RULE FOR EVERY CHANNEL. A walked candidate keeps its walked place, which is
    what `from_place` already records. Any other candidate is rooted through its OWN FIRST ROW:
    a row is a slot, a slot has an address, an address is a place. For connect that recovers
    exactly the neighbour place the value was found at - its rows come from that place's fills,
    best overlap first - so no signature has to change to carry it. For copy and home it is the
    place the value actually stands at, which is the only honest answer to "where does this
    live". Rows from the question's OWN place are skipped, the same exclusion the walk and the
    connect channel already apply, so the second read never starts at home.

    `rc["rows_of"]` is used and not `ev`: the offer's own rows do not depend on --reach-import,
    and a root that moved with the import policy would make depth and evidence one knob.
    """
    j = rc['from_place'].get(value)
    if j is not None and j >= 0:
        return j
    jj = place_of_rows(p, q, rc['rows_of'].get(value, ()))
    return places[0][0] if jj is None else jj

def reach_deep(net, p, q, device, bank, cands, ev, budget, l2):
    """One more read, rooted where the mind's own best shallow candidate lives.

    Returns (logits, candidate names) for the deeper stage, or (None, []) when there is nowhere
    left to go. Values already offered shallower are excluded: a deeper read has to bring
    something NEW or it is not a read.
    """
    rc = reach_candidates(p, q)
    places = rc['places']
    if not places or not cands or len(l2) == 0:
        return (None, [], [])
    ix = reach_index(p)
    if DEEP_ROOT == 'first':
        j0 = places[0][0]
    else:
        best = min(int(l2.argmax()), len(cands) - 1)
        j0 = deep_root_of(p, q, rc, cands[best], places)
    seen_p = {j for j, _it, _s in places} | {j0}
    seen_v = set(cands) | set(rc['own'])
    dc, drows, dplaces = ([], {}, [])
    for j, _it, _sim in reach_places_from(p, j0, REACH_K, seen_p):
        dplaces.append((j, ix['items'][j], 0.0))
        for v, rows, _c in ix['fills'][j]:
            if v in seen_v:
                continue
            if v not in drows:
                if len(dc) >= REACH_CANDS:
                    continue
                dc.append(v)
                drows[v] = []
            drows[v].extend(rows)
    if not dc:
        return (None, [], [])
    dev = {c: reach_rows_for(p, q, c, drows[c]) for c in dc}
    b = min([IMPORT_K] + [len(dev[c]) for c in dc])
    g = []
    for v in dc:
        q.pop('_base', None)
        q['_stage2'] = True
        g.append(reach_world(p, q, bank, device, v, dev[v], b))
    return (torch.stack([net.phi(*x) for x in g]), dc, dplaces)

def reach_bisect(net, p, q, device, bank, cands, ev, budget):
    """Halve the candidate list until one survives, scoring each half as ONE unfilled world.

    EQUAL HALVES BY CONSTRUCTION. When the list is odd the extra candidate stays in its half for
    the purposes of SURVIVING, but only min(|L|, |R|) of each side contribute evidence rows - so
    the two worlds are the same size and the split cannot be decided by having more to show.
    Returns (survivor, split logit pairs, the truth's side at each split) - the last is the exact
    teacher, read off the tape and never off the model.
    """
    order, pairs, sides = (list(cands), [], [])
    truth = q['truth_value']
    while len(order) > 1:
        h = len(order) // 2
        L, R = (order[:h], order[h:])
        n = min(len(L), len(R))
        q.pop('_base', None)
        q['_stage2'] = True
        lw = reach_world(p, q, bank, device, REFUSE_LABEL, [s for v in L[:n] for s in ev[v][:budget]], 10 ** 6)
        q.pop('_base', None)
        rw = reach_world(p, q, bank, device, REFUSE_LABEL, [s for v in R[:n] for s in ev[v][:budget]], 10 ** 6)
        lg = torch.stack([net.phi(*lw), net.phi(*rw)])
        pairs.append(lg)
        sides.append(0 if truth in set(L) else 1 if truth in set(R) else -1)
        order = L if int(lg.argmax()) == 0 else R
    return (order[0] if order else None, pairs, sides)

def reach_reward(q, names, answerable, device):
    R = torch.full((len(names),), -1.0, device=device)
    for i, nm in enumerate(names):
        if nm == REFUSE_LABEL:
            R[i] = 1.0 if not answerable else 0.75
        elif nm == q['truth_value']:
            R[i] = 1.0
    return shift_reward(R)

def shift_reward(R):
    """DISCOUNTING ONLY MEANS A PRICE ON A NON-NEGATIVE SCALE, and 311a is why this exists.

    With rewards in {-1, +1} the expectation at home is negative (truth_in_own 0.13) and so is
    the expectation behind a step (reachable 0.12). Multiplying a NEGATIVE number by gamma
    RAISES it: -0.75 * 0.9 = -0.68. So gamma paid the mind to leave - free money, the one thing
    the design forbade, arriving through the multiplication instead of through an addition. The
    run measured it exactly: step_rate 0.98 and router selectivity 1.02x against counting 1.02x,
    where the same arm at gamma 1.0 stepped on 1.1% and routed at 19.25x.

    The fix is not a smaller gamma - any gamma < 1 has this sign flip. The zero of the scale has
    to MEAN something: zero is the worst outcome, one is the best, and arriving later is worth
    strictly less than arriving now. The map is affine and fixed, never fitted; at gamma 1.0
    nothing is touched, so every run before this one stays comparable.
    """
    return (R + 1.0) / 2.0 if REACH_GAMMA < 1.0 else R

def reach_names(own, cands):
    """Stage-one and stage-two option names, in the order their worlds were built. One
    definition, because the loss, the exam and the report must not disagree about what the
    columns of a softmax mean."""
    tail = [] if REACH_NO_REFUSE else [REFUSE_LABEL]
    return (own + tail, cands + ([REFUSE_LABEL] if tail or not cands else []))

def reach_answerable(p, q):
    return q['truth_value'] in set(reach_candidates(p, q)['cands'])

def speak_term(margins, advs, device):
    """341: THE PRICE OF SPEAKING, PAID ACROSS QUESTIONS INSTEAD OF ON EACH ONE.

    WHY THE PER-QUESTION PRICE CANNOT TEACH THIS. Refusal has been tried three times in this
    project (299_hash, 311's first pair, 339) and failed identically each time, and it was never
    Phi's fault: taken ONE QUESTION AT A TIME, mixed_payoff pays a full 1.0 for correctly
    refusing an unanswerable hole, and 87% of holes are unanswerable, so "always refuse" is the
    arithmetic optimum of the reward and no mind enters into it. Three times we met a base rate
    and read it as a result. Under --reach-no-refuse we then removed the decision entirely - so
    the mind is trained forbidden to be silent, while 339/340 measure exactly the silence
    decision it was never taught.

    THE CHANGE IS TO THE SCOPE OF THE PRICE, NOT TO THE PRICE. The reward is unchanged:
    `advs` is mixed_payoff for answering minus mixed_payoff for refusing, per question,
    derived here rather than restated so it cannot drift from the reward everything else uses.
    What changes is that the mind allocates ONE UNIT OF SPEECH across the batch - a softmax over
    the margins, summing to one - so refusing everything is not expressible. What is left to
    learn is which questions to spend it on, which is precisely the ranking 337 found already
    present untrained (AUC 0.866 on its own correctness, against 0.669 for the best count).

    AND IT IS THE FORM PHI IS DEFINED FOR. 329, 337 and 340 agree that Phi has no absolute
    scale - the raw score is a coin, only the gap carries anything. A per-question speak/refuse
    asks Phi for an absolute judgement. A budgeted choice among questions asks it for a
    comparison, which is the only thing it has ever been able to do.

    NO NEW HEAD, NO NEW PARAMETER, THE SAME R AND THE SAME PHI. It is still a SECOND TERM in the
    loss, and 321 measured a second term on these 5633 parameters at 4x the route - so the arm
    declares its weight, defaults to off, reports the route, and is never pooled with the runs
    it is compared against.
    """
    m = torch.stack(margins)
    a = torch.tensor(advs, device=device, dtype=m.dtype)
    return (torch.softmax(m, 0) * a).sum()

def calib_term(scores, labels, device):
    """389: ONE SCALE ACROSS QUESTIONS. The hole this closes is a GAUGE, not a missing option.

    CLOSED BEFORE IT RAN - DO NOT RE-PROPOSE. See _STATE_353.md section 31. The void check this
    arm declared as gate 4 fired on its own control: raw-score AUC on the standing world is
    0.6385-0.7250 across four seeds, not the 0.50 the argument below predicts, and it already
    beats both counting rivals on all four. The algebra is right and the INFERENCE FROM IT IS
    WRONG: the loss does not constrain a per-question offset, but Phi is a function of the
    world's content with shared weights, so the offset is not a free parameter - unconstrained
    is not arbitrary. The flag is left in place, off, so this note has somewhere to live.

    ------------------------------------------------------------------ the original argument

    WHAT IS ACTUALLY BROKEN. Every gradient Phi has ever received arrives through a softmax over
    the worlds OF ONE QUESTION, and softmax(l + c) = softmax(l). So a constant added to every
    world of a question is invisible to the loss - each question carries its own free offset, and
    Phi's raw value is comparable only WITHIN a question. That is not a shortage of training; it
    is a symmetry of the objective, and it is why 329/337/340 all found the same thing in
    different words: "no absolute scale, only the gap carries".

    WHY THE REFUSAL OPTION DID NOT FIX IT, three times (299_hash, 311's first pair, 339). REFUSE
    is a world in the SAME softmax, so it moves with the offset like everything else; what it
    contributes is again a gap. And because the per-question price asks for an absolute judgement
    on a tape where 87% of holes are unanswerable, "always refuse" is the arithmetic optimum -
    the base rate answers before the mind does. speak_term (341) fixed the SCOPE of that decision
    by spending one unit of speech across a batch, but it compares MARGINS, and a margin is gauge
    invariant too. So the gauge has never been touched.

    THE TERM. B questions are scored together; each contributes the raw Phi of the world it
    settled on. One softmax ACROSS QUESTIONS, against the uniform distribution over the ones the
    tape can actually answer. Nothing is asked of Phi that it has not always done - this is a
    comparison, not an absolute judgement - but the things compared now belong to DIFFERENT
    questions, which is the only way B-1 of the B free offsets can be removed. One global offset
    survives, and one is all a single threshold needs.

    IT CANNOT BE WON BY A CONSTANT, which is exactly what the per-question price could be won by:
    the term is shift invariant, so pushing every score down changes nothing. There is no
    "always refuse" here to meet.

    THE LABEL IS `answerable` - a property of the tape and the walk, never of the mind. Not
    `right`: training on its own current correctness would be a moving target, and it would leave
    no untouched target to read the result on. `right` stays held out, so mind_score's AUC there
    is transfer rather than recall.

    A BATCH WITH NO POSITIVES (or no negatives) HAS NO TARGET and contributes zero rather than a
    fabricated one. Returned to be MAXIMISED, like speak_term.

    SECOND TERM, SO IT IS PRICED LIKE ONE: 321 and 341 each measured a second objective at ~4x the
    route on these 5633 parameters. Declared weight, default off, never pooled with its control.

    389 IS VOID AND THIS FLAG IS CLOSED - _STATE_353.md section 31. Gate 4 of the pre-registration
    ("read this before reading anything else") fired on the CONTROL, before the arm was ever run:
    on 365r3 held, four seeds, the raw mind_score AUC is 0.639-0.725 against the coin's 0.500 and
    beats BOTH counting rivals on 4/4, and on the never-trained target `right` it is 0.616-0.738.
    The scale this term was built to buy, a standing world already holds.

    THE ERROR WAS AN INFERENCE, NOT THE ALGEBRA. softmax(l + c) = softmax(l) says the LOSS does
    not constrain a per-question offset. It does not say the learned function is free: Phi is a
    function of the world's CONTENT with shared weights, so the offset is whatever that shared
    function emits for that question's material. Unconstrained is not arbitrary, and a symmetry
    of the objective is not a symmetry of the learned function.

    The flag stays here, OFF, so that the fourth refusal-shaped proposal meets this paragraph
    first. What survives is the reading above of why the three refusal attempts failed - that was
    never the gauge argument - and the GAUGE line in _read299, which reports a real quantity that
    had gone unread for eighty steps and answered the opposite of the prediction.
    """
    s = torch.stack(scores)
    y = torch.tensor(labels, device=device, dtype=s.dtype)
    npos = float(y.sum())
    if npos <= 0.0 or npos >= float(len(labels)):
        return torch.zeros((), device=device, dtype=s.dtype)
    return (torch.log_softmax(s, 0) * (y / npos)).sum()

def move_term(q, device):
    """391: THE MOVE GETS A GRADIENT. It never had one.

    WHAT WAS ACTUALLY BROKEN, and it is one line. `reach_move_pick` builds one probe world per
    move and takes the argmax of Phi over them; `reach_logits` then kept the CHOSEN NAME and
    threw the logits away - `q["_move_l0"] = (_mv, _mnames)`. Nothing downstream ever touched
    them, so no gradient has ever reached the move decision. 385 and 386 were therefore not
    measurements of "can the mind choose where to look": they measured an argmax of a scorer
    trained to rank FINAL NAMES, applied to a decision nobody had taught. 386's own diagnosis -
    "on three seeds of four the hit of `share` was BELOW the hit of `step`, so the mind picks
    the second move on the questions where that move is worse" - is exactly what an untaught
    policy looks like, and its conclusion ("the probe is one row, and one row is evidently not
    enough") was one of two explanations with no way to tell them apart.

    THE TEACHER, AND IT IS THE TAPE'S. Whether a lane REACHES THE TRUTH - the truth among the
    candidates that lane offers. That is `answerable` per move: a property of the tape and the
    walk, never of the mind's current correctness, which is the same discipline calib_term
    declared and the reason `right` stays untouched. It is counted while the ballot is already
    enumerating each lane, so it costs no world, no head and no parameter.

    WHY AN EXPECTATION AND NOT ONE SAMPLE. Enumerating a lane is counting; only SCORING it is
    expensive. So every arm of this ballot has an exact reward and the term is
    `(softmax(l0) * R).sum()` - the same shape as every other term in this file. A one-sample
    policy gradient would have needed a baseline, and a baseline chosen by hand is precisely the
    class of mistake this project has made three times (317, 383, 387 each found a rule that was
    really a scale).

    WHAT IT IS NOT: the value behind a move. That would be the lane's stage-two expectation, and
    having it for the moves NOT taken means scoring every lane in full - lookahead, which 385
    rejected as today's argmax wearing a different name. Reaching is the EXACT UPPER BOUND of
    that value: a name that is not offered cannot be said. So this term teaches where the answer
    is, and the pick still has to say it - which is why the guard below is on the pick.

    THE SCALE IS THE ONE ALREADY IN USE. `shift_reward`, so the zero means what it means
    everywhere else. 311a is why that sentence exists: a discount on a signed scale paid the
    mind to leave, and the sign flip arrived through a multiplication.

    A BALLOT THAT CANNOT TEACH CONTRIBUTES NOTHING, AND IS COUNTED. With fewer than two live
    moves, or with every lane reaching the same, `(p * R).sum()` is constant in l0 and its
    gradient is exactly zero. Skipping it changes no gradient, and the count it leaves behind is
    the void check of this step: if the lanes almost always agree, the term has nothing to say
    and the arm is void BEFORE its numbers are read. 389's gate 4 is why that is read first.

    SECOND TERM, SO IT IS PRICED LIKE ONE: 321, 341 and 352 each measured a second objective at
    about 4x the route. Declared weight, default off, never pooled with its control.

    THIS IS A FIX AND NOT A STEP - _STATE_353.md section 33-VOID. 387, re-run after the section 27
    leak fix, puts a PERFECT chooser of the lane at oracle - merged = +0.017, and that is the
    ceiling of `reachable_rate`, which is what this term teaches. A gradient does not create
    reach. The gate that was declared for it carried no magnitude and could have PASSED ON NOISE
    against a ceiling of 0.017; it is withdrawn, and those seeds are not to be run.

    The wiring stays because any future chooser needs it - without it the choice is again an
    argmax of a scorer trained for something else. It takes no step number, it is off by default,
    and 34.3's law says what it must clear first: choosing one source has now lost to keeping the
    merge at three different levels (347 the offer, 387 the lane, 393 the place).

    AND ONE RISK, WHICH IS NOT PRICED ANYWHERE ABOVE: with the term on, one Phi is taught on
    ONE-ROW PROBES and on FULL WORLDS. Not a second objective - a second JOB for one function, and
    a row-count marker is what undid 291 and 296.
    """
    if not MOVE_TEACH:
        return None
    ent = q.get('_move_ballot')
    if ent is None:
        return None
    l0, ans = ent
    _MOVE_LIVE['ballot'] += len(ans)
    _MOVE_LIVE['n'] += 1
    if len(ans) < 2 or len(set(ans)) < 2:
        return None
    _MOVE_LIVE['live'] += 1
    R = shift_reward(torch.tensor([1.0 if a else -1.0 for a in ans], device=device, dtype=l0.dtype))
    return MOVE_TEACH * (torch.softmax(l0, 0) * R).sum()

def reach_loss(net, p, q, device, bank):
    l1, l2, own, cands, l3, lcands = reach_logits(net, p, q, device, bank)
    ans = q['truth_value'] in set(cands)
    if _SPEAK_ACC is not None or _CALIB_ACC is not None:
        _said, _st, _dp, _sc, mg = reach_pick(q, l1, l2, own, cands, l3, lcands, keep_graph=True)
        rt = _said == q['truth_value']
        if _SPEAK_ACC is not None:
            _SPEAK_ACC.append((mg, mixed_payoff(False, rt, ans) - mixed_payoff(True, rt, ans)))
        if _CALIB_ACC is not None:
            _CALIB_ACC.append((_sc, 1.0 if ans else 0.0))
    n1, n2 = reach_names(own, cands)
    R1 = reach_reward(q, n1, ans, device)
    R2 = reach_reward(q, n2, ans, device)
    p1 = torch.softmax(l1, 0)
    if ROUTE_ON == 'walk_only':
        _ROUTE_LIVE['n'] += 1
        if ans and q['truth_value'] not in set(own):
            _ROUTE_LIVE['live'] += 1
        else:
            p1 = p1.detach()
    p2 = torch.softmax(l2, 0)
    if REACH_DEPTH > 1:
        ld, dcands, _dp = q.get('_deep', (None, [], []))
        if ld is not None:
            R3 = reach_reward(q, dcands, q['truth_value'] in set(dcands), device)
            v3 = REACH_GAMMA * (torch.softmax(ld, 0) * R3).sum()
            R2 = torch.cat([R2, v3.reshape(1)])
    v2 = REACH_GAMMA * (p2 * R2).sum() - STEP_COST
    mv = move_term(q, device)
    if TWO_WAY:
        lo = q['_own_l']
        v_stay = (torch.softmax(lo, 0) * R1).sum()
        out = p1[0] * v_stay + p1[1] * v2
        return -(out if mv is None else out + mv)
    nk = 2 if REACH_LINE else 1
    out = (p1[:-nk] * R1).sum() + p1[-nk] * v2
    if BISECT:
        _, pairs, sides = reach_bisect(net, p, q, device, bank, cands, ev, budget)
        for lg, side in zip(pairs, sides):
            if side >= 0:
                out = out - torch.nn.functional.cross_entropy(lg.reshape(1, 2), torch.tensor([side], device=device))
    if STAGE2_ALWAYS:
        out = out + STAGE2_ALWAYS * (p2 * R2).sum()
    if REACH_LINE:
        if l3 is not None:
            ans3 = q['truth_value'] in set(lcands)
            R3 = reach_reward(q, lcands, ans3, device)
            p3 = torch.softmax(l3, 0)
            out = out + p1[-1] * (REACH_GAMMA * (p3 * R3).sum() - STEP_COST)
        else:
            out = out + p1[-1] * (REACH_GAMMA * -1.0)
    return -(out if mv is None else out + mv)

def cons_cooc(p, v):
    """What stands with `v`, counted over EVERY place that holds it. Cached per pack.

    This is the whole difference from the walk. reach_places goes to the eight places whose
    filler bag is nearest by fingerprint - an approximation, capped, local. This goes to ALL
    places holding one exact value, however far, and counts. Unbounded neighbourhood, exact
    relation, no cap: the operation that gets BETTER as the corpus grows instead of showing a
    smaller and smaller share of it.

    Exact and cheap at once, and the arithmetic is worth writing down because it is why no cap
    is needed: the counter is over the TAPE, not the corpus, so its total size is bounded by
    sum over places of (fillers at that place) squared - on a 1500-place tape with ~7 fillers
    each that is under a hundred thousand entries for ALL values together. Truncating it would
    buy nothing and would make the subtraction in cons_resolve a lie.
    """
    ix = reach_index(p)
    if ix is None:
        return {}
    cache = p.get('_cons_cooc')
    if cache is None:
        cache = p['_cons_cooc'] = {}
    c = cache.get(v)
    if c is None:
        c = Counter()
        for j, _n in ix['by_val'].get(v, ()):
            for w, _rows, cnt in ix['fills'][j]:
                c[w] += cnt
        cache[v] = c = dict(c)
    return c

def cons_place(p, q, v):
    """384: THE ONE PLACE A LENS SPEAKS FROM - v's home, where it stands most often.

    WHY A SINGLE PLACE AT ALL. The constraint interface was measured twice and closed twice -
    345 L1 by raw count, 345 L2 by share - and the verdict recorded for that whole family was
    "the third time SUMMING LENSES has lost", with the conclusion that what could work is "a
    SELECTION of one lens, never an accumulation". But look where the accumulation actually
    sat: the mind's part was ALREADY a selection - it picks one of its own rows - while the
    TAPE's part summed `cons_cooc`, which adds the fills of EVERY place holding v. Both L1 and
    L2 are that same sum; share only divides it afterwards. Nobody has run a constraint whose
    RESOLUTION is also a selection, and that is what this is.

    The ingredient did not exist in 345. 380 had to answer "where does this value actually
    live" to root the second read, and `place_of_rows` / `real_place` are that answer; this is
    the same question asked of a lens instead of a candidate.

    WHICH PLACE, and it is one rule: where v STANDS MOST OFTEN. A place is a hole and its
    fillers are what the corpus wrote in that hole, so v's biggest place is the hole v most
    belongs to, and the other fillers there are the alternatives the corpus actually offers for
    it. Ties go to where v owns the largest share of the hole, and then to tape order, so two
    runs resolve identically. The question's OWN place is excluded outright rather than
    subtracted afterwards - the hidden truth stands there, and excluding beats subtracting when
    the whole point is that one place is being chosen.

    Returns the place index, or None when v stands nowhere else.
    """
    ix = reach_index(p)
    if ix is None:
        return None
    here = ix['of'].get(q['address'])
    best_j, best_key = (None, None)
    for j, _n in ix['by_val'].get(v, ()):
        if j == here:
            continue
        fills = ix['fills'][j]
        tot = sum((c for _w, _r, c in fills))
        cnt = sum((c for w, _r, c in fills if w == v))
        if tot <= 0 or cnt <= 0:
            continue
        key = (cnt, cnt / tot, -j)
        if best_key is None or key > best_key:
            best_j, best_key = (j, key)
    return best_j

def cons_resolve(p, q, v):
    """The tape's answer through lens `v`: what most often stands with it, elsewhere.

    THE QUESTION'S OWN PLACE COMES OUT, and it is subtracted rather than skipped so the count
    stays exact. Without it the hidden truth leaks straight through the lens - it stands at this
    very place, so any value co-occurring with `v` here would be counted as evidence from
    somewhere else. Same discipline as the fingerprint subtraction in reach_places.

    `v` itself is excluded: a lens cannot answer with itself, and the tape's own row for it is
    already visible to the mind.

    Returns (best, best_count, total, top). Deterministic given the tape, which is what makes
    the teacher exact.
    """
    ix = reach_index(p)
    if ix is None:
        return (None, 0, 0, [])
    if CONS_RESOLVE == 'place':
        j = cons_place(p, q, v)
        if j is None:
            return (None, 0, 0, [])
        c, sub = ({w: cnt for w, _r, cnt in ix['fills'][j]}, {})
    else:
        c = cons_cooc(p, v)
        if not c:
            return (None, 0, 0, [])
        here = ix['of'].get(q['address'])
        sub = {}
        if here is not None and any((j == here for j, _n in ix['by_val'].get(v, ()))):
            sub = {w: cnt for w, _rows, cnt in ix['fills'][here]}
    tot_of = None
    if CONS_RESOLVE == 'share':
        tot_of = {}
        for w, _n in c.items():
            tot_of[w] = sum((cnt for _j, cnt in ix['by_val'].get(w, ()))) or 1
    best, bn, tot = (None, 0, 0)
    top = []
    for w, n in c.items():
        if w == v:
            continue
        n -= sub.get(w, 0)
        if n <= 0:
            continue
        tot += n
        s = n / tot_of[w] if tot_of is not None else n
        top.append((s, w))
        if s > bn:
            best, bn = (w, s)
    top.sort(key=lambda e: (-e[0], e[1]))
    if top and top[0][0] == bn:
        best = top[0][1]
    return (best, bn, tot, [w for _n, w in top[:CONS_TOPM]])

def cons_lenses(p, q):
    """The rows the mind may look through: its own visible values, deduped, in tape order.

    THE OUTPUT SPACE OF PHI, and the reason the invariant is safe here. It is not a vocabulary
    and not a candidate list - it is an index into what is already on the question's own place.
    """
    if '_cons_l' in q:
        return q['_cons_l']
    seen, out = (set(), [])
    for s in q['slots'][:q['query_row']]:
        v = p['tape'].values[s]
        if v not in seen:
            seen.add(v)
            out.append(v)
    out = out[:CONS_LENSES]
    q['_cons_l'] = out
    return out

def cons_rows_for(p, q, v):
    """Where the lens stands ELSEWHERE - the evidence for looking through it.

    Same import machinery as a candidate's evidence, pointed at a different question: not "here
    is a value, is it right" but "here is a row, is it worth following".
    """
    ix = reach_index(p)
    if ix is None:
        return []
    here = ix['of'].get(q['address'])
    rows = []
    js = [cons_place(p, q, v)] if CONS_RESOLVE == 'place' else [j for j, _n in ix['by_val'].get(v, ())]
    for j in js:
        if j is None or j == here:
            continue
        for w, rr, _c in ix['fills'][j]:
            if w == v:
                rows.extend(rr)
    return reach_rows_for(p, q, v, rows)

def cons_logits(net, p, q, device, bank):
    """Stage 1: say a value already here, or CONSTRAIN. Stage 2: which row to look through.

    The shape is reach_logits' shape on purpose - same worlds, same import budget, same
    lookahead rule, same refusal equalisation - so what is being compared between the two arms
    is the OPERATION and not the plumbing.
    """
    own = sorted({p['tape'].values[s] for s in q['slots'][:q['query_row']]})
    lens = cons_lenses(p, q)
    ev = {v: cons_rows_for(p, q, v) for v in lens}
    budget = min([IMPORT_K] + [len(ev[v]) for v in lens]) if lens else 0

    def world(value, rows, b, stage2=False):
        q.pop('_base', None)
        q['_stage2'] = stage2
        return reach_world(p, q, bank, device, value, rows, b)
    g = q.get('_cons_g')
    if g is None:
        g1 = [world(v, [], 0) for v in own]
        if not REACH_NO_REFUSE:
            g1.append(world(REFUSE_LABEL, [], 0))
        if not REACH_LOOKAHEAD:
            g1.append(world(REFUSE_LABEL, [s for v in lens for s in ev[v][:budget]], 10 ** 6))
        g2 = [world(v, ev[v], budget, True) for v in lens]
        if not REACH_NO_REFUSE or not lens:
            g2.append(world(REFUSE_LABEL, [s for v in lens for s in ev[v]], budget, True))
        g = (g1, g2)
        if q.get('_keep_g'):
            q['_cons_g'] = g
    g1, g2 = g
    l1 = torch.stack([net.phi(*x) for x in g1])
    l2 = torch.stack([net.phi(*x) for x in g2])
    if REACH_LOOKAHEAD:
        l1 = torch.cat([l1, l2.max().reshape(1)])
    return (l1, l2, own, lens)

def cons_answers(p, q, lens):
    """What the tape says through each lens, in lens order. The exact teacher."""
    return [cons_resolve(p, q, v)[0] for v in lens]

def cons_loss(net, p, q, device, bank):
    l1, l2, own, lens = cons_logits(net, p, q, device, bank)
    said = cons_answers(p, q, lens)
    ans = q['truth_value'] in set((x for x in said if x is not None))
    R1 = reach_reward(q, own + ([] if REACH_NO_REFUSE else [REFUSE_LABEL]), ans, device)
    names2 = said + ([REFUSE_LABEL] if not REACH_NO_REFUSE or not lens else [])
    R2 = reach_reward(q, [x if x is not None else REFUSE_LABEL for x in names2], ans, device)
    p1 = torch.softmax(l1, 0)
    p2 = torch.softmax(l2, 0)
    v2 = REACH_GAMMA * (p2 * R2).sum() - STEP_COST
    nk = 1
    return -((p1[:-nk] * R1).sum() + p1[-nk] * v2)

def cons_rivals(p, q, lens):
    """THE CLOSED RIVAL SET FOR A LENS, and every member is a count.

    A rival here is not another answer - the tape produces the answer either way - it is another
    RULE FOR CHOOSING THE ROW. Three, all exact, all free:

      rare      the own value standing at the fewest places. The lens that narrows the most.
      frequent  the own value standing at the most. The opposite rule, so the comparison cannot
                be won by whichever direction happened to be right.
      decisive  the own value whose top co-occurrence takes the largest SHARE of its own total -
                the lens the tape itself is most certain about. This is the strongest of the
                three by construction and the one the gate is set against.

    If the mind cannot beat these, choosing the query is a count and step 1 is closed.
    """
    ix = reach_index(p)
    if ix is None or not lens:
        return {}
    homes = {v: len(ix['by_val'].get(v, ())) for v in lens}
    dec = {}
    for v in lens:
        _b, bn, tot, _t = cons_resolve(p, q, v)
        dec[v] = bn / tot if tot else 0.0
    return {'rare': min(lens, key=lambda v: (homes[v], v)), 'frequent': max(lens, key=lambda v: (homes[v], v)), 'decisive': max(lens, key=lambda v: (dec[v], v))}

def cons_question(p, item, rng, hid):
    q = reach_question(p, item, rng, hid)
    if q is not None:
        q['verb'] = 'cons'
        q['cons'] = True
        q.pop('reach', None)
    return q

def cons_questions_for(p, r):
    out = []
    if reach_index(p) is None:
        return out
    for it in p['items']:
        for hid in range(len(it['slots'])):
            if (q := cons_question(p, it, r, hid)) is not None:
                out.append(q)
    if REACH_MAX_Q and len(out) > REACH_MAX_Q:
        out = r.sample(out, REACH_MAX_Q)
    return out
CONS_COLS = ('answerable', 'truth_in_own', 'n_lens', 'silent', 'mind_right', 'constrained', 'lens_idx', 'rare_right', 'frequent_right', 'decisive_right', 'present_topm', 'chosen_share', 'chosen_total', 'walk_answerable', 'walk_rival_right')
CIX = {n: i for i, n in enumerate(CONS_COLS)}

def reach_pick(q, l1, l2, own, cands, l3, lcands, keep_graph=False):
    """The mind's answer, which stage it came from, and how sure it was.

    ONE PLACE FOR THE ARGMAX. This logic used to live inline in the exam, and it is not simple:
    three stages, a two-way variant, and a deeper read that is the last option of stage two.
    337 needs the confidence per question and 338 asks the same of a place, so a second copy
    would have appeared - and a second copy of a staged argmax is exactly how a column starts
    reporting a plausible wrong number. Moved verbatim, nothing changed in what it decides.

    Returns (said, stepped, depth, score, margin):
      said    the filler, or REFUSE_LABEL when there is nothing to say
      stepped 0 stayed, 1 walked, 2 the line channel
      depth   how many reads the answer cost: 0 at home, 1 after a step, 2 after two
      score   Phi's raw value for the world it settled on
      margin  the gap to the runner-up IN THE DISTRIBUTION IT ANSWERED FROM. Not a softmax
              maximum: a max over a normalised vector rises when the option count falls, and
              one of 337's rivals IS the option count, so a probability would have smuggled
              the rival into the mind's own signal. The raw gap has no such term.
    """
    n1, n2 = reach_names(own, cands)
    nk = 2 if REACH_LINE else 1
    pick = int(l1.argmax())
    if TWO_WAY:
        if pick == 1 and len(l2):
            stepped, names, lgr = (1, n2, l2)
        else:
            stepped, names, lgr = (0, n1, q['_own_l'])
    elif pick == len(n1):
        stepped, names, lgr = (1, n2, l2)
    elif REACH_LINE and pick == len(n1) + 1 and (l3 is not None):
        stepped, names, lgr = (2, lcands, l3)
    else:
        stepped, names, lgr = (0, n1, l1[:-nk])
    dep = 1 if stepped else 0
    if stepped == 1 and REACH_DEPTH > 1:
        ld, dc, _dp = q.get('_deep', (None, [], []))
        if ld is not None and int(lgr.argmax()) == len(names):
            names, lgr, dep = (dc, ld, 2)
    said = names[int(lgr.argmax())] if len(names) else REFUSE_LABEL
    if len(lgr):
        if keep_graph:
            sv = lgr.sort(descending=True).values
            return (said, stepped, dep, sv[0], sv[0] - sv[1] if len(sv) > 1 else sv[0] - sv[0])
        sv = lgr.detach().sort(descending=True).values
        score = float(sv[0])
        margin = float(sv[0] - sv[1]) if len(sv) > 1 else 0.0
    elif keep_graph:
        z = l1.sum() * 0.0
        return (said, stepped, dep, z, z)
    else:
        score, margin = (0.0, 0.0)
    return (said, stepped, dep, score, margin)

def rank_auc(scores, labels):
    """Concordant pairs, ties counted half: the share of (positive, negative) pairs the
    ranking gets the right way round. A count over pairs, no threshold and nothing fitted -
    which is why it is the statistic for 337 rather than an accuracy at some chosen cut."""
    n = len(scores)
    if n != len(labels) or n == 0:
        return float('nan')
    npos = sum((1 for l in labels if l))
    nneg = n - npos
    if not npos or not nneg:
        return float('nan')
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    s = sum((r for r, l in zip(ranks, labels) if l))
    return (s - npos * (npos + 1) / 2.0) / (npos * nneg)
GATE_FRACTIONS = (0.05, 0.1, 0.25, 0.5)

def gate_top(scores, k):
    """The k questions a ranking would let through, as a set of indices.

    TIES BREAK BY POSITION, which is not a decision: the exam's question list is a SAMPLE, in
    sampled order, so position is already arbitrary. It matters because the counting rivals tie
    constantly - |own| is a small integer - and a gate cutting inside a tie is choosing at
    random among equals. That is a real limit of gating on a count, not a handicap imposed on
    it: a statistic that cannot separate two questions cannot gate between them. AUC in
    rankblock scores those ties at half credit and is the tie-fair companion to this.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return set(order[:max(0, min(k, len(scores)))])

def prec_at(scores, labels, k):
    """Of the k questions this ranking puts first, how many were really answerable."""
    n = len(scores)
    if k <= 0 or n == 0:
        return float('nan')
    order = sorted(range(n), key=lambda i: -scores[i])
    k = min(k, n)
    return sum((1 for i in order[:k] if labels[i])) / k

def reach_rival(p, q):
    """The same walk without the mind: nearest place, its most frequent filler. The margin is
    the gap to the next place, so it can be thresholded and allowed to stay silent too."""
    rc = reach_candidates(p, q)
    places = rc['places']
    if not places:
        return (None, -1.0)
    j0, _it, s0 = places[0]
    fills = reach_index(p)['fills'][j0]
    best = max(enumerate(fills), key=lambda e: (e[1][2], -e[0]))[1][0] if fills else None
    return (best, s0 - places[1][2] if len(places) > 1 else 1.0)

def reach_count_rival(p, q):
    """THE STRONGEST COUNTING RIVAL THE WALK ALLOWS: argmax of the EXACT count over the SAME
    candidates the mind is offered.

    Why it had to be built. `reach_rival` reads places[0] only - one place - while the mind
    scores candidates drawn from cand_places ~3.46. So on walk_only we have been comparing a
    search over eight candidates from three places against a one-place rule, and part of every
    margin is simply that the mind looked wider. That is a budget difference, not a mind, and
    it is the same two-factor mistake this project has already made three times (the aperture
    in 300, the window in 305, min_fillers in 305).

    This rival sees EXACTLY what the mind sees: the same `cands`, in the same order, produced by
    the same walk. It then applies the one rule counting has - take the value the tape wrote
    most often - summing the exact per-place counts from `fills`, which are counted over the
    whole place and never truncated. Nothing is fitted and nothing is approximate.

    It is deliberately the hardest version: exact counts, full candidate set, ties broken by the
    walk's own order (nearest place first), which is the tie-break most favourable to counting
    because the walk already ordered by similarity.
    """
    rc = reach_candidates(p, q)
    cands = list(rc['cands'])
    ix = reach_index(p)
    if ix is None:
        return (None, 0.0)
    places = list(rc['places'])
    if REACH_DEPTH > 1:
        _ld, _dc, _dp = q.get('_deep', (None, [], []))
        cands = cands + [c for c in _dc if c not in set(cands)]
        places = places + [pl for pl in _dp if pl[0] not in {j for j, _i, _s in places}]
    if not cands:
        return (None, 0.0)
    best, bshare, bcount, order = (None, -1.0, -1, {v: i for i, v in enumerate(cands)})
    seen, scored, ties = (set(cands), set(), Counter())

    def offer(v, sh, cnt):
        nonlocal best, bshare, bcount
        if sh > bshare:
            ties.clear()
        if sh == bshare:
            ties[v] += 1
        if (sh, cnt) > (bshare, bcount) or ((sh, cnt) == (bshare, bcount) and best is not None and (order[v] < order[best])):
            best, bshare, bcount = (v, sh, cnt)
    for j, _it, _sim in places:
        fills = ix['fills'][j]
        tot = sum((c for _v, _r, c in fills))
        if tot <= 0:
            continue
        for v, _rows, c in fills:
            if v not in seen:
                continue
            scored.add(v)
            offer(v, c / tot, c)
    for v in cands:
        if v in scored:
            continue
        j = rc.get('real_place', {}).get(v)
        if j is None:
            continue
        fills = ix['fills'][j]
        tot = sum((c for _v, _r, c in fills))
        if tot <= 0:
            continue
        cnt = sum((c for vv, _r, c in fills if vv == v))
        if cnt > 0:
            offer(v, cnt / tot, cnt)
    if isinstance(q, dict):
        q['_cr_ties'] = len(ties) + 1 if best is not None else 0
    return (best, bshare if best is not None else 0.0)

def reach_questions_for(p, r):
    """Every hidden filler the tape can offer, then capped by SAMPLING.

    A cap is needed and it is a cost bound, not a choice: a 3000-address frame tape gives ~9000
    questions per pack and each one costs a dozen graphs, so scoring all of them is hours of the
    same measurement. Sampled rather than truncated, for the reason 298 taught the hard way -
    a deterministic prefix is the same tape every time and the redraw stops meaning anything.
    """
    out = []
    if reach_index(p) is None:
        return out
    for it in p['items']:
        for hid in range(len(it['slots'])):
            if (q := reach_question(p, it, r, hid)) is not None:
                out.append(q)
    if REACH_MAX_Q and len(out) > REACH_MAX_Q:
        out = r.sample(out, REACH_MAX_Q)
    for it in p['items']:
        if len(it['slots']) >= 2 and (q := count_question(p, it)) is not None:
            out.append(q)
    return out
PAIR = False
PAIR_CANDS = 8
PAIR_MAX_ROWS = 6
PAIR_MAX_Q = 2000
PAIR_PER_LINE = 2
PAIR_BLIND = False
PAIR_FOLLOW = 2
PAIR_COLS = ('both_offered', 'mind_right', 'one_right', 'marg_right', 'joint_seen', 'joint_right', 'in_own_a', 'in_own_b', 'offered_a', 'offered_b', 'n_pairs', 'first_hole', 'world_rows', 'right_a', 'right_b', 'bag_seen', 'bag_right')
PIX = {n: i for i, n in enumerate(PAIR_COLS)}

def pair_offer(p, sub, partners):
    """What one hole may be filled with: its own values, the walk's, and the line's partners.

    Three sources because each is blind somewhere: own-only is 308's marginal rival with extra
    steps, the walk cannot see the confirmations, and neither can name a value whose only tie
    to this line is that it stands NEXT TO this line's visible values elsewhere - which is what
    a composed answer looks like from outside. Partners are counted off the tape (the bag index
    over co-line values, the question's own line subtracted), so the source belongs to the tape
    and the rivals pick from the same offer. Round-robin rather than fixed halves: whichever
    source is short gives its room to the others, and no ranking of mine orders the sources.
    """
    own = sorted({p['tape'].values[s] for s in sub['slots'][:sub['query_row']]})
    srcs = [list(own), list(reach_candidates(p, sub)['cands']), list(partners)]
    offer, seen = ([], set())
    while len(offer) < PAIR_CANDS and any(srcs):
        for src in srcs:
            while src and src[0] in seen:
                src.pop(0)
            if src and len(offer) < PAIR_CANDS:
                seen.add(src[0])
                offer.append(src.pop(0))
    return (offer, set(own))

def pair_question(p, sa, sb, rng):
    """Two holes of one line, each with its place's other rows, in one world.

    Order is by corpus position, so hole A is the earlier one in every world of this question
    and the cosine column the graph ranks against does not move between candidates.
    """
    ix = reach_index(p)
    if ix is None:
        return None
    subs = []
    for sl in (sa, sb):
        j = ix['of'].get(p['straddr'][sl])
        if j is None:
            return None
        it = ix['items'][j]
        if sl not in it['slots'] or len(it['slots']) < 2:
            return None
        sub = reach_question(p, it, rng, it['slots'].index(sl))
        if sub is None:
            return None
        subs.append(sub)
    holes, slots, vals, qrows, n_first = ([], [], [], [], 0)
    for k, (sl, sub) in enumerate(zip((sa, sb), subs)):
        rows = sub['slots'][:sub['query_row']][:PAIR_MAX_ROWS]
        slots.extend(rows)
        vals.extend((p['tape'].values[s] for s in rows))
        qrows.append(len(slots))
        slots.append(sl)
        vals.append(object())
        holes.append({'slot': sl, 'sub': sub, 'address': p['straddr'][sl], 'truth': p['tape'].values[sl], 'rows': rows})
        if k == 0:
            n_first = len(slots)
    return {'verb': 'pair', 'pair': True, 'slots': slots, 'vals': vals, 'holes': holes, 'query_rows': qrows, 'query_row': qrows[0], 'n_first': n_first, 'S': holes[0]['address'], 'S2': holes[1]['address'], 'line': p['line'][sa]}

def pair_offers(p, q):
    """The two offers, walked ON FIRST USE and never at question-build time.

    HALF OF AN OFFER COMES FROM THE WALK, so computing it eagerly puts a matmul over every place
    and an argsort into the construction of every question. The tape is redrawn every
    `--tape-period` steps - eighty times in a 4000-step run - and each redraw builds a few
    thousand questions of which fifty are ever scored, so an eager offer does about a hundred
    walks for every one that is read. That is 299a's stall exactly: not slow arithmetic, the
    same work done for questions nobody looks at. `reach_question` has always been cheap for
    this reason and the walk has always been lazy inside `reach_candidates`; the pair verb has
    to be built the same way, and this is where it is made so.
    """
    if '_pair_ev' in q:
        return q['holes']
    pair_joint_index(p)
    bag = p['_pairbag']
    here = set(q['slots'])
    addrs = {h['address'] for h in q['holes']}
    lslots = reach_line_index(p).get(q['line'], ())
    pool = [sl for sl in lslots if sl not in here and p['straddr'][sl] not in addrs]
    score = Counter()
    for sl in pool:
        for v, c in bag.get(p['tape'].values[sl], {}).items():
            score[v] += c
        for s2 in lslots:
            if s2 != sl and p['straddr'][s2] != p['straddr'][sl]:
                score[p['tape'].values[s2]] -= 1
    partners = [v for v, c in score.most_common(3 * PAIR_CANDS) if c > 0]
    ev = {}
    for h in q['holes']:
        if 'offer' not in h:
            h['offer'], h['own'] = pair_offer(p, h['sub'], partners)
        for v in h['offer']:
            if v not in ev:
                ev[v] = outside_mentions(p, q, v)
    q['_pair_ev'] = ev
    q['_pair_pool'] = pool
    q['_pair_b'] = min([IMPORT_K] + [len(ev[v]) + len(pool) for v in ev]) if ev else 0
    return q['holes']

def pair_world(p, q, bank, device, fills, qrow):
    """One world: the holes' rows, whatever is written into them, and the HOMES of what is
    written. `fills` maps hole index to a value.

    THE FILL CARRIES ITS EVIDENCE, and the first build's fault was exactly here - provable
    without running. With a fixed slot set the only term of the graph where the two filled
    values MEET is the `same` edge between the two query rows, i.e. va == vb, degenerate;
    every other input separates into per-hole contributions, so Phi's pair score FACTORISES
    into the product of its own marginals whatever the weights are. s1337 measured the
    consequence - 1 pair hit where the independent product predicts 1.6 - and the blind arm
    would have tied by construction. Not low power: an identity about the world.

    So a fill imports its value's homes, as the lookup verb has always done for a candidate.
    Worlds now differ in ROWS, and the second fill can agree or clash with the first through
    cos, rare and same on real evidence - which is the only place composition can live.

    SIZES STAY EQUAL inside a softmax: every fill imports exactly `_pair_b` rows, homes first,
    then the shared line pool, which is candidate-independent - the same equalisation the
    reach verb gives its refusal world, for the same reason.
    """
    slots, vals = (list(q['slots']), list(q['vals']))
    b, pool = (q['_pair_b'], q['_pair_pool'])
    for k in sorted(fills):
        v = fills[k]
        vals[q['query_rows'][k]] = v
        rows = q['_pair_ev'].get(v, [])[:b]
        rows = rows + [sl for sl in pool if sl not in rows][:max(0, b - len(rows))]
        for sl in rows:
            if sl not in slots:
                slots.append(sl)
                vals.append(p['tape'].values[sl])
    w = {'verb': 'lookup', 'S': q['S'], 'S2': q['S2'], 'slots': slots, 'vals': vals, 'query_row': qrow, 'query_rows': q['query_rows'], 'n_first': q['n_first']}
    return build_graph(p, w, bank, device, query_value=None, import_k=0)

def pair_logits(net, p, q, device, bank):
    """Stage 1: every value of every hole, each written into the world alone. Stage 2: the other
    hole, written into the world the winner of stage 1 left behind.

    The two stages are the SAME operation on the SAME world - which is the point. Nothing about
    the second fill is a new mechanism; it simply sees more.
    """
    g1 = []
    for k, h in enumerate(pair_offers(p, q)):
        for v in h['offer']:
            g1.append((k, v, pair_world(p, q, bank, device, {k: v}, q['query_rows'][k])))
    l1 = torch.stack([net.phi(*x[2]) for x in g1])
    return (l1, g1)

def pair_second(net, p, q, device, bank, k0, v0):
    """The other hole, scored in a world that already says `v0` - or, under PAIR_BLIND, in one
    that does not. That single dictionary entry is the whole of the composition claim."""
    k1 = 1 - k0
    h = pair_offers(p, q)[k1]
    base = {} if PAIR_BLIND else {k0: v0}
    g2 = [pair_world(p, q, bank, device, {**base, k1: v}, q['query_rows'][k1]) for v in h['offer']]
    return (torch.stack([net.phi(*x) for x in g2]), h['offer'], k1)

def pair_loss(net, p, q, device, bank):
    """Exact teacher at both stages, and the second stage is taught in the world the first one
    actually produced - under the mind's own distribution, not under the truth.

    Teacher forcing would train stage two on worlds it will never see: at exam time the first
    fill is the mind's, and a second fill that only works after a correct first is not
    composition, it is a lookup with a lucky prefix. So stage two is weighted by stage one's
    softmax - every first fill the mind gives weight to is followed through and priced.

    The reward is a COUNT: how many of the two holes the world got right, over two. An exact
    pair pays 1, one hole pays 0.5, neither pays 0. Nothing is tuned, and composing still pays
    strictly more than either half.
    """
    l1, g1 = pair_logits(net, p, q, device, bank)
    p1 = torch.softmax(l1, 0)
    truth = [h['truth'] for h in q['holes']]
    out = 0.0
    order = sorted(range(len(g1)), key=lambda i: -float(l1[i]))[:PAIR_FOLLOW]
    for i in order:
        k0, v0, _ = g1[i]
        l2, offer2, k1 = pair_second(net, p, q, device, bank, k0, v0)
        p2 = torch.softmax(l2, 0)
        r0 = 0.5 * float(v0 == truth[k0])
        R2 = torch.tensor([r0 + 0.5 * float(v == truth[k1]) for v in offer2], device=device)
        out = out + p1[i] * (p2 * R2).sum()
    rest = [i for i in range(len(g1)) if i not in set(order)]
    for i in rest:
        k0, v0, _ = g1[i]
        out = out + p1[i] * (0.5 * float(v0 == truth[k0]))
    return -out

def pair_rivals(p, q):
    """Counting's two ways at this question, both exact.

    MARGINAL: each hole answered on its own by the most frequent value of its place. This is the
    product of two marginals and it is the rival composition has to beat, because it is what a
    perfect index does when the holes are treated separately.

    JOINT: did this exact pair ever stand together at these two places on another line? If it
    did, counting has the pair outright and no composition is required. The subset that decides
    is where BOTH are blind.
    """
    out = []
    for h in q['holes']:
        c = Counter((p['tape'].values[s] for s in h['sub']['slots'][:h['sub']['query_row']]))
        out.append(c.most_common(1)[0][0] if c else None)
    ix = pair_joint_index(p)
    a, b = q['holes']
    seen = ix.get((a['address'], b['address']), {})
    truth = (a['truth'], b['truth'])
    best, n_best = (None, 0)
    for (va, vb), n in seen.items():
        n -= (va, vb) == truth
        if n > n_best:
            best, n_best = ((va, vb), n)
    bag = p['_pairbag']
    lslots = reach_line_index(p).get(q['line'], ())
    adj = Counter()
    for i, s1 in enumerate(lslots):
        for s2 in lslots[i + 1:]:
            if p['straddr'][s1] != p['straddr'][s2]:
                v1, v2 = (p['tape'].values[s1], p['tape'].values[s2])
                adj[v1, v2] += 1
                adj[v2, v1] += 1

    def bagn(x, y):
        return bag.get(x, {}).get(y, 0) - adj.get((x, y), 0)
    pair_offers(p, q)
    bb, bn = (None, 0)
    for va in a['offer']:
        for vb in b['offer']:
            n = bagn(va, vb)
            if n > bn:
                bb, bn = ((va, vb), n)
    return (tuple(out), best, seen.get(truth, 0) > 1, bb, bagn(*truth) > 0)

def pair_joint_index(p):
    """Every (place, place) -> (value, value) the tape actually wrote on one line, counted once
    per pack - the joint statistic counting would need, given to the rival in full. The same
    pass also counts the BAG: value -> values it shares a line with anywhere, at ANY places.
    The bag is the third counting rival and the partner source, so it is one loop, not three."""
    ix = p.get('_pairjoint')
    if ix is not None:
        return ix
    by_line = defaultdict(list)
    for sl, li in enumerate(p['line']):
        if li >= 0:
            by_line[li].append(sl)
    ix = defaultdict(Counter)
    bag = defaultdict(Counter)
    for li, ss in by_line.items():
        for i, s1 in enumerate(ss):
            for s2 in ss[i + 1:]:
                a1, a2 = (p['straddr'][s1], p['straddr'][s2])
                if a1 == a2:
                    continue
                v1, v2 = (p['tape'].values[s1], p['tape'].values[s2])
                if p['pos'][s1] <= p['pos'][s2]:
                    ix[a1, a2][v1, v2] += 1
                else:
                    ix[a2, a1][v2, v1] += 1
                bag[v1][v2] += 1
                bag[v2][v1] += 1
    p['_pairjoint'] = ix
    p['_pairbag'] = dict(bag)
    return ix

def pair_questions_for(p, r):
    """Two holes of one line, far enough apart that neither frame can cover the other's token."""
    out = []
    if reach_index(p) is None or p.get('pos') is None:
        return out
    keep = {s for it in p['items'] for s in it['slots']}
    by_line = defaultdict(list)
    for sl in keep:
        if p['line'][sl] >= 0 and p['pos'][sl] >= 0:
            by_line[p['line'][sl]].append(sl)
    for li, ss in by_line.items():
        if len(ss) < 2:
            continue
        ss.sort(key=lambda s: p['pos'][s])
        cand = [(a, b) for i, a in enumerate(ss) for b in ss[i + 1:] if p['straddr'][a] != p['straddr'][b] and p['pos'][b] - p['pos'][a] > FRAME_MAX]
        if not cand:
            continue
        r.shuffle(cand)
        for a, b in cand[:PAIR_PER_LINE]:
            if (q := pair_question(p, a, b, r)) is not None:
                out.append(q)
    if PAIR_MAX_Q and len(out) > PAIR_MAX_Q:
        out = r.sample(out, PAIR_MAX_Q)
    return out

def open_rival_scored(p, q, bank, device):
    """Whole-tape retrieval, with the confidence it needs to be allowed to stay silent.

    Same rule and same rows as open_rival_cos - kept separate so 292's arms stay bit-identical.
    The margin is the gap from the winning value to the best row of any OTHER value, over the
    spread of the column: a shape, not a magnitude, so no absolute scale is smuggled in.
    """
    ck = p.setdefault('_ctx', {})

    def ctx(sl):
        if sl not in ck:
            c = bank.ctx_fp(p['texts'][sl], exclude=p['tape'].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
        return ck[sl]
    qc = ctx(q['slots'][q['query_row']])
    if qc is None:
        return (None, float('nan'))
    k = shared_import_budget(p, q, list(q['cands']))
    rows, owner = ([], [])
    for c in q['cands']:
        if c == REFUSE_LABEL:
            continue
        for sl in outside_mentions(p, q, c)[:k]:
            r = ctx(sl)
            if r is not None:
                rows.append(r)
                owner.append(c)
    if not rows:
        return (None, float('nan'))
    sims = (torch.stack(rows, 0) @ qc).tolist()
    top = max(range(len(sims)), key=lambda i: sims[i])
    best = owner[top]
    other = [s for s, o in zip(sims, owner) if o != best]
    spread = max(sims) - min(sims)
    if not other:
        return (best, 1.0)
    return (best, (sims[top] - max(other)) / spread if spread > 1e-09 else 0.0)

def lookup_mixed_question(p, item, rng, hid, all_values):
    """One exam question: four drawn values plus refusal, and half the time the truth is not
    among them. Everything else is 294's open question unchanged."""
    q = lookup_open_uniform(p, item, rng, hid, all_values)
    if q is None:
        return None
    truth = q['cands'][q['label']]
    cands = list(q['cands'])
    answerable = rng.random() < 0.5
    if not answerable:
        here = {p['tape'].values[s] for s in item['slots']} | set(cands)
        repl = None
        for _ in range(256):
            v = all_values[rng.randrange(len(all_values))]
            if v not in here:
                repl = v
                break
        if repl is None:
            return None
        cands[cands.index(truth)] = repl
    cands = sorted(cands + [REFUSE_LABEL])
    q['cands'] = cands
    q['label'] = cands.index(truth if answerable else REFUSE_LABEL)
    q['answerable'] = answerable
    q['truth_value'] = truth
    q['mixed'] = True
    for dead in ('open', 'uniform', 'bucket_of', '_base', '_ibudget'):
        q.pop(dead, None)
    if shared_import_budget(p, q, list(q['cands'])) < 1:
        return None
    return q

def mixed_payoff(silent, right, answerable):
    """280's payoff, and the only place the two abilities are weighed against each other.

    Silence on an unanswerable question is not a hedge, it is the correct answer, so it pays
    what a correct answer pays. Silence on an answerable one is the hedge and pays 0.75, which
    is what makes answering worth it above 0.875 confidence - derived, never chosen.
    """
    if silent:
        return 1.0 if not answerable else 0.75
    return 1.0 if right else -1.0

def open_rival_cos(p, q, bank, device):
    """The rival 292 actually has to beat: retrieval over the WHOLE TAPE, not over the address.

    Once every candidate brings its own mentions in, similarity is back in the game - it just
    searches the corpus instead of the address. That is RAG, stated exactly, and it is the fork
    the project has been circling: if nearest-imported-context lands where Phi lands, what we
    built is a search engine with extra steps.

    Same rows Phi is given - the shared import budget - and one rule: the candidate with a
    mention whose context is nearest the query's.
    """
    ck = p.setdefault('_ctx', {})

    def ctx(sl):
        if sl not in ck:
            c = bank.ctx_fp(p['texts'][sl], exclude=p['tape'].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
        return ck[sl]
    qc = ctx(q['slots'][q['query_row']])
    if qc is None:
        return None
    k = shared_import_budget(p, q, list(q['cands']))
    rows, owner = ([], [])
    for c in q['cands']:
        for sl in outside_mentions(p, q, c)[:k]:
            r = ctx(sl)
            if r is not None:
                rows.append(r)
                owner.append(c)
    if not rows:
        return None
    return owner[int((torch.stack(rows, 0) @ qc).argmax())]

def neighbourhood_audit(p, k, ks=(1, 3, 6, 12)):
    """DOES THE NEIGHBOURHOOD CONTAIN THE ANSWER? Per route, per k, before any training.

    290 answered its own falsifier sideways. Phi beat counting at z +2.53 and came within 0.012
    of the retrieval gate - on twenty-one held-out questions, because only twenty-one of roughly
    three hundred were answerable at all. 291's smoke says the same number from the other side:
    unanswerable_rate 0.931.

    That is not a training problem and no objective fixes it. Under 280's rewards, answering is
    worth it exactly when p(correct) > 0.875, so at a 7% hit rate a mind that refuses everything
    is not broken - it is CORRECT, and expected-reward training would arrive at the same policy
    faster. The lever is upstream: how often N(a) holds the value written at a.

    That is a property of the TAPE, computable with no model and no gradient, so it costs
    seconds instead of an hour and it says which route is carrying facts and which is carrying
    noise. rel_nonzero_rate was 0.896 in the full run - the relation channel fires on nine pairs
    in ten - and relations here are words like "and" and "the", so the suspicion to test is that
    the route producing most of the rows produces none of the answers.
    """
    out = {}
    for kk in sorted({k} | set(ks)):
        out[f'k={kk}'] = _audit_at(p, kk)
    return out

def _audit_at(p, k):
    out = {}
    for name, routes in (('anchor', ('anchor',)), ('rel', ('rel',)), ('word', ('word',)), ('anchor+word', ('anchor', 'word')), ('all', ('anchor', 'rel', 'word'))):
        n = ans = rows = 0
        for it in p['items']:
            if not 1 <= len(it['slots']) <= 2:
                continue
            for hid in range(len(it['slots'])):
                own = list(it['slots'])
                hidden = own[hid % len(own)]
                truth = p['tape'].values[hidden]
                rr = [x for x in own if x != hidden]
                for b in neighbourhood(p, it['address'], k, routes):
                    rr += list(p['_addr_index']['slots'].get(b, ()))[:k]
                rr = sorted(set(rr) - {hidden})
                if len(rr) < 1 or len({p['tape'].values[x] for x in rr}) < 2:
                    continue
                n += 1
                rows += len(rr)
                ans += int(truth in {p['tape'].values[x] for x in rr})
        out[name] = {'questions': n, 'answerable': ans, 'hit_rate': ans / n if n else float('nan'), 'mean_rows': rows / n if n else float('nan')}
    return out
IDENTITY = False
IDENT_CORE = 3
IDENT_CANDS = 4
IDENT_VALUES = 'hide'
IDENT_IMPORT = 0
IDENT_TAU = 0.9
IDENT_OVERLAP = 2
IDENT_SUPPLY = Counter()
IDENT_RIVALS = ('cos1nn', 'heur', 'rare')

def str_parts(s):
    """anchor, relation-content-words of a pre-grouping string address."""
    a, r = (s.split('|', 1) + [''])[:2]
    return (a, {w.lower() for w in s279.REL_RE.findall(r) if w.lower() not in s279.VALUE_STOP})

def ident_index(p):
    """The tape read as places rather than as addresses. None when the pack predates `straddr`."""
    ix = p.get('_ident')
    if ix is not None:
        return ix
    sa = p.get('straddr')
    if sa is None:
        return None
    live = sorted({sl for it in p['items'] for sl in it['slots']})
    parts, by_str, by_anc = ({}, defaultdict(list), defaultdict(list))
    by_place = defaultdict(list)
    for sl in live:
        an, rw = str_parts(sa[sl])
        parts[sl] = (an, rw)
        by_str[sa[sl]].append(sl)
        by_anc[an].append(sl)
        by_place[an, p['tape'].values[sl]].append(sl)
    krow = {sl: i for i, sl in enumerate(p['slot_keys_slot'])}
    ix = {'parts': parts, 'by_str': dict(by_str), 'by_anc': dict(by_anc), 'by_place': dict(by_place), 'krow': krow, 'words': {sl: set(context_words(p['texts'][sl], exclude=p['tape'].values[sl])) for sl in live}, 'swords': {sl: {w.lower() for w in s279.REL_RE.findall(p['texts'][sl]) if w.lower() not in s279.VALUE_STOP} - {p['tape'].values[sl].lower()} for sl in live}}
    p['_ident'] = ix
    return ix

def ident_cos(p, ix, a, b, which='ctx'):
    K = p['ctx_keys'] if which == 'ctx' else p['anc_keys']
    ka, kb = (ix['krow'].get(a), ix['krow'].get(b))
    if K is None or ka is None or kb is None:
        return float('nan')
    return float(K[ka] @ K[kb])

def identity_question(p, anchor, value, truth_slot, rng):
    """One question: here are some mentions of one place - which of four joins them?

    The core is the other mentions of this (anchor, value); the truth is the one held out, and
    it has to say it in DIFFERENT words - relation words disjoint from the core's, and not the
    core's string address - or shared-word overlap finds it and nothing has been asked. The
    intruders are the same anchor with a different value under the same disjointness rule, so
    all four candidates stand at the same distance from the core in words and only the reading
    separates them.

    They are ranked nearest-first by ink, so the impostors are the hardest the tape can supply,
    and then the four are shuffled: 292's rungs were readable off their construction and that is
    what produced its inverted landscape, on three seeds out of three.
    """
    ix = ident_index(p)
    if ix is None:
        return None
    sibs = [s for s in ix['by_place'].get((anchor, value), ()) if s != truth_slot]
    if not sibs:
        IDENT_SUPPLY['no_sibling'] += 1
        return None
    core = sibs[:IDENT_CORE]
    core_rel, core_str = (set(), {p['straddr'][s] for s in core})
    for s in core:
        core_rel |= ix['parts'][s][1]
    if ix['parts'][truth_slot][1] & core_rel or p['straddr'][truth_slot] in core_str:
        IDENT_SUPPLY['same_words'] += 1
        return None
    pool = [s for s in ix['by_anc'].get(anchor, ()) if s not in core and s != truth_slot and (p['tape'].values[s] != value) and (not ix['parts'][s][1] & core_rel)]
    if len(pool) < IDENT_CANDS - 1:
        IDENT_SUPPLY['no_intruders'] += 1
        return None
    pool.sort(key=lambda s: -max((ident_cos(p, ix, s, c) for c in core)))
    cands = [truth_slot] + pool[:IDENT_CANDS - 1]
    rng.shuffle(cands)
    IDENT_SUPPLY['built'] += 1
    return {'verb': 'lookup', 'ident': True, 'S': anchor, 'place': [anchor, value], 'address': f'place:{anchor}|{value}', 'hid': truth_slot, 'straddr': p['straddr'][truth_slot], 'slots': list(core), 'vals': [p['tape'].values[s] for s in core], 'cand_slots': cands, 'cands': [f's{c}' for c in cands], 'label': cands.index(truth_slot)}

def ident_budget(p, q):
    """One import budget for all four worlds - the minimum any candidate can supply."""
    b = q.get('_ibudget')
    if b is None:
        ix = ident_index(p)
        b = min([IDENT_IMPORT] + [len([s for s in ix['by_str'].get(p['straddr'][c], ()) if s != c]) for c in q['cand_slots']])
        q['_ibudget'] = b
    return b

def identity_world(p, q, slot):
    """The core plus one candidate mention, scored as one world by the same Phi.

    No new channel and no new head: a place is a set of rows, and Phi already says how well a
    set of rows hangs together. The candidate row is marked as the query row - the same bit a
    completed lookup world carries - so a proposed member is never mistaken for an observed
    one, and the mark is identical across the four worlds, so it cannot carry the label.
    """
    slots = list(q['slots']) + [slot]
    if IDENT_IMPORT:
        ix = ident_index(p)
        sibs = [s for s in ix['by_str'].get(p['straddr'][slot], ()) if s not in slots]
        slots += sibs[:ident_budget(p, q)]
    if IDENT_VALUES == 'hide':
        vals = [object() for _ in slots]
    else:
        vals = [p['tape'].values[s] for s in slots]
    return {'verb': 'lookup', 'ident': True, 'S': q['S'], 'slots': slots, 'vals': vals, 'query_row': len(slots) - 1, 'cands': q['cands'], 'label': q['label']}

def ident_rivals(p, q):
    """The three rules the mind has to beat. Two are already in the project; one IS the tape.

    cos1nn   nearest candidate to any core row by the write ink - the same 1-NN that beat Phi
             in §18 and the honest ceiling of the encoder alone.
    heur     fp_addresses' own decision, reproduced exactly: min(anchor, context) cosine over
             tau AND at least `overlap` shared content words with some core row. This is the
             rule 293 exists to replace, so it is scored as a rival rather than described.
    rare     the discrete channel by itself: most shared rare words with the core.
    """
    ix = ident_index(p)
    core = q['slots']
    out, best = ({}, {})
    accept = []
    for name in ('cos1nn', 'heur', 'rare'):
        best[name] = (float('-inf'), None)
    med = p.get('_median')
    if med is None:
        lens = sorted((len(v) for v in p['postings'].values()))
        med = lens[len(lens) // 2] if lens else 1
        p['_median'] = med
    for c in q['cand_slots']:
        cw = ix['words'][c]
        sc = max((ident_cos(p, ix, c, s) for s in core))
        rare = max((sum((1 for w in cw & ix['words'][s] if len(p['postings'].get(w, ())) < med)) / max(1, min(len(cw), len(ix['words'][s]))) for s in core))
        sw = ix['swords'][c]
        linkable = [s for s in core if len(sw & ix['swords'][s]) >= IDENT_OVERLAP]
        two = max((min(ident_cos(p, ix, c, s, 'anc'), ident_cos(p, ix, c, s)) for s in linkable), default=float('-inf'))
        if two >= IDENT_TAU:
            accept.append(c)
        for name, v in (('cos1nn', sc), ('rare', rare), ('heur', two)):
            if v > best[name][0]:
                best[name] = (v, c)
    for name, (v, c) in best.items():
        out[name] = f's{c}' if c is not None and (name != 'heur' or v >= IDENT_TAU) else None
    out['_heur_accepted'] = len(accept)
    return out

def identity_questions_for(p, r):
    """Every place the tape can put on trial, plus count and compare as the sanity bolt."""
    ix = ident_index(p)
    out = []
    if ix is None:
        return out
    for (anchor, value), slots in sorted(ix['by_place'].items()):
        if len(slots) < 2:
            IDENT_SUPPLY['singleton_place'] += 1
            continue
        for sl in slots:
            if (q := identity_question(p, anchor, value, sl, r)) is not None:
                out.append(q)
    for it in p['items']:
        if len(it['slots']) >= 2 and (q := count_question(p, it)) is not None:
            out.append(q)
    return out

def identity_audit(p, r):
    """The minute that decides whether the hour is worth spending. No model, no gradient.

    Two numbers settle it. SUPPLY: how many places the tape can put on trial - §19 and 291 both
    died of a denominator and it costs nothing to look first. CEILING: what the three rules
    already score. A rival near 1.0 means the question is decided before the mind is asked and
    the construction has to change; a rival near the floor means the question may be undecidable
    from this evidence, which is equally worth knowing before training anything.
    """
    IDENT_SUPPLY.clear()
    qs = [q for q in identity_questions_for(p, r) if q.get('ident')]
    tot, ixp = (Counter(), (ident_index(p) or {'parts': {}})['parts'])
    for q in qs:
        truth = q['cands'][q['label']]
        rv = ident_rivals(p, q)
        tot['n'] += 1
        tot['heur_accepted'] += rv.pop('_heur_accepted')
        tot['core_rows'] += len(q['slots'])
        for name, pick in rv.items():
            tot[name] += int(pick == truth)
            tot[f'{name}_answered'] += int(pick is not None)
        cv = {p['tape'].values[s] for s in q['slots']}
        hits = [c for c in q['cand_slots'] if p['tape'].values[c] in cv]
        tot['value_decides'] += int(len(hits) == 1 and f's{hits[0]}' == truth)
        crel = set()
        for s in q['slots']:
            crel |= ixp[s][1]
        tot['word_leak'] += int(any((ixp[c][1] & crel for c in q['cand_slots'])))
    n = max(1, tot['n'])
    return {'n_questions': tot['n'], 'floor': 1.0 / IDENT_CANDS, 'mean_core_rows': tot['core_rows'] / n, 'supply': dict(IDENT_SUPPLY), 'rival_cos1nn': tot['cos1nn'] / n, 'rival_rare': tot['rare'] / n, 'rival_heuristic': tot['heur'] / n, 'heuristic_answered': tot['heur_answered'] / n, 'heuristic_mean_accepted': tot['heur_accepted'] / n, 'value_identifies_truth': tot['value_decides'] / n, 'word_overlap_leak': tot['word_leak'] / n}
DROPPED = [0, 0]
VIEWS = 1
VIEW_MODE = 'thin'

def view_of(q, rng, keep_p):
    """One VIEW of a question, for reconciliation (ROADMAP 20): thin the redundancy, never the
    candidate set.

    drop_rows is the wrong instrument for views and the difference is a tell. It recomputes the
    candidate set from the surviving rows while guarding only the TRUE value's row - so across
    views the truth would always keep a witness while wrong candidates could lose all of theirs,
    and "which candidate kept support in every view" would leak the label. And with import_k=0 a
    candidate absent from a view's rows is the ladder collision again: every absent value gives
    the same graph.

    A view therefore keeps at least one row for EVERY candidate - "a world where every claim
    keeps a witness" - and thins only the redundancy above that. The candidate set, the label
    and the logit ordering are identical across views, which is what lets their logits be pooled
    by a plain mean. Marginalisation, not noise: a corpus that repeated things less would have
    written exactly such a world.
    """
    qr = q['query_row']
    keep = [i for i in range(qr) if rng.random() < keep_p]
    kept = {q['vals'][i] for i in keep}
    for c in q['cands']:
        wit = [i for i in range(qr) if q['vals'][i] == c]
        if c not in kept and wit:
            keep.append(rng.choice(wit))
    keep = sorted(set(keep))
    out = {**q, 'slots': [q['slots'][i] for i in keep] + [q['slots'][qr]], 'vals': [q['vals'][i] for i in keep] + [q['vals'][qr]], 'query_row': len(keep)}
    out.pop('ladder', None)
    out.pop('_base', None)
    return out

def region_views_of(q, n):
    """Views on DIFFERENT STRETCHES of the tape, not resamples of the same one.

    The evidence rows of a question sit in write order: corpus_assertions walks the lines top
    to bottom, fp_addresses keeps members in ascending original index, pack_from_corpus appends
    in that order, and lookup_question's `slots[:hid] + slots[hid+1:]` preserves it. So row
    order within an address IS the order the corpus said these things in (per phrasing block,
    when fp grouping merged several phrasings - still a systematic coordinate, never a draw).

    Cut that order into min(n, rows) contiguous stretches - the old part of the tape, the
    middle, the new - each view a world written by one stretch of the corpus. Deterministic:
    no rng anywhere, the views are a property of the tape. A region may lack candidates the
    full question has; that is the POINT (a region that never heard of a value has no opinion
    on it) and pool_views handles it by masking, not by moving rows across regions - which is
    exactly what view_of's per-candidate guarantee did and what made its views resamples.
    """
    qr = q['query_row']
    v = min(n, qr)
    cuts = [round(i * qr / v) for i in range(v + 1)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        o = {**q, 'slots': q['slots'][a:b] + [q['slots'][qr]], 'vals': q['vals'][a:b] + [q['vals'][qr]], 'query_row': b - a}
        o.pop('ladder', None)
        o.pop('_base', None)
        out.append(o)
    return out

def views_and_mask(q, rng, device):
    """The question's views plus the candidate-presence mask, under either mode.

    View 0 is always the FULL question, so at VIEWS=1 this is exactly the old single pass and
    the ensemble is a strict superset of the information the single pass had. The mask says
    which candidates each region actually has a witness for; None in thin mode, where view_of
    guarantees every candidate a witness and no masking is needed (or possible - that guarantee
    is what keeps thin views label-tight).
    """
    if VIEW_MODE == 'thin':
        return ([q] + [view_of(q, rng, 1.0 - ROW_DROPOUT) for _ in range(VIEWS - 1)], None)
    qvs = [q] + region_views_of(q, VIEWS)
    M = torch.tensor([[float(c in set(v['vals'][:v['query_row']])) for c in q['cands']] for v in qvs[1:]], device=device)
    return (qvs, M)

def pool_views(L, M):
    """One pooling, used by the training loss, the probe loss and the exam alike - the meter
    and the objective must be the same computation or the curve measures nothing (HANDOFF 9b).

    Thin mode (M is None): plain mean of logits = normalised product of the per-view
    distributions. Unchanged from recon3, bit for bit.

    Region mode: centered log-linear pooling with abstention,

        pooled(c) = (phi_0(c) - mean_c' phi_0(c'))
                  + sum_v M_v(c) * (phi_v(c) - mean_{c' in v} phi_v(c'))

    Centering is derived, not chosen: it is the unique per-view shift that makes the pooled
    score invariant to adding a constant to any one view's logits, so a region with no witness
    for c contributes its AVERAGE opinion - exactly nothing - rather than a vote against.
    Absence of evidence in one stretch of the tape is not evidence against, because the
    stretches partition the rows: every candidate keeps a witness somewhere. A corollary worth
    stating: a region whose rows all carry one value centers to zero and pools nothing - pure
    support counts stay the tape's own channel (the counting rival already has them), and only
    CONTRASTIVE reading pools. Zero parameters throughout.
    """
    if M is None:
        return L.mean(0)
    mu = (L[1:] * M).sum(1) / M.sum(1)
    return L[0] - L[0].mean() + ((L[1:] - mu.unsqueeze(1)) * M).sum(0)

def disagreement(L, mask=None):
    """Generalised Jensen-Shannon divergence of the per-view answer distributions: the mean KL
    of each view to their mixture. Zero when the views agree exactly; label-free by
    construction, so thresholding on it is never conditioning on the outcome.

    With a mask (region mode), each view's distribution lives on ITS candidates - masked
    softmax puts exact zeros elsewhere, and the JS stays finite because 0*log(0/m) = 0 and the
    mixture covers every candidate some view supports. Two regions that put their mass on
    values the other never wrote disagree maximally, which is correct: that address is
    contested across the corpus, and D is the number that says so."""
    if mask is not None:
        L = L.masked_fill(mask == 0, float('-inf'))
    P = torch.softmax(L, dim=1)
    m = P.mean(0).clamp_min(1e-09)
    return float((P * (P.clamp_min(1e-09).log() - m.log())).sum(1).mean())

def reconciled(net, p, q, device, bank, rng):
    """Pooled logits, the single full-pass logits, and D, for one question. Training takes the
    gradient through the pooled logits; the exam reads all three. In thin mode D is over all
    views (view 0 included, as recon3 measured it); in region mode D is over the REGIONS only -
    the full view is their union and would only dilute the cross-region signal."""
    qvs, M = views_and_mask(q, rng, device)
    L = torch.stack([cand_logits_for(net, p, qv, device, bank) for qv in qvs])
    return (pool_views(L, M), L[0], disagreement(L if M is None else L[1:], M))
ROW_DROPOUT = 0.0

def drop_rows(q, rng, keep_p):
    """The same fact, as a thinner corpus would have written it.

    Every tape in a run has the same density - about 2.9 mentions per address - so the mind
    never sees one fact with two witnesses and then the same fact with five. That is exactly
    the axis it fails on: high margin against low margin IS "the witnesses are many and agree"
    against "the witnesses are few", and qrank_big measured opposite behaviour in the two halves
    while training on one density only.

    This is marginalisation, not noise. A subset of the evidence rows is a world the corpus
    could have produced, and the mind has to be right at every density rather than at the one
    the sampler happens to hand it. Nothing is invented and no value is altered.

    The two conditions are answerability, not nudges - they are the same conditions
    lookup_question already imposes: the true value must still be on some surviving row, and
    there must still be at least two candidates to choose between. Weighting is untouched
    because the drop happens inside an already-chosen question, so crowded addresses gain no
    extra share.

    Training only. The held-out tape is never thinned, or its power would be fictional.
    """
    qr = q['query_row']
    truth = q['cands'][q['label']]
    surv = list(range(qr))
    if not any((q['vals'][i] == truth for i in surv)):
        return None
    keep = [i for i in surv if rng.random() < keep_p]
    if not any((q['vals'][i] == truth for i in keep)):
        keep.append(rng.choice([i for i in surv if q['vals'][i] == truth]))
    keep = sorted(set(keep))
    cands = sorted({q['vals'][i] for i in keep})
    if REFUSE_LABEL in q['cands']:
        cands = cands + [REFUSE_LABEL]
    if len(cands) < 2:
        return None
    DROPPED[0] += len(keep)
    DROPPED[1] += len(surv)
    out = {**q, 'slots': [q['slots'][i] for i in keep] + [q['slots'][qr]], 'vals': [q['vals'][i] for i in keep] + [q['vals'][qr]], 'cands': cands, 'label': cands.index(truth), 'query_row': len(keep)}
    out.pop('ladder', None)
    out.pop('_base', None)
    return out
LADDER = ('near', 'middle', 'far')
LADDER_ON = True
IMPORT_K = 2
EDGES = ('same', 'cos', 'rare')
EDGES_NB = ('anchor', 'rel')
EDGES_ON = set(EDGES)
INK = 'mean'
FP = 'arc'
WORDS = 'ascii'
ARC_TRAIN_DENSITY = 2.9701

def tau_for_density(target, iters, log, lo=0.0, hi=0.9995):
    """A tau that writes a tape of the intended DENSITY, for whichever ink is doing the writing.

    279's tau is an absolute cosine. Hash ink and bigram ink have completely different similarity
    distributions, so at 0.90 almost nothing merges: addresses shatter, multi-mention addresses
    vanish, and lookup questions fall from hundreds to single digits. An arm run that way compares
    "a different tape read differently" against "this tape read this way" and the threshold, not
    the ink, is what moved. The code has carried that warning as a comment since the ink patch;
    this is the instrument it asks for.

    Density - mentions per address - is the right invariant to hold fixed, because it is what the
    question generator consumes: lookup_question needs three mentions and two distinct values, so
    density decides how many questions exist and how many candidates each one has. Matching it
    means the two tapes ask comparably hard questions, and any remaining difference in
    FRAGMENTATION (same_anchor_diff_relation, mean_addresses_per_fact) is the ink's doing rather
    than the threshold's. Matching the threshold instead would guarantee the opposite.

    Bisection, because density is decreasing in tau: a higher bar merges less, groups stay small,
    and the min_mentions filter drops them. "Decreasing" is CHECKED at the bracket ends and
    logged, never assumed - the min_mentions filter is not monotone in principle and a silent
    bisection over a non-monotone curve would return a plausible number that means nothing.

    The bracket spans the whole ACHIEVABLE range rather than a plausible-looking window: tau 0 is
    "merge everything the discrete channel allows" - the word-overlap conjunction still applies,
    so it is a real limit and not a degenerate one - and it is where density is maximal. A
    narrower low end would look reasonable and silently clamp on an ink whose cosines run low,
    which is exactly the ink this instrument exists for.

    Calibrated ONCE, on the first pack's assertions, then frozen: tau is a property of the writing
    rule, not of a draw, and re-deriving it per resample would make the tape stop being one tape.
    Cheap despite the repeats because CachedBank memoises the ink - iterations after the first
    change only the threshold and recompute no fingerprints.
    """
    memo = {}

    def resolve(asserts, bank, overlap, min_mentions, addr_key):
        if 'tau' in memo:
            return memo['tau']

        def density(t):
            out, addrs = s279.fp_addresses(asserts, bank, t, overlap, min_mentions, addr_key=addr_key)
            return (len(out) / len(addrs) if addrs else float('nan'), len(addrs))
        t0 = time.time()
        d_lo, n_lo = density(lo)
        d_hi, n_hi = density(hi)
        trace = [(lo, d_lo, n_lo), (hi, d_hi, n_hi)]
        monotone = d_lo > d_hi
        a, b = (lo, hi)
        if not monotone:
            log(f'  tau calibration: density NOT decreasing in tau ({lo}->{d_lo:.3f}, {hi}->{d_hi:.3f}) - bisection is not valid here')
        elif not d_hi <= target <= d_lo:
            log(f'  tau calibration: target {target:.4f} outside the bracket [{d_hi:.3f}, {d_lo:.3f}] - clamping to the nearer end')
        else:
            for _ in range(iters):
                m = 0.5 * (a + b)
                d_m, n_m = density(m)
                trace.append((m, d_m, n_m))
                if d_m > target:
                    a = m
                else:
                    b = m
        best = min(trace, key=lambda r: abs(r[1] - target) if r[1] == r[1] else float('inf'))
        memo['tau'] = best[0]
        memo['trace'] = [{'tau': round(t, 5), 'density': d, 'addresses': n} for t, d, n in trace]
        memo['achieved'] = best[1]
        memo['monotone'] = monotone
        log(f'  tau calibrated: {best[0]:.4f} -> density {best[1]:.4f} (target {target:.4f}, {best[2]} addresses, {len(trace)} probes, {time.time() - t0:.0f}s)')
        return best[0]
    resolve.memo = memo
    return resolve
RARE_NNZ = [0, 0]
FAST_COS = True
RESAMPLE_OVERLAP = []
NB_NNZ = [0, 0, 0]
GRAPH_N = [0, 0, 0]
COS_SPREAD = [0.0, 0.0, 0]

def attach_ladder(pack, q, by_anchor, all_values, rng):
    """Three wrong answers at increasing distance, all free from the tape.

    Phi trained on one right candidate against local wrong ones learns a BOUNDARY. Generation
    needs a LANDSCAPE - how much worse a world gets as the substitution moves further from the
    truth - and a boundary cannot supply one. A mind that cannot rank its own wrong answers by
    how wrong they are has no direction to move in, so this is the precondition for ROADMAP 13
    rather than an accuracy trick.

      near    a value of the SAME anchor under a different relation - right subject, wrong fact
      middle  a value of the address ADJACENT in tape order - right neighbourhood, wrong subject
      far     a value drawn uniformly from the tape - wrong everything

    Nothing is authored: every rung is a value the corpus wrote, and the distance ordering is
    structural (same anchor / adjacent slot / anywhere), not a similarity someone chose. Rungs
    that collide with the true candidate set or with each other are dropped, and a question
    without a full ladder simply trains on the task term alone.
    """
    used = set(q['cands'])
    rungs = {}
    anchor = s289a.anchor_of(q['address'])
    sibs = [it for it in by_anchor.get(anchor, ()) if it['address'] != q['address']]
    for it in rng.sample(sibs, len(sibs)) if sibs else ():
        cand = [pack['tape'].values[sl] for sl in it['slots']]
        cand = [v for v in cand if v not in used]
        if cand:
            rungs['near'] = cand[0]
            used.add(cand[0])
            break
    nxt = max(q['slots']) + 1
    for sl in (nxt, min(q['slots']) - 1):
        if 0 <= sl < pack['n_slots'] and pack['tape'].values[sl] not in used:
            rungs['middle'] = pack['tape'].values[sl]
            used.add(rungs['middle'])
            break
    for _ in range(8):
        v = all_values[rng.randrange(len(all_values))]
        if v not in used:
            rungs['far'] = v
            break
    q['ladder'] = rungs if LADDER_ON and len(rungs) == 3 else {}
    return q

def lookup_rival(q):
    """286's majority rival - over the SURVIVORS only.

    The query row now sits in vals carrying a sentinel that equals nothing. Counting it would
    let the sentinel win any all-distinct address and hand the rival a guaranteed miss, which
    would flatter the mind against an opponent crippled by our own bookkeeping.
    """
    surv = [v for i, v in enumerate(q['vals']) if i != q['query_row']]
    return Counter(surv).most_common(1)[0][0]

def own_row_rival(q):
    """The shortcut, made into an opponent instead of left as a tell.

    Answerability is not independent of the row bookkeeping. A two-mention address that says the
    same thing twice keeps a surviving own row carrying exactly the hidden value, so it is
    answerable; a one-mention address keeps nothing of its own, so it is answerable only if N(a)
    happens to hold the value. `len(own_rows)` therefore predicts the label most of the time -
    and the mind is handed that count as a node indicator.

    A mind could pass 291 on it while understanding nothing. Removing the indicator is the wrong
    fix, because "which rows belong to the address I was asked about" is real information the
    reader needs. So the shortcut is written down as a parameter-free rival instead: answer the
    majority of the OWN rows if any survive, otherwise refuse. If Phi merely ties this, the mind
    learned the bookkeeping and 291 has measured nothing - which is a thing we must be able to
    read off the report, not discover later.
    """
    own = q.get('own_rows') or set()
    ov = [q['vals'][i] for i in range(q['query_row']) if q['slots'][i] in own]
    if ov:
        return Counter(ov).most_common(1)[0][0]
    return REFUSE_LABEL if REFUSE else lookup_rival(q)

def counting_margin(q):
    """Both rivals' own confidence, so they can refuse too.

    291 would be rigged otherwise: the mind gets an "I do not know" world and the rivals are
    parameter-free rules that always name a value, so on an unanswerable question they lose by
    construction and the comparison would measure the option, not the judgment. Each rival
    therefore gets the SAME discipline the mind's refusal gets - its own label-free confidence,
    a threshold read off the probe, applied held out.

    Counting's confidence is the lead its winner holds over the runner-up, as a share of the
    rows. Retrieval's is the gap `lookup_rival_cos` already returns, which is what that rival
    actually decides on - so it is PASSED IN rather than recomputed here. Calling it a second
    time per question is the qcos defect again: a matmul and a device sync bought for nothing.
    """
    surv = [v for i, v in enumerate(q['vals']) if i != q['query_row']]
    c = Counter(surv).most_common(2)
    return (c[0][1] - (c[1][1] if len(c) > 1 else 0)) / max(1, len(surv))

def lookup_rival_cos(p, q, bank, device):
    """The RETRIEVAL rival: nearest context, zero parameters, no training.

    The majority rival knows nothing about context, so a paired win over it proves only that
    the context channel carries information counts do not have. It does NOT prove that reading
    the channel takes a mind. Those two claims have been indistinguishable in every run so far,
    and with hash ink - which is Random Indexing with a fastText-shaped word vector - the
    architecture now sits close enough to a classical retrieval system that the distinction
    stops being academic.

        rival_cos(q) = argmax over candidates c of  max over mentions s of c  cos(ctx_q, ctx_s)

    Same ctx_fp the mind reads, same questions, same run, so the comparison is paired the same
    way. The fork is honest and it is sharp:

      rival_cos lands near the mind      3489 parameters are decoration and what we built is a
                                         search engine - that has to be said, not celebrated
      Phi beats rival_cos paired         the mind does something with the tape that similarity
                                         alone does not, and only then is "mind" earned

    The evidence is the SAME rows the mind gets - the address's own mentions, query row
    excluded - not the whole tape. Handing the rival a wider view would make it a different
    question; handing it a narrower one would rig the fork. Identical evidence, one rule each,
    is the only comparison that answers anything.

    It reuses p["_ctx"], which build_graph has already filled for these slots, and scores every
    row in one matmul rather than one dot product at a time - the cost that made 289a slow.
    """
    ck = p.setdefault('_ctx', {})

    def ctx(sl):
        if sl not in ck:
            c = bank.ctx_fp(p['texts'][sl], exclude=p['tape'].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
        return ck[sl]
    qrow = q['query_row']
    qc = ctx(q['slots'][qrow])
    if qc is None:
        return (None, float('nan'))
    rows = [(i, ctx(sl)) for i, sl in enumerate(q['slots']) if i != qrow]
    rows = [(i, c) for i, c in rows if c is not None]
    if not rows:
        return (None, float('nan'))
    M = torch.stack([c for _, c in rows], 0)
    s = M @ qc
    top = int(s.argmax())
    best = q['vals'][rows[top][0]]
    other = [float(s[j]) for j, (i, _) in enumerate(rows) if q['vals'][i] != best]
    margin = float(s[top]) - (max(other) if other else -1.0)
    return (best, margin)

class Deriver(nn.Module):
    """One body, one scalar. The mind describes a world; the algebra does the arithmetic.

    The body is 286/289a's relational net verbatim: edges carry the same-value indicator and
    two context ranks, nodes carry shares and indicators, identity has nowhere to live. Phi
    pools the whole graph to one number - how well this world hangs together - and that is the
    only trained readout left. The count and compare heads are gone because their tasks moved
    into exact algebra where the invariant says they belong; the interference they caused
    (count 0.965 -> 0.903 as lookup grew) is removed by construction, not compensated.
    """
    MAX_POOL = True
    FLAT = False

    def __init__(self, device, d: int=32, n_edge: int=3, n_node: int=8, grown: int=0):
        super().__init__()
        k = 3 if self.MAX_POOL else 2
        self.edge = nn.Sequential(nn.Linear(n_edge, d), nn.GELU()).to(device)
        self.node = nn.Sequential(nn.Linear(n_node + k * d, d), nn.GELU()).to(device)
        if grown:
            with torch.no_grad():
                self.edge[0].weight[:, n_edge - grown:] = 0.0
        self.lookup = nn.Sequential(nn.Linear((2 if self.MAX_POOL else 1) * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.zeros_(self.lookup[-1].weight)
        nn.init.zeros_(self.lookup[-1].bias)

    def body(self, E, same, nf):
        e = self.edge(E)
        if self.FLAT:
            z = torch.zeros(nf.shape[0], e.shape[-1], device=nf.device, dtype=nf.dtype)
            parts = [nf, z, z] + ([z] if self.MAX_POOL else [])
            return self.node(torch.cat(parts, -1))
        own = (e * same).sum(1) / same.sum(1).clamp(min=1.0)
        parts = [nf, own, e.mean(1)]
        if self.MAX_POOL:
            parts.append(e.max(1).values)
        return self.node(torch.cat(parts, -1))

    def phi(self, E, same, nf):
        """One scalar for one world.

        EVERY aggregation here used to be a mean - over same-value neighbours, over all
        neighbours, over all rows - and a mean cannot express EXISTENCE. A high-margin question
        is decided by one row: the nearest one. "There is a row that is both close to the query
        and carries this candidate's value" is an existential claim, and under a mean that
        decisive row contributes 1/n and is averaged away by n-1 irrelevant ones.

        That is the shape of the measured failure. qmargin gave Phi the confidence signal it was
        missing and the high-margin half did not move by a single item - 0 against 13, z -3.606,
        bit for bit the same as without it. So the information was not the constraint; the
        architecture could not act on it.

        The project already knows this, in the one place it reasoned carefully about
        aggregation: the exact algebra is `new_i = 1 - max_{j<i} s_ij`, a max, chosen precisely
        because "is there any earlier row with this value" is existential.

        So the max joins the mean rather than replacing it. The mean is what carries the
        low-margin win (+2.40) and there is no reason to spend it; the max is what an
        existential question needs. Both are permutation-invariant, so equivariance and the
        invariant are untouched, and --no-max-pool restores the previous body exactly.
        """
        h = self.body(E, same, nf)
        pooled = torch.cat([h.mean(0), h.max(0).values], -1) if self.MAX_POOL else h.mean(0)
        return self.lookup(pooled).squeeze(-1)

def sparse_questions_for(p, r):
    """290's question set: the addresses the dense verb throws away.

    count and compare still come from the dense items - they are exact algebra, they cost
    nothing, and dropping them would remove the sanity bolt that fires if the tape and the
    arithmetic disagree.
    """
    out = []
    for it in p['items']:
        if not 1 <= len(it['slots']) <= 2:
            continue
        for hid in range(len(it['slots'])):
            if (q := lookup_sparse_question(p, it, r, hid, NEIGHBOURS)) is not None:
                out.append(q)
    for it in p['items']:
        if len(it['slots']) >= 2 and (q := count_question(p, it)) is not None:
            out.append(q)
    shuffled = [it for it in p['items'] if len(it['slots']) >= 2]
    r.shuffle(shuffled)
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        if (q := compare_question(p, a, b)) is not None:
            out.append(q)
    return out

def open_questions_for(p, r):
    """292's set. count and compare stay: they are exact algebra and they are the sanity bolt."""
    if MIXED:
        src = anchor_items(p) if ADDRESS_FROM == 'anchor' else [it for it in p['items'] if len(it['slots']) >= 2]
        allv = list(p['tape'].values)
        out = []
        for it in src:
            for hid in range(len(it['slots'])):
                if (q := lookup_mixed_question(p, it, r, hid, allv)) is not None:
                    out.append(q)
        for it in p['items']:
            if len(it['slots']) >= 2 and (q := count_question(p, it)) is not None:
                out.append(q)
        return out
    if ADDRESS_FROM == 'anchor' or OPEN_CANDS == 'uniform':
        src = anchor_items(p) if ADDRESS_FROM == 'anchor' else [it for it in p['items'] if len(it['slots']) >= 2]
        allv = list(p['tape'].values)
        out = []
        for it in src:
            for hid in range(len(it['slots'])):
                if (q := lookup_open_uniform(p, it, r, hid, allv)) is not None:
                    out.append(q)
        for it in p['items']:
            if len(it['slots']) >= 2 and (q := count_question(p, it)) is not None:
                out.append(q)
        shuffled = [it for it in p['items'] if len(it['slots']) >= 2]
        r.shuffle(shuffled)
        for a, b in zip(shuffled[::2], shuffled[1::2]):
            if (q := compare_question(p, a, b)) is not None:
                out.append(q)
        return out
    items = [it for it in p['items'] if len(it['slots']) >= 2]
    by_anchor = defaultdict(list)
    for it in items:
        by_anchor[s289a.anchor_of(it['address'])].append(it)
    all_values = list(p['tape'].values)
    out = []
    for it in items:
        for hid in range(len(it['slots'])):
            if (q := lookup_open_question(p, it, r, hid, by_anchor, all_values)) is not None:
                out.append(q)
        if (q := count_question(p, it)) is not None:
            out.append(q)
    shuffled = list(items)
    r.shuffle(shuffled)
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        if (q := compare_question(p, a, b)) is not None:
            out.append(q)
    return out

def questions_for(p, r):
    """Every question the tape can supply, of all three verbs.

    REACH COMES FIRST AND HAD TO BE PUT HERE. 299 dropped its dispatch inside
    open_questions_for, which is only ever called when OPEN is set - and the flag block below
    REJECTS --reach together with --open ("run it alone"). So in this file the walk's questions
    were unreachable from the moment they were written: `--reach` fell through to the parsed
    tape's count/lookup/compare and reach_questions_for was dead code. Every 299 run happened on
    a working copy that had this line; I never ran the stage, so nothing here could tell me, and
    when I sent whole files to save a confusing patch I would have overwritten the one line that
    made the runs possible. Kostya caught it.
    """
    if CONSTRAIN:
        return cons_questions_for(p, r)
    if PAIR:
        return pair_questions_for(p, r)
    if REACH:
        return reach_questions_for(p, r)
    if IDENTITY:
        return identity_questions_for(p, r)
    if OPEN:
        return open_questions_for(p, r)
    if NEIGHBOURS:
        return sparse_questions_for(p, r)
    items = [it for it in p['items'] if len(it['slots']) >= 2]
    by_anchor = defaultdict(list)
    for it in items:
        by_anchor[s289a.anchor_of(it['address'])].append(it)
    all_values = list(p['tape'].values)
    out = []
    for it in items:
        if (q := count_question(p, it)) is not None:
            out.append(q)
        for hid in range(len(it['slots'])):
            if (q := lookup_question(p, it, r, hid=hid)) is not None:
                out.append(attach_ladder(p, q, by_anchor, all_values, r))
    shuffled = list(items)
    r.shuffle(shuffled)
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        q = compare_question(p, a, b)
        if q is not None:
            out.append(q)
    return out

def n_choices(q) -> int:
    return len(q['cands']) if q['verb'] == 'lookup' else len(COUNT_LABELS) if q['verb'] == 'count' else len(COMPARE_LABELS)

def truth_of(q):
    return q['cands'][q['label']] if q['verb'] == 'lookup' else q['label']

def by_value(p):
    """value -> its slots, once per pack. Also the test for "is this token on the tape at all",
    which the copy lane needs: a token with no slot has no rows and cannot be scored."""
    bv = p.get('_by_value')
    if bv is None:
        bv = defaultdict(list)
        for sl, v in enumerate(p['tape'].values):
            bv[v].append(sl)
        p['_by_value'] = bv
    return bv

def outside_mentions(p, q, value):
    """Mentions of a value that are NOT already in this question's evidence."""
    here = set(q['slots'])
    return [sl for sl in by_value(p).get(value, ()) if sl not in here]

def shared_import_budget(p, q, values):
    """One budget for every world compared in a question, and the reason is a leak.

    A local candidate's mentions are already IN the evidence, so it usually has nothing left to
    import; a ladder rung comes from elsewhere and always has K. Give each world what it
    happens to have and Phi can read "imported rows present" as "this one is wrong" - the
    landscape gate would then pass on a bookkeeping tell rather than on distance. The budget is
    therefore the minimum available across everything being scored, so every completed world
    carries the same number of rows.
    """
    return min([IMPORT_K] + [len(outside_mentions(p, q, v)) for v in values if v != REFUSE_LABEL])

def row_meta(p):
    """slot -> (anchor id, relation id), for 290's two edge channels. Integers, not strings, so
    the channels are one broadcast comparison rather than n^2 python string compares."""
    m = p.get('_rowmeta')
    if m is None:
        an_id, rl_id, m = ({}, {}, {})
        for it in p['items']:
            an, rl = addr_parts(it['address'])
            a = an_id.setdefault(an, len(an_id))
            r = rl_id.setdefault(rl, len(rl_id)) if rl else -1
            for sl in it['slots']:
                m[sl] = (a, r)
        p['_rowmeta'] = m
    return m

def graph_base(p, q, bank, device):
    """Everything about a question's graph that does NOT depend on which candidate is scored.

    THE COST FIX 290 CANNOT RUN WITHOUT, and §22 named it in advance. `build_graph` was called
    once per candidate and rebuilt every channel each time. On the dense verb that was 4 rows
    and 6 candidates; a neighbourhood takes n from ~4 to ~30 and the candidate set from ~6 to
    ~20, so the O(n^2) work would grow ~56x and be repeated ~20 times - a run measured in days,
    which is exactly the "logic fixed, meter sagged" failure this project keeps catching late.

    Only `same` and the count share depend on the candidate. Everything else - the contexts, the
    cosine ranks, the rare-word overlaps, the anchor and relation channels, the query-row rank
    and its margin - is a property of the QUESTION. Computed once here, reused by every world.

    Cached on the question dict itself rather than by id(), because ids are recycled and a stale
    base would silently score the wrong rows. Every function that derives a new question from an
    old one (drop_rows, view_of, region_views_of) drops the cache along with the ladder.
    """
    if '_base' in q:
        return q['_base']
    slots = list(q['slots'])
    n = len(slots)
    ck, ws = (p.setdefault('_ctx', {}), p.setdefault('_words', {}))
    frames = bool(p.get('frame_mode') and p.get('frame_fps') is not None)
    for sl in set(slots):
        if sl not in ws:
            ws[sl] = set(context_words(p['texts'][sl], exclude=p['tape'].values[sl]))
        if not frames and sl not in ck:
            c = bank.ctx_fp(p['texts'][sl], exclude=p['tape'].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
    med = p.get('_median')
    if med is None:
        lens = sorted((len(v) for v in p['postings'].values()))
        med = lens[len(lens) // 2] if lens else 1
        p['_median'] = med
    allw = [ws[s] for s in slots]
    allc = [p['frame_fps'][s] if s < len(p['frame_fps']) else None for s in slots] if frames else [ck[s] for s in slots]
    cos, shared = (torch.zeros(n, n), torch.zeros(n, n))
    if FAST_COS and n > 1 and any((c is not None for c in allc)):
        d0 = next((c for c in allc if c is not None))
        C = torch.stack([c if c is not None else torch.zeros_like(d0) for c in allc])
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        G = (C @ C.T).float().cpu()
        torch.backends.cuda.matmul.allow_tf32 = tf32
        miss = torch.tensor([c is None for c in allc])
        G[miss, :] = 0.0
        G[:, miss] = 0.0
        G.fill_diagonal_(0.0)
        cos = G
    for i in range(n):
        for j in range(i + 1, n):
            if not FAST_COS and allc[i] is not None and (allc[j] is not None):
                cos[i, j] = cos[j, i] = float(allc[i] @ allc[j])
            inter = allw[i] & allw[j]
            rare = sum((1 for w in inter if len(p['postings'].get(w, ())) < med))
            shared[i, j] = shared[j, i] = rare / max(1, min(len(allw[i]), len(allw[j])))
    iu = torch.triu_indices(n, n, offset=1)
    if iu.numel():
        RARE_NNZ[0] += int((shared[iu[0], iu[1]] > 0).sum())
        RARE_NNZ[1] += int(iu.shape[1])
        cv = cos[iu[0], iu[1]]
        COS_SPREAD[0] += float(cv.sum())
        COS_SPREAD[1] += float((cv * cv).sum())
        COS_SPREAD[2] += int(cv.numel())

    def rank_norm(M):
        if iu.numel() == 0:
            return M
        v = M[iu[0], iu[1]]
        order = v.argsort()
        r = torch.empty_like(order, dtype=torch.float32)
        r[order] = torch.arange(len(v), dtype=torch.float32)
        uniq, inv = v.unique(return_inverse=True)
        if len(uniq) > 1:
            mean_r = torch.zeros(len(uniq)).index_reduce_(0, inv, r, 'mean', include_self=False)
            r = mean_r[inv] / (len(v) - 1 if len(v) > 1 else 1)
        else:
            r = torch.zeros_like(r)
        R = torch.zeros_like(M)
        R[iu[0], iu[1]] = r
        R[iu[1], iu[0]] = r
        return R
    chans = [rank_norm(cos) if 'cos' in EDGES_ON else torch.zeros_like(cos), rank_norm(shared) if 'rare' in EDGES_ON else torch.zeros_like(shared)]
    if NEIGHBOURS:
        meta = row_meta(p)
        a = torch.tensor([meta.get(s, (-1, -1))[0] for s in slots])
        r = torch.tensor([meta.get(s, (-1, -2))[1] for s in slots])
        aeq = ((a[:, None] == a[None, :]) & (a[:, None] >= 0)).float()
        req = ((r[:, None] == r[None, :]) & (r[:, None] >= 0)).float()
        aeq.fill_diagonal_(0.0)
        req.fill_diagonal_(0.0)
        chans += [aeq if 'anchor' in EDGES_ON else torch.zeros_like(aeq), req if 'rel' in EDGES_ON else torch.zeros_like(req)]
        iu0 = torch.triu_indices(n, n, offset=1)
        if iu0.numel():
            NB_NNZ[0] += int((aeq[iu0[0], iu0[1]] > 0).sum())
            NB_NNZ[1] += int((req[iu0[0], iu0[1]] > 0).sum())
            NB_NNZ[2] += int(iu0.shape[1])
    qrow = q.get('query_row', -1)
    qcos, qmargin = (torch.zeros(n), 0.0)
    if qrow >= 0 and qrow < n and (allc[qrow] is not None):
        ok = [i for i in range(n) if allc[i] is not None and i != qrow]
        if len(ok) > 1:
            idx = torch.tensor(ok)
            raw = cos[idx, qrow]
            o = raw.argsort()
            rr = torch.empty(len(ok))
            rr[o] = torch.arange(len(ok), dtype=torch.float32)
            qcos[idx] = rr / (len(ok) - 1)
            top = int(raw.argmax())
            bestv = q['vals'][ok[top]]
            other = [float(raw[kk]) for kk, i in enumerate(ok) if q['vals'][i] != bestv]
            spread = float(raw.max() - raw.min())
            if spread > 1e-09:
                qmargin = (float(raw.max()) - max(other)) / spread if other else 1.0
            else:
                qmargin = 1.0 if not other else 0.0
    nfirst = q.get('n_first', n)
    subj = [q['S'].lower() if i < nfirst else q.get('S2', q['S']).lower() for i in range(n)]
    own = q.get('own_rows')
    base = {'n': n, 'slots': slots, 'chans': chans, 'qcos': qcos, 'qmargin': qmargin, 'subj': subj, 'nfirst': nfirst, 'qrow': qrow, 'qrows': sorted(set(q.get('query_rows') or ([qrow] if qrow >= 0 else []))), 'isown': [float(s in own) for s in slots] if own is not None else None}
    GRAPH_N[0] += n
    GRAPH_N[1] = max(GRAPH_N[1], n)
    GRAPH_N[2] += 1
    q['_base'] = base
    return base

def graph_from_base(p, q, bank, device, query_value):
    """One completed world, from the cached base. Only `same` and the count share change."""
    b = graph_base(p, q, bank, device)
    n, slots, qrow = (b['n'], b['slots'], b['qrow'])
    vals = list(q['vals'])
    if query_value is not None:
        vals[qrow] = query_value
    ids, seen = ([], {})
    for v in vals:
        ids.append(seen.setdefault(v if isinstance(v, str) else id(v), len(seen)))
    t = torch.tensor(ids)
    same = (t[:, None] == t[None, :]).float()
    same.fill_diagonal_(0.0)
    E = torch.stack([same if 'same' in EDGES_ON else torch.zeros_like(same)] + b['chans'], -1).to(device)
    cnt = Counter(vals)
    nfill = p.get('frame_nfill') if p.get('frame_mode') else None
    nfill_max = p.get('frame_nfill_max', 1) if nfill is not None else 1
    qrows = set(b['qrows'])
    nf = [[cnt[vals[i]] / n if i not in qrows or query_value is not None else 0.0, float(b['subj'][i] in p['texts_lc'][slots[i]]), float(i >= b['nfirst']), 1.0 / n, float(i in qrows), 0.0, float(b['qcos'][i]), b['qmargin']] + ([math.log1p(nfill[slots[i]]) / math.log1p(nfill_max) if slots[i] < len(nfill) else 0.0] if nfill is not None else []) + ([float(q.get('home_cos', 0.0)) if i == qrow else 0.0] if REACH_HOME_COS else []) + ([float(q.get('confirm', 0.0)) if i == qrow else 0.0] if REACH_CONFIRM else []) + channel_feat(q, i, qrow) for i in range(n)]
    if b['isown'] is not None:
        for i in range(n):
            nf[i].append(b['isown'][i])
    nf = torch.tensor(nf, dtype=torch.float32, device=device)
    return (E, same.unsqueeze(-1).to(device), nf)

def build_graph(p, q, bank, device, query_value=None, import_k=None):
    """286/289a's graph verbatim, plus the side indicator COMPARE needs and, for a completed
    world, the candidate's own mentions imported from elsewhere on the tape."""
    k_eff = IMPORT_K if import_k is None else import_k
    if k_eff == 0 and (not GRAPH_CACHE) and (query_value not in (None, REFUSE_LABEL)):
        pass
    elif k_eff == 0 and query_value != REFUSE_LABEL:
        return graph_from_base(p, q, bank, device, query_value)
    if k_eff == 0 and query_value == REFUSE_LABEL:
        return graph_from_base(p, q, bank, device, None)
    slots, vals = (q['slots'], q['vals'])
    n_evidence = len(slots)
    if query_value is not None:
        slots, vals = (list(slots), list(vals))
        vals[q['query_row']] = query_value
        k = IMPORT_K if import_k is None else import_k
        for sl in outside_mentions(p, q, query_value)[:k]:
            slots.append(sl)
            vals.append(p['tape'].values[sl])
    n = len(slots)
    ck, ws = (p.setdefault('_ctx', {}), p.setdefault('_words', {}))
    frames = bool(p.get('frame_mode') and p.get('frame_fps') is not None)
    for sl in set(slots):
        if sl not in ws:
            ws[sl] = set(context_words(p['texts'][sl], exclude=p['tape'].values[sl]))
        if not frames and sl not in ck:
            c = bank.ctx_fp(p['texts'][sl], exclude=p['tape'].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
    med = p.get('_median')
    if med is None:
        lens = sorted((len(v) for v in p['postings'].values()))
        med = lens[len(lens) // 2] if lens else 1
        p['_median'] = med
    allw = [ws[s] for s in slots]
    allc = [p['frame_fps'][s] if s < len(p['frame_fps']) else None for s in slots] if frames else [ck[s] for s in slots]
    same = torch.zeros(n, n)
    cos = torch.zeros(n, n)
    shared = torch.zeros(n, n)
    if FAST_COS and n > 1 and any((c is not None for c in allc)):
        d0 = next((c for c in allc if c is not None))
        C = torch.stack([c if c is not None else torch.zeros_like(d0) for c in allc])
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        G = (C @ C.T).float().cpu()
        torch.backends.cuda.matmul.allow_tf32 = tf32
        miss = torch.tensor([c is None for c in allc])
        G[miss, :] = 0.0
        G[:, miss] = 0.0
        G.fill_diagonal_(0.0)
        cos = G
    for i in range(n):
        for j in range(i + 1, n):
            same[i, j] = same[j, i] = float(vals[i] == vals[j])
            if not FAST_COS and allc[i] is not None and (allc[j] is not None):
                cos[i, j] = cos[j, i] = float(allc[i] @ allc[j])
            inter = allw[i] & allw[j]
            rare = sum((1 for w in inter if len(p['postings'].get(w, ())) < med))
            shared[i, j] = shared[j, i] = rare / max(1, min(len(allw[i]), len(allw[j])))
    iu = torch.triu_indices(n, n, offset=1)
    if iu.numel():
        RARE_NNZ[0] += int((shared[iu[0], iu[1]] > 0).sum())
        RARE_NNZ[1] += int(iu.shape[1])
        cv = cos[iu[0], iu[1]]
        COS_SPREAD[0] += float(cv.sum())
        COS_SPREAD[1] += float((cv * cv).sum())
        COS_SPREAD[2] += int(cv.numel())

    def rank_norm(M):
        if iu.numel() == 0:
            return M
        v = M[iu[0], iu[1]]
        order = v.argsort()
        r = torch.empty_like(order, dtype=torch.float32)
        r[order] = torch.arange(len(v), dtype=torch.float32)
        uniq, inv = v.unique(return_inverse=True)
        if len(uniq) > 1:
            mean_r = torch.zeros(len(uniq)).index_reduce_(0, inv, r, 'mean', include_self=False)
            r = mean_r[inv] / (len(v) - 1 if len(v) > 1 else 1)
        else:
            r = torch.zeros_like(r)
        R = torch.zeros_like(M)
        R[iu[0], iu[1]] = r
        R[iu[1], iu[0]] = r
        return R
    E = torch.stack([same if 'same' in EDGES_ON else torch.zeros_like(same), rank_norm(cos) if 'cos' in EDGES_ON else torch.zeros_like(cos), rank_norm(shared) if 'rare' in EDGES_ON else torch.zeros_like(shared)], -1).to(device)
    cnt = Counter(vals)
    nfirst = q.get('n_first', n)
    qrow = q.get('query_row', -1)
    subj = [q['S'].lower() if i < nfirst or i >= n_evidence else q.get('S2', q['S']).lower() for i in range(n)]
    qcos = torch.zeros(n)
    qmargin = 0.0
    if qrow >= 0 and allc[qrow] is not None:
        ok = [i for i in range(n) if allc[i] is not None and i != qrow]
        if len(ok) > 1:
            idx = torch.tensor(ok)
            raw = cos[idx, qrow]
            o = raw.argsort()
            r = torch.empty(len(ok))
            r[o] = torch.arange(len(ok), dtype=torch.float32)
            qcos[idx] = r / (len(ok) - 1)
            top = int(raw.argmax())
            bestv = vals[ok[top]]
            other = [float(raw[k]) for k, i in enumerate(ok) if vals[i] != bestv]
            spread = float(raw.max() - raw.min())
            if spread > 1e-09:
                qmargin = (float(raw.max()) - max(other)) / spread if other else 1.0
            else:
                qmargin = 1.0 if not other else 0.0
    nf = torch.tensor([[cnt[vals[i]] / n if i != qrow or query_value is not None else 0.0, float(subj[i] in p['texts_lc'][slots[i]]), float(i >= nfirst), 1.0 / n, float(i == qrow), float(i >= n_evidence), float(qcos[i]), qmargin] + ([math.log1p(p['frame_nfill'][slots[i]]) / math.log1p(p.get('frame_nfill_max', 1)) if slots[i] < len(p['frame_nfill']) else 0.0] if p.get('frame_mode') and p.get('frame_nfill') is not None else []) + ([float(q.get('home_cos', 0.0)) if i == qrow else 0.0] if REACH_HOME_COS else []) + ([float(q.get('confirm', 0.0)) if i == qrow else 0.0] if REACH_CONFIRM else []) + channel_feat(q, i, qrow) for i in range(n)], dtype=torch.float32, device=device)
    return (E, same.unsqueeze(-1).to(device), nf)

def ladder_scores_for(net, p, q, device, bank):
    """Phi on the three wrong worlds, in ladder order. Empty when the tape could not supply one."""
    if not q.get('ladder'):
        return None
    k = shared_import_budget(p, q, list(q['cands']) + [q['ladder'][r] for r in LADDER])
    outs = []
    for rung in LADDER:
        E, same, nf = build_graph(p, q, bank, device, query_value=q['ladder'][rung], import_k=k)
        outs.append(net.phi(E, same, nf))
    return torch.stack(outs)

def cand_logits_for(net, p, q, device, bank):
    """Score one completed world per candidate and let them compete.

    This is 288's repair loop turned inward: instead of preferring a group, the mind writes the
    conjecture into the query row, reads the world that results, and says how well it hangs
    together. The query-row indicator stays set, so a completed world is never mistaken for an
    observed one - the conjecture is marked as a conjecture, which is the derived-slot
    discipline applied to reading.
    """
    if q.get('ident'):
        return torch.stack([net.phi(*build_graph(p, identity_world(p, q, s), bank, device, query_value=None, import_k=0)) for s in q['cand_slots']])
    vals = list(q['cands']) + [q['ladder'][r] for r in LADDER] if q.get('ladder') else list(q['cands'])
    k = shared_import_budget(p, q, vals)
    outs = []
    for c in q['cands']:
        E, same, nf = build_graph(p, q, bank, device, query_value=c, import_k=k)
        outs.append(net.phi(E, same, nf))
    return torch.stack(outs)

def loss_for(net, p, q, device, bank):
    """Only the judged verb trains, and it trains an ORDERING, not a boundary.

    One loss, no weight to choose between a task term and a ladder term - the Plackett-Luce
    likelihood of the whole ordering true > near > middle > far factorises into nested
    softmaxes, and its FIRST factor is exactly the cross-entropy this stage already had, with
    the ladder added to the competitor set:

        L = -log sm(true | all) - log sm(near | near,mid,far) - log sm(mid | mid,far)

    Local wrong candidates sit in `all` but are left unranked among themselves, because the
    tape says they are wrong and says nothing about which is wronger. When the tape cannot
    supply a full ladder this reduces, term for term, to the plain cross-entropy - so the
    change is a strict generalisation and an unladdered run is still the old objective.
    """
    if q.get('cons'):
        return cons_loss(net, p, q, device, bank)
    if q.get('pair'):
        return pair_loss(net, p, q, device, bank)
    if q.get('reach'):
        return reach_loss(net, p, q, device, bank)
    if q['verb'] != 'lookup':
        raise ValueError(f"{q['verb']} is exact algebra now and has no loss")
    if ROUTE and q.get('mixed'):
        return route_loss(net, p, q, device, bank)
    lg = cand_logits_for(net, p, q, device, bank)
    if OBJECTIVE == 'reward':
        pr = torch.softmax(lg, 0)
        R = torch.full_like(pr, -1.0)
        R[q['label']] = 1.0
        if (REFUSE or q.get('mixed')) and q.get('answerable') and (REFUSE_LABEL in q['cands']):
            R[q['cands'].index(REFUSE_LABEL)] = 0.75
        return -(pr * R).sum()
    lad = ladder_scores_for(net, p, q, device, bank)
    if lad is None:
        return F.cross_entropy(lg.unsqueeze(0), torch.tensor([q['label']], device=device))
    allsc = torch.cat([lg, lad])
    loss = -(allsc[q['label']] - torch.logsumexp(allsc, 0))
    for k in range(len(LADDER) - 1):
        loss = loss - (lad[k] - torch.logsumexp(lad[k:], 0))
    return loss

@torch.no_grad()
def predict_with_confidence(net, p, q, device, bank):
    """What would be said, how sure, and what the tape says - the three things an audit needs.

    The exact verbs answer through the algebra with confidence 1.0, which is not flattery: it
    is the honest statement that a computed answer is certain GIVEN the same-value relation.
    When s_ij becomes a trained judgment, its uncertainty enters here and 1.0 stops being the
    right number - that is the seam where the future work plugs in.
    """
    if q.get('reach'):
        l1, l2, own, cands, _l3, _lc = reach_logits(net, p, q, device, bank)
        if int(l1.argmax()) == len(own) + 1:
            names, lg = (cands + [REFUSE_LABEL], l2)
        else:
            names, lg = (own + [REFUSE_LABEL], l1[:-1])
        pr = torch.softmax(lg, -1)
        k = int(pr.argmax())
        return (float(pr[k]), names[k], q['truth_value'])
    if q['verb'] != 'lookup':
        return (1.0, exact_answer(q), truth_of(q))
    lg = cand_logits_for(net, p, q, device, bank)
    pr = torch.softmax(lg, -1)
    k = int(pr.argmax())
    return (float(pr[k]), q['cands'][k], truth_of(q))

def main() -> int:
    global SEED, LOG_PATH, LADDER_ON, EDGES_ON, IMPORT_K, INK, FP, WORDS, FAST_COS, VIEWS, ROW_DROPOUT, VIEW_MODE, NEIGHBOURS, REFUSE, GRAPH_CACHE, OPEN, OBJECTIVE, IDENTITY, IDENT_VALUES, IDENT_TAU, IDENT_OVERLAP, IDENT_CANDS, IDENT_CORE, IDENT_IMPORT, ADDRESS_FROM, OPEN_CANDS, ANCHOR_MAX_ROWS, PATTERNS, PAT_W, MIXED, TAPE, ROUTE, STEP_COST, FRAME_MAX, REACH, REACH_K, REACH_CANDS, REACH_MAX_Q, REACH_MAX_ROWS, REACH_NO_REFUSE, REACH_LOOKAHEAD, FRAME_FP, REACH_IMPORT, REACH_HOME_COS, TAPE_SAMPLE, HOME_COS_STAGE, REACH_LINE, REACH_CONFIRM, CONF_WINDOW, PAIR, PAIR_CANDS, PAIR_MAX_ROWS, PAIR_MAX_Q, PAIR_PER_LINE, PAIR_FOLLOW, PAIR_BLIND, REACH_GAMMA, EQUAL_TAILS, STAGE2_ALWAYS, BISECT, FINETUNE, REACH_DEPTH, REACH_COMPASS, SHUFFLE_TAPE, COHERENCE, DEEP_ROOT, TWO_WAY, RETAIN, RETAIN_BY, RETAIN_CTX, OTHER_NET, SPEAK_BATCH, SPEAK_WEIGHT, CALIB_BATCH, CALIB_WEIGHT, CONSTRAIN, CONS_LENSES, CONS_RESOLVE, TWO_WAY_BY, MIN_FILLERS, CONNECT, CONNECT_MAX, OWN_IMPORT, OWN_IN_OFFER, COPY, COPY_D, COPY_BACKFILL, REACH_CHANNEL, MOVES_ON, MOVES, MOVE_TEACH, ROUTE_ON
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--train-steps', type=int, default=0)
    ap.add_argument('--tape-period', type=int, default=50)
    ap.add_argument('--cpu', action='store_true', help='run on the CPU even when a card is present. Under frames+reach the graphs are a handful of rows and Phi is 5.6k parameters, so the GPU spends its time on launch latency and its memory on the allocator')
    ap.add_argument('--addresses', type=int, default=0)
    ap.add_argument('--min-mentions', type=int, default=2)
    ap.add_argument('--own-in-offer', action='store_true', help="367: rank the home values in the SAME softmax as the walk's, instead of choosing between the two branches. Use with --stage2-always 1.0, which removes the stay/go decision entirely")
    ap.add_argument('--own-import', action='store_true', help="366: build stage one's own worlds with the SAME imported rows every stage-two candidate gets. Without it 'stay' is compared against systematically larger worlds")
    ap.add_argument('--connect', action='store_true', help='365: offer values from RELATED places too, weighted by how many fillers each shares with this place. Interleaved into the existing candidate cap, so the offer does not grow')
    ap.add_argument('--connect-max', type=int, default=4000, help='neighbourhood places scanned, best-overlap first. Bounded so one common filler cannot cost a minute')
    ap.add_argument('--copy', action='store_true', help="376: offer values STANDING IN THE NEIGHBOURING LINES of the question, ranked by how often they stand there, nearest line first on a tie. The question's own line is dropped whole - the hidden value is on it. Interleaved into the existing cap, so the offer does not grow")
    ap.add_argument('--move-set', default=','.join(MOVE_ALL), help='386: WHICH MOVES ARE ON THE BALLOT, comma separated, out of step,share,lines. 385 ran all three and failed its gate 2 seeds of 4 - and the split says why: the two seeds where the mind stayed with `step` BEAT the interleave (+0.099, +0.310) and the two where it went to `lines` collapsed. `lines` is copy, the channel already retired from the standing arm on independent evidence (377r hit .475 against connect-only .599); it was re-enabled in 385 only to give that move a lane. Removing it restores a decision taken earlier, on other data')
    ap.add_argument('--moves', action='store_true', help='385: the mind emits a MOVE - step, share, lines - and the tape executes that one at the unchanged cap, instead of the four channels being merged by a fixed rule with Phi choosing a name. The choice is made on one probe row per move, BEFORE any candidate world is scored')
    ap.add_argument('--route-on', choices=('all', 'walk_only'), default=ROUTE_ON, help="34.4: which questions teach the STAY/GO decision. `all` is every earlier run bit for bit. `walk_only` gives the router a gradient only where staying is arithmetically wrong - the truth not among the values already here, and among the ones the walk reached - by detaching the route's probability elsewhere. Both PICKS keep their gradient on every question: this cuts one decision, not the population. Requires --two-way; read _read394_walkonly.py's void checks off existing dumps BEFORE running it")
    ap.add_argument('--move-teach', type=float, default=MOVE_TEACH, help="391: PAY THE MOVE. The move ballot's logits have never received a gradient - reach_logits kept the chosen name and discarded them - so 385 and 386 measured an argmax of a scorer trained to rank final names, on a decision nobody had taught. This adds one term: the ballot's softmax against whether each lane REACHES THE TRUTH, which is the tape's property and is counted while the lane is already enumerated. Weight, declared, never swept; 0 keeps every earlier run bit for bit. Requires --moves")
    ap.add_argument('--reach-channel', action='store_true', help='379: give the mind THREE INDICATORS saying which channel offered each candidate - connect, home, copy, with the walk as the all-zero baseline. The offer, the head and the budget are unchanged; only the provenance is new. 377 vs 378 showed the merge rule is a constant where a decision belongs')
    ap.add_argument('--copy-backfill', action='store_true', help='378: the copy lane takes ONLY the slots the walk and connect left empty, instead of round-robining for a fixed share of them. 377 lost hit on one seed of four while reach rose on all four, and cand_places fell on all four - the lane was evicting walked candidates that were right')
    ap.add_argument('--copy-d', type=int, default=4, help="lines either side of the question's. 376 read best at 4 on w400 and 16 on w1600, so this travels with the window")
    ap.add_argument('--min-fillers', type=int, default=2, help='how many DIFFERENT values a hole must have taken to be a place. 1 admits constant frames, which is where facts live (359)')
    ap.add_argument('--address-tau', type=float, default=0.9)
    ap.add_argument('--tau-mode', choices=('absolute', 'density'), default='absolute', help="absolute keeps 279's fixed cosine and reproduces every earlier run bit for bit. density derives tau so the WRITE ink produces a tape of --tau-target-density mentions per address - required whenever the write ink changes, because a different ink at a fixed cosine shatters the tape and the threshold becomes what the arm measures")
    ap.add_argument('--tau-target-density', type=float, default=ARC_TRAIN_DENSITY, help='mentions per address to calibrate to. Default is the MEASURED arc/mean train tape (2388 slots / 804 addresses) that every scoreboard number was taken on')
    ap.add_argument('--tau-calib-iters', type=int, default=12, help='bisection steps for --tau-mode density. 12 over the full [0, 1] bracket resolves tau to ~2e-4, which holds the density error under 0.005 even where the merge curve is steep; each extra step is pure arithmetic because CachedBank has already inked the corpus')
    ap.add_argument('--address-overlap', type=int, default=2)
    ap.add_argument('--addr-key', choices=('two', 'set', 'mean'), default='two')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--holdout', choices=('corpus', 'address'), default='corpus')
    ap.add_argument('--no-scan-cache', action='store_true', help='disable the exact corpus-scan memo (use to verify it changes nothing)')
    ap.add_argument('--no-fast-grouping', action='store_true', help='disable the batched single-link grouping (use to verify it changes nothing)')
    ap.add_argument('--wiki-bytes', type=int, default=0)
    ap.add_argument('--train-lines', type=int, default=0)
    ap.add_argument('--eval-lines', type=int, default=0)
    ap.add_argument('--line-max', type=int, default=400, help='keep wiki lines with 80 <= len <= this. 0 drops the upper cap (~4x lines on the local wikitext file). Default 400 keeps every earlier scoreboard bit-identical')
    ap.add_argument('--import-k', type=int, default=IMPORT_K, help='mentions of a candidate imported when completing its world; 0 reproduces the broken ladder where every absent value looked alike')
    ap.add_argument('--edge-channels', type=str, default=','.join(EDGES), help='comma list from same,cos,rare - zero the rest. Ablation to find which channel carries the paired win over counting')
    ap.add_argument('--ink', choices=('mean', 'bigram'), default=INK, help="phrase axis: mean reproduces today's order-blind ctx_fp exactly; bigram binds adjacent words with a fixed non-commutative permutation so the ink can tell `X defeated Y` from `Y defeated X`")
    ap.add_argument('--fp', choices=('arc', 'hash'), default=FP, help='word axis: arc is the frozen stage191 encoder; hash is character n-grams into a blake2b digest - nothing trained, no character vocabulary, no OOV, every script')
    ap.add_argument('--words', choices=('ascii', 'unicode'), default=WORDS, help="what counts as a word. unicode only pays off with --fp hash: arc's stoi has no Cyrillic, so a wider intake would just be discarded")
    ap.add_argument('--fp-ngram', type=int, default=3, help='character n-gram length for --fp hash')
    ap.add_argument('--write-fp', choices=('arc', 'hash'), default='arc', help="ink used to GROUP mentions into addresses. Pinned by default so an ink A/B varies reading only; 279's tau is an absolute cosine and a different ink shatters the tape against it")
    ap.add_argument('--probe-period', type=int, default=250, help='how often to score the fixed probe tape. The training curve is measured on a different tape every resample and cannot tell converged from overfitting; this one can')
    ap.add_argument('--views', type=int, default=1, help="reconciliation (ROADMAP 20): the mind reads V independently thinned views of each question with the SAME weights, logits are pooled by a mean, and the views' disagreement is a label-free confidence signal. 1 reproduces every earlier run bit for bit; V>1 needs --row-dropout as the thinning rate")
    ap.add_argument('--neighbours', type=int, default=0, help='290 (ROADMAP §19): build N(a) from up to this many addresses per route - shared anchor, shared relation, shared rare words - put all their rows in ONE graph, and switch to the sparse verb. 0 reproduces every earlier run bit for bit, including the 5601 parameter count')
    ap.add_argument('--seed', type=int, default=SEED, help='every draw in the run - tapes, questions, probe, views. Added when 292 came back with held z +2.59 against corpus retrieval and train z -0.15 on the same weights: two samples of one quantity disagreeing by 2.7 sigma. A second seed is the only cheap way to tell structure from a lucky split, and there was no way to ask for one')
    ap.add_argument('--objective', choices=('ce', 'reward'), default='ce', help="ce is cross-entropy, every run to date. reward optimises 280's fixed payoff directly: L = -sum_c p(c)R(c), closed form, no new constant. Removes the mismatch between what is trained and what is scored; it does not remove a collapse caused by an unanswerable task")
    ap.add_argument('--open', action='store_true', help="292: the hidden value occurs exactly once at the address, so it is FOREIGN to the evidence and no rule over the address's own rows can reach it. Candidates are the truth and the three ladder rungs, all four importing the same number of rows - the symmetric comparison the ladder could never get in 289. Needs --import-k >= 1, because with 0 imports all four worlds are the same graph")
    ap.add_argument('--no-graph-cache', action='store_true', help='rebuild every channel per candidate, as before graph_base existed. Dense arms only - use it to verify the cache changed nothing')
    ap.add_argument('--refuse', action='store_true', help='291: keep the sparse questions whose answer is on NO row of N(a) and let the mind score the world where the query row stays unknown. Refusal becomes an action with a label the tape supplies, not a threshold on a confidence score. Needs --neighbours')
    ap.add_argument('--view-mode', choices=('thin', 'region'), default='thin', help="how views are cut. thin = recon3's random subsampling (views share ~65%% of rows, D measured model noise, pooled lost to single). region = contiguous stretches of the tape in write order - disjoint by construction, deterministic, so D measures whether the CORPUS agrees with itself at this address rather than whether one sampler agrees with another")
    ap.add_argument('--row-dropout', type=float, default=0.0, help='probability of dropping each evidence row during TRAINING, so the mind sees the same fact at several densities. 0 reproduces every earlier run bit for bit - it draws from its own generator')
    ap.add_argument('--dim', type=int, default=32, help='width of the mind. Exposed so the max-pool result can be checked at MATCHED parameter count: max-pool added 2048 weights along with the max, and one of those two is the cause')
    ap.add_argument('--no-max-pool', action='store_true', help='pool with the mean alone, as every run before this one did. A mean cannot express existence, and a high-margin question is decided by one row')
    ap.add_argument('--no-fast-cos', action='store_true', help='build the pairwise cosine matrix with the original per-pair loop (use to verify the batched version changes nothing)')
    ap.add_argument('--probe-frac', type=int, default=10, help='one anchor in this many is reserved for the probe and excluded from both training and held-out scoring, so the stopping step is never chosen using an anchor the evaluation will ask about')
    ap.add_argument('--probe-size', type=int, default=200, help='how many probe questions to score. Same questions every time - a probe set that changes is the defect this replaces')
    ap.add_argument('--no-early-stop', action='store_true', help='keep the last step instead of the best probe step - reproduces every run before the probe tape existed')
    ap.add_argument('--write-ink', choices=('mean', 'bigram'), default='mean')
    ap.add_argument('--write-words', choices=('ascii', 'unicode'), default='ascii')
    ap.add_argument('--no-ladder', action='store_true', help='ablation: train Phi on the task term alone, the control the ladder is measured against')
    ap.add_argument('--address-from', choices=('fp', 'anchor'), default=ADDRESS_FROM, help="294: what an address IS. fp keeps 279's grouping (cosine tau AND word overlap, plus a relation half the 293 audit found to be a function word); anchor makes it one exact string the corpus wrote, so nothing is approximated where the evidence is decided.")
    ap.add_argument('--open-cands', choices=('ladder', 'uniform'), default=OPEN_CANDS, help='294: ladder builds the three wrong answers BY relatedness, which is what an inverted mean_phi reads back; uniform draws any value the address does not carry and measures the distance afterwards.')
    ap.add_argument('--anchor-max-rows', type=int, default=ANCHOR_MAX_ROWS, help='cost budget for anchor addresses: the graph is O(n^2) and `canada` carries dozens of mentions. Nearest in tape order, same in all worlds.')
    ap.add_argument('--identity', action='store_true', help="293: the verb becomes 'do these mentions name the same place'. The label is the corpus's own pre-grouping string, so the mind is put where fp_addresses' threshold currently stands rather than downstream of it.")
    ap.add_argument('--identity-audit', action='store_true', help="build 293's questions, score the three rivals, print, and stop. No model and no gradient - the minute that says whether the hour is worth spending, which is what 19 and 291 both needed and did not get.")
    ap.add_argument('--ident-values', choices=('hide', 'show'), default=IDENT_VALUES, help='hide (default) gives every row its own sentinel, so the same-value edge cannot decide identity and CONFIRM becomes a result instead of an input; show measures the size of that shortcut.')
    ap.add_argument('--ident-import', type=int, default=IDENT_IMPORT, help="how many of the candidate's own other mentions arrive with it, gathered by its pre-grouping string. 0 compares a place with a row - the evidence 1-NN already has; k > 0 compares a place with a place, which is the thing 1-NN cannot do.")
    ap.add_argument('--ident-cands', type=int, default=IDENT_CANDS)
    ap.add_argument('--ident-core', type=int, default=IDENT_CORE)
    ap.add_argument('--tape', choices=('parser', 'frames'), default=TAPE, help='frames: the write path becomes counting. An address is a hole whose surroundings recur, the width is whatever the corpus supports, and there is no tau, no stopword list and no grammar. 297 measured it: ~10x the addresses, 5.3 mentions each, 22% of rows confirming.')
    ap.add_argument('--frame-max', type=int, default=FRAME_MAX)
    ap.add_argument('--route', action='store_true', help='the mind may READ MORE before answering. `expand` is one more world scored by the same Phi, so the two-step decision is one softmax and stays differentiable - no policy head, no sampling.')
    ap.add_argument('--gamma', type=float, default=REACH_GAMMA, help='the multiplicative price of movement: every terminal pays gamma^reads * R. Below 1.0 it replaces --step-cost entirely')
    ap.add_argument('--deep-root', choices=('mind', 'first'), default=DEEP_ROOT, help="where the second read starts: the place of the mind's own best shallow candidate, or the nearest place. `first` makes reachability a property of the tape again, which is what the depth numbers need")
    ap.add_argument('--two-way', action='store_true', help='stage one becomes STAY vs GO: equal-width maxima to decide, each branch valued by its own expectation. Removes both the dilution and the cardinality asymmetry without adding an objective')
    ap.add_argument('--shuffle-tape', action='store_true', help='THE NULL: permute which filler stood in which hole, keeping every count and every size. The route must collapse to the floor')
    ap.add_argument('--coherence', type=int, default=COHERENCE, help='score N real tape fragments against corrupted ones - Phi asked whether a world hangs together, with no hole and no teacher')
    ap.add_argument('--constrain', action='store_true', help="345 / ladder step 1: the mind chooses WHICH OF ITS OWN ROWS to look through and the tape answers by counting what stands with that value over every place that holds it. Phi's output becomes which QUERY, not which answer, so the answer set is never enumerated by us")
    ap.add_argument('--two-way-by', choices=('max', 'margin'), default=TWO_WAY_BY, help='with --two-way: summarise each branch by its best world (max, every run to date) or by the GAP between its best two (margin) - the quantity that reads AUC 0.969 on the depth arm and decides nothing')
    ap.add_argument('--cons-resolve', choices=('count', 'share', 'place'), default=CONS_RESOLVE, help='how the tape answers through a lens: argmax of the raw co-occurrence count, or of that count divided by how much of the value stands anywhere. 317 measured raw counts at 0.029 against 0.222 because the truths that matter are rare by construction; this is the same fix. 384: `place` answers from the ONE place the lens most stands at instead of summing every place that holds it - a resolution that is a SELECTION, which is the one form of this interface never tried')
    ap.add_argument('--cons-lenses', type=int, default=CONS_LENSES, help='how many of its own rows the mind may choose between - its whole output space')
    ap.add_argument('--calib-batch', type=int, default=CALIB_BATCH, help="389: put the RAW score of B questions into one softmax against which of them the tape can answer, so B-1 of the B free per-question offsets are removed and Phi's value becomes comparable BETWEEN questions. The gauge, not another option: a refusal world lives inside the same per-question softmax and moves with the offset. Costs B questions per step, so divide --train-steps by B for a matched question budget")
    ap.add_argument('--calib-weight', type=float, default=CALIB_WEIGHT, help='weight of the calibration term. Declared, never swept - 321 and 341 each priced a second objective at ~4x the route')
    ap.add_argument('--speak-batch', type=int, default=SPEAK_BATCH, help='341: price the speaking across B questions at once instead of on each one, so `always refuse` stops being expressible and what is learned is which questions to spend speech on. Costs B questions per step, so divide --train-steps by B for a matched question budget')
    ap.add_argument('--speak-weight', type=float, default=SPEAK_WEIGHT, help='weight of the comparative speaking term. Declared, not swept')
    ap.add_argument('--rival-mind', default=None, help='336: a second saved mind that answers the SAME questions in this run, paired question by question. Train natively and pass the transplant here: if Phi is corpus-free the two are indistinguishable')
    ap.add_argument('--retain', type=int, default=RETAIN, help='338: keep only N places for the walk to visit. Questions are still drawn from the whole tape, so the rules are compared on the same questions at the same budget. 0 keeps everything')
    ap.add_argument('--retain-by', choices=('random', 'own', 'share', 'mind'), default=RETAIN_BY, help="how those N are chosen: at random (what the tape does today), by the most mentions, by the most dominant filler, or by the mind's own margin. `mind` requires a frozen --load-mind - see retain_keep")
    ap.add_argument('--reach-compass', choices=('cos', 'share', 'both', 'share1', 'rare', 'common', 'cover', 'jaccard'), default=REACH_COMPASS, help="what the walk follows. cos: the filler-bag fingerprint. share: the exact count of shared mentions. both: interleaved - 323 says the two disagree about direction 75%% of the time at equal yield. 372 adds the rest of 371's family as compasses, each a count with no fitted constant: share1 (distinct values shared), rare (weighted by 1/the value's corpus mentions), common (the opposite, its own control), cover (how much OF THE NEIGHBOUR the sharing covers), jaccard (the two filler sets against their union)")
    ap.add_argument('--reach-depth', type=int, default=REACH_DEPTH, help='how many reads the route may chain. 2 gives the walked place its own walk, paid by the same reward at gamma^2 - one objective, deeper')
    ap.add_argument('--bisect', action='store_true', help='321: measure bisection as a channel. Halves are unfilled worlds of equal evidence, trained by their own exact teacher, and the exam records where the log2(c) descent lands against the flat argmax')
    ap.add_argument('--finetune', action='store_true', help='with --load-mind, keep training on the new corpus instead of freezing')
    ap.add_argument('--stage2-always', type=float, default=STAGE2_ALWAYS, help='teach the pick on EVERY question, off-policy and at this weight, while the route is priced exactly as before. 1.0 = the pick is learned as hard as the route; 0.0 = every run before 314')
    ap.add_argument('--equal-tails', action='store_true', help='the direction choice compares maxima over EQUAL candidate counts, min(|walk|, |line|) - the 304 cardinality fix')
    ap.add_argument('--step-cost', type=float, default=STEP_COST, help='declared price of reading more, like the 0.75 hedge. Not fitted.')
    ap.add_argument('--reach', action='store_true', help='299: no candidate list. The mind walks to the nearest places by frame fingerprint and may say only a filler it reached, or nothing. The floor collapses from 0.25 to ~1/|values|, unanswerable questions arrive on their own, and the rival is the same walk without a mind.')
    ap.add_argument('--pair', action='store_true', help="309: TWO holes on one line, further apart than a frame can reach, filled ONE INTO THE OTHER - the second is scored in the world the first left behind, by the same Phi and with no new parameter. The mind chooses which hole to answer first. Rivals are counting's two ways, the product of marginals and the joint pair where the tape wrote one; COMP_ONLY is where both are blind.")
    ap.add_argument('--pair-cands', type=int, default=PAIR_CANDS, help="values offered per hole, split evenly between the hole's own rows and what the walk reaches")
    ap.add_argument('--pair-max-rows', type=int, default=PAIR_MAX_ROWS)
    ap.add_argument('--pair-max-q', type=int, default=PAIR_MAX_Q)
    ap.add_argument('--pair-per-line', type=int, default=PAIR_PER_LINE)
    ap.add_argument('--pair-blind', action='store_true', help='the ablation: stage two does NOT see the first fill. Same worlds and same Phi, so a tie means the verb composes nothing')
    ap.add_argument('--pair-follow', type=int, default=PAIR_FOLLOW, help='first fills whose second stage is scored. The cost bound that keeps the verb linear in the offer instead of quadratic')
    ap.add_argument('--reach-k', type=int, default=REACH_K)
    ap.add_argument('--reach-cands', type=int, default=REACH_CANDS)
    ap.add_argument('--home-cos-stage', choices=('both', 'stage2'), default=HOME_COS_STAGE, help='where the home summary is visible. At stage one it lifts the LOCAL worlds and so suppresses the step, which is how 299i lost the routing win; at stage two every option is a walked candidate and it only separates them')
    ap.add_argument('--wiki', default=None, help='the corpus to build the tape from. The point of --load-mind is to point this somewhere else: a mind that holds no facts should read a tape from text it was never fitted to')
    ap.add_argument('--save-mind', default=None, help="write Phi's weights and the shape they were trained under")
    ap.add_argument('--load-mind', default=None, help="load Phi and DO NOT TRAIN. The exam then measures a transplanted mind on this corpus's tape - the literal form of the claim that the knowledge is outside the weights. A shape mismatch refuses to run")
    ap.add_argument('--flat', action='store_true', help='withhold the edges from the node layer: same parameters, same node features, same pooling and readout, no message passing. The control for whether Phi is a mind or a good choice of features')
    ap.add_argument('--reach-confirm', action='store_true', help="give the answered row one number: how many RARE words of the question's line also stand around this value elsewhere on the tape. 305 measured 1.67x for the truth over a wrong candidate")
    ap.add_argument('--conf-window', type=int, default=CONF_WINDOW, help='also read this many lines either side of a home; a wider read makes the shared rare words rarer still')
    ap.add_argument('--reach-line', action='store_true', help='give the mind a SECOND kind of step: to the other frames of the same sentence. A different relation from resemblance, exactly counted, and the first time direction is a choice rather than a fixed order')
    ap.add_argument('--reach-home-cos', action='store_true', help='give the answered row one number: the cosine between where this value usually stands on the tape and this place. A summary, not a sample - no row is chosen and no world grows, which is what 299g got wrong')
    ap.add_argument('--reach-import', choices=('walk', 'homes', 'relation'), default=REACH_IMPORT, help="what a stage-two candidate brings with it: the rows at the walked places (`walk`, which makes candidates from one place identical in the cosine channel), its own mentions elsewhere on the tape (`homes`, the lookup verb's import), or its mentions AT PLACES RELATED TO THIS ONE, best overlap first (`relation`, 372b - the walk still steers by the fingerprint and only the EVIDENCE changes)")
    ap.add_argument('--tape-sample', choices=('uniform', 'region'), default=TAPE_SAMPLE, help='uniform draws 3000 addresses from the whole corpus, which dilutes every relation between places by the sampling ratio; region takes every frame of a contiguous stretch of lines instead, and the 300 audit measured what that changes')
    ap.add_argument('--frame-fp', choices=('address', 'fillers'), default=FRAME_FP, help="what a place's fingerprint is. `address` hashes the characters of `left|right`, which at width 1 is six characters and collides - the run measured cos mean 0.918. `fillers` hashes the bag of what stood in the hole, counted with repetition")
    ap.add_argument('--reach-lookahead', action='store_true', help="score the step as max(stage-two logits) instead of as Phi of a separate pile-of-rows world. 299b had step_rate 0: the step's logit and the step's payoff were unrelated quantities, so nothing could teach the one from the other")
    ap.add_argument('--reach-no-refuse', action='store_true', help='take silence away in both stages of the walk. 299_hash was void as a payoff measurement - the truth is reachable 10% of the time, so always-silent is the optimal play and the mind found it. This arm asks the other question on its own: can the walk find anything')
    ap.add_argument('--reach-max-rows', type=int, default=REACH_MAX_ROWS, help="rows of the question's own place, the hidden one included. The frame tape has no ANCHOR_MAX_ROWS and a frame like `the|of` holds hundreds of mentions; graph building is quadratic in rows, so this is what keeps a fat frame from being a question nobody can finish")
    ap.add_argument('--reach-max-q', type=int, default=REACH_MAX_Q, help="questions scored per pack. ~12 graphs each, so this is the run's length dial; 0 scores every one of them.")
    ap.add_argument('--mixed', action='store_true', help='296: one exam, one payoff. Half the questions have the answer on the list, half do not; the mind must find it or say there is none. It refuses by argmax with no threshold; whole-tape retrieval is given a threshold fitted on train, in its favour.')
    ap.add_argument('--frames', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--patterns', action='store_true', help="295: mine value-pair regularities exactly, let Phi judge each rule's WORLD (witness rows, no counters), label by held-out lift > 1, race the rule's own train statistics. The object under judgment is a rule, not a row, so 1-NN cannot play.")
    ap.add_argument('--pat-witnesses', type=int, default=PAT_W)
    ap.add_argument('--run-tag', type=str, default='')
    ap.add_argument('--out', type=str, default='', help='optional extra path for the decision JSON; the report is always also written under results/stage289_decision<tag>.json')
    args = ap.parse_args()
    SEED = args.seed
    LADDER_ON = not args.no_ladder
    IMPORT_K = args.import_k
    INK, FP, WORDS = (args.ink, args.fp, args.words)
    FAST_COS = not args.no_fast_cos
    Deriver.MAX_POOL = not args.no_max_pool
    Deriver.FLAT = args.flat
    VIEWS, ROW_DROPOUT, VIEW_MODE = (args.views, args.row_dropout, args.view_mode)
    NEIGHBOURS, REFUSE, OPEN = (args.neighbours, args.refuse, args.open)
    OBJECTIVE = args.objective
    ADDRESS_FROM, OPEN_CANDS = (args.address_from, args.open_cands)
    ANCHOR_MAX_ROWS = args.anchor_max_rows
    if ADDRESS_FROM == 'anchor' and (not args.open):
        log('  --address-from anchor is 294, which is the open verb: add --open')
        return 1
    if ADDRESS_FROM == 'anchor' and OPEN_CANDS != 'uniform':
        log('  --address-from anchor needs --open-cands uniform: the ladder is built through fp addresses and cannot be attached to one that has none')
        return 1
    PATTERNS, PAT_W = (args.patterns, args.pat_witnesses)
    MIXED = args.mixed
    REACH, REACH_K, REACH_CANDS = (args.reach, args.reach_k, args.reach_cands)
    REACH_MAX_Q, REACH_MAX_ROWS = (args.reach_max_q, args.reach_max_rows)
    REACH_NO_REFUSE, REACH_LOOKAHEAD = (args.reach_no_refuse, args.reach_lookahead)
    FRAME_FP, REACH_IMPORT = (args.frame_fp, args.reach_import)
    TAPE_SAMPLE = args.tape_sample
    MIN_FILLERS = args.min_fillers
    CONNECT, CONNECT_MAX = (args.connect, args.connect_max)
    COPY, COPY_D = (args.copy, args.copy_d)
    COPY_BACKFILL = args.copy_backfill
    REACH_CHANNEL = args.reach_channel
    MOVES_ON = args.moves
    MOVES = tuple((m.strip() for m in args.move_set.split(',') if m.strip()))
    if not MOVES or any((m not in MOVE_ALL for m in MOVES)):
        raise SystemExit(f"--move-set: unknown move in {args.move_set!r}; choose from {','.join(MOVE_ALL)}")
    OWN_IMPORT = args.own_import
    OWN_IN_OFFER = args.own_in_offer
    REACH_HOME_COS, HOME_COS_STAGE = (args.reach_home_cos, args.home_cos_stage)
    REACH_LINE = args.reach_line
    REACH_CONFIRM, CONF_WINDOW = (args.reach_confirm, args.conf_window)
    PAIR, PAIR_CANDS = (args.pair, args.pair_cands)
    PAIR_MAX_ROWS, PAIR_MAX_Q, PAIR_PER_LINE = (args.pair_max_rows, args.pair_max_q, args.pair_per_line)
    PAIR_FOLLOW, PAIR_BLIND = (args.pair_follow, args.pair_blind)
    if PAIR:
        REACH = True
    if PAIR and (REACH_CONFIRM or REACH_HOME_COS):
        log('  --pair does not carry the confirm or home-cos features yet; run it without them')
        return 1
    if REACH and args.tape != 'frames':
        log('  --reach walks by frame fingerprint: it needs --tape frames')
        return 1
    if REACH and (MIXED or OPEN or NEIGHBOURS or IDENTITY):
        log('  --reach replaces the offered-candidate verbs; run it alone')
        return 1
    TAPE, FRAME_MAX, ROUTE, STEP_COST = (args.tape, args.frame_max, args.route, args.step_cost)
    REACH_GAMMA, EQUAL_TAILS = (args.gamma, args.equal_tails)
    STAGE2_ALWAYS = args.stage2_always
    BISECT, FINETUNE = (args.bisect, args.finetune)
    REACH_DEPTH = args.reach_depth
    REACH_COMPASS = args.reach_compass
    SHUFFLE_TAPE, COHERENCE = (args.shuffle_tape, args.coherence)
    DEEP_ROOT, TWO_WAY = (args.deep_root, args.two_way)
    RETAIN, RETAIN_BY = (args.retain, args.retain_by)
    MOVE_TEACH = args.move_teach
    ROUTE_ON = args.route_on
    SPEAK_BATCH, SPEAK_WEIGHT = (args.speak_batch, args.speak_weight)
    CALIB_BATCH, CALIB_WEIGHT = (args.calib_batch, args.calib_weight)
    CONSTRAIN, CONS_LENSES = (args.constrain, args.cons_lenses)
    CONS_RESOLVE = args.cons_resolve
    TWO_WAY_BY = args.two_way_by
    if CONSTRAIN and (args.pair or not args.reach):
        log('  --constrain needs --reach (and not --pair): it answers the reach question by a different operation, and the two arms must share their holes')
        return 1
    if SPEAK_BATCH and (SPEAK_BATCH < 2 or args.pair or (not args.reach)):
        log('  --speak-batch needs at least 2 questions and the reach verb (not pair): a softmax over one margin is a constant, and there is nothing to compare across')
        return 1
    if CALIB_BATCH and (CALIB_BATCH < 2 or args.pair or (not args.reach)):
        log('  --calib-batch needs at least 2 questions and the reach verb (not pair): with one question there is no second scale to tie it to, and the offset stays free')
        return 1
    if ROUTE_ON != 'all' and (not args.two_way):
        log("  --route-on needs --two-way: without it stage one's logits are the own worlds themselves, and cutting the route would cut the home pick with it")
        return 1
    if MOVE_TEACH and (not MOVES_ON):
        log('  --move-teach needs --moves: without a ballot there are no move logits to teach')
        return 1
    if args.rival_mind and args.reach_depth > 1:
        log("  --rival-mind refused at --reach-depth > 1: the deeper walk is rooted at the mind's own pick, so the two minds would not be answering the same question")
        return 1
    if RETAIN_BY == 'mind' and RETAIN and (not args.load_mind or args.finetune):
        log('  --retain-by mind requires a frozen --load-mind (and not --finetune): a mind that both chooses the tape and is trained on it is no longer separate from it')
        return 1
    if TWO_WAY and REACH_LINE:
        log('  --two-way is a binary stay/go decision and has no form for --reach-line yet')
        return 1
    if REACH_GAMMA < 1.0:
        STEP_COST = 0.0
    if args.frames:
        TAPE = 'frames'
    if ROUTE and (not MIXED):
        log("  --route is the mixed exam's step: add --mixed")
        return 1
    if TAPE == 'frames' and args.min_mentions > 1:
        log('  --tape frames writes exact addresses; --min-mentions 1 (a frame already needs two distinct fillers to exist)')
        return 1
    if MIXED and (not args.open):
        log('  --mixed is the open verb with refusal folded in: add --open')
        return 1
    if MIXED and args.import_k < 1:
        log('  --mixed needs --import-k >= 1, or every candidate world is the same graph')
        return 1
    if MIXED and args.objective != 'reward':
        log('  --mixed needs --objective reward: cross-entropy cannot price silence, and the whole point is that one payoff weighs finding against saying there is none')
        return 1
    IDENTITY = args.identity or args.identity_audit
    IDENT_VALUES, IDENT_CANDS, IDENT_CORE = (args.ident_values, args.ident_cands, args.ident_core)
    IDENT_OVERLAP, IDENT_IMPORT = (args.address_overlap, args.ident_import)
    if IDENTITY and (OPEN or NEIGHBOURS):
        log('  --identity is a different verb from --open and --neighbours; running two at once measures their sum and credits whichever was named last')
        return 1
    if IDENTITY and args.views > 1:
        log("  --identity with --views > 1: a region cut is defined over a lookup's evidence rows and a 293 question has no query row until the world is built")
        return 1
    if IDENTITY and args.ident_cands < 2:
        log('  --ident-cands < 2 leaves nothing to choose between')
        return 1
    if OPEN and NEIGHBOURS:
        log('  --open and --neighbours are two different verbs; run them apart or the arm measures their sum and credits whichever was named last')
        return 1
    if OPEN and args.import_k < 1:
        log('  --open needs --import-k >= 1: with nothing imported, the true value and all three rungs give the identical graph and the question has no content')
        return 1
    if OPEN and args.no_ladder:
        log('  --open IS the ladder - the rungs are its candidates - so --no-ladder would leave it with one candidate')
        return 1
    GRAPH_CACHE = not args.no_graph_cache
    if args.no_graph_cache and NEIGHBOURS:
        log("  --no-graph-cache is a dense-arm verification path; it does not build 290's two extra edge channels and would silently score a 3-channel graph")
        return 1
    if REFUSE and (not NEIGHBOURS):
        log('  --refuse needs --neighbours: the unanswerable questions are the sparse ones')
        return 1
    if NEIGHBOURS:
        if args.import_k:
            log('  --neighbours needs --import-k 0: an imported world and the refusal world would carry different row counts, which is a bookkeeping tell, not evidence')
            return 1
    if VIEWS > 1 and VIEW_MODE == 'thin' and (ROW_DROPOUT <= 0):
        log('  --views > 1 with --row-dropout 0: every view is the same graph and the pool is decoration; set a thinning rate')
        return 1
    if TAPE != 'frames' and (args.write_fp, args.write_ink) != ('arc', 'mean') and (args.tau_mode == 'absolute') and (args.address_tau == 0.9):
        log(f"  --write-fp {args.write_fp} --write-ink {args.write_ink} rewrites the tape, and 279's tau is an absolute cosine: at 0.90 a different ink merges almost nothing. Add --tau-mode density (target defaults to the measured arc tape, {ARC_TRAIN_DENSITY} mentions/address).")
        return 1
    for side, w, f in (('read', WORDS, FP), ('write', args.write_words, args.write_fp)):
        if w == 'unicode' and f == 'arc':
            log(f"  --{('' if side == 'read' else 'write-')}words unicode with an arc encoder widens the intake into a vocabulary that cannot represent it; use hash there or ascii")
            return 1
    EDGES_ON = {c.strip() for c in args.edge_channels.split(',') if c.strip()}
    if not EDGES_ON <= set(EDGES):
        log(f'  unknown edge channel in {sorted(EDGES_ON)}; allowed {EDGES}')
        return 1
    if not EDGES_ON:
        log('  every edge channel disabled: nothing to read')
        return 1
    if NEIGHBOURS:
        EDGES_ON |= set(EDGES_NB)
    tag = args.run_tag and f'_{args.run_tag}' or ''
    tag += '_addrholdout' if args.holdout == 'address' else ''
    LOG_PATH = RES / f'_stage289_log{tag}.txt'
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_steps = args.train_steps or (600 if args.smoke else 6000)
    n_addr = args.addresses or (300 if args.smoke else 400)
    log(f'Stage289 derivation start {datetime.now(timezone.utc).isoformat()} device={device} holdout={args.holdout}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    can = None
    if FP != 'hash':
        can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
        can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
        can.eval()
        for p in can.parameters():
            p.requires_grad_(False)
    rule = WORD_RULES[WORDS]

    def make_bank(fp_kind, ink_kind, r):
        base = HashFp(device=device, n=args.fp_ngram, rule=r) if fp_kind == 'hash' else FpBank(can, stoi, device)
        return CachedBank(BigramBank(base, rule=r) if ink_kind == 'bigram' else base)
    bank = make_bank(FP, INK, rule)
    write_bank = bank if (args.write_fp, args.write_ink, args.write_words) == (FP, INK, WORDS) else make_bank(args.write_fp, args.write_ink, WORD_RULES[args.write_words])
    ink_bank = bank._b
    base_bank = ink_bank._b if INK == 'bigram' else ink_bank
    arc0 = s271.arc_enc_hash(can) if can is not None else 'not_loaded'
    _nouns: dict = {}
    _raw_common = s279.common_nouns

    def _cached_common(lines, min_lower: int=3):
        k = (id(lines), len(lines), min_lower)
        if k not in _nouns:
            _nouns[k] = _raw_common(lines, min_lower)
        return _nouns[k]
    s279.common_nouns = _cached_common
    if not args.no_scan_cache:
        install_assertion_cache(s279)
    if not args.no_fast_grouping:
        install_fast_fp_addresses(s279)
    wiki_path = Path(args.wiki) if args.wiki else WIKI
    if not wiki_path.exists():
        log(f'  no corpus at {wiki_path}')
        return 1
    log(f'  corpus {wiki_path}')
    with wiki_path.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(args.wiki_bytes or (4000000 if args.smoke else 30000000))
    lo, hi = (80, args.line_max or None)
    all_lines = [l.strip() for l in wtext.split('\n') if len(l.strip()) >= lo and (hi is None or len(l.strip()) <= hi)]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][:args.train_lines or (3000 if args.smoke else 25000)]
    eval_lines = all_lines[cut:][:args.eval_lines or (1500 if args.smoke else 12000)]
    log(f'  lines: train {len(train_lines)}, eval {len(eval_lines)} (the probe reserves ANCHORS, not lines - see `reserved`)')
    if args.holdout == 'address':
        eval_lines = train_lines
    g_ink = bool(verify_word_rule())
    log(f'  word rule matches stage194: {g_ink}  (rule={WORDS}, fp={FP}, ink={INK})')
    if INK == 'bigram':
        v = bool(ink_bank.verify_mean_path(train_lines[:200]))
        g_ink &= v
        log(f'  bigram tokenisation matches base mean-ink: {v}')
    if FP == 'hash':
        v, notes = verify_hash_ink(base_bank)
        g_ink &= v
        log(f'  hash ink deterministic and digest-faithful: {v}  {notes}')
    if not g_ink:
        log('  ABORT: the ink does not do what it says it does')
        return 1

    def side(address: str) -> int:
        return int(hashlib.sha1(s289a.anchor_of(address).encode('utf-8')).hexdigest(), 16) & 1

    def reserved(address: str) -> bool:
        """Is this anchor set aside for the probe?

        Carving LINES off for the probe was the wrong instrument and the run said so: 3581 lines
        is 11% of the corpus but produced 17 lookup questions against the training tape's 403,
        because a lookup question needs an address carrying two distinct values and that density
        falls faster than linearly. Early stopping on 17 questions selects noise.

        Reserving ANCHORS is the mechanism the project already has - it is what --holdout address
        does - and it lets the probe read the whole corpus while still never seeing a question
        the training tapes could have shown it. Salted separately so the reservation does not
        correlate with the address-holdout split.
        """
        h = hashlib.sha1(f'probe:{s289a.anchor_of(address)}'.encode('utf-8')).hexdigest()
        return int(h, 16) % args.probe_frac == 0
    tau_rule = None if TAPE == 'frames' else args.address_tau if args.tau_mode == 'absolute' else tau_for_density(args.tau_target_density, args.tau_calib_iters, log)

    def new_pack(r, lines, want, probe=False, n_addr_over=None):
        asrt = None
        if TAPE == 'frames':
            asrt, _fa, _pool = tframes.frame_assertions(lines, FRAME_MAX, MIN_FILLERS, n_addr_over or n_addr, r, TAPE_SAMPLE)
            FRAME_POOL[0] = _pool
        p = s280.pack_from_corpus(lines, bank=write_bank, tok=tok, pad_id=pad_id, device=device, rng=r, n_addr=n_addr_over or n_addr, min_mentions=args.min_mentions, tau=tau_rule, overlap=args.address_overlap, soft_match=0.0, min_per_family=8, addr_key=args.addr_key, assertions=asrt, group=asrt is None)
        p = dict(p)
        if SHUFFLE_TAPE:
            sr = random.Random(SEED + 31337)
            vv = list(p['tape'].values)
            sr.shuffle(vv)
            p['tape'].values = vv
        if TAPE == 'frames':
            sa = p['straddr']
            fill = defaultdict(set)
            for sl, ad in enumerate(sa):
                fill[ad].add(p['tape'].values[sl])

            def _fps(strings):
                out = {}
                for b0 in range(0, len(strings), 512):
                    ch = strings[b0:b0 + 512]
                    M = bank.fp(ch).float()
                    out.update({t: M[i] for i, t in enumerate(ch)})
                return out
            keys = list(fill)
            if FRAME_FP == 'fillers':
                vfp = _fps(sorted({p['tape'].values[sl] for sl in range(len(sa))}))
                acc, cnt = ({}, Counter())
                for sl, ad in enumerate(sa):
                    v = vfp[p['tape'].values[sl]]
                    acc[ad] = v.clone() if ad not in acc else acc[ad] + v
                    cnt[ad] += 1
                p['frame_sum'], p['frame_cnt'], p['val_fp'] = (acc, cnt, vfp)
                cache = {ad: F.normalize(acc[ad], dim=-1) for ad in acc}
            else:
                cache = {ad: v for ad, v in zip(keys, [F.normalize(x, dim=-1) for x in (_fps(keys)[k] for k in keys)])}
            p['frame_fps'] = [cache[ad] for ad in sa]
            p['frame_nfill'] = [len(fill[ad]) for ad in sa]
            p['frame_nfill_max'] = max(p['frame_nfill']) if p['frame_nfill'] else 1
            p['frame_mode'] = True
        if args.holdout == 'address':
            p['items'] = [it for it in p['items'] if side(it['address']) == want]
        p['items'] = [it for it in p['items'] if reserved(it['address']) == probe]
        return p
    questions = questions_for

    def by_verb(qq):
        d = defaultdict(list)
        for q in qq:
            d[q['verb']].append(q)
        return d
    n_node_now = 8 + (1 if TAPE == 'frames' else 0) + (1 if NEIGHBOURS else 0) + (1 if REACH_HOME_COS else 0) + (1 if REACH_CONFIRM else 0) + (3 if REACH_CHANNEL else 0)
    net = Deriver(device, d=args.dim, n_edge=3 + (len(EDGES_NB) if NEIGHBOURS else 0), n_node=n_node_now, grown=len(EDGES_NB) if NEIGHBOURS else 0)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = int(sum((x.numel() for x in net.parameters())))

    def cand_logits(p, q):
        return cand_logits_for(net, p, q, device, bank)

    def loss_of(p, q):
        return loss_for(net, p, q, device, bank)
    pack = new_pack(rng, train_lines, 0)
    IDENT_TAU = None if TAPE == 'frames' else tau_rule.memo['tau'] if callable(tau_rule) else tau_rule
    if args.identity_audit:
        au = identity_audit(pack, random.Random(SEED + 293))
        au['tau'] = IDENT_TAU
        au['overlap'] = IDENT_OVERLAP
        au['n_addresses'] = pack['n_addresses']
        au['n_slots'] = pack['n_slots']
        log(f'  IDENT_AUDIT {json.dumps(au)}')
        (RES / f"stage289_ident_audit{('_' + args.run_tag if args.run_tag else '')}.json").write_text(json.dumps(au, indent=2), encoding='utf-8')
        return 0
    qs = questions(pack, rng)
    bv = by_verb(qs)
    TRAIN_VERB = 'cons' if CONSTRAIN else 'pair' if PAIR else 'reach' if REACH else 'lookup'
    n_lad = sum((1 for q in bv.get('lookup', ()) if q.get('ladder')))
    log(f"  tape: {pack['n_addresses']} addresses, {pack['n_slots']} slots | questions {json.dumps({k: len(v) for k, v in bv.items()})} | params {n_params}")
    if bv.get(TRAIN_VERB):
        _q = dict(bv[TRAIN_VERB][0])
        _w = pair_world(pack, _q, bank, device, {0: pair_offers(pack, _q)[0]['offer'][0]}, _q['query_rows'][0])[2].shape[-1] if PAIR else reach_world(pack, _q, bank, device, _q['truth_value'], [], 0)[2].shape[-1] if REACH or CONSTRAIN else build_graph(pack, _q, bank, device, query_value=_q['cands'][0])[2].shape[-1]
        _want = net.node[0].in_features - (3 if Deriver.MAX_POOL else 2) * args.dim
        if _w != _want:
            log(f'  node vector is {_w} wide and the mind expects {_want}: a feature is on in the graph and off in the body')
            return 1
        log(f'  node vector {_w} wide, matches the body')
    log(f"  ladder coverage: {n_lad}/{len(bv.get('lookup', ()))} lookup questions have all three rungs; the rest train on the task term alone")
    if len(bv.get(TRAIN_VERB, ())) < s286.MIN_ANSWERED:
        log('  too few lookup questions: raise --train-lines')
        return 1
    held = new_pack(random.Random(SEED + 99), eval_lines, 1)
    if PATTERNS:
        return run_patterns(pack, held, bank, device, args)
    held_qs = questions(held, random.Random(SEED + 7))
    t_pp = time.time()
    probe = new_pack(random.Random(SEED + 555), train_lines, 0, probe=True, n_addr_over=n_addr * args.probe_frac)
    log(f"  probe pack: {len(probe['items'])} reserved addresses ({time.time() - t_pp:.0f}s to build)")
    probe_qs = [q for q in questions(probe, random.Random(SEED + 556)) if q['verb'] in (('pair',) if PAIR else ('reach',) if REACH else TRAIN_VERBS) and (not q.get('ladder'))][:min(args.probe_size, 60) if REACH or PAIR else args.probe_size]
    t_probe = time.time()
    probe_graphs = []
    for q in probe_qs if REACH else ():
        q['_keep_g'] = True
    pv_rng = random.Random(SEED + 6060)
    for q in () if REACH else probe_qs:
        qvs, qm = views_and_mask(q, pv_rng, device)
        views = []
        for qv in qvs:
            k = shared_import_budget(probe, qv, list(qv['cands']))
            views.append([build_graph(probe, qv, bank, device, query_value=c, import_k=k) for c in qv['cands']])
        probe_graphs.append((views, qm, torch.tensor([q['label']], device=device)))
    log(f'  probe tape: {len(probe_qs)} lookup questions, never trained on; {sum((len(g) for vs, _, _ in probe_graphs for g in vs))} graphs cached ({VIEWS} view(s)/question) in {time.time() - t_probe:.0f}s')

    @torch.no_grad()
    def probe_loss():
        if REACH or PAIR:
            if not probe_qs:
                return float('nan')
            f = pair_loss if PAIR else reach_loss
            return float(sum((float(f(net, probe, q, device, bank)) for q in probe_qs)) / len(probe_qs))
        if not probe_graphs:
            return float('nan')
        net.eval()
        tot = 0.0
        for views, qm, label in probe_graphs:
            L = torch.stack([torch.stack([net.phi(E, same, nf) for E, same, nf in graphs]) for graphs in views])
            tot += float(F.cross_entropy(pool_views(L, qm).unsqueeze(0), label))
        net.train()
        return tot / len(probe_graphs)
    mind_sig = {'dim': args.dim, 'n_node': n_node_now, 'max_pool': bool(Deriver.MAX_POOL), 'edges': sorted(EDGES_ON), 'verb': TRAIN_VERB, 'views': VIEWS, 'frames': TAPE == 'frames', 'frame_fp': FRAME_FP, 'lookahead': REACH_LOOKAHEAD, 'no_refuse': REACH_NO_REFUSE, 'import': REACH_IMPORT, 'home_cos': REACH_HOME_COS, 'import_k': IMPORT_K, 'flat': bool(Deriver.FLAT), 'reach_k': REACH_K, 'reach_cands': REACH_CANDS, 'line': REACH_LINE, 'confirm': REACH_CONFIRM, 'conf_window': CONF_WINDOW, 'pair': PAIR, 'pair_cands': PAIR_CANDS, 'pair_max_rows': PAIR_MAX_ROWS, 'pair_blind': PAIR_BLIND, 'gamma': REACH_GAMMA, 'equal_tails': EQUAL_TAILS, 'stage2_always': STAGE2_ALWAYS, 'bisect': BISECT, 'depth': REACH_DEPTH, 'compass': REACH_COMPASS, 'deep_root': DEEP_ROOT, 'two_way': TWO_WAY, 'two_way_by': TWO_WAY_BY, 'speak_batch': SPEAK_BATCH, 'speak_weight': SPEAK_WEIGHT, 'move_teach': MOVE_TEACH, 'route_on': ROUTE_ON, 'calib_batch': CALIB_BATCH, 'calib_weight': CALIB_WEIGHT, 'constrain': CONSTRAIN, 'cons_lenses': CONS_LENSES, 'cons_resolve': CONS_RESOLVE}
    RETAIN_CTX = (net, device, bank)
    if args.rival_mind:
        rblob = torch.load(args.rival_mind, map_location=device, weights_only=False)
        if rblob.get('sig') != mind_sig:
            rdiff = {k: (rblob.get('sig', {}).get(k), v) for k, v in mind_sig.items() if rblob.get('sig', {}).get(k) != v}
            log(f'  --rival-mind refused: saved under a different shape {json.dumps(rdiff)}')
            return 1
        OTHER_NET = Deriver(device, d=args.dim, n_edge=3 + (len(EDGES_NB) if NEIGHBOURS else 0), n_node=n_node_now, grown=len(EDGES_NB) if NEIGHBOURS else 0)
        OTHER_NET.load_state_dict(rblob['state'])
        OTHER_NET.eval()
        log(f"  rival mind from {args.rival_mind}: {rblob.get('note', '')} - it answers every reach question this run asks, paired. It is never trained here.")
    if args.load_mind:
        blob = torch.load(args.load_mind, map_location=device, weights_only=False)
        if blob.get('sig') != mind_sig:
            diff = {k: (blob.get('sig', {}).get(k), v) for k, v in mind_sig.items() if blob.get('sig', {}).get(k) != v}
            log(f'  --load-mind refused: the saved mind was trained under a different shape {json.dumps(diff)}')
            return 1
        net.load_state_dict(blob['state'])
        if FINETUNE:
            log(f"  transplanted mind from {args.load_mind}: {blob.get('note', '')} - FINE-TUNING on this corpus for {n_steps} steps. This arm is not evidence of separation; the frozen run is.")
        else:
            n_steps = 0
            log(f"  transplanted mind from {args.load_mind}: {blob.get('note', '')} - NO training on this corpus, the exam below is this mind reading a tape it has never been fitted to")
    t0_probe = time.time()
    probe_loss()
    dt_probe = time.time() - t0_probe
    n_evals = len([s for s in range(1, n_steps + 1) if s % args.probe_period == 0 or s == n_steps])
    log(f'  probe eval: {dt_probe:.2f}s x {n_evals} = {dt_probe * n_evals / 60:.1f} min added to this run')
    drop_rng = random.Random(SEED + 4242)
    best = {'loss': float('inf'), 'step': 0, 'state': None}
    probe_curve = []
    losses, curve = ([], [])
    for step in range(1, n_steps + 1):
        if (step - 1) % args.tape_period == 0 and step > 1:
            prev_addr = {it['address'] for it in pack['items']}
            pack = new_pack(rng, train_lines, 0)
            qs = questions(pack, rng)
            bv = by_verb(qs)
            now = {it['address'] for it in pack['items']}
            if prev_addr:
                RESAMPLE_OVERLAP.append(len(prev_addr & now) / max(1, len(prev_addr | now)))
            if not bv.get(TRAIN_VERB):
                log('  empty tape after resample')
                return 1
        q = bv[TRAIN_VERB][rng.randrange(len(bv[TRAIN_VERB]))]
        if SPEAK_BATCH or CALIB_BATCH:
            global _SPEAK_ACC, _CALIB_ACC
            bn = max(SPEAK_BATCH, CALIB_BATCH)
            qs = [q] + [bv[TRAIN_VERB][rng.randrange(len(bv[TRAIN_VERB]))] for _ in range(bn - 1)]
            if args.row_dropout > 0:
                qs = [drop_rows(x, drop_rng, 1.0 - args.row_dropout) or x for x in qs]
            _SPEAK_ACC = [] if SPEAK_BATCH else None
            _CALIB_ACC = [] if CALIB_BATCH else None
            try:
                loss = torch.stack([loss_of(pack, x) for x in qs]).mean()
                acc, cacc = (_SPEAK_ACC, _CALIB_ACC)
            finally:
                _SPEAK_ACC = _CALIB_ACC = None
            if SPEAK_BATCH:
                if len(acc) == len(qs):
                    loss = loss - SPEAK_WEIGHT * speak_term([m for m, _a in acc], [a for _m, a in acc], device)
                elif acc:
                    raise RuntimeError(f'speaking batch recorded {len(acc)} of {len(qs)} questions')
            if CALIB_BATCH:
                if len(cacc) == len(qs):
                    loss = loss - CALIB_WEIGHT * calib_term([s for s, _y in cacc], [y for _s, y in cacc], device)
                elif cacc:
                    raise RuntimeError(f'calibration batch recorded {len(cacc)} of {len(qs)} questions')
        elif VIEWS > 1:
            if VIEW_MODE == 'region' and args.row_dropout > 0:
                thin = drop_rows(q, drop_rng, 1.0 - args.row_dropout)
                if thin is not None:
                    q = thin
            pooled, _, _ = reconciled(net, pack, q, device, bank, drop_rng)
            loss = F.cross_entropy(pooled.unsqueeze(0), torch.tensor([q['label']], device=device))
        else:
            if args.row_dropout > 0:
                thin = drop_rows(q, drop_rng, 1.0 - args.row_dropout)
                if thin is not None:
                    q = thin
            loss = loss_of(pack, q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % args.probe_period == 0 or step == n_steps:
            pl = probe_loss()
            probe_curve.append({'step': step, 'probe_loss': pl})
            if pl < best['loss']:
                best = {'loss': pl, 'step': step, 'state': {k: v.detach().clone() for k, v in net.state_dict().items()}}
        if step % max(1, n_steps // 8) == 0:
            curve.append({'step': step, 'loss': float(np.mean(losses[-200:])), 'probe_loss': probe_curve[-1]['probe_loss'] if probe_curve else None})
            log(f"  step {step}/{n_steps} train={np.mean(losses[-200:]):.4f} probe={(probe_curve[-1]['probe_loss'] if probe_curve else float('nan')):.4f}")
    if not args.no_early_stop and best['state'] is not None:
        net.load_state_dict(best['state'])
        log(f"  early stop: restored step {best['step']} (probe {best['loss']:.4f}) of {n_steps}")
    net.eval()
    if args.save_mind:
        Path(args.save_mind).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'sig': mind_sig, 'state': net.state_dict(), 'note': f'trained on {wiki_path.name}, seed {SEED}, {n_steps} steps'}, args.save_mind)
        log(f'  mind saved to {args.save_mind}')
    arc1 = s271.arc_enc_hash(can) if can is not None else 'not_loaded'

    @torch.no_grad()
    def examine(p, qq):
        st = {v: {'n': 0, 'model': 0, 'rival': 0, 'rival_cos': 0, 'floor': 0.0} for v in ('count', 'compare', 'lookup')}
        conf, hits = (Counter(), [])
        exact_bad = 0
        b10 = b01 = 0
        cos_items = []
        view_rows = []
        sparse_rows = []
        open_rows = []
        ident_rows = []
        mixed_rows = []
        ub = {k: [0.0, 0] for k in ('true', 'same_anchor', 'elsewhere')}
        reach_rows = []
        pair_rows = []
        cons_rows = []
        vrng = random.Random(SEED + 7788)
        wrng = random.Random(SEED + 2991)
        rung_sum = {k: 0.0 for k in ('true',) + LADDER}
        rung_n, concord, pairs, ties = (0, 0, 0, 0)
        budgets = []
        for q in qq:
            v = q['verb']
            if v == 'pair':
                l1, g1 = pair_logits(net, p, q, device, bank)
                k0, v0, _ = g1[int(l1.argmax())]
                l2, offer2, k1 = pair_second(net, p, q, device, bank, k0, v0)
                v1 = offer2[int(l2.argmax())]
                said = [None, None]
                said[k0], said[k1] = (v0, v1)
                a, b = q['holes']
                truth = (a['truth'], b['truth'])
                offered = (truth[0] in set(a['offer']), truth[1] in set(b['offer']))
                marg, joint, seen, bag_best, bag_seen = pair_rivals(p, q)
                right = tuple(said) == truth
                pair_rows.append([int(offered[0] and offered[1]), int(right), int(said[0] == truth[0]) + int(said[1] == truth[1]), int(marg == truth), int(seen), int(joint == truth), int(truth[0] in a['own']), int(truth[1] in b['own']), int(offered[0]), int(offered[1]), len(a['offer']) * len(b['offer']), k0, len(q['slots']), int(said[0] == truth[0]), int(said[1] == truth[1]), int(bag_seen), int(bag_best == truth)])
                continue
            if v == 'cons':
                cl1, cl2, cown, clens = cons_logits(net, p, q, device, bank)
                csaid = cons_answers(p, q, clens)
                c_ans = q['truth_value'] in set((x for x in csaid if x is not None))
                cpick = int(cl1.argmax())
                cnames1 = cown + ([] if REACH_NO_REFUSE else [REFUSE_LABEL])
                if cpick == len(cnames1) and len(cl2):
                    ci = int(cl2.argmax())
                    c_said = (csaid[ci] if ci < len(csaid) else REFUSE_LABEL) or REFUSE_LABEL
                    c_step, c_lens_i = (1, ci)
                else:
                    c_said = cnames1[cpick] if cpick < len(cnames1) else REFUSE_LABEL
                    c_step, c_lens_i = (0, -1)
                riv = cons_rivals(p, q, clens)
                lens_at = {v2: i for i, v2 in enumerate(clens)}

                def _riv_right(nm):
                    v2 = riv.get(nm)
                    return int(v2 is not None and csaid[lens_at[v2]] == q['truth_value'])
                _sh, _tot = (0.0, 0)
                if c_lens_i >= 0:
                    _b, _bn, _tt, _tp = cons_resolve(p, q, clens[c_lens_i])
                    _sh, _tot = (_bn / _tt if _tt else 0.0, _tt)
                _pres = any((q['truth_value'] in set(cons_resolve(p, q, v2)[3]) for v2 in clens))
                _wc = reach_candidates(p, q)
                _wr, _ = reach_rival(p, q)
                cons_rows.append([int(c_ans), int(q['truth_value'] in set(cown)), len(clens), int(c_said == REFUSE_LABEL), int(c_said == q['truth_value']), c_step, c_lens_i, _riv_right('rare'), _riv_right('frequent'), _riv_right('decisive'), int(_pres), float(_sh), int(_tot), int(q['truth_value'] in set(_wc['cands'])), int(_wr == q['truth_value'])])
                continue
            if v == 'reach':
                l1, l2, own, cands, l3, lcands = reach_logits(net, p, q, device, bank)
                _dc_all = q.get('_deep', (None, [], []))[1] if REACH_DEPTH > 1 else []
                ansble = q['truth_value'] in set(cands) | set(_dc_all)
                _deep_only = q['truth_value'] in set(_dc_all) and q['truth_value'] not in set(cands)
                said, stepped, _dep, _pscore, _pmarg = reach_pick(q, l1, l2, own, cands, l3, lcands)
                _o_right, _o_step = (0, 0)
                if OTHER_NET is not None:
                    with torch.no_grad():
                        ol1, ol2, oown, ocands, ol3, olc = reach_logits(OTHER_NET, p, q, device, bank)
                    o_said, _o_step, _od, _os, _om = reach_pick(q, ol1, ol2, oown, ocands, ol3, olc)
                    _o_right = int(o_said == q['truth_value'])
                rv, rmg = reach_rival(p, q)
                _rc = reach_candidates(p, q)
                _ev = {c: reach_rows_for(p, q, c, _rc['rows_of'][c]) for c in cands}
                _b = min([IMPORT_K] + [len(_ev[c]) for c in cands]) if cands else 0
                _base = len(q['slots'])
                _cr_v, _cr_share = reach_count_rival(p, q)
                _cr_ties = q.pop('_cr_ties', 0)
                _bi_v, _bi_pairs = reach_bisect(net, p, q, device, bank, cands, _ev, _b)[:2] if BISECT and cands else (None, [])
                _nexp = _base + len({s for c in cands for s in _ev[c][:_b]} - set(q['slots']))
                reach_rows.append([int(ansble), int(said == REFUSE_LABEL), int(said == q['truth_value']), float(rmg), int(rv == q['truth_value']), stepped, len(cands), _base + _b, _nexp, int(reach_reachable(p, q, REACH_K * 4)), int(reach_reachable(p, q, REACH_K, wrng)), int(q['truth_value'] in set(own)), int(lookup_rival(q) == q['truth_value']), len(own), max(Counter((p['tape'].values[s2] for s2 in q['slots'][:q['query_row']])).values()), _rc['n_places'], int(REACH_LINE and q['truth_value'] in set(reach_line_candidates(p, q)['cands'])), int(REACH_LINE and reach_line_rival(p, q) == q['truth_value']), int(stepped == 2), _base, int(_cr_v == q['truth_value']), float(_cr_share), int(_bi_v == q['truth_value']), len(_bi_pairs), _dep, int(_deep_only), _o_right, _o_step, _pscore, _pmarg, _cr_ties, MOVES.index(q['_move']) if MOVES_ON and q.get('_move') in MOVES else -1])
                continue
            if v == 'lookup':
                if VIEWS > 1:
                    lg, l0, dd = reconciled(net, p, q, device, bank, vrng)
                    view_rows.append([int(int(lg.argmax()) == q['label']), int(int(l0.argmax()) == q['label']), dd])
                elif q.get('mixed') and ROUTE:
                    lg = None
                else:
                    lg = cand_logits(p, q)
                truth = q['cands'][q['label']]
                pred = None if lg is None else q['cands'][int(lg.argmax())]
                if q.get('mixed') and ROUTE:
                    l1, l2 = route_logits(net, p, q, device, bank)
                    if int(l1.argmax()) == len(q['cands']):
                        stepped, pred = (1, q['cands'][int(l2.argmax())])
                    else:
                        stepped, pred = (0, q['cands'][int(l1.argmax())])
                else:
                    stepped = 0
                lad = ladder_scores_for(net, p, q, device, bank)
                if q.get('ladder'):
                    budgets.append(shared_import_budget(p, q, list(q['cands']) + [q['ladder'][r] for r in LADDER]))
                if lad is not None:
                    seq = [float(lg[q['label']])] + [float(x) for x in lad]
                    for name, val in zip(('true',) + LADDER, seq):
                        rung_sum[name] += val
                    rung_n += 1
                    for a_, b_ in zip(seq, seq[1:]):
                        if a_ == b_:
                            ties += 1
                            continue
                        concord += int(a_ > b_)
                        pairs += 1
                riv = None if q.get('ident') or q.get('mixed') else lookup_rival(q)
                st[v]['floor'] += 1.0 / len(q['cands'])
                hits.append({'k': f"{q['address']}#{q.get('hid', len(q['slots']))}", 'hit': int(pred == truth)})
                if q.get('mixed'):
                    rv, rm = open_rival_scored(p, q, bank, device)
                    mixed_rows.append([int(q['answerable']), int(pred == REFUSE_LABEL), int(pred == q['truth_value']), float(rm) if rm == rm else -1.0, int(rv == q['truth_value']), stepped])
                fair = not q.get('open') and (not q.get('ident')) and (not q.get('mixed')) and (q.get('answerable', True) or not REFUSE)
                if fair:
                    if pred == truth and riv != truth:
                        b10 += 1
                    elif pred != truth and riv == truth:
                        b01 += 1
                rcos, rmargin = (None, float('nan')) if q.get('ident') or q.get('mixed') else lookup_rival_cos(p, q, bank, device)
                if q.get('ident'):
                    rv = ident_rivals(p, q)
                    rv.pop('_heur_accepted')
                    ident_rows.append([int(pred == truth)] + [int(rv[nm] == truth) for nm in IDENT_RIVALS] + [int(rv['heur'] is not None), int(p['tape'].values[q['cand_slots'][int(lg.argmax())]] in {p['tape'].values[s] for s in q['slots']})])
                if rcos is not None:
                    st[v]['rival_cos'] += int(rcos == truth)
                    if fair:
                        cos_items.append((int(pred == truth), int(rcos == truth), rmargin))
                if q.get('uniform'):
                    ub['true'][0] += float(lg[q['label']])
                    ub['true'][1] += 1
                    for c_, b_ in q['bucket_of'].items():
                        ub[b_][0] += float(lg[q['cands'].index(c_)])
                        ub[b_][1] += 1
                    orc = open_rival_cos(p, q, bank, device)
                    open_rows.append([int(pred == truth), int(orc == truth), int(orc is not None)])
                elif q.get('open'):
                    byr = {nm: val_ for val_, nm in q['rung_of'].items()}
                    seq = [float(lg[q['cands'].index(truth)])] + [float(lg[q['cands'].index(byr[r_])]) for r_ in LADDER]
                    for name, val in zip(('true',) + LADDER, seq):
                        rung_sum[name] += val
                    rung_n += 1
                    for a_, b_ in zip(seq, seq[1:]):
                        if a_ == b_:
                            ties += 1
                        else:
                            concord += int(a_ > b_)
                            pairs += 1
                    orc = open_rival_cos(p, q, bank, device)
                    open_rows.append([int(pred == truth), int(orc == truth), int(orc is not None)])
                if q.get('sparse'):
                    orp = own_row_rival(q)
                    sparse_rows.append([int(q['answerable']), int(pred == truth), int(pred == REFUSE_LABEL), int(riv == truth), int(rcos == truth), counting_margin(q), rmargin, len(q['own_rows']), int(orp == truth), int(orp == REFUSE_LABEL)])
            else:
                pred = exact_answer(q)
                truth = q['label']
                riv = pred
                st[v]['floor'] += 1.0 / (len(COUNT_LABELS) if v == 'count' else len(COMPARE_LABELS))
                exact_bad += int(pred != truth)
            st[v]['n'] += 1
            st[v]['model'] += int(pred == truth)
            st[v]['rival'] += int(riv == truth)
            conf[v, str(truth), str(pred)] += 1
        out = {}
        for v, t in st.items():
            if not t['n']:
                continue
            out[v] = {'n': t['n'], 'model_accuracy': t['model'] / t['n'], 'rival_accuracy': t['rival'] / t['n'], 'random_floor': t['floor'] / t['n']}
            if v == 'lookup':
                out[v]['rival_cos_accuracy'] = t['rival_cos'] / len(cos_items) if cos_items else float('nan')
                out[v]['rival_cos_answered'] = len(cos_items)
        out_lad = {'n_questions': rung_n, 'pairs': pairs, 'concordant': concord, 'ties_excluded': ties, 'import_budget_zero_rate': sum((1 for b in budgets if b == 0)) / max(1, len(budgets)), 'import_budget_mean': sum(budgets) / max(1, len(budgets)), 'concordance': concord / pairs if pairs else float('nan'), 'z_vs_half': (concord / pairs - 0.5) / math.sqrt(0.25 / pairs) if pairs else float('nan'), 'mean_phi': {k: rung_sum[k] / rung_n if rung_n else float('nan') for k in ('true',) + LADDER}}
        disc = b10 + b01
        out['lookup_paired_vs_rival'] = {'model_only_right': b10, 'rival_only_right': b01, 'discordant': disc, 'mcnemar_z': (b10 - b01) / math.sqrt(disc) if disc else float('nan')}

        def mcnemar(items):
            a = sum((1 for m, r, _ in items if m and (not r)))
            b = sum((1 for m, r, _ in items if r and (not m)))
            d = a + b
            return {'n': len(items), 'model_only_right': a, 'rival_only_right': b, 'discordant': d, 'mcnemar_z': (a - b) / math.sqrt(d) if d else float('nan'), 'max_achievable_z': math.sqrt(d) if d else 0.0, 'underpowered': bool(math.sqrt(d) <= 1.645)}
        out['lookup_paired_vs_rival_cos'] = mcnemar(cos_items)
        ms = sorted((m for _, _, m in cos_items if not math.isnan(m)))
        med = ms[len(ms) // 2] if ms else float('nan')
        out['lookup_paired_vs_rival_cos_by_margin'] = {'median_margin': med, 'low_margin': mcnemar([it for it in cos_items if it[2] <= med]), 'high_margin': mcnemar([it for it in cos_items if it[2] > med])}
        out['ladder'] = out_lad
        out['exact_mismatches'] = exact_bad
        out['confusion'] = {f'{a}|{b}->{c}': k for (a, b, c), k in sorted(conf.items())}
        out['lookup_item_hits'] = sorted(hits, key=lambda h: h['k'])
        out['_views'] = view_rows
        out['_sparse'] = sparse_rows
        out['_mixed'] = mixed_rows
        out['_reach'] = reach_rows
        out['_pair'] = pair_rows
        out['_cons'] = cons_rows
        if open_rows:
            a_ = sum((1 for m, r_, _ in open_rows if m and (not r_)))
            b_ = sum((1 for m, r_, _ in open_rows if r_ and (not m)))
            d_ = a_ + b_
            out['open'] = {'n': len(open_rows), 'random_floor': 0.25, 'accuracy': sum((r[0] for r in open_rows)) / len(open_rows), 'corpus_retrieval_accuracy': sum((r[1] for r in open_rows)) / len(open_rows), 'corpus_retrieval_answered': sum((r[2] for r in open_rows)), 'within_address_rivals_undefined': True, 'landscape_observed': {k: v[0] / v[1] if v[1] else float('nan') for k, v in ub.items()} if ub['true'][1] else None, 'landscape_counts': {k: v[1] for k, v in ub.items()}, 'landscape_near_possible': ADDRESS_FROM != 'anchor', 'paired_vs_corpus_retrieval': {'model_only_right': a_, 'rival_only_right': b_, 'discordant': d_, 'mcnemar_z': (a_ - b_) / math.sqrt(d_) if d_ else float('nan'), 'max_achievable_z': math.sqrt(d_) if d_ else 0.0, 'underpowered': bool(math.sqrt(d_) <= 1.645)}}
        if ident_rows:
            nn_ = len(ident_rows)
            blk = {'n': nn_, 'random_floor': 1.0 / IDENT_CANDS, 'accuracy': sum((r[0] for r in ident_rows)) / nn_, 'values': IDENT_VALUES, 'heuristic_answered': sum((r[1 + len(IDENT_RIVALS)] for r in ident_rows)) / nn_, 'value_agreement': sum((r[-1] for r in ident_rows)) / nn_}
            for i, nm in enumerate(IDENT_RIVALS):
                blk[f'rival_{nm}'] = sum((r[1 + i] for r in ident_rows)) / nn_
                blk[f'paired_vs_{nm}'] = mcnemar([(r[0], r[1 + i], 0.0) for r in ident_rows])
            out['ident'] = blk
        if sparse_rows:
            ans = [r for r in sparse_rows if r[0]]
            una = [r for r in sparse_rows if not r[0]]
            out['sparse'] = {'n': len(sparse_rows), 'n_answerable': len(ans), 'n_unanswerable': len(una), 'acc_answerable': sum((r[1] for r in ans)) / len(ans) if ans else float('nan'), 'refuse_recall': sum((r[2] for r in una)) / len(una) if una else float('nan'), 'false_refusal': sum((r[2] for r in ans)) / len(ans) if ans else float('nan'), 'rival_own_row_accuracy': sum((r[8] for r in sparse_rows)) / len(sparse_rows), 'n_no_own_row': sum((1 for r in sparse_rows if r[7] == 0)), 'acc_no_own_row': sum((r[1] for r in sparse_rows if r[7] == 0)) / max(1, sum((1 for r in sparse_rows if r[7] == 0)))}
        return out
    ex = examine(held, held_qs)
    ctrl = examine(pack, qs)
    tr_v, he_v = (ctrl.pop('_views', []), ex.pop('_views', []))
    pr_v = []
    if VIEWS > 1:
        with torch.no_grad():
            for views, qm, label in probe_graphs:
                L = torch.stack([torch.stack([net.phi(E, same, nf) for E, same, nf in graphs]) for graphs in views])
                lg = pool_views(L, qm)
                pr_v.append([int(int(lg.argmax()) == int(label)), int(int(L[0].argmax()) == int(label)), disagreement(L if qm is None else L[1:], qm)])
    recon = None
    if VIEWS > 1 and tr_v and he_v:

        def paired(rows):
            a = sum((1 for pl, sg, _ in rows if pl and (not sg)))
            b = sum((1 for pl, sg, _ in rows if sg and (not pl)))
            d = a + b
            return {'pooled_only_right': a, 'single_only_right': b, 'discordant': d, 'mcnemar_z': (a - b) / math.sqrt(d) if d else float('nan')}

        def d_auc(rows, i=0):
            err = [r[2] for r in rows if not r[i]]
            okd = [r[2] for r in rows if r[i]]
            if not err or not okd:
                return {'auc': float('nan'), 'z': float('nan')}
            a = s286.auc(err, okd)
            return {'auc': a, 'z': s286.auc_z(a, len(err), len(okd)), 'n_err': len(err), 'n_ok': len(okd)}

        def t_star_of(rows, i=0):
            groups = defaultdict(list)
            for r in rows:
                groups[r[2]].append(r)
            t, cum, seen = (None, 0, 0)
            for d in sorted(groups):
                g = groups[d]
                if (cum + sum((x[i] for x in g))) / (seen + len(g)) < 0.875:
                    break
                cum += sum((x[i] for x in g))
                seen += len(g)
                t = d
            return t

        def refusal_of(i):
            """280's rewards are fixed (+1/-1/+0.75), so answering beats abstaining iff
            p > 0.875 - a constant DERIVED, not chosen.

            Calibrated on the PROBE, applied held out. Train is kept alongside as the record of
            recon3's mistake, never used: the mind fits train, thin views stop disagreeing there,
            and a threshold read off that agreement admits everything or nothing (coverage 0.004).
            Region views largely repaired that - region3 read 0.76 train against 0.72 probe and
            0.69 held - but the probe is what makes the rule sound rather than lucky.

            The margin against BLANKET refusal is the honest comparison and it is reported with
            the count that decides whether it means anything: refusing everything already pays
            0.75, so a selective policy that clears it by a hair on forty answered questions has
            not yet shown it can tell what it knows.
            """
            t = t_star_of(pr_v if pr_v else tr_v, i)
            ans = [r for r in he_v if t is not None and r[2] <= t]
            ref_n = len(he_v) - len(ans)
            n_ok = sum((1 for r in ans if r[i]))
            acc = n_ok / len(ans) if ans else float('nan')
            sel = (n_ok - (len(ans) - n_ok) + 0.75 * ref_n) / len(he_v)
            z_be = (acc - 0.875) / math.sqrt(0.875 * 0.125 / len(ans)) if ans else float('nan')
            return {'p_star': 0.875, 'calibrated_on': 'probe' if pr_v else 'train', 'd_threshold': t, 'd_threshold_from_train': t_star_of(tr_v, i), 'held_coverage': len(ans) / len(he_v), 'held_n_answered': len(ans), 'held_acc_answered': acc, 'z_acc_vs_breakeven': z_be, 'held_reward_selective': sel, 'held_reward_always': sum((1 if r[i] else -1 for r in he_v)) / len(he_v), 'held_reward_blanket_refusal': 0.75}
        pv = paired(he_v)
        hd, rf = (d_auc(he_v), refusal_of(0))
        hd1, rf1 = (d_auc(he_v, 1), refusal_of(1))
        recon = {'views': VIEWS, 'view_mode': VIEW_MODE, 'thin_keep_p': 1.0 - ROW_DROPOUT if VIEW_MODE == 'thin' else None, 'held_pooled_vs_single': pv, 'held_d_auc': hd, 'probe_d_auc': d_auc(pr_v), 'train_d_auc': d_auc(tr_v), 'refusal': rf, 'answer_full': {'held_d_auc': hd1, 'probe_d_auc': d_auc(pr_v, 1), 'train_d_auc': d_auc(tr_v, 1), 'refusal': rf1}, 'gates': {'G_pooled_not_worse': bool(pv['mcnemar_z'] >= 0 if pv['discordant'] else True), 'G_d_predicts_error_held': bool(hd.get('z', 0) == hd.get('z', 0) and hd.get('z', 0) > 1.645), 'G_refusal_beats_blanket': bool(rf['held_reward_selective'] > 0.75)}}
        log(f'  RECON {json.dumps(recon)}')

    def _pearson(a, b):
        n = len(a)
        if n < 2:
            return float('nan')
        ma, mb = (sum(a) / n, sum(b) / n)
        va = sum(((x - ma) ** 2 for x in a))
        vb = sum(((x - mb) ** 2 for x in b))
        if va <= 0 or vb <= 0:
            return float('nan')
        return sum(((x - ma) * (y - mb) for x, y in zip(a, b))) / math.sqrt(va * vb)
    coh_block = None
    if COHERENCE:
        coh_block = {'held_out': coherence_block(net, held, bank, device, COHERENCE, random.Random(SEED + 9111)), 'train': coherence_block(net, pack, bank, device, COHERENCE, random.Random(SEED + 9112))}
        log(f'  COHERENCE {json.dumps(coh_block)}')

    def cblock(rows):
        """Both gates of ladder step 1, computed rather than eyeballed.

        GATE (a) - IS CHOOSING THE ROW A DECISION? The mind's lens against the three counting
        rules for choosing the same row, paired on the questions where an index on the place
        cannot answer at all (`beyond_own` - the constraint arm's analogue of walk-only, and
        mind-independent by construction). `decisive` is the strongest of the three and the one
        the gate is set against; the other two are printed so the comparison cannot be won by
        whichever direction happened to suit the tape.

        GATE (b) - DOES IT REACH MORE THAN THE ENUMERATION? `answerable` here is "the truth is
        what the tape says through SOME lens" - at most |own| shots, each drawn from every place
        on the tape. `walk_answerable` is the incumbent on the SAME question: the truth among
        eight candidates from eight nearby places. Both are properties of the tape and the
        operation, not of the mind, so this is a comparison of interfaces and nothing else.

        READS ARE PART OF THE COMPARISON. A lens costs one read; the walk costs REACH_K places.
        If the constraint reaches more at a fifth of the reads, that is the asymptotic argument
        of 335 turning into a measurement.
        """
        if not rows:
            return None
        n = len(rows)
        bo = [x for x in rows if not x[CIX['truth_in_own']]]

        def mc(sub, a, b):
            i = sum((1 for x in sub if x[a] and (not x[b])))
            j = sum((1 for x in sub if x[b] and (not x[a])))
            return {'mind_only': i, 'rival_only': j, 'n': len(sub), 'mcnemar_z': (i - j) / math.sqrt(i + j) if i + j else float('nan')}
        return {'n': n, 'n_beyond_own': len(bo), 'mean_lenses': sum((x[CIX['n_lens']] for x in rows)) / n, 'constrain_rate': sum((x[CIX['constrained']] for x in rows)) / n, 'hit_rate': sum((x[CIX['mind_right']] for x in rows)) / n, 'own_hit_rate': sum((x[CIX['truth_in_own']] for x in rows)) / n, 'answerable': sum((x[CIX['answerable']] for x in rows)) / n, 'present_topm': sum((x[CIX['present_topm']] for x in rows)) / n, 'walk_answerable': sum((x[CIX['walk_answerable']] for x in rows)) / n, 'reads_constraint': 1.0, 'reads_walk': float(REACH_K), 'beyond_own': {'hit': sum((x[CIX['mind_right']] for x in bo)) / len(bo) if bo else float('nan'), 'rare': sum((x[CIX['rare_right']] for x in bo)) / len(bo) if bo else float('nan'), 'frequent': sum((x[CIX['frequent_right']] for x in bo)) / len(bo) if bo else float('nan'), 'decisive': sum((x[CIX['decisive_right']] for x in bo)) / len(bo) if bo else float('nan'), 'vs_decisive': mc(bo, CIX['mind_right'], CIX['decisive_right']), 'vs_rare': mc(bo, CIX['mind_right'], CIX['rare_right']), 'vs_walk_rival': mc(bo, CIX['mind_right'], CIX['walk_rival_right'])}, 'chosen_share_when_constrained': sum((x[CIX['chosen_share']] for x in rows if x[CIX['constrained']])) / max(1, sum((x[CIX['constrained']] for x in rows)))}
    cons_block = None
    if ex.get('_cons'):
        cons_block = {'lenses': CONS_LENSES, 'topm': CONS_TOPM, 'reach_k': REACH_K, 'resolve': CONS_RESOLVE, 'held_out': cblock(ex['_cons']), 'train_control': cblock(ctrl.get('_cons') or [])}
        log(f'  CONS {json.dumps(cons_block)}')
    reach_block = None
    if ex.get('_reach'):

        def rpay(rows, thr):
            m, r = ([], [])

            def pay(sil, right, ansb):
                v = mixed_payoff(sil, right, ansb)
                return (v + 1.0) / 2.0 if REACH_GAMMA < 1.0 else v
            for ansb, sil, right, mg, rhit, step, *_ in rows:
                m.append((REACH_GAMMA if step else 1.0) * pay(bool(sil), bool(right), bool(ansb)) - STEP_COST * step)
                r.append(pay(mg < thr, bool(rhit), bool(ansb)))
            return (m, r)
        tr = ctrl.get('_reach') or ex['_reach']
        grid = sorted({x[RIX['rival_margin']] for x in tr} | {-2.0, 2.0})
        rthr = max(grid, key=lambda t: sum(rpay(tr, t)[1]) / max(1, len(tr)))

        def othermind(rows):
            """336: A NATIVE MIND AGAINST A TRANSPLANTED ONE, ON THE SAME TAPE.

            This project's strongest claim is that Phi holds no facts: 5633 parameters, ink that
            never saw the corpus, and a mind fitted to wiki that reads a news tape with its
            route and its pick intact. But "intact" was measured against COUNTING RIVALS, never
            against the obvious control - a mind fitted to news itself. If Phi is really free of
            the corpus, the native and the transplanted mind must be INDISTINGUISHABLE here.

            PRE-REGISTERED, before these numbers exist:
              indistinguishable  |z| < 1.645 on the whole exam AND on walk-only, in both arms.
                                 The claim of separation survives and is now controlled.
              native ahead       z >= +1.645 for the run's own mind on >= 3 of 4 seeds. Phi
                                 carries a decision policy fitted to its corpus. The separation
                                 is weaker than claimed and must be restated, not defended.
              transplant ahead   z <= -1.645. Nothing in the theory predicts it; it would mean
                                 the wiki tape teaches a better reader than the news tape does,
                                 which is a claim about corpora and needs its own experiment.

            A NULL RESULT IS THE GOOD ONE HERE, which is unusual in this file and worth saying
            out loud: the standing claim is what indistinguishable supports. That is exactly why
            the comparison is paired inside one run - an underpowered comparison also produces a
            small z, and only the discordant counts show which of the two happened.
            """
            if not rows or OTHER_NET is None:
                return None

            def mc(sub):
                b = sum((1 for x in sub if x[RIX['mind_right']] and (not x[RIX['other_right']])))
                c = sum((1 for x in sub if x[RIX['other_right']] and (not x[RIX['mind_right']])))
                return {'n': len(sub), 'this_only': b, 'other_only': c, 'this': sum((x[RIX['mind_right']] for x in sub)), 'other': sum((x[RIX['other_right']] for x in sub)), 'mcnemar_z': (b - c) / math.sqrt(b + c) if b + c else float('nan'), 'identical': b + c == 0, 'underpowered': 0 < b + c and math.sqrt(b + c) <= 1.645}
            wo = [x for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']])]
            return {'all': mc(rows), 'walk_only': mc(wo), 'confirm': mc([x for x in rows if x[RIX['truth_in_own']]]), 'step_rate': sum((x[RIX['stepped']] for x in rows)) / len(rows), 'other_step_rate': sum((x[RIX['other_stepped']] for x in rows)) / len(rows)}

        def marginblock(rows):
            """341's post-mortem, as a number: IS THE MARGIN A PROPERTY OF THE STAGE?

            341 trained the margin and the ROUTE moved - stepping rose 4-6x and route enrichment
            fell from 11.3x to 2.9x. The suspected mechanism is that the margin is not
            comparable across stages: it is the gap between the best and second-best option of
            WHICHEVER DISTRIBUTION ANSWERED, and stage two's option set is a different size and
            shape from stage one's. If so, a term that rewards large margins can raise them by
            CHANGING WHICH STAGE ANSWERS instead of by being right more often - which is this
            project's oldest fault class (an option's logit being a max over a set) reappearing
            in the quantity used to rank questions.

            That was a hypothesis when 341 closed. It is testable on runs already in hand, from
            columns already written, so it is measured rather than argued: the margin's mean
            where the mind stayed, where it stepped, and where it read twice. If stepping
            carries a systematically larger margin, the coupling is real and any future attempt
            to train the ranking must break it first. If the means are level, the diagnosis is
            wrong and 341's failure needs another explanation.

            `by_right` is the control: the margin SHOULD separate right from wrong - that is
            calibration, and 340 measured it at AUC 0.866. A stage effect on top of that is the
            confound; the two are printed together so one cannot be mistaken for the other.
            """
            if not rows:
                return None

            def mean(sub):
                return sum((x[RIX['pick_margin']] for x in sub)) / len(sub) if sub else float('nan')
            return {'stayed': mean([x for x in rows if not x[RIX['stepped']]]), 'stepped': mean([x for x in rows if x[RIX['stepped']] == 1]), 'line': mean([x for x in rows if x[RIX['stepped']] == 2]), 'depth2': mean([x for x in rows if x[RIX['depth_reached']] == 2]), 'n_stayed': sum((1 for x in rows if not x[RIX['stepped']])), 'n_stepped': sum((1 for x in rows if x[RIX['stepped']] == 1)), 'by_right': {'right': mean([x for x in rows if x[RIX['mind_right']]]), 'wrong': mean([x for x in rows if not x[RIX['mind_right']]])}, 'stayed_right': mean([x for x in rows if not x[RIX['stepped']] and x[RIX['mind_right']]]), 'stayed_wrong': mean([x for x in rows if not x[RIX['stepped']] and (not x[RIX['mind_right']])]), 'stepped_right': mean([x for x in rows if x[RIX['stepped']] == 1 and x[RIX['mind_right']]]), 'stepped_wrong': mean([x for x in rows if x[RIX['stepped']] == 1 and (not x[RIX['mind_right']])])}

        def gateblock(rows):
            """337 USED RATHER THAN ADMIRED: the mind declines the questions it cannot answer.

            337 established that Phi's own margin ranks holes by answerability better than any
            count (AUC 0.731 against top_share 0.668 and |own| 0.464, four seeds of four). That
            was a measurement. This is the operation it licenses: let the mind ANSWER ONLY THE
            TOP FRACTION by its own margin and refuse the rest, and see what the answers are
            worth.

            MATCHED COVERAGE IS THE WHOLE DESIGN. A gate that answers less is more precise for
            free - that is arithmetic, not a faculty - so every ranker here lets through EXACTLY
            k questions, and the only difference between them is WHICH k. The rivals are the two
            counts 337 already beat, gating at the same k, plus the ungated run as the floor.

            THREE NUMBERS PER GATE, because precision alone can be bought by answering nothing:
              precision  of the k it answered, how many it got right
              yield      how many correct answers survive the gate, in absolute count. A gate
                         that doubles precision while keeping a third of the hits has lost.
              payoff     the run's own reward with the refused questions scored as silence -
                         mixed_payoff, unchanged, which already prices correct silence against
                         a false one. This is the only column that can say a gate is WORTH it
                         rather than merely sharper.

            PAIRED, not two rates: question i counts for a ranker when it is both kept and
            right, so McNemar applies over the same questions and the discordant counts are
            what carry the contrast - as everywhere else in this file.

            THE COMPARISON IS AGAINST COUNTING, AND THE INVARIANT IS UNTOUCHED. Nothing is
            trained, no tape changes, no new faculty appears: this reads columns the exam
            already wrote. If the counts gate as well as the mind, 337's AUC was real but
            operationally empty, and that is a finding this block must be able to produce.

            WHERE THE KEPT ANSWERS COME FROM, and this is why `composition` is here. 83-90% of
            everything the mind gets right on this tape is a CONFIRM question - the truth was
            already among the question's own rows, where a lookup on the place is optimal and
            asking Phi to beat it is asking for a better index. A gate scored over all questions
            is therefore mostly a gate over confirms, and its precision says little about the
            half this project's claims actually live on. So the block is also run over the
            WALK-ONLY rows alone (`gate_walk_only` in the report), where a lookup is
            structurally incapable, and the composition of the kept set is printed either way.
            """
            if not rows:
                return None
            n = len(rows)
            right = [bool(x[RIX['mind_right']]) for x in rows]
            ansb = [bool(x[RIX['answerable']]) for x in rows]
            grng = random.Random(SEED + 9340)
            rankers = {'mind': [float(x[RIX['pick_margin']]) for x in rows], 'count_n_own': [float(x[RIX['n_own']]) for x in rows], 'count_top_share': [float(x[RIX['top_share']]) for x in rows], 'random': [grng.random() for _ in rows]}

            def pay(keep):
                v = [mixed_payoff(i not in keep, right[i], ansb[i]) for i in range(n)]
                return sum(((x + 1.0) / 2.0 for x in v)) / n if REACH_GAMMA < 1.0 else sum(v) / n
            sil = pay(set())
            out = {'n': n, 'ungated_hit_rate': sum(right) / n, 'ungated_payoff': pay(set(range(n))), 'always_silent': sil, 'fractions': list(GATE_FRACTIONS)}
            for fr in GATE_FRACTIONS:
                k = max(1, int(round(fr * n)))
                kept = {nm: gate_top(s, k) for nm, s in rankers.items()}
                d = {'k': k}
                for nm, ks in kept.items():
                    hits = sum((1 for i in ks if right[i]))
                    d[nm] = {'precision': hits / k, 'yield': hits, 'payoff': pay(ks), 'gain': pay(ks) - sil}
                km = kept['mind']
                d['composition'] = {'confirm': sum((1 for i in km if rows[i][RIX['truth_in_own']])), 'walk_only': sum((1 for i in km if rows[i][RIX['answerable']] and (not rows[i][RIX['truth_in_own']]))), 'neither': sum((1 for i in km if not rows[i][RIX['answerable']] and (not rows[i][RIX['truth_in_own']]))), 'right_confirm': sum((1 for i in km if right[i] and rows[i][RIX['truth_in_own']])), 'right_walk_only': sum((1 for i in km if right[i] and rows[i][RIX['answerable']] and (not rows[i][RIX['truth_in_own']])))}
                for nm in ('count_n_own', 'count_top_share'):
                    b = sum((1 for i in range(n) if (i in kept['mind'] and right[i]) and (not (i in kept[nm] and right[i]))))
                    c = sum((1 for i in range(n) if (i in kept[nm] and right[i]) and (not (i in kept['mind'] and right[i]))))
                    d[f'vs_{nm}'] = {'mind_only': b, 'rival_only': c, 'mcnemar_z': (b - c) / math.sqrt(b + c) if b + c else float('nan')}
                out[f'{fr:.2f}'] = d
            return out

        def rankblock(rows):
            """337: THE MIND CHOOSES THE QUESTION.

            Every measurement in this project so far hands the mind a hole and grades the
            filler. This one asks the other question: given the whole tape, WHICH HOLES CAN BE
            ANSWERED? Nothing in training says. `answerable` is a property of the tape and the
            walk - it is never an input, never a teacher, and the mind cannot see whether the
            truth is among the values it was offered. So if Phi's own confidence sorts the
            answerable holes above the unanswerable ones, a thing that holds no facts has made
            a judgement about what the facts support.

            THE RIVALS ARE THE COUNTS, as everywhere else: how many rows the place has
            (`n_own`) and how dominant the walk's best offer is at its own place (`top_share`).
            Both are exact, both are free, and either could produce the whole effect - a place
            with many rows is more likely to be answerable for reasons that have nothing to do
            with a mind.

            PRE-REGISTERED, before any of these numbers exist: the direction is real only if
            the mind's AUC beats BOTH counting rivals on at least 3 of 4 seeds, in each arm
            separately. AUC and not accuracy because a cut would have to be chosen, and a
            chosen cut is the fitting this project does not do. precision@k is printed beside
            it as the readable form, at k = a tenth, a quarter, and the true number of
            answerable questions - the last is the one a perfect ranker scores 1.0 on.

            THREE TARGETS, because "could answer" has three honest readings and they are not
            the same question. `answerable` is what the WALK reaches. `ceiling` adds the holes
            already answered by their own rows. `right` is whether THE MIND'S OWN ANSWER WAS
            CORRECT - and that is the one a gate actually needs, because a gate does not choose
            questions whose truth is present, it chooses questions it will get right. The first
            two are properties of the tape; the third is a property of the mind on the tape, so
            it is the only one of the three where the counting rivals are at a real
            disadvantage - they cannot see the answer either.
            """
            if not rows:
                return None
            out = {'n': len(rows)}
            for tgt, lab in (('answerable', [1 if x[RIX['answerable']] else 0 for x in rows]), ('ceiling', [1 if x[RIX['answerable']] or x[RIX['truth_in_own']] else 0 for x in rows]), ('right', [1 if x[RIX['mind_right']] else 0 for x in rows])):
                npos = sum(lab)
                ks = [max(1, len(rows) // 10), max(1, len(rows) // 4), max(1, npos)]
                d = {'base_rate': npos / len(rows), 'k': ks}
                for nm, ix in (('mind_margin', RIX['pick_margin']), ('mind_score', RIX['pick_score']), ('count_n_own', RIX['n_own']), ('count_top_share', RIX['top_share'])):
                    s = [float(x[ix]) for x in rows]
                    d[nm] = {'auc': rank_auc(s, lab), 'prec': [prec_at(s, lab, k) for k in ks]}
                out[tgt] = d
            return out

        def rblock(rows):
            if not rows:
                return None
            mp, rp = rpay(rows, rthr)
            n = len(rows)
            yes = [x for x in rows if x[RIX['answerable']]]
            no = [x for x in rows if not x[RIX['answerable']]]
            w = sum((1 for a, b in zip(mp, rp) if a > b))
            l = sum((1 for a, b in zip(mp, rp) if a < b))
            d = w + l
            return {'n': n, 'reachable_rate': len(yes) / n, 'count_rival_ties': sum((x[RIX['cr_ties']] for x in rows)) / n, 'move_share': {m: sum((1 for x in rows if x[RIX['move_id']] == i)) / n for i, m in enumerate(MOVES)} if MOVES_ON else {}, 'move_hit': {m: sum((x[RIX['mind_right']] for x in rows if x[RIX['move_id']] == i)) / max(1, sum((1 for x in rows if x[RIX['move_id']] == i))) for i, m in enumerate(MOVES)} if MOVES_ON else {}, 'move_teach_live': _MOVE_LIVE['live'] / max(1, _MOVE_LIVE['n']) if MOVE_TEACH else 0.0, 'move_teach_ballot': _MOVE_LIVE['ballot'] / max(1, _MOVE_LIVE['n']) if MOVE_TEACH else 0.0, 'move_teach_seen': _MOVE_LIVE['n'] if MOVE_TEACH else 0, 'route_on': ROUTE_ON, 'route_on_live': _ROUTE_LIVE['live'] / max(1, _ROUTE_LIVE['n']) if ROUTE_ON != 'all' else 1.0, 'route_on_seen': _ROUTE_LIVE['n'] if ROUTE_ON != 'all' else 0, 'mean_candidates': sum((x[RIX['n_cands']] for x in rows)) / n, 'question_rank': rankblock(rows), 'margin_by_stage': marginblock(rows), 'gate': gateblock(rows), 'gate_walk_only': gateblock([x for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']])]), 'other_mind': othermind(rows), 'reachable_wide': sum((x[RIX['reachable_wide']] for x in rows)) / n, 'reachable_random': sum((x[RIX['reachable_random']] for x in rows)) / n, 'random_floor': sum((1.0 / max(1, x[RIX['n_cands']]) for x in rows)) / n, 'payoff_mind': sum(mp) / n, 'payoff_rival': sum(rp) / n, 'always_silent': sum((mixed_payoff(True, False, bool(x[RIX['answerable']])) for x in rows)) / n, 'found_rate': sum((x[RIX['mind_right']] for x in yes)) / len(yes) if yes else float('nan'), 'rival_found_rate': sum((x[RIX['rival_right']] for x in yes)) / len(yes) if yes else float('nan'), 'correct_silence': sum((x[RIX['silent']] for x in no)) / len(no) if no else float('nan'), 'false_silence': sum((x[RIX['silent']] for x in yes)) / len(yes) if yes else float('nan'), 'step_rate': sum((x[RIX['stepped']] for x in rows)) / n, 'own_hit_rate': sum((x[RIX['truth_in_own']] for x in rows)) / n, 'own_rival_hit_rate': sum((x[RIX['own_rival_right']] for x in rows)) / n, 'own_paired': (lambda ow: {'n': len(ow), 'mind_only': sum((1 for x in ow if x[RIX['mind_right']] and (not x[RIX['own_rival_right']]))), 'rival_only': sum((1 for x in ow if x[RIX['own_rival_right']] and (not x[RIX['mind_right']]))), 'both': sum((1 for x in ow if x[RIX['mind_right']] and x[RIX['own_rival_right']])), 'neither': sum((1 for x in ow if not x[RIX['mind_right']] and (not x[RIX['own_rival_right']]))), 'mcnemar_z': (lambda b, c: (b - c) / math.sqrt(b + c) if b + c else float('nan'))(sum((1 for x in ow if x[RIX['mind_right']] and (not x[RIX['own_rival_right']]))), sum((1 for x in ow if x[RIX['own_rival_right']] and (not x[RIX['mind_right']]))))})([x for x in rows if x[RIX['truth_in_own']]]), 'own_rival_of_own': sum((x[RIX['own_rival_right']] for x in rows if x[RIX['truth_in_own']])) / max(1, sum((x[RIX['truth_in_own']] for x in rows))), 'hit_of_own': sum((x[RIX['mind_right']] for x in rows if x[RIX['truth_in_own']])) / max(1, sum((x[RIX['truth_in_own']] for x in rows))), 'ceiling': sum((1 for x in rows if x[RIX['answerable']] or x[RIX['truth_in_own']])) / n, 'walk_only_arrive': (lambda wo: sum((1 for x in wo if x[RIX['stepped']])) / len(wo) if wo else float('nan'))([x for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']])]), 'walk_only_pick': (lambda ws: {'n': len(ws), 'mind': sum((x[RIX['mind_right']] for x in ws)), 'rival': sum((x[RIX['rival_right']] for x in ws)), 'count_rival': sum((x[RIX['count_rival_right']] for x in ws)), 'count_rival_rate': sum((x[RIX['count_rival_right']] for x in ws)) / len(ws) if ws else float('nan'), 'vs_count_mind_only': sum((1 for x in ws if x[RIX['mind_right']] and (not x[RIX['count_rival_right']]))), 'vs_count_rival_only': sum((1 for x in ws if x[RIX['count_rival_right']] and (not x[RIX['mind_right']]))), 'vs_count_z': (lambda b_, c_: (b_ - c_) / math.sqrt(b_ + c_) if b_ + c_ else float('nan'))(sum((1 for x in ws if x[RIX['mind_right']] and (not x[RIX['count_rival_right']]))), sum((1 for x in ws if x[RIX['count_rival_right']] and (not x[RIX['mind_right']])))), 'hit_rate': sum((x[RIX['mind_right']] for x in ws)) / len(ws) if ws else float('nan'), 'rival_rate': sum((x[RIX['rival_right']] for x in ws)) / len(ws) if ws else float('nan'), 'mind_only': sum((1 for x in ws if x[RIX['mind_right']] and (not x[RIX['rival_right']]))), 'rival_only': sum((1 for x in ws if x[RIX['rival_right']] and (not x[RIX['mind_right']]))), 'mcnemar_z': (lambda b_, c_: (b_ - c_) / math.sqrt(b_ + c_) if b_ + c_ else float('nan'))(sum((1 for x in ws if x[RIX['mind_right']] and (not x[RIX['rival_right']]))), sum((1 for x in ws if x[RIX['rival_right']] and (not x[RIX['mind_right']]))))})([x for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]) and x[RIX['stepped']]]), 'walk_only_rate': sum((1 for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / n, 'hit_of_walk_only': sum((x[RIX['mind_right']] for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / max(1, sum((1 for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']])))), 'count_rival_of_walk_only': sum((x[RIX['count_rival_right']] for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / max(1, sum((1 for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']])))), 'count_rival_hit_rate': sum((x[RIX['count_rival_right']] for x in rows)) / n, 'deep_rate': sum((1 for x in rows if x[RIX['depth_reached']] >= 2)) / n, 'deep_only_rate': sum((x[RIX['deep_only']] for x in rows)) / n, 'hit_of_deep_only': sum((x[RIX['mind_right']] for x in rows if x[RIX['deep_only']])) / max(1, sum((x[RIX['deep_only']] for x in rows))) if any((x[RIX['deep_only']] for x in rows)) else float('nan'), 'hit_of_deep': sum((x[RIX['mind_right']] for x in rows if x[RIX['depth_reached']] >= 2)) / max(1, sum((1 for x in rows if x[RIX['depth_reached']] >= 2))) if any((x[RIX['depth_reached']] >= 2 for x in rows)) else float('nan'), 'hit_of_depth1': sum((x[RIX['mind_right']] for x in rows if x[RIX['depth_reached']] == 1)) / max(1, sum((1 for x in rows if x[RIX['depth_reached']] == 1))) if any((x[RIX['depth_reached']] == 1 for x in rows)) else float('nan'), 'bisect': (lambda ws: {'n': len(ws), 'splits_mean': sum((x[RIX['bisect_splits']] for x in ws)) / len(ws) if ws else float('nan'), 'bisect_right': sum((x[RIX['bisect_right']] for x in ws)), 'flat_right': sum((x[RIX['mind_right']] for x in ws)), 'bisect_only': sum((1 for x in ws if x[RIX['bisect_right']] and (not x[RIX['mind_right']]))), 'flat_only': sum((1 for x in ws if x[RIX['mind_right']] and (not x[RIX['bisect_right']]))), 'mcnemar_z': (lambda b_, c_: (b_ - c_) / math.sqrt(b_ + c_) if b_ + c_ else float('nan'))(sum((1 for x in ws if x[RIX['bisect_right']] and (not x[RIX['mind_right']]))), sum((1 for x in ws if x[RIX['mind_right']] and (not x[RIX['bisect_right']]))))})([x for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]) and x[RIX['stepped']]]) if BISECT else None, 'rival_of_walk_only': sum((x[RIX['rival_right']] for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / max(1, sum((1 for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']])))), 'walk_only_paired': (lambda wo: {'n': len(wo), 'mind_only': sum((1 for x in wo if x[RIX['mind_right']] and (not x[RIX['rival_right']]))), 'rival_only': sum((1 for x in wo if x[RIX['rival_right']] and (not x[RIX['mind_right']]))), 'both': sum((1 for x in wo if x[RIX['mind_right']] and x[RIX['rival_right']])), 'neither': sum((1 for x in wo if not x[RIX['mind_right']] and (not x[RIX['rival_right']]))), 'mcnemar_z': (lambda b, c: (b - c) / math.sqrt(b + c) if b + c else float('nan'))(sum((1 for x in wo if x[RIX['mind_right']] and (not x[RIX['rival_right']]))), sum((1 for x in wo if x[RIX['rival_right']] and (not x[RIX['mind_right']])))), 'stepped': sum((x[RIX['stepped']] for x in wo))})([x for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']])]), 'cand_places': sum((x[RIX['n_places']] for x in rows)) / n, 'line_reach_rate': sum((x[RIX['line_reach']] for x in rows)) / n, 'step_line_rate': sum((x[RIX['step_line']] for x in rows)) / n, 'line_only_rate': sum((1 for x in rows if x[RIX['line_reach']] and (not x[RIX['truth_in_own']]) and (not x[RIX['answerable']]))) / n, 'line_only_paired': (lambda lo: {'n': len(lo), 'mind_only': sum((1 for x in lo if x[RIX['mind_right']] and (not x[RIX['line_rival']]))), 'rival_only': sum((1 for x in lo if x[RIX['line_rival']] and (not x[RIX['mind_right']]))), 'mcnemar_z': (lambda b, c: (b - c) / math.sqrt(b + c) if b + c else float('nan'))(sum((1 for x in lo if x[RIX['mind_right']] and (not x[RIX['line_rival']]))), sum((1 for x in lo if x[RIX['line_rival']] and (not x[RIX['mind_right']]))))})([x for x in rows if x[RIX['line_reach']] and (not x[RIX['truth_in_own']]) and (not x[RIX['answerable']])]), 'steps_on_walk_only': sum((x[RIX['stepped']] for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / max(1, sum((x[RIX['stepped']] for x in rows))), 'router': (lambda rows, k: (lambda base: {'n_stepped': k, 'mind_enrichment': sum((x[RIX['stepped']] for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / k / base if k and base else float('nan'), 'count_enrichment': sum((1 for x in sorted(rows, key=lambda y: (y[RIX['n_own']], y[RIX['max_own_count']]))[:k] if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / k / base if k and base else float('nan'), 'share_enrichment': sum((1 for x in sorted(rows, key=lambda y: -y[RIX['top_share']])[:k] if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / k / base if k and base else float('nan'), 'top_share_when_stepped': sum((x[RIX['top_share']] for x in rows if x[RIX['stepped']])) / max(1, sum((x[RIX['stepped']] for x in rows))), 'top_share_when_not': sum((x[RIX['top_share']] for x in rows if not x[RIX['stepped']])) / max(1, sum((1 for x in rows if not x[RIX['stepped']]))), 'n_own_when_stepped': sum((x[RIX['n_own']] for x in rows if x[RIX['stepped']])) / max(1, sum((x[RIX['stepped']] for x in rows))), 'n_own_when_not': sum((x[RIX['n_own']] for x in rows if not x[RIX['stepped']])) / max(1, sum((1 for x in rows if not x[RIX['stepped']])))})(sum((1 for x in rows if x[RIX['answerable']] and (not x[RIX['truth_in_own']]))) / len(rows)))(rows, sum((x[RIX['stepped']] for x in rows))), 'hit_rate': sum((x[RIX['mind_right']] for x in rows)) / n, 'rival_hit_rate': sum((x[RIX['rival_right']] for x in rows)) / n, 'world_rows_own': sum((x[RIX['world_rows_own']] for x in rows)) / n, 'world_rows_candidate': sum((x[RIX['rows_candidate']] for x in rows)) / n, 'world_rows_expand': sum((x[RIX['rows_expand']] for x in rows)) / n, 'world_rows_expand_when_stepped': sum((x[RIX['rows_expand']] for x in rows if x[RIX['stepped']])) / max(1, sum((x[RIX['stepped']] for x in rows))), 'world_rows_expand_when_not': sum((x[RIX['rows_expand']] for x in rows if not x[RIX['stepped']])) / max(1, sum((1 for x in rows if not x[RIX['stepped']]))), 'step_vs_size_r': _pearson([x[RIX['stepped']] for x in rows], [x[RIX['rows_expand']] for x in rows]), 'found_where_rival_missed': sum((1 for x in yes if x[RIX['mind_right']] and (not x[RIX['rival_right']]))), 'missed_where_rival_found': sum((1 for x in yes if x[RIX['rival_right']] and (not x[RIX['mind_right']]))), 'paired_payoff': {'mind_better': w, 'rival_better': l, 'discordant': d, 'mcnemar_z': (w - l) / math.sqrt(d) if d else float('nan'), 'max_achievable_z': math.sqrt(d) if d else 0.0, 'underpowered': bool(math.sqrt(d) <= 1.645)}}
        for _arm in (ex.get('_reach'), ctrl.get('_reach')):
            if _arm:
                _hb = rblock(_arm)
                if _hb and _hb['hit_rate'] > _hb['ceiling'] + 1e-09:
                    log(f"  REACH BOOKKEEPING BROKEN: hit_rate {_hb['hit_rate']:.4f} exceeds ceiling {_hb['ceiling']:.4f} - a reachability the report does not account for. Fix the ceiling before reading anything else.")
                    return 1
        reach_block = {'rival_threshold': rthr, 'places': REACH_K, 'cands_cap': REACH_CANDS, 'no_refuse': REACH_NO_REFUSE, 'lookahead': REACH_LOOKAHEAD, 'frame_fp': FRAME_FP, 'import': REACH_IMPORT, 'home_cos': REACH_HOME_COS, 'line_step': REACH_LINE, 'confirm': REACH_CONFIRM, 'conf_window': CONF_WINDOW, 'home_cos_stage': HOME_COS_STAGE, 'speak_batch': SPEAK_BATCH, 'speak_weight': SPEAK_WEIGHT, 'move_teach': MOVE_TEACH, 'route_on': ROUTE_ON, 'calib_batch': CALIB_BATCH, 'calib_weight': CALIB_WEIGHT, 'two_way_by': TWO_WAY_BY, 'gamma': REACH_GAMMA, 'equal_tails': EQUAL_TAILS, 'deep_root': DEEP_ROOT, 'two_way': TWO_WAY, 'stage2_always': STAGE2_ALWAYS, 'depth': REACH_DEPTH, 'compass': REACH_COMPASS, 'held_out': rblock(ex['_reach']), 'train_control': rblock(ctrl.get('_reach') or [])}
        _h = reach_block['held_out'] or {}
        if _h and _h.get('false_silence', 0.0) >= 0.999 and (_h.get('step_rate', 1.0) <= 0.01):
            reach_block['void_arm'] = 'always-silent'
            log('  REACH ARM IS VOID: false_silence 1.0 and no stepping - the mind matched always-silent, which is correct play against this payoff and says nothing about search. Re-run with --reach-no-refuse.')
        log(f'  REACH {json.dumps(reach_block)}')
    pair_block = None
    if ex.get('_pair'):

        def pblock(rows):
            if not rows:
                return None
            n = len(rows)

            def sub(pred):
                s = [x for x in rows if pred(x)]
                return s

            def paired(s, rival):
                """McNemar, mind against one counting rule, on the questions of `s`."""
                b = sum((1 for x in s if x[PIX['mind_right']] and (not x[PIX[rival]])))
                c = sum((1 for x in s if x[PIX[rival]] and (not x[PIX['mind_right']])))
                return {'n': len(s), 'mind_only': b, 'rival_only': c, 'both': sum((1 for x in s if x[PIX['mind_right']] and x[PIX[rival]])), 'mcnemar_z': (b - c) / math.sqrt(b + c) if b + c else float('nan'), 'max_achievable_z': math.sqrt(b + c) if b + c else 0.0, 'underpowered': bool(math.sqrt(b + c) <= 1.645)}
            off = sub(lambda x: x[PIX['both_offered']])
            comp = sub(lambda x: x[PIX['both_offered']] and (not x[PIX['marg_right']]) and (not x[PIX['joint_seen']]))
            comps = [x for x in comp if not x[PIX['bag_seen']]]

            def co(cs):
                return {'n': len(cs), 'rate': len(cs) / n, 'mind_right': sum((x[PIX['mind_right']] for x in cs)), 'hit_rate': sum((x[PIX['mind_right']] for x in cs)) / len(cs) if cs else float('nan'), 'random_floor': sum((1.0 / max(1, x[PIX['n_pairs']]) for x in cs)) / len(cs) if cs else float('nan'), 'binomial_z': (lambda k, e: (k - e) / math.sqrt(e * (1 - e / max(1, len(cs)))) if e > 0 else float('nan'))(sum((x[PIX['mind_right']] for x in cs)), sum((1.0 / max(1, x[PIX['n_pairs']]) for x in cs))), 'one_hole_mean': sum((x[PIX['one_right']] for x in cs)) / (2 * len(cs)) if cs else float('nan'), 'indep_expected': sum((x[PIX['right_a']] for x in cs)) * sum((x[PIX['right_b']] for x in cs)) / len(cs) ** 2 * len(cs) if cs else float('nan'), 'right_a': sum((x[PIX['right_a']] for x in cs)) / len(cs) if cs else float('nan'), 'right_b': sum((x[PIX['right_b']] for x in cs)) / len(cs) if cs else float('nan')}
            return {'n': n, 'both_offered': len(off) / n, 'mind_exact': sum((x[PIX['mind_right']] for x in rows)) / n, 'mind_exact_of_offered': sum((x[PIX['mind_right']] for x in off)) / len(off) if off else float('nan'), 'holes_right_mean': sum((x[PIX['one_right']] for x in rows)) / (2 * n), 'marginal_exact': sum((x[PIX['marg_right']] for x in rows)) / n, 'joint_exact': sum((x[PIX['joint_right']] for x in rows)) / n, 'joint_seen_rate': sum((x[PIX['joint_seen']] for x in rows)) / n, 'bag_seen_rate': sum((x[PIX['bag_seen']] for x in rows)) / n, 'bag_exact': sum((x[PIX['bag_right']] for x in rows)) / n, 'in_own_both': sum((1 for x in rows if x[PIX['in_own_a']] and x[PIX['in_own_b']])) / n, 'offered_a': sum((x[PIX['offered_a']] for x in rows)) / n, 'offered_b': sum((x[PIX['offered_b']] for x in rows)) / n, 'mean_pair_worlds': sum((x[PIX['n_pairs']] for x in rows)) / n, 'world_rows': sum((x[PIX['world_rows']] for x in rows)) / n, 'first_hole_rate': sum((x[PIX['first_hole']] for x in rows)) / n, 'vs_marginal': paired(rows, 'marg_right'), 'vs_marginal_offered': paired(off, 'marg_right'), 'COMP_ONLY': co(comp), 'COMP_STRICT': co(comps)}
        pair_block = {'cands': PAIR_CANDS, 'max_rows': PAIR_MAX_ROWS, 'per_line': PAIR_PER_LINE, 'follow': PAIR_FOLLOW, 'blind': PAIR_BLIND, 'frame_max': FRAME_MAX, 'held_out': pblock(ex['_pair']), 'train_control': pblock(ctrl.get('_pair') or [])}
        log(f'  PAIR {json.dumps(pair_block)}')
    mixed_block = None
    if ex.get('_mixed'):

        def payoffs(rows, thr):
            """Per question: what the mind earned and what thresholded retrieval earned."""
            m, r = ([], [])
            for ans, sil, right, marg, rhit, step in rows:
                m.append(mixed_payoff(bool(sil), bool(right), bool(ans)) - STEP_COST * step)
                silent = marg < thr
                r.append(mixed_payoff(silent, bool(rhit), bool(ans)))
            return (m, r)
        tr = ctrl.get('_mixed') or ex['_mixed']
        grid = sorted({row[3] for row in tr} | {-1.0, 2.0})
        thr = max(grid, key=lambda t: sum(payoffs(tr, t)[1]) / max(1, len(tr)))

        def block(rows):
            mp, rp = payoffs(rows, thr)
            n = max(1, len(rows))
            ansr = [x for x in rows if x[0]]
            una = [x for x in rows if not x[0]]
            w = sum((1 for a, b in zip(mp, rp) if a > b))
            l = sum((1 for a, b in zip(mp, rp) if a < b))
            d = w + l
            return {'n': len(rows), 'answerable_rate': len(ansr) / n, 'payoff_mind': sum(mp) / n, 'payoff_rival': sum(rp) / n, 'step_rate': sum((x[5] for x in rows)) / n, 'always_answer': sum((mixed_payoff(False, bool(x[2]), bool(x[0])) for x in rows)) / n, 'always_silent': sum((mixed_payoff(True, False, bool(x[0])) for x in rows)) / n, 'found_rate': sum((x[2] for x in ansr)) / len(ansr) if ansr else float('nan'), 'rival_found_rate': sum((x[4] for x in ansr)) / len(ansr) if ansr else float('nan'), 'correct_silence': sum((x[1] for x in una)) / len(una) if una else float('nan'), 'false_silence': sum((x[1] for x in ansr)) / len(ansr) if ansr else float('nan'), 'found_where_rival_missed': sum((1 for x in ansr if x[2] and (not x[4]))), 'missed_where_rival_found': sum((1 for x in ansr if x[4] and (not x[2]))), 'paired_payoff': {'mind_better': w, 'rival_better': l, 'discordant': d, 'mcnemar_z': (w - l) / math.sqrt(d) if d else float('nan'), 'max_achievable_z': math.sqrt(d) if d else 0.0, 'underpowered': bool(math.sqrt(d) <= 1.645)}}
        mixed_block = {'rival_threshold': thr, 'held_out': block(ex['_mixed']), 'train_control': block(ctrl.get('_mixed') or [])}
        log(f'  MIXED {json.dumps(mixed_block)}')
    refuse_block = None
    he_sp, tr_sp = (ex.pop('_sparse', []), ctrl.pop('_sparse', []))
    if REFUSE and he_sp:
        pr_sp = []
        for q in probe_qs:
            if not q.get('sparse'):
                continue
            t = q['cands'][q['label']]
            rc, mx = lookup_rival_cos(probe, q, bank, device)
            pr_sp.append([int(q['answerable']), int(lookup_rival(q) == t), int(rc == t), counting_margin(q), mx])

        def thresh(rows, right_i, margin_i):
            """Largest set of questions, taken in DESCENDING confidence, whose accuracy still
            clears the derived break-even. Whole ties admitted together and the scan stops at
            the first group that breaks it - the same rule the D threshold uses, and for the
            same two reasons: accuracy decays slowly so a tail sneaks in one item at a time, and
            rows tied at one margin cannot be split anyway. Rows with no margin at all (the ink
            gave the query no context, so the retrieval rival cannot rank) are excluded from the
            calibration and refuse at scoring time."""
            g = defaultdict(list)
            for r in rows:
                if r[margin_i] == r[margin_i]:
                    g[r[margin_i]].append(r)
            t, cum, seen = (None, 0, 0)
            for m in sorted(g, reverse=True):
                grp = g[m]
                if (cum + sum((x[right_i] for x in grp))) / (seen + len(grp)) < 0.875:
                    break
                cum += sum((x[right_i] for x in grp))
                seen += len(grp)
                t = m
            return t

        def reward(rows, right_i, margin_i, t):
            tot = 0.0
            for r in rows:
                if t is None or not r[margin_i] >= t:
                    tot += 1.0 if not r[0] else 0.75
                else:
                    tot += 1.0 if r[right_i] else -1.0
            return tot / len(rows)

        def fixed_reward(rows, right_i, refused_i):
            """A rival that refuses by RULE rather than by threshold - scored exactly as the
            mind is, so the only difference between them is the judgment."""
            return sum((1.0 if r[right_i] else 0.75 if r[refused_i] else -1.0 for r in rows)) / len(rows)
        t_cnt = thresh(pr_sp, 1, 3) if pr_sp else None
        t_cos = thresh(pr_sp, 2, 4) if pr_sp else None
        ans = [r for r in he_sp if r[0]]
        una = [r for r in he_sp if not r[0]]
        blanket = (len(una) + 0.75 * len(ans)) / len(he_sp)
        mind_r = sum((1.0 if r[1] else 0.75 if r[2] else -1.0 for r in he_sp)) / len(he_sp)
        refuse_block = {'n': len(he_sp), 'n_answerable': len(ans), 'n_unanswerable': len(una), 'unanswerable_rate': len(una) / len(he_sp), 'mind': {'acc_answerable': sum((r[1] for r in ans)) / len(ans) if ans else float('nan'), 'refuse_recall': sum((r[2] for r in una)) / len(una) if una else float('nan'), 'false_refusal': sum((r[2] for r in ans)) / len(ans) if ans else float('nan'), 'coverage': 1.0 - sum((r[2] for r in he_sp)) / len(he_sp), 'reward': mind_r}, 'rival_counting': {'threshold_from_probe': t_cnt, 'reward': reward(he_sp, 3, 5, t_cnt)}, 'rival_retrieval': {'threshold_from_probe': t_cos, 'reward': reward(he_sp, 4, 6, t_cos)}, 'rival_own_row': {'reward': fixed_reward(he_sp, 8, 9), 'acc_answerable': sum((r[8] for r in ans)) / len(ans) if ans else float('nan'), 'refuse_recall': sum((r[9] for r in una)) / len(una) if una else float('nan')}, 'blanket_refusal_reward': blanket, 'always_answer_ceiling': (len(ans) - len(una)) / len(he_sp), 'gates': {'G_refuse_beats_blanket': bool(mind_r > blanket), 'G_refuse_beats_counting': bool(mind_r > reward(he_sp, 3, 5, t_cnt)), 'G_refuse_beats_retrieval': bool(mind_r > reward(he_sp, 4, 6, t_cos)), 'G_refuse_beats_own_row_shortcut': bool(mind_r > fixed_reward(he_sp, 8, 9))}}
        log(f'  REFUSE {json.dumps(refuse_block)}')

    def paraphrase_split(pk):
        """Does the SAME fact, written two different ways, land at the same address?

        The mind can only use an intersection that the tape actually formed. "Kostya was born in
        1985" and "1985 was a good year; there were no earthquakes, and Kostya was born" are one
        fact, and if extraction sends them to two unrelated addresses then no reader, however
        good, can bring them together - the redundancy that was supposed to be free evidence has
        been thrown away before the mind is asked anything.

        So paraphrase robustness is, in this architecture, an ADDRESSING property, not a reading
        one. Two label-free counts, both computed from the tape alone:

          same_anchor_diff_relation  the anchor survived, the relation did not: one fact spread
                                     across several addresses. Direct fragmentation.
          reversed_pairs             the pair appears in both directions - A|rel -> B somewhere
                                     and B|rel -> A elsewhere.

        The second one carries a warning about a claim made earlier in this session. A third of
        the mind's lookup errors were exact swap pairs (Leipzig<->Weimar, California<->Texas) and
        that was read as evidence of order-blind ink. If the TAPE already holds both directions,
        then both are true at different addresses and the "error" may be the examiner's rather
        than the mind's. Whichever way this number comes out, that earlier reading needs it
        before it can stand.
        """
        by_fact, by_pair, seen = (defaultdict(set), defaultdict(set), Counter())
        for it in pk['items']:
            addr = it['address']
            anchor = s289a.anchor_of(addr)
            for sl in it['slots']:
                val = pk['tape'].values[sl]
                by_fact[anchor, val].add(addr)
                seen[anchor, val] += 1
                by_pair[frozenset((anchor, val.lower()))].add((anchor, val.lower()))
        multi = [k for k in by_fact if seen[k] >= 2]
        rev = [p for p, ends in by_pair.items() if len(p) == 2 and len(ends) == 2]
        return {'facts_written_twice': len(multi), 'same_anchor_diff_relation': sum((1 for k in multi if len(by_fact[k]) > 1)) / len(multi) if multi else float('nan'), 'mean_addresses_per_fact': sum((len(by_fact[k]) for k in multi)) / len(multi) if multi else float('nan'), 'reversed_pairs': len(rev), 'reversed_pair_rate': len(rev) / max(1, len(by_pair))}

    def tape_shape(pk, qq):
        lk = [q for q in qq if q['verb'] == 'lookup']
        return {'slots': len(pk['texts']), 'addresses': len(pk['addr_slots']), 'mentions_per_address': len(pk['texts']) / len(pk['addr_slots']) if pk['addr_slots'] else float('nan'), 'lookup_questions': len(lk), 'mean_candidates': sum((len(q['cands']) for q in lk)) / len(lk) if lk else float('nan'), 'paraphrase': paraphrase_split(pk)}
    log(f"  HELD {json.dumps({k: v for k, v in ex.items() if k != 'lookup_item_hits'})}")
    log(f"  CONTROL {json.dumps({k: v for k, v in ctrl.items() if k != 'lookup_item_hits'})}")
    g_arc = arc0 == arc1
    g_task = ex.get('lookup', {}).get('n', 0) >= 2 * s286.MIN_ANSWERED
    if REACH:
        rh = (reach_block or {}).get('held_out') or {}
        g_task = rh.get('n', 0) >= 2 * s286.MIN_ANSWERED
    g_exact = bool(ex.get('exact_mismatches', 1) == 0 and ctrl.get('exact_mismatches', 1) == 0)
    lk = ex.get('lookup', {})
    pv = ex.get('lookup_paired_vs_rival', {})
    g_floor = bool(lk and lk['model_accuracy'] > lk['random_floor'])
    if REACH:
        rh = (reach_block or {}).get('held_out') or {}
        g_floor = bool(rh and rh['payoff_mind'] > rh['always_silent'])
    g_beats_counts = bool(pv.get('discordant', 0) >= 2 * s286.MIN_ANSWERED and (not math.isnan(pv.get('mcnemar_z', float('nan')))) and (pv['mcnemar_z'] > 1.645))
    pc = ex.get('lookup_paired_vs_rival_cos', {})
    g_beats_retrieval = bool(pc.get('discordant', 0) >= 2 * s286.MIN_ANSWERED and (not math.isnan(pc.get('mcnemar_z', float('nan')))) and (pc['mcnemar_z'] > 1.645))
    if REACH:
        _rh = (reach_block or {}).get('held_out') or {}
        _pp = _rh.get('paired_payoff', {})
        g_beats_retrieval = bool(_pp.get('discordant', 0) >= 8 and (not math.isnan(_pp.get('mcnemar_z', float('nan')))) and (_pp['mcnemar_z'] > 1.645))
        g_beats_counts = g_beats_retrieval
    ld = ex.get('ladder', {})
    g_ladder = bool(ld.get('pairs', 0) >= 6 * s286.MIN_ANSWERED and (not math.isnan(ld.get('z_vs_half', float('nan')))) and (ld['z_vs_half'] > 1.645))
    underpowered = bool(pc.get('underpowered', True))
    bym = ex.get('lookup_paired_vs_rival_cos_by_margin', {})
    lo, hi = (bym.get('low_margin', {}), bym.get('high_margin', {}))
    lo_z, hi_z = (lo.get('mcnemar_z', float('nan')), hi.get('mcnemar_z', float('nan')))
    split_effect = bool(not math.isnan(lo_z) and (not math.isnan(hi_z)) and (lo_z > 1.645) and (hi_z < -1.645))
    overall = 'NO_TASK' if not (g_task and g_arc and g_exact) else 'DERIVATION_OK' if g_floor and g_beats_counts and g_beats_retrieval and g_ladder else 'UNDERPOWERED_VS_RETRIEVAL' if g_floor and g_beats_counts and (not g_beats_retrieval) and underpowered else 'PHI_HELPS_WHERE_SIMILARITY_RUNS_OUT' if g_floor and g_beats_counts and (not g_beats_retrieval) and split_effect else 'PHI_ADDS_NOTHING_ON_LOOKUP' if g_floor and g_beats_counts and (not g_beats_retrieval) else 'DERIVATION_PARTIAL' if g_floor or g_ladder else 'DERIVATION_NO'
    out = {'stage': '289', 'overall': overall, 'seed': SEED, 'smoke': args.smoke, 'holdout': args.holdout, 'run_tag': args.run_tag, 'train_steps': n_steps, 'params': n_params, 'dim': args.dim, 'min_fillers': MIN_FILLERS, 'connect': bool(CONNECT), 'copy': bool(COPY), 'copy_d': COPY_D, 'copy_backfill': bool(COPY_BACKFILL), 'reach_channel': bool(REACH_CHANNEL), 'moves': bool(MOVES_ON), 'move_set': list(MOVES) if MOVES_ON else [], 'own_import': bool(OWN_IMPORT), 'own_in_offer': bool(OWN_IN_OFFER), 'own_import_full': _OWN_IMPORT_N[1] / _OWN_IMPORT_N[0] if _OWN_IMPORT_N[0] else None, 'objective': 'expected_reward_280' if OBJECTIVE == 'reward' else 'plackett_luce_ladder' if LADDER_ON else 'cross_entropy_no_ladder', 'edge_channels': sorted(EDGES_ON), 'import_k': IMPORT_K, 'views': VIEWS, 'reconciliation': recon, 'neighbours': NEIGHBOURS, 'open_verb': OPEN, 'patterns_verb': PATTERNS, 'address_from': ADDRESS_FROM, 'open_cands': OPEN_CANDS, 'anchor_max_rows': ANCHOR_MAX_ROWS if ADDRESS_FROM == 'anchor' else None, 'identity_verb': IDENTITY, 'mixed_verb': MIXED, 'mixed': mixed_block, 'reach_verb': REACH, 'reach': reach_block, 'constrain': CONSTRAIN, 'cons': cons_block, 'coherence': coh_block, 'shuffled_tape': SHUFFLE_TAPE, 'retain': RETAIN, 'retain_by': RETAIN_BY, 'reach_cols': list(REACH_COLS), 'pair_verb': PAIR, 'pair': pair_block, 'tape_cut': TAPE, 'route': ROUTE, 'step_cost': STEP_COST if ROUTE else None, 'frame_max': FRAME_MAX if TAPE == 'frames' else None, 'frame_pool': FRAME_POOL[0] if TAPE == 'frames' else None, 'tape_sample': TAPE_SAMPLE if TAPE == 'frames' else None, 'flat': bool(Deriver.FLAT), 'transplant': args.load_mind or None, 'corpus': str(wiki_path), 'identity': {'tau': IDENT_TAU, 'overlap': IDENT_OVERLAP, 'cands': IDENT_CANDS, 'core': IDENT_CORE, 'values': IDENT_VALUES, 'import': IDENT_IMPORT, 'supply': dict(IDENT_SUPPLY)} if IDENTITY else None, 'open_near_source': {'same_anchor': OPEN_NEAR[0], 'neighbourhood': OPEN_NEAR[1]} if OPEN else None, 'graph_rows': {'mean': GRAPH_N[0] / max(1, GRAPH_N[2]), 'max': GRAPH_N[1], 'graphs': GRAPH_N[2]}, 'neighbourhood_audit': neighbourhood_audit(pack, NEIGHBOURS) if NEIGHBOURS else None, 'nb_channels': {'anchor_nonzero_rate': NB_NNZ[0] / max(1, NB_NNZ[2]), 'rel_nonzero_rate': NB_NNZ[1] / max(1, NB_NNZ[2]), 'pairs': NB_NNZ[2]} if NEIGHBOURS else None, 'refuse': refuse_block, 'ink': INK, 'fp': FP, 'words': WORDS, 'write_ink': args.write_ink, 'write_fp': args.write_fp, 'write_words': args.write_words, 'fp_ngram': args.fp_ngram if FP == 'hash' else None, 'tau': {'mode': 'frames', 'value': None, 'target_density': None, 'achieved_density': None, 'monotone': None, 'trace': None} if TAPE == 'frames' else {'mode': args.tau_mode, 'value': args.address_tau if args.tau_mode == 'absolute' else tau_rule.memo.get('tau'), 'target_density': args.tau_target_density if args.tau_mode == 'density' else None, 'achieved_density': tau_rule.memo.get('achieved') if args.tau_mode == 'density' else None, 'monotone': tau_rule.memo.get('monotone') if args.tau_mode == 'density' else None, 'trace': tau_rule.memo.get('trace') if args.tau_mode == 'density' else None}, 'tape_shape': {'held_out': tape_shape(held, held_qs), 'train': tape_shape(pack, qs)}, 'resample': {'tape_period': args.tape_period, 'mean_overlap': sum(RESAMPLE_OVERLAP) / len(RESAMPLE_OVERLAP) if RESAMPLE_OVERLAP else float('nan'), 'n_resamples': len(RESAMPLE_OVERLAP), 'note': "Jaccard between consecutive tapes' address sets. Near 1 means the redraw returns the same addresses and the anti-memorisation argument in HANDOFF 1 is decorative - the fix is a larger address pool, i.e. more corpus, not fewer parameters"}, 'row_dropout': {'rate': args.row_dropout, 'mean_kept_fraction': DROPPED[0] / DROPPED[1] if DROPPED[1] else float('nan'), 'note': 'training only - the held-out tape is never thinned. Marginalisation, not noise: a subset of the evidence is a world the corpus could have written, and the low/high margin split is a density axis the mind was never trained across'}, 'early_stop': {'enabled': not args.no_early_stop, 'best_step': best['step'], 'best_probe_loss': best['loss'], 'total_steps': n_steps, 'probe_questions': len(probe_qs)}, 'probe_curve': probe_curve, 'rare_nonzero_rate': RARE_NNZ[0] / RARE_NNZ[1] if RARE_NNZ[1] else float('nan'), 'ink_degenerate_rate': INK_DEGENERATE[0] / INK_DEGENERATE[1] if INK_DEGENERATE[1] else float('nan'), 'cos_mean': COS_SPREAD[0] / COS_SPREAD[2] if COS_SPREAD[2] else float('nan'), 'cos_std': math.sqrt(max(0.0, COS_SPREAD[1] / COS_SPREAD[2] - (COS_SPREAD[0] / COS_SPREAD[2]) ** 2)) if COS_SPREAD[2] else float('nan'), 'ladder_coverage_train': {'with_ladder': n_lad, 'lookup_questions': len(bv.get('lookup', ()))}, 'count_labels': list(COUNT_LABELS), 'compare_labels': list(COMPARE_LABELS), 'gates': {'G_arc_enc_frozen': g_arc, 'G_ink_verified': g_ink, 'G_task_exists': g_task, 'G_exact_algebra_matches_tape': g_exact, 'G_lookup_beats_floor': g_floor, 'G_lookup_beats_counts_paired': g_beats_counts, 'G_lookup_beats_retrieval_paired': g_beats_retrieval, 'G_phi_orders_negatives': g_ladder}, 'held_out': ex, 'train_control': ctrl, 'exact_note': 'count and compare left the weights: they are functions of the same-value relation alone (new_i = 1 - max_{j<i} s_ij; count = sum new_i; compare = sign of the side difference), computed exactly with zero parameters and no 5+ cap, because the invariant says whatever decides may not be approximate. Their accuracy is 1.0 by construction and is checked, not celebrated - G_exact_algebra_matches_tape is a sanity bolt. The interference that cost count 0.965 -> 0.903 is removed by construction: one trained task remains', 'ladder_note': 'three wrong answers per question at increasing structural distance - same anchor / adjacent in tape order / anywhere on the tape - every rung a value the corpus wrote, no similarity chosen by anyone. Phi trained only against local wrong candidates learns a BOUNDARY; generation needs a LANDSCAPE, and a mind that cannot rank its own wrong answers by how wrong they are has no direction to move in. The objective is one Plackett-Luce term, not a task loss plus a ladder loss with a weight between them, and it reduces to the previous cross-entropy exactly when the tape cannot supply a ladder', 'retrieval_note': 'two rivals now, because they answer two different questions. The counting rival knows nothing about context, so beating it shows only that the context channel carries information counts lack - not that reading it takes a mind. rival_cos is 1-NN over the same evidence rows by the same ctx_fp cosine, zero parameters, no training. With hash ink the representation IS Random Indexing over fastText-shaped word vectors, so the distance between this architecture and a classical retrieval system is exactly this one number. If rival_cos lands where Phi lands, 3489 parameters are decoration ON THIS VERB and the verdict is PHI_ADDS_NOTHING_ON_LOOKUP. Named for the brick and not for the wall: lookup is one verb, single-hop and retrieval-shaped by construction, and a rival that ties it says nothing about the exact algebra, about verbs where rows must be combined, or about generation, which 1-NN cannot do at all', 'paired_note': 'the rival answers the same lookup questions in the same run, so the gate is McNemar over the discordant items at the usual one-sided 1.645 - never two marginals. The rival over survivors is Bayes-optimal when the query context carries nothing, so a paired win IS the claim that the context channel carries information counts do not have', 'curve': curve, 'arc_enc_hash_before': arc0, 'arc_enc_hash_after': arc1, 'fp_version': s271.fp_version(), 'note': "The derivation moved into exact algebra and the mind kept only the judgment. Two runs measured 7.9k parameters approximating a quantity exactly computable from their own input, and the approximation degraded as the genuinely uncertain task grew beside it. Now count and compare are arithmetic over the same-value relation - exact, uncapped, scale-free - and the one trained surface is Phi, the coherence of a completed world: for each candidate the query row is filled in and the world that results is pooled to one scalar, 288's repair loop turned inward. The two trained surfaces this leaves in the whole architecture are Phi and, once values stop being exact strings, s_ij itself - both judgments, never arithmetic. Confidence for exact verbs reports 1.0, which is the honest statement that a computed answer is certain GIVEN the relation; when s_ij becomes a judgment its uncertainty enters through that same seam.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(out, indent=2)
    (RES / f'stage289_decision{tag}.json').write_text(payload, encoding='utf-8')
    if args.out:
        op = Path(args.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(payload, encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates'], 'lookup': {k: v for k, v in lk.items()}, 'paired': pv, 'ladder': ld}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())