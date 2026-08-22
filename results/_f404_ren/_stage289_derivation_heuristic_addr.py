"""
SNAPSHOT / KEEP — heuristic addressing line (pre «одно / не одно»).

Frozen copy of `_stage289_derivation.py` as of the night when 290 neighbourhood died,
292 open-vs-RAG held across seeds, and the next step was agreed: Φ decides whether places
are the same, and the tau+cosine+word address heuristic stops building the tape.

Why keep this file:
  - Tape addresses are still built by the heuristic (single-link absolute cosine, word
    overlap, tau bisection). That construction may matter again for a future design even
    after a no-heuristic variant lands.
  - Open/import search, sparse neighbours, refuse, ladder, and seed --seed all live here.
  - Do not edit this snapshot into the new question; branch a new file when the
    same-or-not / mind-as-addressing variant arrives.

Active work continues in `_stage289_derivation.py` (or its successor). This file is the
reference for the heuristic-address construction.

---

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
from _tape_speed import INK_DEGENERATE, WORD_RULES, BigramBank, CachedBank, HashFp, install_assertion_cache, install_fast_fp_addresses, verify_hash_ink, verify_word_rule
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
v0 = v40('results')
v1 = v40('checkpoints/stage191_p1_curve.pt')
v2 = v40('data/_wikitext103_train.txt')
v3 = 2890
v4 = 5
v5 = v231((v8(v132) for v132 in v276(1, v4))) + (f'{v4}+',)
v6 = ('first', 'second', 'equal')
v7 = v0 / '_stage289_log.txt'

def log(v41: v8) -> None:
    v42 = v41 if v41.v410('\n') else v41 + '\n'
    try:
        v411(v42, end='', flush=True)
    except v232:
        v411(v42.v648('ascii', 'replace').v610('ascii'), end='', flush=True)
    v7.v412.v233(parents=True, exist_ok=True)
    with v7.v336('a', encoding='utf-8') as v234:
        v234.v413(v42)

def count_label(v43: v10) -> v8:
    return v5[v259(v43, v4) - 1]

def soft_new(v44, v45=None):
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
    v45 = v45 or (lambda v50, v51: 1.0 if v50 == v51 else 0.0)
    return [1.0 - (v311((v45(v44[v300], v44[v132]) for v300 in v276(v132))) if v132 else 0.0) for v132 in v276(v267(v44))]

def soft_count(v44, v45=None) -> v9:
    return v235(v414(v44, v45))

def exact_new(v44, v46=None):
    """The hard special case, kept because integers are what the examiner compares."""
    v45 = None if v46 is None else lambda v50, v51: 1.0 if v46(v50, v51) else 0.0
    return [v10(v428(v415)) for v415 in v414(v44, v45)]

def exact_count(v44, v46=None) -> v10:
    return v235(v416(v44, v46))

def exact_answer(v47):
    """The exact verdict for the exact verbs; raises on lookup, which is judged, not computed."""
    if v47['verb'] == 'count':
        return v417(v418(v47['vals']))
    if v47['verb'] == 'compare':
        v236 = v418(v47['vals'][:v47['n_first']])
        v237 = v418(v47['vals'][v47['n_first']:])
        return 'first' if v236 > v237 else 'second' if v237 > v236 else 'equal'
    raise v238('lookup is not exact: it is the judgment the mind is for')
v11 = ('lookup',)

def count_question(v48, v49):
    """How many distinct values does this address carry? The truth is a property of the tape."""
    v44 = [v48['tape'].v130[v118] for v118 in v49['slots']]
    if v267(v44) < 2:
        return None
    return {'verb': 'count', 'slots': v239(v49['slots']), 'vals': v44, 'label': v417(v267(v102(v44))), 'S': v49['S'], 'address': v49['address']}

def compare_question(v48, v50, v51):
    """Which of two addresses carries more distinct values?

    Both address's mentions go into ONE graph, with a side indicator per row. Nothing tells the
    mind how many rows each side has beyond what it can see, and the answer is not a count but
    an ordering, so the two verbs cannot share a shortcut: a mind that memorised "this many rows
    means this label" for COUNT gets nothing here, where both sides sit in the same graph.
    """
    v52 = [v48['tape'].v130[v118] for v118 in v50['slots']]
    v53 = [v48['tape'].v130[v118] for v118 in v51['slots']]
    if v267(v52) < 2 or v267(v53) < 2:
        return None
    v236, v237 = (v267(v102(v52)), v267(v102(v53)))
    v54 = 'first' if v236 > v237 else 'second' if v237 > v236 else 'equal'
    return {'verb': 'compare', 'slots': v239(v50['slots']) + v239(v51['slots']), 'vals': v52 + v53, 'n_first': v267(v52), 'label': v54, 'S': v50['S'], 'S2': v51['S'], 'address': v50['address'], 'address2': v51['address']}

def lookup_question(v48, v49, v55, v56=None):
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
    v57 = v239(v49['slots'])
    if v267(v57) < 3:
        return None
    v56 = v55.v419(v267(v57)) if v56 is None else v56
    v44 = [v48['tape'].v130[v118] for v118 in v57]
    v58 = v240(v102(v44[:v56] + v44[v56 + 1:]))
    if v267(v58) < 2:
        return None
    if v44[v56] not in v58:
        return None
    v59 = v57[:v56] + v57[v56 + 1:]
    return {'verb': 'lookup', 'slots': v59 + [v57[v56]], 'vals': [v44[v132] for v132 in v276(v267(v57)) if v132 != v56] + [v611()], 'cands': v58, 'label': v58.v420(v44[v56]), 'S': v49['S'], 'address': v49['address'], 'hid': v56, 'query_row': v267(v59)}
v12 = 0
v13 = False
v14 = True
v15 = '\x00REFUSE'

def addr_parts(v60):
    """(anchor, relation) of an fp address, split exactly as pack_from_corpus splits it."""
    v61 = v60.v421(':', 1)[-1]
    v50, v125 = (v61.v421('|', 1) + [''])[:2]
    return (v50, v125)

def neighbourhood(v62, v63, v43, v64=('anchor', 'rel', 'word')):
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
    v65 = v62.v241('_nb', {})
    if (v63, v43, v64) in v65:
        return v65[v63, v43, v64]
    v66 = v62.v242('_addr_index')
    if v66 is None:
        v77, v422, v423, v304 = (v291(v239), v291(v239), v291(v239), {})
        v131 = v62.v242('_median')
        if v131 is None:
            v293 = v240((v267(v92) for v92 in v62['postings'].v130()))
            v131 = v293[v267(v293) // 2] if v293 else 1
            v62['_median'] = v131
        for v132, v113 in v424(v62['items']):
            v304[v113['address']] = v132
            v243, v244 = v245(v113['address'])
            v77[v243].v429(v113['address'])
            if v244:
                v422[v244].v429(v113['address'])
            v246 = v102()
            for v69 in v113['slots']:
                for v248 in v426(v62['texts'][v69], exclude=v62['tape'].v130[v69]):
                    if v267(v62['postings'].v242(v248, ())) < v131 and v248 not in v246:
                        v246.v559(v248)
                        v423[v248].v429(v113['address'])
        v66 = v62['_addr_index'] = {'anchor': v77, 'rel': v422, 'word': v423, 'order': v304, 'slots': {v113['address']: v113['slots'] for v113 in v62['items']}}
    v243, v244 = v245(v63)
    v70, v246 = ([], {v63})
    v67 = v66['order'].v242(v63, 0)

    def take(v58, v91):
        for v85 in v58[:v91]:
            if v85 not in v246:
                v246.v559(v85)
                v70.v429(v85)

    def near(v58):
        return v240((v50 for v50 in v58 if v50 != v63), key=lambda v50: (v661(v66['order'][v50] - v67), v66['order'][v50]))
    if 'anchor' in v64:
        v425(v552(v66['anchor'].v242(v243, ())), v43)
    if 'rel' in v64:
        v425(v552(v66['rel'].v242(v244, ())), v43)
    v68 = v247()
    for v69 in v66['slots'].v242(v63, ()):
        for v248 in v426(v62['texts'][v69], exclude=v62['tape'].v130[v69]):
            if v267(v62['postings'].v242(v248, ())) < v62['_median']:
                for v50 in v66['word'].v242(v248, ()):
                    if v50 != v63:
                        v68[v50] += 1
    if 'word' in v64:
        v425([v50 for v50, v115 in v240(v68.v127(), key=lambda v669: (-v669[1], v66['order'][v669[0]]))], v43)
    v65[v63, v43, v64] = v70
    return v70

def lookup_sparse_question(v62, v49, v55, v56, v43, v64=('anchor', 'rel', 'word')):
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
    v71 = v239(v49['slots'])
    if not 1 <= v267(v71) <= 2:
        return None
    v56 = v56 % v267(v71)
    v72 = v71[v56]
    v73 = v62['tape'].v130[v72]
    v74 = [v118 for v118 in v71 if v118 != v72]
    for v51 in v249(v62, v49['address'], v43, v64):
        v74 += v239(v62['_addr_index']['slots'].v242(v51, ()))[:v43]
    v74 = v240(v102(v74) - {v72})
    if not v74:
        return None
    v58 = v240({v62['tape'].v130[v118] for v118 in v74})
    if v267(v58) < 2:
        return None
    v75 = v73 in v58
    if not v75 and (not v13):
        return None
    if v13:
        v58 = v58 + [v15]
    v76 = v58.v420(v73) if v75 else v58.v420(v15) if v13 else None
    if v76 is None:
        return None
    return {'verb': 'lookup', 'sparse': True, 'answerable': v75, 'slots': v74 + [v72], 'vals': [v62['tape'].v130[v118] for v118 in v74] + [v611()], 'cands': v58, 'label': v76, 'S': v49['S'], 'address': v49['address'], 'hid': v56, 'own_rows': {v118 for v118 in v71 if v118 != v72}, 'query_row': v267(v74)}
v16 = False
v17 = 'ce'
v18 = [0, 0]

def lookup_open_question(v62, v49, v55, v56, v77, v78):
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
    v71 = v239(v49['slots'])
    if v267(v71) < 2:
        return None
    v56 = v56 % v267(v71)
    v72 = v71[v56]
    v73 = v62['tape'].v130[v72]
    v74 = [v118 for v118 in v71 if v118 != v72]
    if v250((v62['tape'].v130[v118] == v73 for v118 in v74)):
        return None
    v79 = {'cands': [v73], 'address': v49['address'], 'slots': v71, 'S': v49['S'], 'query_row': v267(v74)}
    v251(v62, v79, v77, v78, v55)
    v80 = v178(v79.v242('ladder') or {})
    if 'near' not in v80:
        v110 = {v73} | v102(v80.v130())
        for v51 in v249(v62, v49['address'], 3):
            v275 = [v62['tape'].v130[v69] for v69 in v62['_addr_index']['slots'].v242(v51, ())]
            v275 = [v415 for v415 in v275 if v415 not in v110]
            if v275:
                v80['near'] = v275[0]
                break
    if v267(v80) != 3:
        return None
    v18[0 if v79.v242('ladder', {}).v242('near') == v80['near'] else 1] += 1
    v58 = v240([v73] + [v80[v125] for v125 in v23])
    if v267(v58) != 4:
        return None
    v47 = {'verb': 'lookup', 'open': True, 'slots': v74 + [v72], 'vals': [v62['tape'].v130[v118] for v118 in v74] + [v611()], 'cands': v58, 'label': v58.v420(v73), 'rung_of': {v80[v125]: v125 for v125 in v23}, 'S': v49['S'], 'address': v49['address'], 'hid': v56, 'query_row': v267(v74)}
    if v253(v62, v47, v239(v47['cands'])) < 1:
        return None
    return v47

def open_rival_cos(v62, v47, v81, v82):
    """The rival 292 actually has to beat: retrieval over the WHOLE TAPE, not over the address.

    Once every candidate brings its own mentions in, similarity is back in the game - it just
    searches the corpus instead of the address. That is RAG, stated exactly, and it is the fork
    the project has been circling: if nearest-imported-context lands where Phi lands, what we
    built is a search engine with extra steps.

    Same rows Phi is given - the shared import budget - and one rule: the candidate with a
    mention whose context is nearest the query's.
    """
    v83 = v62.v241('_ctx', {})

    def ctx(v69):
        if v69 not in v83:
            v85 = v81.v553(v62['texts'][v69], exclude=v62['tape'].v130[v69])
            v83[v69] = v569.v612(v85, dim=-1) if v85 is not None else None
        return v83[v69]
    v84 = v252(v47['slots'][v47['query_row']])
    if v84 is None:
        return None
    v43 = v253(v62, v47, v239(v47['cands']))
    v74, v254 = ([], [])
    for v85 in v47['cands']:
        for v69 in v554(v62, v47, v85)[:v43]:
            v125 = v252(v69)
            if v125 is not None:
                v74.v429(v125)
                v254.v429(v85)
    if not v74:
        return None
    return v254[v10((v320.v266(v74, 0) @ v84).v443())]

def neighbourhood_audit(v62, v43, v86=(1, 3, 6, 12)):
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
    v70 = {}
    for v87 in v240({v43} | v102(v86)):
        v70[f'k={v87}'] = v427(v62, v87)
    return v70

def _audit_at(v62, v43):
    v70 = {}
    for v255, v64 in (('anchor', ('anchor',)), ('rel', ('rel',)), ('word', ('word',)), ('anchor+word', ('anchor', 'word')), ('all', ('anchor', 'rel', 'word'))):
        v91 = v256 = v74 = 0
        for v113 in v62['items']:
            if not 1 <= v267(v113['slots']) <= 2:
                continue
            for v56 in v276(v267(v113['slots'])):
                v71 = v239(v113['slots'])
                v555 = v71[v56 % v267(v71)]
                v73 = v62['tape'].v130[v555]
                v470 = [v415 for v415 in v71 if v415 != v555]
                for v51 in v249(v62, v113['address'], v43, v64):
                    v470 += v239(v62['_addr_index']['slots'].v242(v51, ()))[:v43]
                v470 = v240(v102(v470) - {v555})
                if v267(v470) < 1 or v267({v62['tape'].v130[v415] for v415 in v470}) < 2:
                    continue
                v91 += 1
                v74 += v267(v470)
                v256 += v10(v73 in {v62['tape'].v130[v415] for v415 in v470})
        v70[v255] = {'questions': v91, 'answerable': v256, 'hit_rate': v256 / v91 if v91 else v9('nan'), 'mean_rows': v74 / v91 if v91 else v9('nan')}
    return v70
v19 = [0, 0]
v20 = 1
v21 = 'thin'

def view_of(v47, v55, v88):
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
    v89 = v47['query_row']
    v59 = [v132 for v132 in v276(v89) if v55.v482() < v88]
    v90 = {v47['vals'][v132] for v132 in v59}
    for v85 in v47['cands']:
        v257 = [v132 for v132 in v276(v89) if v47['vals'][v132] == v85]
        if v85 not in v90 and v257:
            v59.v429(v55.v556(v257))
    v59 = v240(v102(v59))
    v70 = {**v47, 'slots': [v47['slots'][v132] for v132 in v59] + [v47['slots'][v89]], 'vals': [v47['vals'][v132] for v132 in v59] + [v47['vals'][v89]], 'query_row': v267(v59)}
    v70.v258('ladder', None)
    v70.v258('_base', None)
    return v70

def region_views_of(v47, v91):
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
    v89 = v47['query_row']
    v92 = v259(v91, v89)
    v93 = [v428(v132 * v89 / v92) for v132 in v276(v92 + 1)]
    v70 = []
    for v50, v51 in v260(v93, v93[1:]):
        v261 = {**v47, 'slots': v47['slots'][v50:v51] + [v47['slots'][v89]], 'vals': v47['vals'][v50:v51] + [v47['vals'][v89]], 'query_row': v51 - v50}
        v261.v258('ladder', None)
        v261.v258('_base', None)
        v70.v429(v261)
    return v70

def views_and_mask(v47, v55, v82):
    """The question's views plus the candidate-presence mask, under either mode.

    View 0 is always the FULL question, so at VIEWS=1 this is exactly the old single pass and
    the ensemble is a strict superset of the information the single pass had. The mask says
    which candidates each region actually has a witness for; None in thin mode, where view_of
    guarantees every candidate a witness and no masking is needed (or possible - that guarantee
    is what keeps thin views label-tight).
    """
    if v21 == 'thin':
        return ([v47] + [v637(v47, v55, 1.0 - v22) for v115 in v276(v20 - 1)], None)
    v94 = [v47] + v430(v47, v20)
    v95 = v320.v262([[v9(v85 in v102(v92['vals'][:v92['query_row']])) for v85 in v47['cands']] for v92 in v94[1:]], device=v82)
    return (v94, v95)

def pool_views(v96, v95):
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
    if v95 is None:
        return v96.v431(0)
    v97 = (v96[1:] * v95).v235(1) / v95.v235(1)
    return v96[0] - v96[0].v431() + ((v96[1:] - v97.v570(1)) * v95).v235(0)

def disagreement(v96, v98=None):
    """Generalised Jensen-Shannon divergence of the per-view answer distributions: the mean KL
    of each view to their mixture. Zero when the views agree exactly; label-free by
    construction, so thresholding on it is never conditioning on the outcome.

    With a mask (region mode), each view's distribution lives on ITS candidates - masked
    softmax puts exact zeros elsewhere, and the JS stays finite because 0*log(0/m) = 0 and the
    mixture covers every candidate some view supports. Two regions that put their mass on
    values the other never wrote disagree maximally, which is correct: that address is
    contested across the corpus, and D is the number that says so."""
    if v98 is not None:
        v96 = v96.v432(v98 == 0, v9('-inf'))
    v99 = v320.v263(v96, dim=1)
    v41 = v99.v431(0).v264(1e-09)
    return v9((v99 * (v99.v264(1e-09).v105() - v41.v105())).v235(1).v431())

def reconciled(v100, v62, v47, v82, v81, v55):
    """Pooled logits, the single full-pass logits, and D, for one question. Training takes the
    gradient through the pooled logits; the exam reads all three. In thin mode D is over all
    views (view 0 included, as recon3 measured it); in region mode D is over the REGIONS only -
    the full view is their union and would only dilute the cross-region signal."""
    v94, v95 = v265(v47, v55, v82)
    v96 = v320.v266([v317(v100, v62, v368, v82, v81) for v368 in v94])
    return (v433(v96, v95), v96[0], v434(v96 if v95 is None else v96[1:], v95))
v22 = 0.0

def drop_rows(v47, v55, v88):
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
    v89 = v47['query_row']
    v73 = v47['cands'][v47['label']]
    v101 = v239(v276(v89))
    if not v250((v47['vals'][v132] == v73 for v132 in v101)):
        return None
    v59 = [v132 for v132 in v101 if v55.v482() < v88]
    if not v250((v47['vals'][v132] == v73 for v132 in v59)):
        v59.v429(v55.v556([v132 for v132 in v101 if v47['vals'][v132] == v73]))
    v59 = v240(v102(v59))
    v58 = v240({v47['vals'][v132] for v132 in v59})
    if v15 in v47['cands']:
        v58 = v58 + [v15]
    if v267(v58) < 2:
        return None
    v19[0] += v267(v59)
    v19[1] += v267(v101)
    v70 = {**v47, 'slots': [v47['slots'][v132] for v132 in v59] + [v47['slots'][v89]], 'vals': [v47['vals'][v132] for v132 in v59] + [v47['vals'][v89]], 'cands': v58, 'label': v58.v420(v73), 'query_row': v267(v59)}
    v70.v258('ladder', None)
    v70.v258('_base', None)
    return v70
v23 = ('near', 'middle', 'far')
v24 = True
v25 = 2
v26 = ('same', 'cos', 'rare')
v27 = ('anchor', 'rel')
v28 = v102(v26)
v29 = 'mean'
v30 = 'arc'
v31 = 'ascii'
v32 = 2.9701

def tau_for_density(v103, v104, v105, v106=0.0, v107=0.9995):
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
    v108 = {}

    def resolve(v268, v81, v269, v270, v271):
        if 'tau' in v108:
            return v108['tau']

        def density(v139):
            v70, v557 = v353.v558(v268, v81, v139, v269, v270, addr_key=v271)
            return (v267(v70) / v267(v557) if v557 else v9('nan'), v267(v557))
        v165 = v342.v342()
        v435, v436 = v437(v106)
        v438, v439 = v437(v107)
        v272 = [(v106, v435, v436), (v107, v438, v439)]
        v273 = v435 > v438
        v50, v51 = (v106, v107)
        if not v273:
            v105(f'  tau calibration: density NOT decreasing in tau ({v106}->{v435:.3f}, {v107}->{v438:.3f}) - bisection is not valid here')
        elif not v438 <= v103 <= v435:
            v105(f'  tau calibration: target {v103:.4f} outside the bracket [{v438:.3f}, {v435:.3f}] - clamping to the nearer end')
        else:
            for v115 in v276(v104):
                v41 = 0.5 * (v50 + v51)
                v638, v639 = v437(v41)
                v272.v429((v41, v638, v639))
                if v638 > v103:
                    v50 = v41
                else:
                    v51 = v41
        v120 = v259(v272, key=lambda v125: v661(v125[1] - v103) if v125[1] == v125[1] else v9('inf'))
        v108['tau'] = v120[0]
        v108['trace'] = [{'tau': v428(v139, 5), 'density': v279, 'addresses': v91} for v139, v279, v91 in v272]
        v108['achieved'] = v120[1]
        v108['monotone'] = v273
        v105(f'  tau calibrated: {v120[0]:.4f} -> density {v120[1]:.4f} (target {v103:.4f}, {v120[2]} addresses, {v267(v272)} probes, {v342.v342() - v165:.0f}s)')
        return v120[0]
    v109.v108 = v108
    return v109
v33 = [0, 0]
v34 = True
v35 = []
v36 = [0, 0, 0]
v37 = [0, 0, 0]
v38 = [0.0, 0.0, 0]

def attach_ladder(v48, v47, v77, v78, v55):
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
    v110 = v102(v47['cands'])
    v80 = {}
    v111 = v440.v274(v47['address'])
    v112 = [v113 for v113 in v77.v242(v111, ()) if v113['address'] != v47['address']]
    for v113 in v55.v441(v112, v267(v112)) if v112 else ():
        v275 = [v48['tape'].v130[v69] for v69 in v113['slots']]
        v275 = [v92 for v92 in v275 if v92 not in v110]
        if v275:
            v80['near'] = v275[0]
            v110.v559(v275[0])
            break
    v114 = v311(v47['slots']) + 1
    for v69 in (v114, v259(v47['slots']) - 1):
        if 0 <= v69 < v48['n_slots'] and v48['tape'].v130[v69] not in v110:
            v80['middle'] = v48['tape'].v130[v69]
            v110.v559(v80['middle'])
            break
    for v115 in v276(8):
        v92 = v78[v55.v419(v267(v78))]
        if v92 not in v110:
            v80['far'] = v92
            break
    v47['ladder'] = v80 if v24 and v267(v80) == 3 else {}
    return v47

def lookup_rival(v47):
    """286's majority rival - over the SURVIVORS only.

    The query row now sits in vals carrying a sentinel that equals nothing. Counting it would
    let the sentinel win any all-distinct address and hand the rival a guaranteed miss, which
    would flatter the mind against an opponent crippled by our own bookkeeping.
    """
    v101 = [v92 for v132, v92 in v424(v47['vals']) if v132 != v47['query_row']]
    return v247(v101).v277(1)[0][0]

def own_row_rival(v47):
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
    v71 = v47.v242('own_rows') or v102()
    v116 = [v47['vals'][v132] for v132 in v276(v47['query_row']) if v47['slots'][v132] in v71]
    if v116:
        return v247(v116).v277(1)[0][0]
    return v15 if v13 else v442(v47)

def counting_margin(v47):
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
    v101 = [v92 for v132, v92 in v424(v47['vals']) if v132 != v47['query_row']]
    v85 = v247(v101).v277(2)
    return (v85[0][1] - (v85[1][1] if v267(v85) > 1 else 0)) / v311(1, v267(v101))

def lookup_rival_cos(v62, v47, v81, v82):
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
    v83 = v62.v241('_ctx', {})

    def ctx(v69):
        if v69 not in v83:
            v85 = v81.v553(v62['texts'][v69], exclude=v62['tape'].v130[v69])
            v83[v69] = v569.v612(v85, dim=-1) if v85 is not None else None
        return v83[v69]
    v117 = v47['query_row']
    v84 = v252(v47['slots'][v117])
    if v84 is None:
        return (None, v9('nan'))
    v74 = [(v132, v252(v69)) for v132, v69 in v424(v47['slots']) if v132 != v117]
    v74 = [(v132, v85) for v132, v85 in v74 if v85 is not None]
    if not v74:
        return (None, v9('nan'))
    v95 = v320.v266([v85 for v115, v85 in v74], 0)
    v118 = v95 @ v84
    v119 = v10(v118.v443())
    v120 = v47['vals'][v74[v119][0]]
    v121 = [v9(v118[v300]) for v300, (v132, v115) in v424(v74) if v47['vals'][v132] != v120]
    v122 = v9(v118[v119]) - (v311(v121) if v121 else -1.0)
    return (v120, v122)

class Deriver(v123.v39):
    """One body, one scalar. The mind describes a world; the algebra does the arithmetic.

    The body is 286/289a's relational net verbatim: edges carry the same-value indicator and
    two context ranks, nodes carry shares and indicators, identity has nowhere to live. Phi
    pools the whole graph to one number - how well this world hangs together - and that is the
    only trained readout left. The count and compare heads are gone because their tasks moved
    into exact algebra where the invariant says they belong; the interference they caused
    (count 0.965 -> 0.903 as lookup grew) is removed by construction, not compensated.
    """
    v124 = True

    def __init__(v278, v82, v279: v10=32, v280: v10=3, v281: v10=8, v282: v10=0):
        v613().v444()
        v43 = 3 if v278.v124 else 2
        v278.v283 = v123.v614(v123.v640(v280, v279), v123.v641()).v315(v82)
        v278.v284 = v123.v614(v123.v640(v281 + v43 * v279, v279), v123.v641()).v315(v82)
        if v282:
            with v320.v159():
                v278.v283[0].v446[:, v280 - v282:] = 0.0
        v278.v285 = v123.v614(v123.v640((2 if v278.v124 else 1) * v279, v279), v123.v641(), v123.v640(v279, 1)).v315(v82)
        v123.v560.v445(v278.v285[-1].v446)
        v123.v560.v445(v278.v285[-1].v447)

    def body(v278, v140, v46, v142):
        v286 = v278.v283(v140)
        v71 = (v286 * v46).v235(1) / v46.v235(1).v561(min=1.0)
        v287 = [v142, v71, v286.v431(1)]
        if v278.v124:
            v287.v429(v286.v311(1).v130)
        return v278.v284(v320.v319(v287, -1))

    def phi(v278, v140, v46, v142):
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
        v288 = v278.v448(v140, v46, v142)
        v289 = v320.v319([v288.v431(0), v288.v311(0).v130], -1) if v278.v124 else v288.v431(0)
        return v278.v285(v289).v449(-1)

def sparse_questions_for(v62, v125):
    """290's question set: the addresses the dense verb throws away.

    count and compare still come from the dense items - they are exact algebra, they cost
    nothing, and dropping them would remove the sanity bolt that fires if the tape and the
    arithmetic disagree.
    """
    v70 = []
    for v113 in v62['items']:
        if not 1 <= v267(v113['slots']) <= 2:
            continue
        for v56 in v276(v267(v113['slots'])):
            if (v47 := v642(v62, v113, v125, v56, v12)) is not None:
                v70.v429(v47)
    for v113 in v62['items']:
        if v267(v113['slots']) >= 2 and (v47 := v615(v62, v113)) is not None:
            v70.v429(v47)
    v126 = [v113 for v113 in v62['items'] if v267(v113['slots']) >= 2]
    v125.v290(v126)
    for v50, v51 in v260(v126[::2], v126[1::2]):
        if (v47 := v452(v62, v50, v51)) is not None:
            v70.v429(v47)
    return v70

def open_questions_for(v62, v125):
    """292's set. count and compare stay: they are exact algebra and they are the sanity bolt."""
    v127 = [v113 for v113 in v62['items'] if v267(v113['slots']) >= 2]
    v77 = v291(v239)
    for v113 in v127:
        v77[v440.v274(v113['address'])].v429(v113)
    v78 = v239(v62['tape'].v130)
    v70 = []
    for v113 in v127:
        for v56 in v276(v267(v113['slots'])):
            if (v47 := v643(v62, v113, v125, v56, v77, v78)) is not None:
                v70.v429(v47)
        if (v47 := v615(v62, v113)) is not None:
            v70.v429(v47)
    v126 = v239(v127)
    v125.v290(v126)
    for v50, v51 in v260(v126[::2], v126[1::2]):
        if (v47 := v452(v62, v50, v51)) is not None:
            v70.v429(v47)
    return v70

def questions_for(v62, v125):
    """Every question the tape can supply, of all three verbs."""
    if v16:
        return v450(v62, v125)
    if v12:
        return v451(v62, v125)
    v127 = [v113 for v113 in v62['items'] if v267(v113['slots']) >= 2]
    v77 = v291(v239)
    for v113 in v127:
        v77[v440.v274(v113['address'])].v429(v113)
    v78 = v239(v62['tape'].v130)
    v70 = []
    for v113 in v127:
        if (v47 := v615(v62, v113)) is not None:
            v70.v429(v47)
        for v56 in v276(v267(v113['slots'])):
            if (v47 := v644(v62, v113, v125, hid=v56)) is not None:
                v70.v429(v251(v62, v47, v77, v78, v125))
    v126 = v239(v127)
    v125.v290(v126)
    for v50, v51 in v260(v126[::2], v126[1::2]):
        v47 = v452(v62, v50, v51)
        if v47 is not None:
            v70.v429(v47)
    return v70

def n_choices(v47) -> v10:
    return v267(v47['cands']) if v47['verb'] == 'lookup' else v267(v5) if v47['verb'] == 'count' else v267(v6)

def truth_of(v47):
    return v47['cands'][v47['label']] if v47['verb'] == 'lookup' else v47['label']

def outside_mentions(v62, v47, v128):
    """Mentions of a value that are NOT already in this question's evidence."""
    v129 = v62.v242('_by_value')
    if v129 is None:
        v129 = v291(v239)
        for v69, v92 in v424(v62['tape'].v130):
            v129[v92].v429(v69)
        v62['_by_value'] = v129
    v67 = v102(v47['slots'])
    return [v69 for v69 in v129.v242(v128, ()) if v69 not in v67]

def shared_import_budget(v62, v47, v130):
    """One budget for every world compared in a question, and the reason is a leak.

    A local candidate's mentions are already IN the evidence, so it usually has nothing left to
    import; a ladder rung comes from elsewhere and always has K. Give each world what it
    happens to have and Phi can read "imported rows present" as "this one is wrong" - the
    landscape gate would then pass on a bookkeeping tell rather than on distance. The budget is
    therefore the minimum available across everything being scored, so every completed world
    carries the same number of rows.
    """
    return v259([v25] + [v267(v554(v62, v47, v92)) for v92 in v130])

def row_meta(v62):
    """slot -> (anchor id, relation id), for 290's two edge channels. Integers, not strings, so
    the channels are one broadcast comparison rather than n^2 python string compares."""
    v41 = v62.v242('_rowmeta')
    if v41 is None:
        v453, v454, v41 = ({}, {}, {})
        for v113 in v62['items']:
            v243, v244 = v245(v113['address'])
            v50 = v453.v241(v243, v267(v453))
            v125 = v454.v241(v244, v267(v454)) if v244 else -1
            for v69 in v113['slots']:
                v41[v69] = (v50, v125)
        v62['_rowmeta'] = v41
    return v41

def graph_base(v62, v47, v81, v82):
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
    if '_base' in v47:
        return v47['_base']
    v57 = v239(v47['slots'])
    v91 = v267(v57)
    v83, v292 = (v62.v241('_ctx', {}), v62.v241('_words', {}))
    for v69 in v102(v57):
        if v69 not in v83:
            v85 = v81.v553(v62['texts'][v69], exclude=v62['tape'].v130[v69])
            v83[v69] = v569.v612(v85, dim=-1) if v85 is not None else None
            v292[v69] = v102(v426(v62['texts'][v69], exclude=v62['tape'].v130[v69]))
    v131 = v62.v242('_median')
    if v131 is None:
        v293 = v240((v267(v92) for v92 in v62['postings'].v130()))
        v131 = v293[v267(v293) // 2] if v293 else 1
        v62['_median'] = v131
    v146, v147 = ([v83[v118] for v118 in v57], [v292[v118] for v118 in v57])
    v148, v149 = (v320.v316(v91, v91), v320.v316(v91, v91))
    if v34 and v91 > 1 and v250((v85 is not None for v85 in v146)):
        v294 = v455((v85 for v85 in v146 if v85 is not None))
        v295 = v320.v266([v85 if v85 is not None else v320.v467(v294) for v85 in v146])
        v296 = v320.v616.v562.v456.v297
        v320.v616.v562.v456.v297 = False
        v298 = (v295 @ v295.v662).v9().v457()
        v320.v616.v562.v456.v297 = v296
        v299 = v320.v262([v85 is None for v85 in v146])
        v298[v299, :] = 0.0
        v298[:, v299] = 0.0
        v298.v314(0.0)
        v148 = v298
    for v132 in v276(v91):
        for v300 in v276(v132 + 1, v91):
            if not v34 and v146[v132] is not None and (v146[v300] is not None):
                v148[v132, v300] = v148[v300, v132] = v9(v146[v132] @ v146[v300])
            v458 = v147[v132] & v147[v300]
            v459 = v235((1 for v248 in v458 if v267(v62['postings'].v242(v248, ())) < v131))
            v149[v132, v300] = v149[v300, v132] = v459 / v311(1, v259(v267(v147[v132]), v267(v147[v300])))
    v133 = v320.v301(v91, v91, offset=1)
    if v133.v302():
        v33[0] += v10((v149[v133[0], v133[1]] > 0).v235())
        v33[1] += v10(v133.v563[1])
        v303 = v148[v133[0], v133[1]]
        v38[0] += v9(v303.v235())
        v38[1] += v9((v303 * v303).v235())
        v38[2] += v10(v303.v302())

    def rank_norm(v95):
        if v133.v302() == 0:
            return v95
        v92 = v95[v133[0], v133[1]]
        v304 = v92.v460()
        v125 = v320.v461(v304, dtype=v320.v473)
        v125[v304] = v320.v462(v267(v92), dtype=v320.v473)
        v463, v464 = v92.v465(return_inverse=True)
        if v267(v463) > 1:
            v466 = v320.v316(v267(v463)).v564(0, v464, v125, 'mean', include_self=False)
            v125 = v466[v464] / (v267(v92) - 1 if v267(v92) > 1 else 1)
        else:
            v125 = v320.v467(v125)
        v305 = v320.v467(v95)
        v305[v133[0], v133[1]] = v125
        v305[v133[1], v133[0]] = v125
        return v305
    v134 = [v565(v148) if 'cos' in v28 else v320.v467(v148), v565(v149) if 'rare' in v28 else v320.v467(v149)]
    if v12:
        v306 = v468(v62)
        v50 = v320.v262([v306.v242(v118, (-1, -1))[0] for v118 in v57])
        v125 = v320.v262([v306.v242(v118, (-1, -2))[1] for v118 in v57])
        v307 = ((v50[:, None] == v50[None, :]) & (v50[:, None] >= 0)).v9()
        v308 = ((v125[:, None] == v125[None, :]) & (v125[:, None] >= 0)).v9()
        v307.v314(0.0)
        v308.v314(0.0)
        v134 += [v307 if 'anchor' in v28 else v320.v467(v307), v308 if 'rel' in v28 else v320.v467(v308)]
        v309 = v320.v301(v91, v91, offset=1)
        if v309.v302():
            v36[0] += v10((v307[v309[0], v309[1]] > 0).v235())
            v36[1] += v10((v308[v309[0], v309[1]] > 0).v235())
            v36[2] += v10(v309.v563[1])
    v117 = v47.v242('query_row', -1)
    v150, v151 = (v320.v316(v91), 0.0)
    if v117 >= 0 and v117 < v91 and (v146[v117] is not None):
        v310 = [v132 for v132 in v276(v91) if v146[v132] is not None and v132 != v117]
        if v267(v310) > 1:
            v66 = v320.v262(v310)
            v469 = v148[v66, v117]
            v261 = v469.v460()
            v470 = v320.v566(v267(v310))
            v470[v261] = v320.v462(v267(v310), dtype=v320.v473)
            v150[v66] = v470 / (v267(v310) - 1)
            v119 = v10(v469.v443())
            v471 = v47['vals'][v310[v119]]
            v121 = [v9(v469[v87]) for v87, v132 in v424(v310) if v47['vals'][v132] != v471]
            v472 = v9(v469.v311() - v469.v259())
            if v472 > 1e-09:
                v151 = (v9(v469.v311()) - v311(v121)) / v472 if v121 else 1.0
            else:
                v151 = 1.0 if not v121 else 0.0
    v135 = v47.v242('n_first', v91)
    v136 = [v47['S'].v567() if v132 < v135 else v47.v242('S2', v47['S']).v567() for v132 in v276(v91)]
    v71 = v47.v242('own_rows')
    v137 = {'n': v91, 'slots': v57, 'chans': v134, 'qcos': v150, 'qmargin': v151, 'subj': v136, 'nfirst': v135, 'qrow': v117, 'isown': [v9(v118 in v71) for v118 in v57] if v71 is not None else None}
    v37[0] += v91
    v37[1] = v311(v37[1], v91)
    v37[2] += 1
    v47['_base'] = v137
    return v137

def graph_from_base(v62, v47, v81, v82, v138):
    """One completed world, from the cached base. Only `same` and the count share change."""
    v51 = v312(v62, v47, v81, v82)
    v91, v57, v117 = (v51['n'], v51['slots'], v51['qrow'])
    v44 = v239(v47['vals'])
    if v138 is not None:
        v44[v117] = v138
    v313, v246 = ([], {})
    for v92 in v44:
        v313.v429(v246.v241(v92 if v645(v92, v8) else v580(v92), v267(v246)))
    v139 = v320.v262(v313)
    v46 = (v139[:, None] == v139[None, :]).v9()
    v46.v314(0.0)
    v140 = v320.v266([v46 if 'same' in v28 else v320.v467(v46)] + v51['chans'], -1).v315(v82)
    v141 = v247(v44)
    v142 = [[v141[v44[v132]] / v91 if v132 != v117 or v138 is not None else 0.0, v9(v51['subj'][v132] in v62['texts_lc'][v57[v132]]), v9(v132 >= v51['nfirst']), 1.0 / v91, v9(v132 == v117), 0.0, v9(v51['qcos'][v132]), v51['qmargin']] for v132 in v276(v91)]
    if v51['isown'] is not None:
        for v132 in v276(v91):
            v142[v132].v429(v51['isown'][v132])
    v142 = v320.v262(v142, dtype=v320.v473, device=v82)
    return (v140, v46.v570(-1).v315(v82), v142)

def build_graph(v62, v47, v81, v82, v138=None, v143=None):
    """286/289a's graph verbatim, plus the side indicator COMPARE needs and, for a completed
    world, the candidate's own mentions imported from elsewhere on the tape."""
    v144 = v25 if v143 is None else v143
    if v144 == 0 and (not v14) and (v138 not in (None, v15)):
        pass
    elif v144 == 0 and v138 != v15:
        return v474(v62, v47, v81, v82, v138)
    if v144 == 0 and v138 == v15:
        return v474(v62, v47, v81, v82, None)
    v57, v44 = (v47['slots'], v47['vals'])
    v145 = v267(v57)
    if v138 is not None:
        v57, v44 = (v239(v57), v239(v44))
        v44[v47['query_row']] = v138
        v43 = v25 if v143 is None else v143
        for v69 in v554(v62, v47, v138)[:v43]:
            v57.v429(v69)
            v44.v429(v62['tape'].v130[v69])
    v91 = v267(v57)
    v83, v292 = (v62.v241('_ctx', {}), v62.v241('_words', {}))
    for v69 in v102(v57):
        if v69 not in v83:
            v85 = v81.v553(v62['texts'][v69], exclude=v62['tape'].v130[v69])
            v83[v69] = v569.v612(v85, dim=-1) if v85 is not None else None
            v292[v69] = v102(v426(v62['texts'][v69], exclude=v62['tape'].v130[v69]))
    v131 = v62.v242('_median')
    if v131 is None:
        v293 = v240((v267(v92) for v92 in v62['postings'].v130()))
        v131 = v293[v267(v293) // 2] if v293 else 1
        v62['_median'] = v131
    v146 = [v83[v118] for v118 in v57]
    v147 = [v292[v118] for v118 in v57]
    v46 = v320.v316(v91, v91)
    v148 = v320.v316(v91, v91)
    v149 = v320.v316(v91, v91)
    if v34 and v91 > 1 and v250((v85 is not None for v85 in v146)):
        v294 = v455((v85 for v85 in v146 if v85 is not None))
        v295 = v320.v266([v85 if v85 is not None else v320.v467(v294) for v85 in v146])
        v296 = v320.v616.v562.v456.v297
        v320.v616.v562.v456.v297 = False
        v298 = (v295 @ v295.v662).v9().v457()
        v320.v616.v562.v456.v297 = v296
        v299 = v320.v262([v85 is None for v85 in v146])
        v298[v299, :] = 0.0
        v298[:, v299] = 0.0
        v298.v314(0.0)
        v148 = v298
    for v132 in v276(v91):
        for v300 in v276(v132 + 1, v91):
            v46[v132, v300] = v46[v300, v132] = v9(v44[v132] == v44[v300])
            if not v34 and v146[v132] is not None and (v146[v300] is not None):
                v148[v132, v300] = v148[v300, v132] = v9(v146[v132] @ v146[v300])
            v458 = v147[v132] & v147[v300]
            v459 = v235((1 for v248 in v458 if v267(v62['postings'].v242(v248, ())) < v131))
            v149[v132, v300] = v149[v300, v132] = v459 / v311(1, v259(v267(v147[v132]), v267(v147[v300])))
    v133 = v320.v301(v91, v91, offset=1)
    if v133.v302():
        v33[0] += v10((v149[v133[0], v133[1]] > 0).v235())
        v33[1] += v10(v133.v563[1])
        v303 = v148[v133[0], v133[1]]
        v38[0] += v9(v303.v235())
        v38[1] += v9((v303 * v303).v235())
        v38[2] += v10(v303.v302())

    def rank_norm(v95):
        if v133.v302() == 0:
            return v95
        v92 = v95[v133[0], v133[1]]
        v304 = v92.v460()
        v125 = v320.v461(v304, dtype=v320.v473)
        v125[v304] = v320.v462(v267(v92), dtype=v320.v473)
        v463, v464 = v92.v465(return_inverse=True)
        if v267(v463) > 1:
            v466 = v320.v316(v267(v463)).v564(0, v464, v125, 'mean', include_self=False)
            v125 = v466[v464] / (v267(v92) - 1 if v267(v92) > 1 else 1)
        else:
            v125 = v320.v467(v125)
        v305 = v320.v467(v95)
        v305[v133[0], v133[1]] = v125
        v305[v133[1], v133[0]] = v125
        return v305
    v140 = v320.v266([v46 if 'same' in v28 else v320.v467(v46), v565(v148) if 'cos' in v28 else v320.v467(v148), v565(v149) if 'rare' in v28 else v320.v467(v149)], -1).v315(v82)
    v141 = v247(v44)
    v135 = v47.v242('n_first', v91)
    v117 = v47.v242('query_row', -1)
    v136 = [v47['S'].v567() if v132 < v135 or v132 >= v145 else v47.v242('S2', v47['S']).v567() for v132 in v276(v91)]
    v150 = v320.v316(v91)
    v151 = 0.0
    if v117 >= 0 and v146[v117] is not None:
        v310 = [v132 for v132 in v276(v91) if v146[v132] is not None and v132 != v117]
        if v267(v310) > 1:
            v66 = v320.v262(v310)
            v469 = v148[v66, v117]
            v261 = v469.v460()
            v125 = v320.v566(v267(v310))
            v125[v261] = v320.v462(v267(v310), dtype=v320.v473)
            v150[v66] = v125 / (v267(v310) - 1)
            v119 = v10(v469.v443())
            v471 = v44[v310[v119]]
            v121 = [v9(v469[v43]) for v43, v132 in v424(v310) if v44[v132] != v471]
            v472 = v9(v469.v311() - v469.v259())
            if v472 > 1e-09:
                v151 = (v9(v469.v311()) - v311(v121)) / v472 if v121 else 1.0
            else:
                v151 = 1.0 if not v121 else 0.0
    v142 = v320.v262([[v141[v44[v132]] / v91 if v132 != v117 or v138 is not None else 0.0, v9(v136[v132] in v62['texts_lc'][v57[v132]]), v9(v132 >= v135), 1.0 / v91, v9(v132 == v117), v9(v132 >= v145), v9(v150[v132]), v151] for v132 in v276(v91)], dtype=v320.v473, device=v82)
    return (v140, v46.v570(-1).v315(v82), v142)

def ladder_scores_for(v100, v62, v47, v82, v81):
    """Phi on the three wrong worlds, in ladder order. Empty when the tape could not supply one."""
    if not v47.v242('ladder'):
        return None
    v43 = v253(v62, v47, v239(v47['cands']) + [v47['ladder'][v125] for v125 in v23])
    v152 = []
    for v153 in v23:
        v140, v46, v142 = v475(v62, v47, v81, v82, query_value=v47['ladder'][v153], import_k=v43)
        v152.v429(v100.v568(v140, v46, v142))
    return v320.v266(v152)

def cand_logits_for(v100, v62, v47, v82, v81):
    """Score one completed world per candidate and let them compete.

    This is 288's repair loop turned inward: instead of preferring a group, the mind writes the
    conjecture into the query row, reads the world that results, and says how well it hangs
    together. The query-row indicator stays set, so a completed world is never mistaken for an
    observed one - the conjecture is marked as a conjecture, which is the derived-slot
    discipline applied to reading.
    """
    v44 = v239(v47['cands']) + [v47['ladder'][v125] for v125 in v23] if v47.v242('ladder') else v239(v47['cands'])
    v43 = v253(v62, v47, v44)
    v152 = []
    for v85 in v47['cands']:
        v140, v46, v142 = v475(v62, v47, v81, v82, query_value=v85, import_k=v43)
        v152.v429(v100.v568(v140, v46, v142))
    return v320.v266(v152)

def loss_for(v100, v62, v47, v82, v81):
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
    if v47['verb'] != 'lookup':
        raise v238(f"{v47['verb']} is exact algebra now and has no loss")
    v154 = v317(v100, v62, v47, v82, v81)
    if v17 == 'reward':
        v158 = v320.v263(v154, 0)
        v305 = v320.v476(v158, -1.0)
        v305[v47['label']] = 1.0
        if v13 and v47.v242('answerable') and (v15 in v47['cands']):
            v305[v47['cands'].v420(v15)] = 0.75
        return -(v158 * v305).v235()
    v155 = v318(v100, v62, v47, v82, v81)
    if v155 is None:
        return v569.v477(v154.v570(0), v320.v262([v47['label']], device=v82))
    v156 = v320.v319([v154, v155])
    v157 = -(v156[v47['label']] - v320.v571(v156, 0))
    for v43 in v276(v267(v23) - 1):
        v157 = v157 - (v155[v43] - v320.v571(v155[v43:], 0))
    return v157

@v320.v159()
def predict_with_confidence(v100, v62, v47, v82, v81):
    """What would be said, how sure, and what the tape says - the three things an audit needs.

    The exact verbs answer through the algebra with confidence 1.0, which is not flattery: it
    is the honest statement that a computed answer is certain GIVEN the same-value relation.
    When s_ij becomes a trained judgment, its uncertainty enters here and 1.0 stops being the
    right number - that is the seam where the future work plugs in.
    """
    if v47['verb'] != 'lookup':
        return (1.0, v572(v47), v478(v47))
    v154 = v317(v100, v62, v47, v82, v81)
    v158 = v320.v263(v154, -1)
    v43 = v10(v158.v443())
    return (v9(v158[v43]), v47['cands'][v43], v478(v47))

def main() -> v10:
    global SEED, LOG_PATH, LADDER_ON, EDGES_ON, IMPORT_K, INK, FP, WORDS, FAST_COS, VIEWS, ROW_DROPOUT, VIEW_MODE, NEIGHBOURS, REFUSE, GRAPH_CACHE, OPEN, OBJECTIVE
    v160 = v479.v321()
    v160.v322('--smoke', action='store_true')
    v160.v322('--train-steps', type=v10, default=0)
    v160.v322('--tape-period', type=v10, default=50)
    v160.v322('--addresses', type=v10, default=0)
    v160.v322('--min-mentions', type=v10, default=2)
    v160.v322('--address-tau', type=v9, default=0.9)
    v160.v322('--tau-mode', choices=('absolute', 'density'), default='absolute', help="absolute keeps 279's fixed cosine and reproduces every earlier run bit for bit. density derives tau so the WRITE ink produces a tape of --tau-target-density mentions per address - required whenever the write ink changes, because a different ink at a fixed cosine shatters the tape and the threshold becomes what the arm measures")
    v160.v322('--tau-target-density', type=v9, default=v32, help='mentions per address to calibrate to. Default is the MEASURED arc/mean train tape (2388 slots / 804 addresses) that every scoreboard number was taken on')
    v160.v322('--tau-calib-iters', type=v10, default=12, help='bisection steps for --tau-mode density. 12 over the full [0, 1] bracket resolves tau to ~2e-4, which holds the density error under 0.005 even where the merge curve is steep; each extra step is pure arithmetic because CachedBank has already inked the corpus')
    v160.v322('--address-overlap', type=v10, default=2)
    v160.v322('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v160.v322('--lr', type=v9, default=0.001)
    v160.v322('--holdout', choices=('corpus', 'address'), default='corpus')
    v160.v322('--no-scan-cache', action='store_true', help='disable the exact corpus-scan memo (use to verify it changes nothing)')
    v160.v322('--no-fast-grouping', action='store_true', help='disable the batched single-link grouping (use to verify it changes nothing)')
    v160.v322('--wiki-bytes', type=v10, default=0)
    v160.v322('--train-lines', type=v10, default=0)
    v160.v322('--eval-lines', type=v10, default=0)
    v160.v322('--import-k', type=v10, default=v25, help='mentions of a candidate imported when completing its world; 0 reproduces the broken ladder where every absent value looked alike')
    v160.v322('--edge-channels', type=v8, default=','.v573(v26), help='comma list from same,cos,rare - zero the rest. Ablation to find which channel carries the paired win over counting')
    v160.v322('--ink', choices=('mean', 'bigram'), default=v29, help="phrase axis: mean reproduces today's order-blind ctx_fp exactly; bigram binds adjacent words with a fixed non-commutative permutation so the ink can tell `X defeated Y` from `Y defeated X`")
    v160.v322('--fp', choices=('arc', 'hash'), default=v30, help='word axis: arc is the frozen stage191 encoder; hash is character n-grams into a blake2b digest - nothing trained, no character vocabulary, no OOV, every script')
    v160.v322('--words', choices=('ascii', 'unicode'), default=v31, help="what counts as a word. unicode only pays off with --fp hash: arc's stoi has no Cyrillic, so a wider intake would just be discarded")
    v160.v322('--fp-ngram', type=v10, default=3, help='character n-gram length for --fp hash')
    v160.v322('--write-fp', choices=('arc', 'hash'), default='arc', help="ink used to GROUP mentions into addresses. Pinned by default so an ink A/B varies reading only; 279's tau is an absolute cosine and a different ink shatters the tape against it")
    v160.v322('--probe-period', type=v10, default=250, help='how often to score the fixed probe tape. The training curve is measured on a different tape every resample and cannot tell converged from overfitting; this one can')
    v160.v322('--views', type=v10, default=1, help="reconciliation (ROADMAP 20): the mind reads V independently thinned views of each question with the SAME weights, logits are pooled by a mean, and the views' disagreement is a label-free confidence signal. 1 reproduces every earlier run bit for bit; V>1 needs --row-dropout as the thinning rate")
    v160.v322('--neighbours', type=v10, default=0, help='290 (ROADMAP §19): build N(a) from up to this many addresses per route - shared anchor, shared relation, shared rare words - put all their rows in ONE graph, and switch to the sparse verb. 0 reproduces every earlier run bit for bit, including the 5601 parameter count')
    v160.v322('--seed', type=v10, default=v3, help='every draw in the run - tapes, questions, probe, views. Added when 292 came back with held z +2.59 against corpus retrieval and train z -0.15 on the same weights: two samples of one quantity disagreeing by 2.7 sigma. A second seed is the only cheap way to tell structure from a lucky split, and there was no way to ask for one')
    v160.v322('--objective', choices=('ce', 'reward'), default='ce', help="ce is cross-entropy, every run to date. reward optimises 280's fixed payoff directly: L = -sum_c p(c)R(c), closed form, no new constant. Removes the mismatch between what is trained and what is scored; it does not remove a collapse caused by an unanswerable task")
    v160.v322('--open', action='store_true', help="292: the hidden value occurs exactly once at the address, so it is FOREIGN to the evidence and no rule over the address's own rows can reach it. Candidates are the truth and the three ladder rungs, all four importing the same number of rows - the symmetric comparison the ladder could never get in 289. Needs --import-k >= 1, because with 0 imports all four worlds are the same graph")
    v160.v322('--no-graph-cache', action='store_true', help='rebuild every channel per candidate, as before graph_base existed. Dense arms only - use it to verify the cache changed nothing')
    v160.v322('--refuse', action='store_true', help='291: keep the sparse questions whose answer is on NO row of N(a) and let the mind score the world where the query row stays unknown. Refusal becomes an action with a label the tape supplies, not a threshold on a confidence score. Needs --neighbours')
    v160.v322('--view-mode', choices=('thin', 'region'), default='thin', help="how views are cut. thin = recon3's random subsampling (views share ~65%% of rows, D measured model noise, pooled lost to single). region = contiguous stretches of the tape in write order - disjoint by construction, deterministic, so D measures whether the CORPUS agrees with itself at this address rather than whether one sampler agrees with another")
    v160.v322('--row-dropout', type=v9, default=0.0, help='probability of dropping each evidence row during TRAINING, so the mind sees the same fact at several densities. 0 reproduces every earlier run bit for bit - it draws from its own generator')
    v160.v322('--dim', type=v10, default=32, help='width of the mind. Exposed so the max-pool result can be checked at MATCHED parameter count: max-pool added 2048 weights along with the max, and one of those two is the cause')
    v160.v322('--no-max-pool', action='store_true', help='pool with the mean alone, as every run before this one did. A mean cannot express existence, and a high-margin question is decided by one row')
    v160.v322('--no-fast-cos', action='store_true', help='build the pairwise cosine matrix with the original per-pair loop (use to verify the batched version changes nothing)')
    v160.v322('--probe-frac', type=v10, default=10, help='one anchor in this many is reserved for the probe and excluded from both training and held-out scoring, so the stopping step is never chosen using an anchor the evaluation will ask about')
    v160.v322('--probe-size', type=v10, default=200, help='how many probe questions to score. Same questions every time - a probe set that changes is the defect this replaces')
    v160.v322('--no-early-stop', action='store_true', help='keep the last step instead of the best probe step - reproduces every run before the probe tape existed')
    v160.v322('--write-ink', choices=('mean', 'bigram'), default='mean')
    v160.v322('--write-words', choices=('ascii', 'unicode'), default='ascii')
    v160.v322('--no-ladder', action='store_true', help='ablation: train Phi on the task term alone, the control the ladder is measured against')
    v160.v322('--run-tag', type=v8, default='')
    v161 = v160.v323()
    v3 = v161.v162
    v24 = not v161.v324
    v25 = v161.v143
    v29, v30, v31 = (v161.v325, v161.v326, v161.v327)
    v34 = not v161.v328
    v329.v124 = not v161.v330
    v20, v22, v21 = (v161.v331, v161.v332, v161.v333)
    v12, v13, v16 = (v161.v334, v161.v335, v161.v336)
    v17 = v161.v163
    if v16 and v12:
        v105('  --open and --neighbours are two different verbs; run them apart or the arm measures their sum and credits whichever was named last')
        return 1
    if v16 and v161.v143 < 1:
        v105('  --open needs --import-k >= 1: with nothing imported, the true value and all three rungs give the identical graph and the question has no content')
        return 1
    if v16 and v161.v324:
        v105('  --open IS the ladder - the rungs are its candidates - so --no-ladder would leave it with one candidate')
        return 1
    v14 = not v161.v337
    if v161.v337 and v12:
        v105("  --no-graph-cache is a dense-arm verification path; it does not build 290's two extra edge channels and would silently score a 3-channel graph")
        return 1
    if v13 and (not v12):
        v105('  --refuse needs --neighbours: the unanswerable questions are the sparse ones')
        return 1
    if v12:
        if v161.v143:
            v105('  --neighbours needs --import-k 0: an imported world and the refusal world would carry different row counts, which is a bookkeeping tell, not evidence')
            return 1
    if v20 > 1 and v21 == 'thin' and (v22 <= 0):
        v105('  --views > 1 with --row-dropout 0: every view is the same graph and the pool is decoration; set a thinning rate')
        return 1
    if (v161.v407, v161.v406) != ('arc', 'mean') and v161.v480 == 'absolute' and (v161.v360 == 0.9):
        v105(f"  --write-fp {v161.v407} --write-ink {v161.v406} rewrites the tape, and 279's tau is an absolute cosine: at 0.90 a different ink merges almost nothing. Add --tau-mode density (target defaults to the measured arc tape, {v32} mentions/address).")
        return 1
    for v338, v248, v234 in (('read', v31, v30), ('write', v161.v408, v161.v407)):
        if v248 == 'unicode' and v234 == 'arc':
            v105(f"  --{('' if v338 == 'read' else 'write-')}words unicode with an arc encoder widens the intake into a vocabulary that cannot represent it; use hash there or ascii")
            return 1
    v28 = {v85.v481() for v85 in v161.v617.v421(',') if v85.v481()}
    if not v28 <= v102(v26):
        v105(f'  unknown edge channel in {v240(v28)}; allowed {v26}')
        return 1
    if not v28:
        v105('  every edge channel disabled: nothing to read')
        return 1
    if v12:
        v28 |= v102(v27)
    v164 = v161.v405 and f'_{v161.v405}' or ''
    v164 += '_addrholdout' if v161.v359 == 'address' else ''
    v7 = v0 / f'_stage289_log{v164}.txt'
    v7.v412.v233(parents=True, exist_ok=True)
    v7.v339('', encoding='utf-8')
    v82 = v320.v82('cuda' if v320.v562.v574() else 'cpu')
    v55 = v482.v340(v3)
    v320.v341(v3)
    v165 = v342.v342()
    v166 = v161.v343 or (600 if v161.v404 else 6000)
    v167 = v161.v344 or (300 if v161.v404 else 400)
    v105(f'Stage289 derivation start {v659.v512(v660.v636).v549()} device={v82} holdout={v161.v359}')
    v115, v115, v345, v346 = v347()
    v168 = v483.v348(v8(v575.v484))
    v169 = v168.v485(v486) or 0
    v170 = None
    if v30 != 'hash':
        v170 = v618(v346, v168.v646()).v315(v82)
        v170.v487(v320.v619(v1, map_location=v82, weights_only=False)['model'])
        v170.v373()
        for v62 in v170.v488():
            v62.v576(False)
    v171 = v349[v31]

    def make_bank(v350, v351, v125):
        v137 = v577(device=v82, n=v161.v546, rule=v125) if v350 == 'hash' else v578(v170, v345, v82)
        return v489(v620(v137, rule=v125) if v351 == 'bigram' else v137)
    v81 = v352(v30, v29, v171)
    v172 = v81 if (v161.v407, v161.v406, v161.v408) == (v30, v29, v31) else v352(v161.v407, v161.v406, v349[v161.v408])
    v173 = v81.v174
    v175 = v173.v174 if v29 == 'bigram' else v173
    v176 = v579.v490(v170) if v170 is not None else 'not_loaded'
    v177: v178 = {}
    v179 = v353.v180

    def _cached_common(v354, v355: v10=3):
        v43 = (v580(v354), v267(v354), v355)
        if v43 not in v177:
            v177[v43] = v179(v354, v355)
        return v177[v43]
    v353.v180 = v181
    if not v161.v356:
        v491(v353)
    if not v161.v357:
        v492(v353)
    with v2.v336('r', encoding='utf-8', errors='ignore') as v234:
        v358 = v234.v493(v161.v581 or (4000000 if v161.v404 else 30000000))
    v182 = [v494.v481() for v494 in v358.v421('\n') if 80 <= v267(v494.v481()) <= 400]
    v183 = v10(0.7 * v267(v182))
    v184 = v182[:v183][:v161.v184 or (3000 if v161.v404 else 25000)]
    v185 = v182[v183:][:v161.v185 or (1500 if v161.v404 else 12000)]
    v105(f'  lines: train {v267(v184)}, eval {v267(v185)} (the probe reserves ANCHORS, not lines - see `reserved`)')
    if v161.v359 == 'address':
        v185 = v184
    v186 = v187(v495())
    v105(f'  word rule matches stage194: {v186}  (rule={v31}, fp={v30}, ink={v29})')
    if v29 == 'bigram':
        v92 = v187(v173.v582(v184[:200]))
        v186 &= v92
        v105(f'  bigram tokenisation matches base mean-ink: {v92}')
    if v30 == 'hash':
        v92, v496 = v497(v175)
        v186 &= v92
        v105(f'  hash ink deterministic and digest-faithful: {v92}  {v496}')
    if not v186:
        v105('  ABORT: the ink does not do what it says it does')
        return 1

    def side(v60: v8) -> v10:
        return v10(v647.v621(v440.v274(v60).v648('utf-8')).v498(), 16) & 1

    def reserved(v60: v8) -> v187:
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
        v288 = v647.v621(f'probe:{v440.v274(v60)}'.v648('utf-8')).v498()
        return v10(v288, 16) % v161.v583 == 0
    v188 = v161.v360 if v161.v480 == 'absolute' else v499(v161.v500, v161.v501, v105)

    def new_pack(v125, v354, v361, v198=False, v362=None):
        v62 = v584.v502(v354, bank=v172, tok=v168, pad_id=v169, device=v82, rng=v125, n_addr=v362 or v167, min_mentions=v161.v270, tau=v188, overlap=v161.v585, soft_match=0.0, min_per_family=8, addr_key=v161.v271)
        v62 = v178(v62)
        if v161.v359 == 'address':
            v62['items'] = [v113 for v113 in v62['items'] if v338(v113['address']) == v361]
        v62['items'] = [v113 for v113 in v62['items'] if v649(v113['address']) == v198]
        return v62
    v189 = v190

    def by_verb(v363):
        v279 = v291(v239)
        for v47 in v363:
            v279[v47['verb']].v429(v47)
        return v279
    v100 = v329(v82, d=v161.v503, n_edge=3 + (v267(v27) if v12 else 0), n_node=8 + (1 if v12 else 0), grown=v267(v27) if v12 else 0)
    v191 = v320.v504.v364(v100.v488(), lr=v161.v505, weight_decay=0.01)
    v192 = v10(v235((v415.v302() for v415 in v100.v488())))

    def cand_logits(v62, v47):
        return v317(v100, v62, v47, v82, v81)

    def loss_of(v62, v47):
        return v506(v100, v62, v47, v82, v81)
    v48 = v365(v55, v184, 0)
    v193 = v189(v48, v55)
    v129 = v366(v193)
    v194 = v235((1 for v47 in v129.v242('lookup', ()) if v47.v242('ladder')))
    v105(f"  tape: {v48['n_addresses']} addresses, {v48['n_slots']} slots | questions {v609.v550({v43: v267(v92) for v43, v92 in v129.v127()})} | params {v192}")
    v105(f"  ladder coverage: {v194}/{v267(v129.v242('lookup', ()))} lookup questions have all three rungs; the rest train on the task term alone")
    if v267(v129.v242('lookup', ())) < v507.v367:
        v105('  too few lookup questions: raise --train-lines')
        return 1
    v195 = v365(v482.v340(v3 + 99), v185, 1)
    v196 = v189(v195, v482.v340(v3 + 7))
    v197 = v342.v342()
    v198 = v365(v482.v340(v3 + 555), v184, 0, probe=True, n_addr_over=v167 * v161.v583)
    v105(f"  probe pack: {v267(v198['items'])} reserved addresses ({v342.v342() - v197:.0f}s to build)")
    v199 = [v47 for v47 in v189(v198, v482.v340(v3 + 556)) if v47['verb'] in v11 and (not v47.v242('ladder'))][:v161.v508]
    v200 = v342.v342()
    v201 = []
    v202 = v482.v340(v3 + 6060)
    for v47 in v199:
        v94, v509 = v265(v47, v202, v82)
        v331 = []
        for v368 in v94:
            v43 = v253(v198, v368, v239(v368['cands']))
            v331.v429([v475(v198, v368, v81, v82, query_value=v85, import_k=v43) for v85 in v368['cands']])
        v201.v429((v331, v509, v320.v262([v47['label']], device=v82)))
    v105(f'  probe tape: {v267(v199)} lookup questions, never trained on; {v235((v267(v542) for v663, v115, v115 in v201 for v542 in v663))} graphs cached ({v20} view(s)/question) in {v342.v342() - v200:.0f}s')

    @v320.v159()
    def probe_loss():
        if not v201:
            return v9('nan')
        v100.v373()
        v369 = 0.0
        for v331, v509, v76 in v201:
            v96 = v320.v266([v320.v266([v100.v568(v140, v46, v142) for v140, v46, v142 in v650]) for v650 in v331])
            v369 += v9(v569.v477(v433(v96, v509).v570(0), v76))
        v100.v510()
        return v369 / v267(v201)
    v203 = v342.v342()
    v370()
    v204 = v342.v342() - v203
    v205 = v267([v118 for v118 in v276(1, v166 + 1) if v118 % v161.v623 == 0 or v118 == v166])
    v105(f'  probe eval: {v204:.2f}s x {v205} = {v204 * v205 / 60:.1f} min added to this run')
    v206 = v482.v340(v3 + 4242)
    v120 = {'loss': v9('inf'), 'step': 0, 'state': None}
    v207 = []
    v371, v372 = ([], [])
    for v208 in v276(1, v166 + 1):
        if (v208 - 1) % v161.v547 == 0 and v208 > 1:
            v511 = {v113['address'] for v113 in v48['items']}
            v48 = v365(v55, v184, 0)
            v193 = v189(v48, v55)
            v129 = v366(v193)
            v512 = {v113['address'] for v113 in v48['items']}
            if v511:
                v35.v429(v267(v511 & v512) / v311(1, v267(v511 | v512)))
            if not v129.v242('lookup'):
                v105('  empty tape after resample')
                return 1
        v47 = v129['lookup'][v55.v419(v267(v129['lookup']))]
        if v20 > 1:
            if v21 == 'region' and v161.v332 > 0:
                v586 = v622(v47, v206, 1.0 - v161.v332)
                if v586 is not None:
                    v47 = v586
            v289, v115, v115 = v587(v100, v48, v47, v82, v81, v206)
            v157 = v569.v477(v289.v570(0), v320.v262([v47['label']], device=v82))
        else:
            if v161.v332 > 0:
                v586 = v622(v47, v206, 1.0 - v161.v332)
                if v586 is not None:
                    v47 = v586
            v157 = v588(v48, v47)
        v191.v513(set_to_none=True)
        v157.v514()
        v320.v123.v589.v515(v100.v488(), 1.0)
        v191.v208()
        v371.v429(v9(v157))
        if v208 % v161.v623 == 0 or v208 == v166:
            v516 = v370()
            v207.v429({'step': v208, 'probe_loss': v516})
            if v516 < v120['loss']:
                v120 = {'loss': v516, 'step': v208, 'state': {v43: v92.v670().v664() for v43, v92 in v100.v671().v127()}}
        if v208 % v311(1, v166 // 8) == 0:
            v372.v429({'step': v208, 'loss': v9(v667.v431(v371[-200:])), 'probe_loss': v207[-1]['probe_loss'] if v207 else None})
            v105(f"  step {v208}/{v166} train={v667.v431(v371[-200:]):.4f} probe={(v207[-1]['probe_loss'] if v207 else v9('nan')):.4f}")
    if not v161.v517 and v120['state'] is not None:
        v100.v487(v120['state'])
        v105(f"  early stop: restored step {v120['step']} (probe {v120['loss']:.4f}) of {v166}")
    v100.v373()
    v209 = v579.v490(v170) if v170 is not None else 'not_loaded'

    @v320.v159()
    def examine(v62, v363):
        v374 = {v92: {'n': 0, 'model': 0, 'rival': 0, 'rival_cos': 0, 'floor': 0.0} for v92 in ('count', 'compare', 'lookup')}
        v518, v68 = (v247(), [])
        v375 = 0
        v376 = v377 = 0
        v378 = []
        v379 = []
        v380 = []
        v381 = []
        v382 = v482.v340(v3 + 7788)
        v383 = {v43: 0.0 for v43 in ('true',) + v23}
        v519, v520, v521, v522 = (0, 0, 0, 0)
        v384 = []
        for v47 in v363:
            v92 = v47['verb']
            if v92 == 'lookup':
                if v20 > 1:
                    v154, v651, v652 = v587(v100, v62, v47, v82, v81, v382)
                    v379.v429([v10(v10(v154.v443()) == v47['label']), v10(v10(v651.v443()) == v47['label']), v652])
                else:
                    v154 = v653(v62, v47)
                v590 = v47['cands'][v10(v154.v443())]
                v73 = v47['cands'][v47['label']]
                v155 = v318(v100, v62, v47, v82, v81)
                if v47.v242('ladder'):
                    v384.v429(v253(v62, v47, v239(v47['cands']) + [v47['ladder'][v125] for v125 in v23]))
                if v155 is not None:
                    v624 = [v9(v154[v47['label']])] + [v9(v415) for v415 in v155]
                    for v255, v603 in v260(('true',) + v23, v624):
                        v383[v255] += v603
                    v519 += 1
                    for v524, v525 in v260(v624, v624[1:]):
                        if v524 == v525:
                            v522 += 1
                            continue
                        v520 += v10(v524 > v525)
                        v521 += 1
                v591 = v442(v47)
                v374[v92]['floor'] += 1.0 / v267(v47['cands'])
                v68.v429({'k': f"{v47['address']}#{v47.v242('hid', v267(v47['slots']))}", 'hit': v10(v590 == v73)})
                v592 = not v47.v242('open') and (v47.v242('answerable', True) or not v13)
                if v592:
                    if v590 == v73 and v591 != v73:
                        v376 += 1
                    elif v590 != v73 and v591 == v73:
                        v377 += 1
                v625, v626 = v600(v62, v47, v81, v82)
                if v625 is not None:
                    v374[v92]['rival_cos'] += v10(v625 == v73)
                    if v592:
                        v378.v429((v10(v590 == v73), v10(v625 == v73), v626))
                if v47.v242('open'):
                    v627 = {v654: v655 for v655, v654 in v47['rung_of'].v127()}
                    v624 = [v9(v154[v47['cands'].v420(v73)])] + [v9(v154[v47['cands'].v420(v627[v665])]) for v665 in v23]
                    for v255, v603 in v260(('true',) + v23, v624):
                        v383[v255] += v603
                    v519 += 1
                    for v524, v525 in v260(v624, v624[1:]):
                        if v524 == v525:
                            v522 += 1
                        else:
                            v520 += v10(v524 > v525)
                            v521 += 1
                    v628 = v656(v62, v47, v81, v82)
                    v381.v429([v10(v590 == v73), v10(v628 == v73), v10(v628 is not None)])
                if v47.v242('sparse'):
                    v629 = v657(v47)
                    v380.v429([v10(v47['answerable']), v10(v590 == v73), v10(v590 == v15), v10(v591 == v73), v10(v625 == v73), v658(v47), v626, v267(v47['own_rows']), v10(v629 == v73), v10(v629 == v15)])
            else:
                v590 = v572(v47)
                v73 = v47['label']
                v591 = v590
                v374[v92]['floor'] += 1.0 / (v267(v5) if v92 == 'count' else v267(v6))
                v375 += v10(v590 != v73)
            v374[v92]['n'] += 1
            v374[v92]['model'] += v10(v590 == v73)
            v374[v92]['rival'] += v10(v591 == v73)
            v518[v92, v8(v73), v8(v590)] += 1
        v70 = {}
        for v92, v139 in v374.v127():
            if not v139['n']:
                continue
            v70[v92] = {'n': v139['n'], 'model_accuracy': v139['model'] / v139['n'], 'rival_accuracy': v139['rival'] / v139['n'], 'random_floor': v139['floor'] / v139['n']}
            if v92 == 'lookup':
                v70[v92]['rival_cos_accuracy'] = v139['rival_cos'] / v267(v378) if v378 else v9('nan')
                v70[v92]['rival_cos_answered'] = v267(v378)
        v385 = {'n_questions': v519, 'pairs': v521, 'concordant': v520, 'ties_excluded': v522, 'import_budget_zero_rate': v235((1 for v51 in v384 if v51 == 0)) / v311(1, v267(v384)), 'import_budget_mean': v235(v384) / v311(1, v267(v384)), 'concordance': v520 / v521 if v521 else v9('nan'), 'z_vs_half': (v520 / v521 - 0.5) / v635.v608(0.25 / v521) if v521 else v9('nan'), 'mean_phi': {v43: v383[v43] / v519 if v519 else v9('nan') for v43 in ('true',) + v23}}
        v386 = v376 + v377
        v70['lookup_paired_vs_rival'] = {'model_only_right': v376, 'rival_only_right': v377, 'discordant': v386, 'mcnemar_z': (v376 - v377) / v635.v608(v386) if v386 else v9('nan')}

        def mcnemar(v127):
            v50 = v235((1 for v41, v125, v115 in v127 if v41 and (not v125)))
            v51 = v235((1 for v41, v125, v115 in v127 if v125 and (not v41)))
            v279 = v50 + v51
            return {'n': v267(v127), 'model_only_right': v50, 'rival_only_right': v51, 'discordant': v279, 'mcnemar_z': (v50 - v51) / v635.v608(v279) if v279 else v9('nan'), 'max_achievable_z': v635.v608(v279) if v279 else 0.0, 'underpowered': v187(v635.v608(v279) <= 1.645)}
        v70['lookup_paired_vs_rival_cos'] = v523(v378)
        v387 = v240((v41 for v115, v115, v41 in v378 if not v635.v634(v41)))
        v131 = v387[v267(v387) // 2] if v387 else v9('nan')
        v70['lookup_paired_vs_rival_cos_by_margin'] = {'median_margin': v131, 'low_margin': v523([v113 for v113 in v378 if v113[2] <= v131]), 'high_margin': v523([v113 for v113 in v378 if v113[2] > v131])}
        v70['ladder'] = v385
        v70['exact_mismatches'] = v375
        v70['confusion'] = {f'{v50}|{v51}->{v85}': v43 for (v50, v51, v85), v43 in v240(v518.v127())}
        v70['lookup_item_hits'] = v240(v68, key=lambda v288: v288['k'])
        v70['_views'] = v379
        v70['_sparse'] = v380
        if v381:
            v524 = v235((1 for v41, v665, v115 in v381 if v41 and (not v665)))
            v525 = v235((1 for v41, v665, v115 in v381 if v665 and (not v41)))
            v526 = v524 + v525
            v70['open'] = {'n': v267(v381), 'random_floor': 0.25, 'accuracy': v235((v125[0] for v125 in v381)) / v267(v381), 'corpus_retrieval_accuracy': v235((v125[1] for v125 in v381)) / v267(v381), 'corpus_retrieval_answered': v235((v125[2] for v125 in v381)), 'within_address_rivals_undefined': True, 'paired_vs_corpus_retrieval': {'model_only_right': v524, 'rival_only_right': v525, 'discordant': v526, 'mcnemar_z': (v524 - v525) / v635.v608(v526) if v526 else v9('nan'), 'max_achievable_z': v635.v608(v526) if v526 else 0.0, 'underpowered': v187(v635.v608(v526) <= 1.645)}}
        if v380:
            v256 = [v125 for v125 in v380 if v125[0]]
            v396 = [v125 for v125 in v380 if not v125[0]]
            v70['sparse'] = {'n': v267(v380), 'n_answerable': v267(v256), 'n_unanswerable': v267(v396), 'acc_answerable': v235((v125[1] for v125 in v256)) / v267(v256) if v256 else v9('nan'), 'refuse_recall': v235((v125[2] for v125 in v396)) / v267(v396) if v396 else v9('nan'), 'false_refusal': v235((v125[2] for v125 in v256)) / v267(v256) if v256 else v9('nan'), 'rival_own_row_accuracy': v235((v125[8] for v125 in v380)) / v267(v380), 'n_no_own_row': v235((1 for v125 in v380 if v125[7] == 0)), 'acc_no_own_row': v235((v125[1] for v125 in v380 if v125[7] == 0)) / v311(1, v235((1 for v125 in v380 if v125[7] == 0)))}
        return v70
    v210 = v388(v195, v196)
    v211 = v388(v48, v193)
    v389, v390 = (v211.v258('_views', []), v210.v258('_views', []))
    v212 = []
    if v20 > 1:
        with v320.v159():
            for v331, v509, v76 in v201:
                v96 = v320.v266([v320.v266([v100.v568(v140, v46, v142) for v140, v46, v142 in v650]) for v650 in v331])
                v154 = v433(v96, v509)
                v212.v429([v10(v10(v154.v443()) == v10(v76)), v10(v10(v96[0].v443()) == v10(v76)), v434(v96 if v509 is None else v96[1:], v509)])
    v213 = None
    if v20 > 1 and v389 and v390:

        def paired(v74):
            v50 = v235((1 for v516, v666, v115 in v74 if v516 and (not v666)))
            v51 = v235((1 for v516, v666, v115 in v74 if v666 and (not v516)))
            v279 = v50 + v51
            return {'pooled_only_right': v50, 'single_only_right': v51, 'discordant': v279, 'mcnemar_z': (v50 - v51) / v635.v608(v279) if v279 else v9('nan')}

        def d_auc(v74, v132=0):
            v527 = [v125[2] for v125 in v74 if not v125[v132]]
            v528 = [v125[2] for v125 in v74 if v125[v132]]
            if not v527 or not v528:
                return {'auc': v9('nan'), 'z': v9('nan')}
            v50 = v507.v593(v527, v528)
            return {'auc': v50, 'z': v507.v630(v50, v267(v527), v267(v528)), 'n_err': v267(v527), 'n_ok': v267(v528)}

        def t_star_of(v74, v132=0):
            v529 = v291(v239)
            for v125 in v74:
                v529[v125[2]].v429(v125)
            v139, v594, v246 = (None, 0, 0)
            for v279 in v240(v529):
                v542 = v529[v279]
                if (v594 + v235((v415[v132] for v415 in v542))) / (v246 + v267(v542)) < 0.875:
                    break
                v594 += v235((v415[v132] for v415 in v542))
                v246 += v267(v542)
                v139 = v279
            return v139

        def refusal_of(v132):
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
            v139 = v595(v212 if v212 else v389, v132)
            v256 = [v125 for v125 in v390 if v139 is not None and v125[2] <= v139]
            v530 = v267(v390) - v267(v256)
            v531 = v235((1 for v125 in v256 if v125[v132]))
            v532 = v531 / v267(v256) if v256 else v9('nan')
            v533 = (v531 - (v267(v256) - v531) + 0.75 * v530) / v267(v390)
            v534 = (v532 - 0.875) / v635.v608(0.875 * 0.125 / v267(v256)) if v256 else v9('nan')
            return {'p_star': 0.875, 'calibrated_on': 'probe' if v212 else 'train', 'd_threshold': v139, 'd_threshold_from_train': v595(v389, v132), 'held_coverage': v267(v256) / v267(v390), 'held_n_answered': v267(v256), 'held_acc_answered': v532, 'z_acc_vs_breakeven': v534, 'held_reward_selective': v533, 'held_reward_always': v235((1 if v125[v132] else -1 for v125 in v390)) / v267(v390), 'held_reward_blanket_refusal': 0.75}
        v219 = v535(v390)
        v536, v537 = (v596(v390), v597(0))
        v538, v539 = (v596(v390, 1), v597(1))
        v213 = {'views': v20, 'view_mode': v21, 'thin_keep_p': 1.0 - v22 if v21 == 'thin' else None, 'held_pooled_vs_single': v219, 'held_d_auc': v536, 'probe_d_auc': v596(v212), 'train_d_auc': v596(v389), 'refusal': v537, 'answer_full': {'held_d_auc': v538, 'probe_d_auc': v596(v212, 1), 'train_d_auc': v596(v389, 1), 'refusal': v539}, 'gates': {'G_pooled_not_worse': v187(v219['mcnemar_z'] >= 0 if v219['discordant'] else True), 'G_d_predicts_error_held': v187(v536.v242('z', 0) == v536.v242('z', 0) and v536.v242('z', 0) > 1.645), 'G_refusal_beats_blanket': v187(v537['held_reward_selective'] > 0.75)}}
        v105(f'  RECON {v609.v550(v213)}')
    v214 = None
    v391, v392 = (v210.v258('_sparse', []), v211.v258('_sparse', []))
    if v13 and v391:
        v393 = []
        for v47 in v199:
            if not v47.v242('sparse'):
                continue
            v139 = v47['cands'][v47['label']]
            v598, v599 = v600(v198, v47, v81, v82)
            v393.v429([v10(v47['answerable']), v10(v442(v47) == v139), v10(v598 == v139), v658(v47), v599])

        def thresh(v74, v540, v541):
            """Largest set of questions, taken in DESCENDING confidence, whose accuracy still
            clears the derived break-even. Whole ties admitted together and the scan stops at
            the first group that breaks it - the same rule the D threshold uses, and for the
            same two reasons: accuracy decays slowly so a tail sneaks in one item at a time, and
            rows tied at one margin cannot be split anyway. Rows with no margin at all (the ink
            gave the query no context, so the retrieval rival cannot rank) are excluded from the
            calibration and refuse at scoring time."""
            v542 = v291(v239)
            for v125 in v74:
                if v125[v541] == v125[v541]:
                    v542[v125[v541]].v429(v125)
            v139, v594, v246 = (None, 0, 0)
            for v41 in v240(v542, reverse=True):
                v601 = v542[v41]
                if (v594 + v235((v415[v540] for v415 in v601))) / (v246 + v267(v601)) < 0.875:
                    break
                v594 += v235((v415[v540] for v415 in v601))
                v246 += v267(v601)
                v139 = v41
            return v139

        def reward(v74, v540, v541, v139):
            v369 = 0.0
            for v125 in v74:
                if v139 is None or not v125[v541] >= v139:
                    v369 += 1.0 if not v125[0] else 0.75
                else:
                    v369 += 1.0 if v125[v540] else -1.0
            return v369 / v267(v74)

        def fixed_reward(v74, v540, v543):
            """A rival that refuses by RULE rather than by threshold - scored exactly as the
            mind is, so the only difference between them is the judgment."""
            return v235((1.0 if v125[v540] else 0.75 if v125[v543] else -1.0 for v125 in v74)) / v267(v74)
        v394 = v602(v393, 1, 3) if v393 else None
        v395 = v602(v393, 2, 4) if v393 else None
        v256 = [v125 for v125 in v391 if v125[0]]
        v396 = [v125 for v125 in v391 if not v125[0]]
        v397 = (v267(v396) + 0.75 * v267(v256)) / v267(v391)
        v398 = v235((1.0 if v125[1] else 0.75 if v125[2] else -1.0 for v125 in v391)) / v267(v391)
        v214 = {'n': v267(v391), 'n_answerable': v267(v256), 'n_unanswerable': v267(v396), 'unanswerable_rate': v267(v396) / v267(v391), 'mind': {'acc_answerable': v235((v125[1] for v125 in v256)) / v267(v256) if v256 else v9('nan'), 'refuse_recall': v235((v125[2] for v125 in v396)) / v267(v396) if v396 else v9('nan'), 'false_refusal': v235((v125[2] for v125 in v256)) / v267(v256) if v256 else v9('nan'), 'coverage': 1.0 - v235((v125[2] for v125 in v391)) / v267(v391), 'reward': v398}, 'rival_counting': {'threshold_from_probe': v394, 'reward': v631(v391, 3, 5, v394)}, 'rival_retrieval': {'threshold_from_probe': v395, 'reward': v631(v391, 4, 6, v395)}, 'rival_own_row': {'reward': v632(v391, 8, 9), 'acc_answerable': v235((v125[8] for v125 in v256)) / v267(v256) if v256 else v9('nan'), 'refuse_recall': v235((v125[9] for v125 in v396)) / v267(v396) if v396 else v9('nan')}, 'blanket_refusal_reward': v397, 'always_answer_ceiling': (v267(v256) - v267(v396)) / v267(v391), 'gates': {'G_refuse_beats_blanket': v187(v398 > v397), 'G_refuse_beats_counting': v187(v398 > v631(v391, 3, 5, v394)), 'G_refuse_beats_retrieval': v187(v398 > v631(v391, 4, 6, v395)), 'G_refuse_beats_own_row_shortcut': v187(v398 > v632(v391, 8, 9))}}
        v105(f'  REFUSE {v609.v550(v214)}')

    def paraphrase_split(v399):
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
        v544, v545, v246 = (v291(v102), v291(v102), v247())
        for v113 in v399['items']:
            v63 = v113['address']
            v111 = v440.v274(v63)
            for v69 in v113['slots']:
                v603 = v399['tape'].v130[v69]
                v544[v111, v603].v559(v63)
                v246[v111, v603] += 1
                v545[v668((v111, v603.v567()))].v559((v111, v603.v567()))
        v400 = [v43 for v43 in v544 if v246[v43] >= 2]
        v401 = [v62 for v62, v633 in v545.v127() if v267(v62) == 2 and v267(v633) == 2]
        return {'facts_written_twice': v267(v400), 'same_anchor_diff_relation': v235((1 for v43 in v400 if v267(v544[v43]) > 1)) / v267(v400) if v400 else v9('nan'), 'mean_addresses_per_fact': v235((v267(v544[v43]) for v43 in v400)) / v267(v400) if v400 else v9('nan'), 'reversed_pairs': v267(v401), 'reversed_pair_rate': v267(v401) / v311(1, v267(v545))}

    def tape_shape(v399, v363):
        v218 = [v47 for v47 in v363 if v47['verb'] == 'lookup']
        return {'slots': v267(v399['texts']), 'addresses': v267(v399['addr_slots']), 'mentions_per_address': v267(v399['texts']) / v267(v399['addr_slots']) if v399['addr_slots'] else v9('nan'), 'lookup_questions': v267(v218), 'mean_candidates': v235((v267(v47['cands']) for v47 in v218)) / v267(v218) if v218 else v9('nan'), 'paraphrase': v604(v399)}
    v105(f"  HELD {v609.v550({v43: v92 for v43, v92 in v210.v127() if v43 != 'lookup_item_hits'})}")
    v105(f"  CONTROL {v609.v550({v43: v92 for v43, v92 in v211.v127() if v43 != 'lookup_item_hits'})}")
    v215 = v176 == v209
    v216 = v210.v242('lookup', {}).v242('n', 0) >= 2 * v507.v367
    v217 = v187(v210.v242('exact_mismatches', 1) == 0 and v211.v242('exact_mismatches', 1) == 0)
    v218 = v210.v242('lookup', {})
    v219 = v210.v242('lookup_paired_vs_rival', {})
    v220 = v187(v218 and v218['model_accuracy'] > v218['random_floor'])
    v221 = v187(v219.v242('discordant', 0) >= 2 * v507.v367 and (not v635.v634(v219.v242('mcnemar_z', v9('nan')))) and (v219['mcnemar_z'] > 1.645))
    v222 = v210.v242('lookup_paired_vs_rival_cos', {})
    v223 = v187(v222.v242('discordant', 0) >= 2 * v507.v367 and (not v635.v634(v222.v242('mcnemar_z', v9('nan')))) and (v222['mcnemar_z'] > 1.645))
    v224 = v210.v242('ladder', {})
    v225 = v187(v224.v242('pairs', 0) >= 6 * v507.v367 and (not v635.v634(v224.v242('z_vs_half', v9('nan')))) and (v224['z_vs_half'] > 1.645))
    v226 = v187(v222.v242('underpowered', True))
    v227 = v210.v242('lookup_paired_vs_rival_cos_by_margin', {})
    v106, v107 = (v227.v242('low_margin', {}), v227.v242('high_margin', {}))
    v402, v403 = (v106.v242('mcnemar_z', v9('nan')), v107.v242('mcnemar_z', v9('nan')))
    v228 = v187(not v635.v634(v402) and (not v635.v634(v403)) and (v402 > 1.645) and (v403 < -1.645))
    v229 = 'NO_TASK' if not (v216 and v215 and v217) else 'DERIVATION_OK' if v220 and v221 and v223 and v225 else 'UNDERPOWERED_VS_RETRIEVAL' if v220 and v221 and (not v223) and v226 else 'PHI_HELPS_WHERE_SIMILARITY_RUNS_OUT' if v220 and v221 and (not v223) and v228 else 'PHI_ADDS_NOTHING_ON_LOOKUP' if v220 and v221 and (not v223) else 'DERIVATION_PARTIAL' if v220 or v225 else 'DERIVATION_NO'
    v70 = {'stage': '289', 'overall': v229, 'seed': v3, 'smoke': v161.v404, 'holdout': v161.v359, 'run_tag': v161.v405, 'train_steps': v166, 'params': v192, 'objective': 'expected_reward_280' if v17 == 'reward' else 'plackett_luce_ladder' if v24 else 'cross_entropy_no_ladder', 'edge_channels': v240(v28), 'import_k': v25, 'views': v20, 'reconciliation': v213, 'neighbours': v12, 'open_verb': v16, 'open_near_source': {'same_anchor': v18[0], 'neighbourhood': v18[1]} if v16 else None, 'graph_rows': {'mean': v37[0] / v311(1, v37[2]), 'max': v37[1], 'graphs': v37[2]}, 'neighbourhood_audit': v605(v48, v12) if v12 else None, 'nb_channels': {'anchor_nonzero_rate': v36[0] / v311(1, v36[2]), 'rel_nonzero_rate': v36[1] / v311(1, v36[2]), 'pairs': v36[2]} if v12 else None, 'refuse': v214, 'ink': v29, 'fp': v30, 'words': v31, 'write_ink': v161.v406, 'write_fp': v161.v407, 'write_words': v161.v408, 'fp_ngram': v161.v546 if v30 == 'hash' else None, 'tau': {'mode': v161.v480, 'value': v161.v360 if v161.v480 == 'absolute' else v188.v108.v242('tau'), 'target_density': v161.v500 if v161.v480 == 'density' else None, 'achieved_density': v188.v108.v242('achieved') if v161.v480 == 'density' else None, 'monotone': v188.v108.v242('monotone') if v161.v480 == 'density' else None, 'trace': v188.v108.v242('trace') if v161.v480 == 'density' else None}, 'tape_shape': {'held_out': v606(v195, v196), 'train': v606(v48, v193)}, 'resample': {'tape_period': v161.v547, 'mean_overlap': v235(v35) / v267(v35) if v35 else v9('nan'), 'n_resamples': v267(v35), 'note': "Jaccard between consecutive tapes' address sets. Near 1 means the redraw returns the same addresses and the anti-memorisation argument in HANDOFF 1 is decorative - the fix is a larger address pool, i.e. more corpus, not fewer parameters"}, 'row_dropout': {'rate': v161.v332, 'mean_kept_fraction': v19[0] / v19[1] if v19[1] else v9('nan'), 'note': 'training only - the held-out tape is never thinned. Marginalisation, not noise: a subset of the evidence is a world the corpus could have written, and the low/high margin split is a density axis the mind was never trained across'}, 'early_stop': {'enabled': not v161.v517, 'best_step': v120['step'], 'best_probe_loss': v120['loss'], 'total_steps': v166, 'probe_questions': v267(v199)}, 'probe_curve': v207, 'rare_nonzero_rate': v33[0] / v33[1] if v33[1] else v9('nan'), 'ink_degenerate_rate': v607[0] / v607[1] if v607[1] else v9('nan'), 'cos_mean': v38[0] / v38[2] if v38[2] else v9('nan'), 'cos_std': v635.v608(v311(0.0, v38[1] / v38[2] - (v38[0] / v38[2]) ** 2)) if v38[2] else v9('nan'), 'ladder_coverage_train': {'with_ladder': v194, 'lookup_questions': v267(v129.v242('lookup', ()))}, 'count_labels': v239(v5), 'compare_labels': v239(v6), 'gates': {'G_arc_enc_frozen': v215, 'G_ink_verified': v186, 'G_task_exists': v216, 'G_exact_algebra_matches_tape': v217, 'G_lookup_beats_floor': v220, 'G_lookup_beats_counts_paired': v221, 'G_lookup_beats_retrieval_paired': v223, 'G_phi_orders_negatives': v225}, 'held_out': v210, 'train_control': v211, 'exact_note': 'count and compare left the weights: they are functions of the same-value relation alone (new_i = 1 - max_{j<i} s_ij; count = sum new_i; compare = sign of the side difference), computed exactly with zero parameters and no 5+ cap, because the invariant says whatever decides may not be approximate. Their accuracy is 1.0 by construction and is checked, not celebrated - G_exact_algebra_matches_tape is a sanity bolt. The interference that cost count 0.965 -> 0.903 is removed by construction: one trained task remains', 'ladder_note': 'three wrong answers per question at increasing structural distance - same anchor / adjacent in tape order / anywhere on the tape - every rung a value the corpus wrote, no similarity chosen by anyone. Phi trained only against local wrong candidates learns a BOUNDARY; generation needs a LANDSCAPE, and a mind that cannot rank its own wrong answers by how wrong they are has no direction to move in. The objective is one Plackett-Luce term, not a task loss plus a ladder loss with a weight between them, and it reduces to the previous cross-entropy exactly when the tape cannot supply a ladder', 'retrieval_note': 'two rivals now, because they answer two different questions. The counting rival knows nothing about context, so beating it shows only that the context channel carries information counts lack - not that reading it takes a mind. rival_cos is 1-NN over the same evidence rows by the same ctx_fp cosine, zero parameters, no training. With hash ink the representation IS Random Indexing over fastText-shaped word vectors, so the distance between this architecture and a classical retrieval system is exactly this one number. If rival_cos lands where Phi lands, 3489 parameters are decoration ON THIS VERB and the verdict is PHI_ADDS_NOTHING_ON_LOOKUP. Named for the brick and not for the wall: lookup is one verb, single-hop and retrieval-shaped by construction, and a rival that ties it says nothing about the exact algebra, about verbs where rows must be combined, or about generation, which 1-NN cannot do at all', 'paired_note': 'the rival answers the same lookup questions in the same run, so the gate is McNemar over the discordant items at the usual one-sided 1.645 - never two marginals. The rival over survivors is Bayes-optimal when the query context carries nothing, so a paired win IS the claim that the context channel carries information counts do not have', 'curve': v372, 'arc_enc_hash_before': v176, 'arc_enc_hash_after': v209, 'fp_version': v579.v548(), 'note': "The derivation moved into exact algebra and the mind kept only the judgment. Two runs measured 7.9k parameters approximating a quantity exactly computable from their own input, and the approximation degraded as the genuinely uncertain task grew beside it. Now count and compare are arithmetic over the same-value relation - exact, uncapped, scale-free - and the one trained surface is Phi, the coherence of a completed world: for each candidate the query row is filled in and the world that results is pooled to one scalar, 288's repair loop turned inward. The two trained surfaces this leaves in the whole architecture are Phi and, once values stop being exact strings, s_ij itself - both judgments, never arithmetic. Confidence for exact verbs reports 1.0, which is the honest statement that a computed answer is certain GIVEN the relation; when s_ij becomes a judgment its uncertainty enters through that same seam.", 'timestamp': v659.v512(v660.v636).v549(), 'wall_s': v342.v342() - v165}
    v0.v233(parents=True, exist_ok=True)
    (v0 / f'stage289_decision{v164}.json').v339(v609.v550(v70, indent=2), encoding='utf-8')
    v105(v609.v550({'overall': v229, 'gates': v70['gates'], 'lookup': {v43: v92 for v43, v92 in v218.v127()}, 'paired': v219, 'ladder': v224}, indent=2))
    return 0
if v230 == '__main__':
    raise v409(v551())