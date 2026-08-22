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
v0 = v137('results')
v1 = v137('checkpoints/stage191_p1_curve.pt')
v2 = v137('data/_wikitext103_train.txt')
v3 = 2890
v4 = 5
v5 = v502((v8(v217) for v217 in v521(1, v4))) + (f'{v4}+',)
v6 = ('first', 'second', 'equal')
v7 = v0 / '_stage289_log.txt'

def log(v138: v8) -> None:
    v139 = v138 if v138.v862('\n') else v138 + '\n'
    try:
        v863(v139, end='', flush=True)
    except v503:
        v863(v139.v1292('ascii', 'replace').v1196('ascii'), end='', flush=True)
    v7.v864.v504(parents=True, exist_ok=True)
    with v7.v721('a', encoding='utf-8') as v505:
        v505.v865(v139)

def count_label(v140: v10) -> v8:
    return v5[v595(v140, v4) - 1]

def soft_new(v141, v142=None):
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
    v142 = v142 or (lambda v147, v148: 1.0 if v147 == v148 else 0.0)
    return [1.0 - (v576((v142(v141[v236], v141[v217]) for v236 in v521(v217))) if v217 else 0.0) for v217 in v521(v550(v141))]

def soft_count(v141, v142=None) -> v9:
    return v506(v866(v141, v142))

def exact_new(v141, v143=None):
    """The hard special case, kept because integers are what the examiner compares."""
    v142 = None if v143 is None else lambda v147, v148: 1.0 if v143(v147, v148) else 0.0
    return [v10(v944(v185)) for v185 in v866(v141, v142)]

def exact_count(v141, v143=None) -> v10:
    return v506(v867(v141, v143))

def exact_answer(v144):
    """The exact verdict for the exact verbs; raises on lookup, which is judged, not computed."""
    if v144['verb'] == 'count':
        return v868(v869(v144['vals']))
    if v144['verb'] == 'compare':
        v507 = v869(v144['vals'][:v144['n_first']])
        v508 = v869(v144['vals'][v144['n_first']:])
        return 'first' if v507 > v508 else 'second' if v508 > v507 else 'equal'
    raise v509('lookup is not exact: it is the judgment the mind is for')
v11 = ('lookup',)

def count_question(v145, v146):
    """How many distinct values does this address carry? The truth is a property of the tape."""
    v141 = [v145['tape'].v266[v282] for v282 in v146['slots']]
    if v550(v141) < 2:
        return None
    return {'verb': 'count', 'slots': v510(v146['slots']), 'vals': v141, 'label': v868(v550(v360(v141))), 'S': v146['S'], 'address': v146['address']}

def compare_question(v145, v147, v148):
    """Which of two addresses carries more distinct values?

    Both address's mentions go into ONE graph, with a side indicator per row. Nothing tells the
    mind how many rows each side has beyond what it can see, and the answer is not a count but
    an ordering, so the two verbs cannot share a shortcut: a mind that memorised "this many rows
    means this label" for COUNT gets nothing here, where both sides sit in the same graph.
    """
    v149 = [v145['tape'].v266[v282] for v282 in v147['slots']]
    v150 = [v145['tape'].v266[v282] for v282 in v148['slots']]
    if v550(v149) < 2 or v550(v150) < 2:
        return None
    v507, v508 = (v550(v360(v149)), v550(v360(v150)))
    v151 = 'first' if v507 > v508 else 'second' if v508 > v507 else 'equal'
    return {'verb': 'compare', 'slots': v510(v147['slots']) + v510(v148['slots']), 'vals': v149 + v150, 'n_first': v550(v149), 'label': v151, 'S': v147['S'], 'S2': v148['S'], 'address': v147['address'], 'address2': v148['address']}

def lookup_question(v145, v146, v152, v153=None):
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
    v154 = v510(v146['slots'])
    if v550(v154) < 3:
        return None
    v153 = v152.v870(v550(v154)) if v153 is None else v153
    v141 = [v145['tape'].v266[v282] for v282 in v154]
    v155 = v511(v360(v141[:v153] + v141[v153 + 1:]))
    if v550(v155) < 2:
        return None
    if v141[v153] not in v155:
        return None
    v156 = v154[:v153] + v154[v153 + 1:]
    return {'verb': 'lookup', 'slots': v156 + [v154[v153]], 'vals': [v141[v217] for v217 in v521(v550(v154)) if v217 != v153] + [v1102()], 'cands': v155, 'label': v155.v649(v141[v153]), 'S': v146['S'], 'address': v146['address'], 'hid': v153, 'query_row': v550(v156)}
v12 = 0
v13 = False
v14 = True
v15 = '\x00REFUSE'

def addr_parts(v157):
    """(anchor, relation) of an fp address, split exactly as pack_from_corpus splits it."""
    v158 = v157.v871(':', 1)[-1]
    v147, v301 = (v158.v871('|', 1) + [''])[:2]
    return (v147, v301)

def neighbourhood(v159, v160, v140, v161=('anchor', 'rel', 'word')):
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
    v162 = v159.v512('_nb', {})
    if (v160, v140, v161) in v162:
        return v162[v160, v140, v161]
    v163 = v159.v513('_addr_index')
    if v163 is None:
        v174, v872, v873, v226 = (v547(v510), v547(v510), v547(v510), {})
        v350 = v159.v513('_median')
        if v350 is None:
            v296 = v511((v550(v234) for v234 in v159['postings'].v266()))
            v350 = v296[v550(v296) // 2] if v296 else 1
            v159['_median'] = v350
        for v217, v222 in v549(v159['items']):
            v226[v222['address']] = v217
            v514, v515 = v516(v222['address'])
            v174[v514].v876(v222['address'])
            if v515:
                v872[v515].v876(v222['address'])
            v327 = v360()
            for v166 in v222['slots']:
                for v210 in v875(v159['texts'][v166], exclude=v159['tape'].v266[v166]):
                    if v550(v159['postings'].v513(v210, ())) < v350 and v210 not in v327:
                        v327.v1101(v210)
                        v873[v210].v876(v222['address'])
        v163 = v159['_addr_index'] = {'anchor': v174, 'rel': v872, 'word': v873, 'order': v226, 'slots': {v222['address']: v222['slots'] for v222 in v159['items']}}
    v514, v515 = v516(v160)
    v167, v327 = ([], {v160})
    v164 = v163['order'].v513(v160, 0)

    def take(v155, v216):
        for v209 in v155[:v216]:
            if v209 not in v327:
                v327.v1101(v209)
                v167.v876(v209)

    def near(v155):
        return v511((v147 for v147 in v155 if v147 != v160), key=lambda v147: (v1203(v163['order'][v147] - v164), v163['order'][v147]))
    if 'anchor' in v161:
        v874(v1086(v163['anchor'].v513(v514, ())), v140)
    if 'rel' in v161:
        v874(v1086(v163['rel'].v513(v515, ())), v140)
    v165 = v207()
    for v166 in v163['slots'].v513(v160, ()):
        for v210 in v875(v159['texts'][v166], exclude=v159['tape'].v266[v166]):
            if v550(v159['postings'].v513(v210, ())) < v159['_median']:
                for v147 in v163['word'].v513(v210, ()):
                    if v147 != v160:
                        v165[v147] += 1
    if 'word' in v161:
        v874([v147 for v147, v180 in v511(v165.v219(), key=lambda v1341: (-v1341[1], v163['order'][v1341[0]]))], v140)
    v162[v160, v140, v161] = v167
    return v167

def lookup_sparse_question(v159, v146, v152, v153, v140, v161=('anchor', 'rel', 'word')):
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
    v168 = v510(v146['slots'])
    if not 1 <= v550(v168) <= 2:
        return None
    v153 = v153 % v550(v168)
    v169 = v168[v153]
    v170 = v159['tape'].v266[v169]
    v171 = [v282 for v282 in v168 if v282 != v169]
    for v148 in v517(v159, v146['address'], v140, v161):
        v171 += v510(v159['_addr_index']['slots'].v513(v148, ()))[:v140]
    v171 = v511(v360(v171) - {v169})
    if not v171:
        return None
    v155 = v511({v159['tape'].v266[v282] for v282 in v171})
    if v550(v155) < 2:
        return None
    v172 = v170 in v155
    if not v172 and (not v13):
        return None
    if v13:
        v155 = v155 + [v15]
    v173 = v155.v649(v170) if v172 else v155.v649(v15) if v13 else None
    if v173 is None:
        return None
    return {'verb': 'lookup', 'sparse': True, 'answerable': v172, 'slots': v171 + [v169], 'vals': [v159['tape'].v266[v282] for v282 in v171] + [v1102()], 'cands': v155, 'label': v173, 'S': v146['S'], 'address': v146['address'], 'hid': v153, 'own_rows': {v282 for v282 in v168 if v282 != v169}, 'query_row': v550(v171)}
v16 = False
v17 = 'ce'
v18 = [0, 0]

def lookup_open_question(v159, v146, v152, v153, v174, v175):
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
    v168 = v510(v146['slots'])
    if v550(v168) < 2:
        return None
    v153 = v153 % v550(v168)
    v169 = v168[v153]
    v170 = v159['tape'].v266[v169]
    v171 = [v282 for v282 in v168 if v282 != v169]
    if v518((v159['tape'].v266[v282] == v170 for v282 in v171)):
        return None
    v176 = {'cands': [v170], 'address': v146['address'], 'slots': v168, 'S': v146['S'], 'query_row': v550(v171)}
    v519(v159, v176, v174, v175, v152)
    v177 = v439(v176.v513('ladder') or {})
    if 'near' not in v177:
        v368 = {v170} | v360(v177.v266())
        for v148 in v517(v159, v146['address'], 3):
            v644 = [v159['tape'].v266[v166] for v166 in v159['_addr_index']['slots'].v513(v148, ())]
            v644 = [v185 for v185 in v644 if v185 not in v368]
            if v644:
                v177['near'] = v644[0]
                break
    if v550(v177) != 3:
        return None
    v18[0 if v176.v513('ladder', {}).v513('near') == v177['near'] else 1] += 1
    v155 = v511([v170] + [v177[v301] for v301 in v120])
    if v550(v155) != 4:
        return None
    v144 = {'verb': 'lookup', 'open': True, 'slots': v171 + [v169], 'vals': [v159['tape'].v266[v282] for v282 in v171] + [v1102()], 'cands': v155, 'label': v155.v649(v170), 'rung_of': {v177[v301]: v301 for v301 in v120}, 'S': v146['S'], 'address': v146['address'], 'hid': v153, 'query_row': v550(v171)}
    if v540(v159, v144, v510(v144['cands'])) < 1:
        return None
    return v144
v19 = 'fp'
v20 = 'ladder'
v21 = 4
v22 = 8

def anchor_items(v159):
    """Addresses made of one exact string. The grouping rule is not loosened here, it is absent:
    two mentions share an address when the corpus wrote the same anchor, and never otherwise."""
    v178 = v520(v159)
    if v178 is None:
        return []
    return [{'S': v181, 'address': f'anc:{v181}', 'slots': v510(v154), 'kind': 'clean'} for v181, v154 in v511(v178['by_anc'].v219()) if v550(v154) >= 2]

def lookup_open_uniform(v159, v146, v152, v153, v175):
    """292's question, with the wrong answers drawn instead of constructed.

    Everything that made 292 legitimate is kept: the hidden value occurs exactly once at the
    address so it is foreign to every evidence row, and shared_import_budget gives all four
    worlds the same number of imported rows. What goes is the ladder - the three distractors are
    any values the address does not carry, so nothing about a candidate's relation to this
    address says whether it is the answer. The distance is recorded per candidate and read off
    the same logits afterwards, which is the only way to ask whether the inversion is real.
    """
    v168 = v510(v146['slots'])
    if v550(v168) < 2:
        return None
    v153 = v153 % v550(v168)
    v179 = v168[v153]
    v170 = v159['tape'].v266[v179]
    v171 = [v282 for v282 in v168 if v282 != v179]
    if v518((v159['tape'].v266[v282] == v170 for v282 in v171)):
        return None
    if v550(v171) > v22:
        v171 = v511(v511(v171, key=lambda v282: (v1203(v282 - v179), v282))[:v22])
    v164 = {v159['tape'].v266[v282] for v282 in v168}
    v155 = [v170]
    for v180 in v521(64 * v21):
        if v550(v155) == v21:
            break
        v234 = v175[v152.v870(v550(v175))]
        if v234 not in v164 and v234 not in v155:
            v155.v876(v234)
    if v550(v155) != v21:
        return None
    v155 = v511(v155)
    v181 = v146['S']
    v178 = v520(v159)
    v182 = {v159['tape'].v266[v282] for v282 in v178['by_anc'].v513(v181, ())} if v178 else v360()
    v144 = {'verb': 'lookup', 'open': True, 'uniform': True, 'slots': v171 + [v179], 'vals': [v159['tape'].v266[v282] for v282 in v171] + [v1102()], 'cands': v155, 'label': v155.v649(v170), 'bucket_of': {v209: 'same_anchor' if v209 in v182 else 'elsewhere' for v209 in v155 if v209 != v170}, 'S': v181, 'address': v146['address'], 'hid': v153, 'query_row': v550(v171)}
    if v540(v159, v144, v510(v144['cands'])) < 1:
        return None
    return v144
v23 = False
v24 = 2

def pattern_stats(v159, v156=None):
    """Exact co-occurrence counts over anchors. `keep` restricts to an anchor subset (the
    internal split), so train-side labels never touch the anchors the worlds are built from."""
    v178 = v520(v159)
    if v178 is None:
        return None
    v183 = {v147: {v159['tape'].v266[v282] for v282 in v166} for v147, v166 in v178['by_anc'].v219() if v156 is None or v156(v147)}
    v385, v522 = (v207(), v207())
    for v147, v523 in v183.v219():
        for v234 in v523:
            v385[v234] += 1
        for v185 in v523:
            for v186 in v523:
                if v185 < v186:
                    v522[v185, v186] += 1
    return {'av': v183, 'cnt': v385, 'pair': v522, 'N': v576(1, v550(v183)), 'ix': v178}

def pattern_lift(v184, v185, v186):
    """N * P(xy) / (P(x) P(y)). None when either value never occurs on this side - no label."""
    if v184['cnt'][v185] == 0 or v184['cnt'][v186] == 0:
        return None
    return v184['pair'].v513((v185, v186), 0) * v184['N'] / (v184['cnt'][v185] * v184['cnt'][v186])

def pattern_rules(v184):
    """Every pair witnessed by at least PAT_W anchors: enough rows for a fixed-size world."""
    return [v524 for v524, v1087 in v511(v184['pair'].v219()) if v1087 >= v24]

def pattern_world(v159, v184, v185, v186):
    """The rule as rows: from the first PAT_W witness anchors (lexicographic - independent of
    any statistic), one x-row and one y-row each. 2*PAT_W rows for every rule, so the row count
    cannot carry the label; no query row; values visible - the same-value edges ARE the shape
    the rule claims. Phi sees this world and nothing else: no support, no confidence, no lift."""
    v187 = v511((v147 for v147, v523 in v184['av'].v219() if v185 in v523 and v186 in v523))[:v24]
    v154 = []
    for v147 in v187:
        v166 = v184['ix']['by_anc'][v147]
        v154.v876(v595((v282 for v282 in v166 if v159['tape'].v266[v282] == v185)))
        v154.v876(v595((v282 for v282 in v166 if v159['tape'].v266[v282] == v186)))
    v154 = v511(v360(v154))
    if v550(v154) != 2 * v24:
        return None
    return {'verb': 'lookup', 'slots': v154, 'vals': [v159['tape'].v266[v282] for v282 in v154], 'S': v185, 'query_row': -1, 'cands': [v185, v186], 'label': 0}

def run_patterns(v159, v188, v189, v190, v191):
    """Mine on train, judge with Phi, label on held, race the rule's own train statistics."""
    import torch.nn.functional as Fn
    v363(f'  295 patterns: witnesses {v24}')

    def side(v147):
        return v10.v1088(v1291.v1321(v147.v1292(), digest_size=2).v1197(), 'big') % 2
    v192 = v525(v159, keep=lambda v147: v775(v147) == 0)
    v193 = v525(v159, keep=lambda v147: v775(v147) == 1)
    v526, v527 = (v525(v159), v525(v188))
    if not (v192 and v193 and v526 and v527):
        v363('  295 needs straddr packs')
        return 1

    def labelled(v528, v529, v530):
        v167 = []
        for v185, v186 in v877(v528):
            v878 = v1089(v529, v185, v186)
            if v878 is None:
                continue
            v210 = v1090(v530, v528, v185, v186)
            if v210 is not None:
                v167.v876((v210, v10(v878 > 1.0), v1089(v528, v185, v186)))
        return v167
    v194 = v531(v192, v193, v159)
    v195 = v531(v526, v527, v159)
    v363(f'  rules: train {v550(v194)} eval {v550(v195)} | base survive: train {v506((v1014 for v180, v1014, v180 in v194)) / v576(1, v550(v194)):.3f} eval {v506((v1014 for v180, v1014, v180 in v195)) / v576(1, v550(v195)):.3f}')
    if not v194 or not v195:
        v363('  295: empty rule set - need more addresses or lower co-occurrence bar')
        return 1
    v196 = v532(v190, d=v191.v805, n_node=8 + (1 if v26 == 'frames' else 0) + (1 if v88 else 0) + (3 if v65 else 0))
    v197 = v706.v879.v533(v196.v880(), lr=v191.v881, weight_decay=0.01)
    v152 = v882.v534(v3)
    v198 = v191.v535 or 3000
    for v199 in v521(v198):
        v210, v151, v180 = v194[v152.v870(v550(v194))]
        v384, v883, v389 = v539(v159, v210, v189, v190)
        v402 = v1091.v884(v196.v542(v384, v883, v389), v706.v601(v9(v151), device=v190))
        v197.v885()
        v402.v886()
        v197.v199()
    v155 = v511({v878 for v180, v180, v878 in v194})
    v200 = v576(v155, key=lambda v250: v506((v10(v878 >= v250) == v1014 for v180, v1014, v878 in v194))) if v155 else 1.0
    v201 = v202 = v203 = v204 = 0
    with v706.v404():
        for v210, v151, v878 in v195:
            v384, v883, v389 = v539(v159, v210, v189, v190)
            v138 = v10(v9(v196.v542(v384, v883, v389)) > 0.0)
            v301 = v10(v878 >= v200)
            v203 += v10(v138 == v151)
            v204 += v10(v301 == v151)
            v201 += v10(v138 == v151 and v301 != v151)
            v202 += v10(v301 == v151 and v138 != v151)
    v205 = v201 + v202
    v167 = {'stage': '295', 'witnesses': v24, 'rules_train': v550(v194), 'rules_eval': v550(v195), 'rival_threshold': v200, 'base_survive': v506((v1014 for v180, v1014, v180 in v195)) / v576(1, v550(v195)), 'phi_accuracy': v203 / v576(1, v550(v195)), 'rival_accuracy': v204 / v576(1, v550(v195)), 'paired': {'model_only_right': v201, 'rival_only_right': v202, 'discordant': v205, 'mcnemar_z': (v201 - v202) / v1097.v1170(v205) if v205 else v9('nan'), 'max_achievable_z': v1097.v1170(v205) if v205 else 0.0}, 'seed': v3, 'timestamp': v1276.v1035(v1277.v1198).v887()}
    v363(f'  295 {v1084.v859(v167)}')
    v206 = '_' + v191.v536 if v191.v536 else ''
    (v0 / f'stage289_patterns{v206}.json').v537(v1084.v859(v167, indent=2), encoding='utf-8')
    return 0
v25 = False
v26 = 'parser'
v27 = False
v28 = 0.05
v29 = 1.0
v30 = False
v31 = False
v32 = 6
v33 = 8
v34 = 'max'
v35 = 'count'
v36 = 0
v37 = 1.0
v38 = None
v39 = 0
v40 = 1.0
v41 = None
v42 = None
v43 = 0
v44 = 'random'
v45 = None
v46 = False
v47 = 0
v48 = 'mind'
v49 = 'cos'
v50 = 1
v51 = False
v52 = False
v53 = 0.0
v54 = False
v55 = False
v56 = 12
v57 = False
v58 = False
v59 = [0, 0]
v60 = False
v61 = 4000
v62 = False
v63 = 4
v64 = False
v65 = False
v66 = False
v67 = ('step', 'share', 'lines')
v68 = 0.0
v69 = v207()
v70 = 'all'
v71 = v207()
v72 = ('step', 'share', 'lines')
v73 = 2

def expanded_world(v159, v144, v189, v190, v140):
    """What the mind would be looking at if it read on: the question's rows plus everything the
    tape offers about every option. Candidate-independent on purpose - it is ONE world, so Phi
    can score `read more` with the same scalar it scores an answer with, and the route needs no
    second head and no policy network."""
    v208 = []
    for v209 in v144['cands']:
        if v209 == v15:
            continue
        v208 += v897(v159, v144, v209)[:v140]
    v210 = v439(v144)
    v210.v538('_base', None)
    v210.v538('_ibudget', None)
    v210['slots'] = v510(v144['slots']) + v208
    v210['vals'] = v510(v144['vals']) + [v159['tape'].v266[v282] for v282 in v208]
    return v539(v159, v210, v189, v190, query_value=None, import_k=0)

def route_logits(v196, v159, v144, v190, v189):
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
    v140 = v540(v159, v144, v510(v144['cands']))
    v211 = v706.v541([v196.v542(*v539(v159, v144, v189, v190, query_value=v209, import_k=0)) for v209 in v144['cands']])
    v144.v538('_base', None)
    v212 = v196.v542(*v1092(v159, v144, v189, v190, v140))
    v144.v538('_base', None)
    v213 = v706.v541([v196.v542(*v539(v159, v144, v189, v190, query_value=v209, import_k=v140)) for v209 in v144['cands']])
    return (v706.v705([v211, v212.v1099(1)]), v213)

def route_reward(v144, v190):
    v214 = v706.v543((v550(v144['cands']),), -1.0, device=v190)
    v214[v144['label']] = 1.0
    if v144.v513('answerable') and v15 in v144['cands']:
        v214[v144['cands'].v649(v15)] = 0.75
    return v214

def route_loss(v196, v159, v144, v190, v189):
    """Expected payoff of the whole route, in closed form: answer now, or pay the step price and
    answer from what reading brought. No baseline, no sampling, no RL machinery."""
    v211, v213 = v544(v196, v159, v144, v190, v189)
    v214 = v545(v144, v190)
    v288, v289 = (v706.v608(v211, 0), v706.v608(v213, 0))
    v215 = (v289 * v214).v506() - v28
    return -((v288[:-1] * v214).v506() + v288[-1] * v215)
v74 = [0]
v75 = False
v76 = 8
v77 = 8
v78 = 2000
v79 = 12
v80 = 'both'
v81 = ('answerable', 'silent', 'mind_right', 'rival_margin', 'rival_right', 'stepped', 'n_cands', 'rows_candidate', 'rows_expand', 'reachable_wide', 'reachable_random', 'truth_in_own', 'own_rival_right', 'n_own', 'max_own_count', 'n_places', 'line_reach', 'line_rival', 'step_line', 'world_rows_own', 'count_rival_right', 'top_share', 'bisect_right', 'bisect_splits', 'depth_reached', 'deep_only', 'other_right', 'other_stepped', 'pick_score', 'pick_margin', 'cr_ties', 'move_id')
v82 = {v216: v217 for v217, v216 in v549(v81)}
v83 = False
v84 = 3
v85 = 8
v86 = 0
v87 = False
v88 = False
v89 = 'walk'
v90 = 'uniform'
v91 = 'address'
v92 = False
v93 = False
v94 = 4

def reach_index(v159):
    """Every address as one frame fingerprint, stacked once per pack."""
    v178 = v159.v513('_reach')
    if v178 is not None:
        return v178
    v218 = v159.v513('frame_fps')
    v219 = [v222 for v222 in v159['items'] if v222['slots']]
    if v218 is None or not v219:
        return None
    v220 = v706.v541([v218[v222['slots'][0]] for v222 in v219])
    v221 = []
    for v222 in v219:
        v327, v226 = ({}, [])
        for v166 in v222['slots']:
            v234 = v159['tape'].v266[v166]
            if v234 not in v327:
                v327[v234] = [[], 0]
                v226.v876(v234)
            v561 = v327[v234]
            v561[1] += 1
            if v550(v561[0]) < v94:
                v561[0].v876(v166)
        v221.v876([(v234, v327[v234][0], v327[v234][1]) for v234 in v226])
    v245, v546 = ({}, v207())
    if v88:
        for v166, v554 in v549(v159['straddr']):
            v234 = v159['tape'].v266[v166]
            v505 = v218[v166]
            v245[v234] = v505.v557() if v234 not in v245 else v245[v234] + v505
            v546[v234] += 1
    v223 = v547(v510)
    for v236, v548 in v549(v221):
        for v234, v888, v209 in v548:
            v223[v234].v876((v236, v209))
    v178 = {'items': v219, 'M': v220, 'fills': v221, 'home_sum': v245, 'home_n': v546, 'by_val': v439(v223), 'of': {v222['address']: v217 for v217, v222 in v549(v219)}}
    v159['_reach'] = v178
    return v178

def reach_question(v159, v146, v152, v153):
    """A hidden filler and the rows of its own place. No candidates: those are walked to."""
    v168 = v510(v146['slots'])
    if v550(v168) < 2:
        return None
    v153 %= v550(v168)
    v224 = v168[v153]
    v171 = [v282 for v282 in v168 if v282 != v224]
    if v550(v171) > v79 - 1:
        v171 = v511(v152.v892(v171, v79 - 1))
    return {'verb': 'reach', 'reach': True, 'S': v146['S'], 'address': v146['address'], 'hid': v153, 'slots': v171 + [v224], 'vals': [v159['tape'].v266[v282] for v282 in v171] + [v1102()], 'query_row': v550(v171), 'truth_value': v159['tape'].v266[v224]}

def retain_keep(v159):
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
    if not v43 or v46:
        return None
    v138 = v159.v513('_retain')
    if v138 is not None:
        return v138
    v178 = v551(v159)
    if v178 is None:
        return None
    v219 = v178['items']
    v225 = v550(v219)
    if v43 >= v225:
        v159['_retain'] = v706.v889(v225, dtype=v706.v449)
        return v159['_retain']
    v152 = v882.v534(v3 + 9338)
    if v44 == 'random':
        v552 = [v152.v882() for v180 in v521(v225)]
    elif v44 == 'own':
        v552 = [v9(v550(v222['slots'])) for v222 in v219]
    elif v44 == 'share':
        v552 = []
        for v548 in v178['fills']:
            v613 = v506((v209 for v1204, v1202, v209 in v548))
            v552.v876(v576((v209 for v1204, v1202, v209 in v548), default=0) / v576(1, v613))
    else:
        if v45 is None:
            raise v1095('--retain-by mind: the walk ran before the mind existed')
        v196, v190, v189 = v45
        v46 = True
        try:
            v552 = []
            v352 = v882.v534(v3 + 9339)
            with v706.v404():
                for v222 in v219:
                    v144 = v621(v159, v222, v352, 0)
                    if v144 is None:
                        v552.v876(v9('-inf'))
                        continue
                    v211, v213, v168, v155, v264, v302 = v602(v196, v159, v144, v190, v189)
                    v1199, v920, v1332, v1333, v922 = v923(v144, v211, v213, v168, v155, v264, v302)
                    v552.v876(v922)
        finally:
            v46 = False
    v226 = v511(v521(v225), key=lambda v217: (-v552[v217], v217))[:v43]
    v138 = v706.v553(v225, dtype=v706.v449)
    v138[v706.v601(v226, dtype=v706.v1093)] = True
    v159['_retain'] = v138
    return v138

def reach_places(v159, v144, v140):
    """The k nearest places by frame fingerprint, in order, this one excluded."""
    v178 = v551(v159)
    v217 = v178['of'].v513(v144['address']) if v178 else None
    if v217 is None:
        return []
    v227 = v178['M'][v217]
    if v91 == 'fillers' and v159.v513('frame_sum') is not None:
        v554 = v144['address']
        v555 = v159['frame_cnt'].v513(v554, 0)
        v556 = v159['val_fp'].v513(v144['truth_value'])
        if v555 > 1 and v556 is not None:
            v227 = v1119.v1094(v159['frame_sum'][v554] - v556, dim=-1)
    v144['_qv'] = v227
    v228 = (v178['M'] @ v227).v557()
    v228[v217] = -2.0
    v229 = v558(v159)
    if v229 is not None:
        v228 = v228.v890(~v229.v702(v228.v190), -2.0)
    v226 = [v236 for v236 in v228.v975(descending=True)[:v140].v590() if v228[v236] > -2.0]
    v167 = [(v236, v178['items'][v236], v9(v228[v236])) for v236 in v226]
    if v49 == 'cos':
        return v167
    v230 = v207((v159['tape'].v266[v282] for v282 in v144['slots'][:v144['query_row']]))
    v231 = v207()
    if v49 in ('share1', 'rare', 'common', 'cover', 'jaccard'):
        v559 = v550(v230)
        for v234, v385 in v230.v219():
            v165 = v178['by_val'].v513(v234, ())
            v613 = v506((v209 for v1334, v209 in v165)) or 1
            for v236, v209 in v165:
                if v236 == v217 or (v229 is not None and (not v449(v229[v236]))):
                    continue
                if v49 == 'share1':
                    v231[v236] += 1
                elif v49 == 'rare':
                    v231[v236] += 1.0 / v613
                elif v49 == 'common':
                    v231[v236] += v9(v613)
                elif v49 == 'cover':
                    v231[v236] += 1.0 / v576(1, v550(v178['fills'][v236]))
                else:
                    v231[v236] += 1.0 / v576(1, v559 + v550(v178['fills'][v236]) - 1)
    else:
        for v234, v385 in v230.v219():
            for v236, v209 in v178['by_val'].v513(v234, ()):
                if v236 != v217 and (v229 is None or v449(v229[v236])):
                    v231[v236] += v595(v385, v209)
    v232 = [(v236, v178['items'][v236], v9(v216)) for v236, v216 in v231.v562(v140)]
    if v49 != 'both':
        return v232
    v327, v410 = (v360(), [])
    for v147, v148 in v560(v167, v232):
        for v561 in (v147, v148):
            if v561 is not None and v561[0] not in v327:
                v327.v1101(v561[0])
                v410.v876(v561)
    return v410[:2 * v140]

def reach_connect(v159, v144, v140):
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
    v178 = v551(v159)
    v217 = v178['of'].v513(v144['address']) if v178 else None
    if v217 is None:
        return []
    v168 = v207((v159['tape'].v266[v282] for v282 in v144['slots'][:v144['query_row']]))
    v229 = v558(v159)
    v233 = v207()
    for v234 in v168:
        for v236, v891 in v178['by_val'].v513(v234, ()):
            if v236 != v217 and (v229 is None or v449(v229[v236])):
                v233[v236] += 1
    if not v233:
        return []
    v318, v240 = (v207(), {})
    for v236, v370 in v233.v562(v61):
        for v234, v171, v891 in v178['fills'][v236]:
            if v234 in v168:
                continue
            v318[v234] += v370
            if v550(v240.v512(v234, [])) < v94:
                v240[v234].v937(v171[:v94 - v550(v240[v234])])
    return [(v234, v240.v513(v234, []), v216) for v234, v216 in v318.v562(v140)]

def reach_reachable(v159, v144, v140, v235=None):
    """Would the true filler be sayable if the walk went `k` places - or `k` random ones?

    Set membership only, no graphs, so the three variants cost nothing next to the run. The
    random one is the decisive one: if a cosine walk reaches the truth no more often than an
    arbitrary handful of places does, there is no direction in this ink and the verb is
    measuring a lottery. That has to be known before any payoff is read, not after.
    """
    v178 = v551(v159)
    if v178 is None:
        return False
    if v235 is None:
        v237 = [v236 for v236, v565, v1199 in v563(v159, v144, v140)]
    else:
        v229 = v558(v159)
        v317 = [v236 for v236, v222 in v549(v178['items']) if v222['address'] != v144['address'] and (v229 is None or v449(v229[v236]))]
        v237 = v235.v892(v317, v595(v140, v550(v317)))
    v327, v167 = (v360(), [])
    for v236 in v237:
        for v234, v888, v891 in v178['fills'][v236]:
            if v234 not in v327:
                v327.v1101(v234)
                v167.v876(v234)
    return v144['truth_value'] in v360(v167[:v77])

def reach_candidates(v159, v144, v140=None, v238=None):
    """What the walk makes sayable: fillers in place order, deduped, capped by the cost bound.

    Cached on the question so the walk is one object and the rival, the mind and the report all
    grade the SAME traversal rather than three re-derivations of it.

    385: `which` names ONE move's lane instead of the merged offer, and is passed only while the
    mind is choosing between moves - those calls are not cached, because they are proposals and
    not the question's offer. Once the move is chosen it is written to q["_move"] and the
    ordinary cached call builds that lane and only it.
    """
    if v238 is None and '_reach_c' in v144:
        return v144['_reach_c']
    if v238 is None:
        v238 = v144.v513('_move', 'all')
        if v66 and '_move' not in v144:
            raise v1095('385: reach_candidates called before the move was chosen')
    v178 = v551(v159)
    v239 = v563(v159, v144, v76 if v140 is None else v140)
    v327, v155, v240, v564 = (v360(), [], {}, {})
    for v236, v565, v566 in v239:
        for v234, v171, v891 in v178['fills'][v236]:
            v240.v512(v234, []).v937(v171)
            if v234 not in v327:
                v327.v1101(v234)
                v155.v876(v234)
                v564[v234] = v236
    if v238 != 'all':
        if v238 == 'step':
            v567, v568 = ([], [])
        elif v238 == 'share':
            v567, v568 = (v1200(v159, v144, v77), [])
            v155 = [v234 for v234, v1202, v612 in v567]
        elif v238 == 'lines':
            v567, v568 = ([], v1201(v159, v144, v77))
            v155 = [v234 for v234, v1202, v612 in v568]
        else:
            raise v1095(f'385: unknown move {v238!r}')
        for v234, v171, v612 in v567:
            v240[v234], v564[v234] = (v510(v171), -1)
        for v234, v171, v612 in v568:
            v240[v234], v564[v234] = (v510(v171), -3)
    elif v60 or v57 or v62:
        v893 = [v155]
        if v60:
            v567 = v1200(v159, v144, v77)
            v893.v876([v234 for v234, v1202, v612 in v567])
        else:
            v567 = []
        v568 = v1201(v159, v144, v77) if v62 else []
        if v568 and (not v64):
            v893.v876([v234 for v234, v1202, v612 in v568])
        v894 = v511({v159['tape'].v266[v282] for v282 in v144['slots'][:v144['query_row']]})
        if v57:
            v893.v876(v894)
        v1096, v410 = (v360(), [])
        for v895 in v560(*v893):
            for v561 in v895:
                if v561 is not None and v561 not in v1096:
                    v1096.v1101(v561)
                    v410.v876(v561)
        if v568 and v64:
            for v234, v888, v612 in v568:
                if v550(v410) >= v77:
                    break
                if v234 not in v1096:
                    v1096.v1101(v234)
                    v410.v876(v234)
        for v234, v171, v612 in v567:
            if v234 in v1096 and v234 not in v240:
                v240[v234] = v510(v171)
                v564[v234] = -1
        for v234, v171, v612 in v568:
            if v234 in v1096 and v234 not in v240:
                v240[v234] = v510(v171)
                v564[v234] = -3
        if v57:
            for v234 in v894:
                if v234 in v1096 and v234 not in v240:
                    v240[v234] = v510(v897(v159, v144, v234))
                    v564[v234] = -2
        v155 = v410
    v155 = v155[:v77]
    v240 = {v209: v240[v209] for v209 in v155}
    v241 = {v209: v564[v209] if v564[v209] >= 0 else v591(v159, v144, v240[v209]) for v209 in v155}
    v167 = {'cands': v155, 'rows_of': v240, 'places': v239, 'n_places': v550({v236 for v236 in (v241[v209] for v209 in v155) if v236 is not None}), 'from_place': {v209: v564[v209] for v209 in v155}, 'real_place': v241, 'own': v511({v159['tape'].v266[v282] for v282 in v144['slots'][:v144['query_row']]})}
    if v238 == v144.v513('_move', 'all'):
        v144['_reach_c'] = v167
    return v167

def reach_channel(v159, v144, v242):
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
    if not v65:
        return (0.0, 0.0, 0.0)
    v243 = v144.v513('_reach_c')
    v236 = v243['from_place'].v513(v242) if v243 else None
    if v236 is None or v236 >= 0:
        return (0.0, 0.0, 0.0)
    return {-1: (1.0, 0.0, 0.0), -2: (0.0, 1.0, 0.0), -3: (0.0, 0.0, 1.0)}[v236]

def channel_feat(v144, v217, v244):
    """the three indicators as a node-vector tail, carried only by the row the world answers -
    the channel is a property of what the world asserts, not of the evidence around it."""
    if not v65:
        return []
    return v510(v144.v513('channel') or (0.0, 0.0, 0.0)) if v217 == v244 else [0.0, 0.0, 0.0]

def reach_home_cos(v159, v144, v242, v227):
    """cos(where this value usually stands, this place) - with this place taken out of the
    average for EVERY candidate, so the subtraction carries no news about which was hidden."""
    v178 = v159.v513('_reach')
    if not v88 or v178 is None or (not v178['home_sum']) or (v227 is None):
        return 0.0
    v245 = v178['home_sum'].v513(v242)
    if v245 is None:
        return 0.0
    v164 = v178['of'].v513(v144['address'])
    if v164 is None:
        return 0.0
    v246 = v506((v209 for v234, v1202, v209 in v178['fills'][v164] if v234 == v242))
    v216 = v178['home_n'][v242] - v246
    if v216 <= 0:
        return 0.0
    v247 = v245 - v246 * v178['M'][v164]
    return v9(v1119.v1094(v247, dim=-1) @ v227)

def confirm_index(v159):
    """Rare words per line, counted once per pack. Rarity is a document frequency - a count."""
    v178 = v159.v513('_conf')
    if v178 is not None:
        return v178
    v248 = {}
    for v166, v569 in v549(v159.v513('line') or ()):
        if v569 >= 0 and v569 not in v248:
            v248[v569] = v159['texts'][v166]
    v249 = v207()
    for v250 in v248.v266():
        for v210 in v360(v250.v871()):
            v249[v210] += 1
    v178 = {v569: v896((v210 for v210 in v360(v250.v871()) if v249[v210] <= v84)) for v569, v250 in v248.v219()}
    v159['_conf'] = v178
    return v178

def reach_confirm(v159, v144, v242):
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
    if not v83 or not v159.v513('line'):
        return 0.0
    v178 = v570(v159)
    v251 = v159['line'][v144['slots'][v144['query_row']]]
    v164 = v178.v513(v251)
    if not v164:
        return 0.0
    v252 = 0
    for v166 in v897(v159, v144, v242)[:v85]:
        v569 = v159['line'][v166]
        if v569 < 0 or v569 == v251:
            continue
        for v205 in v521(-v86, v86 + 1):
            v332 = v178.v513(v569 + v205)
            if v332:
                v252 = v576(v252, v550(v164 & v332))
    return v1097.v898(v252) / v1097.v898(8.0)

def reach_line_index(v159):
    """Slots by line, once per pack."""
    v178 = v159.v513('_lineix')
    if v178 is None:
        v178 = v547(v510)
        for v166, v569 in v549(v159.v513('line') or ()):
            if v569 >= 0:
                v178[v569].v876(v166)
        v178 = v439(v178)
        v159['_lineix'] = v178
    return v178

def reach_line_candidates(v159, v144):
    """What the OTHER frames of this hole's sentence offer, in line order, deduped and capped.

    The hidden row's own place is excluded, so nothing here is the question's own evidence, and
    positions partition the corpus so a sibling is always another word.
    """
    if '_line_c' in v144:
        return v144['_line_c']
    v155, v240 = ([], {})
    v253 = v159.v513('line')
    if v253:
        v224 = v144['slots'][v144['query_row']]
        for v166 in v1103(v159).v513(v253[v224], ()):
            if v159['straddr'][v166] == v144['address'] or v166 in v360(v144['slots']):
                continue
            v234 = v159['tape'].v266[v166]
            if v234 not in v240:
                if v550(v155) >= v77:
                    continue
                v240[v234] = []
                v155.v876(v234)
            if v550(v240[v234]) < v94:
                v240[v234].v876(v166)
    v167 = {'cands': v155, 'rows_of': v240}
    v144['_line_c'] = v167
    return v167

def copy_index(v159):
    """line -> its text, once per pack. `texts` is the whole line, so this reaches every token
    of a neighbouring sentence and not only the ones the frame cutter kept."""
    v178 = v159.v513('_copyix')
    if v178 is None:
        v178 = {}
        for v166, v569 in v549(v159.v513('line') or ()):
            if v569 >= 0 and v569 not in v178:
                v178[v569] = v159['texts'][v166]
        v159['_copyix'] = v178
    return v178

def reach_copy(v159, v144, v140):
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
    if not v62 or not v159.v513('line'):
        return []
    v178 = v571(v159)
    v251 = v159['line'][v144['slots'][v144['query_row']]]
    if v251 < 0:
        return []
    v254 = v572(v159)
    v168 = {v159['tape'].v266[v282] for v282 in v144['slots'][:v144['query_row']]}
    v318, v573 = (v207(), {})
    for v255 in v521(-v63, v63 + 1):
        if v255 == 0:
            continue
        v574 = v178.v513(v251 + v255)
        if not v574:
            continue
        for v210 in v574.v871():
            if v210 in v168 or v210 not in v254:
                continue
            v318[v210] += 1
            if v1203(v255) < v573.v513(v210, v63 + 1):
                v573[v210] = v1203(v255)
    if not v318:
        return []
    v167 = []
    for v210 in v511(v318, key=lambda v234: (-v318[v234], v573[v234], v234)):
        v171 = v897(v159, v144, v210)
        if not v171:
            continue
        v167.v876((v210, v171, v318[v210]))
        if v550(v167) >= v140:
            break
    return v167

def reach_line_rival(v159, v144):
    """The same sentence read by counting: its most frequent sibling filler."""
    v256 = v575(v159, v144)
    if not v256['cands']:
        return None
    return v576(v256['cands'], key=lambda v234: (v550(v256['rows_of'][v234]), -v256['cands'].v649(v234)))

def reach_relation_rows(v159, v144, v242):
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
    v178 = v551(v159)
    if v178 is None:
        return []
    v217 = v178['of'].v513(v144['address'])
    v168 = {v159['tape'].v266[v282] for v282 in v144['slots'][:v144['query_row']]}
    v229 = v558(v159)
    v233 = v207()
    for v234 in v168:
        for v236, v891 in v178['by_val'].v513(v234, ()):
            if v236 != v217 and (v229 is None or v449(v229[v236])):
                v233[v236] += 1
    v167 = []
    for v236, v577 in v233.v562(v61):
        for v234, v899, v891 in v178['fills'][v236]:
            if v234 == v242:
                v167.v937(v899)
                break
        if v550(v167) >= v94:
            break
    return v167

def reach_rows_for(v159, v144, v242, v171):
    """The evidence a candidate brings: its walked rows, its homes elsewhere on the tape, or
    the mentions it has at places RELATED to this one (372b)."""
    if v89 == 'homes':
        return v897(v159, v144, v242)
    if v89 == 'relation':
        v578 = v900(v159, v144, v242)
        return v578 if v578 else v171
    return v171

def reach_world(v159, v144, v189, v190, v242, v171, v257):
    """The question's rows, the query row filled in, and up to `budget` rows the walk found for
    that value. One budget for every candidate, so no world is larger than another."""
    v154 = v510(v144['slots']) + [v282 for v282 in v171[:v257] if v282 not in v144['slots']]
    v141 = [v159['tape'].v266[v282] for v282 in v144['slots'][:v144['query_row']]]
    v141 += [v242] + [v159['tape'].v266[v282] for v282 in v154[v550(v144['slots']):]]
    v210 = {'verb': 'lookup', 'S': v144['S'], 'slots': v154, 'vals': v141, 'query_row': v144['query_row'], 'n_first': v550(v154), 'home_cos': 0.0 if v80 == 'stage2' and (not v144.v513('_stage2')) else v739(v159, v144, v242, v144.v513('_qv')), 'confirm': v741(v159, v144, v242), 'channel': v414(v159, v144, v242)}
    return v539(v159, v210, v189, v190, query_value=None, import_k=0)

def reach_move_pick(v196, v159, v144, v190, v189):
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
    if not v66:
        return (None, None, [], [])
    v258 = []
    for v138 in v67:
        v243 = v583(v159, v144, which=v138)
        v579 = v243['cands']
        if not v579:
            continue
        v171 = v243['rows_of'].v513(v579[0], [])
        if v171:
            v258.v876((v138, v579[0], v171[:1], v144['truth_value'] in v360(v579)))
    if not v258:
        v144['_move'] = v67[0]
        return (v67[0], None, [], [])
    v259 = []
    for v580, v234, v581, v582 in v258:
        v144.v538('_base', None)
        v144['_stage2'] = True
        v259.v876(v906(v159, v144, v189, v190, v234, v581, 1))
    v260 = v706.v541([v196.v542(*v185) for v185 in v259])
    if '_move' not in v144:
        v144['_move'] = v258[v10(v260.v934())][0]
    v144.v538('_base', None)
    return (v144['_move'], v260, [v138 for v138, v1204, v1202, v582 in v258], [v147 for v580, v1204, v1202, v147 in v258])

def reach_logits(v196, v159, v144, v190, v189):
    """Stage 1: say one of the values already here, refuse, or walk. Stage 2: say one of the
    values the walk reached, or refuse. Both are worlds scored by the same Phi."""
    if v66 and ('_move' not in v144 or v68):
        v901, v902, v903, v904 = v905(v196, v159, v144, v190, v189)
        v144['_move_l0'] = (v901, v903)
        v144['_move_ballot'] = None if v902 is None else (v902, v904)
    v243 = v583(v159, v144)
    v168, v155, v240 = (v243['own'], v243['cands'], v243['rows_of'])
    v261 = {v209: v617(v159, v144, v209, v240[v209]) for v209 in v155}
    v257 = v595([v122] + [v550(v261[v209]) for v209 in v155]) if v155 else 0

    def world(v242, v171, v148, v584=False):
        v144.v538('_base', None)
        v144['_stage2'] = v584
        return v906(v159, v144, v189, v190, v242, v171, v148)
    v256 = v575(v159, v144) if v87 else {'cands': [], 'rows_of': {}}
    v302, v585 = (v256['cands'], v256['rows_of'])
    v262 = v595([v122] + [v550(v585[v209]) for v209 in v302]) if v302 else 0
    v263 = v144.v513('_reach_g')
    if v263 is None:
        v586 = [v282 for v209 in v155 for v282 in v261[v209]]
        if v58:
            v907 = {v234: v897(v159, v144, v234) for v234 in v168}
            for v234 in v168:
                v59[0] += 1
                v59[1] += 1 if v550(v907[v234]) >= v257 else 0
            v320 = [v1098(v234, v907[v234], v257) for v234 in v168]
        else:
            v320 = [v1098(v234, [], 0) for v234 in v168]
        if not v93:
            v320.v876(v1098(v15, [], 0))
        if not v92:
            v320.v876(v1098(v15, [v282 for v209 in v155 for v282 in v261[v209][:v257]], 10 ** 6))
        v325 = [v1098(v209, v261[v209], v257, True) for v209 in v155]
        if not v93 or not v155:
            v325.v876(v1098(v15, v586, v257, True))
        v587 = [v1098(v209, v585[v209], v262) for v209 in v302] if v302 else []
        v263 = (v320, v325, v587)
        if v144.v513('_keep_g'):
            v144['_reach_g'] = v263
    v320, v325, v587 = v263
    v211 = v706.v541([v196.v542(*v185) for v185 in v320])
    v213 = v706.v541([v196.v542(*v185) for v185 in v325])
    v264 = v706.v541([v196.v542(*v185) for v185 in v587]) if v587 else None
    if v50 > 1:
        v494, v908, v593 = v909(v196, v159, v144, v190, v189, v155, v261, v257, v213)
        v144['_deep'] = (v494, v908, v593)
    if v54:
        v144['_own_l'] = v211
        v138 = v595(v550(v211), v550(v213)) if v550(v211) and v550(v213) else 0

        def summary(v185):
            """WHAT A BRANCH IS WORTH, in one number. `max` is the best world behind it - the
            lookahead rule, unchanged. `margin` is the GAP between its best and second-best,
            which is a different claim: not "how good is the best thing here" but "how sure am
            I which of these it is".

            The margin is the quantity 337 measured at AUC 0.866 and 352 at 0.969 - the mind
            knows when it is right better than it knows anything else - and it currently decides
            NOTHING. This is the smallest form in which it decides something. It is a declared
            rule and not a fitted one, and `max` reproduces every earlier run exactly.
            """
            if not v550(v185):
                return v706.v543((1,), -1000000000.0, device=v211.v190)
            if v34 == 'max' or v550(v185) < 2:
                return v185.v576().v1099(1)
            v624 = v185.v616(descending=True).v266
            return (v624[0] - v624[1]).v1099(1)
        v588 = v910(v211[:v138] if v138 else v211)
        v589 = v910(v213[:v138] if v138 else v213) if v550(v213) else v706.v543((1,), -1000000000.0, device=v211.v190)
        v211 = v706.v705([v588, v589])
    if v92 and (not v54):
        v138 = v595(v550(v213), v550(v264)) if v55 and v87 and (v264 is not None) and v550(v213) and v550(v264) else 0
        v158 = [(v213[:v138] if v138 else v213).v576().v1099(1)]
        if v87:
            v158.v876((v264[:v138] if v138 else v264).v576().v1099(1) if v264 is not None else v706.v543((1,), -1000000000.0, device=v211.v190))
        v211 = v706.v705([v211] + v158)
    if v50 > 1:
        v494, v911, v912 = v144.v513('_deep', (None, [], []))
        if v494 is not None:
            v213 = v706.v705([v213, v494.v576().v1099(1)])
    return (v211, v213, v168, v155, v264, v302)

def coherence_block(v196, v159, v189, v190, v265, v152):
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
    v178 = v551(v159)
    if v178 is None or not v178['items']:
        return None
    v219 = [v222 for v222 in v178['items'] if v550(v222['slots']) >= 3]
    if not v219:
        return None
    v141 = v159['tape'].v266
    v267 = v268 = 0
    v269 = []
    with v706.v404():
        for v180 in v521(v265):
            v222 = v219[v152.v870(v550(v219))]
            v171 = v510(v222['slots'])
            if v550(v171) > v79:
                v171 = v152.v892(v171, v79)
            v164 = {v141[v282] for v282 in v171}
            for v913 in v521(8):
                v667 = v178['items'][v152.v870(v550(v178['items']))]
                if v667['address'] == v222['address'] and v550(v178['items']) > 1:
                    continue
                v916 = v141[v667['slots'][v152.v870(v550(v667['slots']))]]
                if v916 not in v164:
                    break
            else:
                continue
            v914 = {'verb': 'lookup', 'S': v222['S'], 'slots': v171, 'vals': [v141[v282] for v282 in v171], 'query_row': -1, 'n_first': v550(v171)}
            v915 = v439(v914)
            v254 = v510(v914['vals'])
            v254[v152.v870(v550(v254))] = v916
            v915['vals'] = v254
            v147 = v9(v196.v542(*v539(v159, v914, v189, v190, query_value=None, import_k=0)))
            v148 = v9(v196.v542(*v539(v159, v915, v189, v190, query_value=None, import_k=0)))
            v267 += v147 > v148
            v268 += v147 == v148
            v269.v876(v147 - v148)
    v216 = v550(v269)
    if not v216:
        return None
    return {'n': v216, 'real_higher': v267 / v216, 'ties': v268 / v216, 'mean_gap': v506(v269) / v216, 'binomial_z': (v267 - 0.5 * v216) / (0.25 * v216) ** 0.5 if v216 else v9('nan')}

def reach_places_from(v159, v270, v140, v271):
    """The k places nearest to PLACE j0, rather than to the question's own place.

    The deeper walk is the same operation one read further on: the same fingerprints, the same
    cosine, the same cap. Places already visited are excluded so a chain cannot pay twice for
    standing still, which would make depth free money.
    """
    v178 = v551(v159)
    if v178 is None:
        return []
    v228 = (v178['M'] @ v178['M'][v270]).v557()
    for v236 in v271:
        v228[v236] = -2.0
    v229 = v558(v159)
    if v229 is not None:
        v228 = v228.v890(~v229.v702(v228.v190), -2.0)
    v226 = v228.v975(descending=True)[:v140].v590()
    return [(v236, v178['items'][v236], v9(v228[v236])) for v236 in v226 if v236 not in v271 and v228[v236] > -2.0]

def place_of_rows(v159, v144, v171):
    """381: THE PLACE A CANDIDATE ACTUALLY STANDS AT - its first row that is not at home.

    A row is a slot, a slot has an address, an address is a place. Rows at the question's own
    address are skipped, the same exclusion the walk and the connect channel already apply.
    Returns None when the candidate has no row anywhere else, which is a real state and not an
    error: `outside_mentions` can be empty for a value whose only standing is the hidden one.
    """
    v178 = v551(v159)
    for v166 in v171:
        if v159['straddr'][v166] == v144['address']:
            continue
        v272 = v178['of'].v513(v159['straddr'][v166])
        if v272 is not None:
            return v272
    return None

def deep_root_of(v159, v144, v243, v242, v239):
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
    v236 = v243['from_place'].v513(v242)
    if v236 is not None and v236 >= 0:
        return v236
    v272 = v591(v159, v144, v243['rows_of'].v513(v242, ()))
    return v239[0][0] if v272 is None else v272

def reach_deep(v196, v159, v144, v190, v189, v155, v261, v257, v213):
    """One more read, rooted where the mind's own best shallow candidate lives.

    Returns (logits, candidate names) for the deeper stage, or (None, []) when there is nowhere
    left to go. Values already offered shallower are excluded: a deeper read has to bring
    something NEW or it is not a read.
    """
    v243 = v583(v159, v144)
    v239 = v243['places']
    if not v239 or not v155 or v550(v213) == 0:
        return (None, [], [])
    v178 = v551(v159)
    if v48 == 'first':
        v270 = v239[0][0]
    else:
        v252 = v595(v10(v213.v934()), v550(v155) - 1)
        v270 = v917(v159, v144, v243, v155[v252], v239)
    v273 = {v236 for v236, v565, v1199 in v239} | {v270}
    v274 = v360(v155) | v360(v243['own'])
    v276, v592, v593 = ([], {}, [])
    for v236, v565, v566 in v594(v159, v270, v76, v273):
        v593.v876((v236, v178['items'][v236], 0.0))
        for v234, v171, v891 in v178['fills'][v236]:
            if v234 in v274:
                continue
            if v234 not in v592:
                if v550(v276) >= v77:
                    continue
                v276.v876(v234)
                v592[v234] = []
            v592[v234].v937(v171)
    if not v276:
        return (None, [], [])
    v275 = {v209: v617(v159, v144, v209, v592[v209]) for v209 in v276}
    v148 = v595([v122] + [v550(v275[v209]) for v209 in v276])
    v263 = []
    for v234 in v276:
        v144.v538('_base', None)
        v144['_stage2'] = True
        v263.v876(v906(v159, v144, v189, v190, v234, v275[v234], v148))
    return (v706.v541([v196.v542(*v185) for v185 in v263]), v276, v593)

def reach_bisect(v196, v159, v144, v190, v189, v155, v261, v257):
    """Halve the candidate list until one survives, scoring each half as ONE unfilled world.

    EQUAL HALVES BY CONSTRUCTION. When the list is odd the extra candidate stays in its half for
    the purposes of SURVIVING, but only min(|L|, |R|) of each side contribute evidence rows - so
    the two worlds are the same size and the split cannot be decided by having more to show.
    Returns (survivor, split logit pairs, the truth's side at each split) - the last is the exact
    teacher, read off the tape and never off the model.
    """
    v226, v596, v597 = (v510(v155), [], [])
    v170 = v144['truth_value']
    while v550(v226) > 1:
        v319 = v550(v226) // 2
        v356, v214 = (v226[:v319], v226[v319:])
        v216 = v595(v550(v356), v550(v214))
        v144.v538('_base', None)
        v144['_stage2'] = True
        v598 = v906(v159, v144, v189, v190, v15, [v282 for v234 in v356[:v216] for v282 in v261[v234][:v257]], 10 ** 6)
        v144.v538('_base', None)
        v581 = v906(v159, v144, v189, v190, v15, [v282 for v234 in v214[:v216] for v282 in v261[v234][:v257]], 10 ** 6)
        v399 = v706.v541([v196.v542(*v598), v196.v542(*v581)])
        v596.v876(v399)
        v597.v876(0 if v170 in v360(v356) else 1 if v170 in v360(v214) else -1)
        v226 = v356 if v10(v399.v934()) == 0 else v214
    return (v226[0] if v226 else None, v596, v597)

def reach_reward(v144, v277, v172, v190):
    v214 = v706.v543((v550(v277),), -1.0, device=v190)
    for v217, v599 in v549(v277):
        if v599 == v15:
            v214[v217] = 1.0 if not v172 else 0.75
        elif v599 == v144['truth_value']:
            v214[v217] = 1.0
    return v600(v214)

def shift_reward(v214):
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
    return (v214 + 1.0) / 2.0 if v29 < 1.0 else v214

def reach_names(v168, v155):
    """Stage-one and stage-two option names, in the order their worlds were built. One
    definition, because the loss, the exam and the report must not disagree about what the
    columns of a softmax mean."""
    v158 = [] if v93 else [v15]
    return (v168 + v158, v155 + ([v15] if v158 or not v155 else []))

def reach_answerable(v159, v144):
    return v144['truth_value'] in v360(v583(v159, v144)['cands'])

def speak_term(v278, v279, v190):
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
    v138 = v706.v541(v278)
    v147 = v706.v601(v279, device=v190, dtype=v138.v918)
    return (v706.v608(v138, 0) * v147).v506()

def calib_term(v280, v281, v190):
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
    v282 = v706.v541(v280)
    v186 = v706.v601(v281, device=v190, dtype=v282.v918)
    v283 = v9(v186.v506())
    if v283 <= 0.0 or v283 >= v9(v550(v281)):
        return v706.v553((), device=v190, dtype=v282.v918)
    return (v706.v1205(v282, 0) * (v186 / v283)).v506()

def move_term(v144, v190):
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
    if not v68:
        return None
    v284 = v144.v513('_move_ballot')
    if v284 is None:
        return None
    v260, v285 = v284
    v69['ballot'] += v550(v285)
    v69['n'] += 1
    if v550(v285) < 2 or v550(v360(v285)) < 2:
        return None
    v69['live'] += 1
    v214 = v600(v706.v601([1.0 if v147 else -1.0 for v147 in v285], device=v190, dtype=v260.v918))
    return v68 * (v706.v608(v260, 0) * v214).v506()

def reach_loss(v196, v159, v144, v190, v189):
    v211, v213, v168, v155, v264, v302 = v602(v196, v159, v144, v190, v189)
    v285 = v144['truth_value'] in v360(v155)
    if v38 is not None or v41 is not None:
        v919, v920, v912, v921, v922 = v923(v144, v211, v213, v168, v155, v264, v302, keep_graph=True)
        v603 = v919 == v144['truth_value']
        if v38 is not None:
            v38.v876((v922, v1265(False, v603, v285) - v1265(True, v603, v285)))
        if v41 is not None:
            v41.v876((v921, 1.0 if v285 else 0.0))
    v604, v605 = v606(v168, v155)
    v286 = v607(v144, v604, v285, v190)
    v287 = v607(v144, v605, v285, v190)
    v288 = v706.v608(v211, 0)
    if v70 == 'walk_only':
        v71['n'] += 1
        if v285 and v144['truth_value'] not in v360(v168):
            v71['live'] += 1
        else:
            v288 = v288.v1100()
    v289 = v706.v608(v213, 0)
    if v50 > 1:
        v494, v908, v912 = v144.v513('_deep', (None, [], []))
        if v494 is not None:
            v924 = v607(v144, v908, v144['truth_value'] in v360(v908), v190)
            v925 = v29 * (v706.v608(v494, 0) * v924).v506()
            v287 = v706.v705([v287, v925.v1099(1)])
    v215 = v29 * (v289 * v287).v506() - v28
    v290 = v609(v144, v190)
    if v54:
        v364 = v144['_own_l']
        v610 = (v706.v608(v364, 0) * v286).v506()
        v167 = v288[0] * v610 + v288[1] * v215
        return -(v167 if v290 is None else v167 + v290)
    v291 = 2 if v87 else 1
    v167 = (v288[:-v291] * v286).v506() + v288[-v291] * v215
    if v51:
        v180, v596, v597 = v926(v196, v159, v144, v190, v189, v155, v261, v257)
        for v399, v775 in v666(v596, v597):
            if v775 >= 0:
                v167 = v167 - v706.v372.v1322.v993(v399.v1099(1, 2), v706.v601([v775], device=v190))
    if v53:
        v167 = v167 + v53 * (v289 * v287).v506()
    if v87:
        if v264 is not None:
            v927 = v144['truth_value'] in v360(v302)
            v924 = v607(v144, v302, v927, v190)
            v928 = v706.v608(v264, 0)
            v167 = v167 + v288[-1] * (v29 * (v928 * v924).v506() - v28)
        else:
            v167 = v167 + v288[-1] * (v29 * -1.0)
    return -(v167 if v290 is None else v167 + v290)

def cons_cooc(v159, v234):
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
    v178 = v551(v159)
    if v178 is None:
        return {}
    v292 = v159.v513('_cons_cooc')
    if v292 is None:
        v292 = v159['_cons_cooc'] = {}
    v209 = v292.v513(v234)
    if v209 is None:
        v209 = v207()
        for v236, v612 in v178['by_val'].v513(v234, ()):
            for v210, v888, v385 in v178['fills'][v236]:
                v209[v210] += v385
        v292[v234] = v209 = v439(v209)
    return v209

def cons_place(v159, v144, v234):
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
    v178 = v551(v159)
    if v178 is None:
        return None
    v164 = v178['of'].v513(v144['address'])
    v293, v611 = (None, None)
    for v236, v612 in v178['by_val'].v513(v234, ()):
        if v236 == v164:
            continue
        v221 = v178['fills'][v236]
        v613 = v506((v209 for v802, v1202, v209 in v221))
        v385 = v506((v209 for v210, v1202, v209 in v221 if v210 == v234))
        if v613 <= 0 or v385 <= 0:
            continue
        v614 = (v385, v385 / v613, -v236)
        if v611 is None or v614 > v611:
            v293, v611 = (v236, v614)
    return v293

def cons_resolve(v159, v144, v234):
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
    v178 = v551(v159)
    if v178 is None:
        return (None, 0, 0, [])
    if v35 == 'place':
        v236 = v929(v159, v144, v234)
        if v236 is None:
            return (None, 0, 0, [])
        v209, v308 = ({v210: v385 for v210, v1202, v385 in v178['fills'][v236]}, {})
    else:
        v209 = v930(v159, v234)
        if not v209:
            return (None, 0, 0, [])
        v164 = v178['of'].v513(v144['address'])
        v308 = {}
        if v164 is not None and v518((v236 == v164 for v236, v612 in v178['by_val'].v513(v234, ()))):
            v308 = {v210: v385 for v210, v888, v385 in v178['fills'][v164]}
    v294 = None
    if v35 == 'share':
        v294 = {}
        for v210, v612 in v209.v219():
            v294[v210] = v506((v385 for v1334, v385 in v178['by_val'].v513(v210, ()))) or 1
    v252, v615, v613 = (None, 0, 0)
    v295 = []
    for v210, v216 in v209.v219():
        if v210 == v234:
            continue
        v216 -= v308.v513(v210, 0)
        if v216 <= 0:
            continue
        v613 += v216
        v282 = v216 / v294[v210] if v294 is not None else v216
        v295.v876((v282, v210))
        if v282 > v615:
            v252, v615 = (v210, v282)
    v295.v616(key=lambda v561: (-v561[0], v561[1]))
    if v295 and v295[0][0] == v615:
        v252 = v295[0][1]
    return (v252, v615, v613, [v210 for v612, v210 in v295[:v33]])

def cons_lenses(v159, v144):
    """The rows the mind may look through: its own visible values, deduped, in tape order.

    THE OUTPUT SPACE OF PHI, and the reason the invariant is safe here. It is not a vocabulary
    and not a candidate list - it is an index into what is already on the question's own place.
    """
    if '_cons_l' in v144:
        return v144['_cons_l']
    v327, v167 = (v360(), [])
    for v282 in v144['slots'][:v144['query_row']]:
        v234 = v159['tape'].v266[v282]
        if v234 not in v327:
            v327.v1101(v234)
            v167.v876(v234)
    v167 = v167[:v32]
    v144['_cons_l'] = v167
    return v167

def cons_rows_for(v159, v144, v234):
    """Where the lens stands ELSEWHERE - the evidence for looking through it.

    Same import machinery as a candidate's evidence, pointed at a different question: not "here
    is a value, is it right" but "here is a row, is it worth following".
    """
    v178 = v551(v159)
    if v178 is None:
        return []
    v164 = v178['of'].v513(v144['address'])
    v171 = []
    v237 = [v929(v159, v144, v234)] if v35 == 'place' else [v236 for v236, v612 in v178['by_val'].v513(v234, ())]
    for v236 in v237:
        if v236 is None or v236 == v164:
            continue
        for v210, v931, v891 in v178['fills'][v236]:
            if v210 == v234:
                v171.v937(v931)
    return v617(v159, v144, v234, v171)

def cons_logits(v196, v159, v144, v190, v189):
    """Stage 1: say a value already here, or CONSTRAIN. Stage 2: which row to look through.

    The shape is reach_logits' shape on purpose - same worlds, same import budget, same
    lookahead rule, same refusal equalisation - so what is being compared between the two arms
    is the OPERATION and not the plumbing.
    """
    v168 = v511({v159['tape'].v266[v282] for v282 in v144['slots'][:v144['query_row']]})
    v296 = v618(v159, v144)
    v261 = {v234: v932(v159, v144, v234) for v234 in v296}
    v257 = v595([v122] + [v550(v261[v234]) for v234 in v296]) if v296 else 0

    def world(v242, v171, v148, v584=False):
        v144.v538('_base', None)
        v144['_stage2'] = v584
        return v906(v159, v144, v189, v190, v242, v171, v148)
    v263 = v144.v513('_cons_g')
    if v263 is None:
        v320 = [v1098(v234, [], 0) for v234 in v168]
        if not v93:
            v320.v876(v1098(v15, [], 0))
        if not v92:
            v320.v876(v1098(v15, [v282 for v234 in v296 for v282 in v261[v234][:v257]], 10 ** 6))
        v325 = [v1098(v234, v261[v234], v257, True) for v234 in v296]
        if not v93 or not v296:
            v325.v876(v1098(v15, [v282 for v234 in v296 for v282 in v261[v234]], v257, True))
        v263 = (v320, v325)
        if v144.v513('_keep_g'):
            v144['_cons_g'] = v263
    v320, v325 = v263
    v211 = v706.v541([v196.v542(*v185) for v185 in v320])
    v213 = v706.v541([v196.v542(*v185) for v185 in v325])
    if v92:
        v211 = v706.v705([v211, v213.v576().v1099(1)])
    return (v211, v213, v168, v296)

def cons_answers(v159, v144, v296):
    """What the tape says through each lens, in lens order. The exact teacher."""
    return [v424(v159, v144, v234)[0] for v234 in v296]

def cons_loss(v196, v159, v144, v190, v189):
    v211, v213, v168, v296 = v619(v196, v159, v144, v190, v189)
    v297 = v620(v159, v144, v296)
    v285 = v144['truth_value'] in v360((v185 for v185 in v297 if v185 is not None))
    v286 = v607(v144, v168 + ([] if v93 else [v15]), v285, v190)
    v298 = v297 + ([v15] if not v93 or not v296 else [])
    v287 = v607(v144, [v185 if v185 is not None else v15 for v185 in v298], v285, v190)
    v288 = v706.v608(v211, 0)
    v289 = v706.v608(v213, 0)
    v215 = v29 * (v289 * v287).v506() - v28
    v291 = 1
    return -((v288[:-v291] * v286).v506() + v288[-v291] * v215)

def cons_rivals(v159, v144, v296):
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
    v178 = v551(v159)
    if v178 is None or not v296:
        return {}
    v299 = {v234: v550(v178['by_val'].v513(v234, ())) for v234 in v296}
    v300 = {}
    for v234 in v296:
        v435, v615, v613, v933 = v424(v159, v144, v234)
        v300[v234] = v615 / v613 if v613 else 0.0
    return {'rare': v595(v296, key=lambda v234: (v299[v234], v234)), 'frequent': v576(v296, key=lambda v234: (v299[v234], v234)), 'decisive': v576(v296, key=lambda v234: (v300[v234], v234))}

def cons_question(v159, v146, v152, v153):
    v144 = v621(v159, v146, v152, v153)
    if v144 is not None:
        v144['verb'] = 'cons'
        v144['cons'] = True
        v144.v538('reach', None)
    return v144

def cons_questions_for(v159, v301):
    v167 = []
    if v551(v159) is None:
        return v167
    for v222 in v159['items']:
        for v153 in v521(v550(v222['slots'])):
            if (v144 := v1278(v159, v222, v301, v153)) is not None:
                v167.v876(v144)
    if v78 and v550(v167) > v78:
        v167 = v301.v892(v167, v78)
    return v167
v95 = ('answerable', 'truth_in_own', 'n_lens', 'silent', 'mind_right', 'constrained', 'lens_idx', 'rare_right', 'frequent_right', 'decisive_right', 'present_topm', 'chosen_share', 'chosen_total', 'walk_answerable', 'walk_rival_right')
v96 = {v216: v217 for v217, v216 in v549(v95)}

def reach_pick(v144, v211, v213, v168, v155, v264, v302, v303=False):
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
    v604, v605 = v606(v168, v155)
    v291 = 2 if v87 else 1
    v304 = v10(v211.v934())
    if v54:
        if v304 == 1 and v550(v213):
            v622, v277, v623 = (1, v605, v213)
        else:
            v622, v277, v623 = (0, v604, v144['_own_l'])
    elif v304 == v550(v604):
        v622, v277, v623 = (1, v605, v213)
    elif v87 and v304 == v550(v604) + 1 and (v264 is not None):
        v622, v277, v623 = (2, v302, v264)
    else:
        v622, v277, v623 = (0, v604, v211[:-v291])
    v305 = 1 if v622 else 0
    if v622 == 1 and v50 > 1:
        v494, v276, v912 = v144.v513('_deep', (None, [], []))
        if v494 is not None and v10(v623.v934()) == v550(v277):
            v277, v623, v305 = (v276, v494, 2)
    v297 = v277[v10(v623.v934())] if v550(v277) else v15
    if v550(v623):
        if v303:
            v624 = v623.v616(descending=True).v266
            return (v297, v622, v305, v624[0], v624[0] - v624[1] if v550(v624) > 1 else v624[0] - v624[0])
        v624 = v623.v1100().v616(descending=True).v266
        v318 = v9(v624[0])
        v371 = v9(v624[0] - v624[1]) if v550(v624) > 1 else 0.0
    elif v303:
        v935 = v211.v506() * 0.0
        return (v297, v622, v305, v935, v935)
    else:
        v318, v371 = (0.0, 0.0)
    return (v297, v622, v305, v318, v371)

def rank_auc(v280, v281):
    """Concordant pairs, ties counted half: the share of (positive, negative) pairs the
    ranking gets the right way round. A count over pairs, no threshold and nothing fitted -
    which is why it is the statistic for 337 rather than an accuracy at some chosen cut."""
    v216 = v550(v280)
    if v216 != v550(v281) or v216 == 0:
        return v9('nan')
    v283 = v506((1 for v1014 in v281 if v1014))
    v306 = v216 - v283
    if not v283 or not v306:
        return v9('nan')
    v226 = v511(v521(v216), key=lambda v217: v280[v217])
    v307 = [0.0] * v216
    v217 = 0
    while v217 < v216:
        v236 = v217
        while v236 + 1 < v216 and v280[v226[v236 + 1]] == v280[v226[v217]]:
            v236 += 1
        v625 = (v217 + v236) / 2.0 + 1.0
        for v250 in v521(v217, v236 + 1):
            v307[v226[v250]] = v625
        v217 = v236 + 1
    v282 = v506((v301 for v301, v1014 in v666(v307, v281) if v1014))
    return (v282 - v283 * (v283 + 1) / 2.0) / (v283 * v306)
v97 = (0.05, 0.1, 0.25, 0.5)

def gate_top(v280, v140):
    """The k questions a ranking would let through, as a set of indices.

    TIES BREAK BY POSITION, which is not a decision: the exam's question list is a SAMPLE, in
    sampled order, so position is already arbitrary. It matters because the counting rivals tie
    constantly - |own| is a small integer - and a gate cutting inside a tie is choosing at
    random among equals. That is a real limit of gating on a count, not a handicap imposed on
    it: a statistic that cannot separate two questions cannot gate between them. AUC in
    rankblock scores those ties at half credit and is the tie-fair companion to this.
    """
    v226 = v511(v521(v550(v280)), key=lambda v217: -v280[v217])
    return v360(v226[:v576(0, v595(v140, v550(v280)))])

def prec_at(v280, v281, v140):
    """Of the k questions this ranking puts first, how many were really answerable."""
    v216 = v550(v280)
    if v140 <= 0 or v216 == 0:
        return v9('nan')
    v226 = v511(v521(v216), key=lambda v217: -v280[v217])
    v140 = v595(v140, v216)
    return v506((1 for v217 in v226[:v140] if v281[v217])) / v140

def reach_rival(v159, v144):
    """The same walk without the mind: nearest place, its most frequent filler. The margin is
    the gap to the next place, so it can be thresholded and allowed to stay silent too."""
    v243 = v583(v159, v144)
    v239 = v243['places']
    if not v239:
        return (None, -1.0)
    v270, v565, v626 = v239[0]
    v221 = v551(v159)['fills'][v270]
    v252 = v576(v549(v221), key=lambda v561: (v561[1][2], -v561[0]))[1][0] if v221 else None
    return (v252, v626 - v239[1][2] if v550(v239) > 1 else 1.0)

def reach_count_rival(v159, v144):
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
    v243 = v583(v159, v144)
    v155 = v510(v243['cands'])
    v178 = v551(v159)
    if v178 is None:
        return (None, 0.0)
    v239 = v510(v243['places'])
    if v50 > 1:
        v936, v911, v912 = v144.v513('_deep', (None, [], []))
        v155 = v155 + [v209 for v209 in v911 if v209 not in v360(v155)]
        v239 = v239 + [v1037 for v1037 in v912 if v1037[0] not in {v236 for v236, v1342, v1199 in v239}]
    if not v155:
        return (None, 0.0)
    v252, v627, v628, v226 = (None, -1.0, -1, {v234: v217 for v217, v234 in v549(v155)})
    v327, v629, v630 = (v360(v155), v360(), v207())

    def offer(v234, v232, v385):
        nonlocal best, bshare, bcount
        if v232 > v627:
            v630.v661()
        if v232 == v627:
            v630[v234] += 1
        if (v232, v385) > (v627, v628) or ((v232, v385) == (v627, v628) and v252 is not None and (v226[v234] < v226[v252])):
            v252, v627, v628 = (v234, v232, v385)
    for v236, v565, v566 in v239:
        v221 = v178['fills'][v236]
        v613 = v506((v209 for v1204, v1202, v209 in v221))
        if v613 <= 0:
            continue
        for v234, v888, v209 in v221:
            if v234 not in v327:
                continue
            v629.v1101(v234)
            v632(v234, v209 / v613, v209)
    for v234 in v155:
        if v234 in v629:
            continue
        v236 = v243.v513('real_place', {}).v513(v234)
        if v236 is None:
            continue
        v221 = v178['fills'][v236]
        v613 = v506((v209 for v1204, v1202, v209 in v221))
        if v613 <= 0:
            continue
        v385 = v506((v209 for v1022, v1202, v209 in v221 if v1022 == v234))
        if v385 > 0:
            v632(v234, v385 / v613, v385)
    if v631(v144, v439):
        v144['_cr_ties'] = v550(v630) + 1 if v252 is not None else 0
    return (v252, v627 if v252 is not None else 0.0)

def reach_questions_for(v159, v301):
    """Every hidden filler the tape can offer, then capped by SAMPLING.

    A cap is needed and it is a cost bound, not a choice: a 3000-address frame tape gives ~9000
    questions per pack and each one costs a dozen graphs, so scoring all of them is hours of the
    same measurement. Sampled rather than truncated, for the reason 298 taught the hard way -
    a deterministic prefix is the same tape every time and the redraw stops meaning anything.
    """
    v167 = []
    if v551(v159) is None:
        return v167
    for v222 in v159['items']:
        for v153 in v521(v550(v222['slots'])):
            if (v144 := v621(v159, v222, v301, v153)) is not None:
                v167.v876(v144)
    if v78 and v550(v167) > v78:
        v167 = v301.v892(v167, v78)
    for v222 in v159['items']:
        if v550(v222['slots']) >= 2 and (v144 := v1214(v159, v222)) is not None:
            v167.v876(v144)
    return v167
v98 = False
v99 = 8
v100 = 6
v101 = 2000
v102 = 2
v103 = False
v104 = 2
v105 = ('both_offered', 'mind_right', 'one_right', 'marg_right', 'joint_seen', 'joint_right', 'in_own_a', 'in_own_b', 'offered_a', 'offered_b', 'n_pairs', 'first_hole', 'world_rows', 'right_a', 'right_b', 'bag_seen', 'bag_right')
v106 = {v216: v217 for v217, v216 in v549(v105)}

def pair_offer(v159, v308, v309):
    """What one hole may be filled with: its own values, the walk's, and the line's partners.

    Three sources because each is blind somewhere: own-only is 308's marginal rival with extra
    steps, the walk cannot see the confirmations, and neither can name a value whose only tie
    to this line is that it stands NEXT TO this line's visible values elsewhere - which is what
    a composed answer looks like from outside. Partners are counted off the tape (the bag index
    over co-line values, the question's own line subtracted), so the source belongs to the tape
    and the rivals pick from the same offer. Round-robin rather than fixed halves: whichever
    source is short gives its room to the others, and no ranking of mine orders the sources.
    """
    v168 = v511({v159['tape'].v266[v282] for v282 in v308['slots'][:v308['query_row']]})
    v310 = [v510(v168), v510(v583(v159, v308)['cands']), v510(v309)]
    v632, v327 = ([], v360())
    while v550(v632) < v99 and v518(v310):
        for v530 in v310:
            while v530 and v530[0] in v327:
                v530.v538(0)
            if v530 and v550(v632) < v99:
                v327.v1101(v530[0])
                v632.v876(v530.v538(0))
    return (v632, v360(v168))

def pair_question(v159, v311, v312, v152):
    """Two holes of one line, each with its place's other rows, in one world.

    Order is by corpus position, so hole A is the earlier one in every world of this question
    and the cosine column the graph ranks against does not move between candidates.
    """
    v178 = v551(v159)
    if v178 is None:
        return None
    v313 = []
    for v166 in (v311, v312):
        v236 = v178['of'].v513(v159['straddr'][v166])
        if v236 is None:
            return None
        v222 = v178['items'][v236]
        if v166 not in v222['slots'] or v550(v222['slots']) < 2:
            return None
        v308 = v621(v159, v222, v152, v222['slots'].v649(v166))
        if v308 is None:
            return None
        v313.v876(v308)
    v633, v154, v141, v388, v634 = ([], [], [], [], 0)
    for v140, (v166, v308) in v549(v666((v311, v312), v313)):
        v171 = v308['slots'][:v308['query_row']][:v100]
        v154.v937(v171)
        v141.v937((v159['tape'].v266[v282] for v282 in v171))
        v388.v876(v550(v154))
        v154.v876(v166)
        v141.v876(v1102())
        v633.v876({'slot': v166, 'sub': v308, 'address': v159['straddr'][v166], 'truth': v159['tape'].v266[v166], 'rows': v171})
        if v140 == 0:
            v634 = v550(v154)
    return {'verb': 'pair', 'pair': True, 'slots': v154, 'vals': v141, 'holes': v633, 'query_rows': v388, 'query_row': v388[0], 'n_first': v634, 'S': v633[0]['address'], 'S2': v633[1]['address'], 'line': v159['line'][v311]}

def pair_offers(v159, v144):
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
    if '_pair_ev' in v144:
        return v144['holes']
    v635(v159)
    v314 = v159['_pairbag']
    v164 = v360(v144['slots'])
    v315 = {v319['address'] for v319 in v144['holes']}
    v316 = v1103(v159).v513(v144['line'], ())
    v317 = [v166 for v166 in v316 if v166 not in v164 and v159['straddr'][v166] not in v315]
    v318 = v207()
    for v166 in v317:
        for v234, v209 in v314.v513(v159['tape'].v266[v166], {}).v219():
            v318[v234] += v209
        for v636 in v316:
            if v636 != v166 and v159['straddr'][v636] != v159['straddr'][v166]:
                v318[v159['tape'].v266[v636]] -= 1
    v309 = [v234 for v234, v209 in v318.v562(3 * v99) if v209 > 0]
    v261 = {}
    for v319 in v144['holes']:
        if 'offer' not in v319:
            v319['offer'], v319['own'] = v1104(v159, v319['sub'], v309)
        for v234 in v319['offer']:
            if v234 not in v261:
                v261[v234] = v897(v159, v144, v234)
    v144['_pair_ev'] = v261
    v144['_pair_pool'] = v317
    v144['_pair_b'] = v595([v122] + [v550(v261[v234]) + v550(v317) for v234 in v261]) if v261 else 0
    return v144['holes']

def pair_world(v159, v144, v189, v190, v221, v244):
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
    v154, v141 = (v510(v144['slots']), v510(v144['vals']))
    v148, v317 = (v144['_pair_b'], v144['_pair_pool'])
    for v140 in v511(v221):
        v234 = v221[v140]
        v141[v144['query_rows'][v140]] = v234
        v171 = v144['_pair_ev'].v513(v234, [])[:v148]
        v171 = v171 + [v166 for v166 in v317 if v166 not in v171][:v576(0, v148 - v550(v171))]
        for v166 in v171:
            if v166 not in v154:
                v154.v876(v166)
                v141.v876(v159['tape'].v266[v166])
    v210 = {'verb': 'lookup', 'S': v144['S'], 'S2': v144['S2'], 'slots': v154, 'vals': v141, 'query_row': v244, 'query_rows': v144['query_rows'], 'n_first': v144['n_first']}
    return v539(v159, v210, v189, v190, query_value=None, import_k=0)

def pair_logits(v196, v159, v144, v190, v189):
    """Stage 1: every value of every hole, each written into the world alone. Stage 2: the other
    hole, written into the world the winner of stage 1 left behind.

    The two stages are the SAME operation on the SAME world - which is the point. Nothing about
    the second fill is a new mechanism; it simply sees more.
    """
    v320 = []
    for v140, v319 in v549(v641(v159, v144)):
        for v234 in v319['offer']:
            v320.v876((v140, v234, v938(v159, v144, v189, v190, {v140: v234}, v144['query_rows'][v140])))
    v211 = v706.v541([v196.v542(*v185[2]) for v185 in v320])
    return (v211, v320)

def pair_second(v196, v159, v144, v190, v189, v321, v322):
    """The other hole, scored in a world that already says `v0` - or, under PAIR_BLIND, in one
    that does not. That single dictionary entry is the whole of the composition claim."""
    v323 = 1 - v321
    v319 = v641(v159, v144)[v323]
    v324 = {} if v103 else {v321: v322}
    v325 = [v938(v159, v144, v189, v190, {**v324, v323: v234}, v144['query_rows'][v323]) for v234 in v319['offer']]
    return (v706.v541([v196.v542(*v185) for v185 in v325]), v319['offer'], v323)

def pair_loss(v196, v159, v144, v190, v189):
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
    v211, v320 = v637(v196, v159, v144, v190, v189)
    v288 = v706.v608(v211, 0)
    v170 = [v319['truth'] for v319 in v144['holes']]
    v167 = 0.0
    v226 = v511(v521(v550(v320)), key=lambda v217: -v9(v211[v217]))[:v104]
    for v217 in v226:
        v321, v322, v180 = v320[v217]
        v213, v939, v323 = v940(v196, v159, v144, v190, v189, v321, v322)
        v289 = v706.v608(v213, 0)
        v638 = 0.5 * v9(v322 == v170[v321])
        v287 = v706.v601([v638 + 0.5 * v9(v234 == v170[v323]) for v234 in v939], device=v190)
        v167 = v167 + v288[v217] * (v289 * v287).v506()
    v326 = [v217 for v217 in v521(v550(v320)) if v217 not in v360(v226)]
    for v217 in v326:
        v321, v322, v180 = v320[v217]
        v167 = v167 + v288[v217] * (0.5 * v9(v322 == v170[v321]))
    return -v167

def pair_rivals(v159, v144):
    """Counting's two ways at this question, both exact.

    MARGINAL: each hole answered on its own by the most frequent value of its place. This is the
    product of two marginals and it is the rival composition has to beat, because it is what a
    perfect index does when the holes are treated separately.

    JOINT: did this exact pair ever stand together at these two places on another line? If it
    did, counting has the pair outright and no composition is required. The subset that decides
    is where BOTH are blind.
    """
    v167 = []
    for v319 in v144['holes']:
        v209 = v207((v159['tape'].v266[v282] for v282 in v319['sub']['slots'][:v319['sub']['query_row']]))
        v167.v876(v209.v562(1)[0][0] if v209 else None)
    v178 = v635(v159)
    v147, v148 = v144['holes']
    v327 = v178.v513((v147['address'], v148['address']), {})
    v170 = (v147['truth'], v148['truth'])
    v252, v639 = (None, 0)
    for (v149, v150), v216 in v327.v219():
        v216 -= (v149, v150) == v170
        if v216 > v639:
            v252, v639 = ((v149, v150), v216)
    v314 = v159['_pairbag']
    v316 = v1103(v159).v513(v144['line'], ())
    v328 = v207()
    for v217, v640 in v549(v316):
        for v636 in v316[v217 + 1:]:
            if v159['straddr'][v640] != v159['straddr'][v636]:
                v1145, v215 = (v159['tape'].v266[v640], v159['tape'].v266[v636])
                v328[v1145, v215] += 1
                v328[v215, v1145] += 1

    def bagn(v185, v186):
        return v314.v513(v185, {}).v513(v186, 0) - v328.v513((v185, v186), 0)
    v641(v159, v144)
    v642, v615 = (None, 0)
    for v149 in v147['offer']:
        for v150 in v148['offer']:
            v216 = v1105(v149, v150)
            if v216 > v615:
                v642, v615 = ((v149, v150), v216)
    return (v502(v167), v252, v327.v513(v170, 0) > 1, v642, v1105(*v170) > 0)

def pair_joint_index(v159):
    """Every (place, place) -> (value, value) the tape actually wrote on one line, counted once
    per pack - the joint statistic counting would need, given to the rival in full. The same
    pass also counts the BAG: value -> values it shares a line with anywhere, at ANY places.
    The bag is the third counting rival and the partner source, so it is one loop, not three."""
    v178 = v159.v513('_pairjoint')
    if v178 is not None:
        return v178
    v329 = v547(v510)
    for v166, v569 in v549(v159['line']):
        if v569 >= 0:
            v329[v569].v876(v166)
    v178 = v547(v207)
    v314 = v547(v207)
    for v569, v643 in v329.v219():
        for v217, v640 in v549(v643):
            for v636 in v643[v217 + 1:]:
                v1206, v1207 = (v159['straddr'][v640], v159['straddr'][v636])
                if v1206 == v1207:
                    continue
                v1145, v215 = (v159['tape'].v266[v640], v159['tape'].v266[v636])
                if v159['pos'][v640] <= v159['pos'][v636]:
                    v178[v1206, v1207][v1145, v215] += 1
                else:
                    v178[v1207, v1206][v215, v1145] += 1
                v314[v1145][v215] += 1
                v314[v215][v1145] += 1
    v159['_pairjoint'] = v178
    v159['_pairbag'] = v439(v314)
    return v178

def pair_questions_for(v159, v301):
    """Two holes of one line, far enough apart that neither frame can cover the other's token."""
    v167 = []
    if v551(v159) is None or v159.v513('pos') is None:
        return v167
    v156 = {v282 for v222 in v159['items'] for v282 in v222['slots']}
    v329 = v547(v510)
    for v166 in v156:
        if v159['line'][v166] >= 0 and v159['pos'][v166] >= 0:
            v329[v159['line'][v166]].v876(v166)
    for v569, v643 in v329.v219():
        if v550(v643) < 2:
            continue
        v643.v616(key=lambda v282: v159['pos'][v282])
        v644 = [(v147, v148) for v217, v147 in v549(v643) for v148 in v643[v217 + 1:] if v159['straddr'][v147] != v159['straddr'][v148] and v159['pos'][v148] - v159['pos'][v147] > v56]
        if not v644:
            continue
        v301.v655(v644)
        for v147, v148 in v644[:v102]:
            if (v144 := v1279(v159, v147, v148, v301)) is not None:
                v167.v876(v144)
    if v101 and v550(v167) > v101:
        v167 = v301.v892(v167, v101)
    return v167

def open_rival_scored(v159, v144, v189, v190):
    """Whole-tape retrieval, with the confidence it needs to be allowed to stay silent.

    Same rule and same rows as open_rival_cos - kept separate so 292's arms stay bit-identical.
    The margin is the gap from the winning value to the best row of any OTHER value, over the
    spread of the column: a shape, not a magnitude, so no absolute scale is smuggled in.
    """
    v330 = v159.v512('_ctx', {})

    def ctx(v166):
        if v166 not in v330:
            v209 = v189.v1106(v159['texts'][v166], exclude=v159['tape'].v266[v166])
            v330[v166] = v1119.v1094(v209, dim=-1) if v209 is not None else None
        return v330[v166]
    v331 = v645(v144['slots'][v144['query_row']])
    if v331 is None:
        return (None, v9('nan'))
    v140 = v540(v159, v144, v510(v144['cands']))
    v171, v646 = ([], [])
    for v209 in v144['cands']:
        if v209 == v15:
            continue
        for v166 in v897(v159, v144, v209)[:v140]:
            v301 = v645(v166)
            if v301 is not None:
                v171.v876(v301)
                v646.v876(v209)
    if not v171:
        return (None, v9('nan'))
    v228 = (v706.v541(v171, 0) @ v331).v590()
    v295 = v576(v521(v550(v228)), key=lambda v217: v228[v217])
    v252 = v646[v295]
    v332 = [v282 for v282, v667 in v666(v228, v646) if v667 != v252]
    v333 = v576(v228) - v595(v228)
    if not v332:
        return (v252, 1.0)
    return (v252, (v228[v295] - v576(v332)) / v333 if v333 > 1e-09 else 0.0)

def lookup_mixed_question(v159, v146, v152, v153, v175):
    """One exam question: four drawn values plus refusal, and half the time the truth is not
    among them. Everything else is 294's open question unchanged."""
    v144 = v647(v159, v146, v152, v153, v175)
    if v144 is None:
        return None
    v170 = v144['cands'][v144['label']]
    v155 = v510(v144['cands'])
    v172 = v152.v882() < 0.5
    if not v172:
        v164 = {v159['tape'].v266[v282] for v282 in v146['slots']} | v360(v155)
        v648 = None
        for v180 in v521(256):
            v234 = v175[v152.v870(v550(v175))]
            if v234 not in v164:
                v648 = v234
                break
        if v648 is None:
            return None
        v155[v155.v649(v170)] = v648
    v155 = v511(v155 + [v15])
    v144['cands'] = v155
    v144['label'] = v155.v649(v170 if v172 else v15)
    v144['answerable'] = v172
    v144['truth_value'] = v170
    v144['mixed'] = True
    for v334 in ('open', 'uniform', 'bucket_of', '_base', '_ibudget'):
        v144.v538(v334, None)
    if v540(v159, v144, v510(v144['cands'])) < 1:
        return None
    return v144

def mixed_payoff(v335, v336, v172):
    """280's payoff, and the only place the two abilities are weighed against each other.

    Silence on an unanswerable question is not a hedge, it is the correct answer, so it pays
    what a correct answer pays. Silence on an answerable one is the hedge and pays 0.75, which
    is what makes answering worth it above 0.875 confidence - derived, never chosen.
    """
    if v335:
        return 1.0 if not v172 else 0.75
    return 1.0 if v336 else -1.0

def open_rival_cos(v159, v144, v189, v190):
    """The rival 292 actually has to beat: retrieval over the WHOLE TAPE, not over the address.

    Once every candidate brings its own mentions in, similarity is back in the game - it just
    searches the corpus instead of the address. That is RAG, stated exactly, and it is the fork
    the project has been circling: if nearest-imported-context lands where Phi lands, what we
    built is a search engine with extra steps.

    Same rows Phi is given - the shared import budget - and one rule: the candidate with a
    mention whose context is nearest the query's.
    """
    v330 = v159.v512('_ctx', {})

    def ctx(v166):
        if v166 not in v330:
            v209 = v189.v1106(v159['texts'][v166], exclude=v159['tape'].v266[v166])
            v330[v166] = v1119.v1094(v209, dim=-1) if v209 is not None else None
        return v330[v166]
    v331 = v645(v144['slots'][v144['query_row']])
    if v331 is None:
        return None
    v140 = v540(v159, v144, v510(v144['cands']))
    v171, v646 = ([], [])
    for v209 in v144['cands']:
        for v166 in v897(v159, v144, v209)[:v140]:
            v301 = v645(v166)
            if v301 is not None:
                v171.v876(v301)
                v646.v876(v209)
    if not v171:
        return None
    return v646[v10((v706.v541(v171, 0) @ v331).v934())]

def neighbourhood_audit(v159, v140, v337=(1, 3, 6, 12)):
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
    v167 = {}
    for v338 in v511({v140} | v360(v337)):
        v167[f'k={v338}'] = v941(v159, v338)
    return v167

def _audit_at(v159, v140):
    v167 = {}
    for v349, v161 in (('anchor', ('anchor',)), ('rel', ('rel',)), ('word', ('word',)), ('anchor+word', ('anchor', 'word')), ('all', ('anchor', 'rel', 'word'))):
        v216 = v285 = v171 = 0
        for v222 in v159['items']:
            if not 1 <= v550(v222['slots']) <= 2:
                continue
            for v153 in v521(v550(v222['slots'])):
                v168 = v510(v222['slots'])
                v179 = v168[v153 % v550(v168)]
                v170 = v159['tape'].v266[v179]
                v931 = [v185 for v185 in v168 if v185 != v179]
                for v148 in v517(v159, v222['address'], v140, v161):
                    v931 += v510(v159['_addr_index']['slots'].v513(v148, ()))[:v140]
                v931 = v511(v360(v931) - {v179})
                if v550(v931) < 1 or v550({v159['tape'].v266[v185] for v185 in v931}) < 2:
                    continue
                v216 += 1
                v171 += v550(v931)
                v285 += v10(v170 in {v159['tape'].v266[v185] for v185 in v931})
        v167[v349] = {'questions': v216, 'answerable': v285, 'hit_rate': v285 / v216 if v216 else v9('nan'), 'mean_rows': v171 / v216 if v216 else v9('nan')}
    return v167
v107 = False
v108 = 3
v109 = 4
v110 = 'hide'
v111 = 0
v112 = 0.9
v113 = 2
v114 = v207()
v115 = ('cos1nn', 'heur', 'rare')

def str_parts(v282):
    """anchor, relation-content-words of a pre-grouping string address."""
    v147, v301 = (v282.v871('|', 1) + [''])[:2]
    return (v147, {v210.v1107() for v210 in v787.v1280.v1208(v301) if v210.v1107() not in v787.v1209})

def ident_index(v159):
    """The tape read as places rather than as addresses. None when the pack predates `straddr`."""
    v178 = v159.v513('_ident')
    if v178 is not None:
        return v178
    v311 = v159.v513('straddr')
    if v311 is None:
        return None
    v339 = v511({v166 for v222 in v159['items'] for v166 in v222['slots']})
    v650, v651, v652 = ({}, v547(v510), v547(v510))
    v340 = v547(v510)
    for v166 in v339:
        v514, v581 = v942(v311[v166])
        v650[v166] = (v514, v581)
        v651[v311[v166]].v876(v166)
        v652[v514].v876(v166)
        v340[v514, v159['tape'].v266[v166]].v876(v166)
    v341 = {v166: v217 for v217, v166 in v549(v159['slot_keys_slot'])}
    v178 = {'parts': v650, 'by_str': v439(v651), 'by_anc': v439(v652), 'by_place': v439(v340), 'krow': v341, 'words': {v166: v360(v875(v159['texts'][v166], exclude=v159['tape'].v266[v166])) for v166 in v339}, 'swords': {v166: {v210.v1107() for v210 in v787.v1280.v1208(v159['texts'][v166]) if v210.v1107() not in v787.v1209} - {v159['tape'].v266[v166].v1107()} for v166 in v339}}
    v159['_ident'] = v178
    return v178

def ident_cos(v159, v178, v147, v148, v238='ctx'):
    v342 = v159['ctx_keys'] if v238 == 'ctx' else v159['anc_keys']
    v507, v508 = (v178['krow'].v513(v147), v178['krow'].v513(v148))
    if v342 is None or v507 is None or v508 is None:
        return v9('nan')
    return v9(v342[v507] @ v342[v508])

def identity_question(v159, v343, v242, v344, v152):
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
    v178 = v520(v159)
    if v178 is None:
        return None
    v345 = [v282 for v282 in v178['by_place'].v513((v343, v242), ()) if v282 != v344]
    if not v345:
        v114['no_sibling'] += 1
        return None
    v346 = v345[:v108]
    v653, v654 = (v360(), {v159['straddr'][v282] for v282 in v346})
    for v282 in v346:
        v653 |= v178['parts'][v282][1]
    if v178['parts'][v344][1] & v653 or v159['straddr'][v344] in v654:
        v114['same_words'] += 1
        return None
    v317 = [v282 for v282 in v178['by_anc'].v513(v343, ()) if v282 not in v346 and v282 != v344 and (v159['tape'].v266[v282] != v242) and (not v178['parts'][v282][1] & v653)]
    if v550(v317) < v109 - 1:
        v114['no_intruders'] += 1
        return None
    v317.v616(key=lambda v282: -v576((v1211(v159, v178, v282, v209) for v209 in v346)))
    v155 = [v344] + v317[:v109 - 1]
    v152.v655(v155)
    v114['built'] += 1
    return {'verb': 'lookup', 'ident': True, 'S': v343, 'place': [v343, v242], 'address': f'place:{v343}|{v242}', 'hid': v344, 'straddr': v159['straddr'][v344], 'slots': v510(v346), 'vals': [v159['tape'].v266[v282] for v282 in v346], 'cand_slots': v155, 'cands': [f's{v209}' for v209 in v155], 'label': v155.v649(v344)}

def ident_budget(v159, v144):
    """One import budget for all four worlds - the minimum any candidate can supply."""
    v148 = v144.v513('_ibudget')
    if v148 is None:
        v178 = v520(v159)
        v148 = v595([v111] + [v550([v282 for v282 in v178['by_str'].v513(v159['straddr'][v209], ()) if v282 != v209]) for v209 in v144['cand_slots']])
        v144['_ibudget'] = v148
    return v148

def identity_world(v159, v144, v347):
    """The core plus one candidate mention, scored as one world by the same Phi.

    No new channel and no new head: a place is a set of rows, and Phi already says how well a
    set of rows hangs together. The candidate row is marked as the query row - the same bit a
    completed lookup world carries - so a proposed member is never mistaken for an observed
    one, and the mark is identical across the four worlds, so it cannot carry the label.
    """
    v154 = v510(v144['slots']) + [v347]
    if v111:
        v178 = v520(v159)
        v345 = [v282 for v282 in v178['by_str'].v513(v159['straddr'][v347], ()) if v282 not in v154]
        v154 += v345[:v1210(v159, v144)]
    if v110 == 'hide':
        v141 = [v1102() for v180 in v154]
    else:
        v141 = [v159['tape'].v266[v282] for v282 in v154]
    return {'verb': 'lookup', 'ident': True, 'S': v144['S'], 'slots': v154, 'vals': v141, 'query_row': v550(v154) - 1, 'cands': v144['cands'], 'label': v144['label']}

def ident_rivals(v159, v144):
    """The three rules the mind has to beat. Two are already in the project; one IS the tape.

    cos1nn   nearest candidate to any core row by the write ink - the same 1-NN that beat Phi
             in §18 and the honest ceiling of the encoder alone.
    heur     fp_addresses' own decision, reproduced exactly: min(anchor, context) cosine over
             tau AND at least `overlap` shared content words with some core row. This is the
             rule 293 exists to replace, so it is scored as a rival rather than described.
    rare     the discrete channel by itself: most shared rare words with the core.
    """
    v178 = v520(v159)
    v346 = v144['slots']
    v167, v252 = ({}, {})
    v348 = []
    for v349 in ('cos1nn', 'heur', 'rare'):
        v252[v349] = (v9('-inf'), None)
    v350 = v159.v513('_median')
    if v350 is None:
        v296 = v511((v550(v234) for v234 in v159['postings'].v266()))
        v350 = v296[v550(v296) // 2] if v296 else 1
        v159['_median'] = v350
    for v209 in v144['cand_slots']:
        v656 = v178['words'][v209]
        v552 = v576((v1211(v159, v178, v209, v282) for v282 in v346))
        v657 = v576((v506((1 for v210 in v656 & v178['words'][v282] if v550(v159['postings'].v513(v210, ())) < v350)) / v576(1, v595(v550(v656), v550(v178['words'][v282]))) for v282 in v346))
        v658 = v178['swords'][v209]
        v659 = [v282 for v282 in v346 if v550(v658 & v178['swords'][v282]) >= v113]
        v660 = v576((v595(v1211(v159, v178, v209, v282, 'anc'), v1211(v159, v178, v209, v282)) for v282 in v659), default=v9('-inf'))
        if v660 >= v112:
            v348.v876(v209)
        for v349, v234 in (('cos1nn', v552), ('rare', v657), ('heur', v660)):
            if v234 > v252[v349][0]:
                v252[v349] = (v234, v209)
    for v349, (v234, v209) in v252.v219():
        v167[v349] = f's{v209}' if v209 is not None and (v349 != 'heur' or v234 >= v112) else None
    v167['_heur_accepted'] = v550(v348)
    return v167

def identity_questions_for(v159, v301):
    """Every place the tape can put on trial, plus count and compare as the sanity bolt."""
    v178 = v520(v159)
    v167 = []
    if v178 is None:
        return v167
    for (v343, v242), v154 in v511(v178['by_place'].v219()):
        if v550(v154) < 2:
            v114['singleton_place'] += 1
            continue
        for v166 in v154:
            if (v144 := v1281(v159, v343, v242, v166, v301)) is not None:
                v167.v876(v144)
    for v222 in v159['items']:
        if v550(v222['slots']) >= 2 and (v144 := v1214(v159, v222)) is not None:
            v167.v876(v144)
    return v167

def identity_audit(v159, v301):
    """The minute that decides whether the hour is worth spending. No model, no gradient.

    Two numbers settle it. SUPPLY: how many places the tape can put on trial - §19 and 291 both
    died of a denominator and it costs nothing to look first. CEILING: what the three rules
    already score. A rival near 1.0 means the question is decided before the mind is asked and
    the construction has to change; a rival near the floor means the question may be undecidable
    from this evidence, which is equally worth knowing before training anything.
    """
    v114.v661()
    v224 = [v144 for v144 in v965(v159, v301) if v144.v513('ident')]
    v613, v662 = (v207(), (v520(v159) or {'parts': {}})['parts'])
    for v144 in v224:
        v170 = v144['cands'][v144['label']]
        v663 = v943(v159, v144)
        v613['n'] += 1
        v613['heur_accepted'] += v663.v538('_heur_accepted')
        v613['core_rows'] += v550(v144['slots'])
        for v349, v304 in v663.v219():
            v613[v349] += v10(v304 == v170)
            v613[f'{v349}_answered'] += v10(v304 is not None)
        v664 = {v159['tape'].v266[v282] for v282 in v144['slots']}
        v165 = [v209 for v209 in v144['cand_slots'] if v159['tape'].v266[v209] in v664]
        v613['value_decides'] += v10(v550(v165) == 1 and f's{v165[0]}' == v170)
        v665 = v360()
        for v282 in v144['slots']:
            v665 |= v662[v282][1]
        v613['word_leak'] += v10(v518((v662[v209][1] & v665 for v209 in v144['cand_slots'])))
    v216 = v576(1, v613['n'])
    return {'n_questions': v613['n'], 'floor': 1.0 / v109, 'mean_core_rows': v613['core_rows'] / v216, 'supply': v439(v114), 'rival_cos1nn': v613['cos1nn'] / v216, 'rival_rare': v613['rare'] / v216, 'rival_heuristic': v613['heur'] / v216, 'heuristic_answered': v613['heur_answered'] / v216, 'heuristic_mean_accepted': v613['heur_accepted'] / v216, 'value_identifies_truth': v613['value_decides'] / v216, 'word_overlap_leak': v613['word_leak'] / v216}
v116 = [0, 0]
v117 = 1
v118 = 'thin'

def view_of(v144, v152, v351):
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
    v352 = v144['query_row']
    v156 = [v217 for v217 in v521(v352) if v152.v882() < v351]
    v353 = {v144['vals'][v217] for v217 in v156}
    for v209 in v144['cands']:
        v187 = [v217 for v217 in v521(v352) if v144['vals'][v217] == v209]
        if v209 not in v353 and v187:
            v156.v876(v152.v1108(v187))
    v156 = v511(v360(v156))
    v167 = {**v144, 'slots': [v144['slots'][v217] for v217 in v156] + [v144['slots'][v352]], 'vals': [v144['vals'][v217] for v217 in v156] + [v144['vals'][v352]], 'query_row': v550(v156)}
    v167.v538('ladder', None)
    v167.v538('_base', None)
    return v167

def region_views_of(v144, v216):
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
    v352 = v144['query_row']
    v234 = v595(v216, v352)
    v354 = [v944(v217 * v352 / v234) for v217 in v521(v234 + 1)]
    v167 = []
    for v147, v148 in v666(v354, v354[1:]):
        v667 = {**v144, 'slots': v144['slots'][v147:v148] + [v144['slots'][v352]], 'vals': v144['vals'][v147:v148] + [v144['vals'][v352]], 'query_row': v148 - v147}
        v667.v538('ladder', None)
        v667.v538('_base', None)
        v167.v876(v667)
    return v167

def views_and_mask(v144, v152, v190):
    """The question's views plus the candidate-presence mask, under either mode.

    View 0 is always the FULL question, so at VIEWS=1 this is exactly the old single pass and
    the ensemble is a strict superset of the information the single pass had. The mask says
    which candidates each region actually has a witness for; None in thin mode, where view_of
    guarantees every candidate a witness and no masking is needed (or possible - that guarantee
    is what keeps thin views label-tight).
    """
    if v118 == 'thin':
        return ([v144] + [v1282(v144, v152, 1.0 - v119) for v180 in v521(v117 - 1)], None)
    v355 = [v144] + v945(v144, v117)
    v220 = v706.v601([[v9(v209 in v360(v234['vals'][:v234['query_row']])) for v209 in v144['cands']] for v234 in v355[1:]], device=v190)
    return (v355, v220)

def pool_views(v356, v220):
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
    if v220 is None:
        return v356.v946(0)
    v357 = (v356[1:] * v220).v506(1) / v220.v506(1)
    return v356[0] - v356[0].v946() + ((v356[1:] - v357.v1120(1)) * v220).v506(0)

def disagreement(v356, v358=None):
    """Generalised Jensen-Shannon divergence of the per-view answer distributions: the mean KL
    of each view to their mixture. Zero when the views agree exactly; label-free by
    construction, so thresholding on it is never conditioning on the outcome.

    With a mask (region mode), each view's distribution lives on ITS candidates - masked
    softmax puts exact zeros elsewhere, and the JS stays finite because 0*log(0/m) = 0 and the
    mixture covers every candidate some view supports. Two regions that put their mass on
    values the other never wrote disagree maximally, which is correct: that address is
    contested across the corpus, and D is the number that says so."""
    if v358 is not None:
        v356 = v356.v890(v358 == 0, v9('-inf'))
    v225 = v706.v608(v356, dim=1)
    v138 = v225.v946(0).v668(1e-09)
    return v9((v225 * (v225.v668(1e-09).v363() - v138.v363())).v506(1).v946())

def reconciled(v196, v159, v144, v190, v189, v152):
    """Pooled logits, the single full-pass logits, and D, for one question. Training takes the
    gradient through the pooled logits; the exam reads all three. In thin mode D is over all
    views (view 0 included, as recon3 measured it); in region mode D is over the REGIONS only -
    the full view is their union and would only dilute the cross-region signal."""
    v355, v220 = v669(v144, v152, v190)
    v356 = v706.v541([v703(v196, v159, v227, v190, v189) for v227 in v355])
    return (v947(v356, v220), v356[0], v948(v356 if v220 is None else v356[1:], v220))
v119 = 0.0

def drop_rows(v144, v152, v351):
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
    v352 = v144['query_row']
    v170 = v144['cands'][v144['label']]
    v359 = v510(v521(v352))
    if not v518((v144['vals'][v217] == v170 for v217 in v359)):
        return None
    v156 = [v217 for v217 in v359 if v152.v882() < v351]
    if not v518((v144['vals'][v217] == v170 for v217 in v156)):
        v156.v876(v152.v1108([v217 for v217 in v359 if v144['vals'][v217] == v170]))
    v156 = v511(v360(v156))
    v155 = v511({v144['vals'][v217] for v217 in v156})
    if v15 in v144['cands']:
        v155 = v155 + [v15]
    if v550(v155) < 2:
        return None
    v116[0] += v550(v156)
    v116[1] += v550(v359)
    v167 = {**v144, 'slots': [v144['slots'][v217] for v217 in v156] + [v144['slots'][v352]], 'vals': [v144['vals'][v217] for v217 in v156] + [v144['vals'][v352]], 'cands': v155, 'label': v155.v649(v170), 'query_row': v550(v156)}
    v167.v538('ladder', None)
    v167.v538('_base', None)
    return v167
v120 = ('near', 'middle', 'far')
v121 = True
v122 = 2
v123 = ('same', 'cos', 'rare')
v124 = ('anchor', 'rel')
v125 = v360(v123)
v126 = 'mean'
v127 = 'arc'
v128 = 'ascii'
v129 = 2.9701

def tau_for_density(v361, v362, v363, v364=0.0, v365=0.9995):
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
    v366 = {}

    def resolve(v670, v189, v233, v671, v672):
        if 'tau' in v366:
            return v366['tau']

        def density(v250):
            v167, v315 = v787.v1109(v670, v189, v250, v233, v671, addr_key=v672)
            return (v550(v167) / v550(v315) if v315 else v9('nan'), v550(v315))
        v426 = v777.v777()
        v949, v950 = v951(v364)
        v952, v953 = v951(v365)
        v673 = [(v364, v949, v950), (v365, v952, v953)]
        v674 = v949 > v952
        v147, v148 = (v364, v365)
        if not v674:
            v363(f'  tau calibration: density NOT decreasing in tau ({v364}->{v949:.3f}, {v365}->{v952:.3f}) - bisection is not valid here')
        elif not v952 <= v361 <= v949:
            v363(f'  tau calibration: target {v361:.4f} outside the bracket [{v952:.3f}, {v949:.3f}] - clamping to the nearer end')
        else:
            for v180 in v521(v362):
                v138 = 0.5 * (v147 + v148)
                v1283, v1284 = v951(v138)
                v673.v876((v138, v1283, v1284))
                if v1283 > v361:
                    v147 = v138
                else:
                    v148 = v138
        v252 = v595(v673, key=lambda v301: v1203(v301[1] - v361) if v301[1] == v301[1] else v9('inf'))
        v366['tau'] = v252[0]
        v366['trace'] = [{'tau': v944(v250, 5), 'density': v205, 'addresses': v216} for v250, v205, v216 in v673]
        v366['achieved'] = v252[1]
        v366['monotone'] = v674
        v363(f'  tau calibrated: {v252[0]:.4f} -> density {v252[1]:.4f} (target {v361:.4f}, {v252[2]} addresses, {v550(v673)} probes, {v777.v777() - v426:.0f}s)')
        return v252[0]
    v367.v366 = v366
    return v367
v130 = [0, 0]
v131 = True
v132 = []
v133 = [0, 0, 0]
v134 = [0, 0, 0]
v135 = [0.0, 0.0, 0]

def attach_ladder(v145, v144, v174, v175, v152):
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
    v368 = v360(v144['cands'])
    v177 = {}
    v343 = v954.v675(v144['address'])
    v345 = [v222 for v222 in v174.v513(v343, ()) if v222['address'] != v144['address']]
    for v222 in v152.v892(v345, v550(v345)) if v345 else ():
        v644 = [v145['tape'].v266[v166] for v166 in v222['slots']]
        v644 = [v234 for v234 in v644 if v234 not in v368]
        if v644:
            v177['near'] = v644[0]
            v368.v1101(v644[0])
            break
    v369 = v576(v144['slots']) + 1
    for v166 in (v369, v595(v144['slots']) - 1):
        if 0 <= v166 < v145['n_slots'] and v145['tape'].v266[v166] not in v368:
            v177['middle'] = v145['tape'].v266[v166]
            v368.v1101(v177['middle'])
            break
    for v180 in v521(8):
        v234 = v175[v152.v870(v550(v175))]
        if v234 not in v368:
            v177['far'] = v234
            break
    v144['ladder'] = v177 if v121 and v550(v177) == 3 else {}
    return v144

def lookup_rival(v144):
    """286's majority rival - over the SURVIVORS only.

    The query row now sits in vals carrying a sentinel that equals nothing. Counting it would
    let the sentinel win any all-distinct address and hand the rival a guaranteed miss, which
    would flatter the mind against an opponent crippled by our own bookkeeping.
    """
    v359 = [v234 for v217, v234 in v549(v144['vals']) if v217 != v144['query_row']]
    return v207(v359).v562(1)[0][0]

def own_row_rival(v144):
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
    v168 = v144.v513('own_rows') or v360()
    v370 = [v144['vals'][v217] for v217 in v521(v144['query_row']) if v144['slots'][v217] in v168]
    if v370:
        return v207(v370).v562(1)[0][0]
    return v15 if v13 else v955(v144)

def counting_margin(v144):
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
    v359 = [v234 for v217, v234 in v549(v144['vals']) if v217 != v144['query_row']]
    v209 = v207(v359).v562(2)
    return (v209[0][1] - (v209[1][1] if v550(v209) > 1 else 0)) / v576(1, v550(v359))

def lookup_rival_cos(v159, v144, v189, v190):
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
    v330 = v159.v512('_ctx', {})

    def ctx(v166):
        if v166 not in v330:
            v209 = v189.v1106(v159['texts'][v166], exclude=v159['tape'].v266[v166])
            v330[v166] = v1119.v1094(v209, dim=-1) if v209 is not None else None
        return v330[v166]
    v244 = v144['query_row']
    v331 = v645(v144['slots'][v244])
    if v331 is None:
        return (None, v9('nan'))
    v171 = [(v217, v645(v166)) for v217, v166 in v549(v144['slots']) if v217 != v244]
    v171 = [(v217, v209) for v217, v209 in v171 if v209 is not None]
    if not v171:
        return (None, v9('nan'))
    v220 = v706.v541([v209 for v180, v209 in v171], 0)
    v282 = v220 @ v331
    v295 = v10(v282.v934())
    v252 = v144['vals'][v171[v295][0]]
    v332 = [v9(v282[v236]) for v236, (v217, v180) in v549(v171) if v144['vals'][v217] != v252]
    v371 = v9(v282[v295]) - (v576(v332) if v332 else -1.0)
    return (v252, v371)

class Deriver(v372.v136):
    """One body, one scalar. The mind describes a world; the algebra does the arithmetic.

    The body is 286/289a's relational net verbatim: edges carry the same-value indicator and
    two context ranks, nodes carry shares and indicators, identity has nowhere to live. Phi
    pools the whole graph to one number - how well this world hangs together - and that is the
    only trained readout left. The count and compare heads are gone because their tasks moved
    into exact algebra where the invariant says they belong; the interference they caused
    (count 0.965 -> 0.903 as lookup grew) is removed by construction, not compensated.
    """
    v373 = True
    v374 = False

    def __init__(v676, v190, v205: v10=32, v677: v10=3, v678: v10=8, v679: v10=0):
        v1212().v956()
        v140 = 3 if v676.v373 else 2
        v676.v680 = v372.v1213(v372.v1285(v677, v205), v372.v1286()).v702(v190)
        v676.v681 = v372.v1213(v372.v1285(v678 + v140 * v205, v205), v372.v1286()).v702(v190)
        if v679:
            with v706.v404():
                v676.v680[0].v958[:, v677 - v679:] = 0.0
        v676.v682 = v372.v1213(v372.v1285((2 if v676.v373 else 1) * v205, v205), v372.v1286(), v372.v1285(v205, 1)).v702(v190)
        v372.v1110.v957(v676.v682[-1].v958)
        v372.v1110.v957(v676.v682[-1].v959)

    def body(v676, v384, v143, v389):
        v561 = v676.v680(v384)
        if v676.v374:
            v935 = v706.v553(v389.v1114[0], v561.v1114[-1], device=v389.v190, dtype=v389.v918)
            v650 = [v389, v935, v935] + ([v935] if v676.v373 else [])
            return v676.v681(v706.v705(v650, -1))
        v168 = (v561 * v143).v506(1) / v143.v506(1).v1111(min=1.0)
        v650 = [v389, v168, v561.v946(1)]
        if v676.v373:
            v650.v876(v561.v576(1).v266)
        return v676.v681(v706.v705(v650, -1))

    def phi(v676, v384, v143, v389):
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
        v319 = v676.v960(v384, v143, v389)
        v683 = v706.v705([v319.v946(0), v319.v576(0).v266], -1) if v676.v373 else v319.v946(0)
        return v676.v682(v683).v961(-1)

def sparse_questions_for(v159, v301):
    """290's question set: the addresses the dense verb throws away.

    count and compare still come from the dense items - they are exact algebra, they cost
    nothing, and dropping them would remove the sanity bolt that fires if the tape and the
    arithmetic disagree.
    """
    v167 = []
    for v222 in v159['items']:
        if not 1 <= v550(v222['slots']) <= 2:
            continue
        for v153 in v521(v550(v222['slots'])):
            if (v144 := v1287(v159, v222, v301, v153, v12)) is not None:
                v167.v876(v144)
    for v222 in v159['items']:
        if v550(v222['slots']) >= 2 and (v144 := v1214(v159, v222)) is not None:
            v167.v876(v144)
    v375 = [v222 for v222 in v159['items'] if v550(v222['slots']) >= 2]
    v301.v655(v375)
    for v147, v148 in v666(v375[::2], v375[1::2]):
        if (v144 := v968(v159, v147, v148)) is not None:
            v167.v876(v144)
    return v167

def open_questions_for(v159, v301):
    """292's set. count and compare stay: they are exact algebra and they are the sanity bolt."""
    if v25:
        v530 = v1112(v159) if v19 == 'anchor' else [v222 for v222 in v159['items'] if v550(v222['slots']) >= 2]
        v684 = v510(v159['tape'].v266)
        v167 = []
        for v222 in v530:
            for v153 in v521(v550(v222['slots'])):
                if (v144 := v1323(v159, v222, v301, v153, v684)) is not None:
                    v167.v876(v144)
        for v222 in v159['items']:
            if v550(v222['slots']) >= 2 and (v144 := v1214(v159, v222)) is not None:
                v167.v876(v144)
        return v167
    if v19 == 'anchor' or v20 == 'uniform':
        v530 = v1112(v159) if v19 == 'anchor' else [v222 for v222 in v159['items'] if v550(v222['slots']) >= 2]
        v684 = v510(v159['tape'].v266)
        v167 = []
        for v222 in v530:
            for v153 in v521(v550(v222['slots'])):
                if (v144 := v647(v159, v222, v301, v153, v684)) is not None:
                    v167.v876(v144)
        for v222 in v159['items']:
            if v550(v222['slots']) >= 2 and (v144 := v1214(v159, v222)) is not None:
                v167.v876(v144)
        v375 = [v222 for v222 in v159['items'] if v550(v222['slots']) >= 2]
        v301.v655(v375)
        for v147, v148 in v666(v375[::2], v375[1::2]):
            if (v144 := v968(v159, v147, v148)) is not None:
                v167.v876(v144)
        return v167
    v219 = [v222 for v222 in v159['items'] if v550(v222['slots']) >= 2]
    v174 = v547(v510)
    for v222 in v219:
        v174[v954.v675(v222['address'])].v876(v222)
    v175 = v510(v159['tape'].v266)
    v167 = []
    for v222 in v219:
        for v153 in v521(v550(v222['slots'])):
            if (v144 := v1288(v159, v222, v301, v153, v174, v175)) is not None:
                v167.v876(v144)
        if (v144 := v1214(v159, v222)) is not None:
            v167.v876(v144)
    v375 = v510(v219)
    v301.v655(v375)
    for v147, v148 in v666(v375[::2], v375[1::2]):
        if (v144 := v968(v159, v147, v148)) is not None:
            v167.v876(v144)
    return v167

def questions_for(v159, v301):
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
    if v31:
        return v962(v159, v301)
    if v98:
        return v963(v159, v301)
    if v75:
        return v964(v159, v301)
    if v107:
        return v965(v159, v301)
    if v16:
        return v966(v159, v301)
    if v12:
        return v967(v159, v301)
    v219 = [v222 for v222 in v159['items'] if v550(v222['slots']) >= 2]
    v174 = v547(v510)
    for v222 in v219:
        v174[v954.v675(v222['address'])].v876(v222)
    v175 = v510(v159['tape'].v266)
    v167 = []
    for v222 in v219:
        if (v144 := v1214(v159, v222)) is not None:
            v167.v876(v144)
        for v153 in v521(v550(v222['slots'])):
            if (v144 := v1289(v159, v222, v301, hid=v153)) is not None:
                v167.v876(v519(v159, v144, v174, v175, v301))
    v375 = v510(v219)
    v301.v655(v375)
    for v147, v148 in v666(v375[::2], v375[1::2]):
        v144 = v968(v159, v147, v148)
        if v144 is not None:
            v167.v876(v144)
    return v167

def n_choices(v144) -> v10:
    return v550(v144['cands']) if v144['verb'] == 'lookup' else v550(v5) if v144['verb'] == 'count' else v550(v6)

def truth_of(v144):
    return v144['cands'][v144['label']] if v144['verb'] == 'lookup' else v144['label']

def by_value(v159):
    """value -> its slots, once per pack. Also the test for "is this token on the tape at all",
    which the copy lane needs: a token with no slot has no rows and cannot be scored."""
    v254 = v159.v513('_by_value')
    if v254 is None:
        v254 = v547(v510)
        for v166, v234 in v549(v159['tape'].v266):
            v254[v234].v876(v166)
        v159['_by_value'] = v254
    return v254

def outside_mentions(v159, v144, v242):
    """Mentions of a value that are NOT already in this question's evidence."""
    v164 = v360(v144['slots'])
    return [v166 for v166 in v572(v159).v513(v242, ()) if v166 not in v164]

def shared_import_budget(v159, v144, v266):
    """One budget for every world compared in a question, and the reason is a leak.

    A local candidate's mentions are already IN the evidence, so it usually has nothing left to
    import; a ladder rung comes from elsewhere and always has K. Give each world what it
    happens to have and Phi can read "imported rows present" as "this one is wrong" - the
    landscape gate would then pass on a bookkeeping tell rather than on distance. The budget is
    therefore the minimum available across everything being scored, so every completed world
    carries the same number of rows.
    """
    return v595([v122] + [v550(v897(v159, v144, v234)) for v234 in v266 if v234 != v15])

def row_meta(v159):
    """slot -> (anchor id, relation id), for 290's two edge channels. Integers, not strings, so
    the channels are one broadcast comparison rather than n^2 python string compares."""
    v138 = v159.v513('_rowmeta')
    if v138 is None:
        v969, v970, v138 = ({}, {}, {})
        for v222 in v159['items']:
            v514, v515 = v516(v222['address'])
            v147 = v969.v512(v514, v550(v969))
            v301 = v970.v512(v515, v550(v970)) if v515 else -1
            for v166 in v222['slots']:
                v138[v166] = (v147, v301)
        v159['_rowmeta'] = v138
    return v138

def graph_base(v159, v144, v189, v190):
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
    if '_base' in v144:
        return v144['_base']
    v154 = v510(v144['slots'])
    v216 = v550(v154)
    v330, v685 = (v159.v512('_ctx', {}), v159.v512('_words', {}))
    v376 = v449(v159.v513('frame_mode') and v159.v513('frame_fps') is not None)
    for v166 in v360(v154):
        if v166 not in v685:
            v685[v166] = v360(v875(v159['texts'][v166], exclude=v159['tape'].v266[v166]))
        if not v376 and v166 not in v330:
            v209 = v189.v1106(v159['texts'][v166], exclude=v159['tape'].v266[v166])
            v330[v166] = v1119.v1094(v209, dim=-1) if v209 is not None else None
    v350 = v159.v513('_median')
    if v350 is None:
        v296 = v511((v550(v234) for v234 in v159['postings'].v266()))
        v350 = v296[v550(v296) // 2] if v296 else 1
        v159['_median'] = v350
    v377 = [v685[v282] for v282 in v154]
    v378 = [v159['frame_fps'][v282] if v282 < v550(v159['frame_fps']) else None for v282 in v154] if v376 else [v330[v282] for v282 in v154]
    v393, v394 = (v706.v553(v216, v216), v706.v553(v216, v216))
    if v131 and v216 > 1 and v518((v209 is not None for v209 in v378)):
        v686 = v971((v209 for v209 in v378 if v209 is not None))
        v687 = v706.v541([v209 if v209 is not None else v706.v982(v686) for v209 in v378])
        v688 = v706.v1215.v1113.v972.v689
        v706.v1215.v1113.v972.v689 = False
        v690 = (v687 @ v687.v1324).v9().v973()
        v706.v1215.v1113.v972.v689 = v688
        v691 = v706.v601([v209 is None for v209 in v378])
        v690[v691, :] = 0.0
        v690[:, v691] = 0.0
        v690.v701(0.0)
        v393 = v690
    for v217 in v521(v216):
        for v236 in v521(v217 + 1, v216):
            if not v131 and v378[v217] is not None and (v378[v236] is not None):
                v393[v217, v236] = v393[v236, v217] = v9(v378[v217] @ v378[v236])
            v974 = v377[v217] & v377[v236]
            v657 = v506((1 for v210 in v974 if v550(v159['postings'].v513(v210, ())) < v350))
            v394[v217, v236] = v394[v236, v217] = v657 / v576(1, v595(v550(v377[v217]), v550(v377[v236])))
    v379 = v706.v692(v216, v216, offset=1)
    if v379.v693():
        v130[0] += v10((v394[v379[0], v379[1]] > 0).v506())
        v130[1] += v10(v379.v1114[1])
        v664 = v393[v379[0], v379[1]]
        v135[0] += v9(v664.v506())
        v135[1] += v9((v664 * v664).v506())
        v135[2] += v10(v664.v693())

    def rank_norm(v220):
        if v379.v693() == 0:
            return v220
        v234 = v220[v379[0], v379[1]]
        v226 = v234.v975()
        v301 = v706.v976(v226, dtype=v706.v986)
        v301[v226] = v706.v977(v550(v234), dtype=v706.v986)
        v978, v979 = v234.v980(return_inverse=True)
        if v550(v978) > 1:
            v981 = v706.v553(v550(v978)).v1115(0, v979, v301, 'mean', include_self=False)
            v301 = v981[v979] / (v550(v234) - 1 if v550(v234) > 1 else 1)
        else:
            v301 = v706.v982(v301)
        v214 = v706.v982(v220)
        v214[v379[0], v379[1]] = v301
        v214[v379[1], v379[0]] = v301
        return v214
    v380 = [v1116(v393) if 'cos' in v125 else v706.v982(v393), v1116(v394) if 'rare' in v125 else v706.v982(v394)]
    if v12:
        v694 = v983(v159)
        v147 = v706.v601([v694.v513(v282, (-1, -1))[0] for v282 in v154])
        v301 = v706.v601([v694.v513(v282, (-1, -2))[1] for v282 in v154])
        v695 = ((v147[:, None] == v147[None, :]) & (v147[:, None] >= 0)).v9()
        v696 = ((v301[:, None] == v301[None, :]) & (v301[:, None] >= 0)).v9()
        v695.v701(0.0)
        v696.v701(0.0)
        v380 += [v695 if 'anchor' in v125 else v706.v982(v695), v696 if 'rel' in v125 else v706.v982(v696)]
        v697 = v706.v692(v216, v216, offset=1)
        if v697.v693():
            v133[0] += v10((v695[v697[0], v697[1]] > 0).v506())
            v133[1] += v10((v696[v697[0], v697[1]] > 0).v506())
            v133[2] += v10(v697.v1114[1])
    v244 = v144.v513('query_row', -1)
    v395, v396 = (v706.v553(v216), 0.0)
    if v244 >= 0 and v244 < v216 and (v378[v244] is not None):
        v698 = [v217 for v217 in v521(v216) if v378[v217] is not None and v217 != v244]
        if v550(v698) > 1:
            v163 = v706.v601(v698)
            v984 = v393[v163, v244]
            v667 = v984.v975()
            v931 = v706.v1117(v550(v698))
            v931[v667] = v706.v977(v550(v698), dtype=v706.v986)
            v395[v163] = v931 / (v550(v698) - 1)
            v295 = v10(v984.v934())
            v985 = v144['vals'][v698[v295]]
            v332 = [v9(v984[v338]) for v338, v217 in v549(v698) if v144['vals'][v217] != v985]
            v333 = v9(v984.v576() - v984.v595())
            if v333 > 1e-09:
                v396 = (v9(v984.v576()) - v576(v332)) / v333 if v332 else 1.0
            else:
                v396 = 1.0 if not v332 else 0.0
    v381 = v144.v513('n_first', v216)
    v382 = [v144['S'].v1107() if v217 < v381 else v144.v513('S2', v144['S']).v1107() for v217 in v521(v216)]
    v168 = v144.v513('own_rows')
    v324 = {'n': v216, 'slots': v154, 'chans': v380, 'qcos': v395, 'qmargin': v396, 'subj': v382, 'nfirst': v381, 'qrow': v244, 'qrows': v511(v360(v144.v513('query_rows') or ([v244] if v244 >= 0 else []))), 'isown': [v9(v282 in v168) for v282 in v154] if v168 is not None else None}
    v134[0] += v216
    v134[1] = v576(v134[1], v216)
    v134[2] += 1
    v144['_base'] = v324
    return v324

def graph_from_base(v159, v144, v189, v190, v383):
    """One completed world, from the cached base. Only `same` and the count share change."""
    v148 = v699(v159, v144, v189, v190)
    v216, v154, v244 = (v148['n'], v148['slots'], v148['qrow'])
    v141 = v510(v144['vals'])
    if v383 is not None:
        v141[v244] = v383
    v700, v327 = ([], {})
    for v234 in v141:
        v700.v876(v327.v512(v234 if v631(v234, v8) else v1129(v234), v550(v327)))
    v250 = v706.v601(v700)
    v143 = (v250[:, None] == v250[None, :]).v9()
    v143.v701(0.0)
    v384 = v706.v541([v143 if 'same' in v125 else v706.v982(v143)] + v148['chans'], -1).v702(v190)
    v385 = v207(v141)
    v386 = v159.v513('frame_nfill') if v159.v513('frame_mode') else None
    v387 = v159.v513('frame_nfill_max', 1) if v386 is not None else 1
    v388 = v360(v148['qrows'])
    v389 = [[v385[v141[v217]] / v216 if v217 not in v388 or v383 is not None else 0.0, v9(v148['subj'][v217] in v159['texts_lc'][v154[v217]]), v9(v217 >= v148['nfirst']), 1.0 / v216, v9(v217 in v388), 0.0, v9(v148['qcos'][v217]), v148['qmargin']] + ([v1097.v898(v386[v154[v217]]) / v1097.v898(v387) if v154[v217] < v550(v386) else 0.0] if v386 is not None else []) + ([v9(v144.v513('home_cos', 0.0)) if v217 == v244 else 0.0] if v88 else []) + ([v9(v144.v513('confirm', 0.0)) if v217 == v244 else 0.0] if v83 else []) + v1118(v144, v217, v244) for v217 in v521(v216)]
    if v148['isown'] is not None:
        for v217 in v521(v216):
            v389[v217].v876(v148['isown'][v217])
    v389 = v706.v601(v389, dtype=v706.v986, device=v190)
    return (v384, v143.v1120(-1).v702(v190), v389)

def build_graph(v159, v144, v189, v190, v383=None, v390=None):
    """286/289a's graph verbatim, plus the side indicator COMPARE needs and, for a completed
    world, the candidate's own mentions imported from elsewhere on the tape."""
    v391 = v122 if v390 is None else v390
    if v391 == 0 and (not v14) and (v383 not in (None, v15)):
        pass
    elif v391 == 0 and v383 != v15:
        return v987(v159, v144, v189, v190, v383)
    if v391 == 0 and v383 == v15:
        return v987(v159, v144, v189, v190, None)
    v154, v141 = (v144['slots'], v144['vals'])
    v392 = v550(v154)
    if v383 is not None:
        v154, v141 = (v510(v154), v510(v141))
        v141[v144['query_row']] = v383
        v140 = v122 if v390 is None else v390
        for v166 in v897(v159, v144, v383)[:v140]:
            v154.v876(v166)
            v141.v876(v159['tape'].v266[v166])
    v216 = v550(v154)
    v330, v685 = (v159.v512('_ctx', {}), v159.v512('_words', {}))
    v376 = v449(v159.v513('frame_mode') and v159.v513('frame_fps') is not None)
    for v166 in v360(v154):
        if v166 not in v685:
            v685[v166] = v360(v875(v159['texts'][v166], exclude=v159['tape'].v266[v166]))
        if not v376 and v166 not in v330:
            v209 = v189.v1106(v159['texts'][v166], exclude=v159['tape'].v266[v166])
            v330[v166] = v1119.v1094(v209, dim=-1) if v209 is not None else None
    v350 = v159.v513('_median')
    if v350 is None:
        v296 = v511((v550(v234) for v234 in v159['postings'].v266()))
        v350 = v296[v550(v296) // 2] if v296 else 1
        v159['_median'] = v350
    v377 = [v685[v282] for v282 in v154]
    v378 = [v159['frame_fps'][v282] if v282 < v550(v159['frame_fps']) else None for v282 in v154] if v376 else [v330[v282] for v282 in v154]
    v143 = v706.v553(v216, v216)
    v393 = v706.v553(v216, v216)
    v394 = v706.v553(v216, v216)
    if v131 and v216 > 1 and v518((v209 is not None for v209 in v378)):
        v686 = v971((v209 for v209 in v378 if v209 is not None))
        v687 = v706.v541([v209 if v209 is not None else v706.v982(v686) for v209 in v378])
        v688 = v706.v1215.v1113.v972.v689
        v706.v1215.v1113.v972.v689 = False
        v690 = (v687 @ v687.v1324).v9().v973()
        v706.v1215.v1113.v972.v689 = v688
        v691 = v706.v601([v209 is None for v209 in v378])
        v690[v691, :] = 0.0
        v690[:, v691] = 0.0
        v690.v701(0.0)
        v393 = v690
    for v217 in v521(v216):
        for v236 in v521(v217 + 1, v216):
            v143[v217, v236] = v143[v236, v217] = v9(v141[v217] == v141[v236])
            if not v131 and v378[v217] is not None and (v378[v236] is not None):
                v393[v217, v236] = v393[v236, v217] = v9(v378[v217] @ v378[v236])
            v974 = v377[v217] & v377[v236]
            v657 = v506((1 for v210 in v974 if v550(v159['postings'].v513(v210, ())) < v350))
            v394[v217, v236] = v394[v236, v217] = v657 / v576(1, v595(v550(v377[v217]), v550(v377[v236])))
    v379 = v706.v692(v216, v216, offset=1)
    if v379.v693():
        v130[0] += v10((v394[v379[0], v379[1]] > 0).v506())
        v130[1] += v10(v379.v1114[1])
        v664 = v393[v379[0], v379[1]]
        v135[0] += v9(v664.v506())
        v135[1] += v9((v664 * v664).v506())
        v135[2] += v10(v664.v693())

    def rank_norm(v220):
        if v379.v693() == 0:
            return v220
        v234 = v220[v379[0], v379[1]]
        v226 = v234.v975()
        v301 = v706.v976(v226, dtype=v706.v986)
        v301[v226] = v706.v977(v550(v234), dtype=v706.v986)
        v978, v979 = v234.v980(return_inverse=True)
        if v550(v978) > 1:
            v981 = v706.v553(v550(v978)).v1115(0, v979, v301, 'mean', include_self=False)
            v301 = v981[v979] / (v550(v234) - 1 if v550(v234) > 1 else 1)
        else:
            v301 = v706.v982(v301)
        v214 = v706.v982(v220)
        v214[v379[0], v379[1]] = v301
        v214[v379[1], v379[0]] = v301
        return v214
    v384 = v706.v541([v143 if 'same' in v125 else v706.v982(v143), v1116(v393) if 'cos' in v125 else v706.v982(v393), v1116(v394) if 'rare' in v125 else v706.v982(v394)], -1).v702(v190)
    v385 = v207(v141)
    v381 = v144.v513('n_first', v216)
    v244 = v144.v513('query_row', -1)
    v382 = [v144['S'].v1107() if v217 < v381 or v217 >= v392 else v144.v513('S2', v144['S']).v1107() for v217 in v521(v216)]
    v395 = v706.v553(v216)
    v396 = 0.0
    if v244 >= 0 and v378[v244] is not None:
        v698 = [v217 for v217 in v521(v216) if v378[v217] is not None and v217 != v244]
        if v550(v698) > 1:
            v163 = v706.v601(v698)
            v984 = v393[v163, v244]
            v667 = v984.v975()
            v301 = v706.v1117(v550(v698))
            v301[v667] = v706.v977(v550(v698), dtype=v706.v986)
            v395[v163] = v301 / (v550(v698) - 1)
            v295 = v10(v984.v934())
            v985 = v141[v698[v295]]
            v332 = [v9(v984[v140]) for v140, v217 in v549(v698) if v141[v217] != v985]
            v333 = v9(v984.v576() - v984.v595())
            if v333 > 1e-09:
                v396 = (v9(v984.v576()) - v576(v332)) / v333 if v332 else 1.0
            else:
                v396 = 1.0 if not v332 else 0.0
    v389 = v706.v601([[v385[v141[v217]] / v216 if v217 != v244 or v383 is not None else 0.0, v9(v382[v217] in v159['texts_lc'][v154[v217]]), v9(v217 >= v381), 1.0 / v216, v9(v217 == v244), v9(v217 >= v392), v9(v395[v217]), v396] + ([v1097.v898(v159['frame_nfill'][v154[v217]]) / v1097.v898(v159.v513('frame_nfill_max', 1)) if v154[v217] < v550(v159['frame_nfill']) else 0.0] if v159.v513('frame_mode') and v159.v513('frame_nfill') is not None else []) + ([v9(v144.v513('home_cos', 0.0)) if v217 == v244 else 0.0] if v88 else []) + ([v9(v144.v513('confirm', 0.0)) if v217 == v244 else 0.0] if v83 else []) + v1118(v144, v217, v244) for v217 in v521(v216)], dtype=v706.v986, device=v190)
    return (v384, v143.v1120(-1).v702(v190), v389)

def ladder_scores_for(v196, v159, v144, v190, v189):
    """Phi on the three wrong worlds, in ladder order. Empty when the tape could not supply one."""
    if not v144.v513('ladder'):
        return None
    v140 = v540(v159, v144, v510(v144['cands']) + [v144['ladder'][v301] for v301 in v120])
    v397 = []
    for v398 in v120:
        v384, v143, v389 = v539(v159, v144, v189, v190, query_value=v144['ladder'][v398], import_k=v140)
        v397.v876(v196.v542(v384, v143, v389))
    return v706.v541(v397)

def cand_logits_for(v196, v159, v144, v190, v189):
    """Score one completed world per candidate and let them compete.

    This is 288's repair loop turned inward: instead of preferring a group, the mind writes the
    conjecture into the query row, reads the world that results, and says how well it hangs
    together. The query-row indicator stays set, so a completed world is never mistaken for an
    observed one - the conjecture is marked as a conjecture, which is the derived-slot
    discipline applied to reading.
    """
    if v144.v513('ident'):
        return v706.v541([v196.v542(*v539(v159, v1335(v159, v144, v282), v189, v190, query_value=None, import_k=0)) for v282 in v144['cand_slots']])
    v141 = v510(v144['cands']) + [v144['ladder'][v301] for v301 in v120] if v144.v513('ladder') else v510(v144['cands'])
    v140 = v540(v159, v144, v141)
    v397 = []
    for v209 in v144['cands']:
        v384, v143, v389 = v539(v159, v144, v189, v190, query_value=v209, import_k=v140)
        v397.v876(v196.v542(v384, v143, v389))
    return v706.v541(v397)

def loss_for(v196, v159, v144, v190, v189):
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
    if v144.v513('cons'):
        return v988(v196, v159, v144, v190, v189)
    if v144.v513('pair'):
        return v989(v196, v159, v144, v190, v189)
    if v144.v513('reach'):
        return v990(v196, v159, v144, v190, v189)
    if v144['verb'] != 'lookup':
        raise v509(f"{v144['verb']} is exact algebra now and has no loss")
    if v27 and v144.v513('mixed'):
        return v991(v196, v159, v144, v190, v189)
    v399 = v703(v196, v159, v144, v190, v189)
    if v17 == 'reward':
        v403 = v706.v608(v399, 0)
        v214 = v706.v992(v403, -1.0)
        v214[v144['label']] = 1.0
        if (v13 or v144.v513('mixed')) and v144.v513('answerable') and (v15 in v144['cands']):
            v214[v144['cands'].v649(v15)] = 0.75
        return -(v403 * v214).v506()
    v400 = v704(v196, v159, v144, v190, v189)
    if v400 is None:
        return v1119.v993(v399.v1120(0), v706.v601([v144['label']], device=v190))
    v401 = v706.v705([v399, v400])
    v402 = -(v401[v144['label']] - v706.v1121(v401, 0))
    for v140 in v521(v550(v120) - 1):
        v402 = v402 - (v400[v140] - v706.v1121(v400[v140:], 0))
    return v402

@v706.v404()
def predict_with_confidence(v196, v159, v144, v190, v189):
    """What would be said, how sure, and what the tape says - the three things an audit needs.

    The exact verbs answer through the algebra with confidence 1.0, which is not flattery: it
    is the honest statement that a computed answer is certain GIVEN the same-value relation.
    When s_ij becomes a trained judgment, its uncertainty enters here and 1.0 stops being the
    right number - that is the seam where the future work plugs in.
    """
    if v144.v513('reach'):
        v211, v213, v168, v155, v994, v995 = v602(v196, v159, v144, v190, v189)
        if v10(v211.v934()) == v550(v168) + 1:
            v277, v399 = (v155 + [v15], v213)
        else:
            v277, v399 = (v168 + [v15], v211[:-1])
        v403 = v706.v608(v399, -1)
        v140 = v10(v403.v934())
        return (v9(v403[v140]), v277[v140], v144['truth_value'])
    if v144['verb'] != 'lookup':
        return (1.0, v1122(v144), v996(v144))
    v399 = v703(v196, v159, v144, v190, v189)
    v403 = v706.v608(v399, -1)
    v140 = v10(v403.v934())
    return (v9(v403[v140]), v144['cands'][v140], v996(v144))

def main() -> v10:
    global SEED, LOG_PATH, LADDER_ON, EDGES_ON, IMPORT_K, INK, FP, WORDS, FAST_COS, VIEWS, ROW_DROPOUT, VIEW_MODE, NEIGHBOURS, REFUSE, GRAPH_CACHE, OPEN, OBJECTIVE, IDENTITY, IDENT_VALUES, IDENT_TAU, IDENT_OVERLAP, IDENT_CANDS, IDENT_CORE, IDENT_IMPORT, ADDRESS_FROM, OPEN_CANDS, ANCHOR_MAX_ROWS, PATTERNS, PAT_W, MIXED, TAPE, ROUTE, STEP_COST, FRAME_MAX, REACH, REACH_K, REACH_CANDS, REACH_MAX_Q, REACH_MAX_ROWS, REACH_NO_REFUSE, REACH_LOOKAHEAD, FRAME_FP, REACH_IMPORT, REACH_HOME_COS, TAPE_SAMPLE, HOME_COS_STAGE, REACH_LINE, REACH_CONFIRM, CONF_WINDOW, PAIR, PAIR_CANDS, PAIR_MAX_ROWS, PAIR_MAX_Q, PAIR_PER_LINE, PAIR_FOLLOW, PAIR_BLIND, REACH_GAMMA, EQUAL_TAILS, STAGE2_ALWAYS, BISECT, FINETUNE, REACH_DEPTH, REACH_COMPASS, SHUFFLE_TAPE, COHERENCE, DEEP_ROOT, TWO_WAY, RETAIN, RETAIN_BY, RETAIN_CTX, OTHER_NET, SPEAK_BATCH, SPEAK_WEIGHT, CALIB_BATCH, CALIB_WEIGHT, CONSTRAIN, CONS_LENSES, CONS_RESOLVE, TWO_WAY_BY, MIN_FILLERS, CONNECT, CONNECT_MAX, OWN_IMPORT, OWN_IN_OFFER, COPY, COPY_D, COPY_BACKFILL, REACH_CHANNEL, MOVES_ON, MOVES, MOVE_TEACH, ROUTE_ON
    v405 = v997.v707()
    v405.v708('--smoke', action='store_true')
    v405.v708('--train-steps', type=v10, default=0)
    v405.v708('--tape-period', type=v10, default=50)
    v405.v708('--cpu', action='store_true', help='run on the CPU even when a card is present. Under frames+reach the graphs are a handful of rows and Phi is 5.6k parameters, so the GPU spends its time on launch latency and its memory on the allocator')
    v405.v708('--addresses', type=v10, default=0)
    v405.v708('--min-mentions', type=v10, default=2)
    v405.v708('--own-in-offer', action='store_true', help="367: rank the home values in the SAME softmax as the walk's, instead of choosing between the two branches. Use with --stage2-always 1.0, which removes the stay/go decision entirely")
    v405.v708('--own-import', action='store_true', help="366: build stage one's own worlds with the SAME imported rows every stage-two candidate gets. Without it 'stay' is compared against systematically larger worlds")
    v405.v708('--connect', action='store_true', help='365: offer values from RELATED places too, weighted by how many fillers each shares with this place. Interleaved into the existing candidate cap, so the offer does not grow')
    v405.v708('--connect-max', type=v10, default=4000, help='neighbourhood places scanned, best-overlap first. Bounded so one common filler cannot cost a minute')
    v405.v708('--copy', action='store_true', help="376: offer values STANDING IN THE NEIGHBOURING LINES of the question, ranked by how often they stand there, nearest line first on a tie. The question's own line is dropped whole - the hidden value is on it. Interleaved into the existing cap, so the offer does not grow")
    v405.v708('--move-set', default=','.v1123(v72), help='386: WHICH MOVES ARE ON THE BALLOT, comma separated, out of step,share,lines. 385 ran all three and failed its gate 2 seeds of 4 - and the split says why: the two seeds where the mind stayed with `step` BEAT the interleave (+0.099, +0.310) and the two where it went to `lines` collapsed. `lines` is copy, the channel already retired from the standing arm on independent evidence (377r hit .475 against connect-only .599); it was re-enabled in 385 only to give that move a lane. Removing it restores a decision taken earlier, on other data')
    v405.v708('--moves', action='store_true', help='385: the mind emits a MOVE - step, share, lines - and the tape executes that one at the unchanged cap, instead of the four channels being merged by a fixed rule with Phi choosing a name. The choice is made on one probe row per move, BEFORE any candidate world is scored')
    v405.v708('--route-on', choices=('all', 'walk_only'), default=v70, help="34.4: which questions teach the STAY/GO decision. `all` is every earlier run bit for bit. `walk_only` gives the router a gradient only where staying is arithmetically wrong - the truth not among the values already here, and among the ones the walk reached - by detaching the route's probability elsewhere. Both PICKS keep their gradient on every question: this cuts one decision, not the population. Requires --two-way; read _read394_walkonly.py's void checks off existing dumps BEFORE running it")
    v405.v708('--move-teach', type=v9, default=v68, help="391: PAY THE MOVE. The move ballot's logits have never received a gradient - reach_logits kept the chosen name and discarded them - so 385 and 386 measured an argmax of a scorer trained to rank final names, on a decision nobody had taught. This adds one term: the ballot's softmax against whether each lane REACHES THE TRUTH, which is the tape's property and is counted while the lane is already enumerated. Weight, declared, never swept; 0 keeps every earlier run bit for bit. Requires --moves")
    v405.v708('--reach-channel', action='store_true', help='379: give the mind THREE INDICATORS saying which channel offered each candidate - connect, home, copy, with the walk as the all-zero baseline. The offer, the head and the budget are unchanged; only the provenance is new. 377 vs 378 showed the merge rule is a constant where a decision belongs')
    v405.v708('--copy-backfill', action='store_true', help='378: the copy lane takes ONLY the slots the walk and connect left empty, instead of round-robining for a fixed share of them. 377 lost hit on one seed of four while reach rose on all four, and cand_places fell on all four - the lane was evicting walked candidates that were right')
    v405.v708('--copy-d', type=v10, default=4, help="lines either side of the question's. 376 read best at 4 on w400 and 16 on w1600, so this travels with the window")
    v405.v708('--min-fillers', type=v10, default=2, help='how many DIFFERENT values a hole must have taken to be a place. 1 admits constant frames, which is where facts live (359)')
    v405.v708('--address-tau', type=v9, default=0.9)
    v405.v708('--tau-mode', choices=('absolute', 'density'), default='absolute', help="absolute keeps 279's fixed cosine and reproduces every earlier run bit for bit. density derives tau so the WRITE ink produces a tape of --tau-target-density mentions per address - required whenever the write ink changes, because a different ink at a fixed cosine shatters the tape and the threshold becomes what the arm measures")
    v405.v708('--tau-target-density', type=v9, default=v129, help='mentions per address to calibrate to. Default is the MEASURED arc/mean train tape (2388 slots / 804 addresses) that every scoreboard number was taken on')
    v405.v708('--tau-calib-iters', type=v10, default=12, help='bisection steps for --tau-mode density. 12 over the full [0, 1] bracket resolves tau to ~2e-4, which holds the density error under 0.005 even where the merge curve is steep; each extra step is pure arithmetic because CachedBank has already inked the corpus')
    v405.v708('--address-overlap', type=v10, default=2)
    v405.v708('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v405.v708('--lr', type=v9, default=0.001)
    v405.v708('--holdout', choices=('corpus', 'address'), default='corpus')
    v405.v708('--no-scan-cache', action='store_true', help='disable the exact corpus-scan memo (use to verify it changes nothing)')
    v405.v708('--no-fast-grouping', action='store_true', help='disable the batched single-link grouping (use to verify it changes nothing)')
    v405.v708('--wiki-bytes', type=v10, default=0)
    v405.v708('--train-lines', type=v10, default=0)
    v405.v708('--eval-lines', type=v10, default=0)
    v405.v708('--line-max', type=v10, default=400, help='keep wiki lines with 80 <= len <= this. 0 drops the upper cap (~4x lines on the local wikitext file). Default 400 keeps every earlier scoreboard bit-identical')
    v405.v708('--import-k', type=v10, default=v122, help='mentions of a candidate imported when completing its world; 0 reproduces the broken ladder where every absent value looked alike')
    v405.v708('--edge-channels', type=v8, default=','.v1123(v123), help='comma list from same,cos,rare - zero the rest. Ablation to find which channel carries the paired win over counting')
    v405.v708('--ink', choices=('mean', 'bigram'), default=v126, help="phrase axis: mean reproduces today's order-blind ctx_fp exactly; bigram binds adjacent words with a fixed non-commutative permutation so the ink can tell `X defeated Y` from `Y defeated X`")
    v405.v708('--fp', choices=('arc', 'hash'), default=v127, help='word axis: arc is the frozen stage191 encoder; hash is character n-grams into a blake2b digest - nothing trained, no character vocabulary, no OOV, every script')
    v405.v708('--words', choices=('ascii', 'unicode'), default=v128, help="what counts as a word. unicode only pays off with --fp hash: arc's stoi has no Cyrillic, so a wider intake would just be discarded")
    v405.v708('--fp-ngram', type=v10, default=3, help='character n-gram length for --fp hash')
    v405.v708('--write-fp', choices=('arc', 'hash'), default='arc', help="ink used to GROUP mentions into addresses. Pinned by default so an ink A/B varies reading only; 279's tau is an absolute cosine and a different ink shatters the tape against it")
    v405.v708('--probe-period', type=v10, default=250, help='how often to score the fixed probe tape. The training curve is measured on a different tape every resample and cannot tell converged from overfitting; this one can')
    v405.v708('--views', type=v10, default=1, help="reconciliation (ROADMAP 20): the mind reads V independently thinned views of each question with the SAME weights, logits are pooled by a mean, and the views' disagreement is a label-free confidence signal. 1 reproduces every earlier run bit for bit; V>1 needs --row-dropout as the thinning rate")
    v405.v708('--neighbours', type=v10, default=0, help='290 (ROADMAP §19): build N(a) from up to this many addresses per route - shared anchor, shared relation, shared rare words - put all their rows in ONE graph, and switch to the sparse verb. 0 reproduces every earlier run bit for bit, including the 5601 parameter count')
    v405.v708('--seed', type=v10, default=v3, help='every draw in the run - tapes, questions, probe, views. Added when 292 came back with held z +2.59 against corpus retrieval and train z -0.15 on the same weights: two samples of one quantity disagreeing by 2.7 sigma. A second seed is the only cheap way to tell structure from a lucky split, and there was no way to ask for one')
    v405.v708('--objective', choices=('ce', 'reward'), default='ce', help="ce is cross-entropy, every run to date. reward optimises 280's fixed payoff directly: L = -sum_c p(c)R(c), closed form, no new constant. Removes the mismatch between what is trained and what is scored; it does not remove a collapse caused by an unanswerable task")
    v405.v708('--open', action='store_true', help="292: the hidden value occurs exactly once at the address, so it is FOREIGN to the evidence and no rule over the address's own rows can reach it. Candidates are the truth and the three ladder rungs, all four importing the same number of rows - the symmetric comparison the ladder could never get in 289. Needs --import-k >= 1, because with 0 imports all four worlds are the same graph")
    v405.v708('--no-graph-cache', action='store_true', help='rebuild every channel per candidate, as before graph_base existed. Dense arms only - use it to verify the cache changed nothing')
    v405.v708('--refuse', action='store_true', help='291: keep the sparse questions whose answer is on NO row of N(a) and let the mind score the world where the query row stays unknown. Refusal becomes an action with a label the tape supplies, not a threshold on a confidence score. Needs --neighbours')
    v405.v708('--view-mode', choices=('thin', 'region'), default='thin', help="how views are cut. thin = recon3's random subsampling (views share ~65%% of rows, D measured model noise, pooled lost to single). region = contiguous stretches of the tape in write order - disjoint by construction, deterministic, so D measures whether the CORPUS agrees with itself at this address rather than whether one sampler agrees with another")
    v405.v708('--row-dropout', type=v9, default=0.0, help='probability of dropping each evidence row during TRAINING, so the mind sees the same fact at several densities. 0 reproduces every earlier run bit for bit - it draws from its own generator')
    v405.v708('--dim', type=v10, default=32, help='width of the mind. Exposed so the max-pool result can be checked at MATCHED parameter count: max-pool added 2048 weights along with the max, and one of those two is the cause')
    v405.v708('--no-max-pool', action='store_true', help='pool with the mean alone, as every run before this one did. A mean cannot express existence, and a high-margin question is decided by one row')
    v405.v708('--no-fast-cos', action='store_true', help='build the pairwise cosine matrix with the original per-pair loop (use to verify the batched version changes nothing)')
    v405.v708('--probe-frac', type=v10, default=10, help='one anchor in this many is reserved for the probe and excluded from both training and held-out scoring, so the stopping step is never chosen using an anchor the evaluation will ask about')
    v405.v708('--probe-size', type=v10, default=200, help='how many probe questions to score. Same questions every time - a probe set that changes is the defect this replaces')
    v405.v708('--no-early-stop', action='store_true', help='keep the last step instead of the best probe step - reproduces every run before the probe tape existed')
    v405.v708('--write-ink', choices=('mean', 'bigram'), default='mean')
    v405.v708('--write-words', choices=('ascii', 'unicode'), default='ascii')
    v405.v708('--no-ladder', action='store_true', help='ablation: train Phi on the task term alone, the control the ladder is measured against')
    v405.v708('--address-from', choices=('fp', 'anchor'), default=v19, help="294: what an address IS. fp keeps 279's grouping (cosine tau AND word overlap, plus a relation half the 293 audit found to be a function word); anchor makes it one exact string the corpus wrote, so nothing is approximated where the evidence is decided.")
    v405.v708('--open-cands', choices=('ladder', 'uniform'), default=v20, help='294: ladder builds the three wrong answers BY relatedness, which is what an inverted mean_phi reads back; uniform draws any value the address does not carry and measures the distance afterwards.')
    v405.v708('--anchor-max-rows', type=v10, default=v22, help='cost budget for anchor addresses: the graph is O(n^2) and `canada` carries dozens of mentions. Nearest in tape order, same in all worlds.')
    v405.v708('--identity', action='store_true', help="293: the verb becomes 'do these mentions name the same place'. The label is the corpus's own pre-grouping string, so the mind is put where fp_addresses' threshold currently stands rather than downstream of it.")
    v405.v708('--identity-audit', action='store_true', help="build 293's questions, score the three rivals, print, and stop. No model and no gradient - the minute that says whether the hour is worth spending, which is what 19 and 291 both needed and did not get.")
    v405.v708('--ident-values', choices=('hide', 'show'), default=v110, help='hide (default) gives every row its own sentinel, so the same-value edge cannot decide identity and CONFIRM becomes a result instead of an input; show measures the size of that shortcut.')
    v405.v708('--ident-import', type=v10, default=v111, help="how many of the candidate's own other mentions arrive with it, gathered by its pre-grouping string. 0 compares a place with a row - the evidence 1-NN already has; k > 0 compares a place with a place, which is the thing 1-NN cannot do.")
    v405.v708('--ident-cands', type=v10, default=v109)
    v405.v708('--ident-core', type=v10, default=v108)
    v405.v708('--tape', choices=('parser', 'frames'), default=v26, help='frames: the write path becomes counting. An address is a hole whose surroundings recur, the width is whatever the corpus supports, and there is no tau, no stopword list and no grammar. 297 measured it: ~10x the addresses, 5.3 mentions each, 22% of rows confirming.')
    v405.v708('--frame-max', type=v10, default=v56)
    v405.v708('--route', action='store_true', help='the mind may READ MORE before answering. `expand` is one more world scored by the same Phi, so the two-step decision is one softmax and stays differentiable - no policy head, no sampling.')
    v405.v708('--gamma', type=v9, default=v29, help='the multiplicative price of movement: every terminal pays gamma^reads * R. Below 1.0 it replaces --step-cost entirely')
    v405.v708('--deep-root', choices=('mind', 'first'), default=v48, help="where the second read starts: the place of the mind's own best shallow candidate, or the nearest place. `first` makes reachability a property of the tape again, which is what the depth numbers need")
    v405.v708('--two-way', action='store_true', help='stage one becomes STAY vs GO: equal-width maxima to decide, each branch valued by its own expectation. Removes both the dilution and the cardinality asymmetry without adding an objective')
    v405.v708('--shuffle-tape', action='store_true', help='THE NULL: permute which filler stood in which hole, keeping every count and every size. The route must collapse to the floor')
    v405.v708('--coherence', type=v10, default=v47, help='score N real tape fragments against corrupted ones - Phi asked whether a world hangs together, with no hole and no teacher')
    v405.v708('--constrain', action='store_true', help="345 / ladder step 1: the mind chooses WHICH OF ITS OWN ROWS to look through and the tape answers by counting what stands with that value over every place that holds it. Phi's output becomes which QUERY, not which answer, so the answer set is never enumerated by us")
    v405.v708('--two-way-by', choices=('max', 'margin'), default=v34, help='with --two-way: summarise each branch by its best world (max, every run to date) or by the GAP between its best two (margin) - the quantity that reads AUC 0.969 on the depth arm and decides nothing')
    v405.v708('--cons-resolve', choices=('count', 'share', 'place'), default=v35, help='how the tape answers through a lens: argmax of the raw co-occurrence count, or of that count divided by how much of the value stands anywhere. 317 measured raw counts at 0.029 against 0.222 because the truths that matter are rare by construction; this is the same fix. 384: `place` answers from the ONE place the lens most stands at instead of summing every place that holds it - a resolution that is a SELECTION, which is the one form of this interface never tried')
    v405.v708('--cons-lenses', type=v10, default=v32, help='how many of its own rows the mind may choose between - its whole output space')
    v405.v708('--calib-batch', type=v10, default=v39, help="389: put the RAW score of B questions into one softmax against which of them the tape can answer, so B-1 of the B free per-question offsets are removed and Phi's value becomes comparable BETWEEN questions. The gauge, not another option: a refusal world lives inside the same per-question softmax and moves with the offset. Costs B questions per step, so divide --train-steps by B for a matched question budget")
    v405.v708('--calib-weight', type=v9, default=v40, help='weight of the calibration term. Declared, never swept - 321 and 341 each priced a second objective at ~4x the route')
    v405.v708('--speak-batch', type=v10, default=v36, help='341: price the speaking across B questions at once instead of on each one, so `always refuse` stops being expressible and what is learned is which questions to spend speech on. Costs B questions per step, so divide --train-steps by B for a matched question budget')
    v405.v708('--speak-weight', type=v9, default=v37, help='weight of the comparative speaking term. Declared, not swept')
    v405.v708('--rival-mind', default=None, help='336: a second saved mind that answers the SAME questions in this run, paired question by question. Train natively and pass the transplant here: if Phi is corpus-free the two are indistinguishable')
    v405.v708('--retain', type=v10, default=v43, help='338: keep only N places for the walk to visit. Questions are still drawn from the whole tape, so the rules are compared on the same questions at the same budget. 0 keeps everything')
    v405.v708('--retain-by', choices=('random', 'own', 'share', 'mind'), default=v44, help="how those N are chosen: at random (what the tape does today), by the most mentions, by the most dominant filler, or by the mind's own margin. `mind` requires a frozen --load-mind - see retain_keep")
    v405.v708('--reach-compass', choices=('cos', 'share', 'both', 'share1', 'rare', 'common', 'cover', 'jaccard'), default=v49, help="what the walk follows. cos: the filler-bag fingerprint. share: the exact count of shared mentions. both: interleaved - 323 says the two disagree about direction 75%% of the time at equal yield. 372 adds the rest of 371's family as compasses, each a count with no fitted constant: share1 (distinct values shared), rare (weighted by 1/the value's corpus mentions), common (the opposite, its own control), cover (how much OF THE NEIGHBOUR the sharing covers), jaccard (the two filler sets against their union)")
    v405.v708('--reach-depth', type=v10, default=v50, help='how many reads the route may chain. 2 gives the walked place its own walk, paid by the same reward at gamma^2 - one objective, deeper')
    v405.v708('--bisect', action='store_true', help='321: measure bisection as a channel. Halves are unfilled worlds of equal evidence, trained by their own exact teacher, and the exam records where the log2(c) descent lands against the flat argmax')
    v405.v708('--finetune', action='store_true', help='with --load-mind, keep training on the new corpus instead of freezing')
    v405.v708('--stage2-always', type=v9, default=v53, help='teach the pick on EVERY question, off-policy and at this weight, while the route is priced exactly as before. 1.0 = the pick is learned as hard as the route; 0.0 = every run before 314')
    v405.v708('--equal-tails', action='store_true', help='the direction choice compares maxima over EQUAL candidate counts, min(|walk|, |line|) - the 304 cardinality fix')
    v405.v708('--step-cost', type=v9, default=v28, help='declared price of reading more, like the 0.75 hedge. Not fitted.')
    v405.v708('--reach', action='store_true', help='299: no candidate list. The mind walks to the nearest places by frame fingerprint and may say only a filler it reached, or nothing. The floor collapses from 0.25 to ~1/|values|, unanswerable questions arrive on their own, and the rival is the same walk without a mind.')
    v405.v708('--pair', action='store_true', help="309: TWO holes on one line, further apart than a frame can reach, filled ONE INTO THE OTHER - the second is scored in the world the first left behind, by the same Phi and with no new parameter. The mind chooses which hole to answer first. Rivals are counting's two ways, the product of marginals and the joint pair where the tape wrote one; COMP_ONLY is where both are blind.")
    v405.v708('--pair-cands', type=v10, default=v99, help="values offered per hole, split evenly between the hole's own rows and what the walk reaches")
    v405.v708('--pair-max-rows', type=v10, default=v100)
    v405.v708('--pair-max-q', type=v10, default=v101)
    v405.v708('--pair-per-line', type=v10, default=v102)
    v405.v708('--pair-blind', action='store_true', help='the ablation: stage two does NOT see the first fill. Same worlds and same Phi, so a tie means the verb composes nothing')
    v405.v708('--pair-follow', type=v10, default=v104, help='first fills whose second stage is scored. The cost bound that keeps the verb linear in the offer instead of quadratic')
    v405.v708('--reach-k', type=v10, default=v76)
    v405.v708('--reach-cands', type=v10, default=v77)
    v405.v708('--home-cos-stage', choices=('both', 'stage2'), default=v80, help='where the home summary is visible. At stage one it lifts the LOCAL worlds and so suppresses the step, which is how 299i lost the routing win; at stage two every option is a walked candidate and it only separates them')
    v405.v708('--wiki', default=None, help='the corpus to build the tape from. The point of --load-mind is to point this somewhere else: a mind that holds no facts should read a tape from text it was never fitted to')
    v405.v708('--save-mind', default=None, help="write Phi's weights and the shape they were trained under")
    v405.v708('--load-mind', default=None, help="load Phi and DO NOT TRAIN. The exam then measures a transplanted mind on this corpus's tape - the literal form of the claim that the knowledge is outside the weights. A shape mismatch refuses to run")
    v405.v708('--flat', action='store_true', help='withhold the edges from the node layer: same parameters, same node features, same pooling and readout, no message passing. The control for whether Phi is a mind or a good choice of features')
    v405.v708('--reach-confirm', action='store_true', help="give the answered row one number: how many RARE words of the question's line also stand around this value elsewhere on the tape. 305 measured 1.67x for the truth over a wrong candidate")
    v405.v708('--conf-window', type=v10, default=v86, help='also read this many lines either side of a home; a wider read makes the shared rare words rarer still')
    v405.v708('--reach-line', action='store_true', help='give the mind a SECOND kind of step: to the other frames of the same sentence. A different relation from resemblance, exactly counted, and the first time direction is a choice rather than a fixed order')
    v405.v708('--reach-home-cos', action='store_true', help='give the answered row one number: the cosine between where this value usually stands on the tape and this place. A summary, not a sample - no row is chosen and no world grows, which is what 299g got wrong')
    v405.v708('--reach-import', choices=('walk', 'homes', 'relation'), default=v89, help="what a stage-two candidate brings with it: the rows at the walked places (`walk`, which makes candidates from one place identical in the cosine channel), its own mentions elsewhere on the tape (`homes`, the lookup verb's import), or its mentions AT PLACES RELATED TO THIS ONE, best overlap first (`relation`, 372b - the walk still steers by the fingerprint and only the EVIDENCE changes)")
    v405.v708('--tape-sample', choices=('uniform', 'region'), default=v90, help='uniform draws 3000 addresses from the whole corpus, which dilutes every relation between places by the sampling ratio; region takes every frame of a contiguous stretch of lines instead, and the 300 audit measured what that changes')
    v405.v708('--frame-fp', choices=('address', 'fillers'), default=v91, help="what a place's fingerprint is. `address` hashes the characters of `left|right`, which at width 1 is six characters and collides - the run measured cos mean 0.918. `fillers` hashes the bag of what stood in the hole, counted with repetition")
    v405.v708('--reach-lookahead', action='store_true', help="score the step as max(stage-two logits) instead of as Phi of a separate pile-of-rows world. 299b had step_rate 0: the step's logit and the step's payoff were unrelated quantities, so nothing could teach the one from the other")
    v405.v708('--reach-no-refuse', action='store_true', help='take silence away in both stages of the walk. 299_hash was void as a payoff measurement - the truth is reachable 10% of the time, so always-silent is the optimal play and the mind found it. This arm asks the other question on its own: can the walk find anything')
    v405.v708('--reach-max-rows', type=v10, default=v79, help="rows of the question's own place, the hidden one included. The frame tape has no ANCHOR_MAX_ROWS and a frame like `the|of` holds hundreds of mentions; graph building is quadratic in rows, so this is what keeps a fat frame from being a question nobody can finish")
    v405.v708('--reach-max-q', type=v10, default=v78, help="questions scored per pack. ~12 graphs each, so this is the run's length dial; 0 scores every one of them.")
    v405.v708('--mixed', action='store_true', help='296: one exam, one payoff. Half the questions have the answer on the list, half do not; the mind must find it or say there is none. It refuses by argmax with no threshold; whole-tape retrieval is given a threshold fitted on train, in its favour.')
    v405.v708('--frames', action='store_true', help=v997.v998)
    v405.v708('--patterns', action='store_true', help="295: mine value-pair regularities exactly, let Phi judge each rule's WORLD (witness rows, no counters), label by held-out lift > 1, race the rule's own train statistics. The object under judgment is a rule, not a row, so 1-NN cannot play.")
    v405.v708('--pat-witnesses', type=v10, default=v24)
    v405.v708('--run-tag', type=v8, default='')
    v405.v708('--out', type=v8, default='', help='optional extra path for the decision JSON; the report is always also written under results/stage289_decision<tag>.json')
    v191 = v405.v709()
    v3 = v191.v406
    v121 = not v191.v710
    v122 = v191.v390
    v126, v127, v128 = (v191.v711, v191.v712, v191.v713)
    v131 = not v191.v714
    v532.v373 = not v191.v715
    v532.v374 = v191.v407
    v117, v119, v118 = (v191.v716, v191.v717, v191.v718)
    v12, v13, v16 = (v191.v719, v191.v720, v191.v721)
    v17 = v191.v408
    v19, v20 = (v191.v722, v191.v723)
    v22 = v191.v409
    if v19 == 'anchor' and (not v191.v721):
        v363('  --address-from anchor is 294, which is the open verb: add --open')
        return 1
    if v19 == 'anchor' and v20 != 'uniform':
        v363('  --address-from anchor needs --open-cands uniform: the ladder is built through fp addresses and cannot be attached to one that has none')
        return 1
    v23, v24 = (v191.v724, v191.v725)
    v25 = v191.v410
    v75, v76, v77 = (v191.v726, v191.v727, v191.v728)
    v78, v79 = (v191.v729, v191.v730)
    v93, v92 = (v191.v731, v191.v732)
    v91, v89 = (v191.v733, v191.v734)
    v90 = v191.v411
    v73 = v191.v412
    v60, v61 = (v191.v735, v191.v736)
    v62, v63 = (v191.v737, v191.v738)
    v64 = v191.v413
    v65 = v191.v414
    v66 = v191.v415
    v67 = v502((v138.v1001() for v138 in v191.v1216.v871(',') if v138.v1001()))
    if not v67 or v518((v138 not in v72 for v138 in v67)):
        raise v861(f"--move-set: unknown move in {v191.v1216!r}; choose from {','.v1123(v72)}")
    v58 = v191.v416
    v57 = v191.v417
    v88, v80 = (v191.v739, v191.v740)
    v87 = v191.v418
    v83, v86 = (v191.v741, v191.v742)
    v98, v99 = (v191.v522, v191.v743)
    v100, v101, v102 = (v191.v744, v191.v745, v191.v746)
    v104, v103 = (v191.v747, v191.v748)
    if v98:
        v75 = True
    if v98 and (v83 or v88):
        v363('  --pair does not carry the confirm or home-cos features yet; run it without them')
        return 1
    if v75 and v191.v749 != 'frames':
        v363('  --reach walks by frame fingerprint: it needs --tape frames')
        return 1
    if v75 and (v25 or v16 or v12 or v107):
        v363('  --reach replaces the offered-candidate verbs; run it alone')
        return 1
    v26, v56, v27, v28 = (v191.v749, v191.v750, v191.v751, v191.v752)
    v29, v55 = (v191.v753, v191.v754)
    v53 = v191.v419
    v51, v52 = (v191.v755, v191.v756)
    v50 = v191.v420
    v49 = v191.v421
    v30, v47 = (v191.v757, v191.v758)
    v48, v54 = (v191.v759, v191.v760)
    v43, v44 = (v191.v761, v191.v762)
    v68 = v191.v422
    v70 = v191.v423
    v36, v37 = (v191.v763, v191.v764)
    v39, v40 = (v191.v765, v191.v766)
    v31, v32 = (v191.v767, v191.v618)
    v35 = v191.v424
    v34 = v191.v425
    if v31 and (v191.v522 or not v191.v726):
        v363('  --constrain needs --reach (and not --pair): it answers the reach question by a different operation, and the two arms must share their holes')
        return 1
    if v36 and (v36 < 2 or v191.v522 or (not v191.v726)):
        v363('  --speak-batch needs at least 2 questions and the reach verb (not pair): a softmax over one margin is a constant, and there is nothing to compare across')
        return 1
    if v39 and (v39 < 2 or v191.v522 or (not v191.v726)):
        v363('  --calib-batch needs at least 2 questions and the reach verb (not pair): with one question there is no second scale to tie it to, and the offset stays free')
        return 1
    if v70 != 'all' and (not v191.v760):
        v363("  --route-on needs --two-way: without it stage one's logits are the own worlds themselves, and cutting the route would cut the home pick with it")
        return 1
    if v68 and (not v66):
        v363('  --move-teach needs --moves: without a ballot there are no move logits to teach')
        return 1
    if v191.v466 and v191.v420 > 1:
        v363("  --rival-mind refused at --reach-depth > 1: the deeper walk is rooted at the mind's own pick, so the two minds would not be answering the same question")
        return 1
    if v44 == 'mind' and v43 and (not v191.v467 or v191.v756):
        v363('  --retain-by mind requires a frozen --load-mind (and not --finetune): a mind that both chooses the tape and is trained on it is no longer separate from it')
        return 1
    if v54 and v87:
        v363('  --two-way is a binary stay/go decision and has no form for --reach-line yet')
        return 1
    if v29 < 1.0:
        v28 = 0.0
    if v191.v376:
        v26 = 'frames'
    if v27 and (not v25):
        v363("  --route is the mixed exam's step: add --mixed")
        return 1
    if v26 == 'frames' and v191.v671 > 1:
        v363('  --tape frames writes exact addresses; --min-mentions 1 (a frame already needs two distinct fillers to exist)')
        return 1
    if v25 and (not v191.v721):
        v363('  --mixed is the open verb with refusal folded in: add --open')
        return 1
    if v25 and v191.v390 < 1:
        v363('  --mixed needs --import-k >= 1, or every candidate world is the same graph')
        return 1
    if v25 and v191.v408 != 'reward':
        v363('  --mixed needs --objective reward: cross-entropy cannot price silence, and the whole point is that one payoff weighs finding against saying there is none')
        return 1
    v107 = v191.v768 or v191.v455
    v110, v109, v108 = (v191.v769, v191.v770, v191.v771)
    v113, v111 = (v191.v772, v191.v773)
    if v107 and (v16 or v12):
        v363('  --identity is a different verb from --open and --neighbours; running two at once measures their sum and credits whichever was named last')
        return 1
    if v107 and v191.v716 > 1:
        v363("  --identity with --views > 1: a region cut is defined over a lookup's evidence rows and a 293 question has no query row until the world is built")
        return 1
    if v107 and v191.v770 < 2:
        v363('  --ident-cands < 2 leaves nothing to choose between')
        return 1
    if v16 and v12:
        v363('  --open and --neighbours are two different verbs; run them apart or the arm measures their sum and credits whichever was named last')
        return 1
    if v16 and v191.v390 < 1:
        v363('  --open needs --import-k >= 1: with nothing imported, the true value and all three rungs give the identical graph and the question has no content')
        return 1
    if v16 and v191.v710:
        v363('  --open IS the ladder - the rungs are its candidates - so --no-ladder would leave it with one candidate')
        return 1
    v14 = not v191.v774
    if v191.v774 and v12:
        v363("  --no-graph-cache is a dense-arm verification path; it does not build 290's two extra edge channels and would silently score a 3-channel graph")
        return 1
    if v13 and (not v12):
        v363('  --refuse needs --neighbours: the unanswerable questions are the sparse ones')
        return 1
    if v12:
        if v191.v390:
            v363('  --neighbours needs --import-k 0: an imported world and the refusal world would carry different row counts, which is a bookkeeping tell, not evidence')
            return 1
    if v117 > 1 and v118 == 'thin' and (v119 <= 0):
        v363('  --views > 1 with --row-dropout 0: every view is the same graph and the pool is decoration; set a thinning rate')
        return 1
    if v26 != 'frames' and (v191.v857, v191.v856) != ('arc', 'mean') and (v191.v999 == 'absolute') and (v191.v1000 == 0.9):
        v363(f"  --write-fp {v191.v857} --write-ink {v191.v856} rewrites the tape, and 279's tau is an absolute cosine: at 0.90 a different ink merges almost nothing. Add --tau-mode density (target defaults to the measured arc tape, {v129} mentions/address).")
        return 1
    for v775, v210, v505 in (('read', v128, v127), ('write', v191.v858, v191.v857)):
        if v210 == 'unicode' and v505 == 'arc':
            v363(f"  --{('' if v775 == 'read' else 'write-')}words unicode with an arc encoder widens the intake into a vocabulary that cannot represent it; use hash there or ascii")
            return 1
    v125 = {v209.v1001() for v209 in v191.v1217.v871(',') if v209.v1001()}
    if not v125 <= v360(v123):
        v363(f'  unknown edge channel in {v511(v125)}; allowed {v123}')
        return 1
    if not v125:
        v363('  every edge channel disabled: nothing to read')
        return 1
    if v12:
        v125 |= v360(v124)
    v206 = v191.v536 and f'_{v191.v536}' or ''
    v206 += '_addrholdout' if v191.v793 == 'address' else ''
    v7 = v0 / f'_stage289_log{v206}.txt'
    v7.v864.v504(parents=True, exist_ok=True)
    v7.v537('', encoding='utf-8')
    v190 = v706.v190('cuda' if v706.v1113.v1218() and (not v191.v973) else 'cpu')
    v152 = v882.v534(v3)
    v706.v776(v3)
    v426 = v777.v777()
    v427 = v191.v535 or (600 if v191.v855 else 6000)
    v428 = v191.v778 or (300 if v191.v855 else 400)
    v363(f'Stage289 derivation start {v1276.v1035(v1277.v1198).v887()} device={v190} holdout={v191.v793}')
    v180, v180, v779, v780 = v781()
    v429 = v1002.v782(v8(v1124.v1003))
    v430 = v429.v1004(v1005) or 0
    v431 = None
    if v127 != 'hash':
        v431 = v1219(v780, v429.v1290()).v702(v190)
        v431.v1006(v706.v1031(v1, map_location=v190, weights_only=False)['model'])
        v431.v811()
        for v159 in v431.v880():
            v159.v1125(False)
    v432 = v783[v128]

    def make_bank(v784, v785, v301):
        v324 = v1126(device=v190, n=v191.v1081, rule=v301) if v784 == 'hash' else v1127(v431, v779, v190)
        return v1007(v1220(v324, rule=v301) if v785 == 'bigram' else v324)
    v189 = v786(v127, v126, v432)
    v433 = v189 if (v191.v857, v191.v856, v191.v858) == (v127, v126, v128) else v786(v191.v857, v191.v856, v783[v191.v858])
    v434 = v189.v435
    v436 = v434.v435 if v126 == 'bigram' else v434
    v437 = v1128.v1008(v431) if v431 is not None else 'not_loaded'
    v438: v439 = {}
    v440 = v787.v441

    def _cached_common(v253, v788: v10=3):
        v140 = (v1129(v253), v550(v253), v788)
        if v140 not in v438:
            v438[v140] = v440(v253, v788)
        return v438[v140]
    v787.v441 = v442
    if not v191.v789:
        v1009(v787)
    if not v191.v790:
        v1010(v787)
    v443 = v137(v191.v791) if v191.v791 else v2
    if not v443.v1011():
        v363(f'  no corpus at {v443}')
        return 1
    v363(f'  corpus {v443}')
    with v443.v721('r', encoding='utf-8', errors='ignore') as v505:
        v792 = v505.v1012(v191.v1130 or (4000000 if v191.v855 else 30000000))
    v364, v365 = (80, v191.v1013 or None)
    v444 = [v1014.v1001() for v1014 in v792.v871('\n') if v550(v1014.v1001()) >= v364 and (v365 is None or v550(v1014.v1001()) <= v365)]
    v445 = v10(0.7 * v550(v444))
    v446 = v444[:v445][:v191.v446 or (3000 if v191.v855 else 25000)]
    v447 = v444[v445:][:v191.v447 or (1500 if v191.v855 else 12000)]
    v363(f'  lines: train {v550(v446)}, eval {v550(v447)} (the probe reserves ANCHORS, not lines - see `reserved`)')
    if v191.v793 == 'address':
        v447 = v446
    v448 = v449(v1015())
    v363(f'  word rule matches stage194: {v448}  (rule={v128}, fp={v127}, ink={v126})')
    if v126 == 'bigram':
        v234 = v449(v434.v1131(v446[:200]))
        v448 &= v234
        v363(f'  bigram tokenisation matches base mean-ink: {v234}')
    if v127 == 'hash':
        v234, v1016 = v1017(v436)
        v448 &= v234
        v363(f'  hash ink deterministic and digest-faithful: {v234}  {v1016}')
    if not v448:
        v363('  ABORT: the ink does not do what it says it does')
        return 1

    def side(v157: v8) -> v10:
        return v10(v1291.v1221(v954.v675(v157).v1292('utf-8')).v1018(), 16) & 1

    def reserved(v157: v8) -> v449:
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
        v319 = v1291.v1221(f'probe:{v954.v675(v157)}'.v1292('utf-8')).v1018()
        return v10(v319, 16) % v191.v1132 == 0
    v450 = None if v26 == 'frames' else v191.v1000 if v191.v999 == 'absolute' else v1133(v191.v1134, v191.v1135, v363)

    def new_pack(v301, v253, v794, v460=False, v795=None):
        v796 = None
        if v26 == 'frames':
            v796, v1136, v1019 = v1222.v1137(v253, v56, v73, v795 or v428, v301, v90)
            v74[0] = v1019
        v159 = v1138.v1020(v253, bank=v433, tok=v429, pad_id=v430, device=v190, rng=v301, n_addr=v795 or v428, min_mentions=v191.v671, tau=v450, overlap=v191.v772, soft_match=0.0, min_per_family=8, addr_key=v191.v672, assertions=v796, group=v796 is None)
        v159 = v439(v159)
        if v30:
            v1021 = v882.v534(v3 + 31337)
            v1022 = v510(v159['tape'].v266)
            v1021.v655(v1022)
            v159['tape'].v266 = v1022
        if v26 == 'frames':
            v311 = v159['straddr']
            v1023 = v547(v360)
            for v166, v554 in v549(v311):
                v1023[v554].v1101(v159['tape'].v266[v166])

            def _fps(v1139):
                v167 = {}
                for v1140 in v521(0, v550(v1139), 512):
                    v1223 = v1139[v1140:v1140 + 512]
                    v220 = v189.v712(v1223).v9()
                    v167.v1293({v250: v220[v217] for v217, v250 in v549(v1223)})
                return v167
            v1024 = v510(v1023)
            if v91 == 'fillers':
                v1141 = v1224(v511({v159['tape'].v266[v166] for v166 in v521(v550(v311))}))
                v1054, v385 = ({}, v207())
                for v166, v554 in v549(v311):
                    v234 = v1141[v159['tape'].v266[v166]]
                    v1054[v554] = v234.v557() if v554 not in v1054 else v1054[v554] + v234
                    v385[v554] += 1
                v159['frame_sum'], v159['frame_cnt'], v159['val_fp'] = (v1054, v385, v1141)
                v292 = {v554: v1119.v1094(v1054[v554], dim=-1) for v554 in v1054}
            else:
                v292 = {v554: v234 for v554, v234 in v666(v1024, [v1119.v1094(v185, dim=-1) for v185 in (v1224(v1024)[v140] for v140 in v1024)])}
            v159['frame_fps'] = [v292[v554] for v554 in v311]
            v159['frame_nfill'] = [v550(v1023[v554]) for v554 in v311]
            v159['frame_nfill_max'] = v576(v159['frame_nfill']) if v159['frame_nfill'] else 1
            v159['frame_mode'] = True
        if v191.v793 == 'address':
            v159['items'] = [v222 for v222 in v159['items'] if v775(v222['address']) == v794]
        v159['items'] = [v222 for v222 in v159['items'] if v1294(v222['address']) == v460]
        return v159
    v451 = v452

    def by_verb(v797):
        v205 = v547(v510)
        for v144 in v797:
            v205[v144['verb']].v876(v144)
        return v205
    v453 = 8 + (1 if v26 == 'frames' else 0) + (1 if v12 else 0) + (1 if v88 else 0) + (1 if v83 else 0) + (3 if v65 else 0)
    v196 = v532(v190, d=v191.v805, n_edge=3 + (v550(v124) if v12 else 0), n_node=v453, grown=v550(v124) if v12 else 0)
    v197 = v706.v879.v533(v196.v880(), lr=v191.v881, weight_decay=0.01)
    v454 = v10(v506((v185.v693() for v185 in v196.v880())))

    def cand_logits(v159, v144):
        return v703(v196, v159, v144, v190, v189)

    def loss_of(v159, v144):
        return v1025(v196, v159, v144, v190, v189)
    v145 = v798(v152, v446, 0)
    v112 = None if v26 == 'frames' else v450.v366['tau'] if v1142(v450) else v450
    if v191.v455:
        v799 = v455(v145, v882.v534(v3 + 293))
        v799['tau'] = v112
        v799['overlap'] = v113
        v799['n_addresses'] = v145['n_addresses']
        v799['n_slots'] = v145['n_slots']
        v363(f'  IDENT_AUDIT {v1084.v859(v799)}')
        (v0 / f"stage289_ident_audit{('_' + v191.v536 if v191.v536 else '')}.json").v537(v1084.v859(v799, indent=2), encoding='utf-8')
        return 0
    v224 = v451(v145, v152)
    v254 = v800(v224)
    v456 = 'cons' if v31 else 'pair' if v98 else 'reach' if v75 else 'lookup'
    v457 = v506((1 for v144 in v254.v513('lookup', ()) if v144.v513('ladder')))
    v363(f"  tape: {v145['n_addresses']} addresses, {v145['n_slots']} slots | questions {v1084.v859({v140: v550(v234) for v140, v234 in v254.v219()})} | params {v454}")
    if v254.v513(v456):
        v801 = v439(v254[v456][0])
        v802 = v938(v145, v801, v189, v190, {0: v641(v145, v801)[0]['offer'][0]}, v801['query_rows'][0])[2].v1114[-1] if v98 else v906(v145, v801, v189, v190, v801['truth_value'], [], 0)[2].v1114[-1] if v75 or v31 else v539(v145, v801, v189, v190, query_value=v801['cands'][0])[2].v1114[-1]
        v803 = v196.v681[0].v1026 - (3 if v532.v373 else 2) * v191.v805
        if v802 != v803:
            v363(f'  node vector is {v802} wide and the mind expects {v803}: a feature is on in the graph and off in the body')
            return 1
        v363(f'  node vector {v802} wide, matches the body')
    v363(f"  ladder coverage: {v457}/{v550(v254.v513('lookup', ()))} lookup questions have all three rungs; the rest train on the task term alone")
    if v550(v254.v513(v456, ())) < v1027.v804:
        v363('  too few lookup questions: raise --train-lines')
        return 1
    v188 = v798(v882.v534(v3 + 99), v447, 1)
    if v23:
        return v1028(v145, v188, v189, v190, v191)
    v458 = v451(v188, v882.v534(v3 + 7))
    v459 = v777.v777()
    v460 = v798(v882.v534(v3 + 555), v446, 0, probe=True, n_addr_over=v428 * v191.v1132)
    v363(f"  probe pack: {v550(v460['items'])} reserved addresses ({v777.v777() - v459:.0f}s to build)")
    v461 = [v144 for v144 in v451(v460, v882.v534(v3 + 556)) if v144['verb'] in (('pair',) if v98 else ('reach',) if v75 else v11) and (not v144.v513('ladder'))][:v595(v191.v1143, 60) if v75 or v98 else v191.v1143]
    v462 = v777.v777()
    v463 = []
    for v144 in v461 if v75 else ():
        v144['_keep_g'] = True
    v464 = v882.v534(v3 + 6060)
    for v144 in () if v75 else v461:
        v355, v1029 = v669(v144, v464, v190)
        v716 = []
        for v227 in v355:
            v140 = v540(v460, v227, v510(v227['cands']))
            v716.v876([v539(v460, v227, v189, v190, query_value=v209, import_k=v140) for v209 in v227['cands']])
        v463.v876((v716, v1029, v706.v601([v144['label']], device=v190)))
    v363(f'  probe tape: {v550(v461)} lookup questions, never trained on; {v506((v550(v263) for v523, v180, v180 in v463 for v263 in v523))} graphs cached ({v117} view(s)/question) in {v777.v777() - v462:.0f}s')

    @v706.v404()
    def probe_loss():
        if v75 or v98:
            if not v461:
                return v9('nan')
            v505 = v989 if v98 else v990
            return v9(v506((v9(v505(v196, v460, v144, v190, v189)) for v144 in v461)) / v550(v461))
        if not v463:
            return v9('nan')
        v196.v811()
        v613 = 0.0
        for v716, v1029, v173 in v463:
            v356 = v706.v541([v706.v541([v196.v542(v384, v143, v389) for v384, v143, v389 in v1295]) for v1295 in v716])
            v613 += v9(v1119.v993(v947(v356, v1029).v1120(0), v173))
        v196.v1030()
        return v613 / v550(v463)
    v465 = {'dim': v191.v805, 'n_node': v453, 'max_pool': v449(v532.v373), 'edges': v511(v125), 'verb': v456, 'views': v117, 'frames': v26 == 'frames', 'frame_fp': v91, 'lookahead': v92, 'no_refuse': v93, 'import': v89, 'home_cos': v88, 'import_k': v122, 'flat': v449(v532.v374), 'reach_k': v76, 'reach_cands': v77, 'line': v87, 'confirm': v83, 'conf_window': v86, 'pair': v98, 'pair_cands': v99, 'pair_max_rows': v100, 'pair_blind': v103, 'gamma': v29, 'equal_tails': v55, 'stage2_always': v53, 'bisect': v51, 'depth': v50, 'compass': v49, 'deep_root': v48, 'two_way': v54, 'two_way_by': v34, 'speak_batch': v36, 'speak_weight': v37, 'move_teach': v68, 'route_on': v70, 'calib_batch': v39, 'calib_weight': v40, 'constrain': v31, 'cons_lenses': v32, 'cons_resolve': v35}
    v45 = (v196, v190, v189)
    if v191.v466:
        v806 = v706.v1031(v191.v466, map_location=v190, weights_only=False)
        if v806.v513('sig') != v465:
            v1032 = {v140: (v806.v513('sig', {}).v513(v140), v234) for v140, v234 in v465.v219() if v806.v513('sig', {}).v513(v140) != v234}
            v363(f'  --rival-mind refused: saved under a different shape {v1084.v859(v1032)}')
            return 1
        v42 = v532(v190, d=v191.v805, n_edge=3 + (v550(v124) if v12 else 0), n_node=v453, grown=v550(v124) if v12 else 0)
        v42.v1006(v806['state'])
        v42.v811()
        v363(f"  rival mind from {v191.v466}: {v806.v513('note', '')} - it answers every reach question this run asks, paired. It is never trained here.")
    if v191.v467:
        v807 = v706.v1031(v191.v467, map_location=v190, weights_only=False)
        if v807.v513('sig') != v465:
            v1033 = {v140: (v807.v513('sig', {}).v513(v140), v234) for v140, v234 in v465.v219() if v807.v513('sig', {}).v513(v140) != v234}
            v363(f'  --load-mind refused: the saved mind was trained under a different shape {v1084.v859(v1033)}')
            return 1
        v196.v1006(v807['state'])
        if v52:
            v363(f"  transplanted mind from {v191.v467}: {v807.v513('note', '')} - FINE-TUNING on this corpus for {v427} steps. This arm is not evidence of separation; the frozen run is.")
        else:
            v427 = 0
            v363(f"  transplanted mind from {v191.v467}: {v807.v513('note', '')} - NO training on this corpus, the exam below is this mind reading a tape it has never been fitted to")
    v468 = v777.v777()
    v808()
    v469 = v777.v777() - v468
    v470 = v550([v282 for v282 in v521(1, v427 + 1) if v282 % v191.v1229 == 0 or v282 == v427])
    v363(f'  probe eval: {v469:.2f}s x {v470} = {v469 * v470 / 60:.1f} min added to this run')
    v471 = v882.v534(v3 + 4242)
    v252 = {'loss': v9('inf'), 'step': 0, 'state': None}
    v472 = []
    v809, v810 = ([], [])
    for v199 in v521(1, v427 + 1):
        if (v199 - 1) % v191.v1082 == 0 and v199 > 1:
            v1034 = {v222['address'] for v222 in v145['items']}
            v145 = v798(v152, v446, 0)
            v224 = v451(v145, v152)
            v254 = v800(v224)
            v1035 = {v222['address'] for v222 in v145['items']}
            if v1034:
                v132.v876(v550(v1034 & v1035) / v576(1, v550(v1034 | v1035)))
            if not v254.v513(v456):
                v363('  empty tape after resample')
                return 1
        v144 = v254[v456][v152.v870(v550(v254[v456]))]
        if v36 or v39:
            global _SPEAK_ACC, _CALIB_ACC
            v615 = v576(v36, v39)
            v224 = [v144] + [v254[v456][v152.v870(v550(v254[v456]))] for v180 in v521(v615 - 1)]
            if v191.v717 > 0:
                v224 = [v1296(v185, v471, 1.0 - v191.v717) or v185 for v185 in v224]
            v38 = [] if v36 else None
            v41 = [] if v39 else None
            try:
                v402 = v706.v541([v1228(v145, v185) for v185 in v224]).v946()
                v1054, v1225 = (v38, v41)
            finally:
                v38 = v41 = None
            if v36:
                if v550(v1054) == v550(v224):
                    v402 = v402 - v37 * v1336([v138 for v138, v582 in v1054], [v147 for v580, v147 in v1054], v190)
                elif v1054:
                    raise v1095(f'speaking batch recorded {v550(v1054)} of {v550(v224)} questions')
            if v39:
                if v550(v1225) == v550(v224):
                    v402 = v402 - v40 * v1337([v282 for v282, v1344 in v1225], [v186 for v1199, v186 in v1225], v190)
                elif v1225:
                    raise v1095(f'calibration batch recorded {v550(v1225)} of {v550(v224)} questions')
        elif v117 > 1:
            if v118 == 'region' and v191.v717 > 0:
                v1226 = v1296(v144, v471, 1.0 - v191.v717)
                if v1226 is not None:
                    v144 = v1226
            v683, v180, v180 = v1227(v196, v145, v144, v190, v189, v471)
            v402 = v1119.v993(v683.v1120(0), v706.v601([v144['label']], device=v190))
        else:
            if v191.v717 > 0:
                v1226 = v1296(v144, v471, 1.0 - v191.v717)
                if v1226 is not None:
                    v144 = v1226
            v402 = v1228(v145, v144)
        v197.v885(set_to_none=True)
        v402.v886()
        v706.v372.v1144.v1036(v196.v880(), 1.0)
        v197.v199()
        v809.v876(v9(v402))
        if v199 % v191.v1229 == 0 or v199 == v427:
            v1037 = v808()
            v472.v876({'step': v199, 'probe_loss': v1037})
            if v1037 < v252['loss']:
                v252 = {'loss': v1037, 'step': v199, 'state': {v140: v234.v1100().v557() for v140, v234 in v196.v1230().v219()}}
        if v199 % v576(1, v427 // 8) == 0:
            v810.v876({'step': v199, 'loss': v9(v1338.v946(v809[-200:])), 'probe_loss': v472[-1]['probe_loss'] if v472 else None})
            v363(f"  step {v199}/{v427} train={v1338.v946(v809[-200:]):.4f} probe={(v472[-1]['probe_loss'] if v472 else v9('nan')):.4f}")
    if not v191.v1038 and v252['state'] is not None:
        v196.v1006(v252['state'])
        v363(f"  early stop: restored step {v252['step']} (probe {v252['loss']:.4f}) of {v427}")
    v196.v811()
    if v191.v473:
        v137(v191.v473).v864.v504(parents=True, exist_ok=True)
        v706.v1039({'sig': v465, 'state': v196.v1230(), 'note': f'trained on {v443.v349}, seed {v3}, {v427} steps'}, v191.v473)
        v363(f'  mind saved to {v191.v473}')
    v474 = v1128.v1008(v431) if v431 is not None else 'not_loaded'

    @v706.v404()
    def examine(v159, v797):
        v184 = {v234: {'n': 0, 'model': 0, 'rival': 0, 'rival_cos': 0, 'floor': 0.0} for v234 in ('count', 'compare', 'lookup')}
        v1040, v165 = (v207(), [])
        v812 = 0
        v201 = v202 = 0
        v813 = []
        v814 = []
        v815 = []
        v816 = []
        v817 = []
        v818 = []
        v819 = {v140: [0.0, 0] for v140 in ('true', 'same_anchor', 'elsewhere')}
        v820 = []
        v821 = []
        v822 = []
        v823 = v882.v534(v3 + 7788)
        v824 = v882.v534(v3 + 2991)
        v825 = {v140: 0.0 for v140 in ('true',) + v120}
        v1041, v1042, v596, v630 = (0, 0, 0, 0)
        v826 = []
        for v144 in v797:
            v234 = v144['verb']
            if v234 == 'pair':
                v211, v320 = v637(v196, v159, v144, v190, v189)
                v321, v322, v180 = v320[v10(v211.v934())]
                v213, v939, v323 = v940(v196, v159, v144, v190, v189, v321, v322)
                v1145 = v939[v10(v213.v934())]
                v297 = [None, None]
                v297[v321], v297[v323] = (v322, v1145)
                v147, v148 = v144['holes']
                v170 = (v147['truth'], v148['truth'])
                v1146 = (v170[0] in v360(v147['offer']), v170[1] in v360(v148['offer']))
                v1184, v1231, v327, v1232, v1233 = v1234(v159, v144)
                v336 = v502(v297) == v170
                v821.v876([v10(v1146[0] and v1146[1]), v10(v336), v10(v297[0] == v170[0]) + v10(v297[1] == v170[1]), v10(v1184 == v170), v10(v327), v10(v1231 == v170), v10(v170[0] in v147['own']), v10(v170[1] in v148['own']), v10(v1146[0]), v10(v1146[1]), v550(v147['offer']) * v550(v148['offer']), v321, v550(v144['slots']), v10(v297[0] == v170[0]), v10(v297[1] == v170[1]), v10(v1233), v10(v1232 == v170)])
                continue
            if v234 == 'cons':
                v1235, v1236, v1237, v1238 = v619(v196, v159, v144, v190, v189)
                v1147 = v620(v159, v144, v1238)
                v1148 = v144['truth_value'] in v360((v185 for v185 in v1147 if v185 is not None))
                v1149 = v10(v1235.v934())
                v1150 = v1237 + ([] if v93 else [v15])
                if v1149 == v550(v1150) and v550(v1236):
                    v1239 = v10(v1236.v934())
                    v1240 = (v1147[v1239] if v1239 < v550(v1147) else v15) or v15
                    v1297, v1244 = (1, v1239)
                else:
                    v1240 = v1150[v1149] if v1149 < v550(v1150) else v15
                    v1297, v1244 = (0, -1)
                v1151 = v1241(v159, v144, v1238)
                v1152 = {v215: v217 for v217, v215 in v549(v1238)}

                def _riv_right(v599):
                    v215 = v1151.v513(v599)
                    return v10(v215 is not None and v1147[v1152[v215]] == v144['truth_value'])
                v1242, v1243 = (0.0, 0)
                if v1244 >= 0:
                    v435, v1298, v1299, v1300 = v424(v159, v144, v1238[v1244])
                    v1242, v1243 = (v1298 / v1299 if v1299 else 0.0, v1299)
                v1153 = v518((v144['truth_value'] in v360(v424(v159, v144, v215)[3]) for v215 in v1238))
                v1154 = v583(v159, v144)
                v1245, v180 = v1246(v159, v144)
                v822.v876([v10(v1148), v10(v144['truth_value'] in v360(v1237)), v550(v1238), v10(v1240 == v15), v10(v1240 == v144['truth_value']), v1297, v1244, v1325('rare'), v1325('frequent'), v1325('decisive'), v10(v1153), v9(v1242), v10(v1243), v10(v144['truth_value'] in v360(v1154['cands'])), v10(v1245 == v144['truth_value'])])
                continue
            if v234 == 'reach':
                v211, v213, v168, v155, v264, v302 = v602(v196, v159, v144, v190, v189)
                v1155 = v144.v513('_deep', (None, [], []))[1] if v50 > 1 else []
                v1156 = v144['truth_value'] in v360(v155) | v360(v1155)
                v1157 = v144['truth_value'] in v360(v1155) and v144['truth_value'] not in v360(v155)
                v297, v622, v1247, v1248, v1249 = v923(v144, v211, v213, v168, v155, v264, v302)
                v1250, v1251 = (0, 0)
                if v42 is not None:
                    with v706.v404():
                        v1305, v1306, v1307, v1308, v1309, v1310 = v602(v42, v159, v144, v190, v189)
                    v1301, v1251, v1302, v1303, v1304 = v923(v144, v1305, v1306, v1307, v1308, v1309, v1310)
                    v1250 = v10(v1301 == v144['truth_value'])
                v663, v1252 = v1246(v159, v144)
                v1158 = v583(v159, v144)
                v1159 = {v209: v617(v159, v144, v209, v1158['rows_of'][v209]) for v209 in v155}
                v435 = v595([v122] + [v550(v1159[v209]) for v209 in v155]) if v155 else 0
                v1160 = v550(v144['slots'])
                v1253, v1254 = v1255(v159, v144)
                v1161 = v144.v538('_cr_ties', 0)
                v1256, v1257 = v926(v196, v159, v144, v190, v189, v155, v1159, v435)[:2] if v51 and v155 else (None, [])
                v1162 = v1160 + v550({v282 for v209 in v155 for v282 in v1159[v209][:v435]} - v360(v144['slots']))
                v820.v876([v10(v1156), v10(v297 == v15), v10(v297 == v144['truth_value']), v9(v1252), v10(v663 == v144['truth_value']), v622, v550(v155), v1160 + v435, v1162, v10(v1339(v159, v144, v76 * 4)), v10(v1339(v159, v144, v76, v824)), v10(v144['truth_value'] in v360(v168)), v10(v955(v144) == v144['truth_value']), v550(v168), v576(v207((v159['tape'].v266[v636] for v636 in v144['slots'][:v144['query_row']])).v266()), v1158['n_places'], v10(v87 and v144['truth_value'] in v360(v575(v159, v144)['cands'])), v10(v87 and v1343(v159, v144) == v144['truth_value']), v10(v622 == 2), v1160, v10(v1253 == v144['truth_value']), v9(v1254), v10(v1256 == v144['truth_value']), v550(v1257), v1247, v10(v1157), v1250, v1251, v1248, v1249, v1161, v67.v649(v144['_move']) if v66 and v144.v513('_move') in v67 else -1])
                continue
            if v234 == 'lookup':
                if v117 > 1:
                    v399, v260, v1311 = v1227(v196, v159, v144, v190, v189, v823)
                    v814.v876([v10(v10(v399.v934()) == v144['label']), v10(v10(v260.v934()) == v144['label']), v1311])
                elif v144.v513('mixed') and v27:
                    v399 = None
                else:
                    v399 = v1326(v159, v144)
                v170 = v144['cands'][v144['label']]
                v1163 = None if v399 is None else v144['cands'][v10(v399.v934())]
                if v144.v513('mixed') and v27:
                    v211, v213 = v544(v196, v159, v144, v190, v189)
                    if v10(v211.v934()) == v550(v144['cands']):
                        v622, v1163 = (1, v144['cands'][v10(v213.v934())])
                    else:
                        v622, v1163 = (0, v144['cands'][v10(v211.v934())])
                else:
                    v622 = 0
                v400 = v704(v196, v159, v144, v190, v189)
                if v144.v513('ladder'):
                    v826.v876(v540(v159, v144, v510(v144['cands']) + [v144['ladder'][v301] for v301 in v120]))
                if v400 is not None:
                    v1258 = [v9(v399[v144['label']])] + [v9(v185) for v185 in v400]
                    for v349, v1191 in v666(('true',) + v120, v1258):
                        v825[v349] += v1191
                    v1041 += 1
                    for v1044, v1045 in v666(v1258, v1258[1:]):
                        if v1044 == v1045:
                            v630 += 1
                            continue
                        v1042 += v10(v1044 > v1045)
                        v596 += 1
                v1151 = None if v144.v513('ident') or v144.v513('mixed') else v955(v144)
                v184[v234]['floor'] += 1.0 / v550(v144['cands'])
                v165.v876({'k': f"{v144['address']}#{v144.v513('hid', v550(v144['slots']))}", 'hit': v10(v1163 == v170)})
                if v144.v513('mixed'):
                    v663, v1312 = v1313(v159, v144, v189, v190)
                    v818.v876([v10(v144['answerable']), v10(v1163 == v15), v10(v1163 == v144['truth_value']), v9(v1312) if v1312 == v1312 else -1.0, v10(v663 == v144['truth_value']), v622])
                v1164 = not v144.v513('open') and (not v144.v513('ident')) and (not v144.v513('mixed')) and (v144.v513('answerable', True) or not v13)
                if v1164:
                    if v1163 == v170 and v1151 != v170:
                        v201 += 1
                    elif v1163 != v170 and v1151 == v170:
                        v202 += 1
                v1259, v1260 = (None, v9('nan')) if v144.v513('ident') or v144.v513('mixed') else v1188(v159, v144, v189, v190)
                if v144.v513('ident'):
                    v663 = v943(v159, v144)
                    v663.v538('_heur_accepted')
                    v817.v876([v10(v1163 == v170)] + [v10(v663[v599] == v170) for v599 in v115] + [v10(v663['heur'] is not None), v10(v159['tape'].v266[v144['cand_slots'][v10(v399.v934())]] in {v159['tape'].v266[v282] for v282 in v144['slots']})])
                if v1259 is not None:
                    v184[v234]['rival_cos'] += v10(v1259 == v170)
                    if v1164:
                        v813.v876((v10(v1163 == v170), v10(v1259 == v170), v1260))
                if v144.v513('uniform'):
                    v819['true'][0] += v9(v399[v144['label']])
                    v819['true'][1] += 1
                    for v1314, v1045 in v144['bucket_of'].v219():
                        v819[v1045][0] += v9(v399[v144['cands'].v649(v1314)])
                        v819[v1045][1] += 1
                    v1261 = v1315(v159, v144, v189, v190)
                    v816.v876([v10(v1163 == v170), v10(v1261 == v170), v10(v1261 is not None)])
                elif v144.v513('open'):
                    v1316 = {v599: v1327 for v1327, v599 in v144['rung_of'].v219()}
                    v1258 = [v9(v399[v144['cands'].v649(v170)])] + [v9(v399[v144['cands'].v649(v1316[v1328])]) for v1328 in v120]
                    for v349, v1191 in v666(('true',) + v120, v1258):
                        v825[v349] += v1191
                    v1041 += 1
                    for v1044, v1045 in v666(v1258, v1258[1:]):
                        if v1044 == v1045:
                            v630 += 1
                        else:
                            v1042 += v10(v1044 > v1045)
                            v596 += 1
                    v1261 = v1315(v159, v144, v189, v190)
                    v816.v876([v10(v1163 == v170), v10(v1261 == v170), v10(v1261 is not None)])
                if v144.v513('sparse'):
                    v1262 = v1317(v144)
                    v815.v876([v10(v144['answerable']), v10(v1163 == v170), v10(v1163 == v15), v10(v1151 == v170), v10(v1259 == v170), v1320(v144), v1260, v550(v144['own_rows']), v10(v1262 == v170), v10(v1262 == v15)])
            else:
                v1163 = v1122(v144)
                v170 = v144['label']
                v1151 = v1163
                v184[v234]['floor'] += 1.0 / (v550(v5) if v234 == 'count' else v550(v6))
                v812 += v10(v1163 != v170)
            v184[v234]['n'] += 1
            v184[v234]['model'] += v10(v1163 == v170)
            v184[v234]['rival'] += v10(v1151 == v170)
            v1040[v234, v8(v170), v8(v1163)] += 1
        v167 = {}
        for v234, v250 in v184.v219():
            if not v250['n']:
                continue
            v167[v234] = {'n': v250['n'], 'model_accuracy': v250['model'] / v250['n'], 'rival_accuracy': v250['rival'] / v250['n'], 'random_floor': v250['floor'] / v250['n']}
            if v234 == 'lookup':
                v167[v234]['rival_cos_accuracy'] = v250['rival_cos'] / v550(v813) if v813 else v9('nan')
                v167[v234]['rival_cos_answered'] = v550(v813)
        v827 = {'n_questions': v1041, 'pairs': v596, 'concordant': v1042, 'ties_excluded': v630, 'import_budget_zero_rate': v506((1 for v148 in v826 if v148 == 0)) / v576(1, v550(v826)), 'import_budget_mean': v506(v826) / v576(1, v550(v826)), 'concordance': v1042 / v596 if v596 else v9('nan'), 'z_vs_half': (v1042 / v596 - 0.5) / v1097.v1170(0.25 / v596) if v596 else v9('nan'), 'mean_phi': {v140: v825[v140] / v1041 if v1041 else v9('nan') for v140 in ('true',) + v120}}
        v828 = v201 + v202
        v167['lookup_paired_vs_rival'] = {'model_only_right': v201, 'rival_only_right': v202, 'discordant': v828, 'mcnemar_z': (v201 - v202) / v1097.v1170(v828) if v828 else v9('nan')}

        def mcnemar(v219):
            v147 = v506((1 for v138, v301, v180 in v219 if v138 and (not v301)))
            v148 = v506((1 for v138, v301, v180 in v219 if v301 and (not v138)))
            v205 = v147 + v148
            return {'n': v550(v219), 'model_only_right': v147, 'rival_only_right': v148, 'discordant': v205, 'mcnemar_z': (v147 - v148) / v1097.v1170(v205) if v205 else v9('nan'), 'max_achievable_z': v1097.v1170(v205) if v205 else 0.0, 'underpowered': v449(v1097.v1170(v205) <= 1.645)}
        v167['lookup_paired_vs_rival_cos'] = v1043(v813)
        v829 = v511((v138 for v180, v180, v138 in v813 if not v1097.v1275(v138)))
        v350 = v829[v550(v829) // 2] if v829 else v9('nan')
        v167['lookup_paired_vs_rival_cos_by_margin'] = {'median_margin': v350, 'low_margin': v1043([v222 for v222 in v813 if v222[2] <= v350]), 'high_margin': v1043([v222 for v222 in v813 if v222[2] > v350])}
        v167['ladder'] = v827
        v167['exact_mismatches'] = v812
        v167['confusion'] = {f'{v147}|{v148}->{v209}': v140 for (v147, v148, v209), v140 in v511(v1040.v219())}
        v167['lookup_item_hits'] = v511(v165, key=lambda v319: v319['k'])
        v167['_views'] = v814
        v167['_sparse'] = v815
        v167['_mixed'] = v818
        v167['_reach'] = v820
        v167['_pair'] = v821
        v167['_cons'] = v822
        if v816:
            v1044 = v506((1 for v138, v1328, v180 in v816 if v138 and (not v1328)))
            v1045 = v506((1 for v138, v1328, v180 in v816 if v1328 and (not v138)))
            v1046 = v1044 + v1045
            v167['open'] = {'n': v550(v816), 'random_floor': 0.25, 'accuracy': v506((v301[0] for v301 in v816)) / v550(v816), 'corpus_retrieval_accuracy': v506((v301[1] for v301 in v816)) / v550(v816), 'corpus_retrieval_answered': v506((v301[2] for v301 in v816)), 'within_address_rivals_undefined': True, 'landscape_observed': {v140: v234[0] / v234[1] if v234[1] else v9('nan') for v140, v234 in v819.v219()} if v819['true'][1] else None, 'landscape_counts': {v140: v234[1] for v140, v234 in v819.v219()}, 'landscape_near_possible': v19 != 'anchor', 'paired_vs_corpus_retrieval': {'model_only_right': v1044, 'rival_only_right': v1045, 'discordant': v1046, 'mcnemar_z': (v1044 - v1045) / v1097.v1170(v1046) if v1046 else v9('nan'), 'max_achievable_z': v1097.v1170(v1046) if v1046 else 0.0, 'underpowered': v449(v1097.v1170(v1046) <= 1.645)}}
        if v817:
            v1047 = v550(v817)
            v1048 = {'n': v1047, 'random_floor': 1.0 / v109, 'accuracy': v506((v301[0] for v301 in v817)) / v1047, 'values': v110, 'heuristic_answered': v506((v301[1 + v550(v115)] for v301 in v817)) / v1047, 'value_agreement': v506((v301[-1] for v301 in v817)) / v1047}
            for v217, v599 in v549(v115):
                v1048[f'rival_{v599}'] = v506((v301[1 + v217] for v301 in v817)) / v1047
                v1048[f'paired_vs_{v599}'] = v1043([(v301[0], v301[1 + v217], 0.0) for v301 in v817])
            v167['ident'] = v1048
        if v815:
            v285 = [v301 for v301 in v815 if v301[0]]
            v844 = [v301 for v301 in v815 if not v301[0]]
            v167['sparse'] = {'n': v550(v815), 'n_answerable': v550(v285), 'n_unanswerable': v550(v844), 'acc_answerable': v506((v301[1] for v301 in v285)) / v550(v285) if v285 else v9('nan'), 'refuse_recall': v506((v301[2] for v301 in v844)) / v550(v844) if v844 else v9('nan'), 'false_refusal': v506((v301[2] for v301 in v285)) / v550(v285) if v285 else v9('nan'), 'rival_own_row_accuracy': v506((v301[8] for v301 in v815)) / v550(v815), 'n_no_own_row': v506((1 for v301 in v815 if v301[7] == 0)), 'acc_no_own_row': v506((v301[1] for v301 in v815 if v301[7] == 0)) / v576(1, v506((1 for v301 in v815 if v301[7] == 0)))}
        return v167
    v475 = v830(v188, v458)
    v476 = v830(v145, v224)
    v831, v832 = (v476.v538('_views', []), v475.v538('_views', []))
    v477 = []
    if v117 > 1:
        with v706.v404():
            for v716, v1029, v173 in v463:
                v356 = v706.v541([v706.v541([v196.v542(v384, v143, v389) for v384, v143, v389 in v1295]) for v1295 in v716])
                v399 = v947(v356, v1029)
                v477.v876([v10(v10(v399.v934()) == v10(v173)), v10(v10(v356[0].v934()) == v10(v173)), v948(v356 if v1029 is None else v356[1:], v1029)])
    v478 = None
    if v117 > 1 and v831 and v832:

        def paired(v171):
            v147 = v506((1 for v1037, v1329, v180 in v171 if v1037 and (not v1329)))
            v148 = v506((1 for v1037, v1329, v180 in v171 if v1329 and (not v1037)))
            v205 = v147 + v148
            return {'pooled_only_right': v147, 'single_only_right': v148, 'discordant': v205, 'mcnemar_z': (v147 - v148) / v1097.v1170(v205) if v205 else v9('nan')}

        def d_auc(v171, v217=0):
            v1049 = [v301[2] for v301 in v171 if not v301[v217]]
            v1050 = [v301[2] for v301 in v171 if v301[v217]]
            if not v1049 or not v1050:
                return {'auc': v9('nan'), 'z': v9('nan')}
            v147 = v1027.v1165(v1049, v1050)
            return {'auc': v147, 'z': v1027.v1263(v147, v550(v1049), v550(v1050)), 'n_err': v550(v1049), 'n_ok': v550(v1050)}

        def t_star_of(v171, v217=0):
            v1051 = v547(v510)
            for v301 in v171:
                v1051[v301[2]].v876(v301)
            v250, v1166, v327 = (None, 0, 0)
            for v205 in v511(v1051):
                v263 = v1051[v205]
                if (v1166 + v506((v185[v217] for v185 in v263))) / (v327 + v550(v263)) < 0.875:
                    break
                v1166 += v506((v185[v217] for v185 in v263))
                v327 += v550(v263)
                v250 = v205
            return v250

        def refusal_of(v217):
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
            v250 = v1167(v477 if v477 else v831, v217)
            v285 = [v301 for v301 in v832 if v250 is not None and v301[2] <= v250]
            v1052 = v550(v832) - v550(v285)
            v1053 = v506((1 for v301 in v285 if v301[v217]))
            v1054 = v1053 / v550(v285) if v285 else v9('nan')
            v1055 = (v1053 - (v550(v285) - v1053) + 0.75 * v1052) / v550(v832)
            v1056 = (v1054 - 0.875) / v1097.v1170(0.875 * 0.125 / v550(v285)) if v285 else v9('nan')
            return {'p_star': 0.875, 'calibrated_on': 'probe' if v477 else 'train', 'd_threshold': v250, 'd_threshold_from_train': v1167(v831, v217), 'held_coverage': v550(v285) / v550(v832), 'held_n_answered': v550(v285), 'held_acc_answered': v1054, 'z_acc_vs_breakeven': v1056, 'held_reward_selective': v1055, 'held_reward_always': v506((1 if v301[v217] else -1 for v301 in v832)) / v550(v832), 'held_reward_blanket_refusal': 0.75}
        v489 = v1057(v832)
        v1058, v1059 = (v1168(v832), v1169(0))
        v1060, v1061 = (v1168(v832, 1), v1169(1))
        v478 = {'views': v117, 'view_mode': v118, 'thin_keep_p': 1.0 - v119 if v118 == 'thin' else None, 'held_pooled_vs_single': v489, 'held_d_auc': v1058, 'probe_d_auc': v1168(v477), 'train_d_auc': v1168(v831), 'refusal': v1059, 'answer_full': {'held_d_auc': v1060, 'probe_d_auc': v1168(v477, 1), 'train_d_auc': v1168(v831, 1), 'refusal': v1061}, 'gates': {'G_pooled_not_worse': v449(v489['mcnemar_z'] >= 0 if v489['discordant'] else True), 'G_d_predicts_error_held': v449(v1058.v513('z', 0) == v1058.v513('z', 0) and v1058.v513('z', 0) > 1.645), 'G_refusal_beats_blanket': v449(v1059['held_reward_selective'] > 0.75)}}
        v363(f'  RECON {v1084.v859(v478)}')

    def _pearson(v147, v148):
        v216 = v550(v147)
        if v216 < 2:
            return v9('nan')
        v1062, v1063 = (v506(v147) / v216, v506(v148) / v216)
        v149 = v506(((v185 - v1062) ** 2 for v185 in v147))
        v150 = v506(((v185 - v1063) ** 2 for v185 in v148))
        if v149 <= 0 or v150 <= 0:
            return v9('nan')
        return v506(((v185 - v1062) * (v186 - v1063) for v185, v186 in v666(v147, v148))) / v1097.v1170(v149 * v150)
    v479 = None
    if v47:
        v479 = {'held_out': v1171(v196, v188, v189, v190, v47, v882.v534(v3 + 9111)), 'train': v1171(v196, v145, v189, v190, v47, v882.v534(v3 + 9112))}
        v363(f'  COHERENCE {v1084.v859(v479)}')

    def cblock(v171):
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
        if not v171:
            return None
        v216 = v550(v171)
        v833 = [v185 for v185 in v171 if not v185[v96['truth_in_own']]]

        def mc(v308, v147, v148):
            v217 = v506((1 for v185 in v308 if v185[v147] and (not v185[v148])))
            v236 = v506((1 for v185 in v308 if v185[v148] and (not v185[v147])))
            return {'mind_only': v217, 'rival_only': v236, 'n': v550(v308), 'mcnemar_z': (v217 - v236) / v1097.v1170(v217 + v236) if v217 + v236 else v9('nan')}
        return {'n': v216, 'n_beyond_own': v550(v833), 'mean_lenses': v506((v185[v96['n_lens']] for v185 in v171)) / v216, 'constrain_rate': v506((v185[v96['constrained']] for v185 in v171)) / v216, 'hit_rate': v506((v185[v96['mind_right']] for v185 in v171)) / v216, 'own_hit_rate': v506((v185[v96['truth_in_own']] for v185 in v171)) / v216, 'answerable': v506((v185[v96['answerable']] for v185 in v171)) / v216, 'present_topm': v506((v185[v96['present_topm']] for v185 in v171)) / v216, 'walk_answerable': v506((v185[v96['walk_answerable']] for v185 in v171)) / v216, 'reads_constraint': 1.0, 'reads_walk': v9(v76), 'beyond_own': {'hit': v506((v185[v96['mind_right']] for v185 in v833)) / v550(v833) if v833 else v9('nan'), 'rare': v506((v185[v96['rare_right']] for v185 in v833)) / v550(v833) if v833 else v9('nan'), 'frequent': v506((v185[v96['frequent_right']] for v185 in v833)) / v550(v833) if v833 else v9('nan'), 'decisive': v506((v185[v96['decisive_right']] for v185 in v833)) / v550(v833) if v833 else v9('nan'), 'vs_decisive': v1264(v833, v96['mind_right'], v96['decisive_right']), 'vs_rare': v1264(v833, v96['mind_right'], v96['rare_right']), 'vs_walk_rival': v1264(v833, v96['mind_right'], v96['walk_rival_right'])}, 'chosen_share_when_constrained': v506((v185[v96['chosen_share']] for v185 in v171 if v185[v96['constrained']])) / v576(1, v506((v185[v96['constrained']] for v185 in v171)))}
    v480 = None
    if v475.v513('_cons'):
        v480 = {'lenses': v32, 'topm': v33, 'reach_k': v76, 'resolve': v35, 'held_out': v1172(v475['_cons']), 'train_control': v1172(v476.v513('_cons') or [])}
        v363(f'  CONS {v1084.v859(v480)}')
    v481 = None
    if v475.v513('_reach'):

        def rpay(v171, v200):
            v138, v301 = ([], [])

            def pay(v1068, v336, v1065):
                v234 = v1265(v1068, v336, v1065)
                return (v234 + 1.0) / 2.0 if v29 < 1.0 else v234
            for v1065, v1068, v336, v922, v1173, v199, *v180 in v171:
                v138.v876((v29 if v199 else 1.0) * v1174(v449(v1068), v449(v336), v449(v1065)) - v28 * v199)
                v301.v876(v1174(v922 < v200, v449(v1173), v449(v1065)))
            return (v138, v301)
        v834 = v476.v513('_reach') or v475['_reach']
        v835 = v511({v185[v82['rival_margin']] for v185 in v834} | {-2.0, 2.0})
        v836 = v576(v835, key=lambda v250: v506(v1179(v834, v250)[1]) / v576(1, v550(v834)))

        def othermind(v171):
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
            if not v171 or v42 is None:
                return None

            def mc(v308):
                v148 = v506((1 for v185 in v308 if v185[v82['mind_right']] and (not v185[v82['other_right']])))
                v209 = v506((1 for v185 in v308 if v185[v82['other_right']] and (not v185[v82['mind_right']])))
                return {'n': v550(v308), 'this_only': v148, 'other_only': v209, 'this': v506((v185[v82['mind_right']] for v185 in v308)), 'other': v506((v185[v82['other_right']] for v185 in v308)), 'mcnemar_z': (v148 - v209) / v1097.v1170(v148 + v209) if v148 + v209 else v9('nan'), 'identical': v148 + v209 == 0, 'underpowered': 0 < v148 + v209 and v1097.v1170(v148 + v209) <= 1.645}
            v1064 = [v185 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']])]
            return {'all': v1264(v171), 'walk_only': v1264(v1064), 'confirm': v1264([v185 for v185 in v171 if v185[v82['truth_in_own']]]), 'step_rate': v506((v185[v82['stepped']] for v185 in v171)) / v550(v171), 'other_step_rate': v506((v185[v82['other_stepped']] for v185 in v171)) / v550(v171)}

        def marginblock(v171):
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
            if not v171:
                return None

            def mean(v308):
                return v506((v185[v82['pick_margin']] for v185 in v308)) / v550(v308) if v308 else v9('nan')
            return {'stayed': v946([v185 for v185 in v171 if not v185[v82['stepped']]]), 'stepped': v946([v185 for v185 in v171 if v185[v82['stepped']] == 1]), 'line': v946([v185 for v185 in v171 if v185[v82['stepped']] == 2]), 'depth2': v946([v185 for v185 in v171 if v185[v82['depth_reached']] == 2]), 'n_stayed': v506((1 for v185 in v171 if not v185[v82['stepped']])), 'n_stepped': v506((1 for v185 in v171 if v185[v82['stepped']] == 1)), 'by_right': {'right': v946([v185 for v185 in v171 if v185[v82['mind_right']]]), 'wrong': v946([v185 for v185 in v171 if not v185[v82['mind_right']]])}, 'stayed_right': v946([v185 for v185 in v171 if not v185[v82['stepped']] and v185[v82['mind_right']]]), 'stayed_wrong': v946([v185 for v185 in v171 if not v185[v82['stepped']] and (not v185[v82['mind_right']])]), 'stepped_right': v946([v185 for v185 in v171 if v185[v82['stepped']] == 1 and v185[v82['mind_right']]]), 'stepped_wrong': v946([v185 for v185 in v171 if v185[v82['stepped']] == 1 and (not v185[v82['mind_right']])])}

        def gateblock(v171):
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
            if not v171:
                return None
            v216 = v550(v171)
            v336 = [v449(v185[v82['mind_right']]) for v185 in v171]
            v1065 = [v449(v185[v82['answerable']]) for v185 in v171]
            v1066 = v882.v534(v3 + 9340)
            v1067 = {'mind': [v9(v185[v82['pick_margin']]) for v185 in v171], 'count_n_own': [v9(v185[v82['n_own']]) for v185 in v171], 'count_top_share': [v9(v185[v82['top_share']]) for v185 in v171], 'random': [v1066.v882() for v180 in v171]}

            def pay(v156):
                v234 = [v1265(v217 not in v156, v336[v217], v1065[v217]) for v217 in v521(v216)]
                return v506(((v185 + 1.0) / 2.0 for v185 in v234)) / v216 if v29 < 1.0 else v506(v234) / v216
            v1068 = v1174(v360())
            v167 = {'n': v216, 'ungated_hit_rate': v506(v336) / v216, 'ungated_payoff': v1174(v360(v521(v216))), 'always_silent': v1068, 'fractions': v510(v97)}
            for v1069 in v97:
                v140 = v576(1, v10(v944(v1069 * v216)))
                v353 = {v599: v1318(v282, v140) for v599, v282 in v1067.v219()}
                v205 = {'k': v140}
                for v599, v337 in v353.v219():
                    v165 = v506((1 for v217 in v337 if v336[v217]))
                    v205[v599] = {'precision': v165 / v140, 'yield': v165, 'payoff': v1174(v337), 'gain': v1174(v337) - v1068}
                v1175 = v353['mind']
                v205['composition'] = {'confirm': v506((1 for v217 in v1175 if v171[v217][v82['truth_in_own']])), 'walk_only': v506((1 for v217 in v1175 if v171[v217][v82['answerable']] and (not v171[v217][v82['truth_in_own']]))), 'neither': v506((1 for v217 in v1175 if not v171[v217][v82['answerable']] and (not v171[v217][v82['truth_in_own']]))), 'right_confirm': v506((1 for v217 in v1175 if v336[v217] and v171[v217][v82['truth_in_own']])), 'right_walk_only': v506((1 for v217 in v1175 if v336[v217] and v171[v217][v82['answerable']] and (not v171[v217][v82['truth_in_own']])))}
                for v599 in ('count_n_own', 'count_top_share'):
                    v148 = v506((1 for v217 in v521(v216) if (v217 in v353['mind'] and v336[v217]) and (not (v217 in v353[v599] and v336[v217]))))
                    v209 = v506((1 for v217 in v521(v216) if (v217 in v353[v599] and v336[v217]) and (not (v217 in v353['mind'] and v336[v217]))))
                    v205[f'vs_{v599}'] = {'mind_only': v148, 'rival_only': v209, 'mcnemar_z': (v148 - v209) / v1097.v1170(v148 + v209) if v148 + v209 else v9('nan')}
                v167[f'{v1069:.2f}'] = v205
            return v167

        def rankblock(v171):
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
            if not v171:
                return None
            v167 = {'n': v550(v171)}
            for v1176, v151 in (('answerable', [1 if v185[v82['answerable']] else 0 for v185 in v171]), ('ceiling', [1 if v185[v82['answerable']] or v185[v82['truth_in_own']] else 0 for v185 in v171]), ('right', [1 if v185[v82['mind_right']] else 0 for v185 in v171])):
                v283 = v506(v151)
                v337 = [v576(1, v550(v171) // 10), v576(1, v550(v171) // 4), v576(1, v283)]
                v205 = {'base_rate': v283 / v550(v171), 'k': v337}
                for v599, v178 in (('mind_margin', v82['pick_margin']), ('mind_score', v82['pick_score']), ('count_n_own', v82['n_own']), ('count_top_share', v82['top_share'])):
                    v282 = [v9(v185[v178]) for v185 in v171]
                    v205[v599] = {'auc': v1330(v282, v151), 'prec': [v1340(v282, v151, v140) for v140 in v337]}
                v167[v1176] = v205
            return v167

        def rblock(v171):
            if not v171:
                return None
            v1177, v1178 = v1179(v171, v836)
            v216 = v550(v171)
            v1070 = [v185 for v185 in v171 if v185[v82['answerable']]]
            v1071 = [v185 for v185 in v171 if not v185[v82['answerable']]]
            v210 = v506((1 for v147, v148 in v666(v1177, v1178) if v147 > v148))
            v1014 = v506((1 for v147, v148 in v666(v1177, v1178) if v147 < v148))
            v205 = v210 + v1014
            return {'n': v216, 'reachable_rate': v550(v1070) / v216, 'count_rival_ties': v506((v185[v82['cr_ties']] for v185 in v171)) / v216, 'move_share': {v138: v506((1 for v185 in v171 if v185[v82['move_id']] == v217)) / v216 for v217, v138 in v549(v67)} if v66 else {}, 'move_hit': {v138: v506((v185[v82['mind_right']] for v185 in v171 if v185[v82['move_id']] == v217)) / v576(1, v506((1 for v185 in v171 if v185[v82['move_id']] == v217))) for v217, v138 in v549(v67)} if v66 else {}, 'move_teach_live': v69['live'] / v576(1, v69['n']) if v68 else 0.0, 'move_teach_ballot': v69['ballot'] / v576(1, v69['n']) if v68 else 0.0, 'move_teach_seen': v69['n'] if v68 else 0, 'route_on': v70, 'route_on_live': v71['live'] / v576(1, v71['n']) if v70 != 'all' else 1.0, 'route_on_seen': v71['n'] if v70 != 'all' else 0, 'mean_candidates': v506((v185[v82['n_cands']] for v185 in v171)) / v216, 'question_rank': v1266(v171), 'margin_by_stage': v1267(v171), 'gate': v1268(v171), 'gate_walk_only': v1268([v185 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']])]), 'other_mind': v1269(v171), 'reachable_wide': v506((v185[v82['reachable_wide']] for v185 in v171)) / v216, 'reachable_random': v506((v185[v82['reachable_random']] for v185 in v171)) / v216, 'random_floor': v506((1.0 / v576(1, v185[v82['n_cands']]) for v185 in v171)) / v216, 'payoff_mind': v506(v1177) / v216, 'payoff_rival': v506(v1178) / v216, 'always_silent': v506((v1265(True, False, v449(v185[v82['answerable']])) for v185 in v171)) / v216, 'found_rate': v506((v185[v82['mind_right']] for v185 in v1070)) / v550(v1070) if v1070 else v9('nan'), 'rival_found_rate': v506((v185[v82['rival_right']] for v185 in v1070)) / v550(v1070) if v1070 else v9('nan'), 'correct_silence': v506((v185[v82['silent']] for v185 in v1071)) / v550(v1071) if v1071 else v9('nan'), 'false_silence': v506((v185[v82['silent']] for v185 in v1070)) / v550(v1070) if v1070 else v9('nan'), 'step_rate': v506((v185[v82['stepped']] for v185 in v171)) / v216, 'own_hit_rate': v506((v185[v82['truth_in_own']] for v185 in v171)) / v216, 'own_rival_hit_rate': v506((v185[v82['own_rival_right']] for v185 in v171)) / v216, 'own_paired': (lambda v1331: {'n': v550(v1331), 'mind_only': v506((1 for v185 in v1331 if v185[v82['mind_right']] and (not v185[v82['own_rival_right']]))), 'rival_only': v506((1 for v185 in v1331 if v185[v82['own_rival_right']] and (not v185[v82['mind_right']]))), 'both': v506((1 for v185 in v1331 if v185[v82['mind_right']] and v185[v82['own_rival_right']])), 'neither': v506((1 for v185 in v1331 if not v185[v82['mind_right']] and (not v185[v82['own_rival_right']]))), 'mcnemar_z': (lambda v148, v209: (v148 - v209) / v1097.v1170(v148 + v209) if v148 + v209 else v9('nan'))(v506((1 for v185 in v1331 if v185[v82['mind_right']] and (not v185[v82['own_rival_right']]))), v506((1 for v185 in v1331 if v185[v82['own_rival_right']] and (not v185[v82['mind_right']]))))})([v185 for v185 in v171 if v185[v82['truth_in_own']]]), 'own_rival_of_own': v506((v185[v82['own_rival_right']] for v185 in v171 if v185[v82['truth_in_own']])) / v576(1, v506((v185[v82['truth_in_own']] for v185 in v171))), 'hit_of_own': v506((v185[v82['mind_right']] for v185 in v171 if v185[v82['truth_in_own']])) / v576(1, v506((v185[v82['truth_in_own']] for v185 in v171))), 'ceiling': v506((1 for v185 in v171 if v185[v82['answerable']] or v185[v82['truth_in_own']])) / v216, 'walk_only_arrive': (lambda v1064: v506((1 for v185 in v1064 if v185[v82['stepped']])) / v550(v1064) if v1064 else v9('nan'))([v185 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']])]), 'walk_only_pick': (lambda v685: {'n': v550(v685), 'mind': v506((v185[v82['mind_right']] for v185 in v685)), 'rival': v506((v185[v82['rival_right']] for v185 in v685)), 'count_rival': v506((v185[v82['count_rival_right']] for v185 in v685)), 'count_rival_rate': v506((v185[v82['count_rival_right']] for v185 in v685)) / v550(v685) if v685 else v9('nan'), 'vs_count_mind_only': v506((1 for v185 in v685 if v185[v82['mind_right']] and (not v185[v82['count_rival_right']]))), 'vs_count_rival_only': v506((1 for v185 in v685 if v185[v82['count_rival_right']] and (not v185[v82['mind_right']]))), 'vs_count_z': (lambda v1045, v1314: (v1045 - v1314) / v1097.v1170(v1045 + v1314) if v1045 + v1314 else v9('nan'))(v506((1 for v185 in v685 if v185[v82['mind_right']] and (not v185[v82['count_rival_right']]))), v506((1 for v185 in v685 if v185[v82['count_rival_right']] and (not v185[v82['mind_right']])))), 'hit_rate': v506((v185[v82['mind_right']] for v185 in v685)) / v550(v685) if v685 else v9('nan'), 'rival_rate': v506((v185[v82['rival_right']] for v185 in v685)) / v550(v685) if v685 else v9('nan'), 'mind_only': v506((1 for v185 in v685 if v185[v82['mind_right']] and (not v185[v82['rival_right']]))), 'rival_only': v506((1 for v185 in v685 if v185[v82['rival_right']] and (not v185[v82['mind_right']]))), 'mcnemar_z': (lambda v1045, v1314: (v1045 - v1314) / v1097.v1170(v1045 + v1314) if v1045 + v1314 else v9('nan'))(v506((1 for v185 in v685 if v185[v82['mind_right']] and (not v185[v82['rival_right']]))), v506((1 for v185 in v685 if v185[v82['rival_right']] and (not v185[v82['mind_right']]))))})([v185 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]) and v185[v82['stepped']]]), 'walk_only_rate': v506((1 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v216, 'hit_of_walk_only': v506((v185[v82['mind_right']] for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v576(1, v506((1 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']])))), 'count_rival_of_walk_only': v506((v185[v82['count_rival_right']] for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v576(1, v506((1 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']])))), 'count_rival_hit_rate': v506((v185[v82['count_rival_right']] for v185 in v171)) / v216, 'deep_rate': v506((1 for v185 in v171 if v185[v82['depth_reached']] >= 2)) / v216, 'deep_only_rate': v506((v185[v82['deep_only']] for v185 in v171)) / v216, 'hit_of_deep_only': v506((v185[v82['mind_right']] for v185 in v171 if v185[v82['deep_only']])) / v576(1, v506((v185[v82['deep_only']] for v185 in v171))) if v518((v185[v82['deep_only']] for v185 in v171)) else v9('nan'), 'hit_of_deep': v506((v185[v82['mind_right']] for v185 in v171 if v185[v82['depth_reached']] >= 2)) / v576(1, v506((1 for v185 in v171 if v185[v82['depth_reached']] >= 2))) if v518((v185[v82['depth_reached']] >= 2 for v185 in v171)) else v9('nan'), 'hit_of_depth1': v506((v185[v82['mind_right']] for v185 in v171 if v185[v82['depth_reached']] == 1)) / v576(1, v506((1 for v185 in v171 if v185[v82['depth_reached']] == 1))) if v518((v185[v82['depth_reached']] == 1 for v185 in v171)) else v9('nan'), 'bisect': (lambda v685: {'n': v550(v685), 'splits_mean': v506((v185[v82['bisect_splits']] for v185 in v685)) / v550(v685) if v685 else v9('nan'), 'bisect_right': v506((v185[v82['bisect_right']] for v185 in v685)), 'flat_right': v506((v185[v82['mind_right']] for v185 in v685)), 'bisect_only': v506((1 for v185 in v685 if v185[v82['bisect_right']] and (not v185[v82['mind_right']]))), 'flat_only': v506((1 for v185 in v685 if v185[v82['mind_right']] and (not v185[v82['bisect_right']]))), 'mcnemar_z': (lambda v1045, v1314: (v1045 - v1314) / v1097.v1170(v1045 + v1314) if v1045 + v1314 else v9('nan'))(v506((1 for v185 in v685 if v185[v82['bisect_right']] and (not v185[v82['mind_right']]))), v506((1 for v185 in v685 if v185[v82['mind_right']] and (not v185[v82['bisect_right']]))))})([v185 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]) and v185[v82['stepped']]]) if v51 else None, 'rival_of_walk_only': v506((v185[v82['rival_right']] for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v576(1, v506((1 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']])))), 'walk_only_paired': (lambda v1064: {'n': v550(v1064), 'mind_only': v506((1 for v185 in v1064 if v185[v82['mind_right']] and (not v185[v82['rival_right']]))), 'rival_only': v506((1 for v185 in v1064 if v185[v82['rival_right']] and (not v185[v82['mind_right']]))), 'both': v506((1 for v185 in v1064 if v185[v82['mind_right']] and v185[v82['rival_right']])), 'neither': v506((1 for v185 in v1064 if not v185[v82['mind_right']] and (not v185[v82['rival_right']]))), 'mcnemar_z': (lambda v148, v209: (v148 - v209) / v1097.v1170(v148 + v209) if v148 + v209 else v9('nan'))(v506((1 for v185 in v1064 if v185[v82['mind_right']] and (not v185[v82['rival_right']]))), v506((1 for v185 in v1064 if v185[v82['rival_right']] and (not v185[v82['mind_right']])))), 'stepped': v506((v185[v82['stepped']] for v185 in v1064))})([v185 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']])]), 'cand_places': v506((v185[v82['n_places']] for v185 in v171)) / v216, 'line_reach_rate': v506((v185[v82['line_reach']] for v185 in v171)) / v216, 'step_line_rate': v506((v185[v82['step_line']] for v185 in v171)) / v216, 'line_only_rate': v506((1 for v185 in v171 if v185[v82['line_reach']] and (not v185[v82['truth_in_own']]) and (not v185[v82['answerable']]))) / v216, 'line_only_paired': (lambda v364: {'n': v550(v364), 'mind_only': v506((1 for v185 in v364 if v185[v82['mind_right']] and (not v185[v82['line_rival']]))), 'rival_only': v506((1 for v185 in v364 if v185[v82['line_rival']] and (not v185[v82['mind_right']]))), 'mcnemar_z': (lambda v148, v209: (v148 - v209) / v1097.v1170(v148 + v209) if v148 + v209 else v9('nan'))(v506((1 for v185 in v364 if v185[v82['mind_right']] and (not v185[v82['line_rival']]))), v506((1 for v185 in v364 if v185[v82['line_rival']] and (not v185[v82['mind_right']]))))})([v185 for v185 in v171 if v185[v82['line_reach']] and (not v185[v82['truth_in_own']]) and (not v185[v82['answerable']])]), 'steps_on_walk_only': v506((v185[v82['stepped']] for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v576(1, v506((v185[v82['stepped']] for v185 in v171))), 'router': (lambda v171, v140: (lambda v324: {'n_stepped': v140, 'mind_enrichment': v506((v185[v82['stepped']] for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v140 / v324 if v140 and v324 else v9('nan'), 'count_enrichment': v506((1 for v185 in v511(v171, key=lambda v186: (v186[v82['n_own']], v186[v82['max_own_count']]))[:v140] if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v140 / v324 if v140 and v324 else v9('nan'), 'share_enrichment': v506((1 for v185 in v511(v171, key=lambda v186: -v186[v82['top_share']])[:v140] if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v140 / v324 if v140 and v324 else v9('nan'), 'top_share_when_stepped': v506((v185[v82['top_share']] for v185 in v171 if v185[v82['stepped']])) / v576(1, v506((v185[v82['stepped']] for v185 in v171))), 'top_share_when_not': v506((v185[v82['top_share']] for v185 in v171 if not v185[v82['stepped']])) / v576(1, v506((1 for v185 in v171 if not v185[v82['stepped']]))), 'n_own_when_stepped': v506((v185[v82['n_own']] for v185 in v171 if v185[v82['stepped']])) / v576(1, v506((v185[v82['stepped']] for v185 in v171))), 'n_own_when_not': v506((v185[v82['n_own']] for v185 in v171 if not v185[v82['stepped']])) / v576(1, v506((1 for v185 in v171 if not v185[v82['stepped']])))})(v506((1 for v185 in v171 if v185[v82['answerable']] and (not v185[v82['truth_in_own']]))) / v550(v171)))(v171, v506((v185[v82['stepped']] for v185 in v171))), 'hit_rate': v506((v185[v82['mind_right']] for v185 in v171)) / v216, 'rival_hit_rate': v506((v185[v82['rival_right']] for v185 in v171)) / v216, 'world_rows_own': v506((v185[v82['world_rows_own']] for v185 in v171)) / v216, 'world_rows_candidate': v506((v185[v82['rows_candidate']] for v185 in v171)) / v216, 'world_rows_expand': v506((v185[v82['rows_expand']] for v185 in v171)) / v216, 'world_rows_expand_when_stepped': v506((v185[v82['rows_expand']] for v185 in v171 if v185[v82['stepped']])) / v576(1, v506((v185[v82['stepped']] for v185 in v171))), 'world_rows_expand_when_not': v506((v185[v82['rows_expand']] for v185 in v171 if not v185[v82['stepped']])) / v576(1, v506((1 for v185 in v171 if not v185[v82['stepped']]))), 'step_vs_size_r': v1270([v185[v82['stepped']] for v185 in v171], [v185[v82['rows_expand']] for v185 in v171]), 'found_where_rival_missed': v506((1 for v185 in v1070 if v185[v82['mind_right']] and (not v185[v82['rival_right']]))), 'missed_where_rival_found': v506((1 for v185 in v1070 if v185[v82['rival_right']] and (not v185[v82['mind_right']]))), 'paired_payoff': {'mind_better': v210, 'rival_better': v1014, 'discordant': v205, 'mcnemar_z': (v210 - v1014) / v1097.v1170(v205) if v205 else v9('nan'), 'max_achievable_z': v1097.v1170(v205) if v205 else 0.0, 'underpowered': v449(v1097.v1170(v205) <= 1.645)}}
        for v837 in (v475.v513('_reach'), v476.v513('_reach')):
            if v837:
                v1180 = v1181(v837)
                if v1180 and v1180['hit_rate'] > v1180['ceiling'] + 1e-09:
                    v363(f"  REACH BOOKKEEPING BROKEN: hit_rate {v1180['hit_rate']:.4f} exceeds ceiling {v1180['ceiling']:.4f} - a reachability the report does not account for. Fix the ceiling before reading anything else.")
                    return 1
        v481 = {'rival_threshold': v836, 'places': v76, 'cands_cap': v77, 'no_refuse': v93, 'lookahead': v92, 'frame_fp': v91, 'import': v89, 'home_cos': v88, 'line_step': v87, 'confirm': v83, 'conf_window': v86, 'home_cos_stage': v80, 'speak_batch': v36, 'speak_weight': v37, 'move_teach': v68, 'route_on': v70, 'calib_batch': v39, 'calib_weight': v40, 'two_way_by': v34, 'gamma': v29, 'equal_tails': v55, 'deep_root': v48, 'two_way': v54, 'stage2_always': v53, 'depth': v50, 'compass': v49, 'held_out': v1181(v475['_reach']), 'train_control': v1181(v476.v513('_reach') or [])}
        v838 = v481['held_out'] or {}
        if v838 and v838.v513('false_silence', 0.0) >= 0.999 and (v838.v513('step_rate', 1.0) <= 0.01):
            v481['void_arm'] = 'always-silent'
            v363('  REACH ARM IS VOID: false_silence 1.0 and no stepping - the mind matched always-silent, which is correct play against this payoff and says nothing about search. Re-run with --reach-no-refuse.')
        v363(f'  REACH {v1084.v859(v481)}')
    v482 = None
    if v475.v513('_pair'):

        def pblock(v171):
            if not v171:
                return None
            v216 = v550(v171)

            def sub(v1163):
                v282 = [v185 for v185 in v171 if v1163(v185)]
                return v282

            def paired(v282, v1182):
                """McNemar, mind against one counting rule, on the questions of `s`."""
                v148 = v506((1 for v185 in v282 if v185[v106['mind_right']] and (not v185[v106[v1182]])))
                v209 = v506((1 for v185 in v282 if v185[v106[v1182]] and (not v185[v106['mind_right']])))
                return {'n': v550(v282), 'mind_only': v148, 'rival_only': v209, 'both': v506((1 for v185 in v282 if v185[v106['mind_right']] and v185[v106[v1182]])), 'mcnemar_z': (v148 - v209) / v1097.v1170(v148 + v209) if v148 + v209 else v9('nan'), 'max_achievable_z': v1097.v1170(v148 + v209) if v148 + v209 else 0.0, 'underpowered': v449(v1097.v1170(v148 + v209) <= 1.645)}
            v1072 = v308(lambda v185: v185[v106['both_offered']])
            v1073 = v308(lambda v185: v185[v106['both_offered']] and (not v185[v106['marg_right']]) and (not v185[v106['joint_seen']]))
            v1074 = [v185 for v185 in v1073 if not v185[v106['bag_seen']]]

            def co(v579):
                return {'n': v550(v579), 'rate': v550(v579) / v216, 'mind_right': v506((v185[v106['mind_right']] for v185 in v579)), 'hit_rate': v506((v185[v106['mind_right']] for v185 in v579)) / v550(v579) if v579 else v9('nan'), 'random_floor': v506((1.0 / v576(1, v185[v106['n_pairs']]) for v185 in v579)) / v550(v579) if v579 else v9('nan'), 'binomial_z': (lambda v140, v561: (v140 - v561) / v1097.v1170(v561 * (1 - v561 / v576(1, v550(v579)))) if v561 > 0 else v9('nan'))(v506((v185[v106['mind_right']] for v185 in v579)), v506((1.0 / v576(1, v185[v106['n_pairs']]) for v185 in v579))), 'one_hole_mean': v506((v185[v106['one_right']] for v185 in v579)) / (2 * v550(v579)) if v579 else v9('nan'), 'indep_expected': v506((v185[v106['right_a']] for v185 in v579)) * v506((v185[v106['right_b']] for v185 in v579)) / v550(v579) ** 2 * v550(v579) if v579 else v9('nan'), 'right_a': v506((v185[v106['right_a']] for v185 in v579)) / v550(v579) if v579 else v9('nan'), 'right_b': v506((v185[v106['right_b']] for v185 in v579)) / v550(v579) if v579 else v9('nan')}
            return {'n': v216, 'both_offered': v550(v1072) / v216, 'mind_exact': v506((v185[v106['mind_right']] for v185 in v171)) / v216, 'mind_exact_of_offered': v506((v185[v106['mind_right']] for v185 in v1072)) / v550(v1072) if v1072 else v9('nan'), 'holes_right_mean': v506((v185[v106['one_right']] for v185 in v171)) / (2 * v216), 'marginal_exact': v506((v185[v106['marg_right']] for v185 in v171)) / v216, 'joint_exact': v506((v185[v106['joint_right']] for v185 in v171)) / v216, 'joint_seen_rate': v506((v185[v106['joint_seen']] for v185 in v171)) / v216, 'bag_seen_rate': v506((v185[v106['bag_seen']] for v185 in v171)) / v216, 'bag_exact': v506((v185[v106['bag_right']] for v185 in v171)) / v216, 'in_own_both': v506((1 for v185 in v171 if v185[v106['in_own_a']] and v185[v106['in_own_b']])) / v216, 'offered_a': v506((v185[v106['offered_a']] for v185 in v171)) / v216, 'offered_b': v506((v185[v106['offered_b']] for v185 in v171)) / v216, 'mean_pair_worlds': v506((v185[v106['n_pairs']] for v185 in v171)) / v216, 'world_rows': v506((v185[v106['world_rows']] for v185 in v171)) / v216, 'first_hole_rate': v506((v185[v106['first_hole']] for v185 in v171)) / v216, 'vs_marginal': v1057(v171, 'marg_right'), 'vs_marginal_offered': v1057(v1072, 'marg_right'), 'COMP_ONLY': v1271(v1073), 'COMP_STRICT': v1271(v1074)}
        v482 = {'cands': v99, 'max_rows': v100, 'per_line': v102, 'follow': v104, 'blind': v103, 'frame_max': v56, 'held_out': v1183(v475['_pair']), 'train_control': v1183(v476.v513('_pair') or [])}
        v363(f'  PAIR {v1084.v859(v482)}')
    v483 = None
    if v475.v513('_mixed'):

        def payoffs(v171, v200):
            """Per question: what the mind earned and what thresholded retrieval earned."""
            v138, v301 = ([], [])
            for v285, v1068, v336, v1184, v1173, v199 in v171:
                v138.v876(v1265(v449(v1068), v449(v336), v449(v285)) - v28 * v199)
                v335 = v1184 < v200
                v301.v876(v1265(v335, v449(v1173), v449(v285)))
            return (v138, v301)
        v834 = v476.v513('_mixed') or v475['_mixed']
        v835 = v511({v1319[3] for v1319 in v834} | {-1.0, 2.0})
        v200 = v576(v835, key=lambda v250: v506(v1185(v834, v250)[1]) / v576(1, v550(v834)))

        def block(v171):
            v1177, v1178 = v1185(v171, v200)
            v216 = v576(1, v550(v171))
            v1075 = [v185 for v185 in v171 if v185[0]]
            v844 = [v185 for v185 in v171 if not v185[0]]
            v210 = v506((1 for v147, v148 in v666(v1177, v1178) if v147 > v148))
            v1014 = v506((1 for v147, v148 in v666(v1177, v1178) if v147 < v148))
            v205 = v210 + v1014
            return {'n': v550(v171), 'answerable_rate': v550(v1075) / v216, 'payoff_mind': v506(v1177) / v216, 'payoff_rival': v506(v1178) / v216, 'step_rate': v506((v185[5] for v185 in v171)) / v216, 'always_answer': v506((v1265(False, v449(v185[2]), v449(v185[0])) for v185 in v171)) / v216, 'always_silent': v506((v1265(True, False, v449(v185[0])) for v185 in v171)) / v216, 'found_rate': v506((v185[2] for v185 in v1075)) / v550(v1075) if v1075 else v9('nan'), 'rival_found_rate': v506((v185[4] for v185 in v1075)) / v550(v1075) if v1075 else v9('nan'), 'correct_silence': v506((v185[1] for v185 in v844)) / v550(v844) if v844 else v9('nan'), 'false_silence': v506((v185[1] for v185 in v1075)) / v550(v1075) if v1075 else v9('nan'), 'found_where_rival_missed': v506((1 for v185 in v1075 if v185[2] and (not v185[4]))), 'missed_where_rival_found': v506((1 for v185 in v1075 if v185[4] and (not v185[2]))), 'paired_payoff': {'mind_better': v210, 'rival_better': v1014, 'discordant': v205, 'mcnemar_z': (v210 - v1014) / v1097.v1170(v205) if v205 else v9('nan'), 'max_achievable_z': v1097.v1170(v205) if v205 else 0.0, 'underpowered': v449(v1097.v1170(v205) <= 1.645)}}
        v483 = {'rival_threshold': v200, 'held_out': v1186(v475['_mixed']), 'train_control': v1186(v476.v513('_mixed') or [])}
        v363(f'  MIXED {v1084.v859(v483)}')
    v484 = None
    v839, v840 = (v475.v538('_sparse', []), v476.v538('_sparse', []))
    if v13 and v839:
        v841 = []
        for v144 in v461:
            if not v144.v513('sparse'):
                continue
            v250 = v144['cands'][v144['label']]
            v243, v1187 = v1188(v460, v144, v189, v190)
            v841.v876([v10(v144['answerable']), v10(v955(v144) == v250), v10(v243 == v250), v1320(v144), v1187])

        def thresh(v171, v1076, v1077):
            """Largest set of questions, taken in DESCENDING confidence, whose accuracy still
            clears the derived break-even. Whole ties admitted together and the scan stops at
            the first group that breaks it - the same rule the D threshold uses, and for the
            same two reasons: accuracy decays slowly so a tail sneaks in one item at a time, and
            rows tied at one margin cannot be split anyway. Rows with no margin at all (the ink
            gave the query no context, so the retrieval rival cannot rank) are excluded from the
            calibration and refuse at scoring time."""
            v263 = v547(v510)
            for v301 in v171:
                if v301[v1077] == v301[v1077]:
                    v263[v301[v1077]].v876(v301)
            v250, v1166, v327 = (None, 0, 0)
            for v138 in v511(v263, reverse=True):
                v1189 = v263[v138]
                if (v1166 + v506((v185[v1076] for v185 in v1189))) / (v327 + v550(v1189)) < 0.875:
                    break
                v1166 += v506((v185[v1076] for v185 in v1189))
                v327 += v550(v1189)
                v250 = v138
            return v250

        def reward(v171, v1076, v1077, v250):
            v613 = 0.0
            for v301 in v171:
                if v250 is None or not v301[v1077] >= v250:
                    v613 += 1.0 if not v301[0] else 0.75
                else:
                    v613 += 1.0 if v301[v1076] else -1.0
            return v613 / v550(v171)

        def fixed_reward(v171, v1076, v1078):
            """A rival that refuses by RULE rather than by threshold - scored exactly as the
            mind is, so the only difference between them is the judgment."""
            return v506((1.0 if v301[v1076] else 0.75 if v301[v1078] else -1.0 for v301 in v171)) / v550(v171)
        v842 = v1190(v841, 1, 3) if v841 else None
        v843 = v1190(v841, 2, 4) if v841 else None
        v285 = [v301 for v301 in v839 if v301[0]]
        v844 = [v301 for v301 in v839 if not v301[0]]
        v845 = (v550(v844) + 0.75 * v550(v285)) / v550(v839)
        v846 = v506((1.0 if v301[1] else 0.75 if v301[2] else -1.0 for v301 in v839)) / v550(v839)
        v484 = {'n': v550(v839), 'n_answerable': v550(v285), 'n_unanswerable': v550(v844), 'unanswerable_rate': v550(v844) / v550(v839), 'mind': {'acc_answerable': v506((v301[1] for v301 in v285)) / v550(v285) if v285 else v9('nan'), 'refuse_recall': v506((v301[2] for v301 in v844)) / v550(v844) if v844 else v9('nan'), 'false_refusal': v506((v301[2] for v301 in v285)) / v550(v285) if v285 else v9('nan'), 'coverage': 1.0 - v506((v301[2] for v301 in v839)) / v550(v839), 'reward': v846}, 'rival_counting': {'threshold_from_probe': v842, 'reward': v1272(v839, 3, 5, v842)}, 'rival_retrieval': {'threshold_from_probe': v843, 'reward': v1272(v839, 4, 6, v843)}, 'rival_own_row': {'reward': v1273(v839, 8, 9), 'acc_answerable': v506((v301[8] for v301 in v285)) / v550(v285) if v285 else v9('nan'), 'refuse_recall': v506((v301[9] for v301 in v844)) / v550(v844) if v844 else v9('nan')}, 'blanket_refusal_reward': v845, 'always_answer_ceiling': (v550(v285) - v550(v844)) / v550(v839), 'gates': {'G_refuse_beats_blanket': v449(v846 > v845), 'G_refuse_beats_counting': v449(v846 > v1272(v839, 3, 5, v842)), 'G_refuse_beats_retrieval': v449(v846 > v1272(v839, 4, 6, v843)), 'G_refuse_beats_own_row_shortcut': v449(v846 > v1273(v839, 8, 9))}}
        v363(f'  REFUSE {v1084.v859(v484)}')

    def paraphrase_split(v847):
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
        v1079, v1080, v327 = (v547(v360), v547(v360), v207())
        for v222 in v847['items']:
            v160 = v222['address']
            v343 = v954.v675(v160)
            for v166 in v222['slots']:
                v1191 = v847['tape'].v266[v166]
                v1079[v343, v1191].v1101(v160)
                v327[v343, v1191] += 1
                v1080[v896((v343, v1191.v1107()))].v1101((v343, v1191.v1107()))
        v848 = [v140 for v140 in v1079 if v327[v140] >= 2]
        v849 = [v159 for v159, v1274 in v1080.v219() if v550(v159) == 2 and v550(v1274) == 2]
        return {'facts_written_twice': v550(v848), 'same_anchor_diff_relation': v506((1 for v140 in v848 if v550(v1079[v140]) > 1)) / v550(v848) if v848 else v9('nan'), 'mean_addresses_per_fact': v506((v550(v1079[v140]) for v140 in v848)) / v550(v848) if v848 else v9('nan'), 'reversed_pairs': v550(v849), 'reversed_pair_rate': v550(v849) / v576(1, v550(v1080))}

    def tape_shape(v847, v797):
        v488 = [v144 for v144 in v797 if v144['verb'] == 'lookup']
        return {'slots': v550(v847['texts']), 'addresses': v550(v847['addr_slots']), 'mentions_per_address': v550(v847['texts']) / v550(v847['addr_slots']) if v847['addr_slots'] else v9('nan'), 'lookup_questions': v550(v488), 'mean_candidates': v506((v550(v144['cands']) for v144 in v488)) / v550(v488) if v488 else v9('nan'), 'paraphrase': v1192(v847)}
    v363(f"  HELD {v1084.v859({v140: v234 for v140, v234 in v475.v219() if v140 != 'lookup_item_hits'})}")
    v363(f"  CONTROL {v1084.v859({v140: v234 for v140, v234 in v476.v219() if v140 != 'lookup_item_hits'})}")
    v485 = v437 == v474
    v486 = v475.v513('lookup', {}).v513('n', 0) >= 2 * v1027.v804
    if v75:
        v850 = (v481 or {}).v513('held_out') or {}
        v486 = v850.v513('n', 0) >= 2 * v1027.v804
    v487 = v449(v475.v513('exact_mismatches', 1) == 0 and v476.v513('exact_mismatches', 1) == 0)
    v488 = v475.v513('lookup', {})
    v489 = v475.v513('lookup_paired_vs_rival', {})
    v490 = v449(v488 and v488['model_accuracy'] > v488['random_floor'])
    if v75:
        v850 = (v481 or {}).v513('held_out') or {}
        v490 = v449(v850 and v850['payoff_mind'] > v850['always_silent'])
    v491 = v449(v489.v513('discordant', 0) >= 2 * v1027.v804 and (not v1097.v1275(v489.v513('mcnemar_z', v9('nan')))) and (v489['mcnemar_z'] > 1.645))
    v492 = v475.v513('lookup_paired_vs_rival_cos', {})
    v493 = v449(v492.v513('discordant', 0) >= 2 * v1027.v804 and (not v1097.v1275(v492.v513('mcnemar_z', v9('nan')))) and (v492['mcnemar_z'] > 1.645))
    if v75:
        v851 = (v481 or {}).v513('held_out') or {}
        v852 = v851.v513('paired_payoff', {})
        v493 = v449(v852.v513('discordant', 0) >= 8 and (not v1097.v1275(v852.v513('mcnemar_z', v9('nan')))) and (v852['mcnemar_z'] > 1.645))
        v491 = v493
    v494 = v475.v513('ladder', {})
    v495 = v449(v494.v513('pairs', 0) >= 6 * v1027.v804 and (not v1097.v1275(v494.v513('z_vs_half', v9('nan')))) and (v494['z_vs_half'] > 1.645))
    v496 = v449(v492.v513('underpowered', True))
    v497 = v475.v513('lookup_paired_vs_rival_cos_by_margin', {})
    v364, v365 = (v497.v513('low_margin', {}), v497.v513('high_margin', {}))
    v853, v854 = (v364.v513('mcnemar_z', v9('nan')), v365.v513('mcnemar_z', v9('nan')))
    v498 = v449(not v1097.v1275(v853) and (not v1097.v1275(v854)) and (v853 > 1.645) and (v854 < -1.645))
    v499 = 'NO_TASK' if not (v486 and v485 and v487) else 'DERIVATION_OK' if v490 and v491 and v493 and v495 else 'UNDERPOWERED_VS_RETRIEVAL' if v490 and v491 and (not v493) and v496 else 'PHI_HELPS_WHERE_SIMILARITY_RUNS_OUT' if v490 and v491 and (not v493) and v498 else 'PHI_ADDS_NOTHING_ON_LOOKUP' if v490 and v491 and (not v493) else 'DERIVATION_PARTIAL' if v490 or v495 else 'DERIVATION_NO'
    v167 = {'stage': '289', 'overall': v499, 'seed': v3, 'smoke': v191.v855, 'holdout': v191.v793, 'run_tag': v191.v536, 'train_steps': v427, 'params': v454, 'dim': v191.v805, 'min_fillers': v73, 'connect': v449(v60), 'copy': v449(v62), 'copy_d': v63, 'copy_backfill': v449(v64), 'reach_channel': v449(v65), 'moves': v449(v66), 'move_set': v510(v67) if v66 else [], 'own_import': v449(v58), 'own_in_offer': v449(v57), 'own_import_full': v59[1] / v59[0] if v59[0] else None, 'objective': 'expected_reward_280' if v17 == 'reward' else 'plackett_luce_ladder' if v121 else 'cross_entropy_no_ladder', 'edge_channels': v511(v125), 'import_k': v122, 'views': v117, 'reconciliation': v478, 'neighbours': v12, 'open_verb': v16, 'patterns_verb': v23, 'address_from': v19, 'open_cands': v20, 'anchor_max_rows': v22 if v19 == 'anchor' else None, 'identity_verb': v107, 'mixed_verb': v25, 'mixed': v483, 'reach_verb': v75, 'reach': v481, 'constrain': v31, 'cons': v480, 'coherence': v479, 'shuffled_tape': v30, 'retain': v43, 'retain_by': v44, 'reach_cols': v510(v81), 'pair_verb': v98, 'pair': v482, 'tape_cut': v26, 'route': v27, 'step_cost': v28 if v27 else None, 'frame_max': v56 if v26 == 'frames' else None, 'frame_pool': v74[0] if v26 == 'frames' else None, 'tape_sample': v90 if v26 == 'frames' else None, 'flat': v449(v532.v374), 'transplant': v191.v467 or None, 'corpus': v8(v443), 'identity': {'tau': v112, 'overlap': v113, 'cands': v109, 'core': v108, 'values': v110, 'import': v111, 'supply': v439(v114)} if v107 else None, 'open_near_source': {'same_anchor': v18[0], 'neighbourhood': v18[1]} if v16 else None, 'graph_rows': {'mean': v134[0] / v576(1, v134[2]), 'max': v134[1], 'graphs': v134[2]}, 'neighbourhood_audit': v1193(v145, v12) if v12 else None, 'nb_channels': {'anchor_nonzero_rate': v133[0] / v576(1, v133[2]), 'rel_nonzero_rate': v133[1] / v576(1, v133[2]), 'pairs': v133[2]} if v12 else None, 'refuse': v484, 'ink': v126, 'fp': v127, 'words': v128, 'write_ink': v191.v856, 'write_fp': v191.v857, 'write_words': v191.v858, 'fp_ngram': v191.v1081 if v127 == 'hash' else None, 'tau': {'mode': 'frames', 'value': None, 'target_density': None, 'achieved_density': None, 'monotone': None, 'trace': None} if v26 == 'frames' else {'mode': v191.v999, 'value': v191.v1000 if v191.v999 == 'absolute' else v450.v366.v513('tau'), 'target_density': v191.v1134 if v191.v999 == 'density' else None, 'achieved_density': v450.v366.v513('achieved') if v191.v999 == 'density' else None, 'monotone': v450.v366.v513('monotone') if v191.v999 == 'density' else None, 'trace': v450.v366.v513('trace') if v191.v999 == 'density' else None}, 'tape_shape': {'held_out': v1194(v188, v458), 'train': v1194(v145, v224)}, 'resample': {'tape_period': v191.v1082, 'mean_overlap': v506(v132) / v550(v132) if v132 else v9('nan'), 'n_resamples': v550(v132), 'note': "Jaccard between consecutive tapes' address sets. Near 1 means the redraw returns the same addresses and the anti-memorisation argument in HANDOFF 1 is decorative - the fix is a larger address pool, i.e. more corpus, not fewer parameters"}, 'row_dropout': {'rate': v191.v717, 'mean_kept_fraction': v116[0] / v116[1] if v116[1] else v9('nan'), 'note': 'training only - the held-out tape is never thinned. Marginalisation, not noise: a subset of the evidence is a world the corpus could have written, and the low/high margin split is a density axis the mind was never trained across'}, 'early_stop': {'enabled': not v191.v1038, 'best_step': v252['step'], 'best_probe_loss': v252['loss'], 'total_steps': v427, 'probe_questions': v550(v461)}, 'probe_curve': v472, 'rare_nonzero_rate': v130[0] / v130[1] if v130[1] else v9('nan'), 'ink_degenerate_rate': v1195[0] / v1195[1] if v1195[1] else v9('nan'), 'cos_mean': v135[0] / v135[2] if v135[2] else v9('nan'), 'cos_std': v1097.v1170(v576(0.0, v135[1] / v135[2] - (v135[0] / v135[2]) ** 2)) if v135[2] else v9('nan'), 'ladder_coverage_train': {'with_ladder': v457, 'lookup_questions': v550(v254.v513('lookup', ()))}, 'count_labels': v510(v5), 'compare_labels': v510(v6), 'gates': {'G_arc_enc_frozen': v485, 'G_ink_verified': v448, 'G_task_exists': v486, 'G_exact_algebra_matches_tape': v487, 'G_lookup_beats_floor': v490, 'G_lookup_beats_counts_paired': v491, 'G_lookup_beats_retrieval_paired': v493, 'G_phi_orders_negatives': v495}, 'held_out': v475, 'train_control': v476, 'exact_note': 'count and compare left the weights: they are functions of the same-value relation alone (new_i = 1 - max_{j<i} s_ij; count = sum new_i; compare = sign of the side difference), computed exactly with zero parameters and no 5+ cap, because the invariant says whatever decides may not be approximate. Their accuracy is 1.0 by construction and is checked, not celebrated - G_exact_algebra_matches_tape is a sanity bolt. The interference that cost count 0.965 -> 0.903 is removed by construction: one trained task remains', 'ladder_note': 'three wrong answers per question at increasing structural distance - same anchor / adjacent in tape order / anywhere on the tape - every rung a value the corpus wrote, no similarity chosen by anyone. Phi trained only against local wrong candidates learns a BOUNDARY; generation needs a LANDSCAPE, and a mind that cannot rank its own wrong answers by how wrong they are has no direction to move in. The objective is one Plackett-Luce term, not a task loss plus a ladder loss with a weight between them, and it reduces to the previous cross-entropy exactly when the tape cannot supply a ladder', 'retrieval_note': 'two rivals now, because they answer two different questions. The counting rival knows nothing about context, so beating it shows only that the context channel carries information counts lack - not that reading it takes a mind. rival_cos is 1-NN over the same evidence rows by the same ctx_fp cosine, zero parameters, no training. With hash ink the representation IS Random Indexing over fastText-shaped word vectors, so the distance between this architecture and a classical retrieval system is exactly this one number. If rival_cos lands where Phi lands, 3489 parameters are decoration ON THIS VERB and the verdict is PHI_ADDS_NOTHING_ON_LOOKUP. Named for the brick and not for the wall: lookup is one verb, single-hop and retrieval-shaped by construction, and a rival that ties it says nothing about the exact algebra, about verbs where rows must be combined, or about generation, which 1-NN cannot do at all', 'paired_note': 'the rival answers the same lookup questions in the same run, so the gate is McNemar over the discordant items at the usual one-sided 1.645 - never two marginals. The rival over survivors is Bayes-optimal when the query context carries nothing, so a paired win IS the claim that the context channel carries information counts do not have', 'curve': v810, 'arc_enc_hash_before': v437, 'arc_enc_hash_after': v474, 'fp_version': v1128.v1083(), 'note': "The derivation moved into exact algebra and the mind kept only the judgment. Two runs measured 7.9k parameters approximating a quantity exactly computable from their own input, and the approximation degraded as the genuinely uncertain task grew beside it. Now count and compare are arithmetic over the same-value relation - exact, uncapped, scale-free - and the one trained surface is Phi, the coherence of a completed world: for each candidate the query row is filled in and the world that results is pooled to one scalar, 288's repair loop turned inward. The two trained surfaces this leaves in the whole architecture are Phi and, once values stop being exact strings, s_ij itself - both judgments, never arithmetic. Confidence for exact verbs reports 1.0, which is the honest statement that a computed answer is certain GIVEN the relation; when s_ij becomes a judgment its uncertainty enters through that same seam.", 'timestamp': v1276.v1035(v1277.v1198).v887(), 'wall_s': v777.v777() - v426}
    v0.v504(parents=True, exist_ok=True)
    v500 = v1084.v859(v167, indent=2)
    (v0 / f'stage289_decision{v206}.json').v537(v500, encoding='utf-8')
    if v191.v167:
        v860 = v137(v191.v167)
        v860.v864.v504(parents=True, exist_ok=True)
        v860.v537(v500, encoding='utf-8')
    v363(v1084.v859({'overall': v499, 'gates': v167['gates'], 'lookup': {v140: v234 for v140, v234 in v488.v219()}, 'paired': v489, 'ladder': v494}, indent=2))
    return 0
if v501 == '__main__':
    raise v861(v1085())