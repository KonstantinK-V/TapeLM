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
from _tape_speed import (INK_DEGENERATE, WORD_RULES, BigramBank, CachedBank, HashFp,
                         install_assertion_cache, install_fast_fp_addresses, verify_hash_ink,
                         verify_word_rule)
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 2890   # settable from --seed; see the flag's help for why it became necessary
# The closed answer sets. COUNT is capped because a derivation whose answer set grows with the
# tape is not a derivation, it is a lookup with extra steps; everything at or above the cap is
# one class and the mind must say "many".
COUNT_MAX = 5
COUNT_LABELS = tuple(str(i) for i in range(1, COUNT_MAX)) + (f"{COUNT_MAX}+",)
COMPARE_LABELS = ("first", "second", "equal")
LOG_PATH = RES / "_stage289_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def count_label(k: int) -> str:
    return COUNT_LABELS[min(k, COUNT_MAX) - 1]


# ------------------------------------------------------------------------- the exact algebra
# Whatever decides may not be approximate. count and compare are functions of the same-value
# relation alone, so they are computed, not learned. Written over a pluggable s_ij so that the
# day values stop being exact strings, the TRAINED sameness judgment slots in here and the
# arithmetic above it does not change.

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
    return [1.0 - (max(sim(vals[j], vals[i]) for j in range(i)) if i else 0.0)
            for i in range(len(vals))]


def soft_count(vals, sim=None) -> float:
    return sum(soft_new(vals, sim))


def exact_new(vals, same=None):
    """The hard special case, kept because integers are what the examiner compares."""
    sim = None if same is None else (lambda a, b: 1.0 if same(a, b) else 0.0)
    return [int(round(x)) for x in soft_new(vals, sim)]


def exact_count(vals, same=None) -> int:
    return sum(exact_new(vals, same))


def exact_answer(q):
    """The exact verdict for the exact verbs; raises on lookup, which is judged, not computed."""
    if q["verb"] == "count":
        return count_label(exact_count(q["vals"]))
    if q["verb"] == "compare":
        ka = exact_count(q["vals"][: q["n_first"]])
        kb = exact_count(q["vals"][q["n_first"]:])
        return "first" if ka > kb else "second" if kb > ka else "equal"
    raise ValueError("lookup is not exact: it is the judgment the mind is for")


# the one trained verb. 289c trains the same mind and must never feed it an exact verb.
TRAIN_VERBS = ("lookup",)


# ------------------------------------------------------------------------------- the questions

def count_question(pack, item):
    """How many distinct values does this address carry? The truth is a property of the tape."""
    vals = [pack["tape"].values[s] for s in item["slots"]]
    if len(vals) < 2:
        return None
    return {"verb": "count", "slots": list(item["slots"]), "vals": vals,
            "label": count_label(len(set(vals))), "S": item["S"],
            "address": item["address"]}


def compare_question(pack, a, b):
    """Which of two addresses carries more distinct values?

    Both address's mentions go into ONE graph, with a side indicator per row. Nothing tells the
    mind how many rows each side has beyond what it can see, and the answer is not a count but
    an ordering, so the two verbs cannot share a shortcut: a mind that memorised "this many rows
    means this label" for COUNT gets nothing here, where both sides sit in the same graph.
    """
    va = [pack["tape"].values[s] for s in a["slots"]]
    vb = [pack["tape"].values[s] for s in b["slots"]]
    if len(va) < 2 or len(vb) < 2:
        return None
    ka, kb = len(set(va)), len(set(vb))
    lab = "first" if ka > kb else "second" if kb > ka else "equal"
    return {"verb": "compare", "slots": list(a["slots"]) + list(b["slots"]),
            "vals": va + vb, "n_first": len(va), "label": lab,
            "S": a["S"], "S2": b["S"], "address": a["address"], "address2": b["address"]}


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
    slots = list(item["slots"])
    if len(slots) < 3:
        return None
    hid = rng.randrange(len(slots)) if hid is None else hid
    vals = [pack["tape"].values[s] for s in slots]
    cands = sorted(set(vals[:hid] + vals[hid + 1:]))
    if len(cands) < 2:
        return None                       # nothing to choose between: not a question
    if vals[hid] not in cands:
        return None                       # 286 failure mode 12: target not a function of input
    # survivors first, the query row last: its value is replaced by a sentinel that matches
    # nobody, so the same-value channel stays silent and only the context channels speak
    keep = slots[:hid] + slots[hid + 1:]
    return {"verb": "lookup", "slots": keep + [slots[hid]],
            "vals": [vals[i] for i in range(len(slots)) if i != hid] + [object()],
            "cands": cands, "label": cands.index(vals[hid]),
            "S": item["S"], "address": item["address"], "hid": hid,
            "query_row": len(keep)}


# ------------------------------------------------------- 290: the graph holds several addresses

# How many addresses each of the three relations may contribute. 0 keeps every earlier run bit
# for bit - the neighbourhood is not built, the two extra edge channels do not exist, and the
# parameter count stays 5601.
NEIGHBOURS = 0
REFUSE = False
# The per-question graph base (graph_base). --no-graph-cache falls back to the original
# per-candidate builder so the claim "caching changed nothing" can be CHECKED on a dense arm
# rather than asserted - the same role --no-fast-cos plays for the batched cosine.
GRAPH_CACHE = True
REFUSE_LABEL = "\x00REFUSE"     # cannot collide: no corpus value contains a NUL


def addr_parts(address):
    """(anchor, relation) of an fp address, split exactly as pack_from_corpus splits it."""
    tail = address.split(":", 1)[-1]
    a, r = (tail.split("|", 1) + [""])[:2]
    return a, r


def neighbourhood(p, addr, k, routes=("anchor", "rel", "word")):
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
    nb = p.setdefault("_nb", {})
    if (addr, k, routes) in nb:
        return nb[(addr, k, routes)]
    idx = p.get("_addr_index")
    if idx is None:
        by_anchor, by_rel, by_word, order = defaultdict(list), defaultdict(list), \
            defaultdict(list), {}
        med = p.get("_median")
        if med is None:
            lens = sorted(len(v) for v in p["postings"].values())
            med = lens[len(lens) // 2] if lens else 1
            p["_median"] = med
        for i, it in enumerate(p["items"]):
            order[it["address"]] = i
            an, rl = addr_parts(it["address"])
            by_anchor[an].append(it["address"])
            if rl:
                by_rel[rl].append(it["address"])
            seen = set()
            for sl in it["slots"]:
                for w in context_words(p["texts"][sl], exclude=p["tape"].values[sl]):
                    if len(p["postings"].get(w, ())) < med and w not in seen:
                        seen.add(w)
                        by_word[w].append(it["address"])
        idx = p["_addr_index"] = {"anchor": by_anchor, "rel": by_rel, "word": by_word,
                                  "order": order,
                                  "slots": {it["address"]: it["slots"] for it in p["items"]}}
    an, rl = addr_parts(addr)
    out, seen = [], {addr}
    here = idx["order"].get(addr, 0)

    def take(cands, n):
        for c in cands[:n]:
            if c not in seen:
                seen.add(c)
                out.append(c)

    # NEAREST IN TAPE ORDER, not first in tape order, and the difference is not cosmetic.
    # Relations like "and" or "the" are shared by hundreds of addresses, so "first k" would hand
    # the SAME three neighbours to every one of them - a constant appended to every graph, which
    # is a bias the mind can absorb and not a neighbourhood. Distance in tape order gives each
    # address its own, is deterministic, and uses a coordinate the project already treats as
    # structural (the ladder's `middle` rung is "adjacent in tape order"). Ties break on index.
    def near(cands):
        return sorted((a for a in cands if a != addr),
                      key=lambda a: (abs(idx["order"][a] - here), idx["order"][a]))

    if "anchor" in routes:
        take(near(idx["anchor"].get(an, ())), k)
    if "rel" in routes:
        take(near(idx["rel"].get(rl, ())), k)
    hits = Counter()
    for sl in idx["slots"].get(addr, ()):
        for w in context_words(p["texts"][sl], exclude=p["tape"].values[sl]):
            if len(p["postings"].get(w, ())) < p["_median"]:
                for a in idx["word"].get(w, ()):
                    if a != addr:
                        hits[a] += 1
    if "word" in routes:
        take([a for a, _ in sorted(hits.items(),
                                   key=lambda kv: (-kv[1], idx["order"][kv[0]]))], k)
    nb[(addr, k, routes)] = out
    return out


def lookup_sparse_question(p, item, rng, hid, k, routes=("anchor", "rel", "word")):
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
    own = list(item["slots"])
    if not (1 <= len(own) <= 2):
        return None
    hid = hid % len(own)
    hidden_slot = own[hid]
    truth = p["tape"].values[hidden_slot]
    rows = [s for s in own if s != hidden_slot]
    # AT MOST k ROWS PER NEIGHBOUR, and k is the constant already declared, not a new knob.
    # Without it one popular anchor with fifty mentions decides the graph on its own: n explodes,
    # the O(n^2) base with it, and the question stops being about the neighbourhood and becomes
    # about that one address. First k in tape order, so it stays deterministic like the
    # neighbourhood itself. "k addresses, k rows each" bounds n at 1 + 3k^2.
    for b in neighbourhood(p, item["address"], k, routes):
        rows += list(p["_addr_index"]["slots"].get(b, ()))[:k]
    rows = sorted(set(rows) - {hidden_slot})
    if not rows:
        return None
    cands = sorted({p["tape"].values[s] for s in rows})
    if len(cands) < 2:
        return None                       # nothing to choose between: not a question
    answerable = truth in cands
    if not answerable and not REFUSE:
        return None
    if REFUSE:
        cands = cands + [REFUSE_LABEL]
    label = cands.index(truth) if answerable else cands.index(REFUSE_LABEL) if REFUSE else None
    if label is None:
        return None
    return {"verb": "lookup", "sparse": True, "answerable": answerable,
            "slots": rows + [hidden_slot],
            "vals": [p["tape"].values[s] for s in rows] + [object()],
            "cands": cands, "label": label,
            "S": item["S"], "address": item["address"], "hid": hid,
            "own_rows": {s for s in own if s != hidden_slot},
            "query_row": len(rows)}


# ------------------------------------------------- 292: the value is not on the tape's own rows

OPEN = False

# "ce" is every run to date. "reward" optimises 280's payoff directly - see loss_for.
OBJECTIVE = "ce"

# which route supplied the `near` rung: [same anchor, neighbourhood]. If the first is always 0
# the sibling route is dead and only the composition keeps 292 alive - worth seeing, not
# assuming.
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
    own = list(item["slots"])
    if len(own) < 2:
        return None                       # nothing survives to be evidence
    hid = hid % len(own)
    hidden_slot = own[hid]
    truth = p["tape"].values[hidden_slot]
    rows = [s for s in own if s != hidden_slot]
    if any(p["tape"].values[s] == truth for s in rows):
        return None                       # the truth is still on a row: not a foreign target
    stub = {"cands": [truth], "address": item["address"], "slots": own,
            "S": item["S"], "query_row": len(rows)}
    attach_ladder(p, stub, by_anchor, all_values, rng)
    rungs = dict(stub.get("ladder") or {})
    # THE `near` RUNG ALMOST NEVER EXISTS, and the smoke said so before the run did: ladder
    # coverage has been 0 on every tape this project has built, 0/403 and again 0/20. The reason
    # is that `near` wants a sibling ADDRESS under the same anchor, and fp grouping makes those
    # rare. Requiring all three rungs would therefore have made 292 produce no questions at all
    # - an hour spent to print n = 0.
    #
    # The fix is not a looser rung, it is the neighbourhood 290 already defines: an address that
    # shares this one's anchor, relation or rare words is structurally near by a rule already
    # declared and already deterministic. The relation route alone fires on ~70% of addresses.
    # Nothing new is chosen; two existing structural notions are composed.
    if "near" not in rungs:
        used = {truth} | set(rungs.values())
        for b in neighbourhood(p, item["address"], 3):
            cand = [p["tape"].values[sl] for sl in p["_addr_index"]["slots"].get(b, ())]
            cand = [x for x in cand if x not in used]
            if cand:
                rungs["near"] = cand[0]
                break
    if len(rungs) != 3:
        return None                       # the tape could not supply a full ladder
    OPEN_NEAR[0 if stub.get("ladder", {}).get("near") == rungs["near"] else 1] += 1
    cands = sorted([truth] + [rungs[r] for r in LADDER])
    if len(cands) != 4:
        return None                       # a rung collided with the truth
    q = {"verb": "lookup", "open": True,
         "slots": rows + [hidden_slot],
         "vals": [p["tape"].values[s] for s in rows] + [object()],
         "cands": cands, "label": cands.index(truth),
         "rung_of": {rungs[r]: r for r in LADDER},
         "S": item["S"], "address": item["address"], "hid": hid,
         "query_row": len(rows)}
    # AND THE TRUTH MUST HAVE A MENTION SOMEWHERE ELSE, or the question has no content.
    #
    # The shared budget is the minimum available across all four candidates, so if the true
    # value occurs nowhere on the tape but at this one hidden row, that minimum is zero, nothing
    # is imported for anybody, and the four worlds are the SAME GRAPH four times. The mind would
    # then be scoring a coin flip and the run would report it as an accuracy. Requiring one
    # outside mention is what makes the completed world differ from the empty one at all - the
    # same reason IMPORT_K exists.
    if shared_import_budget(p, q, list(q["cands"])) < 1:
        return None
    return q


def open_rival_cos(p, q, bank, device):
    """The rival 292 actually has to beat: retrieval over the WHOLE TAPE, not over the address.

    Once every candidate brings its own mentions in, similarity is back in the game - it just
    searches the corpus instead of the address. That is RAG, stated exactly, and it is the fork
    the project has been circling: if nearest-imported-context lands where Phi lands, what we
    built is a search engine with extra steps.

    Same rows Phi is given - the shared import budget - and one rule: the candidate with a
    mention whose context is nearest the query's.
    """
    ck = p.setdefault("_ctx", {})

    def ctx(sl):
        if sl not in ck:
            c = bank.ctx_fp(p["texts"][sl], exclude=p["tape"].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
        return ck[sl]

    qc = ctx(q["slots"][q["query_row"]])
    if qc is None:
        return None
    k = shared_import_budget(p, q, list(q["cands"]))
    rows, owner = [], []
    for c in q["cands"]:
        for sl in outside_mentions(p, q, c)[:k]:
            r = ctx(sl)
            if r is not None:
                rows.append(r)
                owner.append(c)
    if not rows:
        return None
    # one matmul, not one device sync per imported row. Four candidates times k rows is small
    # per question and enormous over a run, and it is the same cost that made 289a slow.
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
    # k costs nothing here - no model, no gradient, and the pack is already built - so the run
    # reports the whole (route x k) surface instead of the single point it happens to use.
    # Whether more neighbours raise the hit rate or only add rows is the question that decides
    # if the sparse verb is starved or mis-specified, and it should not need four runs.
    for kk in sorted({k} | set(ks)):
        out[f"k={kk}"] = _audit_at(p, kk)
    return out


def _audit_at(p, k):
    out = {}
    for name, routes in (("anchor", ("anchor",)), ("rel", ("rel",)), ("word", ("word",)),
                         ("anchor+word", ("anchor", "word")),
                         ("all", ("anchor", "rel", "word"))):
        n = ans = rows = 0
        for it in p["items"]:
            if not (1 <= len(it["slots"]) <= 2):
                continue
            for hid in range(len(it["slots"])):
                own = list(it["slots"])
                hidden = own[hid % len(own)]
                truth = p["tape"].values[hidden]
                rr = [x for x in own if x != hidden]
                for b in neighbourhood(p, it["address"], k, routes):
                    rr += list(p["_addr_index"]["slots"].get(b, ()))[:k]
                rr = sorted(set(rr) - {hidden})
                if len(rr) < 1 or len({p["tape"].values[x] for x in rr}) < 2:
                    continue
                n += 1
                rows += len(rr)
                ans += int(truth in {p["tape"].values[x] for x in rr})
        out[name] = {"questions": n, "answerable": ans,
                     "hit_rate": ans / n if n else float("nan"),
                     "mean_rows": rows / n if n else float("nan")}
    return out


DROPPED = [0, 0]   # rows kept, rows offered - the density the mind actually trained on


# How many views reconciliation runs. 1 reproduces every earlier run bit for bit; set from
# --views. Module-level for the same reason as LADDER_ON: 289c imports these paths too.
VIEWS = 1

# How the views are cut. "thin" is recon3's random subsampling; "region" cuts the evidence
# rows into contiguous stretches of the tape. recon3 measured the difference between the two:
# random views share ~65% of their rows pairwise, so they are resamples of one reading and
# their disagreement is model noise - pooled LOST to single (z -1.67) and D was blind on train
# (auc 0.485). Regions are disjoint by construction, so agreement between them is evidence
# reconciling and disagreement is a property of the TAPE - the address is contested across
# the corpus - which is the thing worth refusing on.
VIEW_MODE = "thin"


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
    qr = q["query_row"]
    keep = [i for i in range(qr) if rng.random() < keep_p]
    kept = {q["vals"][i] for i in keep}
    for c in q["cands"]:
        # a candidate with no witness at all - REFUSE, or 292's foreign target - cannot be given
        # one, and asking for a choice from an empty list is a crash rather than a leak.
        wit = [i for i in range(qr) if q["vals"][i] == c]
        if c not in kept and wit:
            keep.append(rng.choice(wit))
    keep = sorted(set(keep))
    out = {**q, "slots": [q["slots"][i] for i in keep] + [q["slots"][qr]],
           "vals": [q["vals"][i] for i in keep] + [q["vals"][qr]],
           "query_row": len(keep)}
    out.pop("ladder", None)   # rungs were chosen against the full row set
    out.pop("_base", None)    # and the cached graph was built on that row set
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
    qr = q["query_row"]
    v = min(n, qr)
    cuts = [round(i * qr / v) for i in range(v + 1)]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        o = {**q, "slots": q["slots"][a:b] + [q["slots"][qr]],
             "vals": q["vals"][a:b] + [q["vals"][qr]],
             "query_row": b - a}
        o.pop("ladder", None)   # rungs were chosen against the full row set
        o.pop("_base", None)    # and the cached graph was built on that row set
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
    if VIEW_MODE == "thin":
        return [q] + [view_of(q, rng, 1.0 - ROW_DROPOUT) for _ in range(VIEWS - 1)], None
    qvs = [q] + region_views_of(q, VIEWS)
    M = torch.tensor([[float(c in set(v["vals"][:v["query_row"]])) for c in q["cands"]]
                      for v in qvs[1:]], device=device)
    return qvs, M


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
        L = L.masked_fill(mask == 0, float("-inf"))
    P = torch.softmax(L, dim=1)
    m = P.mean(0).clamp_min(1e-9)
    return float((P * (P.clamp_min(1e-9).log() - m.log())).sum(1).mean())


def reconciled(net, p, q, device, bank, rng):
    """Pooled logits, the single full-pass logits, and D, for one question. Training takes the
    gradient through the pooled logits; the exam reads all three. In thin mode D is over all
    views (view 0 included, as recon3 measured it); in region mode D is over the REGIONS only -
    the full view is their union and would only dilute the cross-region signal."""
    qvs, M = views_and_mask(q, rng, device)
    L = torch.stack([cand_logits_for(net, p, qv, device, bank) for qv in qvs])
    return pool_views(L, M), L[0], disagreement(L if M is None else L[1:], M)


ROW_DROPOUT = 0.0   # set from --row-dropout; view_of reads it so 289c can replay views too


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
    qr = q["query_row"]
    truth = q["cands"][q["label"]]
    surv = list(range(qr))
    if not any(q["vals"][i] == truth for i in surv):
        # 291's unanswerable questions have no witness for their answer - that IS the answer -
        # and 292's target is foreign to every row by construction. Thinning cannot preserve a
        # witness that does not exist, and the old line would have indexed an empty choice and
        # crashed an hour into the run. Such a question is simply not thinned.
        return None
    keep = [i for i in surv if rng.random() < keep_p]
    if not any(q["vals"][i] == truth for i in keep):
        keep.append(rng.choice([i for i in surv if q["vals"][i] == truth]))
    keep = sorted(set(keep))
    cands = sorted({q["vals"][i] for i in keep})
    if REFUSE_LABEL in q["cands"]:
        # rebuilding the candidate set from the surviving rows would silently delete the refusal
        # option, since nothing witnesses it - turning a 291 question into a plain one partway
        # through training. Only reachable with --refuse --row-dropout together, which no arm
        # currently uses, and left correct rather than left to be discovered.
        cands = cands + [REFUSE_LABEL]
    if len(cands) < 2:
        return None
    DROPPED[0] += len(keep)
    DROPPED[1] += len(surv)
    out = {**q, "slots": [q["slots"][i] for i in keep] + [q["slots"][qr]],
           "vals": [q["vals"][i] for i in keep] + [q["vals"][qr]],
           "cands": cands, "label": cands.index(truth), "query_row": len(keep)}
    out.pop("ladder", None)   # the rungs were chosen against the full row set
    out.pop("_base", None)    # and the cached graph was built on that row set
    return out


# ------------------------------------------------------------------- the ladder of wrong answers

LADDER = ("near", "middle", "far")

# Module-level rather than an argument because 289c calls questions_for too and would
# otherwise need the flag threaded through it. Set once from --no-ladder; never touched again.
LADDER_ON = True

# Which edge channels the graph may carry. Ablated one at a time to find out which one holds
# the 4.42-sigma paired win over counting, because the answer decides where the next effort
# goes: `rare` is exact set intersection and keeps word order irrelevant but loses nothing;
# `cos` is ctx_fp, a mean over words of a mean over characters - a bag of a bag, blurred twice.
# If `rare` alone carries the result the ink is not the bottleneck and bigrams would be wasted
# work; if `cos` carries it the blur is costing us and an order-aware encoder is worth building.
# How many mentions of a candidate are imported when its world is completed. THE LADDER RUN
# PROVED THIS IS NOT OPTIONAL: a value absent from the evidence produces a bit-identical graph
# whatever it is - its same-value row is all zeros, its share is 1/n, and the query row's
# CONTEXT never depended on the substituted value at all. So near, middle and far were the same
# input and Phi returned -19.535 for all three, to fifteen decimal places. Completing a world
# has to mean bringing in what the tape already says about that value, not writing a label on
# an empty row.
IMPORT_K = 2

EDGES = ("same", "cos", "rare")
# 290's two, added only when a neighbourhood is built. They are separate names so a dense run
# keeps exactly three channels and exactly 5601 parameters, and stays comparable to the board.
EDGES_NB = ("anchor", "rel")
EDGES_ON = set(EDGES)

# The ablation answered the question the comment above was asking. `rare` carries nothing -
# zeroing it leaves the run BIT-IDENTICAL, and `rare` alone scores 0.365 against a 0.413 rival
# - so the entire result lives in `cos`, which is order-blind. This is a change to KNOWLEDGE,
# not to the mind: no parameter moves anywhere on either axis.
#
# The ink is two independent choices and they are separate flags on purpose - one change per
# run, or the A/B measures their sum and attributes it to whichever was named last.
#
#   --fp    how a WORD becomes a vector.   arc  = the frozen stage191 encoder, order-blind
#                                                 inside the word and English-only by vocabulary
#                                          hash = character n-grams into a digest; nothing
#                                                 trained, no vocabulary, no OOV, any script
#   --ink   how a PHRASE becomes a vector. mean = today's average, order-blind
#                                          bigram = non-commutative binding of adjacent words
#   --words what counts as a word at all.  ascii = the stage194 rule
#                                          unicode = the same idea without Latin-only
INK = "mean"
FP = "arc"
WORDS = "ascii"


# The measured arc/mean tape, tau 0.90, 200k train lines - maxpool and qrank_big report it
# identically, and it is the shape every scoreboard number was taken on. It is the calibration
# TARGET for a write-ink change, and it is a measurement carrying its conditions, not a knob.
ARC_TRAIN_DENSITY = 2.9701          # mentions per address (2388 slots / 804 addresses)
# and the fragmentation this tape has, which a better write ink has to beat:
#   same_anchor_diff_relation 0.0785, mean_addresses_per_fact 1.0900   (held: 0.0841, 1.0886)


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
        if "tau" in memo:
            return memo["tau"]

        def density(t):
            out, addrs = s279.fp_addresses(asserts, bank, t, overlap, min_mentions,
                                           addr_key=addr_key)
            return (len(out) / len(addrs)) if addrs else float("nan"), len(addrs)

        t0 = time.time()
        d_lo, n_lo = density(lo)
        d_hi, n_hi = density(hi)
        trace = [(lo, d_lo, n_lo), (hi, d_hi, n_hi)]
        # the direction check. If a higher bar does not lower the density, the assumption behind
        # the search is false on this corpus and the number it would return is not interpretable.
        monotone = d_lo > d_hi
        a, b = lo, hi
        if not monotone:
            log(f"  tau calibration: density NOT decreasing in tau "
                f"({lo}->{d_lo:.3f}, {hi}->{d_hi:.3f}) - bisection is not valid here")
        elif not (d_hi <= target <= d_lo):
            log(f"  tau calibration: target {target:.4f} outside the bracket "
                f"[{d_hi:.3f}, {d_lo:.3f}] - clamping to the nearer end")
        else:
            for _ in range(iters):
                m = 0.5 * (a + b)
                d_m, n_m = density(m)
                trace.append((m, d_m, n_m))
                if d_m > target:
                    a = m           # too dense: raise the bar
                else:
                    b = m
        best = min(trace, key=lambda r: abs(r[1] - target) if r[1] == r[1] else float("inf"))
        memo["tau"] = best[0]
        memo["trace"] = [{"tau": round(t, 5), "density": d, "addresses": n} for t, d, n in trace]
        memo["achieved"] = best[1]
        memo["monotone"] = monotone
        log(f"  tau calibrated: {best[0]:.4f} -> density {best[1]:.4f} "
            f"(target {target:.4f}, {best[2]} addresses, {len(trace)} probes, "
            f"{time.time() - t0:.0f}s)")
        return best[0]

    resolve.memo = memo
    return resolve

# how often the rare channel is actually nonzero. If the answer is never, the channel is not
# "learned to be useless", it is structurally empty and the ablation measured nothing about
# word overlap at all. Diagnostic: reported, never read by the graph.
RARE_NNZ = [0, 0]

# one matmul for the pairwise cosines instead of a scalar read per pair. --no-fast-cos restores
# the original loop so the two can be compared rather than trusted.
FAST_COS = True

# Jaccard overlap of address sets between consecutive training tapes. Tape resampling is one of
# the four separation proofs; if a redraw returns the same addresses it proves nothing.
RESAMPLE_OVERLAP = []

# How often 290's two channels are actually nonzero. `rare` turned out structurally empty and
# the run said nothing until the rate was printed; these are the same instrument, installed
# before the fact rather than after it. A rate of 0 means the neighbourhood route that feeds
# the channel never fired and the arm measured something other than what it claims.
NB_NNZ = [0, 0, 0]

# rows per graph: sum, max, count. The bound above is a promise; this is the measurement.
GRAPH_N = [0, 0, 0]

# The failure mode that would actually kill bigram ink is not cancellation, it is COLLAPSE.
# Elementwise binding amplifies coordinates where both vectors are large; if arc_enc has a few
# dominant coordinates, every bigram bind aligns along them and all contexts come out cosine
# ~0.999 apart. rank_norm hides this from the mind - ranks are invariant to scale and offset -
# so the mind would read pure noise and the report would show nothing wrong. The spread of the
# raw cosines is what detects it. Sum and sum-of-squares over off-diagonal pairs.
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
    used = set(q["cands"])
    rungs = {}
    anchor = s289a.anchor_of(q["address"])
    sibs = [it for it in by_anchor.get(anchor, ())
            if it["address"] != q["address"]]
    for it in (rng.sample(sibs, len(sibs)) if sibs else ()):
        cand = [pack["tape"].values[sl] for sl in it["slots"]]
        cand = [v for v in cand if v not in used]
        if cand:
            rungs["near"] = cand[0]
            used.add(cand[0])
            break
    # adjacency in TAPE ORDER: the slot that arrived next to this address's last mention. That
    # is a property of when the corpus wrote things, not of what they mean.
    nxt = max(q["slots"]) + 1
    for sl in (nxt, min(q["slots"]) - 1):
        if 0 <= sl < pack["n_slots"] and pack["tape"].values[sl] not in used:
            rungs["middle"] = pack["tape"].values[sl]
            used.add(rungs["middle"])
            break
    for _ in range(8):
        v = all_values[rng.randrange(len(all_values))]
        if v not in used:
            rungs["far"] = v
            break
    q["ladder"] = rungs if (LADDER_ON and len(rungs) == 3) else {}
    return q


# ---------------------------------------------------------------------------------- the rivals

# count and compare have no rival any more: the exact algebra IS the optimal rule, and the
# label was built by an independent set-based count in the question builder. exact_mismatches
# compares the two, which is the sanity bolt that would fire if the algebra were wrong.

def lookup_rival(q):
    """286's majority rival - over the SURVIVORS only.

    The query row now sits in vals carrying a sentinel that equals nothing. Counting it would
    let the sentinel win any all-distinct address and hand the rival a guaranteed miss, which
    would flatter the mind against an opponent crippled by our own bookkeeping.
    """
    surv = [v for i, v in enumerate(q["vals"]) if i != q["query_row"]]
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
    own = q.get("own_rows") or set()
    ov = [q["vals"][i] for i in range(q["query_row"]) if q["slots"][i] in own]
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
    surv = [v for i, v in enumerate(q["vals"]) if i != q["query_row"]]
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
    ck = p.setdefault("_ctx", {})

    def ctx(sl):
        if sl not in ck:
            c = bank.ctx_fp(p["texts"][sl], exclude=p["tape"].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
        return ck[sl]

    qrow = q["query_row"]
    qc = ctx(q["slots"][qrow])
    if qc is None:
        return None, float("nan")
    rows = [(i, ctx(sl)) for i, sl in enumerate(q["slots"]) if i != qrow]
    rows = [(i, c) for i, c in rows if c is not None]
    if not rows:
        return None, float("nan")
    M = torch.stack([c for _, c in rows], 0)
    s = M @ qc
    top = int(s.argmax())
    best = q["vals"][rows[top][0]]
    # The MARGIN: how far the winning value's best row beats the best row of any OTHER value.
    # Label-free by construction, so splitting on it is not conditioning on the outcome.
    #
    # It exists to keep one specific excuse unavailable later. If Phi ties 1-NN, the tempting
    # move is "lookup was too retrieval-shaped a task" - true, but worthless said afterwards.
    # Declared in advance instead: on the half of the questions where similarity is CONFIDENT,
    # retrieval is close to a ceiling and a tie means little; on the half where the margin is
    # small, similarity has run out and anything that wins there is winning on something else.
    # Both halves are reported whatever happens.
    other = [float(s[j]) for j, (i, _) in enumerate(rows) if q["vals"][i] != best]
    margin = float(s[top]) - (max(other) if other else -1.0)
    return best, margin


# ------------------------------------------------------------------------------------- the mind

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

    def __init__(self, device, d: int = 32, n_edge: int = 3, n_node: int = 8, grown: int = 0):
        super().__init__()
        k = 3 if self.MAX_POOL else 2            # own-mean, all-mean, and the max
        self.edge = nn.Sequential(nn.Linear(n_edge, d), nn.GELU()).to(device)
        self.node = nn.Sequential(nn.Linear(n_node + k * d, d), nn.GELU()).to(device)
        # §19.7 - GROWING WITHOUT LOSING WHAT IS THERE. New channels enter with zero-initialised
        # INPUT columns, so at the instant of widening the function is bit-identical to the
        # narrower mind and training continues from where it stopped, rather than restarting.
        # `grown` says how many of the trailing edge inputs are new. This run trains from
        # scratch, where the idiom is inert - it is the mechanism for resuming a 289 checkpoint
        # into a 290 body, and _check290_sparse proves the identity holds.
        if grown:
            with torch.no_grad():
                self.edge[0].weight[:, n_edge - grown:] = 0.0
        self.lookup = nn.Sequential(nn.Linear((2 if self.MAX_POOL else 1) * d, d), nn.GELU(),
                                    nn.Linear(d, 1)).to(device)
        nn.init.zeros_(self.lookup[-1].weight)
        nn.init.zeros_(self.lookup[-1].bias)

    def body(self, E, same, nf):
        e = self.edge(E)
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


# --------------------------------------------------- module level, so 289c can reuse them
# 289c audits THIS mind on THESE questions. If it built its own it would be a second mind
# grading the first, so everything the audit needs lives here and is imported, not copied.

def sparse_questions_for(p, r):
    """290's question set: the addresses the dense verb throws away.

    count and compare still come from the dense items - they are exact algebra, they cost
    nothing, and dropping them would remove the sanity bolt that fires if the tape and the
    arithmetic disagree.
    """
    out = []
    for it in p["items"]:
        if not (1 <= len(it["slots"]) <= 2):
            continue
        for hid in range(len(it["slots"])):
            if (q := lookup_sparse_question(p, it, r, hid, NEIGHBOURS)) is not None:
                out.append(q)
    for it in p["items"]:
        if len(it["slots"]) >= 2 and (q := count_question(p, it)) is not None:
            out.append(q)
    shuffled = [it for it in p["items"] if len(it["slots"]) >= 2]
    r.shuffle(shuffled)
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        if (q := compare_question(p, a, b)) is not None:
            out.append(q)
    return out


def open_questions_for(p, r):
    """292's set. count and compare stay: they are exact algebra and they are the sanity bolt."""
    items = [it for it in p["items"] if len(it["slots"]) >= 2]
    by_anchor = defaultdict(list)
    for it in items:
        by_anchor[s289a.anchor_of(it["address"])].append(it)
    all_values = list(p["tape"].values)
    out = []
    for it in items:
        for hid in range(len(it["slots"])):
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
    """Every question the tape can supply, of all three verbs."""
    if OPEN:
        return open_questions_for(p, r)
    if NEIGHBOURS:
        return sparse_questions_for(p, r)
    items = [it for it in p["items"] if len(it["slots"]) >= 2]
    by_anchor = defaultdict(list)
    for it in items:
        by_anchor[s289a.anchor_of(it["address"])].append(it)
    all_values = list(p["tape"].values)
    out = []
    for it in items:
        if (q := count_question(p, it)) is not None:
            out.append(q)
        # every hidden position is its own question. One random draw per address gave lookup
        # n=21 held out, and the depth gate is a DIFFERENCE between two runs, which needs a
        # denominator one order larger. Enumerating costs nothing and is the same fix
        # wrong_relation needed in 289a.
        for hid in range(len(it["slots"])):
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
    return (len(q["cands"]) if q["verb"] == "lookup"
            else len(COUNT_LABELS) if q["verb"] == "count" else len(COMPARE_LABELS))


def truth_of(q):
    return q["cands"][q["label"]] if q["verb"] == "lookup" else q["label"]


def outside_mentions(p, q, value):
    """Mentions of a value that are NOT already in this question's evidence."""
    bv = p.get("_by_value")
    if bv is None:
        bv = defaultdict(list)
        for sl, v in enumerate(p["tape"].values):
            bv[v].append(sl)
        p["_by_value"] = bv
    here = set(q["slots"])
    return [sl for sl in bv.get(value, ()) if sl not in here]


def shared_import_budget(p, q, values):
    """One budget for every world compared in a question, and the reason is a leak.

    A local candidate's mentions are already IN the evidence, so it usually has nothing left to
    import; a ladder rung comes from elsewhere and always has K. Give each world what it
    happens to have and Phi can read "imported rows present" as "this one is wrong" - the
    landscape gate would then pass on a bookkeeping tell rather than on distance. The budget is
    therefore the minimum available across everything being scored, so every completed world
    carries the same number of rows.
    """
    return min([IMPORT_K] + [len(outside_mentions(p, q, v)) for v in values])


def row_meta(p):
    """slot -> (anchor id, relation id), for 290's two edge channels. Integers, not strings, so
    the channels are one broadcast comparison rather than n^2 python string compares."""
    m = p.get("_rowmeta")
    if m is None:
        an_id, rl_id, m = {}, {}, {}
        for it in p["items"]:
            an, rl = addr_parts(it["address"])
            a = an_id.setdefault(an, len(an_id))
            r = rl_id.setdefault(rl, len(rl_id)) if rl else -1
            for sl in it["slots"]:
                m[sl] = (a, r)
        p["_rowmeta"] = m
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
    if "_base" in q:
        return q["_base"]
    slots = list(q["slots"])
    n = len(slots)
    ck, ws = p.setdefault("_ctx", {}), p.setdefault("_words", {})
    for sl in set(slots):
        if sl not in ck:
            c = bank.ctx_fp(p["texts"][sl], exclude=p["tape"].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
            ws[sl] = set(context_words(p["texts"][sl], exclude=p["tape"].values[sl]))
    med = p.get("_median")
    if med is None:
        lens = sorted(len(v) for v in p["postings"].values())
        med = lens[len(lens) // 2] if lens else 1
        p["_median"] = med
    allc, allw = [ck[s] for s in slots], [ws[s] for s in slots]
    cos, shared = torch.zeros(n, n), torch.zeros(n, n)
    if FAST_COS and n > 1 and any(c is not None for c in allc):
        d0 = next(c for c in allc if c is not None)
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
            if not FAST_COS and allc[i] is not None and allc[j] is not None:
                cos[i, j] = cos[j, i] = float(allc[i] @ allc[j])
            inter = allw[i] & allw[j]
            rare = sum(1 for w in inter if len(p["postings"].get(w, ())) < med)
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
            mean_r = torch.zeros(len(uniq)).index_reduce_(0, inv, r, "mean", include_self=False)
            r = mean_r[inv] / (len(v) - 1 if len(v) > 1 else 1)
        else:
            r = torch.zeros_like(r)
        R = torch.zeros_like(M)
        R[iu[0], iu[1]] = r
        R[iu[1], iu[0]] = r
        return R

    chans = [rank_norm(cos) if "cos" in EDGES_ON else torch.zeros_like(cos),
             rank_norm(shared) if "rare" in EDGES_ON else torch.zeros_like(shared)]
    if NEIGHBOURS:
        # DISCRETE, like `same`: two rows either were written at the same anchor or were not.
        # No rank and no similarity - the tape already decided this at write time, and the
        # invariant says whatever is exact stays exact.
        meta = row_meta(p)
        a = torch.tensor([meta.get(s, (-1, -1))[0] for s in slots])
        r = torch.tensor([meta.get(s, (-1, -2))[1] for s in slots])
        aeq = ((a[:, None] == a[None, :]) & (a[:, None] >= 0)).float()
        req = ((r[:, None] == r[None, :]) & (r[:, None] >= 0)).float()
        aeq.fill_diagonal_(0.0)
        req.fill_diagonal_(0.0)
        chans += [aeq if "anchor" in EDGES_ON else torch.zeros_like(aeq),
                  req if "rel" in EDGES_ON else torch.zeros_like(req)]
        iu0 = torch.triu_indices(n, n, offset=1)
        if iu0.numel():
            NB_NNZ[0] += int((aeq[iu0[0], iu0[1]] > 0).sum())
            NB_NNZ[1] += int((req[iu0[0], iu0[1]] > 0).sum())
            NB_NNZ[2] += int(iu0.shape[1])
    qrow = q.get("query_row", -1)
    qcos, qmargin = torch.zeros(n), 0.0
    if qrow >= 0 and qrow < n and allc[qrow] is not None:
        ok = [i for i in range(n) if allc[i] is not None and i != qrow]
        if len(ok) > 1:
            idx = torch.tensor(ok)
            raw = cos[idx, qrow]
            o = raw.argsort()
            rr = torch.empty(len(ok))
            rr[o] = torch.arange(len(ok), dtype=torch.float32)
            qcos[idx] = rr / (len(ok) - 1)
            top = int(raw.argmax())
            bestv = q["vals"][ok[top]]
            other = [float(raw[kk]) for kk, i in enumerate(ok) if q["vals"][i] != bestv]
            spread = float(raw.max() - raw.min())
            if spread > 1e-9:
                qmargin = ((float(raw.max()) - max(other)) / spread) if other else 1.0
            else:
                qmargin = 1.0 if not other else 0.0
    nfirst = q.get("n_first", n)
    subj = [q["S"].lower() if (i < nfirst) else q.get("S2", q["S"]).lower() for i in range(n)]
    own = q.get("own_rows")
    base = {"n": n, "slots": slots, "chans": chans, "qcos": qcos, "qmargin": qmargin,
            "subj": subj, "nfirst": nfirst, "qrow": qrow,
            # 290's node indicator: is this row the query address's own, or a neighbour's? One
            # bit, no identity. Without it the mind cannot tell what it was asked about.
            "isown": ([float(s in own) for s in slots] if own is not None else None)}
    GRAPH_N[0] += n
    GRAPH_N[1] = max(GRAPH_N[1], n)
    GRAPH_N[2] += 1
    q["_base"] = base
    return base


def graph_from_base(p, q, bank, device, query_value):
    """One completed world, from the cached base. Only `same` and the count share change."""
    b = graph_base(p, q, bank, device)
    n, slots, qrow = b["n"], b["slots"], b["qrow"]
    vals = list(q["vals"])
    if query_value is not None:
        vals[qrow] = query_value
    # integer ids, one broadcast comparison - n^2 python string compares per candidate is the
    # same round-trip cost, in a different costume
    ids, seen = [], {}
    for v in vals:
        ids.append(seen.setdefault(v if isinstance(v, str) else id(v), len(seen)))
    t = torch.tensor(ids)
    same = (t[:, None] == t[None, :]).float()
    same.fill_diagonal_(0.0)
    E = torch.stack([same if "same" in EDGES_ON else torch.zeros_like(same)] + b["chans"],
                    -1).to(device)
    cnt = Counter(vals)
    nf = [[cnt[vals[i]] / n if (i != qrow or query_value is not None) else 0.0,
           float(b["subj"][i] in p["texts_lc"][slots[i]]),
           float(i >= b["nfirst"]),
           1.0 / n,
           float(i == qrow),
           0.0,                                  # imported: never, on this path k is 0
           float(b["qcos"][i]),
           b["qmargin"]]
          for i in range(n)]
    if b["isown"] is not None:
        for i in range(n):
            nf[i].append(b["isown"][i])
    nf = torch.tensor(nf, dtype=torch.float32, device=device)
    return E, same.unsqueeze(-1).to(device), nf


def build_graph(p, q, bank, device, query_value=None, import_k=None):
    """286/289a's graph verbatim, plus the side indicator COMPARE needs and, for a completed
    world, the candidate's own mentions imported from elsewhere on the tape."""
    k_eff = IMPORT_K if import_k is None else import_k
    if k_eff == 0 and not GRAPH_CACHE and query_value not in (None, REFUSE_LABEL):
        pass          # --no-graph-cache: fall through to the original per-candidate builder
    elif k_eff == 0 and query_value != REFUSE_LABEL:
        return graph_from_base(p, q, bank, device, query_value)
    if k_eff == 0 and query_value == REFUSE_LABEL:
        # THE REFUSAL WORLD, and it costs nothing to build: it is the question as it stands,
        # with the query row left unknown. Phi already scores exactly this object - the query
        # row carries a sentinel that matches nothing and its count share is zeroed - so
        # "I do not know" competes with the candidates through the SAME scalar, with no extra
        # parameter, no threshold and no reward term. See cand_logits_for.
        return graph_from_base(p, q, bank, device, None)
    slots, vals = q["slots"], q["vals"]
    n_evidence = len(slots)
    if query_value is not None:
        # the completed world: the query row is filled in with the candidate...
        slots, vals = list(slots), list(vals)
        vals[q["query_row"]] = query_value
        # ...and the tape's own mentions of that candidate come with it. Without this the
        # completion is a label on an empty row and every absent value gives the same graph.
        # Importing cannot leak the answer: a mention elsewhere carries no mark saying it is
        # the one that was hidden, and every world in a question gets the SAME budget - see
        # shared_import_budget for why an unequal one would be a tell rather than evidence.
        k = IMPORT_K if import_k is None else import_k
        for sl in outside_mentions(p, q, query_value)[:k]:
            slots.append(sl)
            vals.append(p["tape"].values[sl])
    n = len(slots)
    ck, ws = p.setdefault("_ctx", {}), p.setdefault("_words", {})
    for sl in set(slots):
        if sl not in ck:
            c = bank.ctx_fp(p["texts"][sl], exclude=p["tape"].values[sl])
            ck[sl] = F.normalize(c, dim=-1) if c is not None else None
            ws[sl] = set(context_words(p["texts"][sl], exclude=p["tape"].values[sl]))
    med = p.get("_median")
    if med is None:
        lens = sorted(len(v) for v in p["postings"].values())
        med = lens[len(lens) // 2] if lens else 1
        p["_median"] = med
    allc = [ck[s] for s in slots]
    allw = [ws[s] for s in slots]
    same = torch.zeros(n, n)
    cos = torch.zeros(n, n)
    shared = torch.zeros(n, n)
    # ONE matmul for every cosine, not n(n-1)/2 scalar reads off the GPU.
    #
    # `float(allc[i] @ allc[j])` synchronises the device once per pair. On a small address that
    # is invisible; on the probe's reserved addresses it is not - caching 2000 graphs took 864
    # seconds, which is 0.43s per graph, and the graphs are tiny. The work was never the
    # arithmetic, it was the round trips, for the third time in this stage.
    #
    # Same values up to float32 matmul associativity, and TF32 is disabled so the accumulation
    # stays float32. rank_norm reads only the ORDER of these numbers, so a difference in the
    # seventh decimal cannot move anything unless two pairs are that close; --no-fast-cos runs
    # the original loop so that can be checked rather than asserted.
    if FAST_COS and n > 1 and any(c is not None for c in allc):
        d0 = next(c for c in allc if c is not None)
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
            if not FAST_COS and allc[i] is not None and allc[j] is not None:
                cos[i, j] = cos[j, i] = float(allc[i] @ allc[j])
            inter = allw[i] & allw[j]
            rare = sum(1 for w in inter if len(p["postings"].get(w, ())) < med)
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
            mean_r = torch.zeros(len(uniq)).index_reduce_(0, inv, r, "mean", include_self=False)
            r = mean_r[inv] / (len(v) - 1 if len(v) > 1 else 1)
        else:
            r = torch.zeros_like(r)
        R = torch.zeros_like(M)
        R[iu[0], iu[1]] = r
        R[iu[1], iu[0]] = r
        return R

    # a disabled channel is zeroed, not removed: the tensor keeps its shape so the mind keeps
    # its parameter count and the arms stay the same size, which is what makes them comparable
    E = torch.stack([same if "same" in EDGES_ON else torch.zeros_like(same),
                     rank_norm(cos) if "cos" in EDGES_ON else torch.zeros_like(cos),
                     rank_norm(shared) if "rare" in EDGES_ON else torch.zeros_like(shared)],
                    -1).to(device)
    cnt = Counter(vals)
    nfirst = q.get("n_first", n)
    qrow = q.get("query_row", -1)
    # COMPARE puts two addresses in one graph, so the subject indicator has to ask each row
    # about ITS OWN subject. Asking every row about the first address's subject makes the
    # indicator false on the whole second side by construction - a second copy of the side
    # flag dressed as evidence, and the actual signal thrown away.
    subj = [q["S"].lower() if (i < nfirst or i >= n_evidence) else q.get("S2", q["S"]).lower()
            for i in range(n)]
    # HOW CLOSE IS THIS ROW TO THE QUERY ROW - as a rank, per row.
    #
    # ink_mean measured Phi at 0.692 against 1-NN over the same rows at 0.760, paired z -2.33,
    # with the entire loss on the low-margin half. The diagnosis is in the two lines above:
    # rank_norm ranks the WHOLE upper triangle at once, so the query row's similarities arrive
    # mixed into every other pair's, and no node feature says "you are the near one". Phi had to
    # reconstruct an argmax over one column out of a globally ranked matrix, through message
    # passing, on 3489 weights. The rival just takes the argmax.
    #
    # This adds NO information - the numbers were already in the graph - only access to them.
    # It is a rank, so the invariant is untouched, and if Phi still loses with it the loss is
    # real rather than an artefact of how the evidence was presented.
    # read the column out of `cos`, do not recompute it. The pairwise loop above already did
    # every one of these dot products, and each `float(a @ b)` forces a GPU synchronisation -
    # recomputing n of them per graph is ~475k extra syncs over a run, which is the same
    # per-item-round-trip cost that made 289a slow. Same numbers, no new work.
    qcos = torch.zeros(n)
    # HOW CONFIDENT that ordering is - the thing a rank cannot say.
    #
    # qrank_big split the retrieval comparison exactly where it was pre-declared to split, and
    # the two halves came back significant in OPPOSITE directions: where similarity is uncertain
    # Phi wins (z +2.40 held out, +1.50 train), and where similarity is confident Phi loses
    # every single discordant item (0 against 13, z -3.61 held out, -3.21 train). The same shape
    # on both sides, so it is structure and not overfitting.
    #
    # The reason is in the line above. A rank is scale-free by construction: it says which row is
    # nearest and cannot say whether it is nearest by a mile or by a hair. So Phi has no way to
    # know when to defer to the obvious answer, and it overrides it.
    #
    # The gap is given as a SHAPE, not a magnitude: the distance from the winning row to the best
    # row of any other value, over the full spread of the column. Invariant to any affine
    # rescaling of the similarity distribution, so it adds no absolute coordinate - and it
    # mirrors what rival_cos actually decides on, which is a gap between VALUES, not between rows.
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
            if spread > 1e-9:
                qmargin = ((float(raw.max()) - max(other)) / spread) if other else 1.0
            else:
                qmargin = 1.0 if not other else 0.0
    nf = torch.tensor(
        [[cnt[vals[i]] / n if (i != qrow or query_value is not None) else 0.0,
          float(subj[i] in p["texts_lc"][slots[i]]),
          float(i >= nfirst),                    # which side, only COMPARE ever sets it
          1.0 / n,                               # tape scale, a share not a count
          float(i == qrow),                      # the query row, only LOOKUP ever sets it
          float(i >= n_evidence),                # imported: the tape's own word on this value
          float(qcos[i]),                        # rank of closeness to the query row
          qmargin]                               # and how confident that ordering is
         for i in range(n)], dtype=torch.float32, device=device)
    return E, same.unsqueeze(-1).to(device), nf


def ladder_scores_for(net, p, q, device, bank):
    """Phi on the three wrong worlds, in ladder order. Empty when the tape could not supply one."""
    if not q.get("ladder"):
        return None
    k = shared_import_budget(p, q, list(q["cands"]) + [q["ladder"][r] for r in LADDER])
    outs = []
    for rung in LADDER:
        E, same, nf = build_graph(p, q, bank, device,
                                  query_value=q["ladder"][rung], import_k=k)
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
    vals = list(q["cands"]) + [q["ladder"][r] for r in LADDER] if q.get("ladder") \
        else list(q["cands"])
    k = shared_import_budget(p, q, vals)
    outs = []
    for c in q["cands"]:
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
    if q["verb"] != "lookup":
        raise ValueError(f"{q['verb']} is exact algebra now and has no loss")
    lg = cand_logits_for(net, p, q, device, bank)
    if OBJECTIVE == "reward":
        # TRAIN ON THE PAYOFF THE RUN IS SCORED BY, not on a proxy for it.
        #
        #     L = - sum_c p(c) R(c),   p = softmax(phi),   R from 280's fixed +1 / -1 / +0.75
        #
        # Differentiable in closed form: no sampling, no baseline, no RL machinery, and not one
        # new constant - R is the reward matrix the report already uses. Cross-entropy treats
        # every error alike, while the payoff says refusing an answerable question costs 0.25
        # and answering it wrongly costs 2.0, so optimising one and reporting the other is a
        # mismatch worth removing on its own merits.
        #
        # It will NOT rescue 291, and saying so is the point of writing it down. Under this
        # payoff answering beats abstaining exactly when p(correct) > 0.875 - the same derived
        # threshold - so at a 7% answerable rate the reward-optimal policy IS to refuse
        # everything. The collapse is not a defect of the loss; it is the correct response to a
        # neighbourhood that does not hold the answer. See neighbourhood_audit.
        pr = torch.softmax(lg, 0)
        R = torch.full_like(pr, -1.0)
        R[q["label"]] = 1.0
        if REFUSE and q.get("answerable") and REFUSE_LABEL in q["cands"]:
            R[q["cands"].index(REFUSE_LABEL)] = 0.75
        return -(pr * R).sum()
    lad = ladder_scores_for(net, p, q, device, bank)
    if lad is None:
        return F.cross_entropy(lg.unsqueeze(0), torch.tensor([q["label"]], device=device))
    allsc = torch.cat([lg, lad])
    loss = -(allsc[q["label"]] - torch.logsumexp(allsc, 0))
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
    if q["verb"] != "lookup":
        return 1.0, exact_answer(q), truth_of(q)
    lg = cand_logits_for(net, p, q, device, bank)
    pr = torch.softmax(lg, -1)
    k = int(pr.argmax())
    return float(pr[k]), q["cands"][k], truth_of(q)


def main() -> int:
    global SEED, LOG_PATH, LADDER_ON, EDGES_ON, IMPORT_K, INK, FP, WORDS, FAST_COS, VIEWS, \
        ROW_DROPOUT, VIEW_MODE, NEIGHBOURS, REFUSE, GRAPH_CACHE, OPEN, OBJECTIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--tape-period", type=int, default=50)
    ap.add_argument("--addresses", type=int, default=0)
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--address-tau", type=float, default=0.90)
    ap.add_argument("--tau-mode", choices=("absolute", "density"), default="absolute",
                    help="absolute keeps 279's fixed cosine and reproduces every earlier run "
                         "bit for bit. density derives tau so the WRITE ink produces a tape of "
                         "--tau-target-density mentions per address - required whenever the "
                         "write ink changes, because a different ink at a fixed cosine shatters "
                         "the tape and the threshold becomes what the arm measures")
    ap.add_argument("--tau-target-density", type=float, default=ARC_TRAIN_DENSITY,
                    help="mentions per address to calibrate to. Default is the MEASURED arc/mean "
                         "train tape (2388 slots / 804 addresses) that every scoreboard number "
                         "was taken on")
    ap.add_argument("--tau-calib-iters", type=int, default=12,
                    help="bisection steps for --tau-mode density. 12 over the full [0, 1] "
                         "bracket resolves tau to ~2e-4, which holds the density error under "
                         "0.005 even where the merge curve is steep; each extra step is pure "
                         "arithmetic because CachedBank has already inked the corpus")
    ap.add_argument("--address-overlap", type=int, default=2)
    ap.add_argument("--addr-key", choices=("two", "set", "mean"), default="two")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", choices=("corpus", "address"), default="corpus")
    ap.add_argument("--no-scan-cache", action="store_true",
                    help="disable the exact corpus-scan memo "
                         "(use to verify it changes nothing)")
    ap.add_argument("--no-fast-grouping", action="store_true",
                    help="disable the batched single-link grouping "
                         "(use to verify it changes nothing)")
    ap.add_argument("--wiki-bytes", type=int, default=0)
    ap.add_argument("--train-lines", type=int, default=0)
    ap.add_argument("--eval-lines", type=int, default=0)
    ap.add_argument("--import-k", type=int, default=IMPORT_K,
                    help="mentions of a candidate imported when completing its world; 0 "
                         "reproduces the broken ladder where every absent value looked alike")
    ap.add_argument("--edge-channels", type=str, default=",".join(EDGES),
                    help="comma list from same,cos,rare - zero the rest. Ablation to find "
                         "which channel carries the paired win over counting")
    ap.add_argument("--ink", choices=("mean", "bigram"), default=INK,
                    help="phrase axis: mean reproduces today's order-blind ctx_fp exactly; "
                         "bigram binds adjacent words with a fixed non-commutative permutation "
                         "so the ink can tell `X defeated Y` from `Y defeated X`")
    ap.add_argument("--fp", choices=("arc", "hash"), default=FP,
                    help="word axis: arc is the frozen stage191 encoder; hash is character "
                         "n-grams into a blake2b digest - nothing trained, no character "
                         "vocabulary, no OOV, every script")
    ap.add_argument("--words", choices=("ascii", "unicode"), default=WORDS,
                    help="what counts as a word. unicode only pays off with --fp hash: arc's "
                         "stoi has no Cyrillic, so a wider intake would just be discarded")
    ap.add_argument("--fp-ngram", type=int, default=3,
                    help="character n-gram length for --fp hash")
    ap.add_argument("--write-fp", choices=("arc", "hash"), default="arc",
                    help="ink used to GROUP mentions into addresses. Pinned by default so an "
                         "ink A/B varies reading only; 279's tau is an absolute cosine and a "
                         "different ink shatters the tape against it")
    ap.add_argument("--probe-period", type=int, default=250,
                    help="how often to score the fixed probe tape. The training curve is "
                         "measured on a different tape every resample and cannot tell "
                         "converged from overfitting; this one can")
    ap.add_argument("--views", type=int, default=1,
                    help="reconciliation (ROADMAP 20): the mind reads V independently thinned "
                         "views of each question with the SAME weights, logits are pooled by a "
                         "mean, and the views' disagreement is a label-free confidence signal. "
                         "1 reproduces every earlier run bit for bit; V>1 needs --row-dropout "
                         "as the thinning rate")
    ap.add_argument("--neighbours", type=int, default=0,
                    help="290 (ROADMAP §19): build N(a) from up to this many addresses per "
                         "route - shared anchor, shared relation, shared rare words - put all "
                         "their rows in ONE graph, and switch to the sparse verb. 0 reproduces "
                         "every earlier run bit for bit, including the 5601 parameter count")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="every draw in the run - tapes, questions, probe, views. Added when "
                         "292 came back with held z +2.59 against corpus retrieval and train "
                         "z -0.15 on the same weights: two samples of one quantity disagreeing "
                         "by 2.7 sigma. A second seed is the only cheap way to tell structure "
                         "from a lucky split, and there was no way to ask for one")
    ap.add_argument("--objective", choices=("ce", "reward"), default="ce",
                    help="ce is cross-entropy, every run to date. reward optimises 280's fixed "
                         "payoff directly: L = -sum_c p(c)R(c), closed form, no new constant. "
                         "Removes the mismatch between what is trained and what is scored; it "
                         "does not remove a collapse caused by an unanswerable task")
    ap.add_argument("--open", action="store_true",
                    help="292: the hidden value occurs exactly once at the address, so it is "
                         "FOREIGN to the evidence and no rule over the address's own rows can "
                         "reach it. Candidates are the truth and the three ladder rungs, all "
                         "four importing the same number of rows - the symmetric comparison the "
                         "ladder could never get in 289. Needs --import-k >= 1, because with 0 "
                         "imports all four worlds are the same graph")
    ap.add_argument("--no-graph-cache", action="store_true",
                    help="rebuild every channel per candidate, as before graph_base existed. "
                         "Dense arms only - use it to verify the cache changed nothing")
    ap.add_argument("--refuse", action="store_true",
                    help="291: keep the sparse questions whose answer is on NO row of N(a) and "
                         "let the mind score the world where the query row stays unknown. "
                         "Refusal becomes an action with a label the tape supplies, not a "
                         "threshold on a confidence score. Needs --neighbours")
    ap.add_argument("--view-mode", choices=("thin", "region"), default="thin",
                    help="how views are cut. thin = recon3's random subsampling (views share "
                         "~65%% of rows, D measured model noise, pooled lost to single). "
                         "region = contiguous stretches of the tape in write order - disjoint "
                         "by construction, deterministic, so D measures whether the CORPUS "
                         "agrees with itself at this address rather than whether one sampler "
                         "agrees with another")
    ap.add_argument("--row-dropout", type=float, default=0.0,
                    help="probability of dropping each evidence row during TRAINING, so the "
                         "mind sees the same fact at several densities. 0 reproduces every "
                         "earlier run bit for bit - it draws from its own generator")
    ap.add_argument("--dim", type=int, default=32,
                    help="width of the mind. Exposed so the max-pool result can be checked at "
                         "MATCHED parameter count: max-pool added 2048 weights along with the "
                         "max, and one of those two is the cause")
    ap.add_argument("--no-max-pool", action="store_true",
                    help="pool with the mean alone, as every run before this one did. A mean "
                         "cannot express existence, and a high-margin question is decided by "
                         "one row")
    ap.add_argument("--no-fast-cos", action="store_true",
                    help="build the pairwise cosine matrix with the original per-pair loop "
                         "(use to verify the batched version changes nothing)")
    ap.add_argument("--probe-frac", type=int, default=10,
                    help="one anchor in this many is reserved for the probe and excluded from "
                         "both training and held-out scoring, so the stopping step is never "
                         "chosen using an anchor the evaluation will ask about")
    ap.add_argument("--probe-size", type=int, default=200,
                    help="how many probe questions to score. Same questions every time - a "
                         "probe set that changes is the defect this replaces")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="keep the last step instead of the best probe step - reproduces every "
                         "run before the probe tape existed")
    ap.add_argument("--write-ink", choices=("mean", "bigram"), default="mean")
    ap.add_argument("--write-words", choices=("ascii", "unicode"), default="ascii")
    ap.add_argument("--no-ladder", action="store_true",
                    help="ablation: train Phi on the task term alone, the control the ladder "
                         "is measured against")
    ap.add_argument("--run-tag", type=str, default="")
    args = ap.parse_args()

    SEED = args.seed
    LADDER_ON = not args.no_ladder
    IMPORT_K = args.import_k
    INK, FP, WORDS = args.ink, args.fp, args.words
    FAST_COS = not args.no_fast_cos
    Deriver.MAX_POOL = not args.no_max_pool
    VIEWS, ROW_DROPOUT, VIEW_MODE = args.views, args.row_dropout, args.view_mode
    NEIGHBOURS, REFUSE, OPEN = args.neighbours, args.refuse, args.open
    OBJECTIVE = args.objective
    if OPEN and NEIGHBOURS:
        log("  --open and --neighbours are two different verbs; run them apart or the arm "
            "measures their sum and credits whichever was named last")
        return 1
    if OPEN and args.import_k < 1:
        log("  --open needs --import-k >= 1: with nothing imported, the true value and all "
            "three rungs give the identical graph and the question has no content")
        return 1
    if OPEN and args.no_ladder:
        log("  --open IS the ladder - the rungs are its candidates - so --no-ladder would "
            "leave it with one candidate")
        return 1
    GRAPH_CACHE = not args.no_graph_cache
    if args.no_graph_cache and NEIGHBOURS:
        log("  --no-graph-cache is a dense-arm verification path; it does not build 290's "
            "two extra edge channels and would silently score a 3-channel graph")
        return 1
    if REFUSE and not NEIGHBOURS:
        log("  --refuse needs --neighbours: the unanswerable questions are the sparse ones")
        return 1
    if NEIGHBOURS:
        if args.import_k:
            # the refusal world imports nothing - there is no value to import - so with k > 0 it
            # would carry fewer rows than every candidate world and Phi could read the row count
            # as the answer. That is the ladder's budget defect exactly, and it was settled by
            # refusing the construction rather than by compensating for it.
            log("  --neighbours needs --import-k 0: an imported world and the refusal world "
                "would carry different row counts, which is a bookkeeping tell, not evidence")
            return 1
    if VIEWS > 1 and VIEW_MODE == "thin" and ROW_DROPOUT <= 0:
        log("  --views > 1 with --row-dropout 0: every view is the same graph and the pool "
            "is decoration; set a thinning rate")
        return 1
    # region views need no thinning rate - the cut is deterministic - so --row-dropout there
    # keeps its rowdrop04 meaning: a training-only density augmentation, applied BEFORE the cut
    # A WRITE ink change at a fixed absolute tau is the failure the tau comment has warned about
    # since the ink patch: the tape shatters, lookup questions collapse to single digits, and the
    # arm reports an ink result that is a threshold artefact. Refused rather than warned about,
    # because the run would otherwise burn an hour producing an uninterpretable number.
    # An explicitly given absolute tau is a DECLARED writing rule, not the arc default left in
    # place by accident, and 290 needs one: with --min-mentions 1 the natural density is far
    # below the arc tape's, so calibrating to 2.97 would merge away exactly the single-mention
    # addresses the sparse verb exists for. Reusing the tau write_hash derived keeps the WRITING
    # RULE identical while letting the tape's shape be whatever the wider pool makes it - which
    # is the right invariant to hold when the question set, not the reader, is what changed.
    if (args.write_fp, args.write_ink) != ("arc", "mean") and args.tau_mode == "absolute" \
            and args.address_tau == 0.90:
        log(f"  --write-fp {args.write_fp} --write-ink {args.write_ink} rewrites the tape, and "
            f"279's tau is an absolute cosine: at 0.90 a different ink merges almost nothing. "
            f"Add --tau-mode density (target defaults to the measured arc tape, "
            f"{ARC_TRAIN_DENSITY} mentions/address).")
        return 1
    # both sides, not just the read side: a unicode WRITE rule against an arc encoder shatters
    # the addressing for the same reason and would look like an ink result
    for side, w, f in (("read", WORDS, FP), ("write", args.write_words, args.write_fp)):
        if w == "unicode" and f == "arc":
            log(f"  --{'' if side == 'read' else 'write-'}words unicode with an arc encoder "
                f"widens the intake into a vocabulary that cannot represent it; use hash there "
                f"or ascii")
            return 1
    EDGES_ON = {c.strip() for c in args.edge_channels.split(",") if c.strip()}
    if not EDGES_ON <= set(EDGES):
        log(f"  unknown edge channel in {sorted(EDGES_ON)}; allowed {EDGES}")
        return 1
    if not EDGES_ON:
        log("  every edge channel disabled: nothing to read")
        return 1
    # AFTER the parse, not before it. The first version turned the two channels on above and
    # this line then overwrote the set, so 290 would have run with anchor and rel silently all
    # zero - a dense arm carrying two dead channels and 64 spare weights, reporting a plausible
    # number. That is the rare_nonzero_rate failure exactly, and it is why the report prints
    # edge_channels: an empty channel has to be visible.
    if NEIGHBOURS:
        EDGES_ON |= set(EDGES_NB)

    tag = (args.run_tag and f"_{args.run_tag}") or ""
    tag += "_addrholdout" if args.holdout == "address" else ""
    LOG_PATH = RES / f"_stage289_log{tag}.txt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_steps = args.train_steps or (600 if args.smoke else 6000)
    n_addr = args.addresses or (300 if args.smoke else 400)

    log(f"Stage289 derivation start {datetime.now(timezone.utc).isoformat()} "
        f"device={device} holdout={args.holdout}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    # Under --fp hash the checkpoint is not loaded AT ALL - not loaded and frozen, absent. The
    # ink is a pure function, so there is no encoder to put on the device and no character
    # vocabulary to carry. That matters beyond tidiness: "the knowledge must not have to sit in
    # VRAM" is the point of the tape, and the character table was the last piece of knowledge
    # still living inside a weight file. G_arc_enc_frozen then holds in the strongest possible
    # way - weights that were never in the process cannot have moved - and the report says
    # `not_loaded` rather than a hash, so nobody can mistake it for a check that ran.
    can = None
    if FP != "hash":
        can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
        can.load_state_dict(torch.load(CKPT_P1, map_location=device,
                                       weights_only=False)["model"])
        can.eval()
        for p in can.parameters():
            p.requires_grad_(False)
    # CachedBank stays OUTERMOST so it memoises the finished vector whichever ink produced it.
    # The cache is in-memory and built fresh per run, so two inks can never contaminate one
    # another through it - but the report stamps both axes anyway, because a number without its
    # ink is not comparable to anything.
    # TWO BANKS, because one bank made the ink A/B measure two things at once.
    #
    # pack_from_corpus groups mentions into addresses with bank.fp, and 279's merge threshold
    # tau is an absolute cosine. Bigram and hash inks have a completely different similarity
    # distribution, so at the same tau almost nothing merges: addresses shatter, multi-mention
    # addresses vanish, and lookup questions fall from hundreds to single digits. That is not a
    # broken gate - it is a distribution shift against a fixed threshold, and it means the arms
    # were comparing "a different tape read differently" against "this tape read this way".
    #
    # So the WRITE ink is pinned and the READ ink is what --fp / --ink vary. Same tape in every
    # arm, one variable, and the A/B answers the question it was asked. Varying the write ink is
    # a separate experiment and needs tau re-derived as a QUANTILE of the observed cosine
    # distribution rather than kept as an absolute - otherwise the threshold, not the ink, is
    # what is being measured.
    rule = WORD_RULES[WORDS]

    def make_bank(fp_kind, ink_kind, r):
        base = (HashFp(device=device, n=args.fp_ngram, rule=r) if fp_kind == "hash"
                else FpBank(can, stoi, device))
        return CachedBank(BigramBank(base, rule=r) if ink_kind == "bigram" else base)

    # ONE bank when the two configurations agree, which is the default. Building two would give
    # the run two independent word caches and two independent context caches computing the same
    # vectors - every fingerprint inked twice, every context inked twice, for no gain. Splitting
    # the banks was meant to remove a confound, not to double the ink.
    bank = make_bank(FP, INK, rule)
    write_bank = (bank if (args.write_fp, args.write_ink, args.write_words) == (FP, INK, WORDS)
                  else make_bank(args.write_fp, args.write_ink, WORD_RULES[args.write_words]))
    ink_bank = bank._b
    base_bank = ink_bank._b if INK == "bigram" else ink_bank
    arc0 = s271.arc_enc_hash(can) if can is not None else "not_loaded"

    _nouns: dict = {}
    _raw_common = s279.common_nouns

    def _cached_common(lines, min_lower: int = 3):
        k = (id(lines), len(lines), min_lower)
        if k not in _nouns:
            _nouns[k] = _raw_common(lines, min_lower)
        return _nouns[k]

    s279.common_nouns = _cached_common
    if not args.no_scan_cache:
        install_assertion_cache(s279)
    if not args.no_fast_grouping:
        install_fast_fp_addresses(s279)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(args.wiki_bytes or (4_000_000 if args.smoke else 30_000_000))
    all_lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][: (args.train_lines or (3000 if args.smoke else 25000))]
    eval_lines = all_lines[cut:][: (args.eval_lines or (1500 if args.smoke else 12000))]
    # THE PROBE NEEDS ITS OWN LINES, or it is not a probe.
    #
    # The first version drew the probe tape from train_lines - the same lines the ~30 training
    # tapes are drawn from - so its addresses were very likely trained on, and a probe that has
    # seen its own questions under-reports the thing it exists to detect. It still showed a gap
    # (train 2.19 against probe 2.48 at step 750) because a resample never covers everything,
    # but the measurement was blunted by construction.
    #
    # Carved off the train side, so model selection still never touches the corpus holdout.
    log(f"  lines: train {len(train_lines)}, eval {len(eval_lines)} "
        f"(the probe reserves ANCHORS, not lines - see `reserved`)")
    if args.holdout == "address":
        eval_lines = train_lines

    # Everything the ink claims about itself, checked BEFORE six thousand steps are spent on it.
    # A wrong ink does not crash: it produces a plausible number that means nothing, and the
    # only moment it is cheap to catch is now.
    g_ink = bool(verify_word_rule())
    log(f"  word rule matches stage194: {g_ink}  (rule={WORDS}, fp={FP}, ink={INK})")
    if INK == "bigram":
        # the bigram bank re-derives the word list rather than borrowing the base bank's, so
        # prove the copy is faithful: if the two paths tokenise differently, an A/B between the
        # inks measures tokenisation and not order. torch.equal, not allclose.
        v = bool(ink_bank.verify_mean_path(train_lines[:200]))
        g_ink &= v
        log(f"  bigram tokenisation matches base mean-ink: {v}")
    if FP == "hash":
        # G_arc_enc_frozen is vacuous here - there are no weights to move - so this is the gate
        # that stands in its place, and it checks the digest rather than re-running the code.
        v, notes = verify_hash_ink(base_bank)
        g_ink &= v
        log(f"  hash ink deterministic and digest-faithful: {v}  {notes}")
    if not g_ink:
        log("  ABORT: the ink does not do what it says it does")
        return 1

    def side(address: str) -> int:
        return int(hashlib.sha1(s289a.anchor_of(address).encode("utf-8")).hexdigest(), 16) & 1

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
        h = hashlib.sha1(f"probe:{s289a.anchor_of(address)}".encode("utf-8")).hexdigest()
        return int(h, 16) % args.probe_frac == 0

    # one rule for every pack in the run - train, held and probe alike. In absolute mode it is
    # the plain number and nothing changes; in density mode it calibrates on the first pack and
    # every later pack reuses the frozen tau, so all three tapes are written by the same rule.
    tau_rule = (args.address_tau if args.tau_mode == "absolute"
                else tau_for_density(args.tau_target_density, args.tau_calib_iters, log))

    def new_pack(r, lines, want, probe=False, n_addr_over=None):
        # write_bank, not bank: the tape must be identical across reading arms
        p = s280.pack_from_corpus(lines, bank=write_bank, tok=tok, pad_id=pad_id, device=device,
                                  rng=r, n_addr=n_addr_over or n_addr,
                                  min_mentions=args.min_mentions,
                                  tau=tau_rule, overlap=args.address_overlap,
                                  soft_match=0.0, min_per_family=8, addr_key=args.addr_key)
        p = dict(p)
        if args.holdout == "address":
            p["items"] = [it for it in p["items"] if side(it["address"]) == want]
        # reserved anchors go to the probe and to nothing else - not to training, and not to the
        # held-out tape either, so the step that gets selected was never chosen using an anchor
        # that scoring will later ask about
        p["items"] = [it for it in p["items"] if reserved(it["address"]) == probe]
        return p

    questions = questions_for

    def by_verb(qq):
        d = defaultdict(list)
        for q in qq:
            d[q["verb"]].append(q)
        return d

    net = Deriver(device, d=args.dim,
                  n_edge=3 + (len(EDGES_NB) if NEIGHBOURS else 0),
                  n_node=8 + (1 if NEIGHBOURS else 0),
                  grown=len(EDGES_NB) if NEIGHBOURS else 0)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = int(sum(x.numel() for x in net.parameters()))

    def cand_logits(p, q):
        return cand_logits_for(net, p, q, device, bank)

    def loss_of(p, q):
        return loss_for(net, p, q, device, bank)

    pack = new_pack(rng, train_lines, 0)
    qs = questions(pack, rng)
    bv = by_verb(qs)
    n_lad = sum(1 for q in bv.get("lookup", ()) if q.get("ladder"))
    log(f"  tape: {pack['n_addresses']} addresses, {pack['n_slots']} slots | "
        f"questions {json.dumps({k: len(v) for k, v in bv.items()})} | params {n_params}")
    # printed early because a low number is the one thing that makes the landscape gate
    # unanswerable, and it is fixed with --addresses (more sibling relations), not the model
    log(f"  ladder coverage: {n_lad}/{len(bv.get('lookup', ()))} lookup questions have all "
        f"three rungs; the rest train on the task term alone")
    if len(bv.get("lookup", ())) < s286.MIN_ANSWERED:
        log("  too few lookup questions: raise --train-lines")
        return 1

    held = new_pack(random.Random(SEED + 99), eval_lines, 1)
    held_qs = questions(held, random.Random(SEED + 7))

    # THE PROBE TAPE - one tape, built once, never trained on, used only to decide when to stop.
    #
    # ink_mean showed why this is not housekeeping. Against the retrieval rival Phi scored
    # z +2.60 on train and z -2.33 held out: a 4.9-sigma swing on a paired comparison, which
    # rules out task difficulty as the cause because the rival answers the same questions. That
    # is overfitting, and the run had no way to see it happening - the training curve is measured
    # on a DIFFERENT tape every 200 steps, so its points are not comparable to each other and it
    # cannot tell converged from diverged.
    #
    # It is drawn from TRAIN lines, so model selection stays on the training side of the corpus
    # split and the held-out tape remains untouched by any choice made here. A different rng draw
    # than the training tapes, and no gradient ever flows from it.
    # the whole train corpus, and enough addresses drawn that ~n_addr survive the reservation
    t_pp = time.time()
    probe = new_pack(random.Random(SEED + 555), train_lines, 0, probe=True,
                     n_addr_over=n_addr * args.probe_frac)
    log(f"  probe pack: {len(probe['items'])} reserved addresses "
        f"({time.time() - t_pp:.0f}s to build)")
    # capped, and the cap is a prefix of a deterministic order: the probe set must be the SAME
    # questions at every evaluation or the curve stops being comparable to itself, which is the
    # exact defect it exists to fix.
    # laddered questions are excluded rather than special-cased: the cached path computes the
    # plain cross-entropy, which is what loss_for reduces to without a ladder, and a question
    # scored by a different objective than the training loss would make the curve meaningless.
    # Coverage has been 0/403 on every tape so far, so this costs nothing today and stays
    # correct if that changes.
    probe_qs = [q for q in questions(probe, random.Random(SEED + 556))
                if q["verb"] in TRAIN_VERBS and not q.get("ladder")][:args.probe_size]
    # BUILD THE PROBE'S GRAPHS ONCE.
    #
    # build_graph never touches the net - it reads the pack, the question and the bank - so for
    # a FIXED tape and a FIXED question set the graphs are constant for the whole run. Rebuilding
    # them at every probe evaluation was 200 x ~9 = 1800 O(n^2) python loops repeated two dozen
    # times, which is the same mistake fp_addresses made and HANDOFF 9b was written about: the
    # cost was not compute, it was identical work redone. Precomputed, a probe evaluation is
    # 1800 forward passes through 3521 parameters and nothing else.
    #
    # Memory is a few megabytes: n is a handful of rows, so each graph is a few hundred floats.
    t_probe = time.time()
    probe_graphs = []
    pv_rng = random.Random(SEED + 6060)   # deterministic views: the probe must be the same
    for q in probe_qs:                     # measurement at every evaluation
        qvs, qm = views_and_mask(q, pv_rng, device)
        views = []
        for qv in qvs:
            k = shared_import_budget(probe, qv, list(qv["cands"]))
            views.append([build_graph(probe, qv, bank, device, query_value=c, import_k=k)
                          for c in qv["cands"]])
        probe_graphs.append((views, qm, torch.tensor([q["label"]], device=device)))
    log(f"  probe tape: {len(probe_qs)} lookup questions, never trained on; "
        f"{sum(len(g) for vs, _, _ in probe_graphs for g in vs)} graphs cached "
        f"({VIEWS} view(s)/question) in {time.time() - t_probe:.0f}s")

    @torch.no_grad()
    def probe_loss():
        if not probe_graphs:
            return float("nan")
        net.eval()
        tot = 0.0
        for views, qm, label in probe_graphs:
            L = torch.stack([torch.stack([net.phi(E, same, nf) for E, same, nf in graphs])
                             for graphs in views])
            tot += float(F.cross_entropy(pool_views(L, qm).unsqueeze(0), label))
        net.train()
        return tot / len(probe_graphs)

    # time it before trusting it. Every cost regression in this project came from adding work
    # and estimating its price instead of measuring it - HANDOFF 9b - and this line is what makes
    # the price visible on the first minute of a run rather than in hour three.
    t0_probe = time.time()
    probe_loss()
    dt_probe = time.time() - t0_probe
    n_evals = len([s for s in range(1, n_steps + 1)
                   if s % args.probe_period == 0 or s == n_steps])
    log(f"  probe eval: {dt_probe:.2f}s x {n_evals} = {dt_probe * n_evals / 60:.1f} min added "
        f"to this run")

    drop_rng = random.Random(SEED + 4242)
    best = {"loss": float("inf"), "step": 0, "state": None}
    probe_curve = []

    losses, curve = [], []
    for step in range(1, n_steps + 1):
        if (step - 1) % args.tape_period == 0 and step > 1:
            prev_addr = {it["address"] for it in pack["items"]}
            pack = new_pack(rng, train_lines, 0)
            qs = questions(pack, rng)
            bv = by_verb(qs)
            # HOW MUCH OF THE TAPE ACTUALLY CHANGED.
            #
            # Resampling is one of the four proofs of separation in HANDOFF 1 - the mind cannot
            # memorise a tape it keeps losing. That argument holds only if a resample brings new
            # addresses, and qrank says it may not: the tape carried 681 addresses out of a pool
            # of roughly 835, so a redraw returns ~82% of the same ones and the mechanism is
            # decorative. Measured now rather than assumed, because it decides whether
            # overfitting is a model problem or a corpus problem.
            now = {it["address"] for it in pack["items"]}
            if prev_addr:
                RESAMPLE_OVERLAP.append(len(prev_addr & now) / max(1, len(prev_addr | now)))
            if not bv.get("lookup"):
                log("  empty tape after resample")
                return 1
        # only the judged verb trains; the exact verbs have nothing to teach a mind
        q = bv["lookup"][rng.randrange(len(bv["lookup"]))]
        # its OWN generator: drawing from rng would shift every later resample and the arm
        # would stop being comparable to the runs it is meant to be compared with
        if VIEWS > 1:
            # ROADMAP 20: one mind, V worlds, one loss. softmax of the pooled logits is the
            # normalised product of the per-view distributions, so the single cross-entropy IS
            # the reconciliation - no second term, no weight between views, and at V=1 this
            # reduces exactly to the old objective. In region mode the gradient teaches the
            # SKILL the run measures: read one stretch of the tape and be poolable with the
            # other stretches - not just read everything at once.
            if VIEW_MODE == "region" and args.row_dropout > 0:
                thin = drop_rows(q, drop_rng, 1.0 - args.row_dropout)
                if thin is not None:
                    q = thin
            pooled, _, _ = reconciled(net, pack, q, device, bank, drop_rng)
            loss = F.cross_entropy(pooled.unsqueeze(0),
                                   torch.tensor([q["label"]], device=device))
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
            probe_curve.append({"step": step, "probe_loss": pl})
            if pl < best["loss"]:
                best = {"loss": pl, "step": step,
                        "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
        if step % max(1, n_steps // 8) == 0:
            curve.append({"step": step, "loss": float(np.mean(losses[-200:])),
                          "probe_loss": probe_curve[-1]["probe_loss"] if probe_curve else None})
            log(f"  step {step}/{n_steps} train={np.mean(losses[-200:]):.4f} "
                f"probe={probe_curve[-1]['probe_loss'] if probe_curve else float('nan'):.4f}")

    # restore the weights the probe liked best. Selection on train-side data; the held-out tape
    # never entered the choice. --no-early-stop keeps the last step, which is what every run
    # before this one did, so the two are comparable.
    if not args.no_early_stop and best["state"] is not None:
        net.load_state_dict(best["state"])
        log(f"  early stop: restored step {best['step']} (probe {best['loss']:.4f}) "
            f"of {n_steps}")
    net.eval()
    arc1 = s271.arc_enc_hash(can) if can is not None else "not_loaded"

    # ------------------------------------------------------------------------------- score
    @torch.no_grad()
    def examine(p, qq):
        st = {v: {"n": 0, "model": 0, "rival": 0, "rival_cos": 0, "floor": 0.0} for v in
              ("count", "compare", "lookup")}
        conf, hits = Counter(), []
        exact_bad = 0
        b10 = b01 = 0                      # model-only-right / rival-only-right, paired in-run
        cos_items = []                     # (model_right, rival_cos_right, margin) per question
        view_rows = []                     # (pooled_right, single_right, disagreement D)
        sparse_rows = []                   # 290/291: answerable, right, refused, rivals, margins
        open_rows = []                     # 292: mind right, corpus-retrieval right, answered
        vrng = random.Random(SEED + 7788)  # fresh per examine call: deterministic views
        rung_sum = {k: 0.0 for k in ("true",) + LADDER}
        rung_n, concord, pairs, ties = 0, 0, 0, 0
        budgets = []
        for q in qq:
            v = q["verb"]
            if v == "lookup":
                if VIEWS > 1:
                    # the pooled logits ARE the model's answer; view 0 is the full graph, so the
                    # in-run single-view baseline makes ensemble-vs-single a PAIRED comparison
                    # on the same questions, and D is the label-free confidence signal
                    lg, l0, dd = reconciled(net, p, q, device, bank, vrng)
                    view_rows.append([int(int(lg.argmax()) == q["label"]),
                                      int(int(l0.argmax()) == q["label"]), dd])
                else:
                    lg = cand_logits(p, q)
                pred = q["cands"][int(lg.argmax())]
                truth = q["cands"][q["label"]]
                lad = ladder_scores_for(net, p, q, device, bank)
                if q.get("ladder"):
                    budgets.append(shared_import_budget(
                        p, q, list(q["cands"]) + [q["ladder"][r] for r in LADDER]))
                if lad is not None:
                    # the landscape check: does Phi fall as the substitution moves away? Three
                    # ordered pairs per question, each 50/50 under a Phi that carries no
                    # distance information, so the null is exactly 0.5 and needs no choosing.
                    seq = [float(lg[q["label"]])] + [float(x) for x in lad]
                    for name, val in zip(("true",) + LADDER, seq):
                        rung_sum[name] += val
                    rung_n += 1
                    for a_, b_ in zip(seq, seq[1:]):
                        if a_ == b_:
                            ties += 1          # unmeasurable, not a failure - see below
                            continue
                        concord += int(a_ > b_)
                        pairs += 1
                riv = lookup_rival(q)
                st[v]["floor"] += 1.0 / len(q["cands"])
                hits.append({"k": f"{q['address']}#{q.get('hid', len(q['slots']))}",
                             "hit": int(pred == truth)})
                # WHICH QUESTIONS THE CLASSIC PAIRED GATES MAY SEE.
                #
                # Both rivals answer with a value lying on the evidence rows. On a 292 question
                # the truth is foreign to every row by construction, and on an unanswerable 291
                # question the truth is "refuse", so in both cases a rival with no refusal option
                # CANNOT be right - ever. Scoring those questions here would print z near +18
                # and read as a triumph while being a tautology about the construction.
                #
                # So they are excluded from the paired gates and answered elsewhere: 292 has its
                # own rival (retrieval over the whole tape, which CAN reach a foreign value) and
                # 291 gives both rivals a probe-calibrated refusal threshold. The rivals' raw
                # accuracy is still reported, because "retrieval inside the address scores zero
                # by construction" is exactly §19.6's claim and deserves to be a number.
                fair = not q.get("open") and (q.get("answerable", True) or not REFUSE)
                if fair:
                    if (pred == truth) and (riv != truth):
                        b10 += 1
                    elif (pred != truth) and (riv == truth):
                        b01 += 1
                # and against the retrieval rival, paired the same way. A question where the
                # ink gave no query context is skipped rather than scored: rival_cos cannot
                # answer it, and counting that as a loss for the rival would flatter the mind.
                rcos, rmargin = lookup_rival_cos(p, q, bank, device)
                if rcos is not None:
                    st[v]["rival_cos"] += int(rcos == truth)
                    if fair:
                        cos_items.append((int(pred == truth), int(rcos == truth), rmargin))
                if q.get("open"):
                    # the landscape, read off the SAME logits that produced the answer rather
                    # than a second pass. Three ordered pairs, each 50/50 under a Phi carrying
                    # no distance information, so the null is exactly 0.5 and needs no choosing.
                    byr = {nm: val_ for val_, nm in q["rung_of"].items()}
                    seq = [float(lg[q["cands"].index(truth)])] + \
                          [float(lg[q["cands"].index(byr[r_])]) for r_ in LADDER]
                    for name, val in zip(("true",) + LADDER, seq):
                        rung_sum[name] += val
                    rung_n += 1
                    for a_, b_ in zip(seq, seq[1:]):
                        if a_ == b_:
                            ties += 1
                        else:
                            concord += int(a_ > b_)
                            pairs += 1
                    orc = open_rival_cos(p, q, bank, device)
                    open_rows.append([int(pred == truth), int(orc == truth),
                                      int(orc is not None)])
                if q.get("sparse"):
                    # how many rows of ITS OWN the address still has. 0 is the case 19.5 is
                    # about - nothing of the address survives the hiding, so 1-NN inside it is
                    # undefined and the answer can only come from the neighbourhood. Reported,
                    # because --min-mentions decides whether that case exists at all and a run
                    # that never produced one has not tested the claim.
                    orp = own_row_rival(q)
                    sparse_rows.append([int(q["answerable"]), int(pred == truth),
                                        int(pred == REFUSE_LABEL), int(riv == truth),
                                        int(rcos == truth), counting_margin(q), rmargin,
                                        len(q["own_rows"]), int(orp == truth),
                                        int(orp == REFUSE_LABEL)])
            else:
                pred = exact_answer(q)
                truth = q["label"]
                riv = pred                 # the rival IS the same computation; kept to prove it
                st[v]["floor"] += 1.0 / (len(COUNT_LABELS) if v == "count"
                                         else len(COMPARE_LABELS))
                exact_bad += int(pred != truth)
            st[v]["n"] += 1
            st[v]["model"] += int(pred == truth)
            st[v]["rival"] += int(riv == truth)
            conf[(v, str(truth), str(pred))] += 1
        out = {}
        for v, t in st.items():
            if not t["n"]:
                continue
            out[v] = {"n": t["n"],
                      "model_accuracy": t["model"] / t["n"],
                      "rival_accuracy": t["rival"] / t["n"],
                      "random_floor": t["floor"] / t["n"]}
            if v == "lookup":
                # over the questions rival_cos could ANSWER, not over all of them. A question
                # where the ink gave no query context is skipped from the paired comparison, so
                # counting it in the denominator here would report a rival the comparison never
                # faced. All 104 were answerable in ink_mean, so this moved nothing yet - it is
                # a denominator that would have started lying the moment one was not.
                out[v]["rival_cos_accuracy"] = (t["rival_cos"] / len(cos_items)
                                                if cos_items else float("nan"))
                out[v]["rival_cos_answered"] = len(cos_items)
        out_lad = {"n_questions": rung_n, "pairs": pairs,
                   "concordant": concord,
                   # A tie means the two completed worlds were the SAME INPUT, which is a fact
                   # about the construction and not evidence against the mind. The first ladder
                   # run scored ties as failures and printed z = -3.44 while Phi was in fact
                   # right on 25 of the 27 pairs it could actually see. Ties are counted and
                   # excluded; a high count means the ladder is broken again.
                   "ties_excluded": ties,
                   # The budget is why the ladder cannot be settled here. The true value LIVES
                   # at this address, so it has little to import; a rung comes from elsewhere
                   # and has everything. Unequal budgets let Phi read "imported rows present"
                   # as "wrong" - a bookkeeping tell, not distance - and equal budgets collapse
                   # to zero, which makes the rungs the same input again. There is no third
                   # option on THIS construction. The ladder belongs to 292, where the true
                   # value is foreign too and the comparison is symmetric by construction.
                   "import_budget_zero_rate": (sum(1 for b in budgets if b == 0)
                                               / max(1, len(budgets))),
                   "import_budget_mean": (sum(budgets) / max(1, len(budgets))),
                   "concordance": (concord / pairs) if pairs else float("nan"),
                   "z_vs_half": (((concord / pairs) - 0.5) / math.sqrt(0.25 / pairs))
                   if pairs else float("nan"),
                   "mean_phi": {k: (rung_sum[k] / rung_n if rung_n else float("nan"))
                                for k in ("true",) + LADDER}}
        disc = b10 + b01
        out["lookup_paired_vs_rival"] = {
            "model_only_right": b10, "rival_only_right": b01, "discordant": disc,
            "mcnemar_z": ((b10 - b01) / math.sqrt(disc)) if disc else float("nan")}
        def mcnemar(items):
            a = sum(1 for m, r, _ in items if m and not r)
            b = sum(1 for m, r, _ in items if r and not m)
            d = a + b
            # what z would be if the mind won EVERY discordant item. If that ceiling is under
            # the gate, the comparison cannot pass however good the mind is, and reporting a
            # failed gate would be reporting a conclusion where there is only a shortage of
            # data. qrank had 2 discordant items of 56 questions: ceiling 1.41 against a 1.645
            # gate, so the verdict was decided by the corpus and not by the model.
            return {"n": len(items), "model_only_right": a, "rival_only_right": b,
                    "discordant": d,
                    "mcnemar_z": ((a - b) / math.sqrt(d)) if d else float("nan"),
                    "max_achievable_z": math.sqrt(d) if d else 0.0,
                    "underpowered": bool(math.sqrt(d) <= 1.645)}

        out["lookup_paired_vs_rival_cos"] = mcnemar(cos_items)
        # The pre-declared split. Median of a label-free quantity, so which questions land in
        # which half was fixed before any answer was scored. LOW margin is where similarity has
        # run out and where a reader that adds anything has to show it; HIGH margin is where
        # 1-NN is near a ceiling and a tie proves little either way. Both are printed whatever
        # they say - the point of declaring it now is that neither can be chosen afterwards.
        ms = sorted(m for _, _, m in cos_items if not math.isnan(m))
        med = ms[len(ms) // 2] if ms else float("nan")
        out["lookup_paired_vs_rival_cos_by_margin"] = {
            "median_margin": med,
            "low_margin": mcnemar([it for it in cos_items if it[2] <= med]),
            "high_margin": mcnemar([it for it in cos_items if it[2] > med])}
        out["ladder"] = out_lad
        out["exact_mismatches"] = exact_bad
        out["confusion"] = {f"{a}|{b}->{c}": k for (a, b, c), k in sorted(conf.items())}
        out["lookup_item_hits"] = sorted(hits, key=lambda h: h["k"])
        out["_views"] = view_rows
        out["_sparse"] = sparse_rows
        if open_rows:
            a_ = sum(1 for m, r_, _ in open_rows if m and not r_)
            b_ = sum(1 for m, r_, _ in open_rows if r_ and not m)
            d_ = a_ + b_
            out["open"] = {
                "n": len(open_rows), "random_floor": 0.25,
                "accuracy": sum(r[0] for r in open_rows) / len(open_rows),
                "corpus_retrieval_accuracy": sum(r[1] for r in open_rows) / len(open_rows),
                "corpus_retrieval_answered": sum(r[2] for r in open_rows),
                # 19.6, as a number rather than an argument: a rule over the address's own rows
                # cannot reach a value that is on none of them, so its accuracy here is zero by
                # construction and it is excluded from the paired gates above.
                "within_address_rivals_undefined": True,
                # paired in-run: same questions, same imported rows, one rule each
                "paired_vs_corpus_retrieval": {
                    "model_only_right": a_, "rival_only_right": b_, "discordant": d_,
                    "mcnemar_z": ((a_ - b_) / math.sqrt(d_)) if d_ else float("nan"),
                    "max_achievable_z": math.sqrt(d_) if d_ else 0.0,
                    "underpowered": bool(math.sqrt(d_) <= 1.645)}}
        if sparse_rows:
            ans = [r for r in sparse_rows if r[0]]
            una = [r for r in sparse_rows if not r[0]]
            out["sparse"] = {
                "n": len(sparse_rows), "n_answerable": len(ans), "n_unanswerable": len(una),
                # the two halves are reported apart, always. A single accuracy over both would
                # let a mind that refuses everything look competent on a tape with many
                # unanswerable questions, and a mind that never refuses look competent on one
                # with few - the mix is a property of the corpus, not of the reader.
                "acc_answerable": (sum(r[1] for r in ans) / len(ans)) if ans else float("nan"),
                "refuse_recall": (sum(r[2] for r in una) / len(una)) if una else float("nan"),
                "false_refusal": (sum(r[2] for r in ans) / len(ans)) if ans else float("nan"),
                # the 19.5 case, split out: no own row survives, so 1-NN inside the address is
                # undefined and only the neighbourhood can answer
                # the control for "did the neighbourhood add anything": answer from the
                # address's OWN surviving rows alone. Without it a 290 arm cannot tell a
                # neighbourhood that carries facts from one that carries the row it already had.
                "rival_own_row_accuracy": sum(r[8] for r in sparse_rows) / len(sparse_rows),
                "n_no_own_row": sum(1 for r in sparse_rows if r[7] == 0),
                "acc_no_own_row": (sum(r[1] for r in sparse_rows if r[7] == 0)
                                   / max(1, sum(1 for r in sparse_rows if r[7] == 0))),
            }
        return out

    ex = examine(held, held_qs)
    ctrl = examine(pack, qs)

    # ------------------------------------------------------- reconciliation (ROADMAP 20)
    # Three numbers, each answering one clause of the design, all from data already computed:
    #   pooled_vs_single  does reading V views beat reading one? Paired in-run - view 0 IS the
    #                     single pass, so McNemar on the same questions, never two marginals.
    #   d_auc             does disagreement predict error? Hanley-McNeil z via s286, errors as
    #                     positives, so the confidence signal has to clear its own noise.
    #   refusal           280's rewards are fixed (+1/-1/+0.75), so answering beats abstaining
    #                     iff p > 0.875 - a constant DERIVED, not chosen. The threshold on D is
    #                     read off the TRAIN curve (largest prefix by ascending D whose accuracy
    #                     still clears 0.875) and only then applied held out.
    tr_v, he_v = ctrl.pop("_views", []), ex.pop("_views", [])
    # the probe's rows, from graphs already cached - forwards only. recon3 is why these exist:
    # train D-auc 0.485 against held 0.702, because the mind fits the training tapes and its
    # views stop disagreeing there whatever the tape says. A threshold read off train was
    # therefore blind (coverage 0.004) while the SIGNAL was real held out (z 4.71). The probe
    # was never trained on, so its D-error relation is the held one - and it is train-side,
    # so calibrating there still never touches the held-out tape.
    pr_v = []
    if VIEWS > 1:
        with torch.no_grad():
            for views, qm, label in probe_graphs:
                L = torch.stack([torch.stack([net.phi(E, same, nf) for E, same, nf in graphs])
                                 for graphs in views])
                lg = pool_views(L, qm)
                pr_v.append([int(int(lg.argmax()) == int(label)),
                             int(int(L[0].argmax()) == int(label)),
                             disagreement(L if qm is None else L[1:], qm)])
    recon = None
    if VIEWS > 1 and tr_v and he_v:
        def paired(rows):
            a = sum(1 for pl, sg, _ in rows if pl and not sg)
            b = sum(1 for pl, sg, _ in rows if sg and not pl)
            d = a + b
            return {"pooled_only_right": a, "single_only_right": b, "discordant": d,
                    "mcnemar_z": ((a - b) / math.sqrt(d)) if d else float("nan")}

        # WHICH ANSWER the confidence signal is about. Column 0 is the pooled answer, column 1 is
        # the full graph read once. Two runs now say pooling does not answer better - recon3
        # z -1.67, region3 z -0.50, never a win - and the reason is structural rather than
        # incidental: a region cannot see an edge to a row in another region, while the full
        # graph sees all of them. Splitting is what makes DISAGREEMENT measurable; it is not a
        # better way to decide. So both are reported and neither is assumed, because a refusal
        # rule calibrated against the pooled answer is calibrated against the answer we are about
        # to stop giving.
        def d_auc(rows, i=0):
            err = [r[2] for r in rows if not r[i]]
            okd = [r[2] for r in rows if r[i]]
            if not err or not okd:
                return {"auc": float("nan"), "z": float("nan")}
            a = s286.auc(err, okd)          # D higher on errors
            return {"auc": a, "z": s286.auc_z(a, len(err), len(okd)),
                    "n_err": len(err), "n_ok": len(okd)}

        # Per DISTINCT D, admitted as a whole group, and stop at the first group that breaks
        # the bound. Scanning row by row and keeping the last index that cleared is wrong twice:
        # accuracy degrades slowly so a tail of errors sneaks in one at a time, and rows tied at
        # one D cannot be split anyway - admitting any admits all. Verified offline: the row-wise
        # rule kept a 30-error tail at coverage 0.96 and accuracy 0.750, this one cuts it.
        def t_star_of(rows, i=0):
            groups = defaultdict(list)
            for r in rows:
                groups[r[2]].append(r)
            t, cum, seen = None, 0, 0
            for d in sorted(groups):
                g = groups[d]
                if (cum + sum(x[i] for x in g)) / (seen + len(g)) < 0.875:
                    break
                cum += sum(x[i] for x in g)
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
            n_ok = sum(1 for r in ans if r[i])
            acc = (n_ok / len(ans)) if ans else float("nan")
            sel = (n_ok - (len(ans) - n_ok) + 0.75 * ref_n) / len(he_v)
            # is the answered accuracy actually above the break-even, or is it 0.875 plus noise?
            z_be = ((acc - 0.875) / math.sqrt(0.875 * 0.125 / len(ans))) if ans else float("nan")
            return {
                "p_star": 0.875, "calibrated_on": "probe" if pr_v else "train",
                "d_threshold": t, "d_threshold_from_train": t_star_of(tr_v, i),
                "held_coverage": len(ans) / len(he_v), "held_n_answered": len(ans),
                "held_acc_answered": acc,
                "z_acc_vs_breakeven": z_be,
                "held_reward_selective": sel,
                "held_reward_always": sum(1 if r[i] else -1 for r in he_v) / len(he_v),
                "held_reward_blanket_refusal": 0.75,
            }

        pv = paired(he_v)
        hd, rf = d_auc(he_v), refusal_of(0)
        hd1, rf1 = d_auc(he_v, 1), refusal_of(1)
        recon = {
            "views": VIEWS, "view_mode": VIEW_MODE,
            "thin_keep_p": (1.0 - ROW_DROPOUT) if VIEW_MODE == "thin" else None,
            "held_pooled_vs_single": pv,
            # the pooled answer - what region3 and recon3 reported, kept under the same keys so
            # those two runs stay directly comparable to whatever comes next
            "held_d_auc": hd, "probe_d_auc": d_auc(pr_v), "train_d_auc": d_auc(tr_v),
            "refusal": rf,
            # and the same two numbers about the FULL-graph answer, which is the one the evidence
            # says to keep. Nothing is recomputed and no run is repeated: both columns were
            # already scored on every question, so this only stops the refusal rule from being
            # calibrated against an answer we are about to stop giving.
            "answer_full": {"held_d_auc": hd1, "probe_d_auc": d_auc(pr_v, 1),
                            "train_d_auc": d_auc(tr_v, 1), "refusal": rf1},
            # declared before the number, all three from constants already in the file:
            #   pooled must at least not lose to single (recon3: z -1.67);
            #   D must predict error held out (recon3 passed at 4.71 - regions must not break it);
            #   answering must beat BLANKET refusal, whose payoff is exactly 0.75 by 280's
            #   rewards - recon3's 0.743 was blanket refusal minus epsilon, not a result.
            # The gates are read on the pooled answer, unchanged, so region3's verdict is not
            # restated by moving the goalposts; the full-graph column sits beside it as evidence.
            "gates": {
                "G_pooled_not_worse": bool(pv["mcnemar_z"] >= 0 if pv["discordant"] else True),
                "G_d_predicts_error_held": bool(hd.get("z", 0) == hd.get("z", 0)
                                                and hd.get("z", 0) > 1.645),
                "G_refusal_beats_blanket": bool(rf["held_reward_selective"] > 0.75),
            },
        }
        log(f"  RECON {json.dumps(recon)}")

    # ------------------------------------------------------------------- 291: refusal as an act
    # The mind refuses by SCORING a world - the one where the query row stays unknown - so it
    # needs no threshold and no reward term; the rivals are parameter-free rules that always
    # name a value, so without a refusal option of their own they would lose every unanswerable
    # question by construction and this would measure the option rather than the judgment.
    # They therefore get the same discipline: their own label-free confidence, a threshold read
    # off the PROBE by the same derived 0.875, applied held out and never refitted.
    refuse_block = None
    he_sp, tr_sp = ex.pop("_sparse", []), ctrl.pop("_sparse", [])
    if REFUSE and he_sp:
        pr_sp = []
        for q in probe_qs:
            if not q.get("sparse"):
                continue
            t = q["cands"][q["label"]]
            rc, mx = lookup_rival_cos(probe, q, bank, device)
            pr_sp.append([int(q["answerable"]), int(lookup_rival(q) == t), int(rc == t),
                          counting_margin(q), mx])

        # WHAT REFUSING IS WORTH, and the first version of this got it wrong in a way that
        # would have decided the verdict.
        #
        # 280's +0.75 is the price of a HEDGE: forgoing an answer that might have been right.
        # On an unanswerable question refusing is not a hedge, it is the correct answer, and
        # paying it 0.75 would have made the capability 291 exists to measure worth less than
        # getting an ordinary question right. So:
        #
        #     correct           +1     (including a refusal of an unanswerable question)
        #     wrong             -1
        #     refused an ANSWERABLE question   +0.75, the hedge, 280's constant untouched
        #
        # And blanket refusal is therefore no longer the constant 0.75 - it is
        # unanswerable_rate + 0.75 x answerable_rate, a property of the tape. Computed, not
        # assumed, because on a tape that is mostly unanswerable that floor is high and a mind
        # that refuses everything must not be able to clear it by accident.
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
            t, cum, seen = None, 0, 0
            for m in sorted(g, reverse=True):
                grp = g[m]
                if (cum + sum(x[right_i] for x in grp)) / (seen + len(grp)) < 0.875:
                    break
                cum += sum(x[right_i] for x in grp)
                seen += len(grp)
                t = m
            return t

        def reward(rows, right_i, margin_i, t):
            tot = 0.0
            for r in rows:
                # `not (m >= t)` rather than `m < t`, so a NaN margin refuses instead of
                # answering. A rival that cannot rank must abstain, not guess.
                if t is None or not (r[margin_i] >= t):
                    tot += 1.0 if not r[0] else 0.75
                else:
                    tot += 1.0 if r[right_i] else -1.0
            return tot / len(rows)

        def fixed_reward(rows, right_i, refused_i):
            """A rival that refuses by RULE rather than by threshold - scored exactly as the
            mind is, so the only difference between them is the judgment."""
            return sum(1.0 if r[right_i] else (0.75 if r[refused_i] else -1.0)
                       for r in rows) / len(rows)

        t_cnt = thresh(pr_sp, 1, 3) if pr_sp else None
        t_cos = thresh(pr_sp, 2, 4) if pr_sp else None
        ans = [r for r in he_sp if r[0]]
        una = [r for r in he_sp if not r[0]]
        blanket = (len(una) + 0.75 * len(ans)) / len(he_sp)
        mind_r = sum(1.0 if r[1] else (0.75 if r[2] else -1.0) for r in he_sp) / len(he_sp)
        refuse_block = {
            "n": len(he_sp), "n_answerable": len(ans), "n_unanswerable": len(una),
            "unanswerable_rate": len(una) / len(he_sp),
            "mind": {"acc_answerable": (sum(r[1] for r in ans) / len(ans)) if ans else
                     float("nan"),
                     "refuse_recall": (sum(r[2] for r in una) / len(una)) if una else
                     float("nan"),
                     "false_refusal": (sum(r[2] for r in ans) / len(ans)) if ans else
                     float("nan"),
                     "coverage": 1.0 - sum(r[2] for r in he_sp) / len(he_sp),
                     "reward": mind_r},
            "rival_counting": {"threshold_from_probe": t_cnt,
                               "reward": reward(he_sp, 3, 5, t_cnt)},
            "rival_retrieval": {"threshold_from_probe": t_cos,
                                "reward": reward(he_sp, 4, 6, t_cos)},
            # the bookkeeping shortcut, scored the same way. Phi tying this means the mind read
            # "does this address still have a row of its own" and nothing else.
            "rival_own_row": {"reward": fixed_reward(he_sp, 8, 9),
                              "acc_answerable": (sum(r[8] for r in ans) / len(ans)) if ans
                              else float("nan"),
                              "refuse_recall": (sum(r[9] for r in una) / len(una)) if una
                              else float("nan")},
            "blanket_refusal_reward": blanket,
            "always_answer_ceiling": (len(ans) - len(una)) / len(he_sp),
            # declared before the number. Refusal is a capability, so the claim is not "the mind
            # is more accurate" but "the mind knows when it cannot answer" - and the only honest
            # way to say that is against rivals that were given the best refusal rule available
            # to them, on the same questions, calibrated the same way.
            "gates": {
                "G_refuse_beats_blanket": bool(mind_r > blanket),
                "G_refuse_beats_counting": bool(mind_r > reward(he_sp, 3, 5, t_cnt)),
                "G_refuse_beats_retrieval": bool(mind_r > reward(he_sp, 4, 6, t_cos)),
                "G_refuse_beats_own_row_shortcut": bool(mind_r > fixed_reward(he_sp, 8, 9)),
            },
        }
        log(f"  REFUSE {json.dumps(refuse_block)}")

    # The ink is upstream of the ADDRESSING, not only of the reading: pack_from_corpus groups
    # mentions into an address with bank.fp and bank.ctx_fp, so changing --fp rebuilds the tape
    # rather than re-reading the same one. That is legitimate - addressing is knowledge, not
    # mind - but it means an arm can win by writing a better tape rather than by reading better,
    # and the two are not the same claim. These four numbers say whether the tapes are even
    # comparable; if they move a lot, the lookup delta is about grouping and must be said so.
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
        by_fact, by_pair, seen = defaultdict(set), defaultdict(set), Counter()
        for it in pk["items"]:
            addr = it["address"]
            anchor = s289a.anchor_of(addr)
            for sl in it["slots"]:
                val = pk["tape"].values[sl]
                by_fact[(anchor, val)].add(addr)
                seen[(anchor, val)] += 1
                by_pair[frozenset((anchor, val.lower()))].add((anchor, val.lower()))
        multi = [k for k in by_fact if seen[k] >= 2]
        rev = [p for p, ends in by_pair.items() if len(p) == 2 and len(ends) == 2]
        return {"facts_written_twice": len(multi),
                "same_anchor_diff_relation": (sum(1 for k in multi if len(by_fact[k]) > 1)
                                              / len(multi)) if multi else float("nan"),
                "mean_addresses_per_fact": (sum(len(by_fact[k]) for k in multi) / len(multi))
                if multi else float("nan"),
                "reversed_pairs": len(rev),
                "reversed_pair_rate": len(rev) / max(1, len(by_pair))}

    def tape_shape(pk, qq):
        lk = [q for q in qq if q["verb"] == "lookup"]
        return {"slots": len(pk["texts"]), "addresses": len(pk["addr_slots"]),
                "mentions_per_address": (len(pk["texts"]) / len(pk["addr_slots"])
                                         if pk["addr_slots"] else float("nan")),
                "lookup_questions": len(lk),
                "mean_candidates": (sum(len(q["cands"]) for q in lk) / len(lk)
                                    if lk else float("nan")),
                "paraphrase": paraphrase_split(pk)}
    log(f"  HELD {json.dumps({k: v for k, v in ex.items() if k != 'lookup_item_hits'})}")
    log(f"  CONTROL {json.dumps({k: v for k, v in ctrl.items() if k != 'lookup_item_hits'})}")

    g_arc = arc0 == arc1
    g_task = ex.get("lookup", {}).get("n", 0) >= 2 * s286.MIN_ANSWERED
    # sanity, not achievement: the algebra must reproduce the tape's truth exactly, on every
    # question, both splits. One mismatch means the algebra is wrong, not that the mind is.
    g_exact = bool(ex.get("exact_mismatches", 1) == 0 and ctrl.get("exact_mismatches", 1) == 0)
    lk = ex.get("lookup", {})
    pv = ex.get("lookup_paired_vs_rival", {})
    g_floor = bool(lk and lk["model_accuracy"] > lk["random_floor"])
    # the claim, paired at the project's usual one-sided 1.645: the completed-world score wins
    # against the count-optimal rival on the items where they disagree
    g_beats_counts = bool(pv.get("discordant", 0) >= 2 * s286.MIN_ANSWERED
                          and not math.isnan(pv.get("mcnemar_z", float("nan")))
                          and pv["mcnemar_z"] > 1.645)
    # The gate that can end the project as a MIND rather than as a search engine.
    #
    # Beating the counting rival proves the context channel carries information counts do not
    # have. It does not prove that reading it takes a mind - and with hash ink the ink is
    # Random Indexing with a fastText-shaped word vector, so "the mind reads the tape" and
    # "cosine similarity picks the answer" are one measurement apart. rival_cos is 1-NN over
    # the SAME evidence rows with zero parameters. If it lands where the mind lands, the 3489
    # weights are decoration and what we built is retrieval; that has to be reported, not
    # explained away.
    pc = ex.get("lookup_paired_vs_rival_cos", {})
    g_beats_retrieval = bool(pc.get("discordant", 0) >= 2 * s286.MIN_ANSWERED
                             and not math.isnan(pc.get("mcnemar_z", float("nan")))
                             and pc["mcnemar_z"] > 1.645)
    # the landscape, not the boundary: Phi must FALL as the substitution moves away. Each of
    # the three ordered pairs is 50/50 under a Phi carrying no distance information, so the
    # null is 0.5 by construction and the gate is the same one-sided 1.645 used everywhere.
    ld = ex.get("ladder", {})
    g_ladder = bool(ld.get("pairs", 0) >= 6 * s286.MIN_ANSWERED
                    and not math.isnan(ld.get("z_vs_half", float("nan")))
                    and ld["z_vs_half"] > 1.645)

    # g_beats_retrieval joins the OK condition rather than sitting beside it as decoration: a
    # run that beats counting but ties 1-NN has not shown a mind, and a verdict that says
    # otherwise is the kind of thing this project exists not to do.
    # A comparison that COULD NOT have passed is not a comparison that failed. qrank had 2
    # discordant items across 56 held-out questions: even winning both gives z = 1.41 against a
    # 1.645 gate, so the corpus decided the verdict and the model never got a vote. Calling that
    # PHI_ADDS_NOTHING_ON_LOOKUP would be reporting a conclusion where there is only a shortage
    # of data, which is the failure this project exists not to commit.
    underpowered = bool(pc.get("underpowered", True))
    # A null overall z is not "adds nothing" when the pre-declared halves are individually
    # significant in OPPOSITE directions. qrank_big: +2.40 where similarity is uncertain, -3.61
    # where it is confident, cancelling to -0.34. Averaging those into one verdict destroys the
    # only finding in the run.
    bym = ex.get("lookup_paired_vs_rival_cos_by_margin", {})
    lo, hi = bym.get("low_margin", {}), bym.get("high_margin", {})
    lo_z, hi_z = lo.get("mcnemar_z", float("nan")), hi.get("mcnemar_z", float("nan"))
    split_effect = bool(not math.isnan(lo_z) and not math.isnan(hi_z)
                        and lo_z > 1.645 and hi_z < -1.645)
    overall = ("NO_TASK" if not (g_task and g_arc and g_exact)
               else "DERIVATION_OK" if (g_floor and g_beats_counts and g_beats_retrieval
                                        and g_ladder)
               else "UNDERPOWERED_VS_RETRIEVAL" if (g_floor and g_beats_counts
                                                    and not g_beats_retrieval and underpowered)
               else "PHI_HELPS_WHERE_SIMILARITY_RUNS_OUT" if (g_floor and g_beats_counts
                                                              and not g_beats_retrieval
                                                              and split_effect)
               else "PHI_ADDS_NOTHING_ON_LOOKUP" if (g_floor and g_beats_counts
                                                     and not g_beats_retrieval)
               else "DERIVATION_PARTIAL" if (g_floor or g_ladder)
               else "DERIVATION_NO")

    out = {
        "stage": "289", "overall": overall, "seed": SEED, "smoke": args.smoke,
        "holdout": args.holdout, "run_tag": args.run_tag,
        "train_steps": n_steps, "params": n_params,
        # provenance readable from the file alone: --run-tag is a human choice and a run can be
        # mislabelled, but this says which objective actually produced the numbers. An older
        # report has no such field, so absence identifies the pre-ladder arm.
        "objective": ("expected_reward_280" if OBJECTIVE == "reward" else
                      "plackett_luce_ladder" if LADDER_ON else "cross_entropy_no_ladder"),
        "edge_channels": sorted(EDGES_ON), "import_k": IMPORT_K,
        "views": VIEWS,
        "reconciliation": recon,
        "neighbours": NEIGHBOURS, "open_verb": OPEN,
        "open_near_source": ({"same_anchor": OPEN_NEAR[0], "neighbourhood": OPEN_NEAR[1]}
                             if OPEN else None),
        "graph_rows": {"mean": GRAPH_N[0] / max(1, GRAPH_N[2]), "max": GRAPH_N[1],
                       "graphs": GRAPH_N[2]},
        "neighbourhood_audit": (neighbourhood_audit(pack, NEIGHBOURS) if NEIGHBOURS else None),
        "nb_channels": ({"anchor_nonzero_rate": NB_NNZ[0] / max(1, NB_NNZ[2]),
                         "rel_nonzero_rate": NB_NNZ[1] / max(1, NB_NNZ[2]),
                         "pairs": NB_NNZ[2]} if NEIGHBOURS else None),
        "refuse": refuse_block,
        "ink": INK, "fp": FP, "words": WORDS,
        "write_ink": args.write_ink, "write_fp": args.write_fp, "write_words": args.write_words,
        "fp_ngram": args.fp_ngram if FP == "hash" else None,
        # the threshold that wrote this tape, and how it was arrived at. A write-ink arm is only
        # readable next to this block: if the achieved density missed the target, the tapes are
        # not comparable and the fragmentation numbers below say nothing about the ink.
        "tau": {
            "mode": args.tau_mode,
            "value": (args.address_tau if args.tau_mode == "absolute"
                      else tau_rule.memo.get("tau")),
            "target_density": (args.tau_target_density if args.tau_mode == "density" else None),
            "achieved_density": (tau_rule.memo.get("achieved")
                                 if args.tau_mode == "density" else None),
            "monotone": (tau_rule.memo.get("monotone") if args.tau_mode == "density" else None),
            "trace": (tau_rule.memo.get("trace") if args.tau_mode == "density" else None),
        },
        "tape_shape": {"held_out": tape_shape(held, held_qs), "train": tape_shape(pack, qs)},
        "resample": {
            "tape_period": args.tape_period,
            "mean_overlap": (sum(RESAMPLE_OVERLAP) / len(RESAMPLE_OVERLAP))
            if RESAMPLE_OVERLAP else float("nan"),
            "n_resamples": len(RESAMPLE_OVERLAP),
            "note": ("Jaccard between consecutive tapes' address sets. Near 1 means the redraw "
                     "returns the same addresses and the anti-memorisation argument in HANDOFF "
                     "1 is decorative - the fix is a larger address pool, i.e. more corpus, not "
                     "fewer parameters"),
        },
        "row_dropout": {
            "rate": args.row_dropout,
            "mean_kept_fraction": (DROPPED[0] / DROPPED[1]) if DROPPED[1] else float("nan"),
            "note": ("training only - the held-out tape is never thinned. Marginalisation, not "
                     "noise: a subset of the evidence is a world the corpus could have written, "
                     "and the low/high margin split is a density axis the mind was never "
                     "trained across"),
        },
        "early_stop": {"enabled": not args.no_early_stop, "best_step": best["step"],
                       "best_probe_loss": best["loss"], "total_steps": n_steps,
                       "probe_questions": len(probe_qs)},
        "probe_curve": probe_curve,
        # two diagnostics, read by nobody: how often the rare channel is nonzero at all (if it
        # is never, the ablation said nothing about word overlap) and how often the bigram bind
        # cancels to a vector too short to normalise (if it often does, the binding is
        # destroying context rather than orienting it).
        "rare_nonzero_rate": (RARE_NNZ[0] / RARE_NNZ[1]) if RARE_NNZ[1] else float("nan"),
        "ink_degenerate_rate": ((INK_DEGENERATE[0] / INK_DEGENERATE[1])
                                if INK_DEGENERATE[1] else float("nan")),
        # mean and standard deviation of the RAW cosines, before rank_norm. A spread near zero
        # says every context looks alike and the mind is ranking noise, whatever the mean is.
        "cos_mean": (COS_SPREAD[0] / COS_SPREAD[2]) if COS_SPREAD[2] else float("nan"),
        "cos_std": (math.sqrt(max(0.0, COS_SPREAD[1] / COS_SPREAD[2]
                                  - (COS_SPREAD[0] / COS_SPREAD[2]) ** 2))
                    if COS_SPREAD[2] else float("nan")),
        "ladder_coverage_train": {"with_ladder": n_lad,
                                  "lookup_questions": len(bv.get("lookup", ()))},
        "count_labels": list(COUNT_LABELS), "compare_labels": list(COMPARE_LABELS),
        "gates": {
            # vacuous when fp=hash: there are no weights to move. G_ink_verified is the gate
            # that carries the weight in that arm.
            "G_arc_enc_frozen": g_arc,
            "G_ink_verified": g_ink,
            "G_task_exists": g_task,
            "G_exact_algebra_matches_tape": g_exact,
            "G_lookup_beats_floor": g_floor,
            "G_lookup_beats_counts_paired": g_beats_counts,
            "G_lookup_beats_retrieval_paired": g_beats_retrieval,
            "G_phi_orders_negatives": g_ladder,
        },
        "held_out": ex, "train_control": ctrl,
        "exact_note": ("count and compare left the weights: they are functions of the "
                       "same-value relation alone (new_i = 1 - max_{j<i} s_ij; count = sum "
                       "new_i; compare = sign of the side difference), computed exactly with "
                       "zero parameters and no 5+ cap, because the invariant says whatever "
                       "decides may not be approximate. Their accuracy is 1.0 by construction "
                       "and is checked, not celebrated - G_exact_algebra_matches_tape is a "
                       "sanity bolt. The interference that cost count 0.965 -> 0.903 is "
                       "removed by construction: one trained task remains"),
        "ladder_note": ("three wrong answers per question at increasing structural distance - "
                        "same anchor / adjacent in tape order / anywhere on the tape - every "
                        "rung a value the corpus wrote, no similarity chosen by anyone. Phi "
                        "trained only against local wrong candidates learns a BOUNDARY; "
                        "generation needs a LANDSCAPE, and a mind that cannot rank its own "
                        "wrong answers by how wrong they are has no direction to move in. The "
                        "objective is one Plackett-Luce term, not a task loss plus a ladder "
                        "loss with a weight between them, and it reduces to the previous "
                        "cross-entropy exactly when the tape cannot supply a ladder"),
        "retrieval_note": ("two rivals now, because they answer two different questions. The "
                           "counting rival knows nothing about context, so beating it shows "
                           "only that the context channel carries information counts lack - "
                           "not that reading it takes a mind. rival_cos is 1-NN over the same "
                           "evidence rows by the same ctx_fp cosine, zero parameters, no "
                           "training. With hash ink the representation IS Random Indexing over "
                           "fastText-shaped word vectors, so the distance between this "
                           "architecture and a classical retrieval system is exactly this one "
                           "number. If rival_cos lands where Phi lands, 3489 parameters are "
                           "decoration ON THIS VERB and the verdict is "
                           "PHI_ADDS_NOTHING_ON_LOOKUP. Named for the brick and not for the wall: lookup is one verb, single-hop and retrieval-shaped by construction, and a rival that ties it says nothing about the exact algebra, about verbs where rows must be combined, or about generation, which 1-NN cannot do at all"),
        "paired_note": ("the rival answers the same lookup questions in the same run, so the "
                        "gate is McNemar over the discordant items at the usual one-sided "
                        "1.645 - never two marginals. The rival over survivors is "
                        "Bayes-optimal when the query context carries nothing, so a paired "
                        "win IS the claim that the context channel carries information counts "
                        "do not have"),
        "curve": curve, "arc_enc_hash_before": arc0, "arc_enc_hash_after": arc1,
        "fp_version": s271.fp_version(),
        "note": (
            "The derivation moved into exact algebra and the mind kept only the judgment. Two "
            "runs measured 7.9k parameters approximating a quantity exactly computable from "
            "their own input, and the approximation degraded as the genuinely uncertain task "
            "grew beside it. Now count and compare are arithmetic over the same-value relation "
            "- exact, uncapped, scale-free - and the one trained surface is Phi, the coherence "
            "of a completed world: for each candidate the query row is filled in and the world "
            "that results is pooled to one scalar, 288's repair loop turned inward. The two "
            "trained surfaces this leaves in the whole architecture are Phi and, once values "
            "stop being exact strings, s_ij itself - both judgments, never arithmetic. "
            "Confidence for exact verbs reports 1.0, which is the honest statement that a "
            "computed answer is certain GIVEN the relation; when s_ij becomes a judgment its "
            "uncertainty enters through that same seam."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"stage289_decision{tag}.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")
    log(json.dumps({"overall": overall, "gates": out["gates"],
                    "lookup": {k: v for k, v in lk.items()},
                    "paired": pv, "ladder": ld}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
