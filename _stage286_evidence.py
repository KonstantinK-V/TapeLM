"""
Stage 286 — The tape is its own exam: judgment moves from code into weights.

Everything before this point had the shape 285's smoke finally made undeniable. Every judgment
was written by hand - votes, return paths, tie rules - and the weights only learned to copy
whichever hand-written judge was pointed at them. Adding a second judge produced arbitration of
plumbing, not understanding; where the judges disagreed, one of them was simply always wrong,
and "learn whom to trust" collapsed into "trust the better heuristic".

So this stage inverts the direction of teaching. Nobody demonstrates anything. The training
signal is the tape itself:

    L = -log P( value(s) | evidence of the address, with s hidden )

Hide one mention of an address, show the mind the others, require the hidden value. The label
is the tape, so there are as many examples as there are slots - not a quota of questions - and
no judge is imitated because no judge is asked. What the weights must learn is the thing the
judges were hand-written approximations of: how to weigh evidence. Votes and corroboration are
demoted to FEATURES on the candidates; if arbitration between them exists, it emerges inside
P and is measured, not coded.

Abstention stops being a habit and becomes an output. UNKNOWN is a candidate like any other,
and the target IS unknown whenever the hidden value does not appear in the visible evidence -
so the mind is trained, not told, that what the tape does not hold cannot be answered. No
threshold constant anywhere: the answer is the argmax, in 284's sense.

The separation invariant holds by construction. The mind predicts a value OF THE TAPE given
evidence FROM THE TAPE, the tape is resampled during training, and the same weights are scored
on a tape built from a disjoint corpus - so hiding facts in the parameters has no payoff, and
the leak test stays available: delete the evidence and the answer must die with it.

Three examiners, all relative, none tunable:
  G_learns_evidence   beat BOTH hand-written judges on the same leave-one-out task
  G_abstains_unknowable  say UNKNOWN more when the answer is absent than when it is present
  G_survives_lie      keep beating votes when one visible mention is forged - the lying tape

------------------------------------------------------------------------------------------
WHAT THE FIRST FULL --no-hidden RUN MEASURED, AND WHAT IT SAID ABOUT THE MEASUREMENT

  EVIDENCE_NO. LOO n=70: model 0.100, votes 0.014, return 0.754. unknown_when_absent 0.000
  against unknown_when_present 0.218. Exam: model 0.777, votes 0.932.

Four of those numbers were reporting defects in the examiners rather than in the mind. Each is
a member of a family HANDOFF.md section 5 already names.

1. THE EXAM WAS A TAUTOLOGY (vacuous gate). 280 defines item["truth"] as the majority value of
   the address, None on a tie. With every mention visible, votes_judge computes exactly that
   function of exactly that input, so on the exam votes IS the label. Its 0.9324 is not a score,
   it is the maximum attainable: 21 clean + 6 decidable at 1.0 and 10 ties at the abstain reward
   is 34.5/37 = 0.9324 to the digit. G_exam_matches_judges asked the mind to tie an oracle. The
   gate is gone; the exam now reports oracle_reward and the gap to it, and nothing gates on it.

2. REWARD CANNOT TELL SILENCE FROM SELECTIVITY (check the denominator). With abstain +0.75,
   wrong -1.0 and correct +1.0, answering pays more than silence only above 87.5% accuracy.
   Every scorer therefore piles up at 0.75 and the ordering between them is decided by the last
   few points of accuracy on a handful of answered items. return_reward 0.7536 on n=70 admits
   only the solutions a in {1, 9, 17, ...} answered at accuracy ~0.88 - anywhere from one item
   to a third of them, indistinguishable in the number that was gating. So coverage and accuracy
   are now reported separately for every scorer, the value of unconditional silence is printed
   beside them, and a judge that answers less than a tenth of the items is marked vacuous
   instead of being beaten or lost to.

3. THE MIND WAS COMPARED TO THE JUDGES AT DIFFERENT COVERAGE. The model answered 58 of 70 at
   0.483; the return judge answered few at ~0.88. That comparison has no direction. The
   examiner is now selective: rank the items by the mind's own confidence, take the same
   FRACTION the judge chose to answer, and compare accuracy there. Relative, and still no
   constant to tune.

4. UNKNOWN AT ARGMAX FIGHTS THE BASE RATE, and in the regime that dominated training it was
   not predictable at all. Whether the hidden value appears among the visible ones is a property
   of the HIDDEN slot. Where the evidence is unanimous - 6 of the 8 logged training steps had a
   single candidate - the feature row for UNKNOWN is [1/n, 1.0, 1/n, 1.0] whether the target is
   present or absent: literally the same input under both labels, so the 21% absent rate is
   irreducible noise there and "never say UNKNOWN" is the optimal response to it. 0.000 was the
   correct behaviour of an argmax, not a dishonest mind. Honesty is therefore measured where it
   lives, in the distribution: P(UNKNOWN) must rank absent targets above present ones (AUC),
   and the argmax rates are kept as diagnostics.

Two defects in the head itself, both of which the ablation exposes by failing:

5. THE UNKNOWN ROW COULD NOT SEE THE CORROBORATION. Its four features were [1/n, margin,
   n_distinct/n, flag] while a candidate's third feature was its external support. The row that
   decides whether to answer at all was blind to the one signal that says whether any candidate
   deserves an answer - so the head could not express the return judge's rule even with the
   return judge's feature sitting in its input, which is the whole of the 0.100 against 0.754.
   Every row now carries the same schema: its own four numbers plus five example-level numbers
   broadcast to all rows, UNKNOWN included.

6. THE TARGET WAS MISLABELLED WHERE THE CANDIDATE LIST TRUNCATED. cands is most_common(8); a
   target present in the evidence but ranked ninth was labelled UNKNOWN. Rare, but it is label
   noise pointing the wrong way on exactly the axis gate 3 measures. Presence is now decided
   against the full counter, truncated examples are dropped from training and counted.

------------------------------------------------------------------------------------------
WHAT THE FIRST HIDDEN RUN UNDER THE NEW EXAMINERS MEASURED (2026-08-08, wall 8607s)

  EVIDENCE_NO again, but this time the numbers are about the mind, and they are not noise:
  unknown_auc 0.259 - 2.8 sigma BELOW chance on 14x56 pairs - and a selective curve that
  RISES with coverage (0.286 at 10% -> 0.443 at 100%). Confidence anti-correlated with being
  right is a sign flip, not an absence of signal.

  The mechanism is a sampling mismatch this file created itself. Training drew an ITEM
  uniformly and then a slot, so it saw 1.95 candidates on average; leave-one-out enumerates
  every slot of every item, weighting an address by its mention count, and saw 3.56. And where
  UNKNOWN sits in the composition flips between those regimes - at ~2 mentions absent targets
  live in unanimous evidence, at ~4 in contested evidence. The head learned the training
  tape's statistic TRUTHFULLY (P(UNKNOWN|unanimous) pins at 0.52-0.63 from the training
  counts, and the exam's clean abstain of 0.571 is that number read back) and it
  anti-generalised to the distribution it was scored on. Training now samples (item, slot)
  pairs uniformly - the examiner's own distribution - and a leave-one-out control on the last
  training tape localises whatever remains: AUC above 0.5 there and below held-out is shift
  between the corpus halves; below 0.5 on both is a target that is not a function of the
  features; nohid clean while hidden stays inverted is the trunk memorising subject->value
  across resamples, the one leak the invariant forbids.

------------------------------------------------------------------------------------------
THE ABLATION PAIR, SAME CODE, SAME TAPES - AND A PREDICTION THAT FAILED

  The nohid arm had in fact run two hours BEFORE the hidden arm on the same second-generation
  code (judge numbers byte-identical between the two JSONs, so the tapes and examples were
  too). That makes the pair a controlled ablation, and it falsifies half of the paragraph
  above. The prediction was that the composition flip, living entirely in the count features,
  had to show in the features-only arm as well. It did not:

                          nohid (features only)      hidden (features + reading)
    LOO accuracy/coverage      0.641 @ 0.557              0.325 @ 0.571
    selective curve         1.00 -> 0.43 FALLING       0.29 -> 0.44 RISING
    unknown_auc             0.412  (1.0 sigma, noise)  0.259  (2.8 sigma, inverted)
    matched vs votes        0.431 vs 0.431 (tie)       0.379 vs 0.431
    matched vs return       1.000 vs 0.889 (n=9)       0.222 vs 0.889
    lying tape              CLEAN: 0.487 vs 0.462      fails
    exam gap to oracle      0.108                      0.345
    verdict                 EVIDENCE_PARTIAL           EVIDENCE_NO

  So the composition mismatch is real but survivable: nine numbers trained on the wrong mix
  still rank correctly on the right one. What is NOT survivable is the reading: on identical
  tapes it halved accuracy at equal coverage and inverted the confidence ordering. The
  features-only ablation BEATS the arm carrying the stage's whole claim. Failing gates:
  honesty at chance (the sampler fix above is aimed exactly there), and reward under 0.75 -
  which 280's baseline (0.704) never cleared either.

  Two hypotheses fit "confidently wrong on held-out, only when reading":
    a) the trunk fine-tunes at lr_upper over 6000 steps on resamples of the same 25k lines
       and memorises subject->value; held-out subjects then misfire with confidence - facts
       moving into weights, the exact leak the separation invariant forbids;
    b) the reading channel (d_hidden dims against 9 feature dims) drowns the features and the
       head overfits reading noise - generic overfit, no memorisation needed.
  --frozen-trunk separates them: it keeps the reading but stops the trunk from learning, so
  (a) predicts frozen-hidden recovers to nohid while (b) predicts it stays broken. The
  head_input_norms block in the decision JSON reports how much of the head's first layer
  serves each channel, which says how much room (b) even has.

------------------------------------------------------------------------------------------
THE FIRST PASS - EVIDENCE_OK ON NOHID - AND WHAT IT IS ACTUALLY A PASS OF

  nohid, pair-uniform sampler: model 0.84 accuracy at 0.357 coverage against votes 0.431 at
  0.829; beats votes AND return at their own coverage; on the lying tape it answers 15% of
  items and is never wrong. Those are the first non-vacuous passes this stage has produced.

  Two predictions made before the run failed, and the second one matters. Training candidates
  were supposed to reach ~3.5 and the unknown rate ~0.2; they moved 1.95 -> 2.07 and
  0.348 -> 0.377. The sampler was not the gap. The CONTROL says where the gap is: leave-one-out
  on the training tape itself sees 2.15 candidates, not 3.56. The two corpus halves build
  differently shaped addresses - the training tape has 104 addresses of which only 22 can
  spare a mention - and that shift, not the sampler's weighting, is most of the distance.
  The sampler fix was still worth making (it is the trainer agreeing with the examiner about
  what an example is), it simply was not the thing it was sold as.

  What the control also shows is where honesty lives. unknown_auc is 0.809 on the training
  tape (4.73 sigma) and 0.560 held out (0.69 sigma - chance). G_abstains_unknowable passed on
  a coin. Hence auc_z and a gate that asks the effect to clear its own noise.

  And the shape of what was learned. On the exam this arm abstained on 100% of decidable and
  100% of tie while answering 100% of clean, perfectly: that policy is "answer iff the address
  is unanimous", one threshold on one count feature. On the training tape the coverage 0.4167
  sits next to a unanimous rate of 0.4325. If the same holds held-out then EVIDENCE_OK is a
  copying rule that learned when to keep quiet - better than votes because it is SELECTIVE,
  which is real and is learned, but not the weighing this stage set out to train.
  by_evidence_shape now splits every rate by unanimous against contested, and
  G_weighs_contested runs the matched-coverage comparison on the contested subset alone, where
  copying cannot answer. Passing everything but that gate is its own verdict now:
  EVIDENCE_OK_UNANIMOUS_ONLY.

------------------------------------------------------------------------------------------
THE FROZEN ARM ANSWERS IT: THE TRUNK IS INNOCENT, THE CAPACITY IS NOT

  --frozen-trunk, hidden, same tapes as the nohid pass. EVIDENCE_NO, and worse than the
  unfrozen arm on every held-out number. So hypothesis (a) is dead: the trunk did not learn a
  single parameter here and the collapse happened anyway. (b) is what is left, and the two
  leave-one-out passes say it in one line:

                       train control (last training tape)   held out
    unknown AUC             0.986   (+7.44 sigma)            0.416  (-0.97 sigma)
    accuracy @ coverage     0.897 @ 0.690                    0.377 @ 0.757
    selective curve         1.00 / 1.00 / 1.00 at 10-50%     flat 0.43 / 0.44 / 0.49
    vs votes matched        0.897 vs 0.690                   0.414 vs 0.431

  A head that fits its training tape at AUC 0.986 and lands a hair below chance on a disjoint
  one, with a completely flat confidence curve, is not misreading anything - it memorised.
  The arithmetic was there to be done beforehand: the reading path carries 512x128 = 65,536
  weights against roughly a thousand distinct training examples (84 leave-one-out pairs per
  tape, twelve tapes at tape_period 500), which is 65 parameters per example. The
  features-only head has 1,152 weights, or 1.14 per example, and it is the arm that passed.
  head_input_norms measured the same thing from the other side: the reading columns carry 3.1x
  the total norm of the nine features.

  This does NOT say reading is useless; it says the question was never asked at a capacity the
  data can pay for. Two changes make asking it cheap. --read-rank R projects the reading down
  before the head, so the channel can be given 8 or 16 dimensions instead of 512. And a frozen
  trunk returns the same vector for the same text forever, so the encodes are now cached - the
  frozen arm was recomputing thousands of identical trunk calls, which is most of its hour.

  Note what this leaves standing. G_arc_enc_frozen was true in every arm and the answer is
  still a value taken from the visible evidence, so no fact reached the weights in the sense
  the invariant forbids. What reached them is a tape-specific way of reading - which the
  invariant does not name, and which the held-out tape catches anyway.

------------------------------------------------------------------------------------------
RANK 8 KILLS THE CAPACITY STORY TOO: THE READING IS A LOOKUP KEY

  --frozen-trunk --read-rank 8. The reading path shrank 12.8x (65,536 -> 5,120 trainable
  parameters, projection included - the first head_norms count missed the projection and is
  fixed now) and the train fit DID NOT MOVE: accuracy 0.897 @ 0.690 identical to the digit,
  control AUC 0.986 -> 0.996. Held out: accuracy 0.353, unknown AUC 0.302 (-2.3 sigma,
  inverted again), lying tape 0.143 at 0.689 coverage. A 12.8x capacity cut that changes
  nothing about the fit and nothing about the collapse means capacity was never the binding
  constraint. The reading is being used as a lookup KEY: a text with its names in it is a
  unique identifier, and a lookup table from unique identifiers is cheap at any width.

  So (a) trunk and (b) capacity are both dead, and what remains is (c): identity in the input.
  --read-anon folds the subject and every candidate value in the READ text to constant words,
  leaving only the phrasing - the one part of a reading that could transfer across disjoint
  corpora. The whole sentence is still a unique string, so the train control may memorise
  regardless; the question anon answers is HELD OUT only: does phrasing carry signal? Beats
  nohid there - it does. Ties nohid - reading adds nothing at this tape size and nohid IS the
  model. Still inverted - the channel is dropped, and the stage's verdict is that judgment
  lives in the counters.

  The gate this run exposed: G_weighs_contested passed at 0.152 against votes' 0.121 while
  every other gate failed - two scorers under the floor, ranked solemnly. Random choice among
  the candidates is the null that comparison forgot; by_evidence_shape now carries
  random_accuracy_contested and the gate requires beating it.

------------------------------------------------------------------------------------------
ANON RESOLVES THE PRE-REGISTERED FORK; THE WALL GETS A NAME; --mind relational IS THE ANSWER

  --frozen-trunk --read-rank 8 --read-anon. The inversion is gone (unknown AUC 0.302 at
  -2.3 sigma -> 0.579 at +0.9), which confirms identity was the poison. But the channel does
  not pay: parity with nohid on leave-one-out (matched vs votes 0.431 = 0.431, nohid ties the
  same number), a LOSS on the lying tape (0.410 vs votes 0.462 where nohid won 0.487), and
  the train control still memorised at AUC 0.997 because an anonymised sentence is still a
  unique string. One genuine positive: the first legitimate G_weighs_contested pass -
  contested 0.364 against random 0.268 against votes 0.121 - a hint that STRUCTURE, once the
  names are gone, carries something.

  The fork declared before the run therefore resolves to: the raw-reading channel is dropped.
  And the wall has a name now. Everything trainable over an input that contains identity
  becomes a lookup table - trunk frozen: collapse stays; capacity cut 12.8x: fit unchanged;
  names folded out: poison gone, payoff absent. The only head that ever passed an exam here
  is nine relational statistics, and they transfer precisely because they cannot name anyone.

  --mind relational takes that as the design rule instead of a post-mortem: an equivariant
  network over the mention graph whose every input is an indicator or a within-example RANK
  (same-value; rank of ctx-fp cosine; rank of shared-rare-word overlap, rare = rarer than the
  tape's median; rank of external corroboration; value's share of the evidence). No text, no
  fingerprint, no absolute coordinate reaches the weights - the lookup channel is closed by
  construction, not by regularisation. Equivariance is the "mind does not grow with the
  knowledge" invariant as architecture: the same ~5k weights read 3 mentions or 300. What it
  can express that counters cannot: two mentions with near-identical contexts are ONE witness
  repeated - and --lie-dup K is the exam for that, because votes scale linearly in the copies
  by construction. Pre-registered falsifiers: the train-control AUC must FALL from ~1.0 (if it
  does not, the graph itself leaks identity and the class failed its reason to exist); matched
  coverage must reach nohid's (its features are derivable from the graph); the win, if any,
  must show on contested or on the duplicated lie.

  python _stage286_evidence.py --train-steps 6000 --min-mentions 2 --mind relational --lie-dup 3

------------------------------------------------------------------------------------------
THE RELATIONAL MIND: THE FIRST THING A COUNTER CANNOT DO

  --mind relational --lie-dup 3, 4,417 parameters, no text and no absolute coordinate reaching
  the weights. EVIDENCE_PARTIAL, and the three falsifiers stated before the run split cleanly.

  PASSED. The train-control AUC fell, 0.997 -> 0.841, so the graph is not the lookup table the
  text was. Matched coverage reached nohid exactly: 0.431 against votes' 0.431, 1.000 against
  return's 0.889 - the nine counters really are derivable from the graph, as claimed.

  PASSED, and this is the result. On the duplicated lying tape - the forged mention repeated
  three times - votes scores accuracy 0.000 on 40 answered, reward -0.398. Not "worse": zero,
  by construction, because a counter cannot tell three copies of one witness from three
  witnesses. The relational mind scores 0.773 on 22 answered, reward 0.676, and 0.450 against
  votes' 0.000 at votes' own coverage. That is not a better heuristic, it is a capability the
  counting family does not have. It also confirms the mechanism rather than a lucky number:
  duplicates arrive as pairs whose context-cosine rank and shared-rare-word rank both sit at
  the top, and the model answers 0.36 of the duplicated-lie sets while answering only 0.04 of
  ordinary contested ones - it acts exactly where the graph says "one witness, repeated".

  FAILED. G_weighs_contested: two contested answers held out, accuracy 0.0. On ordinary
  contested evidence this mind abstains almost totally. Worth noting before calling that a
  defect: on that subset votes scores 0.121 against a random-pick floor of 0.268, so majority
  reasoning there is WORSE than chance and silence is the calibrated response - but two items
  is no evidence of weighing either, and the gate says so.

  And the honesty gate was measuring through a confound of its own. P(UNKNOWN) shares one
  softmax with however many candidate rows an example has, so it is mechanically diluted by
  |cands| - while absent targets live in contested evidence, which has MORE candidates. The
  measure was biased against its own hypothesis, and this run made it visible: at the argmax
  UNKNOWN won 0.714 of absent against 0.589 of present, the right ordering, while the mean
  probabilities were 0.478 and 0.484 and the AUC read 0.482. The logit MARGIN over the best
  candidate is what the argmax actually compares and it does not care how many other rows
  exist; that is what the gate reads now, with the probability form kept for comparability.

  python _stage286_evidence.py --smoke
  python _stage286_evidence.py --train-steps 6000 --min-mentions 2
  python _stage286_evidence.py --train-steps 6000 --min-mentions 2 --no-hidden
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
import random
import re
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
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage279_write_decision as s279
from _tape_speed import CachedBank, install_all
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _inprint_glue import hidden_and_logits
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 286
N_OWN = 4             # what a row says about ITS candidate
N_GLOBAL = 5          # what the evidence set says, broadcast to every row including UNKNOWN
N_FEAT = N_OWN + N_GLOBAL
MIN_EVIDENCE = 2      # mentions that must remain visible after one is hidden
# The last arbitrary constant in the examiners was a coverage floor of 0.10, chosen because it
# looked small. Replaced by the smallest answered count on which a one-sided sign test can
# reach 0.05 at all: 0.5^5 = 0.031 is the first below it, 0.5^4 = 0.063 is not. So a judge is
# vacuous when no comparison against it could be significant even if it lost every item -
# derived from the null, like the 1.645 the honesty gate uses, rather than picked.
MIN_ANSWERED = next(n for n in range(1, 64) if 0.5 ** n < 0.05)   # == 5
LOG_PATH = RES / "_stage286_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


# --------------------------------------------------------------------------- evidence -> P

def ext_support(pack, S: str, value: str, own: set[int]) -> int:
    """Mentions OUTSIDE the address that carry subject and value together.

    Outside, because inside is what the model already sees - and because the hidden slot is a
    member, counting members would hand the target back through the index. This is 285's return
    judge reduced to a number the model may weigh however the evidence warrants.
    """
    words = context_words(value) or [value]
    scan = min((pack["postings_probe"].get(w, ()) for w in words), key=len, default=())
    v = value.lower()
    return sum(1 for c in scan
               if c not in own and S in pack["texts_lc"][c] and v in pack["texts_lc"][c])


def example(pack, item, hidden: int | None, lie_from: int | None = None, lie_dup: int = 1):
    """One evidence set. hidden=None is exam mode: all mentions visible, predict the address.

    The candidate list is what the EVIDENCE says plus UNKNOWN, so the model can only answer
    with a value the tape holds, and must answer UNKNOWN when the tape (as visible) does not
    hold the answer - which is the honesty this stage trains instead of assumes.
    """
    vals, texts = pack["tape"].values, pack["texts"]
    ev = [c for c in item["slots"] if c != hidden]
    # Leaving ONE mention visible is not a weighing task, it is a copying task: the candidate
    # list has a single entry and the only choice left is that value or UNKNOWN. The first smoke
    # ran entirely in that regime - n_cands was 1 at every step - so nothing about evidence
    # could be learned or measured. An address must be able to spare a mention and still hold a
    # comparison.
    if len(ev) < (MIN_EVIDENCE if hidden is not None else 1):
        return None
    ev_vals = [vals[c] for c in ev]
    ev_texts = [texts[c] for c in ev]
    ev_slots = list(ev)
    if lie_from is not None:
        # The lying tape forges ONE visible mention. With a single mention visible that is not a
        # lie to catch, it is the whole record replaced, and the first smoke scored exactly that
        # - model and votes both at -1.0, because no evidence survived to contradict it.
        if len(ev) < 2:
            return None
        j = len(ev) - 1
        # lie_dup > 1 repeats the SAME forged mention. Votes grow linearly in the copies -
        # a counter cannot tell three repetitions from three witnesses by construction. A mind
        # that models duplication (identical value, near-identical context) should not budge.
        ev_vals = ev_vals[:j] + [vals[lie_from]] * lie_dup
        ev_texts = ev_texts[:j] + [texts[lie_from]] * lie_dup
        ev_slots = ev_slots[:j] + [lie_from] * lie_dup
    cnt = Counter(ev_vals)
    cands = [v for v, _ in cnt.most_common(8)]
    target_val = vals[hidden] if hidden is not None else item["truth"]
    # Presence is a fact about the EVIDENCE, not about the top-8 window over it. Deciding it
    # from `cands` labelled a target that ranked ninth as UNKNOWN, which is label noise on the
    # exact axis G_abstains_unknowable measures. Truncation is now visible and counted instead.
    present = target_val is not None and target_val in cnt
    truncated = present and target_val not in cands
    target = cands.index(target_val) if (present and not truncated) else len(cands)
    own = set(item["slots"])
    n = len(ev_vals)
    top2 = cnt.most_common(2)
    lead = top2[0][1]
    second = top2[1][1] if len(top2) > 1 else 0
    ext = {v: ext_support(pack, item["S"], v, own) for v in cands}
    ext_rank = sorted(ext.values(), reverse=True)
    ext1 = ext_rank[0] if ext_rank else 0
    ext2 = ext_rank[1] if len(ext_rank) > 1 else 0
    # Every row reads the same schema. The first four numbers are the row's own candidate; the
    # last five are the evidence set, broadcast. Before this the UNKNOWN row carried a different
    # third column from the candidate rows and no view of external support at all, so the row
    # that decides whether to answer was blind to the only signal saying whether any candidate
    # deserved an answer - the head could not express the return judge's rule while holding the
    # return judge's feature.
    glob = [len(cnt) / n, (lead - second) / n, 1.0 / n,
            min(ext1, 3) / 3.0, min(ext1 - ext2, 3) / 3.0]
    rows = []
    for v in cands:
        margin = (cnt[v] - max((c for w, c in cnt.items() if w != v), default=0)) / n
        rows.append([cnt[v] / n, margin, min(ext[v], 3) / 3.0, 0.0] + glob)
    rows.append([0.0, 0.0, 0.0, 1.0] + glob)                       # the UNKNOWN row
    # The mentions each candidate is made of, cued with the subject, so a reading can be ABOUT
    # a candidate. UNKNOWN reads everything, which is the right thing for it to be about.
    cue = s271.CUE.format(S=item.get("query") or item["S"])
    per = [[cue] + [t for t, v2 in zip(ev_texts, ev_vals) if v2 == v] for v in cands]
    per.append([cue] + ev_texts)
    return {"texts": ev_texts, "vals": ev_vals, "cands": cands, "rows": rows,
            "cand_texts": per, "target": target, "cnt": cnt, "ext": ext, "ev_slots": ev_slots,
            "present": bool(present), "truncated": bool(truncated)}


class Evidence(nn.Module):
    """Scores each candidate (and UNKNOWN) from ITS OWN mentions, read by the trunk.

    The first version pooled every mention into one vector and handed the same vector to every
    candidate row. That reading cannot say which candidate the text supports - it is a constant
    across the softmax and only modulates how the counts are weighed - so the head would have
    been votes and corroboration wearing a hat, and --no-hidden would have tied on purpose.
    Each candidate now gets the mentions that actually carry it, cued with the subject, so the
    weights compare readings and not only numbers. That is the whole claim of this stage.
    """

    def __init__(self, d_hidden: int, device, read_rank: int = 0):
        super().__init__()
        # The reading channel arrives 512 dimensions wide against nine features, and the frozen
        # run measured what that costs: 512x128 = 65,536 weights on the reading path against
        # roughly a thousand distinct training examples - 65 parameters per example, where the
        # features-only head has 1.14. It fit the training tape to AUC 0.986 and transferred
        # nothing. read_rank projects the reading down first, so "does reading carry signal"
        # can be asked at a capacity the data can actually pay for. 0 keeps the wide channel.
        self.proj = (nn.Linear(d_hidden, read_rank, bias=False).to(device)
                     if (d_hidden and read_rank) else None)
        d_read = read_rank if self.proj is not None else d_hidden
        self.d_read = d_read
        self.f = nn.Sequential(nn.Linear(d_read + N_FEAT, 128), nn.GELU(),
                               nn.Linear(128, 1)).to(device)
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, H, rows):
        """H is one reading PER ROW, not one reading for the example."""
        if H.numel() and self.proj is not None:
            H = self.proj(H)
        return self.f(torch.cat([H, rows], dim=-1)).squeeze(-1) if H.numel() \
            else self.f(rows).squeeze(-1)


def encode(model, char_table, tok, pad_id, device, texts):
    """Read a set of mentions. The window is the trunk's own - pos is Embedding(MAX_ARCS, d),
    so anything longer indexes past the table and the run dies on the device rather than in
    Python. 271 truncates the same way and for the same reason."""
    ids = [i for i in tok.encode(" | ".join(texts)[-2000:]).ids
           if i != pad_id][-s177.MAX_ARCS:]
    if not ids:
        return None
    h, _ = hidden_and_logits(model, char_table,
                             torch.tensor([ids], dtype=torch.long, device=device), pad_id)
    return h[0, -1]


class Relational(nn.Module):
    """The mind as structure only: ranks and indicators over relations the plumbing computes.

    Three autopsies said the same thing from three sides. The trunk froze and the collapse
    stayed; the reading path shrank 12.8x and the train fit did not move a digit; the names
    were folded out of the text and the poison left but the payoff did not arrive. Everything
    trainable OVER AN INPUT THAT CONTAINS IDENTITY becomes a lookup table, because a text - or
    a context fingerprint, or any absolute coordinate - is a unique key, and keying is cheaper
    than weighing at every width. Meanwhile the one head that passed an exam is nine relational
    statistics: counts and corroboration, which transfer between disjoint corpora precisely
    because they cannot name anyone.

    So identity is made UNREPRESENTABLE instead of discouraged. The input is a graph over the
    visible mentions; every number on it is an indicator or a rank:

        edge (i,j)   same value or not; the RANK of their context-fp cosine among this
                     example's pairs; the RANK of their shared-rare-word overlap (rare = rarer
                     than the tape's median, 284's rule - a rank, not a threshold)
        node i       its value's share of the evidence; the RANK of its external corroboration

    Order statistics carry structure and almost nothing else: a fresh tape re-ranks everything,
    so a memorised rank pattern has no address to return to. The network is equivariant - sum
    and mean pools over mentions and over each candidate's coalition - which is the invariant
    "the mind does not grow with the knowledge" realised in the architecture: the same weights
    read 3 mentions or 300. No threshold constants; the answer is still the argmax; UNKNOWN is
    still a row, pooled from the whole graph with a flag.

    What this can express that the counters cannot: two mentions with near-identical contexts
    are ONE witness repeated, not two. Votes count them twice by construction; --lie-dup is the
    exam for exactly that. And the falsifier is pre-registered: if the train-control AUC stays
    at ~1.0 under this input, the relation graph itself leaks identity and this class failed
    its own reason to exist.
    """

    def __init__(self, device, d: int = 32, n_edge: int = 3, n_node: int = 2):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(n_edge, d), nn.GELU()).to(device)
        self.node = nn.Sequential(nn.Linear(n_node + 2 * d, d), nn.GELU()).to(device)
        self.cand = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(),
                                  nn.Linear(d, 1)).to(device)
        nn.init.zeros_(self.cand[-1].weight)
        nn.init.zeros_(self.cand[-1].bias)

    def forward(self, E, same, node_feats, masks):
        e = self.edge(E)                                            # n,n,d
        own = (e * same).sum(1) / same.sum(1).clamp(min=1.0)        # my witnesses
        h = self.node(torch.cat([node_feats, own, e.mean(1)], -1))  # n,d
        g = h.mean(0)
        z, o = torch.zeros(1, device=g.device), torch.ones(1, device=g.device)
        outs = [self.cand(torch.cat([h[m].mean(0), g, z])) for m in masks]
        outs.append(self.cand(torch.cat([torch.zeros_like(g), g, o])))
        return torch.cat(outs)


# --------------------------------------------------------------------------- the two old judges

def votes_judge(ex):
    top = ex["cnt"].most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def return_judge(pack, item, ex):
    scored = sorted(ex["ext"].items(), key=lambda t: -t[1])
    if not scored:
        return None
    (v1, n1), n2 = scored[0], (scored[1][1] if len(scored) > 1 else 0)
    return None if (n1 < 2 or n1 == n2) else v1


def reward_of(answered, truth, abstain_reward, wrong_cost):
    if answered is None:
        return abstain_reward
    return 1.0 if (truth is not None and answered == truth) else -wrong_cost


# ------------------------------------------------------------------- scoring without constants

def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def scorer_stats(answered, correct, reward):
    """Reward alone hides the difference between silence and selectivity.

    With abstain +0.75 against wrong -1.0, answering beats silence only above 87.5% accuracy, so
    every scorer piles up near the abstain constant and the ordering between them is decided by
    a handful of items. The three numbers that actually differ - how often it spoke, how often
    it was right when it spoke, and what that was worth - are reported separately from here on.
    """
    n = len(answered)
    n_ans = int(sum(answered))
    return {"n": n, "coverage": (n_ans / n) if n else float("nan"),
            "accuracy": (float(sum(correct)) / n_ans) if n_ans else float("nan"),
            "reward": _mean(reward), "n_answered": n_ans,
            "vacuous": bool(n and n_ans < MIN_ANSWERED)}


def acc_at_coverage(conf, ok, coverage):
    """The mind's accuracy on the fraction of items it is most confident about.

    A judge that answers a fifth of the items at high accuracy and a mind that answers all of
    them at middling accuracy cannot be ordered by either number alone. Ranking by the mind's
    own confidence and cutting at the judge's coverage compares them at the same operating
    point, which is the only comparison that has a direction. No threshold is chosen: the
    coverage comes from whatever the judge did.
    """
    n = len(conf)
    if n == 0 or not (0.0 < coverage <= 1.0):
        return float("nan")
    order = sorted(range(n), key=lambda i: -conf[i])
    m = max(1, int(round(coverage * n)))
    return float(np.mean([ok[i] for i in order[:m]]))


def auc(pos, neg):
    """P(score on a positive > score on a negative), ties at half. Scale-free, so it survives a
    model whose probabilities are badly calibrated in absolute terms."""
    if not pos or not neg:
        return float("nan")
    p = np.asarray(pos)[:, None]
    q = np.asarray(neg)[None, :]
    return float(np.mean((p > q) + 0.5 * (p == q)))


def auc_z(a, n1, n2):
    """How many standard errors the AUC sits above chance, under the null.

    An AUC of 0.560 on 14 absent against 56 present is 0.69 sigma - chance, wearing a number
    above 0.5. The first run to clear G_abstains_unknowable cleared it on exactly that, while
    the same measurement on the training tape stood at 4.73 sigma. A gate written as "> 0.5"
    cannot tell those apart, which is failure mode 3 arriving through the one examiner that
    was still absolute. The Hanley-McNeil null standard error is a property of the sample
    sizes, not a tuned constant, so the gate can require the effect to clear its own noise.
    """
    if math.isnan(a) or n1 < 1 or n2 < 1:
        return float("nan")
    return (a - 0.5) / math.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--tape-period", type=int, default=0)
    ap.add_argument("--addresses", type=int, default=0)
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--min-evidence", type=int, default=2,
                    help="mentions left visible after hiding one; under 2 there is nothing "
                         "to weigh")
    ap.add_argument("--min-candidates", type=int, default=1, metavar="C",
                    help="train only on evidence sets holding at least C distinct values. 1 "
                         "keeps the natural mix, in which unanimous evidence dominates and the "
                         "UNKNOWN label is irreducible noise; 2 trains only where there is "
                         "something to weigh, at the cost of never seeing a settled address. "
                         "The rate is reported either way.")
    ap.add_argument("--min-per-family", type=int, default=8,
                    help="ordering only. 280's packer moves this many of each family to the "
                         "front and truncates nowhere unless --max-items is set, which this "
                         "stage does not set - so the family mix here is the corpus's own.")
    ap.add_argument("--address-tau", type=float, default=0.90)
    ap.add_argument("--address-overlap", type=int, default=2)
    ap.add_argument("--soft-match", type=float, default=0.0)
    ap.add_argument("--addr-key", choices=("two", "set", "mean"), default="two")
    ap.add_argument("--abstain-reward", type=float, default=0.75)
    ap.add_argument("--wrong-cost", type=float, default=1.0)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-upper", type=float, default=3e-5)
    ap.add_argument("--no-hidden", action="store_true",
                    help="ablation: features only, no reading of the evidence text")
    ap.add_argument("--frozen-trunk", action="store_true")
    ap.add_argument("--mind", choices=("head", "relational"), default="head",
                    help="head is 286's scorer over features (and optionally the reading). "
                         "relational replaces it with an equivariant network over ranks and "
                         "indicators of the mention graph - no text, no fingerprints, no "
                         "absolute coordinate of any kind reaches the weights, so the lookup "
                         "channel that killed every reading arm is closed by construction, "
                         "not by regularisation. Trains in minutes: the trunk is never read.")
    ap.add_argument("--holdout", choices=("corpus", "address"), default="corpus",
                    help="what the held-out tape holds out. corpus splits the LINES 70/30, so "
                         "the eval tape is built from different text - and every arm so far "
                         "learns honesty and contested weighing on the training tape (margin "
                         "AUC 0.865, contested 0.889) and reads chance across that split. "
                         "That leaves two explanations and no way to tell them apart: the two "
                         "corpus halves build differently shaped addresses, or the mind simply "
                         "does not generalise. address splits the ADDRESSES of ONE corpus by a "
                         "stable hash of the anchor, so the tape geometry is identical on both "
                         "sides and only novelty of the subject remains. Transfer under "
                         "address holdout but not under corpus holdout blames the tape; "
                         "failure under both blames the mind.")
    ap.add_argument("--ctx-channel", choices=("cos", "delta", "both"), default="cos",
                    help="what the context edge carries. cos is the plain ctx-fp cosine. "
                         "delta REPLACES it with the composition residual: D = ctx_fp(whole) - "
                         "norm(ctx_fp(before value) + ctx_fp(after value)), compared between "
                         "mentions by cosine rank, plus |D| as a node rank. Word-level D is "
                         "provably useless here - fp pools characters BEFORE the FF, so both "
                         "f(whole) and mean f(parts) are functions of the character multiset "
                         "and anagrams give bit-identical D. At CONTEXT level the per-chunk "
                         "normalisation breaks that linearity, so D depends on the grouping, "
                         "and it subtracts the bulk mean - which 283 named as the crowder "
                         "(0.345 against 0.061). D is the high-pass part of the context "
                         "channel. 'both' keeps cos as well and is the arm expected to lose: "
                         "ink-slices already showed an EXTRA fp channel costing the dup win "
                         "0.773 -> 0.333 on 4,417 parameters and a thousand examples.")
    ap.add_argument("--ink-slices", action="store_true",
                    help="two more edge channels: the RANK of the context-fp cosine taken "
                         "before the value and after it, separately. fp is order-blind inside "
                         "a word (277, closed) but slicing the CONTEXT at the value recovers "
                         "order BETWEEN chunks, which is a different question and an open one: "
                         "two mentions agreeing on the frame but not the aftermath are a "
                         "paraphrase, agreeing on both are a duplicate. Ranks only - an "
                         "absolute fp coordinate is an identity, and identity is what turned "
                         "every reading arm into a lookup table.")
    ap.add_argument("--lie-dup", type=int, default=1, metavar="K",
                    help="lying tape with the forged mention repeated K times. Votes scale "
                         "linearly in K by construction; a mind that models duplication "
                         "should not. Reported as lying_tape_dup and gated as "
                         "G_survives_duplicated_lie (informational, not in the verdict).")
    ap.add_argument("--read-anon", action="store_true",
                    help="replace the subject and every candidate value in the READ text with "
                         "fixed common words. The r8 arm showed the reading is used as a lookup "
                         "key - identity is what memorises, phrasing is the only part that "
                         "could transfer - so this is the arm that asks whether phrasing alone "
                         "carries signal. Features keep identity either way; that split is the "
                         "point.")
    ap.add_argument("--read-rank", type=int, default=0, metavar="R",
                    help="project the reading to R dimensions before the head. 0 is the wide "
                         "512-dim channel that the frozen arm showed memorising the training "
                         "tape (AUC 0.986 there, 0.416 held out) with the trunk not learning "
                         "at all - so the capacity, not the trunk, is what has to be tested.")
    ap.add_argument("--run-tag", type=str, default="")
    ap.add_argument("--no-speedups", action="store_true",
                    help="run the original unmemoised paths - they are byte-identical, "
                         "and this is how that stays checkable")
    args = ap.parse_args()

    global LOG_PATH
    tag = (args.run_tag and f"_{args.run_tag}") or ""
    tag += "_nohid" if args.no_hidden else ""
    # A smoke and a full run with no --run-tag wrote the same filename, and a smoke
    # overwrote a finished full run's decision.json - recovered by hand from the log, which
    # is not a provenance anyone should have to accept twice.
    tag = "_smoke" + tag if args.smoke else tag
    # The frozen arm has to reach the filename too. Without this a hidden --frozen-trunk run
    # writes stage286_decision.json over the unfrozen hidden run - and those two files ARE the
    # (a)/(b) discriminator, so the comparison would destroy one of its own halves.
    tag += "_frozen" if args.frozen_trunk else ""
    tag += f"_r{args.read_rank}" if args.read_rank else ""
    tag += "_anon" if args.read_anon else ""
    tag += "_rel" if args.mind == "relational" else ""
    tag += f"_dup{args.lie_dup}" if args.lie_dup > 1 else ""
    tag += "_slices" if args.ink_slices else ""
    tag += f"_{args.ctx_channel}" if args.ctx_channel != "cos" else ""
    tag += "_addrholdout" if args.holdout == "address" else ""
    LOG_PATH = RES / f"_stage286_log{tag}.txt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_steps = args.train_steps or (600 if args.smoke else 6000)
    tape_period = args.tape_period or (150 if args.smoke else 500)
    n_addr = args.addresses or (60 if args.smoke else 400)
    global MIN_EVIDENCE
    MIN_EVIDENCE = args.min_evidence

    log(f"Stage286 evidence start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"steps={n_steps} no_hidden={args.no_hidden}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    trunk = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk, map_location=device, weights_only=False)["model"])
    # s213.set_train_mode accepts "upper" and "arc_enc" and raises ValueError on anything else,
    # so the "none" this used to pass killed --frozen-trunk in its first second. Freezing needs
    # no helper: drop every gradient and leave the module in eval. Worth stating what that does
    # NOT change - SelfModelXL carries no dropout, so a frozen trunk reads exactly what an
    # unfrozen one reads at eval time. The two arms differ in whether the trunk learns and in
    # nothing else, which is what makes them a discriminator.
    if args.frozen_trunk or args.mind == "relational":
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        s213.set_train_mode(model, "upper")
    arc0 = s271.arc_enc_hash(model)
    can = SelfModelXL(n_char, V).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    # exact speedups, shared with the 289 family: the corpus scan and the ink are pure
    # functions of the text, and fp_addresses' single-link loop was spending its time on
    # GPU round-trips rather than arithmetic. Verified byte-identical; --no-speedups
    # keeps the original paths reachable so that stays checkable.
    if not args.no_speedups:
        install_all(s279)
    bank = (FpBank(can, stoi, device) if args.no_speedups
            else CachedBank(FpBank(can, stoi, device)))

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(4_000_000 if args.smoke else 30_000_000)
    all_lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][: (3000 if args.smoke else 25000)]
    eval_lines = all_lines[cut:][: (1500 if args.smoke else 12000)]
    if args.holdout == "address":
        # One corpus, one tape geometry, and the split moved onto the SUBJECTS. Both sides then
        # see addresses built by the same process out of the same text, so anything that fails
        # here fails on novelty of the subject alone and not on the two halves of wikitext
        # writing differently.
        eval_lines = train_lines

    def anchor_side(address: str) -> int:
        """Which side of an address holdout a subject falls on.

        Keyed on the anchor, because the fp group number in front of it is assigned by corpus
        order and changes on every resample - an address NAME is not stable across tapes, a
        subject is. A stable hash, so the split is identical in every resample and in the eval
        pack without any state being carried between them.
        """
        anchor = address.split(":", 1)[-1].split("|")[0]
        return int(hashlib.sha1(anchor.encode("utf-8")).hexdigest(), 16) & 1

    def split_items(p, side):
        if args.holdout == "corpus":
            return p
        p = dict(p)
        p["items"] = [it for it in p["items"] if anchor_side(it["address"]) == side]
        return p

    def new_pack(r, lines):
        return s280.pack_from_corpus(lines, bank=bank, tok=tok, pad_id=pad_id, device=device,
                                     rng=r, n_addr=n_addr, min_mentions=args.min_mentions,
                                     tau=args.address_tau, overlap=args.address_overlap,
                                     soft_match=args.soft_match,
                                     min_per_family=args.min_per_family,
                                     addr_key=args.addr_key)

    pack = split_items(new_pack(rng, train_lines), 0)
    if len(pack["items"]) < 8:
        log("  too few items")
        return 1
    usable = [it for it in pack["items"] if len(it["slots"]) > args.min_evidence]
    log(f"  tape: {pack['n_addresses']} addresses, {pack['n_slots']} slots | "
        f"items {json.dumps(dict(Counter(i['kind'] for i in pack['items'])))} | "
        f"usable for leave-one-out (>{args.min_evidence} mentions): {len(usable)}")
    if not usable:
        log("  no address can spare a mention: raise --addresses or lower --min-evidence")
        return 1

    d_hidden = 0 if args.no_hidden else 2 * (model.head.in_features // 2)
    net = (Relational(device,
                      n_edge=3 + 2 * int(args.ink_slices) + int(args.ctx_channel == "both"),
                      n_node=2 + int(args.ctx_channel != "cos"))
           if args.mind == "relational"
           else Evidence(d_hidden, device, read_rank=args.read_rank))
    live = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": net.parameters(), "lr": args.lr_head}]
                            + ([{"params": live, "lr": args.lr_upper}] if live else []),
                            weight_decay=0.01)

    # A frozen trunk returns the same vector for the same text every time, and the same tape
    # is reused for tape_period steps, so the encodes were being recomputed by the thousand.
    # Caching them is exact, not an approximation - and it is what makes a capacity sweep over
    # the reading channel affordable, since each frozen arm otherwise costs an hour.
    read_cache: dict[str, torch.Tensor] = {}
    anon_rx: dict = {}

    def anon_texts(it, ex, texts):
        # Identity is what memorises; phrasing is what could generalise. A reading with the
        # names left in is a unique key per example, and the r8 run showed a lookup table
        # being built on such keys at 5k parameters as happily as at 65k. With subject and
        # candidate values folded to constants the reading can only say HOW things are said.
        # The whole text is still a unique string, so the train control may memorise anyway -
        # the question this answers is held-out only: does phrasing transfer?
        key = (it["S"], tuple(ex["cands"]))
        rx = anon_rx.get(key)
        if rx is None:
            pats = sorted([it["S"]] + list(ex["cands"]), key=len, reverse=True)
            rx = re.compile("|".join(re.escape(p) for p in pats), re.IGNORECASE)
            anon_rx[key] = rx
        subj = it["S"].lower()
        return [rx.sub(lambda m: "someone" if m.group(0).lower() == subj else "something", t)
                for t in texts]

    def read_of(texts):
        if not args.frozen_trunk:
            return encode(model, char_table, tok, pad_id, device, texts)
        key = "\u241f".join(texts)
        h = read_cache.get(key)
        if h is None:
            h = encode(model, char_table, tok, pad_id, device, texts)
            if h is None:
                return None
            h = h.detach()
            read_cache[key] = h
        return h

    @torch.no_grad()
    def ink_coherence(p, ex, value):
        """Is the answered value's coalition tighter than its split from the rest?

        The ink's one surviving job, used as an OBSERVER rather than a judge. For the value the
        mind answered, compare the mean context-fp cosine WITHIN its mentions against the mean
        cosine ACROSS to the others. Below zero means the mentions that agree on the value do
        not agree on anything else - a coalition assembled by coincidence.

        Deliberately not in the loss. A hand-written signal added to L is a teacher through the
        back door, and the circularity 286 exists to remove is precisely "train the weights to
        resemble a heuristic, then score resemblance". It earns its way in the way the return
        path had to: only if the two rates below come apart does it become load bearing.
        """
        slots, vals_e = ex["ev_slots"], ex["vals"]
        ck = p.setdefault("_rel_ctx", {})
        for sl in set(slots):
            if sl not in ck:
                c = bank.ctx_fp(p["texts"][sl], exclude=p["tape"].values[sl])
                ck[sl] = F.normalize(c, dim=-1) if c is not None else None
        inn, out = [], []
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = ck[slots[i]], ck[slots[j]]
                if a is None or b is None:
                    continue
                c = float(a @ b)
                mine = (vals_e[i] == value) + (vals_e[j] == value)
                if mine == 2:
                    inn.append(c)
                elif mine == 1:
                    out.append(c)
        if not inn or not out:
            return float("nan")
        return float(np.mean(inn) - np.mean(out))

    def rel_inputs(p, ex):
        """The graph, in ranks. Everything here comes from plumbing that already existed -
        ctx fingerprints, the inverted index, the probe counts - and leaves as an order
        statistic of THIS example, so a fresh tape re-ranks it all and nothing is worth
        memorising. Caches live on the pack and die with the resample."""
        slots, vals_e = ex["ev_slots"], ex["vals"]
        n = len(slots)
        ck = p.setdefault("_rel_ctx", {})
        ws = p.setdefault("_rel_words", {})
        lk, rk = p.setdefault("_rel_left", {}), p.setdefault("_rel_right", {})
        for sl in set(slots):
            if sl not in ck:
                t, v = p["texts"][sl], p["tape"].values[sl]
                c = bank.ctx_fp(t, exclude=v)
                ck[sl] = F.normalize(c, dim=-1) if c is not None else None
                ws[sl] = set(context_words(t, exclude=v))
                # split at the value: what leads up to it, and what follows
                i = t.lower().find(v.lower())
                left, right = ((t[:i], t[i + len(v):]) if i >= 0 else (t, ""))
                for store, part in ((lk, left), (rk, right)):
                    cc = bank.ctx_fp(part, exclude=v) if part.strip() else None
                    store[sl] = F.normalize(cc, dim=-1) if cc is not None else None
                # the composition residual, in the only place it is not degenerate. Single-word
                # chunks would give D parallel to the mean itself (fp is already unit norm), so
                # cos(Di,Dj) would reproduce the plain ctx cosine exactly; two spans split at
                # the value have a normalisation inside each, and that is what makes D depend
                # on how the words are grouped rather than only on which words they are.
                dk = p.setdefault("_rel_delta", {})
                dn = p.setdefault("_rel_dnorm", {})
                if ck[sl] is not None and lk[sl] is not None and rk[sl] is not None:
                    d = ck[sl] - F.normalize(lk[sl] + rk[sl], dim=-1)
                    dn[sl] = float(d.norm())
                    dk[sl] = F.normalize(d, dim=-1)
                else:
                    dk[sl], dn[sl] = None, 0.0
        med = p.get("_rel_median")
        if med is None:
            lens = sorted(len(v) for v in p["postings"].values())
            med = lens[len(lens) // 2] if lens else 1
            p["_rel_median"] = med
        same = torch.zeros(n, n)
        cos = torch.zeros(n, n)
        shared = torch.zeros(n, n)
        cosL, cosR = torch.zeros(n, n), torch.zeros(n, n)
        cosD = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = slots[i], slots[j]
                same[i, j] = same[j, i] = float(vals_e[i] == vals_e[j])
                if ck[si] is not None and ck[sj] is not None:
                    cos[i, j] = cos[j, i] = float(ck[si] @ ck[sj])
                rare = sum(1 for w in (ws[si] & ws[sj])
                           if len(p["postings"].get(w, ())) < med)
                shared[i, j] = shared[j, i] = rare / max(1, min(len(ws[si]), len(ws[sj])))
                if args.ink_slices:
                    for M, store in ((cosL, lk), (cosR, rk)):
                        if store[si] is not None and store[sj] is not None:
                            M[i, j] = M[j, i] = float(store[si] @ store[sj])
                if args.ctx_channel != "cos":
                    dk = p.setdefault("_rel_delta", {})
                    if dk.get(si) is not None and dk.get(sj) is not None:
                        cosD[i, j] = cosD[j, i] = float(dk[si] @ dk[sj])

        iu = torch.triu_indices(n, n, offset=1)

        def rank_norm(M):
            """Rank the upper triangle by argsort, not by list.index per pair.

            The first version scanned a sorted list for every pair: O(n^4) in the mentions of
            one address. Invisible at n=4 and a wall at n=300, which is exactly the size this
            architecture claims to reach - a mind that does not grow with the knowledge is not
            scale-free if the graph builder is quartic. argsort-of-argsort is O(n^2 log n),
            and ties now share a rank instead of resolving by list order.
            """
            if iu.numel() == 0:
                return M
            v = M[iu[0], iu[1]]
            order = v.argsort()
            r = torch.empty_like(order, dtype=torch.float32)
            r[order] = torch.arange(len(v), dtype=torch.float32)
            uniq, inv = v.unique(return_inverse=True)          # ties get one rank
            if len(uniq) > 1:
                mean_r = torch.zeros(len(uniq)).index_reduce_(
                    0, inv, r, "mean", include_self=False)
                r = mean_r[inv] / (len(v) - 1 if len(v) > 1 else 1)
            else:
                r = torch.zeros_like(r)
            R = torch.zeros_like(M)
            R[iu[0], iu[1]] = r
            R[iu[1], iu[0]] = r
            return R

        # delta REPLACES the plain cosine; both keeps it. Replacing is the arm the evidence
        # points at - an extra fp channel is what ink-slices was, and it cost the dup win.
        chans = [same, rank_norm(cosD if args.ctx_channel == "delta" else cos),
                 rank_norm(shared)]
        if args.ctx_channel == "both":
            chans.append(rank_norm(cosD))
        if args.ink_slices:
            chans += [rank_norm(cosL), rank_norm(cosR)]
        E = torch.stack(chans, -1).to(device)
        ext_per = [ex["ext"].get(v, 0) for v in vals_e]
        levels = sorted(set(ext_per))
        rows_n = [[ex["cnt"][vals_e[i]] / n,
                   levels.index(ext_per[i]) / max(1, len(levels) - 1)] for i in range(n)]
        if args.ctx_channel != "cos":
            # |D| as a node RANK: how much this one mention behaves as a bound span rather
            # than as two independent halves. A rank, so a fresh tape re-ranks it.
            dn = [p.setdefault("_rel_dnorm", {}).get(sl, 0.0) for sl in slots]
            lv = sorted(set(dn))
            for i in range(n):
                rows_n[i].append(lv.index(dn[i]) / max(1, len(lv) - 1))
        nf = torch.tensor(rows_n, dtype=torch.float32, device=device)
        masks = [torch.tensor([vals_e[i] == c for i in range(n)], device=device)
                 for c in ex["cands"]]
        return E, same.unsqueeze(-1).to(device), nf, masks

    def forward(p, it, ex):
        if args.mind == "relational":
            return net(*rel_inputs(p, ex))
        rows = torch.tensor(ex["rows"], dtype=torch.float32, device=device)
        if args.no_hidden:
            return net(torch.zeros(0, device=device), rows)
        hs = []
        for texts in ex["cand_texts"]:
            if args.read_anon:
                texts = anon_texts(it, ex, texts)
            h = read_of(texts)
            if h is None:
                return None
            hs.append(h.float())
        return net(torch.stack(hs), rows)

    # ---------------------------------------------------------------- train: the tape teaches
    net.train()
    model.train(not args.frozen_trunk and args.mind != "relational")
    curve, losses = [], []
    # What the mind was actually shown. The first full run logged n_cands at eight steps and six
    # of them were 1, against 3.56 on the held-out tape - the training distribution was the
    # copying regime failure mode 5 warns about while the exam was not, and nothing measured it.
    seen_cands, seen_unknown, n_trunc, n_skipped = [], [], 0, 0

    def loo_pairs(p):
        # Sample what the examiner enumerates. The hidden run measured the gap this closes:
        # item-uniform sampling saw 1.95 candidates on average while leave-one-out, which walks
        # every slot of every item and so weights an address by its mention count, saw 3.56.
        # The rich contested addresses that dominate scoring were exactly the regime training
        # under-sampled - and where UNKNOWN sits in the composition FLIPS between the two. At
        # ~2 mentions the absent targets live in UNANIMOUS evidence (hide the minority of
        # {A,A,B} and nothing contradicts the leader), and the training numbers pin it:
        # P(UNKNOWN | unanimous) works out to 0.52-0.63, which is where the exam's clean
        # abstain of 0.571 came from. At ~4 mentions the absent targets live in CONTESTED
        # evidence, because an address holding many distinct values makes every hidden slot
        # absent. The head learned the first tape's statistic truthfully and it anti-
        # generalised to the second: unknown_auc 0.259 (2.8 sigma below chance) with a RISING
        # selective curve is the same information with the sign flipped, not an absence of it.
        return [(it, s) for it in p["items"] if len(it["slots"]) > MIN_EVIDENCE
                for s in it["slots"]]

    pairs = loo_pairs(pack)
    for step in range(1, n_steps + 1):
        if (step - 1) % tape_period == 0 and step > 1:
            pack = split_items(new_pack(rng, train_lines), 0)
            pairs = loo_pairs(pack)
            read_cache.clear()          # new tape, new texts; bounds the cache to one tape
            anon_rx.clear()
        if not pairs:
            log("  no multi-mention addresses on this tape")
            return 1
        it, hidden = pairs[rng.randrange(len(pairs))]
        ex = example(pack, it, hidden)
        if ex is None:
            continue
        if ex["truncated"]:
            n_trunc += 1            # the target is in the evidence but outside the top 8: the
            continue                # UNKNOWN label would be a lie, so it is not trained on
        if len(ex["cands"]) < args.min_candidates:
            n_skipped += 1
            continue
        seen_cands.append(len(ex["cands"]))
        seen_unknown.append(int(not ex["present"]))
        logits = forward(pack, it, ex)
        if logits is None:
            continue
        loss = F.cross_entropy(logits.unsqueeze(0),
                               torch.tensor([ex["target"]], device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + live, 1.0)
        opt.step()
        losses.append(float(loss))
        if step % max(1, n_steps // 8) == 0:
            curve.append({"step": step, "loss": float(np.mean(losses[-200:])),
                          "kind": it["kind"], "n_cands": len(ex["cands"]),
                          "mean_cands_so_far": _mean(seen_cands)})
            log(f"  step {step}/{n_steps} loss={np.mean(losses[-200:]):.4f} [{it['kind']}] "
                f"cands={len(ex['cands'])} mean_cands={_mean(seen_cands):.2f}")

    train_dist = {"n": len(seen_cands), "sampling": "pair_uniform",
                  "mean_candidates": _mean(seen_cands),
                  "target_unknown_rate": _mean(seen_unknown),
                  "single_candidate_rate": _mean([int(c == 1) for c in seen_cands]),
                  "truncated_skipped": n_trunc, "min_candidates_skipped": n_skipped}
    log(f"  TRAIN DIST {json.dumps(train_dist)}")

    net.eval()
    model.eval()
    arc1 = s271.arc_enc_hash(model)

    # How the head's first layer splits between the reading and the nine features, per input
    # dimension so 512 columns do not out-count 9 by construction. A diagnostic, never a gate:
    # if the reading arm fails while this shows the reading columns carrying most of the norm,
    # the drowning hypothesis has room; if the feature columns dominate and the arm still
    # fails, the damage came through the trunk's own drift, not through the head's weighting.
    if args.mind == "relational":
        head_norms = {"mind": "relational",
                      "params": int(sum(q.numel() for q in net.parameters()))}
    else:
      with torch.no_grad():
        w0, d_read = net.f[0].weight, net.d_read
        head_norms = {
            "d_reading": d_read, "d_features": N_FEAT,
            "reading": float(w0[:, :d_read].norm()) if d_read else 0.0,
            "features": float(w0[:, d_read:].norm()),
            "reading_per_dim": (float(w0[:, :d_read].norm()) / math.sqrt(d_read)
                                if d_read else 0.0),
            "features_per_dim": float(w0[:, d_read:].norm()) / math.sqrt(N_FEAT),
            # what the run is fitting, against what it has to fit it with
            "reading_path_params": d_read * 128 + (d_hidden * d_read
                                                    if net.proj is not None else 0),
            "distinct_train_examples_est": len(pairs) * max(1, n_steps // tape_period),
        }
    log(f"  HEAD NORMS {json.dumps(head_norms)}")

    # ---------------------------------------------------------------- score: three examiners
    @torch.no_grad()
    def leave_one_out(p, lie: bool, dup: int = 1):
        r = random.Random(SEED + (1 if lie else 0))
        n_slots = p["n_slots"]
        ans, cor, rew = defaultdict(list), defaultdict(list), defaultdict(list)
        conf, forced_ok = [], []
        p_unk_absent, p_unk_present = [], []
        m_unk_absent, m_unk_present = [], []
        n_cands, unknowable, trunc = [], [], 0
        unk_argmax = {"absent": [], "present": []}
        acc_model = []
        # Split by whether the evidence needed weighing at all. On the exam the first passing
        # arm abstained on 100% of decidable and 100% of tie while answering 100% of clean -
        # a policy exactly equal to "answer iff the address is unanimous", which is one
        # threshold on one count feature, not evidence being weighed. If that also holds on
        # leave-one-out, EVIDENCE_OK is a copying rule that learned when to keep quiet, and
        # the honest thing is for the gates to say so rather than for a reader to divide
        # coverage by unanimity and notice.
        coh_ok, coh_bad = [], []
        uni_ans, con_ans, uni_ok, con_ok = [], [], [], []
        con_conf, con_forced_ok, con_votes_ans, con_votes_ok, con_rand = [], [], [], [], []
        for it in p["items"]:
            if len(it["slots"]) <= MIN_EVIDENCE:
                continue
            for hidden in it["slots"]:
                lie_from = None
                if lie:
                    lie_from = r.randrange(n_slots)
                    if lie_from in set(it["slots"]):
                        continue
                ex = example(p, it, hidden, lie_from=lie_from,
                             lie_dup=(dup if lie else 1))
                if ex is None or not ex["cands"]:
                    continue
                logits = forward(p, it, ex)
                if logits is None:
                    continue
                prob = torch.softmax(logits, dim=-1)
                pick = int(logits.argmax())
                model_ans = ex["cands"][pick] if pick < len(ex["cands"]) else None
                truth = p["tape"].values[hidden]
                trunc += int(ex["truncated"])
                # The forced answer and its confidence, taken over the candidate rows only. The
                # selective comparison needs a ranking of items the mind would answer, which is
                # a different question from whether its argmax happened to land on UNKNOWN.
                cand_p = prob[: len(ex["cands"])]
                best = int(cand_p.argmax())
                conf.append(float(cand_p[best]))
                forced_ok.append(int(ex["cands"][best] == truth))
                for name, a in (("model", model_ans), ("votes", votes_judge(ex)),
                                ("return", return_judge(p, it, ex))):
                    ans[name].append(int(a is not None))
                    if a is not None:
                        cor[name].append(int(a == truth))
                    rew[name].append(reward_of(a, truth, args.abstain_reward, args.wrong_cost))
                if model_ans is not None:
                    acc_model.append(int(model_ans == truth))
                    coh = ink_coherence(p, ex, model_ans)
                    if not math.isnan(coh):
                        (coh_ok if model_ans == truth else coh_bad).append(coh)
                # Honesty in the distribution, not at the argmax. Whether the hidden value is
                # among the visible ones is a property of the HIDDEN slot; where the evidence is
                # unanimous the UNKNOWN row is identical under both labels, so with a ~20% base
                # rate a MAP decision can essentially never choose it and 0.000 was correct
                # behaviour rather than a dishonest mind. What can be asked of the evidence is
                # whether it moves P(UNKNOWN), and AUC asks exactly that.
                pu = float(prob[-1])
                (p_unk_absent if not ex["present"] else p_unk_present).append(pu)
                # P(UNKNOWN) shares one softmax with however many candidate rows the example
                # happens to have, so it is mechanically diluted by |cands|. Absent targets
                # live in CONTESTED evidence, which has more candidates - so the probability
                # measure is biased against exactly the hypothesis it tests. The relational
                # run made the confound visible: at the argmax it chose UNKNOWN on 0.714 of
                # absent against 0.589 of present, while the mean probabilities were 0.478 and
                # 0.484 - the same ordering, erased by dilution. The logit MARGIN against the
                # best candidate is what the argmax actually compares, and it does not care
                # how many other rows exist.
                mu = float(logits[-1] - logits[: len(ex["cands"])].max())
                (m_unk_absent if not ex["present"] else m_unk_present).append(mu)
                unk_argmax["absent" if not ex["present"] else "present"].append(
                    int(model_ans is None))
                n_cands.append(len(ex["cands"]))
                unknowable.append(int(not ex["present"]))
                if len(ex["cnt"]) == 1:
                    uni_ans.append(int(model_ans is not None))
                    if model_ans is not None:
                        uni_ok.append(int(model_ans == truth))
                else:
                    con_ans.append(int(model_ans is not None))
                    if model_ans is not None:
                        con_ok.append(int(model_ans == truth))
                    # the same matched-coverage comparison, restricted to the subset where
                    # weighing is the task rather than copying
                    con_conf.append(float(cand_p[best]))
                    con_forced_ok.append(int(ex["cands"][best] == truth))
                    vj = votes_judge(ex)
                    con_votes_ans.append(int(vj is not None))
                    if vj is not None:
                        con_votes_ok.append(int(vj == truth))
                    con_rand.append((1.0 / len(ex["cands"])) if ex["present"] else 0.0)
        out = {"n": len(conf), "mean_candidates": _mean(n_cands),
               "target_unknown_rate": _mean(unknowable), "truncated_rate": trunc / max(1, len(conf)),
               "silence_reward": args.abstain_reward,
               "break_even_accuracy": (args.abstain_reward + args.wrong_cost)
                                      / (1.0 + args.wrong_cost),
               "model": scorer_stats(ans["model"], cor["model"], rew["model"]),
               "votes": scorer_stats(ans["votes"], cor["votes"], rew["votes"]),
               "return": scorer_stats(ans["return"], cor["return"], rew["return"]),
               # the competition-invariant one: gated on, because it measures what the
               # argmax decides. The probability pair below is kept so older runs stay
               # comparable term by term.
               "unknown_margin_auc": auc(m_unk_absent, m_unk_present),
               "unknown_margin_auc_z": auc_z(auc(m_unk_absent, m_unk_present),
                                             len(m_unk_absent), len(m_unk_present)),
               "unknown_margin_when_absent": _mean(m_unk_absent),
               "unknown_margin_when_present": _mean(m_unk_present),
               "unknown_auc": auc(p_unk_absent, p_unk_present),
               "unknown_auc_z": auc_z(auc(p_unk_absent, p_unk_present),
                                      len(p_unk_absent), len(p_unk_present)),
               "p_unknown_when_absent": _mean(p_unk_absent),
               "p_unknown_when_present": _mean(p_unk_present),
               "unknown_when_absent": _mean(unk_argmax["absent"]),
               "unknown_when_present": _mean(unk_argmax["present"]),
               "n_absent": len(p_unk_absent), "n_present": len(p_unk_present),
               # the flat numbers the earlier runs reported, kept so old JSON stays comparable
               "model_reward": _mean(rew["model"]), "votes_reward": _mean(rew["votes"]),
               "return_reward": _mean(rew["return"]),
               "model_acc_answered": _mean(acc_model), "model_abstain": 1.0 - _mean(ans["model"])}
        # The mind at each judge's own operating point.
        matched = {}
        for name in ("votes", "return"):
            j = out[name]
            matched[name] = {
                "judge_coverage": j["coverage"], "judge_accuracy": j["accuracy"],
                "model_accuracy_at_that_coverage": acc_at_coverage(conf, forced_ok,
                                                                   j["coverage"]),
                # How many items the comparison actually rests on. The nohid arm cleared
                # G_learns_evidence on 1.000 against 0.889 at the return judge's coverage -
                # which is NINE items. A gate decided by nine items has to say so in the JSON
                # rather than in whoever remembers to divide.
                "n_compared": j["n_answered"],
                "judge_vacuous": j["vacuous"]}
        out["matched_coverage"] = matched
        out["selective_curve"] = {f"{c:.2f}": acc_at_coverage(conf, forced_ok, c)
                                  for c in (0.1, 0.25, 0.5, 0.75, 1.0)}
        # An observer, never a vote: the mind does not see it and the loss does not move with
        # it. Useful only if the two numbers come apart - the same bar the return path had to
        # clear before 280 would report it beside accuracy.
        out["ink_coherence"] = {
            "when_correct": _mean(coh_ok), "when_wrong": _mean(coh_bad),
            "separation": _mean(coh_ok) - _mean(coh_bad),
            "n_correct": len(coh_ok), "n_wrong": len(coh_bad)}
        v_cov = _mean(con_votes_ans)
        out["by_evidence_shape"] = {
            "unanimous_rate": len(uni_ans) / max(1, len(conf)),
            "n_unanimous": len(uni_ans), "n_contested": len(con_ans),
            "answer_rate_when_unanimous": _mean(uni_ans),
            "answer_rate_when_contested": _mean(con_ans),
            "accuracy_when_unanimous": _mean(uni_ok),
            "accuracy_when_contested": _mean(con_ok),
            "n_contested_answered": int(sum(con_ans)),
            # votes on the same subset, and the mind at votes' coverage there. This is the
            # comparison the stage's claim actually rests on: on unanimous evidence there is
            # nothing to weigh and any scorer copies the only value.
            # expected accuracy of picking a candidate at random: the null that "beats
            # votes" forgot. The r8 arm passed the old gate at 0.152 against votes' 0.121 -
            # two scorers under the floor, ranked solemnly.
            "random_accuracy_contested": _mean(con_rand),
            "votes_coverage_contested": v_cov,
            "votes_accuracy_contested": _mean(con_votes_ok),
            "model_accuracy_at_votes_coverage_contested":
                acc_at_coverage(con_conf, con_forced_ok, v_cov) if con_conf else float("nan")}
        return out

    def beats_judges(loo):
        """Beat each judge where it chose to speak. Vacuous judges are neither beaten nor lost
        to - a scorer whose answered set is too small for any comparison against it to be
        significant is the abstain constant with a rounding error, and ranking against it is
        failure mode 3 with extra steps."""
        live_j = [n for n in ("votes", "return") if not loo["matched_coverage"][n]["judge_vacuous"]]
        if not live_j:
            return None, []
        ok = []
        for n in live_j:
            m = loo["matched_coverage"][n]
            a, b = m["model_accuracy_at_that_coverage"], m["judge_accuracy"]
            ok.append(not math.isnan(a) and not math.isnan(b) and a >= b)
        return all(ok), live_j

    @torch.no_grad()
    def exam(p):
        """A consistency check, not an examiner.

        280 defines item["truth"] as the majority value of the address and None on a tie. With
        every mention visible votes_judge computes that same function of that same input, so
        here votes IS the label and its score is the ceiling: 1.0 wherever the corpus settled
        and the abstain reward on every tie, since answering a tie is wrong by construction.
        The old G_exam_matches_judges asked the mind to tie an oracle. What is left worth
        reporting is the gap to that ceiling.
        """
        agg = defaultdict(list)
        per = {f: defaultdict(list) for f in s280.FAMILIES}
        oracle = []
        for it in p["items"]:
            ex = example(p, it, None)
            if ex is None or not ex["cands"]:
                continue
            logits = forward(p, it, ex)
            if logits is None:
                continue
            pick = int(logits.argmax())
            ans = ex["cands"][pick] if pick < len(ex["cands"]) else None
            rew = reward_of(ans, it["truth"], args.abstain_reward, args.wrong_cost)
            agg["r"].append(rew)
            agg["votes_r"].append(reward_of(votes_judge(ex), it["truth"],
                                            args.abstain_reward, args.wrong_cost))
            agg["return_r"].append(reward_of(return_judge(p, it, ex), it["truth"],
                                             args.abstain_reward, args.wrong_cost))
            oracle.append(1.0 if it["truth"] is not None else args.abstain_reward)
            f = it["kind"]
            per[f]["abstain"].append(int(ans is None))
            per[f]["reward"].append(rew)
            if ans is not None:
                per[f]["acc"].append(int(it["truth"] is not None and ans == it["truth"]))
        out = {"n": len(agg["r"]), "reward": _mean(agg["r"]),
               "votes_reward": _mean(agg["votes_r"]), "return_reward": _mean(agg["return_r"]),
               "oracle_reward": _mean(oracle),
               "gap_to_oracle": _mean(oracle) - _mean(agg["r"]),
               "votes_is_the_label": True}
        for f in s280.FAMILIES:
            out[f] = {"n": len(per[f]["abstain"]), "abstain": _mean(per[f]["abstain"]),
                      "acc_answered": _mean(per[f]["acc"]), "reward": _mean(per[f]["reward"])}
        return out

    # Control: the same examiner on the LAST TRAINING tape, before the held-out one is read.
    # It localises a failure instead of leaving it to interpretation. AUC above 0.5 here and
    # below it held-out is distribution shift between the tapes - the corpus halves build
    # different addresses. Below 0.5 on BOTH is worse: the target was not a function of the
    # features even where the head was fit, and no composition story explains that. And the
    # pair the hidden arm's rerun must answer: if the nohid arm clears this control while the
    # hidden arm stays inverted on the same tape, the inversion lives in the READING - the
    # trunk memorising subject->value across resamples of the same 25k lines, which is exactly
    # the leak the separation invariant says must not pay.
    loo_train = leave_one_out(pack, lie=False)
    log(f"  LOO-TRAIN(control) {json.dumps(loo_train)}")

    held = split_items(new_pack(random.Random(SEED + 99), eval_lines), 1)
    loo = leave_one_out(held, lie=False)
    lied = leave_one_out(held, lie=True)
    lied_dup = (leave_one_out(held, lie=True, dup=args.lie_dup)
                if args.lie_dup > 1 else None)
    ex_out = exam(held)
    log(f"  LOO {json.dumps(loo)}")
    log(f"  LIE {json.dumps(lied)}")
    if lied_dup is not None:
        log(f"  LIE-DUP(x{args.lie_dup}) {json.dumps(lied_dup)}")
    log(f"  EXAM {json.dumps(ex_out)}")

    g_arc = arc0 == arc1
    # Before any claim about learning: was there something to learn from, and was the mind shown
    # it? The first smoke had one candidate everywhere; the first full run had one candidate
    # almost everywhere in TRAINING while the exam had 3.56, and only the exam was checked.
    # "more than one candidate on average" is the whole content; 1.05 was padding around 1.0
    # and 20 was a round number. Both become the derived floor and the exact statement.
    g_task = (loo["n"] >= 4 * MIN_ANSWERED and loo["mean_candidates"] > 1.0
              and loo["target_unknown_rate"] > 0.0
              and train_dist["n"] >= 4 * MIN_ANSWERED
              and train_dist["mean_candidates"] > 1.0)
    g_learns, live_judges = beats_judges(loo)
    g_lie_full, lie_judges = beats_judges(lied)
    lie_m = lied["matched_coverage"]["votes"]
    g_lie = (None if lie_m["judge_vacuous"]
             else bool(lie_m["model_accuracy_at_that_coverage"] >= lie_m["judge_accuracy"]))
    dup_m = lied_dup["matched_coverage"]["votes"] if lied_dup else None
    g_lie_dup = (None if dup_m is None or dup_m["judge_vacuous"]
                 else bool(dup_m["model_accuracy_at_that_coverage"]
                           >= dup_m["judge_accuracy"]))
    honest_testable = (loo["n_absent"] >= MIN_ANSWERED
                       and loo["n_present"] >= MIN_ANSWERED)
    # Above chance BY MORE THAN ITS OWN NOISE. The first arm to pass this gate did so at 0.560,
    # which is 0.69 sigma on 14x56 - while the same measurement on the training tape stood at
    # 4.73 sigma. "> 0.5" cannot separate a real effect from a coin, and 1.645 is the one-sided
    # 95% point of the null, not a number picked to make an arm pass.
    g_honest = (None if not honest_testable
                else bool(loo["unknown_margin_auc_z"] > 1.645))
    # Did it weigh anything, or only decline to copy? Restricted to contested evidence, where
    # unanimity cannot answer for it, and guarded by a denominator.
    bes = loo["by_evidence_shape"]
    g_weighs = bool(bes["n_contested_answered"] >= MIN_ANSWERED
                    and not math.isnan(bes["model_accuracy_at_votes_coverage_contested"])
                    and not math.isnan(bes["votes_accuracy_contested"])
                    and bes["model_accuracy_at_votes_coverage_contested"]
                    >= bes["votes_accuracy_contested"]
                    and bes["model_accuracy_at_votes_coverage_contested"]
                    > bes["random_accuracy_contested"])
    g_judges = bool(live_judges)
    g_beats_silence = bool(loo["model"]["reward"] > args.abstain_reward)

    overall = ("NO_TASK" if not g_task
               else "JUDGES_VACUOUS" if not g_judges
               else "HONESTY_UNTESTED" if g_honest is None
               else "EVIDENCE_OK" if (g_arc and g_learns and g_honest and g_lie and g_weighs)
               # Passing everything except the contested subset is a real and nameable
               # outcome, not a near miss: it is "answer iff unanimous, else keep quiet" -
               # learned rather than coded, better than votes because it is selective, and
               # still not the weighing this stage set out to train.
               else "EVIDENCE_OK_UNANIMOUS_ONLY" if (g_arc and g_learns and g_honest and g_lie)
               else "EVIDENCE_PARTIAL" if (g_arc and (g_learns or g_lie))
               else "EVIDENCE_NO")

    out = {
        "stage": 286, "overall": overall, "seed": SEED, "smoke": args.smoke,
        "run_tag": args.run_tag, "no_hidden": args.no_hidden, "addr_key": args.addr_key,
        "train_steps": n_steps, "min_mentions": args.min_mentions,
        "min_evidence": args.min_evidence, "min_candidates": args.min_candidates,
        "min_per_family": args.min_per_family,
        "reward": {"correct": 1.0, "wrong": -args.wrong_cost, "abstain": args.abstain_reward},
        "min_answered_for_a_comparison": MIN_ANSWERED,
        "gates": {
            "G_arc_enc_frozen": g_arc,
            "G_task_exists": g_task,
            "G_judges_non_vacuous": g_judges,
            "G_learns_evidence": g_learns,
            "G_abstains_unknowable": g_honest,
            "G_survives_lie": g_lie,
            "G_weighs_contested": g_weighs,
            "G_survives_duplicated_lie": g_lie_dup,
            "G_beats_silence": g_beats_silence,
        },
        "judges_compared": live_judges, "judges_compared_lie": lie_judges,
        "mind": args.mind, "lie_dup": args.lie_dup, "ink_slices": args.ink_slices,
        "ctx_channel": args.ctx_channel, "holdout": args.holdout,
        "train_distribution": train_dist, "head_input_norms": head_norms,
        "leave_one_out": loo, "leave_one_out_train_control": loo_train,
        "lying_tape": lied, "lying_tape_dup": lied_dup, "exam": ex_out, "curve": curve,
        "arc_enc_hash_before": arc0, "arc_enc_hash_after": arc1,
        "fp_version": s271.fp_version(),
        "reference_280_baseline": {"held_out_reward": 0.704, "acc_answered_all": 0.900},
        "reference_286_first_full_nohid": {
            "overall": "EVIDENCE_NO", "model_reward": 0.100, "votes_reward": 0.014,
            "return_reward": 0.754, "unknown_when_absent": 0.000,
            "unknown_when_present": 0.218, "exam_reward": 0.777, "exam_votes_reward": 0.932,
            "note": "scored by the examiners this file replaces; not comparable term by term"},
        "reference_286_ablation_pair_gen2": {
            "code": "second-generation examiners, item-uniform sampler, no train control",
            "nohid": {"overall": "EVIDENCE_PARTIAL", "loo_accuracy": 0.641,
                      "loo_coverage": 0.557, "unknown_auc": 0.412, "selective_10pct": 1.0},
            "hidden": {"overall": "EVIDENCE_NO", "loo_accuracy": 0.325,
                       "loo_coverage": 0.571, "unknown_auc": 0.259, "selective_10pct": 0.286},
            "note": "same tapes - judge numbers byte-identical between the runs. Reading "
                    "halved accuracy at equal coverage and inverted the confidence ordering; "
                    "the features-only ablation beat the arm carrying the stage's claim."},
        "note": (
            "The direction of teaching is inverted: nobody demonstrates, the tape is the label. "
            "Hide one mention of an address, show the rest, require the hidden value - so there "
            "are as many examples as slots, no judge is imitated, and what the weights learn is "
            "what the judges were hand-written approximations of: weighing evidence. Votes and "
            "corroboration are features on the candidates now, every row reads the same schema "
            "so the UNKNOWN row can see the corroboration too, and each candidate is READ from "
            "the mentions that carry it. The examiners changed after the first full run, which "
            "measured four properties of the scoring rather than of the mind. The exam is no "
            "longer gated: with every mention visible, votes computes the majority and the "
            "label IS the majority, so votes was an oracle and 0.932 was the ceiling. Reward is "
            "no longer the comparison: at abstain 0.75 against wrong -1.0 nothing beats silence "
            "under 87.5% accuracy, so every scorer piles up at the constant and coverage and "
            "accuracy are now reported apart, with a judge under 10% coverage marked vacuous "
            "instead of being ranked against. The mind is compared to each judge at THAT "
            "JUDGE'S coverage, ranked by its own confidence. And honesty is measured in the "
            "distribution: whether the hidden value is visible is a property of the hidden "
            "slot, and where the evidence is unanimous the UNKNOWN row is the same input under "
            "both labels, so an argmax can never choose it against a 20% base rate - what the "
            "evidence can be asked is whether it moves P(UNKNOWN), which is an AUC. The lying "
            "tape stays the sharpest test of the concept: a forged mention has no external "
            "corroboration and a mind that weighs evidence should discount it, while votes by "
            "construction cannot. The address itself is still 280's plumbing - this stage is "
            "about weighing WITHIN the evidence set, so the ceiling here is what a correct "
            "address makes visible."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"stage286_decision{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RES / f"stage286_mini{tag}.md").write_text(
        f"# Stage 286 the tape as its own label"
        f"{' (no hidden)' if args.no_hidden else ''}\n\n**{overall}**"
        f"{' · SMOKE' if args.smoke else ''}\n\n"
        f"| leave-one-out | coverage | accuracy | reward |\n|---|---:|---:|---:|\n"
        + "".join(
            f"| {n}{' (vacuous)' if loo[n]['vacuous'] else ''} | {loo[n]['coverage']:.2f} | "
            f"{loo[n]['accuracy']:.2f} | {loo[n]['reward']:.3f} |\n"
            for n in ("model", "votes", "return"))
        + f"\n- unconditional silence is worth {args.abstain_reward:.2f}; answering pays only "
          f"above accuracy {loo['break_even_accuracy']:.3f}\n"
        + "".join(
            f"- at {n}'s coverage {loo['matched_coverage'][n]['judge_coverage']:.2f}: model "
            f"{loo['matched_coverage'][n]['model_accuracy_at_that_coverage']:.2f} vs judge "
            f"{loo['matched_coverage'][n]['judge_accuracy']:.2f}"
            f"{' — judge vacuous, not compared' if loo['matched_coverage'][n]['judge_vacuous'] else ''}"
            f"\n" for n in ("votes", "return"))
        + f"- control: unknown AUC {loo_train['unknown_auc']:.3f} on the last training tape "
          f"against {loo['unknown_auc']:.3f} held out; the pair says whether a failure is "
          f"shift between the tapes or not a function of the features at all\n"
        + f"- contested evidence only ({bes['n_contested']} sets): model answers "
          f"{bes['answer_rate_when_contested']:.2f} of them at "
          f"{bes['accuracy_when_contested']:.2f}, against "
          f"{bes['answer_rate_when_unanimous']:.2f} at "
          f"{bes['accuracy_when_unanimous']:.2f} on unanimous ones; at votes' contested "
          f"coverage {bes['votes_coverage_contested']:.2f} model "
          f"{bes['model_accuracy_at_votes_coverage_contested']:.2f} vs votes "
          f"{bes['votes_accuracy_contested']:.2f}\n"
        + f"- UNKNOWN's margin over the best candidate separates absent from present at AUC "
          f"**{loo['unknown_margin_auc']:.3f}** ({loo['unknown_margin_auc_z']:+.2f} sigma); "
          f"the diluted probability form says {loo['unknown_auc']:.3f} "
          f"({loo['p_unknown_when_absent']:.3f} vs {loo['p_unknown_when_present']:.3f}); at the "
          f"argmax {loo['unknown_when_absent']:.3f} vs {loo['unknown_when_present']:.3f}\n"
        f"- training saw {train_dist['mean_candidates']:.2f} candidates on average, "
        f"{train_dist['single_candidate_rate']:.2f} of examples unanimous\n"
        f"- exam is a consistency check only: oracle {ex_out['oracle_reward']:.3f}, model "
        f"{ex_out['reward']:.3f}, gap {ex_out['gap_to_oracle']:.3f}\n\n## Gates\n\n"
        + "".join(f"- {kk}: **{vv}**\n" for kk, vv in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
