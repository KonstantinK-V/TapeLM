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
v0 = v13('results')
v1 = v13('checkpoints/stage191_p1_curve.pt')
v2 = v13('checkpoints/stage253_joint_l02.pt')
v3 = v13('data/_wikitext103_train.txt')
v4 = 286
v5 = 4
v6 = 5
v7 = v5 + v6
v8 = 2
v9 = v14((v38 for v38 in v238(1, 64) if 0.5 ** v38 < 0.05))
v10 = v0 / '_stage286_log.txt'

def log(v15: v131) -> None:
    v16 = v15 if v15.v270('\n') else v15 + '\n'
    try:
        v271(v16, end='', flush=True)
    except v132:
        v271(v16.v398('ascii', 'replace').v436('ascii'), end='', flush=True)
    v10.v272.v133(parents=True, exist_ok=True)
    with v10.v273('a', encoding='utf-8') as v134:
        v134.v274(v16)

def ext_support(v17, v18: v131, v19: v131, v20: v144[v11]) -> v11:
    """Mentions OUTSIDE the address that carry subject and value together.

    Outside, because inside is what the model already sees - and because the hidden slot is a
    member, counting members would hand the target back through the index. This is 285's return
    judge reduced to a number the model may weigh however the evidence warrants.
    """
    v21 = v275(v19) or [v19]
    v22 = v135((v17['postings_probe'].v318(v371, ()) for v371 in v21), key=v145, default=())
    v23 = v19.v136()
    return v137((1 for v140 in v22 if v140 not in v20 and v18 in v17['texts_lc'][v140] and (v23 in v17['texts_lc'][v140])))

def example(v17, v24, v25: v11 | None, v26: v11 | None=None, v27: v11=1):
    """One evidence set. hidden=None is exam mode: all mentions visible, predict the address.

    The candidate list is what the EVIDENCE says plus UNKNOWN, so the model can only answer
    with a value the tape holds, and must answer UNKNOWN when the tape (as visible) does not
    hold the answer - which is the honesty this stage trains instead of assumes.
    """
    v138, v56 = (v17['tape'].v139, v17['texts'])
    v28 = [v140 for v140 in v24['slots'] if v140 != v25]
    if v145(v28) < (v8 if v25 is not None else 1):
        return None
    v29 = [v138[v140] for v140 in v28]
    v30 = [v56[v140] for v140 in v28]
    v31 = v141(v28)
    if v26 is not None:
        if v145(v28) < 2:
            return None
        v142 = v145(v28) - 1
        v29 = v29[:v142] + [v138[v26]] * v27
        v30 = v30[:v142] + [v56[v26]] * v27
        v31 = v31[:v142] + [v26] * v27
    v32 = v143(v29)
    v33 = [v23 for v23, v157 in v32.v146(8)]
    v34 = v138[v25] if v25 is not None else v24['truth']
    v35 = v34 is not None and v34 in v32
    v36 = v35 and v34 not in v33
    v37 = v33.v276(v34) if v35 and (not v36) else v145(v33)
    v20 = v144(v24['slots'])
    v38 = v145(v29)
    v39 = v32.v146(2)
    v40 = v39[0][1]
    v41 = v39[1][1] if v145(v39) > 1 else 0
    v42 = {v23: v277(v17, v24['S'], v23, v20) for v23 in v33}
    v43 = v147(v42.v139(), reverse=True)
    v44 = v43[0] if v43 else 0
    v45 = v43[1] if v145(v43) > 1 else 0
    v46 = [v145(v32) / v38, (v40 - v41) / v38, 1.0 / v38, v135(v44, 3) / 3.0, v135(v44 - v45, 3) / 3.0]
    v47 = []
    for v23 in v33:
        v148 = (v32[v23] - v173((v140 for v371, v140 in v32.v288() if v371 != v23), default=0)) / v38
        v47.v149([v32[v23] / v38, v148, v135(v42[v23], 3) / 3.0, 0.0] + v46)
    v47.v149([0.0, 0.0, 0.0, 1.0] + v46)
    v48 = v302.v278.v150(S=v24.v318('query') or v24['S'])
    v49 = [[v48] + [v372 for v372, v462 in v463(v30, v29) if v462 == v23] for v23 in v33]
    v49.v149([v48] + v30)
    return {'texts': v30, 'vals': v29, 'cands': v33, 'rows': v47, 'cand_texts': v49, 'target': v37, 'cnt': v32, 'ext': v42, 'ev_slots': v31, 'present': v262(v35), 'truncated': v262(v36)}

class Evidence(v50.v12):
    """Scores each candidate (and UNKNOWN) from ITS OWN mentions, read by the trunk.

    The first version pooled every mention into one vector and handed the same vector to every
    candidate row. That reading cannot say which candidate the text supports - it is a constant
    across the softmax and only modulates how the counts are weighed - so the head would have
    been votes and corroboration wearing a hat, and --no-hidden would have tied on purpose.
    Each candidate now gets the mentions that actually carry it, cued with the subject, so the
    weights compare readings and not only numbers. That is the whole claim of this stage.
    """

    def __init__(v151, v99: v11, v55, v152: v11=0):
        v437().v279()
        v151.v153 = v50.v464(v99, v152, bias=False).v195(v55) if v99 and v152 else None
        v154 = v152 if v151.v153 is not None else v99
        v151.v154 = v154
        v151.v134 = v50.v438(v50.v464(v154 + v7, 128), v50.v465(), v50.v464(128, 1)).v195(v55)
        v50.v373.v280(v151.v134[-1].v281)
        v50.v373.v280(v151.v134[-1].v282)

    def forward(v151, v155, v47):
        """H is one reading PER ROW, not one reading for the example."""
        if v155.v374() and v151.v153 is not None:
            v155 = v151.v153(v155)
        return v151.v134(v294.v287([v155, v47], dim=-1)).v375(-1) if v155.v374() else v151.v134(v47).v375(-1)

def encode(v51, v52, v53, v54, v55, v56):
    """Read a set of mentions. The window is the trunk's own - pos is Embedding(MAX_ARCS, d),
    so anything longer indexes past the table and the run dies on the device rather than in
    Python. 271 truncates the same way and for the same reason."""
    v57 = [v216 for v216 in v53.v398(' | '.v320(v56)[-2000:]).v57 if v216 != v54][-v384.v376:]
    if not v57:
        return None
    v156, v157 = v158(v51, v52, v294.v283([v57], dtype=v294.v377, device=v55), v54)
    return v156[0, -1]

class Relational(v50.v12):
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

    def __init__(v151, v55, v159: v11=32, v160: v11=3, v161: v11=2):
        v437().v279()
        v151.v162 = v50.v438(v50.v464(v160, v159), v50.v465()).v195(v55)
        v151.v163 = v50.v438(v50.v464(v161 + 2 * v159, v159), v50.v465()).v195(v55)
        v151.v164 = v50.v438(v50.v464(2 * v159 + 1, v159), v50.v465(), v50.v464(v159, 1)).v195(v55)
        v50.v373.v280(v151.v164[-1].v281)
        v50.v373.v280(v151.v164[-1].v282)

    def forward(v151, v165, v166, v167, v168):
        v169 = v151.v162(v165)
        v20 = (v169 * v166).v137(1) / v166.v137(1).v378(min=1.0)
        v156 = v151.v163(v294.v287([v167, v20, v169.v284(1)], -1))
        v170 = v156.v284(0)
        v285, v286 = (v294.v328(1, device=v170.v55), v294.v379(1, device=v170.v55))
        v171 = [v151.v164(v294.v287([v156[v15].v284(0), v170, v285])) for v15 in v168]
        v171.v149(v151.v164(v294.v287([v294.v410(v170), v170, v286])))
        return v294.v287(v171)

def votes_judge(v58):
    v59 = v58['cnt'].v146(2)
    if v145(v59) > 1 and v59[0][1] == v59[1][1]:
        return None
    return v59[0][0]

def return_judge(v17, v24, v58):
    v60 = v147(v58['ext'].v288(), key=lambda v372: -v372[1])
    if not v60:
        return None
    (v172, v78), v79 = (v60[0], v60[1][1] if v145(v60) > 1 else 0)
    return None if v78 < 2 or v78 == v79 else v172

def reward_of(v61, v62, v63, v64):
    if v61 is None:
        return v63
    return 1.0 if v62 is not None and v61 == v62 else -v64

def _mean(v65):
    return v174(v381.v284(v65)) if v145(v65) else v174('nan')

def scorer_stats(v61, v66, v67):
    """Reward alone hides the difference between silence and selectivity.

    With abstain +0.75 against wrong -1.0, answering beats silence only above 87.5% accuracy, so
    every scorer piles up near the abstain constant and the ordering between them is decided by
    a handful of items. The three numbers that actually differ - how often it spoke, how often
    it was right when it spoke, and what that was worth - are reported separately from here on.
    """
    v38 = v145(v61)
    v68 = v11(v137(v61))
    return {'n': v38, 'coverage': v68 / v38 if v38 else v174('nan'), 'accuracy': v174(v137(v66)) / v68 if v68 else v174('nan'), 'reward': v289(v67), 'n_answered': v68, 'vacuous': v262(v38 and v68 < v9)}

def acc_at_coverage(v69, v70, v71):
    """The mind's accuracy on the fraction of items it is most confident about.

    A judge that answers a fifth of the items at high accuracy and a mind that answers all of
    them at middling accuracy cannot be ordered by either number alone. Ranking by the mind's
    own confidence and cutting at the judge's coverage compares them at the same operating
    point, which is the only comparison that has a direction. No threshold is chosen: the
    coverage comes from whatever the judge did.
    """
    v38 = v145(v69)
    if v38 == 0 or not 0.0 < v71 <= 1.0:
        return v174('nan')
    v72 = v147(v238(v38), key=lambda v216: -v69[v216])
    v15 = v173(1, v11(v380(v71 * v38)))
    return v174(v381.v284([v70[v216] for v216 in v72[:v15]]))

def auc(v73, v74):
    """P(score on a positive > score on a negative), ties at half. Scale-free, so it survives a
    model whose probabilities are badly calibrated in absolute terms."""
    if not v73 or not v74:
        return v174('nan')
    v75 = v381.v290(v73)[:, None]
    v76 = v381.v290(v74)[None, :]
    return v174(v381.v284((v75 > v76) + 0.5 * (v75 == v76)))

def auc_z(v77, v78, v79):
    """How many standard errors the AUC sits above chance, under the null.

    An AUC of 0.560 on 14 absent against 56 present is 0.69 sigma - chance, wearing a number
    above 0.5. The first run to clear G_abstains_unknowable cleared it on exactly that, while
    the same measurement on the training tape stood at 4.73 sigma. A gate written as "> 0.5"
    cannot tell those apart, which is failure mode 3 arriving through the one examiner that
    was still absolute. The Hanley-McNeil null standard error is a property of the sample
    sizes, not a tuned constant, so the gate can require the effect to clear its own noise.
    """
    if v382.v291(v77) or v78 < 1 or v79 < 1:
        return v174('nan')
    return (v77 - 0.5) / v382.v292((v78 + v79 + 1) / (12.0 * v78 * v79))

def main() -> v11:
    v80 = v293.v175()
    v80.v176('--smoke', action='store_true')
    v80.v176('--train-steps', type=v11, default=0)
    v80.v176('--tape-period', type=v11, default=0)
    v80.v176('--addresses', type=v11, default=0)
    v80.v176('--min-mentions', type=v11, default=2)
    v80.v176('--min-evidence', type=v11, default=2, help='mentions left visible after hiding one; under 2 there is nothing to weigh')
    v80.v176('--min-candidates', type=v11, default=1, metavar='C', help='train only on evidence sets holding at least C distinct values. 1 keeps the natural mix, in which unanimous evidence dominates and the UNKNOWN label is irreducible noise; 2 trains only where there is something to weigh, at the cost of never seeing a settled address. The rate is reported either way.')
    v80.v176('--min-per-family', type=v11, default=8, help="ordering only. 280's packer moves this many of each family to the front and truncates nowhere unless --max-items is set, which this stage does not set - so the family mix here is the corpus's own.")
    v80.v176('--address-tau', type=v174, default=0.9)
    v80.v176('--address-overlap', type=v11, default=2)
    v80.v176('--soft-match', type=v174, default=0.0)
    v80.v176('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v80.v176('--abstain-reward', type=v174, default=0.75)
    v80.v176('--wrong-cost', type=v174, default=1.0)
    v80.v176('--lr-head', type=v174, default=0.001)
    v80.v176('--lr-upper', type=v174, default=3e-05)
    v80.v176('--no-hidden', action='store_true', help='ablation: features only, no reading of the evidence text')
    v80.v176('--frozen-trunk', action='store_true')
    v80.v176('--mind', choices=('head', 'relational'), default='head', help="head is 286's scorer over features (and optionally the reading). relational replaces it with an equivariant network over ranks and indicators of the mention graph - no text, no fingerprints, no absolute coordinate of any kind reaches the weights, so the lookup channel that killed every reading arm is closed by construction, not by regularisation. Trains in minutes: the trunk is never read.")
    v80.v176('--holdout', choices=('corpus', 'address'), default='corpus', help='what the held-out tape holds out. corpus splits the LINES 70/30, so the eval tape is built from different text - and every arm so far learns honesty and contested weighing on the training tape (margin AUC 0.865, contested 0.889) and reads chance across that split. That leaves two explanations and no way to tell them apart: the two corpus halves build differently shaped addresses, or the mind simply does not generalise. address splits the ADDRESSES of ONE corpus by a stable hash of the anchor, so the tape geometry is identical on both sides and only novelty of the subject remains. Transfer under address holdout but not under corpus holdout blames the tape; failure under both blames the mind.')
    v80.v176('--ctx-channel', choices=('cos', 'delta', 'both'), default='cos', help="what the context edge carries. cos is the plain ctx-fp cosine. delta REPLACES it with the composition residual: D = ctx_fp(whole) - norm(ctx_fp(before value) + ctx_fp(after value)), compared between mentions by cosine rank, plus |D| as a node rank. Word-level D is provably useless here - fp pools characters BEFORE the FF, so both f(whole) and mean f(parts) are functions of the character multiset and anagrams give bit-identical D. At CONTEXT level the per-chunk normalisation breaks that linearity, so D depends on the grouping, and it subtracts the bulk mean - which 283 named as the crowder (0.345 against 0.061). D is the high-pass part of the context channel. 'both' keeps cos as well and is the arm expected to lose: ink-slices already showed an EXTRA fp channel costing the dup win 0.773 -> 0.333 on 4,417 parameters and a thousand examples.")
    v80.v176('--ink-slices', action='store_true', help='two more edge channels: the RANK of the context-fp cosine taken before the value and after it, separately. fp is order-blind inside a word (277, closed) but slicing the CONTEXT at the value recovers order BETWEEN chunks, which is a different question and an open one: two mentions agreeing on the frame but not the aftermath are a paraphrase, agreeing on both are a duplicate. Ranks only - an absolute fp coordinate is an identity, and identity is what turned every reading arm into a lookup table.')
    v80.v176('--lie-dup', type=v11, default=1, metavar='K', help='lying tape with the forged mention repeated K times. Votes scale linearly in K by construction; a mind that models duplication should not. Reported as lying_tape_dup and gated as G_survives_duplicated_lie (informational, not in the verdict).')
    v80.v176('--read-anon', action='store_true', help='replace the subject and every candidate value in the READ text with fixed common words. The r8 arm showed the reading is used as a lookup key - identity is what memorises, phrasing is the only part that could transfer - so this is the arm that asks whether phrasing alone carries signal. Features keep identity either way; that split is the point.')
    v80.v176('--read-rank', type=v11, default=0, metavar='R', help='project the reading to R dimensions before the head. 0 is the wide 512-dim channel that the frozen arm showed memorising the training tape (AUC 0.986 there, 0.416 held out) with the trunk not learning at all - so the capacity, not the trunk, is what has to be tested.')
    v80.v176('--run-tag', type=v131, default='')
    v80.v176('--no-speedups', action='store_true', help='run the original unmemoised paths - they are byte-identical, and this is how that stays checkable')
    v81 = v80.v177()
    global LOG_PATH
    v82 = v81.v263 and f'_{v81.v263}' or ''
    v82 += '_nohid' if v81.v178 else ''
    v82 = '_smoke' + v82 if v81.v179 else v82
    v82 += '_frozen' if v81.v180 else ''
    v82 += f'_r{v81.v152}' if v81.v152 else ''
    v82 += '_anon' if v81.v181 else ''
    v82 += '_rel' if v81.v241 == 'relational' else ''
    v82 += f'_dup{v81.v27}' if v81.v27 > 1 else ''
    v82 += '_slices' if v81.v182 else ''
    v82 += f'_{v81.v268}' if v81.v268 != 'cos' else ''
    v82 += '_addrholdout' if v81.v202 == 'address' else ''
    v10 = v0 / f'_stage286_log{v82}.txt'
    v10.v272.v133(parents=True, exist_ok=True)
    v10.v183('', encoding='utf-8')
    v55 = v294.v55('cuda' if v294.v439.v383() else 'cpu')
    v83 = v295.v184(v4)
    v294.v185(v4)
    v84 = v186.v186()
    v85 = v81.v187 or (600 if v81.v179 else 6000)
    v86 = v81.v86 or (150 if v81.v179 else 500)
    v87 = v81.v188 or (60 if v81.v179 else 400)
    global MIN_EVIDENCE
    v8 = v81.v88
    v189(f'Stage286 evidence start {v474.v460(v475.v461).v368()} device={v55} steps={v85} no_hidden={v81.v178}')
    v157, v157, v190, v191 = v192()
    v53 = v296.v193(v131(v384.v297))
    v89 = v53.v194()
    v54 = v53.v298(v299) or 0
    v52 = v440.v385(v53, v190, v54, v89).v195(v55)
    v90 = v2 if v2.v300() else v1
    v51 = v386(v191, v89).v195(v55)
    v51.v196(v294.v387(v90, map_location=v55, weights_only=False)['model'])
    if v81.v180 or v81.v241 == 'relational':
        v51.v198()
        for v75 in v51.v199():
            v75.v303(False)
    else:
        v388.v301(v51, 'upper')
    v91 = v302.v197(v51)
    v92 = v386(v191, v89).v195(v55)
    v92.v196(v294.v387(v1, map_location=v55, weights_only=False)['model'])
    v92.v198()
    for v75 in v92.v199():
        v75.v303(False)
    if not v81.v200:
        v304(v305)
    v93 = v306(v92, v190, v55) if v81.v200 else v307(v306(v92, v190, v55))
    with v3.v273('r', encoding='utf-8', errors='ignore') as v134:
        v201 = v134.v308(4000000 if v81.v179 else 30000000)
    v94 = [v310.v309() for v310 in v201.v389('\n') if 80 <= v145(v310.v309()) <= 400]
    v95 = v11(0.7 * v145(v94))
    v96 = v94[:v95][:3000 if v81.v179 else 25000]
    v97 = v94[v95:][:1500 if v81.v179 else 12000]
    if v81.v202 == 'address':
        v97 = v96

    def anchor_side(v203: v131) -> v11:
        """Which side of an address holdout a subject falls on.

        Keyed on the anchor, because the fp group number in front of it is assigned by corpus
        order and changes on every resample - an address NAME is not stable across tapes, a
        subject is. A stable hash, so the split is identical in every resample and in the eval
        pack without any state being carried between them.
        """
        v204 = v203.v389(':', 1)[-1].v389('|')[0]
        return v11(v480.v476(v204.v398('utf-8')).v441(), 16) & 1

    def split_items(v75, v205):
        if v81.v202 == 'corpus':
            return v75
        v75 = v105(v75)
        v75['items'] = [v209 for v209 in v75['items'] if v466(v209['address']) == v205]
        return v75

    def new_pack(v206, v207):
        return v366.v311(v207, bank=v93, tok=v53, pad_id=v54, device=v55, rng=v206, n_addr=v87, min_mentions=v81.v265, tau=v81.v390, overlap=v81.v391, soft_match=v81.v392, min_per_family=v81.v267, addr_key=v81.v264)
    v17 = v208(v312(v83, v96), 0)
    if v145(v17['items']) < 8:
        v189('  too few items')
        return 1
    v98 = [v209 for v209 in v17['items'] if v145(v209['slots']) > v81.v88]
    v189(f"  tape: {v17['n_addresses']} addresses, {v17['n_slots']} slots | items {v435.v369(v105(v143((v216['kind'] for v216 in v17['items']))))} | usable for leave-one-out (>{v81.v88} mentions): {v145(v98)}")
    if not v98:
        v189('  no address can spare a mention: raise --addresses or lower --min-evidence')
        return 1
    v99 = 0 if v81.v178 else 2 * (v51.v442.v393 // 2)
    v100 = v313(v55, n_edge=3 + 2 * v11(v81.v182) + v11(v81.v268 == 'both'), n_node=2 + v11(v81.v268 != 'cos')) if v81.v241 == 'relational' else v314(v99, v55, read_rank=v81.v152)
    v101 = [v75 for v75 in v51.v199() if v75.v315]
    v102 = v294.v316.v210([{'params': v100.v199(), 'lr': v81.v443}] + ([{'params': v101, 'lr': v81.v467}] if v101 else []), weight_decay=0.01)
    v103: v105[v131, v294.v317] = {}
    v104: v105 = {}

    def anon_texts(v209, v58, v56):
        v211 = (v209['S'], v394(v58['cands']))
        v212 = v104.v318(v211)
        if v212 is None:
            v319 = v147([v209['S']] + v141(v58['cands']), key=v145, reverse=True)
            v212 = v444.v395('|'.v320((v444.v477(v75) for v75 in v319)), v444.v396)
            v104[v211] = v212
        v213 = v209['S'].v136()
        return [v212.v397(lambda v15: 'someone' if v15.v481(0).v136() == v213 else 'something', v372) for v372 in v56]

    def read_of(v56):
        if not v81.v180:
            return v398(v51, v52, v53, v54, v55, v56)
        v211 = '␟'.v320(v56)
        v156 = v103.v318(v211)
        if v156 is None:
            v156 = v398(v51, v52, v53, v54, v55, v56)
            if v156 is None:
                return None
            v156 = v156.v399()
            v103[v211] = v156
        return v156

    @v294.v217()
    def ink_coherence(v75, v58, v19):
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
        v321, v322 = (v58['ev_slots'], v58['vals'])
        v214 = v75.v323('_rel_ctx', {})
        for v215 in v144(v321):
            if v215 not in v214:
                v140 = v93.v445(v75['texts'][v215], exclude=v75['tape'].v139[v215])
                v214[v215] = v418.v468(v140, dim=-1) if v140 is not None else None
        v324, v129 = ([], [])
        for v216 in v238(v145(v321)):
            for v142 in v238(v216 + 1, v145(v321)):
                v77, v432 = (v214[v321[v216]], v214[v321[v142]])
                if v77 is None or v432 is None:
                    continue
                v140 = v174(v77 @ v432)
                v400 = (v322[v216] == v19) + (v322[v142] == v19)
                if v400 == 2:
                    v324.v149(v140)
                elif v400 == 1:
                    v129.v149(v140)
        if not v324 or not v129:
            return v174('nan')
        return v174(v381.v284(v324) - v381.v284(v129))

    def rel_inputs(v75, v58):
        """The graph, in ranks. Everything here comes from plumbing that already existed -
        ctx fingerprints, the inverted index, the probe counts - and leaves as an order
        statistic of THIS example, so a fresh tape re-ranks it all and nothing is worth
        memorising. Caches live on the pack and die with the resample."""
        v321, v322 = (v58['ev_slots'], v58['vals'])
        v38 = v145(v321)
        v214 = v75.v323('_rel_ctx', {})
        v218 = v75.v323('_rel_words', {})
        v325, v326 = (v75.v323('_rel_left', {}), v75.v323('_rel_right', {}))
        for v215 in v144(v321):
            if v215 not in v214:
                v372, v23 = (v75['texts'][v215], v75['tape'].v139[v215])
                v140 = v93.v445(v372, exclude=v23)
                v214[v215] = v418.v468(v140, dim=-1) if v140 is not None else None
                v218[v215] = v144(v275(v372, exclude=v23))
                v216 = v372.v136().v446(v23.v136())
                v447, v448 = (v372[:v216], v372[v216 + v145(v23):]) if v216 >= 0 else (v372, '')
                for v449, v450 in ((v325, v447), (v326, v448)):
                    v451 = v93.v445(v450, exclude=v23) if v450.v309() else None
                    v449[v215] = v418.v468(v451, dim=-1) if v451 is not None else None
                v401 = v75.v323('_rel_delta', {})
                v334 = v75.v323('_rel_dnorm', {})
                if v214[v215] is not None and v325[v215] is not None and (v326[v215] is not None):
                    v159 = v214[v215] - v418.v468(v325[v215] + v326[v215], dim=-1)
                    v334[v215] = v174(v159.v470())
                    v401[v215] = v418.v468(v159, dim=-1)
                else:
                    v401[v215], v334[v215] = (None, 0.0)
        v219 = v75.v318('_rel_median')
        if v219 is None:
            v327 = v147((v145(v23) for v23 in v75['postings'].v139()))
            v219 = v327[v145(v327) // 2] if v327 else 1
            v75['_rel_median'] = v219
        v166 = v294.v328(v38, v38)
        v220 = v294.v328(v38, v38)
        v221 = v294.v328(v38, v38)
        v329, v330 = (v294.v328(v38, v38), v294.v328(v38, v38))
        v222 = v294.v328(v38, v38)
        for v216 in v238(v38):
            for v142 in v238(v216 + 1, v38):
                v452, v453 = (v321[v216], v321[v142])
                v166[v216, v142] = v166[v142, v216] = v174(v322[v216] == v322[v142])
                if v214[v452] is not None and v214[v453] is not None:
                    v220[v216, v142] = v220[v142, v216] = v174(v214[v452] @ v214[v453])
                v402 = v137((1 for v371 in v218[v452] & v218[v453] if v145(v75['postings'].v318(v371, ())) < v219))
                v221[v216, v142] = v221[v142, v216] = v402 / v173(1, v135(v145(v218[v452]), v145(v218[v453])))
                if v81.v182:
                    for v332, v449 in ((v329, v325), (v330, v326)):
                        if v449[v452] is not None and v449[v453] is not None:
                            v332[v216, v142] = v332[v142, v216] = v174(v449[v452] @ v449[v453])
                if v81.v268 != 'cos':
                    v401 = v75.v323('_rel_delta', {})
                    if v401.v318(v452) is not None and v401.v318(v453) is not None:
                        v222[v216, v142] = v222[v142, v216] = v174(v401[v452] @ v401[v453])
        v223 = v294.v331(v38, v38, offset=1)

        def rank_norm(v332):
            """Rank the upper triangle by argsort, not by list.index per pair.

            The first version scanned a sorted list for every pair: O(n^4) in the mentions of
            one address. Invisible at n=4 and a wall at n=300, which is exactly the size this
            architecture claims to reach - a mind that does not grow with the knowledge is not
            scale-free if the graph builder is quartic. argsort-of-argsort is O(n^2 log n),
            and ties now share a rank instead of resolving by list order.
            """
            if v223.v374() == 0:
                return v332
            v23 = v332[v223[0], v223[1]]
            v72 = v23.v403()
            v206 = v294.v404(v72, dtype=v294.v412)
            v206[v72] = v294.v405(v145(v23), dtype=v294.v412)
            v406, v407 = v23.v408(return_inverse=True)
            if v145(v406) > 1:
                v409 = v294.v328(v145(v406)).v454(0, v407, v206, 'mean', include_self=False)
                v206 = v409[v407] / (v145(v23) - 1 if v145(v23) > 1 else 1)
            else:
                v206 = v294.v410(v206)
            v333 = v294.v410(v332)
            v333[v223[0], v223[1]] = v206
            v333[v223[1], v223[0]] = v206
            return v333
        v224 = [v166, v411(v222 if v81.v268 == 'delta' else v220), v411(v221)]
        if v81.v268 == 'both':
            v224.v149(v411(v222))
        if v81.v182:
            v224 += [v411(v329), v411(v330)]
        v165 = v294.v414(v224, -1).v195(v55)
        v225 = [v58['ext'].v318(v23, 0) for v23 in v322]
        v226 = v147(v144(v225))
        v227 = [[v58['cnt'][v322[v216]] / v38, v226.v276(v225[v216]) / v173(1, v145(v226) - 1)] for v216 in v238(v38)]
        if v81.v268 != 'cos':
            v334 = [v75.v323('_rel_dnorm', {}).v318(v215, 0.0) for v215 in v321]
            v335 = v147(v144(v334))
            for v216 in v238(v38):
                v227[v216].v149(v335.v276(v334[v216]) / v173(1, v145(v335) - 1))
        v228 = v294.v283(v227, dtype=v294.v412, device=v55)
        v168 = [v294.v283([v322[v216] == v140 for v216 in v238(v38)], device=v55) for v140 in v58['cands']]
        return (v165, v166.v419(-1).v195(v55), v228, v168)

    def forward(v75, v209, v58):
        if v81.v241 == 'relational':
            return v100(*v469(v75, v58))
        v47 = v294.v283(v58['rows'], dtype=v294.v412, device=v55)
        if v81.v178:
            return v100(v294.v328(0, device=v55), v47)
        v229 = []
        for v56 in v58['cand_texts']:
            if v81.v181:
                v56 = v455(v209, v58, v56)
            v156 = v413(v56)
            if v156 is None:
                return None
            v229.v149(v156.v174())
        return v100(v294.v414(v229), v47)
    v100.v230()
    v51.v230(not v81.v180 and v81.v241 != 'relational')
    v231, v232 = ([], [])
    v233, v234, v235, v236 = ([], [], 0, 0)

    def loo_pairs(v75):
        return [(v209, v415) for v209 in v75['items'] if v145(v209['slots']) > v8 for v415 in v209['slots']]
    v106 = v237(v17)
    for v107 in v238(1, v85 + 1):
        if (v107 - 1) % v86 == 0 and v107 > 1:
            v17 = v208(v312(v83, v96), 0)
            v106 = v237(v17)
            v103.v416()
            v104.v416()
        if not v106:
            v189('  no multi-mention addresses on this tape')
            return 1
        v209, v25 = v106[v83.v417(v145(v106))]
        v58 = v336(v17, v209, v25)
        if v58 is None:
            continue
        if v58['truncated']:
            v235 += 1
            continue
        if v145(v58['cands']) < v81.v266:
            v236 += 1
            continue
        v233.v149(v145(v58['cands']))
        v234.v149(v11(not v58['present']))
        v239 = v337(v17, v209, v58)
        if v239 is None:
            continue
        v240 = v418.v338(v239.v419(0), v294.v283([v58['target']], device=v55))
        v102.v339(set_to_none=True)
        v240.v340()
        v294.v50.v420.v341(v141(v100.v199()) + v101, 1.0)
        v102.v107()
        v232.v149(v174(v240))
        if v107 % v173(1, v85 // 8) == 0:
            v231.v149({'step': v107, 'loss': v174(v381.v284(v232[-200:])), 'kind': v209['kind'], 'n_cands': v145(v58['cands']), 'mean_cands_so_far': v289(v233)})
            v189(f"  step {v107}/{v85} loss={v381.v284(v232[-200:]):.4f} [{v209['kind']}] cands={v145(v58['cands'])} mean_cands={v289(v233):.2f}")
    v108 = {'n': v145(v233), 'sampling': 'pair_uniform', 'mean_candidates': v289(v233), 'target_unknown_rate': v289(v234), 'single_candidate_rate': v289([v11(v140 == 1) for v140 in v233]), 'truncated_skipped': v235, 'min_candidates_skipped': v236}
    v189(f'  TRAIN DIST {v435.v369(v108)}')
    v100.v198()
    v51.v198()
    v109 = v302.v197(v51)
    if v81.v241 == 'relational':
        v242 = {'mind': 'relational', 'params': v11(v137((v76.v374() for v76 in v100.v199())))}
    else:
        with v294.v217():
            v421, v154 = (v100.v134[0].v281, v100.v154)
            v242 = {'d_reading': v154, 'd_features': v7, 'reading': v174(v421[:, :v154].v470()) if v154 else 0.0, 'features': v174(v421[:, v154:].v470()), 'reading_per_dim': v174(v421[:, :v154].v470()) / v382.v292(v154) if v154 else 0.0, 'features_per_dim': v174(v421[:, v154:].v470()) / v382.v292(v7), 'reading_path_params': v154 * 128 + (v99 * v154 if v100.v153 is not None else 0), 'distinct_train_examples_est': v145(v106) * v173(1, v85 // v86)}
    v189(f'  HEAD NORMS {v435.v369(v242)}')

    @v294.v217()
    def leave_one_out(v75, v243: v262, v244: v11=1):
        v206 = v295.v184(v4 + (1 if v243 else 0))
        v245 = v75['n_slots']
        v342, v343, v344 = (v364(v141), v364(v141), v364(v141))
        v69, v345 = ([], [])
        v346, v347 = ([], [])
        v348, v349 = ([], [])
        v350, v351, v352 = ([], [], 0)
        v246 = {'absent': [], 'present': []}
        v247 = []
        v353, v354 = ([], [])
        v355, v356, v357, v358 = ([], [], [], [])
        v359, v360, v361, v362, v363 = ([], [], [], [], [])
        for v209 in v75['items']:
            if v145(v209['slots']) <= v8:
                continue
            for v25 in v209['slots']:
                v26 = None
                if v243:
                    v26 = v206.v417(v245)
                    if v26 in v144(v209['slots']):
                        continue
                v58 = v336(v75, v209, v25, lie_from=v26, lie_dup=v244 if v243 else 1)
                if v58 is None or not v58['cands']:
                    continue
                v239 = v337(v75, v209, v58)
                if v239 is None:
                    continue
                v422 = v294.v456(v239, dim=-1)
                v365 = v11(v239.v459())
                v423 = v58['cands'][v365] if v365 < v145(v58['cands']) else None
                v62 = v75['tape'].v139[v25]
                v352 += v11(v58['truncated'])
                v424 = v422[:v145(v58['cands'])]
                v425 = v11(v424.v459())
                v69.v149(v174(v424[v425]))
                v345.v149(v11(v58['cands'][v425] == v62))
                for v249, v77 in (('model', v423), ('votes', v472(v58)), ('return', v473(v75, v209, v58))):
                    v342[v249].v149(v11(v77 is not None))
                    if v77 is not None:
                        v343[v249].v149(v11(v77 == v62))
                    v344[v249].v149(v434(v77, v62, v81.v63, v81.v64))
                if v423 is not None:
                    v247.v149(v11(v423 == v62))
                    v457 = v471(v75, v58, v423)
                    if not v382.v291(v457):
                        (v353 if v423 == v62 else v354).v149(v457)
                v426 = v174(v422[-1])
                (v346 if not v58['present'] else v347).v149(v426)
                v427 = v174(v239[-1] - v239[:v145(v58['cands'])].v173())
                (v348 if not v58['present'] else v349).v149(v427)
                v246['absent' if not v58['present'] else 'present'].v149(v11(v423 is None))
                v350.v149(v145(v58['cands']))
                v351.v149(v11(not v58['present']))
                if v145(v58['cnt']) == 1:
                    v355.v149(v11(v423 is not None))
                    if v423 is not None:
                        v357.v149(v11(v423 == v62))
                else:
                    v356.v149(v11(v423 is not None))
                    if v423 is not None:
                        v358.v149(v11(v423 == v62))
                    v359.v149(v174(v424[v425]))
                    v360.v149(v11(v58['cands'][v425] == v62))
                    v458 = v472(v58)
                    v361.v149(v11(v458 is not None))
                    if v458 is not None:
                        v362.v149(v11(v458 == v62))
                    v363.v149(1.0 / v145(v58['cands']) if v58['present'] else 0.0)
        v129 = {'n': v145(v69), 'mean_candidates': v289(v350), 'target_unknown_rate': v289(v351), 'truncated_rate': v352 / v173(1, v145(v69)), 'silence_reward': v81.v63, 'break_even_accuracy': (v81.v63 + v81.v64) / (1.0 + v81.v64), 'model': v428(v342['model'], v343['model'], v344['model']), 'votes': v428(v342['votes'], v343['votes'], v344['votes']), 'return': v428(v342['return'], v343['return'], v344['return']), 'unknown_margin_auc': v429(v348, v349), 'unknown_margin_auc_z': v430(v429(v348, v349), v145(v348), v145(v349)), 'unknown_margin_when_absent': v289(v348), 'unknown_margin_when_present': v289(v349), 'unknown_auc': v429(v346, v347), 'unknown_auc_z': v430(v429(v346, v347), v145(v346), v145(v347)), 'p_unknown_when_absent': v289(v346), 'p_unknown_when_present': v289(v347), 'unknown_when_absent': v289(v246['absent']), 'unknown_when_present': v289(v246['present']), 'n_absent': v145(v346), 'n_present': v145(v347), 'model_reward': v289(v344['model']), 'votes_reward': v289(v344['votes']), 'return_reward': v289(v344['return']), 'model_acc_answered': v289(v247), 'model_abstain': 1.0 - v289(v342['model'])}
        v248 = {}
        for v249 in ('votes', 'return'):
            v142 = v129[v249]
            v248[v249] = {'judge_coverage': v142['coverage'], 'judge_accuracy': v142['accuracy'], 'model_accuracy_at_that_coverage': v431(v69, v345, v142['coverage']), 'n_compared': v142['n_answered'], 'judge_vacuous': v142['vacuous']}
        v129['matched_coverage'] = v248
        v129['selective_curve'] = {f'{v140:.2f}': v431(v69, v345, v140) for v140 in (0.1, 0.25, 0.5, 0.75, 1.0)}
        v129['ink_coherence'] = {'when_correct': v289(v353), 'when_wrong': v289(v354), 'separation': v289(v353) - v289(v354), 'n_correct': v145(v353), 'n_wrong': v145(v354)}
        v250 = v289(v361)
        v129['by_evidence_shape'] = {'unanimous_rate': v145(v355) / v173(1, v145(v69)), 'n_unanimous': v145(v355), 'n_contested': v145(v356), 'answer_rate_when_unanimous': v289(v355), 'answer_rate_when_contested': v289(v356), 'accuracy_when_unanimous': v289(v357), 'accuracy_when_contested': v289(v358), 'n_contested_answered': v11(v137(v356)), 'random_accuracy_contested': v289(v363), 'votes_coverage_contested': v250, 'votes_accuracy_contested': v289(v362), 'model_accuracy_at_votes_coverage_contested': v431(v359, v360, v250) if v359 else v174('nan')}
        return v129

    def beats_judges(v112):
        """Beat each judge where it chose to speak. Vacuous judges are neither beaten nor lost
        to - a scorer whose answered set is too small for any comparison against it to be
        significant is the abstain constant with a rounding error, and ranking against it is
        failure mode 3 with extra steps."""
        v251 = [v38 for v38 in ('votes', 'return') if not v112['matched_coverage'][v38]['judge_vacuous']]
        if not v251:
            return (None, [])
        v70 = []
        for v38 in v251:
            v15 = v112['matched_coverage'][v38]
            v77, v432 = (v15['model_accuracy_at_that_coverage'], v15['judge_accuracy'])
            v70.v149(not v382.v291(v77) and (not v382.v291(v432)) and (v77 >= v432))
        return (v433(v70), v251)

    @v294.v217()
    def exam(v75):
        """A consistency check, not an examiner.

        280 defines item["truth"] as the majority value of the address and None on a tie. With
        every mention visible votes_judge computes that same function of that same input, so
        here votes IS the label and its score is the ceiling: 1.0 wherever the corpus settled
        and the abstain reward on every tie, since answering a tie is wrong by construction.
        The old G_exam_matches_judges asked the mind to tie an oracle. What is left worth
        reporting is the gap to that ceiling.
        """
        v252 = v364(v141)
        v49 = {v134: v364(v141) for v134 in v366.v254}
        v253 = []
        for v209 in v75['items']:
            v58 = v336(v75, v209, None)
            if v58 is None or not v58['cands']:
                continue
            v239 = v337(v75, v209, v58)
            if v239 is None:
                continue
            v365 = v11(v239.v459())
            v342 = v58['cands'][v365] if v365 < v145(v58['cands']) else None
            v344 = v434(v342, v209['truth'], v81.v63, v81.v64)
            v252['r'].v149(v344)
            v252['votes_r'].v149(v434(v472(v58), v209['truth'], v81.v63, v81.v64))
            v252['return_r'].v149(v434(v473(v75, v209, v58), v209['truth'], v81.v63, v81.v64))
            v253.v149(1.0 if v209['truth'] is not None else v81.v63)
            v134 = v209['kind']
            v49[v134]['abstain'].v149(v11(v342 is None))
            v49[v134]['reward'].v149(v344)
            if v342 is not None:
                v49[v134]['acc'].v149(v11(v209['truth'] is not None and v342 == v209['truth']))
        v129 = {'n': v145(v252['r']), 'reward': v289(v252['r']), 'votes_reward': v289(v252['votes_r']), 'return_reward': v289(v252['return_r']), 'oracle_reward': v289(v253), 'gap_to_oracle': v289(v253) - v289(v252['r']), 'votes_is_the_label': True}
        for v134 in v366.v254:
            v129[v134] = {'n': v145(v49[v134]['abstain']), 'abstain': v289(v49[v134]['abstain']), 'acc_answered': v289(v49[v134]['acc']), 'reward': v289(v49[v134]['reward'])}
        return v129
    v110 = v255(v17, lie=False)
    v189(f'  LOO-TRAIN(control) {v435.v369(v110)}')
    v111 = v208(v312(v295.v184(v4 + 99), v97), 1)
    v112 = v255(v111, lie=False)
    v113 = v255(v111, lie=True)
    v114 = v255(v111, lie=True, dup=v81.v27) if v81.v27 > 1 else None
    v115 = v256(v111)
    v189(f'  LOO {v435.v369(v112)}')
    v189(f'  LIE {v435.v369(v113)}')
    if v114 is not None:
        v189(f'  LIE-DUP(x{v81.v27}) {v435.v369(v114)}')
    v189(f'  EXAM {v435.v369(v115)}')
    v116 = v91 == v109
    v117 = v112['n'] >= 4 * v9 and v112['mean_candidates'] > 1.0 and (v112['target_unknown_rate'] > 0.0) and (v108['n'] >= 4 * v9) and (v108['mean_candidates'] > 1.0)
    v257, v258 = v259(v112)
    v260, v261 = v259(v113)
    v118 = v113['matched_coverage']['votes']
    v119 = None if v118['judge_vacuous'] else v262(v118['model_accuracy_at_that_coverage'] >= v118['judge_accuracy'])
    v120 = v114['matched_coverage']['votes'] if v114 else None
    v121 = None if v120 is None or v120['judge_vacuous'] else v262(v120['model_accuracy_at_that_coverage'] >= v120['judge_accuracy'])
    v122 = v112['n_absent'] >= v9 and v112['n_present'] >= v9
    v123 = None if not v122 else v262(v112['unknown_margin_auc_z'] > 1.645)
    v124 = v112['by_evidence_shape']
    v125 = v262(v124['n_contested_answered'] >= v9 and (not v382.v291(v124['model_accuracy_at_votes_coverage_contested'])) and (not v382.v291(v124['votes_accuracy_contested'])) and (v124['model_accuracy_at_votes_coverage_contested'] >= v124['votes_accuracy_contested']) and (v124['model_accuracy_at_votes_coverage_contested'] > v124['random_accuracy_contested']))
    v126 = v262(v258)
    v127 = v262(v112['model']['reward'] > v81.v63)
    v128 = 'NO_TASK' if not v117 else 'JUDGES_VACUOUS' if not v126 else 'HONESTY_UNTESTED' if v123 is None else 'EVIDENCE_OK' if v116 and v257 and v123 and v119 and v125 else 'EVIDENCE_OK_UNANIMOUS_ONLY' if v116 and v257 and v123 and v119 else 'EVIDENCE_PARTIAL' if v116 and (v257 or v119) else 'EVIDENCE_NO'
    v129 = {'stage': 286, 'overall': v128, 'seed': v4, 'smoke': v81.v179, 'run_tag': v81.v263, 'no_hidden': v81.v178, 'addr_key': v81.v264, 'train_steps': v85, 'min_mentions': v81.v265, 'min_evidence': v81.v88, 'min_candidates': v81.v266, 'min_per_family': v81.v267, 'reward': {'correct': 1.0, 'wrong': -v81.v64, 'abstain': v81.v63}, 'min_answered_for_a_comparison': v9, 'gates': {'G_arc_enc_frozen': v116, 'G_task_exists': v117, 'G_judges_non_vacuous': v126, 'G_learns_evidence': v257, 'G_abstains_unknowable': v123, 'G_survives_lie': v119, 'G_weighs_contested': v125, 'G_survives_duplicated_lie': v121, 'G_beats_silence': v127}, 'judges_compared': v258, 'judges_compared_lie': v261, 'mind': v81.v241, 'lie_dup': v81.v27, 'ink_slices': v81.v182, 'ctx_channel': v81.v268, 'holdout': v81.v202, 'train_distribution': v108, 'head_input_norms': v242, 'leave_one_out': v112, 'leave_one_out_train_control': v110, 'lying_tape': v113, 'lying_tape_dup': v114, 'exam': v115, 'curve': v231, 'arc_enc_hash_before': v91, 'arc_enc_hash_after': v109, 'fp_version': v302.v367(), 'reference_280_baseline': {'held_out_reward': 0.704, 'acc_answered_all': 0.9}, 'reference_286_first_full_nohid': {'overall': 'EVIDENCE_NO', 'model_reward': 0.1, 'votes_reward': 0.014, 'return_reward': 0.754, 'unknown_when_absent': 0.0, 'unknown_when_present': 0.218, 'exam_reward': 0.777, 'exam_votes_reward': 0.932, 'note': 'scored by the examiners this file replaces; not comparable term by term'}, 'reference_286_ablation_pair_gen2': {'code': 'second-generation examiners, item-uniform sampler, no train control', 'nohid': {'overall': 'EVIDENCE_PARTIAL', 'loo_accuracy': 0.641, 'loo_coverage': 0.557, 'unknown_auc': 0.412, 'selective_10pct': 1.0}, 'hidden': {'overall': 'EVIDENCE_NO', 'loo_accuracy': 0.325, 'loo_coverage': 0.571, 'unknown_auc': 0.259, 'selective_10pct': 0.286}, 'note': "same tapes - judge numbers byte-identical between the runs. Reading halved accuracy at equal coverage and inverted the confidence ordering; the features-only ablation beat the arm carrying the stage's claim."}, 'note': "The direction of teaching is inverted: nobody demonstrates, the tape is the label. Hide one mention of an address, show the rest, require the hidden value - so there are as many examples as slots, no judge is imitated, and what the weights learn is what the judges were hand-written approximations of: weighing evidence. Votes and corroboration are features on the candidates now, every row reads the same schema so the UNKNOWN row can see the corroboration too, and each candidate is READ from the mentions that carry it. The examiners changed after the first full run, which measured four properties of the scoring rather than of the mind. The exam is no longer gated: with every mention visible, votes computes the majority and the label IS the majority, so votes was an oracle and 0.932 was the ceiling. Reward is no longer the comparison: at abstain 0.75 against wrong -1.0 nothing beats silence under 87.5% accuracy, so every scorer piles up at the constant and coverage and accuracy are now reported apart, with a judge under 10% coverage marked vacuous instead of being ranked against. The mind is compared to each judge at THAT JUDGE'S coverage, ranked by its own confidence. And honesty is measured in the distribution: whether the hidden value is visible is a property of the hidden slot, and where the evidence is unanimous the UNKNOWN row is the same input under both labels, so an argmax can never choose it against a 20% base rate - what the evidence can be asked is whether it moves P(UNKNOWN), which is an AUC. The lying tape stays the sharpest test of the concept: a forged mention has no external corroboration and a mind that weighs evidence should discount it, while votes by construction cannot. The address itself is still 280's plumbing - this stage is about weighing WITHIN the evidence set, so the ceiling here is what a correct address makes visible.", 'timestamp': v474.v460(v475.v461).v368(), 'wall_s': v186.v186() - v84}
    v0.v133(parents=True, exist_ok=True)
    (v0 / f'stage286_decision{v82}.json').v183(v435.v369(v129, indent=2), encoding='utf-8')
    (v0 / f'stage286_mini{v82}.md').v183(f"# Stage 286 the tape as its own label{(' (no hidden)' if v81.v178 else '')}\n\n**{v128}**{(' · SMOKE' if v81.v179 else '')}\n\n| leave-one-out | coverage | accuracy | reward |\n|---|---:|---:|---:|\n" + ''.v320((f"| {v38}{(' (vacuous)' if v112[v38]['vacuous'] else '')} | {v112[v38]['coverage']:.2f} | {v112[v38]['accuracy']:.2f} | {v112[v38]['reward']:.3f} |\n" for v38 in ('model', 'votes', 'return'))) + f"\n- unconditional silence is worth {v81.v63:.2f}; answering pays only above accuracy {v112['break_even_accuracy']:.3f}\n" + ''.v320((f"- at {v38}'s coverage {v112['matched_coverage'][v38]['judge_coverage']:.2f}: model {v112['matched_coverage'][v38]['model_accuracy_at_that_coverage']:.2f} vs judge {v112['matched_coverage'][v38]['judge_accuracy']:.2f}{(' — judge vacuous, not compared' if v112['matched_coverage'][v38]['judge_vacuous'] else '')}\n" for v38 in ('votes', 'return'))) + f"- control: unknown AUC {v110['unknown_auc']:.3f} on the last training tape against {v112['unknown_auc']:.3f} held out; the pair says whether a failure is shift between the tapes or not a function of the features at all\n" + f"- contested evidence only ({v124['n_contested']} sets): model answers {v124['answer_rate_when_contested']:.2f} of them at {v124['accuracy_when_contested']:.2f}, against {v124['answer_rate_when_unanimous']:.2f} at {v124['accuracy_when_unanimous']:.2f} on unanimous ones; at votes' contested coverage {v124['votes_coverage_contested']:.2f} model {v124['model_accuracy_at_votes_coverage_contested']:.2f} vs votes {v124['votes_accuracy_contested']:.2f}\n" + f"- UNKNOWN's margin over the best candidate separates absent from present at AUC **{v112['unknown_margin_auc']:.3f}** ({v112['unknown_margin_auc_z']:+.2f} sigma); the diluted probability form says {v112['unknown_auc']:.3f} ({v112['p_unknown_when_absent']:.3f} vs {v112['p_unknown_when_present']:.3f}); at the argmax {v112['unknown_when_absent']:.3f} vs {v112['unknown_when_present']:.3f}\n- training saw {v108['mean_candidates']:.2f} candidates on average, {v108['single_candidate_rate']:.2f} of examples unanimous\n- exam is a consistency check only: oracle {v115['oracle_reward']:.3f}, model {v115['reward']:.3f}, gap {v115['gap_to_oracle']:.3f}\n\n## Gates\n\n" + ''.v320((f'- {v478}: **{v479}**\n' for v478, v479 in v129['gates'].v288())), encoding='utf-8')
    v189(v435.v369({'overall': v128, 'gates': v129['gates']}, indent=2))
    return 0
if v130 == '__main__':
    raise v269(v370())