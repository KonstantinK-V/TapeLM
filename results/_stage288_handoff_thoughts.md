# TapeLM / Inprint — handoff for a fresh session

> **ПОМЕТКА (мысли):** §6b *The path past one-slot answers* — дизайн/мысли из разговора,
> **не измерялось, не стадия к сборке**. Остальной handoff — измеренные факты, как в шапке.
> Источник: `HANDOFF(9).md` + патч `0001handoffrecordthepathpastoneslotanswersbefore.patch`.

Written to replace a very long conversation. Read this, attach the four files listed at the
bottom, and you have everything the previous session had. Every number below was measured;
nothing here is a plan or a hope — **except §6b, which is explicitly design-only thoughts**.

---

## 1. The concept

Separate **knowledge** from **mind**. Facts live on a **tape** — an append-only store of
mentions extracted from raw text. The mind is a small set of weights that decides *how to
read* the tape. It never holds a fact.

Two consequences the project takes seriously:

- The mind can never be better than its tape. If the tape is wrong the answer is wrong, and
  that is **fidelity, not failure**. Improvement comes from writing, not from tuning weights.
- The mind must not grow with the knowledge. 7,564,290 parameters addressed 91 slots and 4,730
  slots identically in stage 283.

**The invariant, in one line:**

> Whatever decides may not be approximate. Whatever is trained may not hold facts.

**How separation is proven today:**

| proof | mechanism |
|---|---|
| `G_arc_enc_frozen` | `arc_enc_hash(model)` identical before and after training |
| `G_answer_is_slot` | the answer is a slot index, so it cannot be generated |
| tape resampling | the tape is rebuilt every ~200 steps during training |
| held-out tape | scoring uses a tape built from a disjoint 30% of the corpus |

The second one dies as soon as answers stop being indices. Its only label-free replacement is
the **return path** (§5), which is why that check is kept alive even though it lost its own
ablation.

---

## 2. Architecture as it stands

**Fingerprints ("ink").** `fp(w) = normalize(arc_enc(chars of w))` on a frozen encoder
(`checkpoints/stage191_p1_curve.pt`), d=256. `ArcEncoder` mean-pools character embeddings
*before* the feed-forward, so **fp is order-blind** — a normalized character histogram.
Anagrams collide bit-exactly. Participation ratio 26.06 at d=256, i.e. ~26 effective dimensions.

**Write (stage 279).** Sentences yield assertions `(anchor, relation words, value, context)`.
The structural rule: **at least one token must stand between the two names** — language-free,
and it alone lifted the teacher ceiling from −0.189 to +0.483 on raw text. Assertions merge
into addresses when their keys agree **and** their contexts share words. Three write actions:
`WRITE` / `CONFIRM` / `DISPUTE`. Repeats become counters; support count is a free reliability
estimate.

**Address key.** Was `norm(fp(anchor) + ctx_fp(context))`, one point per address (mean of
members). Now: **a set of member keys, scored on two channels** —
`score = min(cos_anchor, cos_context)` — because averaging was the thing filling the space
(§4).

**Read (stage 280).** Word votes over an inverted index (idf-weighted), then a subject filter,
then an fp "hop" when the words go silent. Families are **not declared**: an address with one
value is `clean`, a leader beating the runner-up is `decidable`, a tie at the top is a `tie`.
Which questions exist is a property of the corpus.

**Reward:** correct +1.0, wrong −1.0, abstain +0.75, read −0.02 per read. Abstention is
*cheap on purpose* — the project values calibrated silence over coverage.

**Policy:** BC from an executable teacher, then RL with a value baseline `V(s)` on `[h, feats]`
and advantage `R − V(s_t)`, with the BC anchor kept on during RL.

---

## 3. The baseline — measure everything against this

`_stage280_raw_exam.py --run-tag casefold`, held-out tape, **all ten gates true**:

| | |
|---|---:|
| `reward_total` | **0.704** |
| teacher ceiling | 0.625 |
| `acc_answered_all` | **0.900** |
| `coverage_all` | 0.526 |
| tie abstain | **0.85** |
| clean / decidable `acc_answered` | **1.0 / 1.0** |
| `retrieval_precision` | 0.823 |

The strength here is **calibration, not accuracy**: it abstains on 85% of genuinely
contradictory addresses with zero wrong answers there, and is perfect where the corpus settled.
Note the task is "pick from ≤7 retrieved slots or stay silent" — not open QA. Do not compare
0.900 to QA benchmarks.

---

## 4. Closed questions (do not redo these)

**The ink is closed as a search channel (277).** Whitening, n-gram variants, MaxSim, and
dimension changes were all measured. `dims_95pct = 48`, participation ratio 26.06, nearest-other
cosine p50 0.913. Width is not the bottleneck; order-blindness is.

**Votes are the τ→1 limit of fingerprint scoring.** `score = Σ idf(w)·max_c relu(cos−τ)^p`
reduces exactly to word votes at τ→1 because `content()` dedups.

**The register bug was the whole story of stages 280–282.** Addresses are lowercased at write
time (`name.lower()`); contexts are stored as they appear in the corpus. Every "does this slot
mention the subject" test is a substring test between the two, so it was **False for every
capitalised name** — that is, every entity. `'tommy lawton' in 'Tommy Lawton was born'` → False.
Fixing it moved retrieval precision 0.33 → 0.80 and accounted for nearly all of the apparent
gain attributed to new actions. **The pack now carries `texts_lc` and both readers use it.**

**Stage 282's three additions do not pay** (measured, held-out):

| arm | reward |
|---|---:|
| 280 baseline, no additions | **0.704** |
| 282 with the return-path probe | 0.594 |
| 282 without it | 0.564 |

- *Typed silence* (`STOP_CONFLICT` / `STOP_UNKNOWN`) cost abstention: tie abstain 0.85 → 0.70,
  and `STOP_UNKNOWN` never fired once on either tape.
- *Editable query* (`DROP_i`/`ADD_i`) never fired at all — the teacher only edits when the
  candidate list is empty, and votes always return k slots, so the branch is unreachable at any
  corpus size.
- *Return path* helped slightly (+0.03 reward, `acc_when_probe_hit` 0.82 vs miss 0.67) but hurt
  the teacher when allowed to break ties. It now lives in 280 as an **eval-time observer**
  (`return_path_when_correct` vs `return_path_when_wrong`), invisible to the policy.

**Two teachers gave arbitration, not understanding (285).** Votes and return-path disagree on
~50% of held-out items, but the disagreement is **one-sided**: votes accuracy 0.0 on contested
items, return 0.75. "Learn whom to trust" collapses into "trust the better heuristic". Also,
skipping contested items in BC destroyed abstention. Verdict `NO_DISAGREEMENT`.

**Scale holds on the tape side (283, 284).** Ladder 91 → 792 → 4,730 slots (52× growth), one
configuration, mind constant:

| | 4M | 30M | 120M |
|---|---:|---:|---:|
| retrieval precision | 0.796 | 0.740 | 0.664 |
| return-path separation | 0.96 | 0.986 | 0.953 |
| `foreign_member_rate` (real merge damage) | 0.000 | 0.019 | 0.052 → **0.0086** after two channels |
| address crowding, summed key | 0.00 | 0.195 | 0.345 |
| address crowding, **two channels** | 0.00 | 0.008 | **0.061** |
| bare anchors alone | 0.00 | 0.000 | 0.018 |
| anchor growth exponent | | | **0.923** |

Two findings that reversed my predictions:

1. **Context is the crowder, not the protector.** Bare anchors barely collide (0.018); adding a
   context fingerprint pushes crowding to 0.345, because every English sentence has much the
   same character statistics. Hence two channels scored by their minimum.
2. **Names do not run out.** Growth exponent 0.923 — nearly linear in corpus size at this scale,
   so vocabulary saturation will not save the address space.

---

## 5. Recurring failure modes — check these first on any new result

These caused most of the wasted runs. They repeat.

1. **A fixed window over a growing tape.** `top-k` retrieval lost the corroborating mention
   (return-path true rate 1.00 → 0.62) even though mentions per address *rose*. The fix is
   never a bigger k: scan the posting list of the rarest word, or cut at the largest score gap.
2. **Absolute thresholds in a space whose occupancy grows.** `tau=0.90`, `k=7`, `overlap=2`,
   `hop_min` all need re-picking at new N — that is re-fitting, not scaling. 284 replaces them
   with rank rules: **mutual nearest neighbour** for merging (no threshold at all), **largest
   gap** for reading, **rarer than the tape's median** for shared words.
3. **Vacuous gates.** A gate that passes through an `isnan` escape, or because both sides did
   nothing, proves nothing. This happened three times: a teacher abstaining on everything scored
   the abstain reward and cleared `G_teacher_usable`; a probe that always hit left the miss
   bucket empty and cleared `G_probe_filters`; a policy matching a silent teacher by being
   silent cleared `G_reaches_teacher`. **Always check the denominator.**
4. **Silent fallbacks hide version skew.** `pack.get("postings_probe", pack["postings"])` made a
   missing index indistinguishable from a real miss. Raise instead.
5. **Degenerate tasks that still produce numbers.** With `min_mentions=2`, leave-one-out leaves
   one mention — the candidate list has one entry and the task is copying, not weighing. Gate
   the task's existence *before* gating performance (`G_task_exists`).
6. **Case folding.** See §4. Any string comparison between an address and a context needs
   `texts_lc`.
7. **Trunk window.** `nn.Embedding(MAX_ARCS=64, d)` — truncate token ids to `[-64:]` or the run
   dies on the device with `srcIndex < srcSelectDimSize`.
8. **Averaging collapses the space.** Twice: characters into a word (fp, unfixable without
   n-grams) and mentions into an address key (fixed — sets).
9. **A baseline computed from the label.** 286's exam defined `truth` as the majority value and
   then scored `votes_judge`, which *is* that function of that input — so votes scored the
   ceiling (0.9324 = 21+6+10·0.75 over 37) and the gate asked the mind to tie an oracle. Before
   comparing to a baseline, write down what the baseline would score if it were perfect. If that
   equals what it did score, the baseline is the label.
10. **A reward that compresses every scorer onto one constant.** With abstain +0.75 and wrong
    −1.0, answering pays more than silence only above **87.5%** accuracy, so on a task whose
    reachable accuracy is ~0.5 every scorer lands near 0.75 and the ordering is decided by a
    handful of items. 286's `return_reward` 0.7536 on n=70 is consistent with answering 1 item
    or 65 of them — coverage was simply not in the number that was gating. **Report coverage and
    accuracy apart from reward, and compare selective scorers at matched coverage.**
11. **The trainer and the examiner weight the data differently.** 286 sampled an *item*
    uniformly, then a slot; leave-one-out enumerates every slot of every item, weighting an
    address by its mention count. Training saw 1.95 candidates on average, scoring saw 3.56 —
    and the conditional P(UNKNOWN | features) *flips sign* between those regimes, so the head
    learned the training statistic truthfully and scored 2.8σ below chance. Sample what the
    examiner enumerates, and always compare `train_distribution` to the eval's own shape.
12. **A target that is not a function of the input.** Whether a hidden mention's value appears
    among the visible ones is a property of the *hidden* slot. Where evidence is unanimous the
    feature row is identical under both labels, so the ~20% absent rate is irreducible noise and
    an argmax can never choose `UNKNOWN` against it — `unknown_when_absent = 0.000` was correct
    behaviour, not dishonesty. Measure such things in the distribution (AUC of `P(UNKNOWN)`),
    never at the argmax.

---

## 6. The open front: stage 286

**The circularity this fixes.** A hand-written heuristic is called the teacher, the mind is
trained to resemble it, success is measured as resemblance (`G_reaches_teacher` says so
literally). Every judgment lived in code; the weights only routed to it. The one thing that was
never imitation is the reward, which comes from the tape's verdict — which is why the policy
ends up above its teacher at all.

**The inversion.** Nobody demonstrates. The tape is the label:

```
L = -log P( value(s) | evidence of the address, with s hidden )
```

Hide one mention, show the rest, require the hidden value.

- as many examples as slots, not a quota of questions;
- no judge is imitated because no judge is asked;
- votes and external corroboration become **features** on candidates;
- each candidate is **read from its own mentions** (a single pooled vector is constant across
  the softmax and cannot say which candidate the text supports);
- `UNKNOWN` is a candidate whose target is real whenever the visible evidence cannot supply the
  answer — honesty is trained, not assumed;
- the answer is the argmax: no threshold constants anywhere.

**Why the invariant survives:** the mind predicts a value *of the tape* given evidence *from the
tape*, the tape resamples during training, and scoring runs on a disjoint corpus. Hiding facts
in parameters has no payoff, and the leak test (delete the evidence, the answer must die) stays
available.

**Gates, in order:**

| gate | meaning |
|---|---|
| `G_task_exists` | ≥20 examples, >1 candidate on average **in training as well as in scoring**, some unknowable target — else `NO_TASK` |
| `G_judges_non_vacuous` | at least one judge answers ≥10% of items — else `JUDGES_VACUOUS` |
| `G_learns_evidence` | beat each non-vacuous judge **at that judge's own coverage**, ranking by the mind's confidence |
| `G_abstains_unknowable` | `P(UNKNOWN)` ranks absent targets above present ones, AUC > 0.5 |
| `G_survives_lie` | keep beating votes at matched coverage when one visible mention is forged |
| `G_beats_silence` | reported, not required: reward above the abstain constant needs accuracy > 87.5% |

**The lying tape** is the sharpest test of the concept: a forged mention has no external
corroboration, so a mind that weighs evidence should discount it, while votes by construction
cannot.

**State.** First smoke: `EVIDENCE_NO`, measuring nothing — `n_cands` was 1 at every step
(failure mode 5). `--min-evidence 2` and `G_task_exists` were added in response.

First **full `--no-hidden` run** (6000 steps, `results/stage286_decision_nohid.json`):
`EVIDENCE_NO`, LOO n=70, model 0.100 / votes 0.014 / return 0.754, `unknown_when_absent` 0.000
against `unknown_when_present` 0.218, exam model 0.777 against votes 0.932. Four of those are
properties of the examiners, not of the mind — see failure modes 9, 10 and 11, plus:

- the head could not express the return judge's rule while holding the return judge's feature,
  because the `UNKNOWN` row carried a *different* feature schema from the candidate rows and no
  view of external support at all. Every row now shares one schema: four own numbers plus five
  example-level numbers broadcast to all rows.
- training ran in the copying regime (6 of 8 logged steps at `n_cands` 1) while scoring did not
  (3.56). The training distribution is now measured and gated, and `--min-candidates` exists.
- a target present in the evidence but outside `most_common(8)` was labelled `UNKNOWN` — label
  noise pointing the wrong way on the very axis gate 3 measures. Presence is now decided against
  the full counter and truncated examples are dropped from training and counted.

**First hidden run under the new examiners** (6000 steps, wall 8607 s): `EVIDENCE_NO`, and this
time the numbers are about the mind. `unknown_auc` **0.259** — 2.8σ *below* chance on 14×56
pairs — and the selective curve **rises** with coverage (0.286 at 10% → 0.443 at 100%):
confidence anti-correlated with correctness. That is a sign flip, not absence of signal. The
mechanism is failure mode 11: item-uniform training saw 1.95 candidates while slot-enumerating
scoring saw 3.56, and where UNKNOWN sits in the composition flips between the regimes (at ~2
mentions absent targets live in *unanimous* evidence — hide the minority of {A,A,B}; at ~4 in
*contested* evidence — many distinct values make every hidden slot absent). The head learned
the training statistic truthfully: P(UNKNOWN | unanimous) pins at 0.52–0.63 from the training
counts, and the exam's clean abstain of 0.571 is that same number read back.

Fixed by sampling (item, slot) pairs uniformly — the examiner's own distribution — plus a
leave-one-out **control on the last training tape** (`leave_one_out_train_control`). The next
runs read as a 2×2:

| | train-tape AUC > 0.5 | train-tape AUC ≤ 0.5 |
|---|---|---|
| held-out AUC > 0.5 | fixed | — |
| held-out AUC ≤ 0.5 | shift between corpus halves | target not a function of the features |

**The ablation pair resolved it — and the composition prediction failed.** The nohid arm had
run two hours *before* the hidden arm on the same second-generation code; judge numbers are
byte-identical between the two JSONs, so tapes and examples were too. A controlled pair:

| same code, same tapes | nohid (features only) | hidden (+ reading) |
|---|---|---|
| verdict | **EVIDENCE_PARTIAL** | EVIDENCE_NO |
| LOO accuracy @ coverage | **0.641 @ 0.557** | 0.325 @ 0.571 |
| selective curve | 1.00 → 0.43 **falling** | 0.29 → 0.44 rising |
| `unknown_auc` | 0.412 (1.0σ — noise) | 0.259 (2.8σ — inverted) |
| vs votes / return at their coverage | 0.431 vs 0.431 · **1.000 vs 0.889** (n=9) | loses both |
| lying tape at votes' coverage | **0.487 vs 0.462** | loses |
| exam gap to oracle | 0.108 | 0.345 |

The prediction that the composition flip *must* show in the features-only arm was wrong: nine
numbers trained on the wrong mix still rank correctly on the right one (`G_learns_evidence`
and `G_survives_lie` both true — the first non-vacuous pass of either). The inversion is a
property of **reading**, on identical tapes. Two hypotheses fit "confidently wrong held-out,
only when reading": **(a)** the trunk fine-tunes at `lr_upper` over 6000 steps on resamples of
the same 25k lines and memorises subject→value — facts moving into weights, the leak the
invariant forbids; **(b)** the reading channel (~512 dims vs 9) drowns the features and the
head overfits reading noise. `--frozen-trunk` separates them: reading stays, trunk learning
stops — (a) predicts recovery to nohid, (b) predicts no change. `head_input_norms` in the JSON
now reports the head's per-dim norm split between the channels.

**First pass: nohid under the pair-uniform sampler → `EVIDENCE_OK`.** Model 0.84 accuracy at
0.357 coverage against votes 0.431 at 0.829; beats votes *and* return at their own coverage;
on the lying tape it answers 15% of items and is **never wrong**. First non-vacuous passes of
`G_learns_evidence` and `G_survives_lie` the stage has produced. Three cautions, all measured:

- **Both pre-run predictions failed.** Training candidates were to reach ~3.5 and the unknown
  rate ~0.2; they moved 1.95 → 2.07 and 0.348 → 0.377. The sampler was not the gap. The new
  control locates it: leave-one-out *on the training tape* sees 2.15 candidates, not 3.56 —
  the corpus halves build differently shaped addresses (train tape: 104 addresses, only 22 can
  spare a mention). The sampler fix was still right in principle, just not what it was sold as.
- **`G_abstains_unknowable` passed on a coin.** `unknown_auc` 0.560 held-out is **0.69σ** on
  14×56 — chance. The same measurement on the training tape is 0.809 = **4.73σ**. The honesty
  signal is strong where the head was fit and does not survive the tape shift. Gate now
  requires `unknown_auc_z > 1.645` (the null's own one-sided 95% point, not a fitted constant),
  and `auc_z` is reported everywhere.
- **What was learned may be "answer iff unanimous".** On the exam this arm abstained on 100% of
  decidable and 100% of tie while answering 100% of clean, perfectly — one threshold on one
  count feature. On the training tape coverage 0.4167 sits beside a unanimous rate of 0.4325.
  `by_evidence_shape` now splits every rate by unanimous vs contested, and
  **`G_weighs_contested`** runs the matched-coverage comparison on contested sets alone, where
  copying cannot answer. Passing everything except it is now its own verdict,
  `EVIDENCE_OK_UNANIMOUS_ONLY` — selective copying is real and learned, but it is not weighing.

**Frozen-trunk resolved (a) vs (b): the trunk is innocent, the capacity is not.** `EVIDENCE_NO`,
worse than the unfrozen arm on every held-out number, with the trunk not learning a single
parameter. The two leave-one-out passes:

| frozen, hidden | train control | held out |
|---|---|---|
| `unknown_auc` | 0.986 (**+7.44σ**) | 0.416 (−0.97σ) |
| accuracy @ coverage | 0.897 @ 0.690 | 0.377 @ 0.757 |
| selective curve | 1.00 / 1.00 / 1.00 | flat 0.43 / 0.44 / 0.49 |
| vs votes at its coverage | 0.897 vs 0.690 | 0.414 vs 0.431 |

Fitting the training tape at AUC 0.986 and landing below chance on a disjoint one, with a flat
confidence curve, is memorisation. The arithmetic was available in advance: **65,536 weights on
the reading path against ~1,000 distinct training examples** (84 LOO pairs × 12 tapes) = 65
params/example, against **1.14** for the features-only head — which is the arm that passed.
`head_input_norms` agrees from the other side: reading carries 3.1× the features' total norm.

This does not say reading is useless — it says the question was never asked at a capacity the
data can pay for. `--read-rank R` now projects the reading to R dims before the head, and a
frozen trunk's encodes are cached (same text → same vector, forever), which was most of that
arm's hour. Next, cheapest first: `--frozen-trunk --read-rank 8` and `--read-rank 16`. If a
narrow reading beats nohid, reading carries signal; if it merely ties, it carries none at this
tape size and the stage's verdict is that judgment lives in the counters.

**Rank 8 killed the capacity story too.** `--frozen-trunk --read-rank 8`: the reading path
shrank **12.8×** (65,536 → 5,120 trainable params, projection included) and the train fit did
not move — 0.897 @ 0.690 identical to the digit, control AUC 0.986 → 0.996. Held out: 0.353
accuracy, `unknown_auc` 0.302 (−2.3σ, inverted again), lying tape 0.143. A capacity cut that
changes nothing means capacity was never binding: **the reading is a lookup key** — a text
with its names in it is a unique identifier, and a lookup table over identifiers is cheap at
any width. (a) trunk ✝, (b) capacity ✝; what remains is (c) identity in the input.
`--read-anon` folds subject and candidate values in the read text to constant words, leaving
phrasing — the only part that could transfer. Train control may still memorise (the sentence
remains a unique string); the question is held-out only. Next run:
`--frozen-trunk --read-rank 8 --read-anon` (cheap, cache applies). Beats nohid → phrasing
carries signal; ties → nohid *is* the model; still inverted → drop the channel, verdict
"judgment lives in the counters".

Also: `G_weighs_contested` passed on that run at 0.152 vs votes' 0.121 — two scorers under the
floor, ranked solemnly. The gate now also requires beating `random_accuracy_contested`
(expected accuracy of a uniform pick among the candidates), the null the comparison forgot.

**Anon resolved the fork: the raw-reading channel is dropped.** Inversion gone (0.302/−2.3σ →
0.579/+0.9σ) — identity was the poison. But no payoff: LOO parity with nohid (matched vs votes
0.431 = 0.431), a *loss* on the lying tape (0.410 vs 0.462 where nohid won), train control
still memorised at 0.997 (an anonymised sentence is still a unique string). One genuine
positive: the first legitimate `G_weighs_contested` pass — contested **0.364 > random 0.268 >
votes 0.121** — structure, once names are gone, carries something.

**The wall, named.** Everything trainable over an input containing identity becomes a lookup
table: trunk frozen → collapse stays; capacity ÷12.8 → fit unchanged; names removed → poison
gone, payoff absent. The only head that ever passed is nine *relational* statistics — they
transfer precisely because they cannot name anyone.

**Stage 287 inside 286: `--mind relational`.** The design rule instead of the post-mortem: an
equivariant network over the mention graph whose every input is an indicator or a
within-example **rank** — same-value; rank of ctx-fp cosine; rank of shared-rare-word overlap
(rare = rarer than the tape's median, 284's rule); rank of external corroboration; value's
share of the evidence. No text, fingerprint, or absolute coordinate reaches the weights: the
lookup channel is closed **by construction**. Equivariance is "the mind does not grow with the
knowledge" as architecture — the same ~5k weights read 3 mentions or 300, which is also the
scale story. It can express what counters cannot: two near-identical contexts are *one witness
repeated* — `--lie-dup K` is the exam (votes scale linearly in K by construction;
`G_survives_duplicated_lie`, informational). Pre-registered falsifiers: train-control AUC must
**fall** from ~1.0, else the graph itself leaks identity and the class failed its reason to
exist; matched coverage must reach nohid's (its features are derivable from the graph); any
win must show on contested or the duplicated lie. Runs in minutes — the trunk is never read:

```
python _stage286_evidence.py --train-steps 6000 --min-mentions 2 --mind relational --lie-dup 3
```

**The relational mind works, and produced the first counter-impossible result.** 4,417
parameters, no text or absolute coordinate in the weights. `EVIDENCE_PARTIAL`; the three
pre-registered falsifiers split cleanly:

- ✅ **train-control AUC fell** 0.997 → 0.841 — the graph is not the lookup table the text was.
- ✅ **matched coverage reached nohid** exactly: 0.431 vs votes 0.431, **1.000** vs return 0.889.
- ✅ **the duplicated lie** — the result. Forged mention ×3: votes accuracy **0.000** on 40
  answered (reward −0.398), *by construction* — a counter cannot tell three copies of one
  witness from three witnesses. The relational mind: **0.773** on 22 answered (reward 0.676),
  **0.450 vs 0.000** at votes' own coverage. Mechanism confirmed, not luck: it answers 36% of
  duplicated-lie sets versus 4% of ordinary contested ones — it acts where the graph says
  "one witness, repeated".
- ❌ **`G_weighs_contested`**: 2 contested answers held out, accuracy 0.0. Caveat before calling
  it a defect — on that subset votes scores 0.121 against a **random floor of 0.268**, so
  majority reasoning is worse than chance there and silence is calibrated; but two items is not
  evidence of weighing either.

**Failure mode 13: a softmax probability is diluted by however many rows share it.**
`P(UNKNOWN)` competes with |cands| candidate rows, and absent targets live in contested sets
with *more* candidates — so the honesty measure was biased against its own hypothesis. This run
exposed it: at the argmax UNKNOWN won 0.714 of absent vs 0.589 of present (right ordering),
while mean probabilities were 0.478 vs 0.484 and AUC read 0.482. Gate now reads
`unknown_margin_auc` — the logit margin over the best candidate, which is what the argmax
compares and is invariant to row count.

Next: rerun `--mind relational --lie-dup 3` under the margin gate (minutes), and a `--lie-dup`
ladder (1/2/3/5) — the duplication gap is the one axis where this mind provably leaves the
counting family, so it deserves a curve rather than a point.

**Address holdout answered the question.** Honesty crosses SUBJECTS (margin AUC 0.754,
+2.6σ, train control 0.896) while it failed the corpus split — so the mind generalises and the
tape halves differ; the bottleneck is data, not the head. `--tape-period 50` had already shown
the same from the other side (held-out honesty 0.486 → 0.686) at the price of the dup win
(0.773 → 0.174): more tapes made it talkative, and talkative loses the forgery exam.

**Stage 288 (`_stage288_repair.py`) is the response: repair, not reading.** Break a clean
address (NONE / REPLACE / DUP ×2–3), require WHERE and WHAT — or UNKNOWN when the surviving
evidence cannot say. The truth is free because the break is ours: unlimited examples, honesty
trained on breaks *known* to be unrecoverable, and the counting detector wrong by construction
where the forgery is the majority. The proposal's original reward — "consistency by the
judges" — is refused as gradient (its fixed point is an empty tape) and kept as the observer
`verdict_restored_rate`. Cloze, the lying tape and lie-dup were points in this corruption
family; 288 trains on the family. Diagnosis + repair share one relational embedding (~9k
params, ranks and indicators only); CLEAN and UNKNOWN are rows; answers are indices. Gates:
beat the minority-flag detector and the random floor, beat it on dup where counting cannot,
clean-margin AUC above noise, beat majority-repair with the true flag, UNKNOWN margin AUC on
unrecoverable breaks. Runs in minutes; `--holdout corpus|address` both available.

Nothing here breaches the invariant: `G_arc_enc_frozen` held in every arm and answers are still
values from the visible evidence. What reached the weights is a tape-specific *way of reading* —
which the invariant does not name, and which the held-out tape catches regardless.

Arm 3 (regular hidden) was correctly skipped by the queue: frozen restored neither gate.
All arms so far ran before `auc_z`, `by_evidence_shape` and `G_weighs_contested` existed; σ is
recoverable from the reported AUCs and counts, the contested split is not.

---

## 6b. The path past one-slot answers (мысли / design, not yet built)

> **Мысли, не план прогона.** Записано, чтобы не потерять нить. Ничего здесь не измерено и
> не поставлено в очередь на код, пока отдельно не скажут строить.

Recorded because it was reasoned through and would otherwise be lost. Nothing here is measured.

**The blocker is the answer interface, not the mind.** Answers are slot indices, and
`G_answer_is_slot` is what proves separation. Composite questions — compare, count, two-hop —
are not answerable by one slot, so the moment the interface changes the proof dies. §7 has said
this from the start and it has never been attacked.

**289 — the answer becomes a derivation.** A pair `(verb, slot set)` over a small closed verb
set (COUNT, COMPARE, MAX; JOIN later). Questions are *built from the tape*, so labels stay free
exactly as 288's corruption made them free. The answer is **executed**, not generated, so
`G_answer_is_derivation` replaces `G_answer_is_slot` and is strictly stronger: delete the
slots, the derivation fails. 288 is already the two-verb case — repair answers WHERE + WHAT.

**Thought = a write with provenance.** An intermediate result that is not on the tape must live
somewhere: in activations (transient, unauditable) or written back as a *derived slot* marked
"from slots {i,j} by verb v". The second keeps the invariant, extends 279's
WRITE/CONFIRM/DISPUTE with DERIVE, makes conclusions revocable (delete a premise, the
conclusion dies — the existing leak test), and lets 288's repair apply to conclusions.

**No emergence, by design — with one correction.** New *verbs* never appear from data;
capability is installed. But new *content* can, because a discrete space has search where a
continuous one has interpolation. GPT gets more than its dataset from smoothness; a tape cannot,
and that is the same coin as auditability.

**Generate-and-filter converges unless selection is external.** Accepting what agrees with the
tape yields only its deductive closure — bounded, and it saturates. The environment that exists
for free is *incoming text*: a derived slot is a **prediction with provenance**, later confirmed
or refuted by new corpus, and the verb that produced it gains or loses support. That is the only
mechanism sketched so far by which a capability could improve without being built.

**Collapse guard, required.** Training on own output destroys models. Provenance already
carries the fix: a derived slot holds a **derivation depth**; only depth-0 slots count as
evidence for further derivation; confirmation *by text* resets depth to 0. A thought becomes
knowledge only by passing through the world, never through itself.

**Generation, if it ever comes**, is the same interface at sequence length: the mind emits
structural tokens plus slot indices, content words are never sampled. Strip the tape and content
dies while structure remains. Strictly: idiom and collocation are facts about language, so the
clean version needs two tapes — world and language — with the mind holding neither.

---

## 7. What is deliberately *not* solved

- **Understanding is unmeasurable by construction.** Every question is "which slot holds it", so
  the answer already sits in one slot and no operation on data is required. The step after 286
  is **composite questions** — comparison, counting, two-hop — where no single slot answers.
  Until then "understanding" cannot be trained because it cannot be scored.
- **`ANSWER_i` is scaffolding.** When it goes, `G_answer_is_slot` goes with it, and the return
  path has to carry the proof.
- **284's `--rule fixed` arm** lacks the subject filter and the fp hop, so its precision numbers
  are not comparable to 283's. Its `margin` arm has not been reported yet.
- **The mind cannot learn from being wrong about the world.** By design. Only writing fixes the
  tape.

---

## 8. Files to attach in a new session

Minimum, in this order:

1. `HANDOFF.md` (this file)
2. `_stage286_evidence.py` — the live front
3. `_stage280_raw_exam.py` — the baseline mind, the tape builder, retrieval; everything imports it
4. `_stage279_write_decision.py` — the write side and address merging

Add only when the work touches them:

- `_stage284_scalefree.py` — if the topic is thresholds or scale
- `_stage283_scale.py` — if the topic is tape geometry
- `_stage278_value_baseline.py` — if the topic is the policy or the old teacher
- the newest `results/stage28*_decision*.json` for whatever is being discussed

Do **not** attach stages below 277; `STATE.md` and this file cover them.

## 9. Environment

- Runs happen on the user's local Windows machine; the assistant's container has the repo but
  no corpus runs. Results arrive as uploaded JSON.
- `git push` returns **HTTP 403** and commit signing is impossible. Deliverables are handed over
  as **files plus `git format-patch`**. A stop-hook complains about unverified commits on every
  turn; the email is already correct and it requires no action.
- Branch: `claude/repo-review-strategy-xnewqd`.
- `results/` is no longer tracked; `archive/results_pre277.tar.gz` holds the 894 historical
  files, and git history holds them anyway.
- Wall times for calibration: a full 280/285 run ≈ 1.5–2.5 h, a 283 ladder ≈ 10–35 min, a 286
  smoke ≈ 9 min.

## 10. How to work with this user

- They are a solo researcher, Russian-speaking, running everything locally, and **tokens are
  expensive** — answer briefly and put the reasoning in commit messages and file docstrings
  instead.
- They catch real defects (the value-head width bug, the frame-whitelist brittleness, the
  encoder window overflow, the pooled-vector weakness). **Take their objections seriously; they
  have been right more than once.**
- They want measurement, not reassurance. State what a number falsifies before the run, and say
  plainly when a prediction failed — that has happened twice and both times it mattered.
