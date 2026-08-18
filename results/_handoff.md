# TapeLM / Inprint — handoff for a fresh session

Written to replace a very long conversation. Read this, attach the four files listed at the
bottom, and you have everything the previous session had. Every number below was measured;
nothing here is a plan or a hope.

---

---

## 0. Where this stands right now — read this first

**Stage 289, retrieval line. Held out, 266 lookup questions, same tape in every arm below.**

| arm | Φ | vs counting | vs 1-NN | low margin | high margin |
|---|---|---|---|---|---|
| qrank_big | 0.7932 | z 8.08 | −0.34 | +2.40 | 0v13 |
| qmargin | 0.7895 | z 8.02 | −0.52 | +2.24 | 0v13 |
| **maxpool** | **0.8158** | z 8.20 | **+1.00** | +1.94 | 0v3 |
| mean_matched (d=41, no max) | 0.7932 | z 8.16 | −0.34 | +2.40 | 0v13 |
| mean_wide (d=64, no max) | 0.8195 | z 8.26 | +1.04 | +2.67 | 0v6 |
| rowdrop04 | 0.8120 | z 8.14 | +0.73 | +1.60 | 0v3 |
| recon3 (thin views ×3) | 0.7820 | z 7.44 | −1.09 | 0.00 | 0v5 |
| region3 pooled | 0.8120 | z 7.79 | +0.73 | +1.29 | 0v2 |
| region3 **full graph** | **0.8195** | — | — | — | — |

**Settled.** Counting is beaten by ~8σ everywhere. Max-pooling is a structural win, not capacity:
at matched parameters the mean-only arm reproduces the old failure (0v13) and max keeps the
repair (0v3); width alone fixes low-margin but never high. Overfitting was resample overlap
(0.82 → 2.8e-05, fixed by corpus size). Three hypotheses were ruled out by pre-declared
predictions: missing information (qmargin), density (rowdrop), capacity (mean_matched/wide).

**Open.** `G_lookup_beats_retrieval_paired` has never passed; best is +1.04 against a 1.645 gate.
The whole disagreement with 1-NN is 16–23 questions of 266, so **the verb is saturated** — lookup
inside one address is retrieval-shaped by construction.

**recon3 verdict (2026-08-11).** Random-thin ensembling is dead: pooled lost to single
(z −1.67), so the carry is **max alone** on the answering channel. But the run found the first
real brick of 291: disagreement D predicts error held out at AUC 0.702 (z 4.71) while being at
chance on train (0.485) — the mind fits the training tapes, its views stop disagreeing there,
and the train-calibrated refusal threshold therefore went degenerate (coverage 0.004; the
"selective reward 0.743" was blanket refusal minus epsilon — 280's rewards pay 0.75 for
refusing everything). Two structural corrections are now in the code: refusal calibrates on
the PROBE, and views are cut by REGION.

**region3 verdict (ROADMAP §21.3). Two gates of three, and the one that matters most.**

| | recon3 (thin) | region3 |
|---|---|---|
| pooled vs single | −1.67 | −0.50 |
| D-AUC train / probe / held | 0.485 / — / 0.702 | **0.761 / 0.722 / 0.687** |
| threshold probe / train | — / degenerate | 0.5246 / 0.5669 |
| selective reward | 0.743 < 0.75 | **0.767 > 0.75** |

The result is the third row, not the fourth. Thin views had D at chance on train and real held
out — a signal existing only where it cannot be calibrated. Region views read 0.76 / 0.72 / 0.69
with the probe and train thresholds nearly agreeing: **D became a property of the tape** rather
than of the sampler, which is exactly what cutting by region was for.

**Pooling is dropped, and two runs say so** — z −1.67 then −0.50, never a win. Structural, not
incidental: a region cannot see an edge into another region, the full graph sees all of them.
Splitting is what makes disagreement *measurable*; it is not a better way to decide. Single-view
accuracy is computable from the same run — 0.8120 + (9−7)/266 = **0.8195**, above the pooled
0.8120 and level with mean_wide. **The carry: answer with the full graph, refuse on region D.**

Stated honestly about the reward: the gate passed by 0.017 — 4.5 questions of 266. The policy
answers 42 (16%) at accuracy 0.929 against a break-even of 0.875, `z = +1.05`, i.e.
indistinguishable from break-even. What *is* solid is the AUC (z 4.13). And the finding worth
naming: under 280's rewards, being right 81% of the time is **not good enough to answer**, and
the derived policy correctly concludes it — hence 84% refusals.

**`region3_rowdrop35` (same code, `--row-dropout 0.35`) settles the carry.** Dropout narrowed
the train↔held D gap from 0.074 to 0.024 (0.728/0.723/0.704) and lifted `z_acc_vs_breakeven`
1.05 → 1.48: it teaches a disagreement that transfers rather than one that fits. But the result
is the new `answer_full` column —

— but the no-dropout rerun supplies the other two cells, and they change the reading:

| | coverage | accuracy | reward |
|---|---|---|---|
| nodrop / pooled | 0.158 | 0.929 | 0.767 |
| nodrop / full | 0.158 | 0.952 | 0.774 |
| drop / pooled | 0.154 | 0.951 | 0.773 |
| **drop / full** | **0.605** | 0.888 | 0.766 |

The coverage jump is **one cell of four**, it needs dropout and the full graph together, and the
reward does not corroborate it — all four sit within 0.008. With 200 probe questions the
admissible threshold is placed noisily, so this is a hypothesis to confirm, not a carry. (An
earlier note here credited the jump to the full graph alone; the four-cell grid says otherwise.)
**Pooling is dead for the third time** (z −1.67 → −0.50 → −0.47); it is not tested again.

The no-dropout rerun also served as the sanity check on the new column: `held_d_auc` 0.6874537
and refusal 0.15789 / 0.9285714 / 0.7669173 reproduce the pre-patch JSON exactly.

**`ink_hash` — the best number the project has on the open gate.** Same tape (write pinned to
arc, 2078/729/2.8505), one variable.

| | arc | hash |
|---|---|---|
| Φ held | 0.8120 | 0.8083 |
| rival_cos | 0.8008 | 0.7895 |
| vs 1-NN | +0.73 | **+1.39** |
| low margin | +1.29 | **+1.73** |
| probe loss | 1.081 | **0.831** |
| cos mean / std | 0.894 / 0.046 | **0.447 / 0.235** |

Non-inferiority passes: −0.0075 on Φ is two questions of 266. But the gap to 1-NN *grew*,
because the rival weakened while Φ held. That refines the claim made when the ink arms were
queued — better ink lifts both, but hash is not better, it is **different**, and Φ turns out
more robust to an ink change than 1-NN is. Signs now agree across train (+1.0) and held (+1.39)
where earlier arms flipped.

**`ink_bigram` closes ROADMAP §18.3, in the second direction.** Same tape again.

| | arc-mean | arc-bigram | hash |
|---|---|---|---|
| Φ held | 0.8120 | **0.7895** | 0.8083 |
| rival_cos | 0.8008 | 0.7895 | 0.7895 |
| vs 1-NN held | +0.73 | **0.00** (4v4) | +1.39 |
| vs 1-NN train | 0.00 | **+3.21** (13v1) | +1.00 |
| cos mean / std | 0.894 / 0.046 | 0.649 / 0.121 | 0.447 / 0.235 |

Bigram ties 1-NN to the last digit (0.7895 both) and flips sign train → held: the order binding
was learned and did not transfer. So of §18.3's two branches the second holds — **287 was right,
order is not the problem here**, and the symmetric swap pairs in the confusion table belong to
the tape (`reversed_pairs`), not to the ink. A week of bigram work bought one negative number,
which is the normal price.

That also corrects the mechanism claimed for hash one entry above. Cosine spread is not it:
bigram's 0.121 is three times arc's and its result is the worst of the three, so the relation is
not monotone. What distinguishes hash is **morphology and sub-words** — inflections and OOV land
near their relatives. Order was noise; sub-words are signal.

**Hash should become the default**: no worse on accuracy, clean on the third invariant line (the
checkpoint is never loaded — `not_loaded`), best margin against retrieval. Pending `write_hash`
for the fragmentation half.

**`write_hash` picks the winner and closes the ink line.** Calibration: tau 0.4712 → density
2.9720 against 2.9701, monotone, 14 probes. The train tapes are comparable in shape (811
addresses vs 804, density 2.999 vs 2.970), so the write comparison is clean.

| train | arc | hash |
|---|---|---|
| `same_anchor_diff_relation` | 0.0785 | **0.0416** |
| `mean_addresses_per_fact` | 1.0900 | **1.0524** |
| lookup questions | 265 | **365** |
| held `reversed_pair_rate` | 0.00598 | **0.00236** |

Fragmentation **halved**, and the ceiling rose with it: a third more answerable questions from
the same corpus. Reversed pairs fell 2.5×, which turns yesterday's inference — that the swap
pairs were an addressing defect, not an ink one — into a measurement.

Φ on the new tape is 0.7486 against 1-NN's 0.7514, **z −0.23**. Not a contradiction of the
+1.39: different tape, harder questions (350 vs 266, 7.63 candidates vs 6.32, floor 0.285 vs
0.302), and Φ was never comparable across tapes. **That is the run's real finding: better
writing cannot close the retrieval gate.** It buys more questions and harder ones, and Φ ties
retrieval on those too — because lookup inside one address is retrieval-shaped by construction,
whatever the ink or the tape quality. Only §19 changes that.

**Winner: hash on both sides.** Reading — non-inferior with the best margin against 1-NN.
Writing — fragmentation halved, a third more questions. Invariant — the checkpoint is never
loaded. Cost — `--tau-mode density` is mandatory, tau 0.47 rather than 0.90. **No further ink
arms.** `write_bigram` need not be waited for: bigram already lost on reading.

**`tau_smoke` — the density instrument works.** `monotone: true`, the bracket covers the target
(tau 0 → density 18.58, tau 0.9995 → 2.0), converged in 14 probes to **tau 0.3162 → 2.9638**
against a target of 2.9701. The number that matters is 0.3162 **against arc's 0.90**: refusing a
write-ink change at absolute tau was not caution, the tape would have shattered outright. The
residual 0.0063 exceeds the 0.005 claimed for 12 steps, but it is not a bisection miss — density
is discrete here (2.9638 at 138 addresses, 3.0074 at 136, nothing between), so this is the
optimum available on this corpus.

Offline proof before any long run, like the ink check: `python _check289_region.py`.

**The ink was never tested — both arms are still queued (ROADMAP §18.3a).** Every number above
is `ink: mean, fp: arc`; the bigram and hash inks passed their offline checks and were then lost
behind the instrument work. Note before reading their results: better ink *cannot* close the
retrieval gate — Φ and `rival_cos` read the same ctx vectors — so `--fp hash` is a
**non-inferiority** test (pass = `|z| < 1.645`) whose prize is the invariant: hash is the only
ink that does not know the corpus. The arm that could raise the ceiling is the **write**-side
one, and it needed `--tau-mode density` (§18.3b) because 279's tau is an absolute cosine.

**The default configuration from here on:** `--fp hash --write-fp hash --tau-mode density`,
answer from the full graph, region views for refusal. Every number taken before this was under
arc ink; §18.5's rule applies to them.

**290 ran, and it answered a different question than it asked.** All instruments healthy: 5697
parameters, five edge channels, `nb_channels` 0.364 / 0.896, `graph_rows` mean 5.8 max 15 (the
per-neighbour cap held). Φ beat counting at z **+2.53** and came within 0.012 of the retrieval
gate at z **+1.63** — the closest this project has ever been. On **21 held-out questions**, where
the achievable ceiling is 2.45. The number is not a result.

**Why 21.** Of roughly three hundred sparse questions, twenty-one were answerable.
`smoke291b` states it from the other side: `unanswerable_rate = 0.931`. **The neighbourhood
holds the value written at its own address 7% of the time.** That is the finding — a property of
the tape, not of the reader.

**And it settles what to do about cross-entropy, in the opposite direction to the obvious one.**
Training on likelihood while scoring on 280's payoff is a real mismatch and it is now removable
in closed form — `L = −Σ_c p(c)R(c)`, `p = softmax(φ)`, R the fixed +1 / −1 / +0.75, no sampling,
no baseline, no new constant (`--objective reward`). But it will **not** rescue 291. Under that
payoff answering beats abstaining exactly when p > 0.875, so at a 7% answerable rate refusing
everything is not a collapse, it is the correct policy, and reward training reaches it sooner.

The headroom is the number that decides this. A perfect discriminator beats blanket refusal by
`p_answerable × (2·accuracy − 1.75)` — at 7% and no errors at all, **+0.0175**, about 2.5
questions of reward on a 145-question held set. Undetectable by construction. **291 is
unmeasurable until the answerable rate reaches roughly 0.3**, whatever the objective.

So the lever is upstream, and `neighbourhood_audit` measures it with no model and no gradient:
the hit rate per route — anchor, relation, rare words — and per k. The suspicion it exists to
test: `rel_nonzero_rate` was 0.896 in the full run, so the relation route supplies nine pairs in
ten, and relations here are words like "and" and "the". The route producing most of the rows may
produce none of the answers.

**READ THIS BEFORE THE 292 BLOCK BELOW: THE 2×2 IS IN AND THE WIN WAS THE LADDER.**
One seed (1337), three cells, everything else identical:

| address | distractors | Φ | retrieval | paired z |
|---|---|---|---|---|
| fp | ladder, built by relatedness | 0.416 | 0.348 | **+1.64** |
| fp | uniform, drawn | 0.378 | 0.375 | **+0.12** |
| anchor | uniform, drawn | 0.397 | 0.387 | +0.31 |

The addressing rule moves almost nothing. The distractor source moves everything: on the SAME
tape and the SAME evidence, replacing constructed rungs with drawn values collapses held z from
+1.64 to +0.12 (37 v 36 of 315). So **Φ's advantage over 1-NN existed only against distractors
whose relatedness was perfectly correlated with being wrong** — which is a fact about the
construction, not about the mind. The inverted `mean_phi` was the tell, and removing the ladder
confirms it by removal. The pooled +3.24 below must be read as that artefact until two more
uniform seeds say otherwise; it is no longer "the first win over the rival that matters".

`same_anchor` came back 0 here too, this time by base rate rather than by construction: an fp
address holds ~3.13 mentions and an anchor holds barely more, so `at_anchor \ here` is nearly
empty and a uniform draw lands in it ~0.3% of the time. A two-bucket categorical can never
measure this landscape. It needs a CONTINUOUS distance carried by every drawn candidate —
correlate Φ against the candidate's cosine to the address rows — which also needs no
construction and so cannot leak.

**Where this leaves the line.** Φ has now failed to beat retrieval on every construction that
was not leaking. Lookup inside an address is retrieval-shaped (§0 said "the verb is saturated"),
291 starved on 7% answerable, 293 starved on function-word relations, and 292's margin was its
rungs. Every verb tried so far is single-hop selection among candidates, which is exactly what
1-NN does. The untried structural move is the one ROADMAP §6b already describes: **a verb whose
answer requires two rows to be COMBINED**, so that no single nearest row contains it and the
rival has to become two-step retrieval. That is the next thing worth an hour, not another seed
of a selection verb.

**292 is the first arm where Φ beats the rival that matters — and the run says do not celebrate.**
The rival here is retrieval over the WHOLE TAPE (nearest imported context), which is RAG stated
exactly; the within-address rules cannot reach a foreign value at all and are excluded from the
gates rather than scored as a walkover.

| | Φ | corpus retrieval | paired |
|---|---|---|---|
| held (n=159) | **0.4528** | 0.3459 | **z +2.59** (30 v 13) |
| train (n=127) | 0.3386 | 0.3465 | z −0.15 (21 v 22) |
| pooled | | | z **+1.73** (51 v 35) |

Held clears the floor at z +5.91, so the task is real. But the two halves are **2.74σ apart on the
same weights**, and pooled the margin is +1.73 against a 1.645 gate — a hair. Held better than
train is backwards; both are unseen tapes, so these are two draws of one quantity that disagree.

**And the mind never learned.** Early stop chose **step 300** of 6000; the probe loss best is
1.3615 against ln 4 = 1.3863, i.e. **0.025 below chance**, and it rises monotonically after step
300 to ~1.5 and never returns. Six thousand steps made it worse throughout. Whatever produced
0.4528 was present in a barely-trained network, and further training destroyed it.

**The landscape gate fails cleanly, with a signature worth keeping.** Concordance 0.5054 held
(459 pairs, not underpowered), z +0.23. And `mean_phi` reads true 0.219, **near −0.409**,
middle −0.071, far +0.146 — Φ separates the truth from everything else but ranks the wrong
answers *backwards*, scoring the structurally NEAREST one worst. That is the boundary-not-a-
landscape distinction the ladder note predicted, now measured rather than argued.

`--seed` was added because of this run: two samples of one number disagreeing by 2.74σ, and no
way to ask for a third. **Six more seeds were run; the pool settles positive but the 2×2 above
downgrades how to read it.**

**292 AT SIX SEEDS — the pool, now to be read against the 2×2 above.** Three more (8642 / 5200 /
1337) give held z **+2.67 / +0.27 / +1.64**, so five of six seeds are positive and none of the
new ones reverses. Pooled held **201 v 141 discordant on 1056 questions, z +3.24**; accuracy
0.412 against whole-tape retrieval 0.348. Read against uniform distractors (+0.12 on the same
seed): the margin was the ladder, not a general beat over 1-NN.

Also: `open_near_source` was 50827 same-anchor and 0 neighbourhood, so the sibling route is
alive on a real tape and last night's fallback was never needed — it was dead only on the toy
smoke tape.

**Next, in order:** 290 = ROADMAP §19 (graph spans several addresses + `lookup_sparse`, where
1-NN inside an address is undefined and most of the mention distribution is currently thrown
away) with §22's batching written into it, then 291 (refusal — the D channel above plus §20.3's
derived 0.875 rule), then 292 (generation).

**Designs recorded, not yet run:** §19 multi-address, §21.2 tape growth (train at N, score at
2N/4N/10N), §22 batching, §23 rows→value-class compression.

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

**A third line, added when hash ink made the loophole reachable:**

> The ink may not know the corpus.

**And a correction to how the second line has been read.** "Whatever is trained may not hold
facts" has been applied as though it meant *the weights may know nothing about the corpus*. It
does not, and reading it that way is a mistake — see ROADMAP §15. Knowledge may reach the
weights; the text and the facts may not. A child knows a dropped mug will break without knowing
an equation. The operational form — stated positively, because "must not answer" says nothing
about what should be there instead:

> **The weights carry the skill of looking and the understanding, never the answer. Given any
> tape they must look; finding nothing, they must say so.**

Three clauses, and two are already measured rather than hoped:

| clause | status |
|---|---|
| skill not answer → transfers to a tape the weights never saw | **shown**: the held-out tape is built from a disjoint 30% of the corpus — different lines, different addresses, different facts — and 289 scores 0.6923 there with paired z 4.32 |
| finding nothing, says so | **shown in part**: 205 `g_forget` + `g_no_collateral` — delete the slot and the fact is gone with no collateral. Refusal as a first-class *action* is 291 |
| **any** tape | **open, and this is the real gap**: the held-out tape is another slice of the same wikitext — same language, same style, same entity distribution. Transfer across *samples*, not across *corpora*. Costs one path change to test |

That is testable where "nothing leaks" never was. And it reframes the parameter budget: 3489 is
not a wall against knowledge but a **sieve**, because a type-level regularity is compressible
across thousands of facts while a specific fact is not. Only what generalises fits — which is
the definition of understanding, not an obstacle to it.

The first two say "not trained". That is not the same thing and the gap is a real door. Random
Indexing over co-occurrence, a PCA fitted to the corpus, a co-occurrence count table - none of
them is *trained*, none involves a gradient, and all of them would hand the ink "car is like
automobile" for free. Under the letter of the two-line invariant they are all legal.

They must not be legal, because an ink that knows the corpus makes the central claim
unfalsifiable: when the mind answers well, nothing distinguishes "the mind read the tape" from
"the index already knew". The distinction between knowledge and mind survives only if the
representation is content-free.

What this buys, stated positively rather than as a prohibition: **any semantics must be derived
from the tape's structure** - co-occurrence of addresses, shared contexts, support counts -
because there is nowhere else for it to come from. Ink with no semantics is not a hole in the
design. It is the design, and it is what makes "the mind reads" a claim that can fail.

Corollary for choosing an ink: orthographic similarity (hash n-grams) is legal, visual
similarity (glyph bitmaps - homoglyphs, CJK radicals, OCR robustness, the open fork for the
multilingual stage) is legal, distributional similarity is not, however it is obtained.

**How separation is proven today:**

| proof | mechanism |
|---|---|
| `G_arc_enc_frozen` | `arc_enc_hash(model)` identical before and after training — **vacuous under `--fp hash`**, where `G_ink_verified` replaces it: a fresh bank reproduces every probe bit-for-bit and every coordinate sign matches hashlib |
| `G_answer_is_slot` | the answer is a slot index, so it cannot be generated |
| tape resampling | the tape is rebuilt every ~200 steps during training |
| held-out tape | scoring uses a tape built from a disjoint 30% of the corpus |

The second one dies as soon as answers stop being indices. Its only label-free replacement is
the **return path** (§5), which is why that check is kept alive even though it lost its own
ablation.

---

## 2. Architecture as it stands

**One page, current as of the reconciliation commit. Everything below is in
`_stage289_derivation.py` unless marked.**

```
raw text ──WRITE(279)──▶  TAPE: slots (value, context, support), addresses fp{g}:anchor|relation
                              │
                              │  a question hides one value; the row STAYS, carrying a sentinel
                              ▼
            ┌─────────── build_graph ───────────┐
            │ rows   = mentions of the address  │   edges: same-value | rank(ctx cos) | rank(rare)
            │ nodes  = shares + indicators      │   nodes: share, subject, side, 1/n, query,
            └───────────────┬───────────────────┘          imported, qcos rank, qmargin
                            ▼
      Φ = lookup( [mean ; max] over rows of node( nf, own-mean, all-mean, all-max ) )
                            │
        V views of the same question ──▶ pooled logits (1/V)Σφ_v ──▶ answer = argmax
                            └──────────▶ D = JS(views) ──▶ confidence, refusal
```

**Ink.** `fp(w)=normalize(arc_enc(chars))`, frozen, d=256, order-blind. `ctx_fp`= mean over
words — order-blind twice. Alternatives exist behind flags and are unmeasured: `--fp hash`
(blake2b over character n-grams: nothing trained, no character vocabulary, no OOV, any script),
`--ink bigram` (non-commutative binding of adjacent words), `--words unicode`. The WRITE ink is
pinned separately (`--write-fp/--write-ink`) because 279's merge threshold is an absolute
cosine and a different ink shatters the addressing.

**Exact algebra, zero parameters.** `new_i = 1 − max_{j<i} s_ij`; `count = Σ new_i`;
`compare = sign(count_A − count_B)`. Both 1.0 by construction, uncapped, scale-free.
`exact_mismatches` checks them against the tape and has never fired.

**The one trained surface** is Φ: the coherence of a completed world. For each candidate the
query row is filled in and the resulting world is pooled to one scalar. 5601 weights at d=32
with max-pooling; equivariant, so the same weights read 3 rows or 300.

**Reconciliation (new).** `--views V` reads V independently thinned views with the *same*
weights and pools their logits by a mean; `softmax(Σφ_v)` is the normalised product of the
per-view distributions, so one cross-entropy is the whole objective and V=1 reduces to the
previous loss exactly. A view keeps ≥1 row for **every** candidate — thinning only redundancy —
so candidates, label and logit order are identical across views and the label cannot leak.
`D` = Jensen-Shannon divergence between views = label-free confidence. Refusal answers iff
`p > 0.875`, a constant *derived* from 280's fixed rewards (+1/−1/+0.75), with the cut on `D`
read off the train curve.

**Rivals, both parameter-free, both paired in-run.** Majority-over-survivors (counting) and
1-NN by context cosine over the same rows (retrieval). Counting is beaten by ~8σ in every arm.
Retrieval is the open one.

**Discipline.** Held-out tape from a disjoint 30% of the corpus; the probe reserves *anchors*
(never lines) and is excluded from both training and scoring; tape resampled every 50 steps
with measured overlap (2.8e-05); early stopping on the probe; `max_achievable_z` marks a
comparison that could not have passed.

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
13. **The rival reads something the mind cannot.** 289a's first run: the asserted value reached
    the graph only through the query row's same-value edges, and that row is all-zero both when
    the question claims nothing *and* when its claim is missing from the evidence, while
    `ctx_fp(..., exclude=asserted)` stripped the value from the text too. So `ok`,
    `wrong_relation` and `false_premise` were **the same input** — three of four classes
    indistinguishable in principle — and the mind put all of them in one bucket (recall 0.0/0.0,
    identically on the train control, which is the tell: a generalisation failure does not
    reproduce on the training tape). Meanwhile `counting_rival` read `q["asserted"]` directly in
    Python. **Before comparing, check that every field the rival branches on is reachable from
    the mind's inputs**; when a class is defined by a fact, that fact needs a channel, and an
    identity-free indicator is enough. Corollary: build the collision table by hand (which pairs
    of classes are input-identical to a counter?) and confirm the intended count is exactly one.
14. **A class with no denominator.** Same run: `wrong_relation` came out at n=6 held out with
    `MIN_ANSWERED = 5` — the class the entire stage existed for could not have shown a
    significant win even from a perfect model, because one random sibling was drawn per item.
    Enumerate every eligible construction, and sample the classes uniformly in training: an
    imbalanced mix pays a mind to never name the rare class, which is the same 0.0 recall
    arriving by a second road.

15. **I wrote the collision-table rule and then did not run it.** 289's negative ladder scored
    `mean_phi` near / middle / far identical to fifteen decimals, because a candidate value
    enters the completed world ONLY through same-value edges and the value-share feature, and
    the query row's context comes from `p["texts"][slot]` and does not depend on the
    substitution at all. Any value absent from the survivors therefore produces a
    **bit-identical graph** whatever it is — the three rungs were one input, exactly failure
    mode 13 in a new place, in code I wrote after writing mode 13 down. The reported
    `z_vs_half = -3.44` is not an inverted landscape: it is ties scored as failures by a strict
    `>`. The measurable part says the opposite of despair — `true > rung` held in **25 of 27**
    held-out questions.
    **The fix is on the critical path anyway**: completing a world with a value must import
    that value's OWN mentions as rows, not merely write a label. Then near drags in mentions of
    the same anchor and far drags in strangers, and the worlds genuinely differ. 292 needs the
    same thing to score a value from the whole tape, so this is the generator's first brick
    rather than a repair.

---

**15. A closure quoted past its expiry date.** Every result carries the conditions it was taken
under, and a gate that established *possibility* is not a measurement of a *setting*. The ink
was recorded as "closed (277, 287)" on a 1.3% order-content finding, and that line sat in
ROADMAP §0 while this project spent a week building order-aware ink on the opposite premise. The
tell is a document asserting something the current work contradicts without either being marked.
Where the difference was small and the capability was already technically confirmed, the number
is not an argument — only qualitative collapses cross regimes. See ROADMAP §18.


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

## 6b. The path past one-slot answers (design, not yet built)

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

## 6b-bis. The whole frame and the plan: see `ROADMAP.md`

Five parts — ink (closed), fact tape (done, scales), mind (done, 288's seven gates), mind tape
(289b decides), conjecture loop (planned) — plus the stage plan 289 → 292 with a falsifier on
each, the three controls that keep interpolation from being hallucination, and the four ways
this concept could fail, written before the runs. This file keeps the measured state;
`ROADMAP.md` keeps the frame.

## 6d. Hand-choices, wishes, and the holes found on re-reading

Full text in `ROADMAP.md` §7–§11. Compressed:

**Still chosen by hand** (named so "derived" cannot cover them): the smoothness radius needed
a definition and now has one — majority-class floor, exceeded by more than its own noise at
1.645; `1.645`/`0.05` are conventions and the only constants left in the examiners; the **verb
vocabulary is installed by hand, deliberately** — the verb curve is what judges that.

**Two holes in the plan, found before they cost a run.**

1. **The invariant was one level too low.** A pattern in the mind tape *is* knowledge — about
   situations. The right statement: **the mind holds no content of any kind.** Facts on one
   tape, patterns on another, the mind is a fixed comparison procedure over both. A nearly
   empty reader is the goal arriving, not a retreat.
2. **No arrival time on the tape, and 290 needs it.** "New text confirms or refutes" is
   unenforceable without an order: a conjecture would be graded against text that was already
   there. **Mandatory before 290:** a monotone arrival index per slot; a conjecture is scored
   only against slots with a strictly greater index. Held-out discipline, applied along time.
3. **Nothing forgets.** Weights don't grow; both tapes grow forever. Prune by *use* — never
   retrieved, never cited, support decayed. Falsifier: pruned tape within noise of unpruned.

**Two cheap mechanisms that separate a mind from a reference book:**

- **Unprompted audit.** 288's diagnosis head, run as a sweep with nobody asking. Anomalies
  become requests to the write side. A reference book answers; this notices. Costs one loop.
- **Confidence as empirical frequency.** Store per pattern how often the mind was right in
  situations like this one. Calibration by memory, not by softmax — on a tape, improving
  without retraining. Falsifier: must beat softmax confidence at matched coverage.

## 6e. Six more from the inside (`ROADMAP.md` §12) — two are buildable now

- **Confidence is real; provenance is not.** A model can often tell it is unsure but never
  *why*. Untraceable confidence cannot be acted on; a support count of 4 against 2 with the
  dissenters nameable can. Warning light versus diagnosis.
- **Retrieval and reasoning share capacity → a new gate.** In one set of weights, recall
  degrades as inference deepens. A separated system must be **flat**. Score factual accuracy
  against derivation depth: `G_depth_does_not_cost_recall`. Free with 289's machinery, and the
  only proposed gate that tests separation **under load** rather than at rest.
- **The real product claim is correction without retraining.** Hallucination is what people
  complain about; un-correctability is what costs them. A correction here is one write with
  provenance, persistent and revertible.
- **Understanding ≈ catching the malformed question.** Not a verb count. False presupposition,
  absent entity, incomparable comparison — all detectable by 288's diagnosis head pointed at
  the *query*. **Proposed as the first new verb, before `COUNT`/`COMPARE`:
  `REJECT_PRESUPPOSITION`.** Falsifier: build false-presupposition questions from the tape and
  require rejection above the rate of rejecting good ones.
- **The right benchmark is mostly unanswerable questions.** Every existing one asks what has an
  answer — the single regime where a fluent interpolator wins and this design deliberately does
  not compete. Hold out addresses, ask about them, score the refusals as the primary quantity.
  That is the honest home for `G_beats_silence`.
- **The system can publish a map of what it knows badly.** Support, agreement and external
  corroboration are known per address before any question is asked. One loop over the tape
  turns that into a map — likely the most immediately useful artefact, and impossible for a
  model whose competence is not represented anywhere.

**If the choice were mine:** `REJECT_PRESUPPOSITION` first, and `G_depth_does_not_cost_recall`
written into 289 as it is built.

## 6c. The derived slot — design agreed, not yet built

GPT: point → decoder → text; the blend lives one forward pass and leaves no trace. Here:
**point → a written slot → the output only reads.** A thought becomes an *object* — citable,
disputable, revocable with its premises — instead of a flash in activations.

- **Hallucination is a labelled state, not a failure mode.** `depth > 0` marks conjecture.
  Being wrong is allowed; *not knowing that it was a guess* is not.
- **Storage is not new.** 279 writes (WRITE/CONFIRM/DISPUTE + support counts), 288 repairs and
  catches forgeries, the leak test already kills conclusions with their premises. A derived
  slot is an ordinary slot with provenance.
- **One new rule:** only `depth 0` counts as evidence; only text from the world resets depth.

Four things the sketch still needs, each plugging a leak:

1. **The mind proposes, `Tape.decide` writes.** A conjecture that restates an existing value
   must become `CONFIRM`, not a new slot — otherwise the tape inflates with paraphrases of
   itself and support counts stop meaning anything.
2. **Depth must reach the OUTPUT, not just the index.** An answer served from a `depth > 0`
   slot is labelled a conjecture to the caller. Marking that stays internal is not honesty.
3. **Refutation must cost the MECHANISM.** When the world contradicts a derived slot, deleting
   the slot is not enough — the pattern that produced it loses support, or it re-derives the
   same error forever. That loss *is* the selection loop; without it there is no evolution,
   only churn.
4. **Two gates, or it is philosophy.** (a) A tape with conjectures must predict *incoming*
   text better than the same tape without them — selection against something the mind cannot
   edit. (b) The conjecture rate must not grow without bound as the tape grows — the
   self-inflation check, and the direct measurement of model collapse.

**Invariant, third line:** the mind holds no facts; a thought must be written; a written
thought is not evidence until the world confirms it.

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

## 9a. The result the 289 line existed to test — and one caveat on how it reads

**The context channel beats counting, paired, held out, at 4.42 sigma.** 289 `exact` arm:

| | held out | train |
|---|---:|---:|
| lookup accuracy | **0.692** | 0.712 |
| majority rival | 0.413 | 0.485 |
| random floor | 0.330 | 0.298 |
| McNemar z, paired, same items | **4.42** (36 vs 7 of 43) | 7.26 (111 vs 26 of 137) |

The rival over survivors is Bayes-optimal when the query context carries nothing, so this is
the direct statement that the hidden mention's context carries information counts do not have.
It is the first time anything in this project has beaten the counting rule on producing a
value rather than on judging a situation.

**Removing the two heads bought 27 points.** Same 104 held-out items, same tape, same lookup
construction: 0.423 with count and compare sharing the body, **0.692** with them gone. The
interference was far larger than the 9 points I estimated from the earlier marginals. A
description and a preference ordering fighting over one 32-wide vector cost more than half the
available signal, and 3457 parameters now do what 7881 did worse.

**The exact algebra is verified on real data**, not only on synthetic tapes: `exact_mismatches`
= 0 over 385 held-out and 1002 train questions, on both splits.

**Caveat, and it matters for how 289c is cited.** The audit's Spearman rose to 0.961 held out
and 0.979 transfer, but **nine of its twelve regions are now exact** and sit at stated 1.0 /
realised 1.0. Ordering regions is much easier when three quarters of them are deterministic.
Inside the three judged regions the ordering is flat or wrong:

    lookup|mid|split    stated 0.982  realised 0.629
    lookup|deep|major   stated 0.894  realised 0.667
    lookup|deep|split   stated 0.823  realised 0.667

So the calibration problem is unchanged and confined to lookup: still overconfident, still
unable to rank its own weak spots. Do not read 0.961 as the audit having improved. The honest
number to quote is the judged-region behaviour, and `G_calibrated` is still false (ECE 0.061
against a 0.022 noise floor).

`G_refusal_is_informed` is false on its denominator only: one question was refused and
refusing it was **right** (`refused_would_have_been` 0.0), but 1 < MIN_ANSWERED.

---

## 9b. Before proposing any long run — the cost check

Learned the hard way on 289a: `--wiki-bytes` went from 30M to 150M and `--addresses` from 400
to 1200, and a 70-minute stage silently became a 7-hour one. Nobody noticed until the run was
already an hour in. Three rules, all cheap, all mandatory:

1. **Recompute wall time whenever a size knob moves.** Cost is linear in
   `train_lines × (n_steps / tape_period)` — every resample re-scans the whole corpus slice
   (`common_nouns`, `corpus_assertions`) and re-inks every assertion. Multiplying the corpus
   multiplies the *resample* cost, and the resample *count* is a separate knob that must be
   re-tuned alongside it. State the estimate in hours **before** the user starts anything.
2. **A run with no intermediate output is a run that can lose everything.** These stages write
   their JSON only at the end; a crash at hour six leaves a loss curve and nothing else — 288
   already cost one hand-reconstructed result this way. Anything over ~2 hours needs a periodic
   measurement in the log, not just a loss.
3. **If a knob would have to be guessed, measure it instead.** "Are 120 tapes needed or would
   50 do" is a saturation curve from one run (`--eval-period`, `tape_curve` in 289a), not an
   opinion. One run answers it for every later stage. Prefer the curve to the argument.

Corollary on what resampling *is*: real use is one growing tape, so many tapes is not a
simulation of deployment. It is the proof that no single tape was memorised — the separation
claim itself — so the number of tapes is a scientific parameter, not a convenience.

---

## 9c. Two numbers that are settled — do not re-derive them

**Twenty tapes is enough.** 289a's `tape_curve` measured blind-pair accuracy on one fixed
held-out tape every ten distinct training tapes, across 120 of them: 0.225 at ten, 0.570 at
twenty, and then flat between 0.535 and 0.565 all the way to 120. Resampling more often buys
nothing after about twenty tapes. Use `--tape-period` ≈ `train_steps / 20`; the old default of
50 was doing 120 tapes and paying for a hundred of them for no return. This does not mean the
mind is finished — it means whatever is limiting it is the channel, not the supply of tapes.

**Class-uniform training changes what the examiner may compare.** Failure mode 14's fix draws
the verdicts uniformly, so the mind's logits are calibrated to 1/4 each while the examiner sees
the tape's natural mix (76/24 on the blind pair). An argmax accuracy compared against a
majority floor then charges the mind for a prior shift the examiner introduced, and it will
fail even when the discrimination is strong: 289a scored 0.558 against a 0.764 floor while
separating the classes at TPR 0.700 vs FPR 0.487, z = 3.35. Whenever training is class-uniform,
gate on the AUC (prior-free, null point 1.645 as everywhere else) and report the argmax only
after the exact Bayes correction using the TRAIN tape's frequencies.

**A marginal difference between two runs is not the test when the items are shared.** 289's
depth ablation scores the SAME lookup questions twice. Read off the marginals it looked like
derivation cost nine points of recall; the paired McNemar on `lookup_item_hits` says 11
discordant one way and 6 the other, z = 0.97, p = 0.17 — no detectable cost. I reported the
marginal difference first and it was wrong. Emit per-item hits from any stage whose claim is a
between-run difference, and test them paired.

**The real non-additivity showed up in the other verb.** Across three versions of 289, as
lookup grew from 21 questions to a real task, `count` fell 0.965 → 0.934 → 0.903 and `compare`
0.883 → 0.859, with 289c localising it exactly: `count|*|unanimous` and `count|thin|*` stay at
1.0 while `count|mid|split` and `count|deep|split` collapse to 0.0. Three verbs share one
7.9k-parameter body and displace each other. Fixing one degrades another, which is what "the
capability is not additive" actually looks like — not a loss in the verb you were repairing.

---

## 9e. The road to generation — see ROADMAP §13

Written down because the question "does the exact algebra close the path to a mind that
produces?" deserves an answer that survives the session it was asked in. Short form:

- **Phi is already general.** It scores any completed world and does not know where the
  candidate came from. The restriction lives in the CANDIDATE GENERATOR, not the judge, so
  generation means widening the generator and leaving Phi alone.
- **Next cheap step: a ladder of wrong answers** — near (same anchor, other relation), middle
  (neighbouring address), far (random tape value). Labels free, no authoring. Trains Phi into
  a LANDSCAPE instead of a boundary, gated by monotonicity `G_phi_orders_negatives`. A mind
  that cannot rank its own wrong answers by how wrong they are cannot steer.
- **Controlled interpolation needs three parts and has two**: the smooth space (289b, 0.82/0.73
  vs 0.46/0.41) and the differentiable judge (Phi over the soft algebra — this is why §9d
  insisted the soft form be written). Missing: a generator of points, answered by the nearest
  value on the WHOLE tape rather than local candidates. No text is generated; the answer stays
  something the world wrote, found by a thought.
- **Two named ways it fails**: the landscape is flat (Phi ranks near/middle/far alike), or the
  nearest tape value is always the majority value anyway.

---

## 9d. The ratchet — when NOT to take something out of the weights

289 moved count and compare into exact algebra because the weights were approximating a
quantity computable from their own input. The rule that justified it is dangerous in the
general form and must be stated with its limit, or it eats the project:

> "Whatever is computable, compute it" ends with no mind at all. Every answer about a finite
> tape is computable in principle.

The rule that is actually right:

> Whatever the input **determines** is arithmetic. Whatever the input **underdetermines** is
> judgment, and belongs to the mind.

`count` given `s_ij` is determined. `s_ij` itself is not. Which value fills a gap is not.
Whether a question should be answered at all is not. Those stay.

**What makes the move safe here, and the test to apply next time.** The arithmetic that
replaced the heads is not a wall, it is the degenerate case of a learnable computation:

    new_i = 1 - max_{j<i} s_ij        count = sum_i new_i

With `s_ij` in {0,1} this is union-find and exact. With `s_ij` in [0,1] it is differentiable,
the count goes fractional, and the gradient flows into whatever produced `s_ij` — measured:
s(USA, United States) at 0.0 / 0.3 / 0.7 / 1.0 gives count 3.00 / 2.70 / 2.30 / 2.00 on the
same three values. So nothing was closed; today's degenerate sameness was substituted into a
formula that stays open.

**Before taking any future capability out of the weights, require both:**
1. it is determined by the input, not underdetermined; and
2. the replacement is written so the learnable version is a strict generalisation of it — if
   you cannot write the soft form, you are building a wall, not a floor.

What the move genuinely costs: the mind no longer *chooses* which operation to apply, nor
composes them. It never did — the verb was given from outside in 289 — so an imitation was
removed rather than a capability. Choosing among operators is 291, and it is easier against
well-defined operators than against three heads approximating them unevenly.

---

## 9g. Three false forks about the ink — do not spend a session on them again

Raised as "could the ink be PCA, or neuron activations per character, or character images —
was picking vectors a mistake?" All three deserve an answer and two of them dissolve.

**Vector versus activation versus PCA output is not a distinction.** An activation is a vector;
a PCA output is a vector. The real questions are what fixes the coordinates and what operation
joins them, and every defect this project has found in the ink lives in the *join*, never in
the coordinates.

**PCA was not skipped; its trainless twin was taken.** Random projection is the untrained
analogue and Johnson-Lindenstrauss bounds how little is lost. Hash ink *is* that projection.
Fitting a real PCA would additionally violate §1's third line, so the trainless version is the
better object, not the compromise.

**"Averaged neuron activations over the characters" is exactly what `arc_enc` already does** —
per-character embedding, mean, feed-forward. The idea was implemented at stage 191. Its defect
is the `mean`, which is order-blind, and that is a property of the join, not of using
activations. Nothing was lost by "switching to vectors" because there was nothing to switch
from.

**Character images are the one part never tried, and there is a trainless version.** Render each
character to an N×N bitmap and use the bitmap as the character's vector: no training, fully
determined by the font. It buys homoglyph unification (Cyrillic О and Latin O become one),
robustness to OCR confusions (1/l, rn/m), and CJK radicals for free — 江 河 湖 share the water
key, so a semantic family falls out of pixels. It costs a font file, which is a hand-choice of
the kind this project avoids, and it fixes the *character* layer while the measured defect was
in the *order* layer — glyphs still need binding on top. Not worth it on English wikitext where
none of the three payoffs exists. First candidate the moment the tape carries a second script.

---

## 9f. Which advantages over RAG are real, and which side of the wall they live on

Asked directly — "we had a pile of advantages over RAG because of the ink; were we wrong
everywhere?" — and answerable from the decision files rather than from memory. Go to the
journal before reasoning; this section exists so the next session does not re-derive it.

```
197 EDIT_CLEAN_WIN          g_vs_rag=True  g_acquire=True  g_select=True
205 UNLEARN_PROVENANCE_WIN  g_forget=True  g_no_collateral=True  g_provenance=True
                            g_conflict=True  g_gpt_collateral=True
200 COMPOSE_CHAINS_BUT_RAG_PARITY   g_vs_rag=False
199 SEM_INV_PARTIAL         g_scale_monotone=False  inversion_at_full=False
201 SEM_HARDNEG_NO          g_invert=False  g_trend=False
202 SEM_B_NO                g_paws=False    g_inversion=False
```

**The wins are TAPE-side and no ink change can touch them.** Editing a fact, forgetting one
without collateral damage, naming the provenance of an answer, resolving a conflict: these beat
RAG because in RAG the fact also lives in the language model's weights, where it cannot be
reached. That argument does not mention the encoder. Hash ink strengthens it if anything — the
encoder has not seen the corpus either.

**The one INK-side advantage was already measured, three times, and it is not there.** "Same
meaning, same fingerprint" is 199 PARTIAL, 201 NO, 202 NO. So an untrained digest cannot lose
paraphrase invariance: `arc_enc` never demonstrated any. Anyone tempted to defend `arc_enc` on
semantic grounds should be shown 201 and 202 first.

This is also the same fact as the 289 edge ablation, seen from the other end. If `arc_enc`
carries no semantics, then of course the only channel with signal is a surface-overlap cosine,
and of course a hash can match it. Ninety stages apart, one finding.

**Scope the `rival_cos` risk honestly.** It threatens ONE verb. `count` and `compare` are exact
algebra at 1.0 with zero parameters and no `5+` cap — a 1-NN has nothing to say about them —
and the answer is a slot value, so hallucination stays impossible by construction. The worst
case is "the trained judge is decoration on `lookup`", which leaves a knowledge system that is
exact, traceable, editable, forgetful on request and script-agnostic. That is not the goal, and
the verdict must say `RETRIEVAL_NOT_MIND` when it happens, but it is not "wrong everywhere".

---

## 10. How to work with this user

- They are a solo researcher, Russian-speaking, running everything locally, and **tokens are
  expensive** — answer briefly and put the reasoning in commit messages and file docstrings
  instead.
- They catch real defects (the value-head width bug, the frame-whitelist brittleness, the
  encoder window overflow, the pooled-vector weakness). **Take their objections seriously; they
  have been right more than once.**
- They want measurement, not reassurance. State what a number falsifies before the run, and say
  plainly when a prediction failed — that has happened twice and both times it mattered.
