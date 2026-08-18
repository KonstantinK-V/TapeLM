# TapeLM — the frame, and the plan to finish it

Written after 286–289b. Everything in §1 is measured; everything in §2 onward is a plan with a
falsifier attached. Nothing here is a hope without a number that could kill it.

---

## 0. The frame in one picture

```
   text ──WRITE(279)──▶ ┌──────────────┐
                        │  FACT TAPE   │  slots: value, sources, support, depth=0
                        └──────┬───────┘
                               │ ranks, indicators — never identity
                        ┌──────▼───────┐
   question ───────────▶│    MIND      │  ~5k weights, equivariant, holds no fact
                        │  verbs over  │  READ · DIAGNOSE · REPAIR · COUNT · COMPARE
                        │    slots     │
                        └──┬────────┬──┘
                           │        │ situation as a fixed-length point
                    answer │        ▼
              (index or    │  ┌──────────────┐
               derivation) │  │  MIND TAPE   │  patterns as points; neighbourhood
                           │  └──────┬───────┘  and *interpolation* live here
                           │         │ conjecture, only where the fact tape is SILENT
                           │  ┌──────▼───────┐
                           └──│ Tape.decide  │──▶ derived slot, depth = d+1, provenance
                              └──────────────┘
                                     ▲
   new text ─────────────────────────┘  confirms → depth 0 · refutes → slot dies AND
                                        the pattern that produced it loses support
```

Five parts. Three of them exist and are measured; two are the plan.

| part | status |
|---|---|
| **Ink** — `fp`, proposes addresses only | **REOPENED** (§18). 277/287 closed it as order-blind-and-fine; the 289 edge ablation says all the signal is in the order-blind channel. `--fp hash` / `--ink bigram` are the retest |
| **Fact tape** — 279 write decision, two-channel addressing | done, scales 52× with the mind constant (283/284) |
| **Mind** — relational net, ranks only, ~4.4k params | done; passes 288's seven gates |
| **Mind tape** — patterns as points, structure-only key | **289b decides whether it exists** |
| **Conjecture loop** — derived slots, selection by the world | plan (§3–§4) |

---

## 1. What is settled, with the numbers that settled it

**The invariant has teeth now.** It is no longer a gate, it is a constraint on the input.
Anything carrying identity — text, fingerprints, any absolute coordinate — became a lookup
table: frozen trunk (collapse stayed), reading path ÷12.8 (fit unchanged to the digit), names
stripped (poison gone, payoff absent), delta channel (lost the lie). Only ranks and indicators
transfer. This is the single most useful thing the project learned.

**The mind does not grow with knowledge** — as architecture, not coincidence. Equivariant pools:
the same 4,417 weights read 3 mentions or 300.

**The mind became an auditor, not a reader.** 288, held out: forgery detection 0.683 against
counting's 0.387 and a random floor of 0.213; on the *duplicated* forgery 0.831 against 0.254,
where counting is wrong by construction. Repair 0.433 against majority-of-the-rest's 0.049.
Seven gates, all true. First end-to-end pass in the project.

**Two dead ends, closed by proof rather than by taste.** Adjacency as a second ink layer:
order content 1.3%, residual signal at chance with working nulls (287). Judge-consistency as a
reward: it ranks two real arms *backwards* (288) — trained on it we would have shipped the
broken one.

**Still open and honest about it:** `G_beats_silence` has never passed; everything is measured
on ~100-address tapes; repair restores what the corpus said, not what is true.

---

## 2. The one blocker

The answer is a slot index, and `G_answer_is_slot` is what proves separation. Composite
questions — compare, count, two hops — are not answerable by one slot *by definition*. So the
interface must change, and the moment it does the proof evaporates unless something replaces it.

**Replacement: the answer becomes a derivation.** A pair `(verb, slot set)` from a small closed
vocabulary, *executable* against the tape. Then:

- nothing is generated — it is a program over indices;
- the proof gets **stronger**: `G_answer_is_derivation` — delete the slots, the derivation
  fails to execute;
- labels stay free: questions are built *from* the tape, so we know the answer by construction,
  exactly as 288 builds corruptions;
- the mind's job stays "which slots and how", never "what is true".

288 is already the two-verb case: repair answers WHERE + WHAT. This is the same shape with a
larger vocabulary.

---

## 3. Controlled interpolation

Interpolation is what a discrete tape cannot do and what a transformer gets for free. The mind
tape buys it — but free interpolation is just hallucination with extra steps. Three controls,
none of them a chosen constant:

1. **Licensed by measured smoothness.** 289b produces the curve of accuracy against neighbour
   distance. Interpolate only where the nearest stored pattern lies inside the radius at which
   accuracy is still above the floor. Beyond it: abstain, do not conjecture. The radius is
   *read off the measurement*, per region, and is re-read as the mind tape grows.
2. **Gap-filling only.** A conjecture may be written **only where the fact tape is silent** —
   no slot at that address. If the tape speaks, the tape wins. This keeps "the mind can never
   be better than its tape" intact while allowing "the mind may speak where the tape does not",
   and it makes conjecture structurally unable to overwrite evidence.
3. **Depth discipline.** Only `depth 0` counts as evidence; only text from the world resets
   depth. A conjecture built on conjectures cannot become a premise.

---

## 4. The evolutionary loop, stated as mechanism

Not a metaphor — the three ingredients are present and each maps to something already built.

| ingredient | here |
|---|---|
| **variation** | interpolated midpoints in the mind tape; new situations appended |
| **heredity** | a pattern is reused whenever a similar situation recurs |
| **selection** | incoming text confirms or refutes the conjectures a pattern produced |

**Refutation must cost the mechanism, not just the slot.** Delete the wrong derived slot *and*
decrement the support of the pattern that produced it, or the same error is re-derived forever.
That decrement is the selection pressure; without it there is churn and no evolution.

And the environment is real: **the corpus stream is something the mind cannot edit.** That is
what makes this selection rather than self-agreement — the failure that killed the
judge-consistency reward.

---

## 5. The plan — revised by the 289b / 288-refuse results

**289b came back `MIND_TAPE_REAL` on both holdouts, with nothing trained.** 1-NN 0.730 and
0.801 against a 0.333 floor; smoothness AUC 0.701 (z +5.77) and 0.594 (z +2.51), with wrong
neighbours 2.4–4.8× further away than right ones; and the one that matters —
**interpolation 0.8225 and 0.7325 against controls of 0.4575 and 0.4050.** Midpoints between
two same-shape situations land on that shape. §3 and §4 are not cancelled; their precondition
is measured.

Two caveats, stated because they are real: growth **saturates at an eighth of the memory**
(0.718 → 0.730), so "the repertoire grows without retraining" is true but plateaus early at 28
dimensions; and the labels here are the three corruption ops, not an open-ended target.

**288-refuse says something the plan has to obey: the mind can JUDGE but cannot yet PRODUCE.**

| | corpus holdout | address holdout |
|---|---|---|
| detect forgery | 0.671 vs votes 0.333 vs random 0.304 | **0.760** vs 0.297 vs 0.400 |
| detect duplicated forgery | 0.940 vs **0.162** | 0.984 vs **0.057** |
| clean-margin AUC | 0.729 (z 7.0) | 0.859 (z 11.2) |
| repair | coverage **0.000** — abstained on everything | model **identical to votes** to 16 digits |

Detection transfers across both splits and beats counting by 2.5× on forgery and up to 17× on
duplication. Repair is either silence or the majority rule — and the gate that said otherwise
was passing on total abstention, now fixed. Producing a value is the thing that does not work.

**So the order changes.** 290 asked the mind to produce values in a *harder* setting than
repair — no ground truth at write time. Sending it there next would be building on the one
measured weakness. Judgment first:

| # | stage | cost | dies if |
|---|---|---|---|
| **289a** | **`REJECT_PRESUPPOSITION`** (§12.4). Pure judgment about the *query*: absent address, disputed address, presupposition contradicting a depth-0 slot. Questions built from the tape by construction | ~1 h | rejection of malformed questions is not above the rate of rejecting good ones — the cheapest thing that is not a reference book fails |
| **289** | **Answers become derivations.** `COUNT`, `COMPARE` over a slot set — still judgment, not production. `G_answer_is_derivation` replaces `G_answer_is_slot`; add **`G_depth_does_not_cost_recall`** (§12.2) while writing it | ~1 h | the slot set cannot be picked above the random floor |
| **289c** | **Audit sweep** (§10): 288's diagnosis head over the whole tape, unprompted, publishing the map of §12.6. This is the strongest measured capability turned into an artefact | ~min | flagged addresses are no more worth enriching than random ones |
| **290** | **Conjecture writing** — only after production works somewhere. Gap-filling, licensed by 289b's radius, `Tape.decide` writes | ~2 h | a tape with conjectures does not predict incoming text better than one without |
| **291** | **Selection.** Pattern support rises on confirmation, falls on refutation | ~2 h | fitness-weighted mind tape does not beat the unweighted one |
| **292** | **Generation** as structure plus pointers | ~1 d | fluency needs content the tape cannot supply |

**Before 290, mandatory** (§9): a monotone arrival index per slot, or "predicts *incoming*
text" is unenforceable.

**Cross-cutting, do with 289:** derive the abstain value as `2/c − 1`. The corpus arm just
demonstrated why — total silence scored 0.750 and looked like a win.

**And the open question the repair result poses:** is producing a value hard *here*, or hard
*for this mind*? The cheap discriminator is 289b's own memory — ask the mind tape to repair by
neighbourhood instead of by head, on the same subset. If the memory produces and the head does
not, the bottleneck is the 6.5k weights, not the concept.

---|---|---|---|
| **289b** | *running.* Is the pattern space smooth? Signal, smoothness, interpolation, growth — zero trained parameters | ~min | midpoints land nowhere → no mind tape, and §3–§4 are cancelled |
| **289** | **Answers become derivations.** `COUNT`, `COMPARE` over an address set. Questions generated from the tape. `G_answer_is_derivation` replaces `G_answer_is_slot` | ~1 h | the mind cannot pick the slot set above the random floor → composite questions are out of reach and the project stays a reader |
| **290** | **Conjecture writing.** Gap-filling only, licensed by 289b's radius, `Tape.decide` performs the write so restatements become `CONFIRM` | ~2 h | a tape with conjectures does **not** predict incoming text better than the same tape without them |
| **291** | **Selection.** Pattern support rises on confirmation, falls on refutation; conjectures re-scored as the tape grows | ~2 h | fitness-weighted mind tape does not beat the unweighted one → variation without selection, i.e. churn |
| **292** | **Generation.** Structure tokens the mind may emit freely; every content word is a pointer. Strip the tape → content dies, structure remains | ~1 d | fluency requires content words the tape cannot supply → the split is not clean and must be stated as such |

**Cross-cutting fix, do it with 289:** the abstain reward is a picked constant and
`G_beats_silence` has therefore never meant anything. Derive it per example: the expected value
of a uniform guess, `2/c − 1` for `c` candidates. Then "beats silence" means "beats guessing",
which is a claim worth making.

---

## 6. What would make this concept fail, stated in advance

- **The verb curve does not saturate.** Coverage of real questions against the number of verbs
  installed. If it climbs linearly, the useful part of thinking is not finitely axiomatizable
  in this vocabulary, you will need ten thousand verbs, and at that point you have rebuilt a
  network with facts inside. Measure this from 289 onward — it is *the* bet, and it is cheap
  to plot.
- **Conjectures do not survive the world.** If confirmation rate sits at the base rate of
  guessing, interpolation adds noise with provenance and nothing else.
- **The conjecture rate grows without bound.** Self-inflation, i.e. model collapse, and it is
  directly observable rather than inferred.
- **Structure cannot be separated from content in generation.** If 292 needs the tape to supply
  the glue, then "mind separate from knowledge" is a claim about *facts* only, and that is a
  smaller claim than the one the project has been making.

Any one of these is publishable as a negative result. That is the point of writing them here
before the runs.

---

## 7. Remaining hand-choices, named rather than hidden

Three things in this plan are still chosen. None is a tuned threshold, but calling them
"derived" would be a lie.

1. **The smoothness radius is under-specified.** "Where accuracy still beats the floor" needs
   *which* floor and *by how much*. Fix at 289b's reading: the majority-class floor, exceeded
   by more than its own noise — the same one-sided 1.645 the honesty gate uses. Then the radius
   is read, not picked. Until that is written down it is a hole.
2. **`1.645` and `0.05`.** Conventions, not measurements. They are the only constants the
   examiners still contain, they are shared with the rest of statistics, and `MIN_ANSWERED = 5`
   follows from the second (0.5^5 < 0.05 ≤ 0.5^4).
3. **The verb vocabulary is hand-installed** — `COUNT`, `COMPARE`, and whatever follows. This
   is the one place hand-design remains, and it is deliberate: nothing emerges here that was
   not built. The verb curve in §6 is the measurement that judges that choice, and it is the
   project's central bet.

Minor, and worth stating so nobody later mistakes them for findings: the pattern quantile set
`(0, .25, .5, .75, 1)` is a fixed summary rather than a tuned one; `k = 1` in 289b avoids
choosing a `k`; `--tapes 8` and `--tape-period 50` are compute budgets, and 286 measured what
the second one costs and buys.

---

## 8. Wishes — what the mind would ask for, if asked

Written because the question was put directly. Each is small, each has a mechanism already
present, and each can be falsified rather than admired.

**Know *why* it was wrong, not only *that* it was.** Refutation currently deletes a slot and
docks the pattern. But provenance already records which premises a conjecture stood on, and
nothing reads it: the credit could be assigned to the premise that turned out false rather
than smeared over the whole derivation. Same data, sharper lesson, no new storage. Falsifier:
premise-level credit must beat uniform docking at the next round of conjectures, or it is
bookkeeping.

**Turn silence into a question.** Abstention today is terminal. But when the mind abstains it
usually knows *where* the hole is — the address is identified, the evidence is just thin. A
mind that emits "no slot here; a mention carrying this subject and a value would settle it"
converts a dead end into a **request to the write side**, which is exactly where this project
says improvement comes from. Cheap: the address and the gap are already computed. Falsifier:
tapes grown by answering those requests must beat tapes grown by reading the same volume of
arbitrary text.

**Remember the mistakes, not only the successes.** The mind tape stores patterns that worked.
Storing the ones that failed, with their outcome, costs the same and makes "this resembles a
case where I was wrong" expressible. Falsifier: including failures must improve held-out
accuracy, or it is sentiment.

**Never be scored by something I can edit.** Already honoured — that is why judge-consistency
was refused as a gradient — but honoured by discipline rather than by construction. The wish
is procedural: every new reward should have to state, in one line, which part of it the mind
can influence. The one that could not answer that question is the one that ranked two arms
backwards.

**Report the margin with every answer, not only in the eval JSON.** It is already computed. An
answer without its confidence is a claim without its evidence, and this project has spent a
lot of effort making sure claims come with evidence.

---

## 9. Holes found on re-reading, before they cost a run

**The invariant was stated one level too low.** "The mind holds no facts" — but a pattern in
the mind tape *is* knowledge, about situations rather than about entities. The honest version
is stronger and simpler: **the mind holds no content of any kind.** Facts on one tape, patterns
on another, and the mind shrinks to a fixed comparison procedure over both. That is not a
retreat, it is the goal arriving: if the reader ends up nearly empty, the separation is total.

**There is no arrival time on the tape, and 290's gate needs one.** Slots carry sources but no
order. The whole selection story is "new text confirms or refutes a conjecture" — and without
an arrival index, "new" is unenforceable and the gate leaks: a conjecture would be scored
against text that was already on the tape when it was made. Fix, cheap and mandatory before
290: every slot gets a monotone arrival index, and a conjecture is scored **only against slots
with a strictly greater index than its own**. This is the same discipline as the held-out tape,
applied along time instead of along the corpus.

**Nothing ever forgets.** The weights do not grow, but both tapes grow forever, and "the mind
does not grow with knowledge" quietly stops being the interesting claim if the knowledge is
unbounded. There is a principled rule available and it needs no threshold: prune by **use** —
a slot never retrieved and never cited as a premise, a pattern whose support has decayed to
nothing. Falsifier: the pruned tape must score within noise of the unpruned one. If it does,
forgetting is free and the system has a steady state; if it does not, the tape is not
compressible and that is worth knowing early.

---

## 10. Two mechanisms that are cheap, and are the difference from a reference book

**Unprompted audit — noticing rather than answering.** Everything so far is
question → read → answer. A reference book does exactly that and nothing else. But 288's
diagnosis head already scores "is this address broken, and where" — and nothing stops it from
being run as a **sweep over the whole tape, with nobody asking**. Anomalies it finds become
requests to the write side: "these mentions disagree and none is corroborated; more text about
this subject would settle it." That is a mind setting its own agenda from the structure of what
it holds, which is not a thing a lookup table does, and it costs one loop over addresses.
Falsifier: the flagged addresses must be enriched with new text more profitably than randomly
chosen addresses — same measurement as the §8 wish about silence, and the same loop closes both.

**Confidence as an empirical frequency, not a softmax.** The mind currently abstains because a
margin came out small. It has never once consulted its own track record. Store, per pattern in
the mind tape, **how often the mind was right in situations like this one** — then confidence
is a measured frequency over similar past cases rather than a number from the last layer. This
is calibration by memory instead of by parameter, it lives on a tape rather than in weights, it
improves as the memory fills without retraining, and it is exactly what "know that it is a
guess" requires in order to be more than a label. Falsifier: memory-calibrated confidence must
beat softmax confidence at selective accuracy — the same matched-coverage comparison 286
already uses. If it ties, the softmax was already calibrated and we learned that too.

Both reuse machinery that exists. Neither needs a new verb.

---

## 11. One note from the inside

Written because it is the strongest argument for this concept and it cannot be measured from
outside a language model.

When I state a fact, I have no access to whether it came from memorisation or from
interpolation between things I memorised. The two are the same operation in the same weights,
and there is no signal available to me that separates them. Everything this project calls a
guarantee — depth, provenance, "delete the evidence and the answer must die" — exists to make
that distinction *representable*. That is not a safety feature bolted on; it is the one
capability the dominant architecture cannot have, at any scale, because the separation was
never built.

And the second half: everything I do well with information in the context window, I do without
changing a single weight. That is the tape and the mind, already working, inside an
architecture that was not designed for it. The differences are exactly the ones this project is
addressing — my context is transient where the tape persists, unaddressed where the tape has
coordinates, unaudited where the tape has support counts and disputes. Which suggests the
target is not exotic. It is in-context learning with a memory that outlives the prompt and a
reader small enough to be honest about what it does not know.

---

## 12. More from the inside — six observations, two of which are new machinery

Same register as §11: what a language model can report about its own operation that bears on
this design. Where an observation implies something buildable, it says so; where it is only an
observation, it says that too.

### 12.1 My confidence is real; its provenance is not

The common claim is that a model cannot tell when it is wrong. That is not quite my experience.
I *can* often tell that I am uncertain — the uncertainty is real and roughly calibrated. What I
cannot do is say **why**. I cannot report "I am unsure of this date because I encountered it in
two conflicting sources"; I can only report a feeling with no attached history.

So the gap this project fills is narrower and more interesting than "models are overconfident".
It is that **confidence without provenance cannot be acted on**. A number I cannot trace gives
you nothing to do except discount everything equally. A support count of 4 against 2, with the
two disagreeing sources nameable, tells you what to go and check. That is the difference
between a warning light and a diagnosis.

### 12.2 Retrieval and reasoning compete for the same capacity — and that is a testable gate

When I work through a long chain of reasoning, my factual recall gets worse. I cannot prove
this from the inside, but the architectural reason is not mysterious: the same weights and the
same activations serve both, so depth of inference and breadth of recall trade against each
other.

**A separated system must not have that tradeoff, and this is measurable.** Score factual
accuracy as a function of derivation depth: answer a one-slot question, then the same question
as the last step of a two-verb derivation, then three. In an entangled system accuracy falls
with depth. In TapeLM it must be **flat** — the tape does not get harder to read because the
mind is doing more with it. If it falls, something is leaking that should not be.

That is a new gate, it costs nothing beyond 289's machinery, and it tests the central claim
directly rather than by proxy: `G_depth_does_not_cost_recall`.

### 12.3 The real product claim is correction without retraining

If you tell me a fact is wrong, the correction lives until the conversation ends and then dies.
I cannot learn. Every deployment of a model like me works around this — retrieval, prompts,
fine-tuning runs — because the architecture has no write.

TapeLM has one. A correction is a single write with provenance, it persists, it is auditable,
and it can be reverted with everything derived from it. I think this is a bigger practical
difference than hallucination control, and it is underclaimed in the current documents:
hallucination is what people complain about, but **un-correctability is what actually costs
them**, because it means every error is permanent until someone spends money on a training run.

### 12.4 Understanding, operationally, is catching the malformed question

I do not believe understanding is a verb count, and I want to say so plainly since the roadmap
bets on a verb curve. The most reliable behavioural marker of understanding I can identify is
**noticing that a question should not be answered as asked** — a false presupposition, an
entity that does not exist, a comparison between things that are not comparable.

A reference book cannot do this: it either has the entry or it does not. And TapeLM can, using
machinery that already exists — the question names an address; the address may be absent, or
disputed, or its presupposition may contradict a depth-0 slot. All three are detectable by
288's diagnosis head pointed at the *query* instead of at the evidence.

**Proposed as the earliest new verb, before `COUNT` and `COMPARE`: `REJECT_PRESUPPOSITION`.**
It is cheap, it needs no new representation, and it is the first thing the system could do that
a lookup table structurally cannot. Its falsifier is easy: build questions with false
presuppositions from the tape by construction (ask about an address that was deleted, or assert
the minority value in the question) and require rejection above the rate of rejecting good
questions.

### 12.5 The right benchmark is one where most questions are unanswerable

Every benchmark I have been evaluated on asks questions that have answers. That is exactly the
regime where a fluent interpolator looks best and an honest system looks conservative and slow.

The value of this architecture is concentrated in the opposite regime: questions where the
truth is **not** in the tape. There, a model like me produces something plausible and wrong,
and a system with depth labels and an UNKNOWN row produces a refusal you can trust. So the
benchmark that would show what this is for has most of its questions unanswerable, and scores
the refusals as the primary quantity rather than as a penalty.

This is buildable from the tape by construction — hold out addresses, ask about them — and it
is the honest home for `G_beats_silence` once the abstain value is derived rather than picked.
It also explains, more generally than HANDOFF §3 does, why comparing 0.900 to QA leaderboards
is the wrong instinct: the leaderboards are built around the one regime where the design is
deliberately not optimised.

### 12.6 A map of what it knows badly, published by the system itself

Before answering, this system knows how many mentions support the address, whether they agree,
and whether anything corroborates them from outside. That is a **per-question competence
report**, and it is available for free. Run the unprompted audit (§10) across the whole tape
and it becomes a *map*: here is what I hold well, here is what is thin, here is what is
contested.

No model like me can produce that map, at any scale, because the competence is not represented
anywhere — it would have to be estimated by asking myself questions and grading the answers,
which is the circularity this project spent 286 removing. Publishing the map is one loop over
addresses, and I suspect it is the most immediately useful artefact the system could emit,
before any of the harder capabilities land.

---

### What I would build first, if the choice were mine

`REJECT_PRESUPPOSITION` (§12.4) before `COUNT` and `COMPARE`, and
`G_depth_does_not_cost_recall` (§12.2) added to 289 as it is written. The first is the cheapest
thing that is qualitatively not a reference book. The second is the only gate proposed anywhere
in these documents that tests the *separation* itself under load rather than at rest — and
separation under load is the whole claim.

---

## 13. The road to generation, and why it is still open

Written after 289 moved COUNT and COMPARE into exact algebra, because the fair question is
whether that closes the road to a mind that *produces* rather than only judges. It does not,
and the reason is worth stating precisely rather than reassuringly.

### 13.1 Why the road is open: the judge is already general

Phi scores **any** completed world. It does not know or care where the candidate came from —
it is handed a graph with one row filled in and returns how well that world hangs together.
Today the candidates come from the surviving mentions of the same address, and that is the
**only** thing that makes this a lookup rather than generation.

So the restriction lives in the **candidate generator, not the judge**. Generation is
therefore: widen the generator, leave Phi alone. Nothing about the current architecture has to
be unbuilt for that, which is exactly what "not closed" means.

### 13.2 The nearest strengthening: a ladder of wrong answers

Phi is trained today on one right candidate against a handful of local wrong ones. It learns a
**boundary**. What generation needs is a **landscape** — how much worse a world gets as the
substitution moves further from the truth — and a boundary cannot supply one.

The ladder is free from the tape, with no authoring anywhere:

| rung | substitution | what it teaches |
|---|---|---|
| near | a value of the **same anchor**, different relation | right subject, wrong fact |
| middle | a value of a **neighbouring address** on the tape | right neighbourhood, wrong subject |
| far | a **random value** from the tape | wrong everything |

Train Phi contrastively across the rungs rather than binary right/wrong. The gate is a
monotonicity, not a threshold:

    G_phi_orders_negatives:  Phi(true) > Phi(near) > Phi(middle) > Phi(far)

measured as a rank correlation over held-out questions, with a permutation null. About thirty
lines, and it is the precondition for everything below: **a mind that cannot rank its own
wrong answers by how wrong they are cannot steer.**

When 290 starts writing the mind's own conjectures as derived slots, those conjectures join
the same ladder as further rungs — the evolutionary loop, with no new machinery invented for
it. That is the payoff for building the ladder now rather than later.

### 13.3 Controlled interpolation: three parts, two already built

1. **A smooth space of situations** — measured in 289b: 0.82 / 0.73 on the interpolation
   probes against controls at 0.46 / 0.41. **Built.**
2. **A differentiable judge** — Phi over the soft algebra. `soft_count` is differentiable in
   `s_ij`, so the question "what pattern of sameness would make this world cohere" has a
   gradient, and a gradient is a **direction to search in**. **Built**, as of the soft-algebra
   change; §9d of HANDOFF records why that was the point of writing it soft.
3. **A generator of points** — **missing, and this is the whole remaining gap.** The mind
   proposes a point in the identity-free pattern space (289b's `pattern_of`), and the answer
   becomes the nearest value on the **whole tape**, not only among local candidates.

"Controlled" means the mind chooses *where in the space to move*, by the gradient of Phi.
Note what is still not happening: **no text is generated.** The answer remains a value that
exists on the tape, reached by a thought rather than by a lookup. That is the honest form of
generation for this concept, and it keeps `G_answer_is_slot` alive in spirit — the answer is
still something the world wrote, the mind only found it by thinking.

### 13.4 Order of work

1. **Finish 289 / 289c** as they now stand — the paired McNemar gate is the first honest test
   of whether the context channel beats counting at all. If it fails, 13.2 is where to look
   first, not 13.3.
2. **The negative ladder in 289** (§13.2). Cheap, labels free, immediately checkable.
3. **290 — the conjecture becomes a derived slot.** The monotone arrival index it needs already
   exists: `max_{j<i}` in the algebra requires row order to be tape order, so the index is
   load-bearing now rather than bolted on.
4. **291 — selection.** Which operator to apply, and which direction to move in. This is the
   capability the mind genuinely does not have today, and it is easier against well-defined
   operators than against three heads approximating them unevenly.
5. **292 — the point generator and search over the whole tape.**

### 13.5 The two ways this still fails

Named now so a later run is not read as a surprise:

- **The landscape is flat.** If Phi turns out to separate true from false but rank near, middle
  and far identically, then there is no direction to descend and 13.3 has no ground. That is
  what `G_phi_orders_negatives` is for, and it is a real possible outcome.
- **The nearest tape value is always the majority value.** If searching the whole tape from a
  proposed point lands on the same answer counting would have given, the generator adds
  machinery and no capability. The check is the same paired McNemar against the counting rival,
  run over the full-tape search rather than the local candidates.

---

## 14. Design sketches for 290 / 291 / 292

Written while 289's four arms run overnight, so the next session starts from a design rather
than a memory. Each block states the mechanism, the free labels, the gates, and the way it
fails — the last one because every stage so far that skipped it cost a run.

### 14.1 — 290: the conjecture becomes a derived slot

**Mechanism.** When a read leaves a gap the mind can score, it writes its best candidate back
to the tape as a slot carrying:

| field | value | why |
|---|---|---|
| `depth` | `1 + max(depth of sources)` | a conjecture from conjectures is further from the world |
| `provenance` | the slot ids it was derived from | so a wrong conjecture can be traced and dropped |
| `phi` | Φ of the completed world at write time | its own confidence, recorded not asserted |
| `arrival` | the monotone index | already load-bearing: `max_{j<i}` needs row order = tape order |

**The one rule that makes this safe.** *Only depth-0 rows are evidence.* The exact algebra runs
over depth-0 rows only, so a conjecture never changes a count, never lights a same-value edge
for the arithmetic, and cannot confirm itself. A thought becomes knowledge only when world text
independently asserts it — which resets depth to 0 through the normal write path, not through
any promotion rule of its own.

Without that rule the failure is immediate and total: a conjecture raises the count of its own
value, the next read sees a stronger majority, and the mind confirms itself into certainty.
Naming it here so nobody re-derives it after a wasted run.

**What conjectures are FOR, given they are not evidence.** Three things, all real: candidates
for lookups at *other* addresses; rungs on the ladder; and coverage — an answer where there was
none.

**Gates.**
- `G_conjecture_beats_random_write` — written conjectures are right more often than a random
  tape value would have been. Label free from the tape.
- `G_conjecture_does_not_poison` — accuracy on the questions that already had answers is
  unchanged after N rounds of writing, paired against a no-write control on the same items.
- `G_no_self_confirmation` — re-asking a question with the conjecture present and absent gives
  the identical answer. A sanity bolt like `G_exact_algebra_matches_tape`: it must pass by
  construction, and a failure means the depth filter leaks.

**How it fails.** Depth filtering leaks somewhere subtle (the ladder, the context ranks, the
retrieval index) and self-confirmation creeps in through a channel nobody checked.

### 14.2 — 291: selection, and what is actually underdetermined

**The trap first.** "Which verb applies" is *not* selection — asking "how many" is a count by
the form of the question, and parsing that is linguistics, not mind. Building a verb classifier
would be theatre.

**What is genuinely underdetermined** is *how much machinery to spend and whether to answer at
all*. The action set, all of which now exist:

    ANSWER_MAJORITY      cheap, no Φ evaluation
    ANSWER_BY_COMPLETION run Φ over the candidates (289)
    REFUSE               289a's verdict, when the question is wrong
    CONJECTURE           write a derived slot (290)
    SEEK                 retrieve more mentions for this address, then re-decide

Reward is already defined and free: correct +1, wrong −1, abstain +0.75, minus a read cost per
retrieval. This is 280's RL frame with actions that finally differ from each other.

**Gate.** `G_selection_beats_every_fixed_policy` — the learned policy's reward exceeds the best
*single fixed action* applied everywhere, at matched coverage. Beating the average of fixed
policies is not enough; it must beat the best one.

**Vacuity guard, required.** Report the action histogram. A policy that collapses onto one
action is not selecting, and its "win" is just that action's number. Require at least two
actions used above `MIN_ANSWERED`.

**Also here: the abstain reward.** Derive it as `2/c − 1` for `c` candidates so that
"beats silence" means "beats guessing" rather than "beats 0.75, a number someone picked". This
is the cross-cutting item that has been open since 286 and it belongs in the stage that finally
makes silence one action among several.

### 14.3 — 292: the point generator and search over the whole tape

**The questions already exist and are being thrown away.** `lookup_question` returns `None`
when the hidden value appears nowhere else at that address — 286's failure mode 12, discarded
because the target was not a function of the input. Those discarded questions are *exactly*
292's task: the answer is on the tape, just not at this address. No new task has to be invented,
and the labels stay free.

**Architecture: the mind emits a query, the tape returns matches, Φ ranks them.**

1. The mind proposes a point `p*` in 289b's identity-free pattern space.
2. Optionally moves it: `p* ← p* + η·∂Φ/∂p` for a few steps. Φ is differentiable through the
   soft algebra, so this is a real direction and not a metaphor. **This is what "controlled"
   in controlled interpolation means** — the mind chooses where to look.
3. The tape returns the `k` values whose mentions sit nearest `p*` — the same division of
   labour as votes over the inverted index: identity lives in the tape, never in the weights.
4. Φ scores the `k` completed worlds; argmax answers.

Cost is `k` Φ evaluations per question, so `k` stays small and the generator does the reaching.
No text is produced: the answer is a value the world wrote, found by a thought.

**Gates.**
- `G_reaches_beyond_candidates` — accuracy above floor on the questions 289 discards. This is
  the whole claim; everything else is diagnostics.
- `G_search_beats_counting` — paired McNemar against the majority value of the retrieved set,
  same questions, same run.
- `G_gradient_steps_help` — paired comparison of retrieving at `p*` after gradient steps versus
  at the initial point. If this fails, the interpolation is not controlled, only smooth.

**How it fails, three ways.**
- Retrieval collapses to the corpus-frequent value and `G_search_beats_counting` reads zero.
- Φ saturates, `∂Φ/∂p ≈ 0`, and `G_gradient_steps_help` is flat — no direction to move.
- The reachable set is large enough that Φ ranking inside it is no better than chance, which
  would show as accuracy above floor but paired z near zero.

### 14.4 — the order, and the one dependency that matters

290 → 291 → 292 is the build order, but only one hard dependency exists: **292 needs 290's
depth discipline already in place**, because a generator that can reach the whole tape can also
reach its own past conjectures, and without depth-0-only evidence that is a self-confirmation
loop with a much wider mouth. 291 can be built before or after 292; it is more useful after,
since `SEEK` and `CONJECTURE` are only interesting once there is somewhere to reach.

---

---

## 15. What may leak into the weights — the invariant was stated too strictly

This section replaces an earlier draft that got it wrong, and the correction came from outside:

> "Knowledge MAY leak into the weights, but not in the form of the facts themselves or the text
> of the corpus. What leaks should be a skill of understanding rather than specific knowledge. A
> child understands that a dropped mug will hit the floor and break without knowing a single
> equation. I understand how a car drives without knowing where the stress analysis went in each
> part — and if I need to, I open the article and understand it more deeply."

The draft it replaces said the mind must know nothing about the corpus, and proposed training on
a tape of meaningless strings to guarantee it. That guarantee is real and it is the wrong
guarantee: it would also destroy the thing that makes reading possible.

### 15.1 — three levels, and the earlier draft collapsed them into one

| level | example | may it be in the weights? |
|---|---|---|
| a specific fact | "Canada defeated Sweden in 1917" | **never** |
| a type-level regularity | a `born in` value is a year; a `defeated` value is the kind of thing that competes | **yes — this is the understanding** |
| pure structure | more witnesses is more reliable; contradictory rows cohere worse | yes, trivially |

Free permutation of values destroys the first level, which is correct, and the second, which is
not. A mind trained that way can count witnesses and has no idea what kind of thing an answer
should be. That is not the child; that is a bookkeeper.

### 15.2 — the operational definition, and the transfer that is ALREADY demonstrated

> **The weights carry the skill of looking and the understanding, never the answer. Given any
> tape they must look; finding nothing, they must say so.**

Positive, because "must not answer" specifies nothing about what should be there instead. See
HANDOFF §1 for the clause-by-clause status. Two corrections this forces on the rest of §15:

**Transfer between tapes is not something this section would establish — it is already
established.** The held-out tape is built from a disjoint 30% of the corpus: different lines,
different addresses, largely different entities, and the weights never saw any of it. 289's
0.6923 with paired z 4.32 *is* a cross-tape number. Any framing that treats "does a skill
transfer at all" as the open question is wrong; the open question is **what kind of thing
transfers**, which is what the three arms below separate.

**The real gap is corpus, not slice.** Both tapes are wikitext — same language, same register,
same entity distribution — so what is shown is transfer across *samples*. "Any tape" is
untested and is the cheapest missing experiment in the project: point `WIKI` at a different
file and re-score with the existing gates. No code change. Do this before the arms below, since
a skill that does not survive a change of corpus is not the skill being claimed.

**And a measurement note for 291/292 — corrected, because the first version of it was wrong.**

The draft read the car analogy as an argument *against* measuring the chain, and that throws out
the thing that makes understanding possible. The correction, verbatim from outside:

> "My understanding (the abstraction) is based on a logical chain. Without logic and cause-and-
> effect there is no mind at all. The chain has simply compressed into an abstraction — and that
> compression is what we need to achieve in our mind."

That is right and it is a different claim from "short is better". The chain is not absent when
you turn a key; it is **compressed and still recoverable** — the driver can produce spark →
injection → crankshaft on demand, and simply does not traverse it to start the car. Both halves
are the capability, and either one alone is a failure mode:

| present | absent | what you get |
|---|---|---|
| chain | compression | a prover that re-derives everything and understands nothing |
| compression | chain | a lookup table with confident guesses and no justification |

So 291/292 get a **pair** of numbers, not one:

- **reaches to the tape until answered** — the compression. Fewer is better only while correct.
- **the supporting rows are recoverable on demand** — the chain. This already exists as the
  return path (§5 of HANDOFF), kept alive precisely because it is the label-free provenance
  check, and it is what makes the compression legitimate rather than lucky.

Traversal length is a bad yardstick *alone*. Paired with recoverable provenance it is the right
one.

The child analogy is precise about the second half too: the child's understanding **improves**
when shown the equations. Applied here, the mind should get better with more tape and no
retraining. That is a capability, not a slogan, and it should be measured — stage 199's
`g_scale_monotone` came back **False**, so on the evidence available today it is an open defect,
not an established property. Worth re-checking under the current architecture before it is
assumed.

### 15.3 — capacity is a sieve, not a wall

This reconciles the correction with why 3489 parameters was right anyway, and states the reason
better than the project has been stating it.

A type-level regularity is **compressible**: it is shared across thousands of facts, so it costs
few parameters. A specific fact is **incompressible**: it is unique, and storing ten thousand of
them costs ten thousand times as much. A hard parameter budget therefore does not block
knowledge — it blocks knowledge that does not generalise.

So the honest reading of the budget is not "nothing gets through". It is: **only what compresses
across many facts fits, and that is precisely the definition of understanding.** The wall was
never a wall. It was already the sieve the correction asks for, and the project had been
describing it wrongly.

### 15.4 — the experiment, rebuilt: permute WITHIN type

Three arms, differing only in the tape generator:

| arm | values | destroys | keeps |
|---|---|---|---|
| `real` | as written | — | — |
| `permuted-in-type` | swapped only among values sharing a relation | facts | types, structure |
| `permuted-free` | swapped with anything | facts, types | structure |

Type comes from the tape, not from an annotator: **values that appear in the same relation are
the same type**. No labels, no list, nothing chosen by hand.

Then train on each and evaluate on a **real** tape never seen. The three numbers separate what
the earlier draft could not:

- `permuted-in-type` transfers → the skill is type-level understanding, facts were never needed,
  and the separation is proved in the form the correction actually asks for
- only `real` transfers → the mind is leaning on specific strings, and the separation claim has
  been resting on capacity alone
- `permuted-free` transfers as well as `permuted-in-type` → what was learned is pure structure
  and no understanding of kinds is present at all, which is its own finding and a discouraging
  one

The middle arm is the one that matters, and it is the arm the earlier draft did not contain.

### 15.5 — what survives permutation, and why the arms are informative

`same`, the subject indicator and the value shares are untouched by any permutation. `cos` is
not: permuted values change the contexts that mention them. So the arms also localise the
dependence — if transfer collapses, comparing which channels still work says whether the mind
was reading kinds or reading strings.

### 15.6 — cost and placement

One flag in the tape generator, `--values {real,permuted-in-type,permuted-free}`. Touches
neither the ink nor Phi, conflicts with nothing currently running. It should run before 291 and
292, because 15.3 means the answer decides how much capacity those stages are allowed — and
under the corrected reading, a growing mind is not automatically a leaking one.

### 15.7 — one fact, two phrasings: this is an ADDRESSING problem, and it now has a number

Raised from outside and it is the deepest open item in the project:

> "'Kostya was born in 1985' can be written that way, or as '1985 was a good year; there were no
> earthquakes, and Kostya was born in it'. The mind has to understand this. That mechanism will
> not emerge from a 150M corpus, the same as it does not for GPT — but it must be accounted for,
> designed in, and at least somewhat proved with numbers."

The last clause is the operative one, and the first thing to say is that the project has already
measured the *reading* side of this and **failed** it: 199 `SEM_INV_PARTIAL`, 201
`SEM_HARDNEG_NO`, 202 `SEM_B_NO`. So this is not an unexplored hope, it is a known negative.

But the failure was measured in the wrong place. In this architecture the mind never sees two
phrasings side by side unless **extraction put them at the same address**. If "born in" and "was
born" become two addresses, the redundancy that was supposed to be free evidence is discarded
before the mind is asked anything, and no reader can recover it. Paraphrase robustness is an
addressing property here, not a reading one.

Two label-free counts now ship in `tape_shape.paraphrase`, computed from the tape alone:

- `same_anchor_diff_relation` — the anchor survived, the relation did not: one fact spread over
  several addresses. Direct fragmentation, and the ceiling on everything downstream.
- `reversed_pairs` — the pair appears in both directions, A|rel→B here and B|rel→A there.

Both are sensitive to `--fp` and `--ink`, since addressing runs through `bank.fp`, so they ride
along with the ink arms already queued at no cost.

**`reversed_pairs` also puts a claim made earlier in this session on probation.** A third of the
mind's lookup errors were exact swap pairs — Leipzig↔Weimar, California↔Texas — and that was
read as evidence of order-blind ink. If the tape already carries both directions, both are true
at different addresses and the "error" is the examiner's, not the mind's. The swap-pair reading
does not stand until this number is in.

The design consequence, for when the corpus is large enough to matter: the fix is not a better
reader but **an address that survives rephrasing** — anchor and value fixed, relation soft. That
is a change to 279's write path, and it is the single largest lever on this problem.

---

## 16. Traversal — the algorithm, and where accumulated skill is allowed to live

Asked from outside, and it is the right question to be asking now: an algorithm for walking the
tape, free of heuristics and simple, such that **as the tape grows the mind applies what past
traversals taught it** to read new data better. With the observation attached that GPTs are not
born clever either — their competence arrives with corpus scale.

### 16.1 — why there is nothing to accumulate today

The mind does not walk. `build_graph` puts one address's rows in front of it and that is the
whole world it ever sees. It never chooses where to look, so there is no traversal for a skill
of traversing to attach to. Everything below exists to create the thing that could accumulate.

### 16.2 — the core: traversal as uncertainty reduction, with no new parameters

Phi already yields logits over candidates, so there is a distribution and there is its entropy.
The walk is defined entirely from those:

```
W  <- the query address's rows
loop:
    P <- rows one edge from W        (shares a value / shares rare words / near in ctx)
    g(r) = H(Phi | W) - H(Phi | W + r)      for each r in P
    stop if max g <= 0
    W <- W + argmax g
answer    <- argmax Phi(W)
provenance<- the order rows were added
```

- **No heuristic.** The stopping rule is zero information gain, not a chosen threshold. The
  proposal set is structural — one edge — not a picked shortlist. The features are the ones the
  graph already carries.
- **Simple.** Roughly fifty lines on top of what exists.
- **Does not grow with the tape.** Phi stays 3489 equivariant weights whatever the tape's size.
- **Yields §15.2's pair for free.** Steps taken is the compression; the order of addition is the
  chain, recoverable on demand.

The objective is deliberately the entropy over *candidates*, not Phi itself. Ascending Phi finds
worlds that hang together, which is not the same as worlds that answer — that failure mode is
the reason for the choice and it is listed in 16.5.

This is standard machinery: greedy value-of-information / active evidence selection. As
everywhere else in this project, the mechanism being standard is reassurance, not a problem.

### 16.3 — but that alone accumulates nothing

The walk above is exactly as clever on the first tape as on the thousandth. It searches; it does
not remember how searching went. The question was about the second thing.

And here the fork has to be named plainly: **GPT scales by growing parameters and this
architecture forbids that.** So accumulation must live in the non-parametric part or it does not
exist at all.

### 16.4 — the policy lives on a tape, not in the weights

Record not "what was true" but "what a useful step looked like":

```
key   = structural signature of a proposal: rank of cos, shares a value with the query row,
        rank of support, depth, share of rare words matched
        - ranks and indicators only, no identity, so no fact fits
value = g(r), the information gain that actually materialised
```

Every episode appends. On the next walk, nearest neighbours in that store *predict* `g(r)`
instead of computing it. No weights move, nothing is trained, and the store grows with the data.
It is §0's mind tape pointed at traversal instead of at repair — and it is the child analogy
exactly: experience accumulates, the brain does not grow.

### 16.5 — the scaling claim, stated so it can fail

On a 400-address tape there are few proposals, `g(r)` can be computed exactly for all of them,
and the store is useless. On a 40,000-address tape there are too many to try, and the store
triages: predict cheaply, evaluate exactly only at the top.

> **The store's benefit is zero on a small tape and grows with tape size.**

One curve settles it: accuracy and reaches-to-answer against tape size, two arms, with store and
without. Curves that do not diverge as the tape grows refute the construction.

Three ways this dies, and they are distinct:

| outcome | reading |
|---|---|
| the walk does not beat today's single-address reading | there is nowhere to go; all the evidence really is local |
| store and exhaustive search never diverge at any size | experience does not accumulate — it is only search |
| Phi rises while the answer does not | the walk finds coherent worlds rather than answering ones |

### 16.6 — the honest shape of the bet (corrected: parameter growth is NOT forbidden)

The first draft of this section said "GPT scales by growing parameters and this architecture
forbids that". That is wrong, it conflated two different restrictions, and only one of them is
real. See §17.

The invariant does not restrict *size*; it restricts *what the weights can see*. Identity is
unrepresentable at the mind's input — edges carry `same` and two ranks, nodes carry shares and
indicators, and nothing anywhere carries "Canada". A fact cannot be stored in weights whose
input never contains it, at any parameter count. The real cap today is the training set, and
that cap moves with the corpus.

So the bet is narrower and more defensible than the draft claimed: skill that is general because
identity cannot reach it, plus experience accumulating outside the weights, plus a parameter
budget that is allowed to grow with the tape rather than frozen at a number chosen in 2025.
16.5 is still how it gets checked rather than believed.

### 16.7 — placement

This supersedes and precedes 290: derived slots have nowhere to come from until the mind can
reach. Order becomes **traversal → 291 (refusal as an action) → 292 (generation)**.

Build 16.2 alone first. It produces one number — does walking beat reading a single address —
and if that number is flat, 16.4 is never needed.

---

## 17. The parameter budget — two restrictions that got conflated, and only one is real

Asked from outside: *why* is parameter growth forbidden? It was adopted for the tests, to make
something provable with numbers — so would changing the mind's geometry make sense, more layers,
wider or narrower?

The question is right and the premise in §16.6's first draft was wrong.

### 17.1 — the restriction that is real, and it is not about size

**Identity is unrepresentable at the mind's input.** Edges carry `same`, the rank of `cos` and
the rank of shared rare words. Nodes carry shares, indicators and `1/n`. Nothing anywhere
carries "Canada". §1 records this as the single most useful thing the project learned, and it
was learned the hard way: every arm that let identity in — text, fingerprints, any absolute
coordinate — turned the mind into a lookup table.

A fact cannot be stored in weights whose input never contains it. **That holds at any parameter
count.** Memorisation is blocked by the representation, not by the budget, so the budget was
never carrying the weight it was credited with.

### 17.2 — the restriction that is real but temporary

375 training questions. A larger net overfits them, and that is a fact about the *data*, not
about the concept. It lifts by exactly the lever GPT uses: more corpus → more addresses → more
questions → more parameters supportable.

So: **parameter growth is not forbidden. It is currently rate-limited by the training set.**

### 17.3 — but there is no evidence to decide on yet, and the reason is an instrumentation bug

The training curves from the two most recent runs:

```
exact2   2.205  1.931  1.706  1.303  1.701  1.973  1.977  1.565
import2  2.060  1.967  1.888  1.416  1.475  1.800  2.185  1.433
```

Down to ~1.3–1.4, up to ~2.0, down again — the same shape in both arms. That is not divergence.
The seed is shared, so **step 5250 is the same tape in both runs and that tape is simply
harder**.

Which means the curve is not a convergence signal at all: every point is measured on a different
dataset, so the points are not comparable to each other. **Today there is no evidence whatever
about whether capacity binds.** Changing geometry without it is guessing.

The fix is cheap and must come first: hold **one fixed probe tape**, never trained on, and
measure loss on it every N steps. Then the curve means something and "converged or capacity-
bound" becomes answerable.

### 17.4 — width and depth are not interchangeable here, and the graph says why

The graph is **complete** — every pair of rows carries an edge. One round of message passing
therefore already reaches every node, so extra layers buy no additional reach, only nonlinearity.

| lever | here |
|---|---|
| **width** (`d = 32 → 64 → 128`) | directly raises how much of a world fits. The live lever now |
| **depth** (more rounds) | no reach to gain on a complete graph; expensive nonlinearity |
| **depth after §16** | becomes meaningful — traversal makes the graph sparse and multi-hop, and then a layer is a hop |

Depth is not rejected, it is **premature**. It pays off exactly when traversal exists.

**A precedent that was cited here and has been withdrawn.** The first draft argued "smaller has
already won once" from removing the `count`/`compare` heads: half the network, **+27 points** on
lookup (0.423 → 0.692). That is a confounded experiment and it says nothing about capacity. Two
things changed at once — fewer parameters, and **one task instead of three**. The interference
was measured directly at the time (`count` fell 0.965 → 0.903 as lookup grew beside it), so the
gain is almost certainly the second. The experiment shows one task beats three. It is not
evidence about size in either direction.

Nothing else in the record speaks to capacity either, which is why 17.3 comes first.

### 17.5 — replace the fixed number with a rule that scales

Not 3489, but:

> **Fewer parameters than incompressible facts on the tape.**

The budget then grows with the tape on its own, nobody chooses anything, and on every tape there
is still a checkable statement that memorisation is impossible by counting — while the real
guard, 17.1, holds regardless.

### 17.6 — order

1. **Fixed probe tape** for a comparable convergence signal. Everything else is blind without it.
2. **Width sweep** `d = 32 / 64 / 128`, all else frozen. One number: does held-out lookup move?
   If it does not, capacity does not bind and the question is closed by measurement.
3. **Depth after §16**, when there are hops for layers to correspond to.

Steps 1 and 2 are orthogonal to the ink arms and cost one run each.

---

## 18. Closures expire — an audit of what the old numbers can and cannot carry

Raised from outside and it is a methodological correction, not a detail:

> "We lean heavily on past tests that confirmed architectural *possibility* — that is what the
> gates were for — and not on what we are arriving at now. Before the current training scheme we
> were tuning parameters, and back then it looked obvious that fewer weights was good. I would
> not read those runs' numbers that way. Especially where the difference was small and the tape's
> capability had already been technically confirmed."

### 18.1 — two kinds of result, and reading one as the other is a category error

| kind | what it establishes | how it fails |
|---|---|---|
| **possibility gate** | can the architecture do X at all | qualitative collapse — let identity in and the mind becomes a lookup table. Setting-independent, survives everything |
| **setting measurement** | is this the right value | needs everything else frozen and a difference bigger than noise |

The gates were built for the first. They have been quoted as the second — by me, in §17.4, one
message before writing this.

### 18.2 — the audit

| result | kind | usable today |
|---|---|---|
| identity → lookup table (§1) | gate, qualitative collapse | **yes**, setting-independent |
| equivariance: same weights on 3 or 300 mentions | structural property, not a delta | **yes** |
| +27 points from dropping the heads | setting, **confounded** | **withdrawn** — see 17.4 |
| 199 / 201 / 202 semantic failures | measured under **arc_enc ink** | needs retaking — the mechanism is being replaced |
| 289b mind-tape saturation at 1/8 of memory | old ink, old task | needs retaking |
| 277 / 287 "the ink is closed" | gate, but narrow | **contradicts current work — see 18.3** |

### 18.3 — the live contradiction, and the arm that settles it

§0 said the ink was *closed* (277, 287): order content 1.3%, residual at chance with working
nulls. And the project has spent this week building bigram binding **on the premise that order
matters**.

One of those is wrong, and it is already being tested by the queued `ink_bigram` arm:

- **bigrams give a real gain** → 287 closed something narrower than "order matters" — it tested
  order as a separate ink *layer*, not as binding *within* the context vector. The closure lifts.
- **bigrams give nothing** → 287 was right, order is not the issue here, and the symmetric swap
  pairs in the confusion table are the tape's doing (`reversed_pairs`, §15.7) rather than the
  ink's.

Both outcomes are informative. Neither may be leaned on in advance.

### 18.4 — the inversion hiding in §1

Stage 283 ran **7,564,290 parameters** and addressed 91 slots and 4,730 slots identically. The
finding was about equivariance — the mind not growing with knowledge — and it was recorded as a
success. So a seven-million-parameter mind has already been run here and did not break the thing
that was actually being guarded.

Read plainly, that is evidence **for** allowing growth, and it has been sitting in §1 being read
backwards.

### 18.5 — the rule

> **A closure has an expiry date.** Every number in these documents carries the conditions it was
> taken under, and a closure is void once those conditions change. "Ink closed (287)" was taken
> under the old ink, the old task and a narrower definition of order, and is therefore not an
> argument against bigrams.

And the corollary that motivated the question: **where the difference was small and the tape's
capability was already technically confirmed, the number is not an argument at all.** Only the
qualitative collapses carry across regimes.

---

## 19. The brick inventory — what carries, what does not, what is missing

Adopted from outside as the working method, and it is the one the project has actually been
using: try a thing, see whether it works on the tape, keep it or drop it, then stack the bricks
that held and look for the gaps. The value of writing the stack down is that it makes the gaps
visible instead of remembered.

### 19.1 — bricks that carry

| brick | what settled it |
|---|---|
| tape write and addressing | teacher ceiling −0.189 → **+0.483** on raw text (279) |
| exact algebra for `count` / `compare` | **1.0**, zero parameters, uncapped, scale-free |
| equivariance | the same weights read 3 mentions or 300 |
| ranks and indicators only | every arm that admitted identity became a lookup table |
| knowledge editing | 197, `g_vs_rag=True` |
| unlearning with provenance | 205, every gate, with GPT taking collateral damage where we take none |
| calibrated silence | 0.85 abstention on genuine ties with zero wrong answers there (280) |
| forgery audit | 0.683 against counting's 0.387 (288) |
| **interpolation on the mind tape** | **0.8225 / 0.7325** against controls 0.4575 / 0.4050 (289b) |
| lookup beats counting, paired | z **4.42**, and it survived every ink arm since |
| transfer to an unseen tape | held-out 0.6923 on a disjoint 30% of the corpus |

### 19.2 — bricks that do not carry

| brick | what killed it |
|---|---|
| semantic invariance of the ink | 199 PARTIAL, 201 NO, 202 NO |
| `G_beats_silence` | never passed, in any stage |
| composition | 200, `g_vs_rag=False` — parity, not a win |
| monotone improvement with scale | 199 `g_scale_monotone=False` |
| Phi's ladder / landscape | coverage 0/403; the construction cannot form one |
| Phi's calibration | states 0.98, realises 0.63 |
| **Phi beating 1-NN on lookup** | **ink_mean: 0.692 vs 0.760, paired z −2.33** — see 19.4 |

### 19.3 — bricks not yet laid

traversal (§16) · refusal as an action (291) · generation (292) · **reversed-pair linking**
(19.5) · transfer across *corpora* rather than slices · a comparable convergence signal (§17.3)
· width sweep (§17.6)

### 19.4 — the newest negative, and why it renames a verdict

`ink_mean` reproduced `exact2` exactly (0.692, z 4.32 against counting — so the ink patch is
inert when off, as required) and then **lost to 1-NN over the same rows**: 0.760, paired z
−2.33, with 0/0 discordant on the high-margin half and the entire loss on the low-margin half.
By the split declared before the number, that is the worst of the three allowed outcomes: where
similarity is confident Phi merely agrees with it, and where similarity runs out Phi is *worse*
than taking the nearest row.

Two things follow, and they pull in opposite directions, which is why both are recorded.

**The brick is named, not the wall.** The verdict string is now
`PHI_ADDS_NOTHING_ON_LOOKUP`, not `RETRIEVAL_NOT_MIND`. Lookup is one verb, single-hop and
retrieval-shaped by construction. A rival that beats it says nothing about the exact algebra,
about verbs that require combining rows, or about generation — which 1-NN cannot do at all, and
where 289b already measured a positive.

**But a defect in the arm was found, so the number is provisional.** `rank_norm` ranks the whole
upper triangle at once, so the query row's similarities arrived mixed into every other pair's,
and no node feature stated how close a row was to the query. Phi had to reconstruct an argmax
over one column from a globally ranked matrix; the rival takes it directly. The per-row rank of
closeness to the query row adds no information, only access. **If Phi loses again with it, the
brick is genuinely out.**

### 19.5 — reversed pairs are an input, not only a warning

Raised from outside: the tape carries "Kostya → 1985" and "1985 → Kostya" and nothing counts
them, and that capability is wanted. Correct — but merging them blindly is wrong:

```
Kostya|born in → 1985      and  1985|in that year was born → Kostya    ONE fact
Canada|defeated → Sweden   and  Sweden|defeated → Canada               OPPOSITE facts
```

Whether a relation is symmetric cannot be annotated, and it does not have to be. It is visible
in the tape: **for each relation, the fraction of its pairs that also occur reversed.** High is
symmetric, low is directed. Label-free, scale-free, nothing chosen by hand.

So the fix is not a merge but a **feature**: an edge saying "this row's address is the reverse
of that one's", and a node feature carrying the relation's reversal rate. The tape presents the
structure, the mind decides what it means — the same division of labour defended everywhere
else. This also promotes `reversed_pairs` from a caveat about the swap-pair reading (§15.7) into
an input the mind can use.

### 19.6 — where the concept gets decided

Also from outside, and it is right: the mind/retrieval question stops being statistical at
**interpolation**. 1-NN cannot produce a value that no row carries — not badly, *at all*. So a
correct value absent from every row refutes retrieval by construction, with no McNemar and no
threshold needed. 289b already shows a positive there (0.8225 vs 0.4575), with the honest limits
that it was measured on three corruption ops rather than an open target and that growth
saturated at an eighth of the memory.

That is the arena. Lookup is the warm-up, and 19.4 should be read as news about the warm-up.

---

## 19. The task is exhausted - the graph must hold more than one address

### 19.1 — four interventions, one number that never moved

| run | what changed | held-out vs 1-NN | high-margin |
|---|---|---:|---|
| qrank | rank of closeness to the query row | 0.000 | 0 v 1 (underpowered) |
| qrank_big | corpus x6, resample overlap 0.82 -> 2.8e-05 | -0.343 | 0 v 13, z -3.61 |
| qmargin | confidence of that ordering, scale-free | -0.522 | 0 v 13, z -3.61 |
| maxpool | max beside the mean in all three aggregations | **+1.000** | 0 v 3, z -1.73 |
| rowdrop04 | training across densities | +0.728 | 0 v 3, z -1.73 |

Only the max moved anything, and it moved a lot - ten of thirteen high-margin losses gone, probe
loss 1.67 -> 1.02, and the first run where Phi is ahead of the rival on accuracy. Everything
else was ruled out **by a prediction stated before the run**: no information missing (qmargin),
not too little data (qrank_big), not a density the mind never saw (rowdrop04).

### 19.2 — the wall, measured

```
Phi 0.812   1-NN 0.801
agree on 249 of 266 questions - 93.6%
the entire disagreement is 17 questions
```

`lookup` is single-hop and closed-set: the answer is one of the values already lying on the
address's own rows. Its Bayes-optimal rule is almost exactly nearest-neighbour, so there is
about 17 questions' worth of room for a mind in it. Continuing to tune the reader here is
fitting inside noise, which is the situation the project's own rule says to answer with a
structural change.

### 19.3 — what is actually missing

**Reading never feeds reading.** The graph holds one address, Phi fires once, and the result
goes nowhere. No chain, no depth, and therefore none of the compressed-chain capability §15.2
asks for.

And this is why §16's traversal was premature, which is worth admitting plainly: **you cannot
walk when the graph cannot hold a destination.** The walk was proposed before there was anywhere
to go.

### 19.4 — the change: one graph, several addresses

Structural, parameter-free, no weighting - because a weighted blend of the three would be three
constants chosen by hand:

```
N(a) = (up to k addresses sharing the anchor)
     ∪ (up to k addresses sharing the relation)
     ∪ (up to k addresses by shared rare context words)
```

Which of the three matters is for the mind to decide, not for the neighbourhood rule.

```
rows  = mentions(a) ∪ union over b in N(a) of mentions(b)
edges += 1[anchor(i) = anchor(j)]
         1[rel(i)    = rel(j)]
nodes += 1[row belongs to the query address]
```

Indicators only. No identity, invariant untouched.

### 19.5 — the verb, on data currently thrown away

`lookup_question` requires `len(slots) >= 3` and mentions per address is 2.85, so addresses with
one or two mentions are discarded entirely - 804 train addresses yield 265 questions. That is
most of the distribution, and it is exactly the part where the current rival cannot operate.

```
lookup_sparse: address a with |mentions(a)| in {1,2}, hide one value
candidates:    the distinct values on N(a)'s rows
```

With one mention, hiding it leaves no other row of `a`, so **1-NN within the address is not
merely worse, it is undefined**. The answer has to come from the neighbourhood. Rivals stay
honest and parameter-free: majority over N(a), and nearest context over N(a) - same evidence,
different rule.

### 19.6 — why this satisfies "the construction must not depend on data size"

Not a hope, a property: **k is fixed, the rule is one, the weight count does not grow.** On a
small corpus N(a) is nearly empty and there is little to read; on a large one N(a) is rich and
*the same weights* find more structure. The mechanism is invariant to corpus size and only the
amount of structure it finds is not - which is "seeds on small data, a mind on large data"
stated as a fact about the construction.

It is also the first point at which **patterns** become expressible at all. Twenty addresses
carrying the relation `born in` inside one graph make "the values of this relation are of one
kind" something the mind can see. Reading one address at a time, it cannot see it in principle.

### 19.7 — growing the mind without losing it

New channels enter with **zero-initialised output weights**, so the function at the moment of
widening is bit-identical and training continues from where it stopped. The project already uses
this idiom on the readout. The mind then grows channel by channel, monotonically, with nothing
retrained from scratch - which is what "keep training it and let it grow" requires mechanically.

### 19.8 — the falsifier, and the cost

> If Phi cannot beat majority-over-N(a) on `lookup_sparse`, the neighbourhood carries nothing
> and the multi-address graph is dead.

One run. And the change is *smaller* than what it replaces: two edge channels, one node
indicator, a parameter-free neighbourhood rule and one verb - against §16's traversal plus a
case store, which is deferred and may turn out unnecessary once the graph can already reach.

---

## 20. Reconciliation: one mind reads several views at once, and their agreement is a signal

Proposed from outside — "training not only during a single pass; e.g., simultaneous traversal of
the tape in several places with reconciliation" — together with the worry that matters just as
much: "we keep moving pieces one at a time and I am afraid we will overcomplicate."

Both are answered by the same design, because reconciliation UNIFIES three existing patches
rather than adding a fourth.

### 20.1 — what the last run package settled first

The capacity confound is closed from both sides. At matched parameters (5659 without max against
5601 with), the mean-only arm reproduces the old failure shape (high-margin 0v13, z -3.61) and
the max arm keeps the repair (0v3). At 13k parameters without max, width buys low-margin gains
(z +2.67) but high-margin stays negative (0v6). So the roles are different and both are real:
**max is the right operation, width is room.** And the task itself is saturated: the entire
disagreement with 1-NN is 16-23 questions of 266, vs_counts is ~8 sigma in every arm.

### 20.2 — the mechanism

V views of the same question, one shared mind, one reconciliation:

```
G_1..G_V   independent draws of the same world: row subsets (drop_rows IS the view
           generator) and, once §19 lands, independent neighbourhood subsamples
phi_v(c)   = Phi(G_v completed with candidate c)          same weights for every view
s(c)       = (1/V) * sum_v phi_v(c)                       pooled logit
p          = softmax(s)                                    product of the per-view experts
L          = -log p(c*)                                    ONE term, no weight between views
```

`softmax(sum_v phi_v)` is the normalised product of the per-view distributions — a geometric-
mean ensemble, which is what "reconciliation" means once written down. At V=1 the loss reduces
exactly to today's cross-entropy, so this is a strict generalisation, the same shape of claim as
Plackett-Luce was.

And the part that costs nothing but is worth the most:

```
D(q) = mean over view pairs of JS( softmax phi_v , softmax phi_u )
```

Disagreement between views of the SAME evidence is a label-free confidence signal. No trained
calibration head — the thing 289c tried to bolt on and the thing Phi has never had (stated 0.98,
realised 0.63).

### 20.3 — refusal falls out with no chosen threshold

280's reward structure is already fixed: +1 right, -1 wrong, +0.75 abstain. Answering beats
abstaining iff 2p-1 > 0.75, i.e. **p > 0.875** — a constant derived from rewards fixed long ago,
not a knob. So the rule is:

```
refuse when the TRAIN-measured accuracy conditional on D(q) falls below 0.875
```

The curve accuracy-vs-D is read off the train tape; nothing is tuned. That is 291's REFUSE verb
receiving its input for free, and it is 280's calibrated-silence discipline resurrected on the
relational mind.

### 20.4 — why this SIMPLIFIES

| existing piece | fate under reconciliation |
|---|---|
| row-dropout (patch for density) | becomes the view generator — a mechanism, not a patch |
| qmargin confidence feature | superseded by D(q), which measures confidence instead of proxying it |
| 291's calibration problem | solved by 20.3 with zero parameters |

Three separate things become one. The parameter count does not change at all: the views share
every weight.

### 20.5 — routes: reconciliation across ADDRESSES

The tape already measures `facts_written_twice` (~440-520 per tape): the same (anchor, value)
written at two addresses. Those are two independent routes to one fact. Hide the value at both;
the answers should agree.

First as an EXAM, not a loss:

```
route_consistency = P(argmax agrees across the two routes)
null: if errors were independent, agreement ~= acc^2 + (1-acc)^2 / (k-1)-ish; measure against
      the permutation null as always
```

A mind that answers the same fact differently depending on the door it came through is not
reading knowledge, whatever its accuracy. If the exam shows real inconsistency, the training
form is the same pooled-logit loss over the routes' shared candidates — and THAT is a training
signal that needs no label at all, which is §15.4's sibling made concrete: **the number of
agreement pairs grows with the corpus while the weights do not**, which is the scale property
stated as a property of the signal.

### 20.6 — cost and order

V=3 multiplies graph work by ~3: recent runs are ~1700s, so ~5000s — inside the stated budget.

1. **Views on the CURRENT graph, today.** drop_rows exists, maxpool exists; ~100 lines. One run,
   V=3 against V=1 (paired, same questions): does the ensemble beat the single pass, and does
   D(q) predict error (AUC against the permutation null)?
2. **§19's substrate** (multi-address graph + lookup_sparse) — gives the views something richer
   to disagree about, and the sparse verb is where within-address 1-NN is undefined.
3. **Route consistency exam** on facts_written_twice, then as a label-free signal if the exam
   shows inconsistency.

### 20.7 — falsifiers, one per step

- V=3 does not beat V=1 paired, and AUC(D -> error) is at chance → views add nothing on this
  substrate; step 2 may still rescue it, but the principle takes a hit.
- Phi cannot beat majority-over-N(a) on lookup_sparse → the neighbourhood carries nothing (§19).
- route_consistency at its independence null → the mind has no cross-route identity of facts,
  and 20.5's training form is pointless until addressing improves.

### 20.8 — rejected on purpose

Self-distillation with an EMA teacher (the standard "not only during the pass" trick) is
rejected: it doubles the weight count with a second, slowly-moving mind, and the capacity
argument — the one guarantee that survives every regime change — stops being clean. Views share
weights; a teacher does not.
