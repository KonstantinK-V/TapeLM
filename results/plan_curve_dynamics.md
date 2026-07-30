# Contract — Curve dynamics LM substrate

**Status:** LOCKED 2026-07-28  
**Plan file:** this document  
**Smoke:** Stage170 `CURVE_DYN_SMOKE_YES` (permission to proceed, not understanding)  
**Prior:** Stage169 word-CE — FROZEN

Machine copy: `results/stage170_contract.json`

---

## One-line

**Text draws a continuous curve; we train only on how that curve moves; text decode is later and not the teacher.**

---

## Ontology (what exists)

| Name | What it is |
|------|------------|
| **Stream** | Raw character sequence from corpus (WikiText / later OWT). Not words, not BPE. |
| **Pen** | Map `char_t →` contribution to the curve. May be learnable or frozen. |
| **Curve** | Sequence of latent points `z_0, z_1, …, z_T` in `R^d`. **This is the object of learning.** |
| **Arc / chunk** | Contiguous segment `z_{t:t+L}` (a piece of the curve). |
| **Δ (delta)** | Change along the curve: `Δ_t = z_{t+1} − z_t` (or multi-step `z_{t+k} − z_t`). |
| **Dynamics** | Model that predicts future `z` / `Δ` from past arc. |
| **Decoder** | Optional read-out `z → text`. **Not** in the train loss until explicitly unlocked. |

---

## What the curve is made of (LOCKED for next stage)

1. **Time index** = one step per **character** in the stream (not per word).  
2. **Point `z_t`** = latent state after consuming chars `…, c_t` through the pen (causal).  
3. **Space** = continuous `R^d` (`d` in 64–256 for 4GB; smoke used **d=96**).  
4. **Not** discrete token ids as the learning target.  
5. **Not** word codebook atoms (that was 169).  
6. Letters/words appear only as **ink** that draws `z`; they are not what loss names.

**Smoke pen (170, reference):** `Embedding(char) → GRU → LayerNorm → z_t`.  
**Next hardening (171):** same curve definition, but **freeze pen** after a short fit (or freeze SOTE letter stack as pen) and train **dynamics only** — to kill “jointly make an easy curve.”

---

## Train contract (LOCKED)

| Allowed | Forbidden as main loss / gate |
|---------|--------------------------------|
| Predict `z_{t+1}` from past arc | Char / word / BPE cross-entropy |
| Predict `Δ_t` (and later `Δ` over k steps) | Battery / majority-last as primary gate |
| Contrastive: true future arc vs wrong arc | Fluency / ALL% / STORY as success |
| Geometry regularizers (energy, non-collapse) | Declaring “understands language” from cosΔ alone |

**Loss language:** cosine / MSE / contrastive **in `z`-space only**.

---

## Eval contract (LOCKED)

Primary gate = **dynamics quality vs trivial baselines in `z`-space**:

| Baseline | Meaning |
|----------|---------|
| zero-Δ | always predict no change |
| mean-Δ | always predict corpus-mean change |
| copy-Δ | repeat last observed Δ |

**Pass (smoke):** beat mean-Δ and copy-Δ by margin (170 used >0.02 lift) → `CURVE_DYN_SMOKE_YES`.  
**Pass (hardened):** same with **frozen pen** + multi-step and/or contrastive arc.  
**Fail:** ≤ baselines → redesign dynamics/pen geometry — **do not** add text CE to “fix” it.

Secondary probes (optional, never promote to teacher): weak decoder accuracy, retrieval of arcs, domain transfer.

---

## Role of text

- **Bootstrap / ink only:** provides the path through `z`.  
- **Not** the definition of intelligence for this line.  
- Corpus choice matters only as “which curves we draw,” not as LM pretraining objective.

## Role of SOTE letter-assembly

- Optional **pen** (orthographic drawing tool).  
- **Not** the learner’s objective.  
- Must not reintroduce word-id CE through the back door.

## Role of decoder

- **After** dynamics works under hardened gates.  
- Read-out / interface.  
- If decoder loss dominates training, contract is broken — stop and strip it.

---

## Explicit non-goals

- Revive 169 battery-vs-majority as the north star  
- “SOTE understands because letters compose words”  
- Soft morph / dual-channel as substitute for this contract  
- Scaling word-CE and calling it curve learning  

---

## Stage174 early context falsify (DONE)

| Probe | Result |
|-------|--------|
| A same-suffix | **FAIL** — endpoint `z` cos=1.0 for same last 24 chars (prefix wiped) |
| B paraphrase | **FAIL** — form≫meaning (hard spelling cousins closer than paraphrases) |
| C sentence shuffle | **PASS-ish** — order moves whole-window `z`, not proof of meaning |

**Practical call:** `CONTEXT_WALL_FOR_UNDERSTANDING_GOALS` — do not scale dyn hoping for understanding on this pen.

---

## Known limits (after 170–172) — honest

What we **have**: frozen-pen dynamics beat mean/copy Δ; contrastive arcs work; scale early-stop YES @40k.

What we **do not** have yet (main bottlenecks, rough priority):

1. **Gate ≠ meaning** — success is geometric predictability of `z`/Δ, not semantics/context. Easy to overread YES as “language.”  
2. **Pen is orthographic ink** — char/GRU curve encodes form (spaces = sharp turns). Dynamics may mostly learn **local script physics**, not discourse.  
3. **Same-domain hold** — train/hold from same Wiki stream → weak test of transfer.  
4. **Horizon soft** — k16 still good but weaker than k1; long-arc competence unproven (k32+).  
5. **Probe ceiling** — ~54% next-char readout with stop-grad; shows leftover letter signal in `z`, not that decoder should become teacher.  
6. **Identity of curve underdetermined** — many pens could yield equally predictable Δ; we haven’t stress-tested alternate frozen pens / scrambled ink.

**Next should falsify (1–4), not inflate capacity.**

---

## Stage map

| Stage | Intent |
|-------|--------|
| **170** | Smoke: joint pen+dyn, char-step curve — **DONE YES** |
| **171** | Hardening: frozen pen; multi-step Δ; contrastive arcs — **DONE YES** |
| **172** | Scale: more data, stronger dyn (attn), k≤16, weak readout — **DONE YES (early stop @~40k plateau)** |
| **173** | Ortho vs language falsify — **LETTER_SEQUENCE_SENSITIVE** |
| **174** | Context A/B/C — **CONTEXT_WALL** (A wipe, B form≫meaning) |
| **175** | Causal Transformer pen → **NULL** (A_same≈1.0; Δ-loss still suffix-wipes) |
| **176** | whitespace “word” arcs + next-arc → **WEAK/practical FAIL** (not real BPE) |
| **177** | **Curve-BPE:** ByteLevel merges + space-in-token → **WEAK / STILL_LAST_UNIT_WIPE** (A_same≈0.96; same as 176) |
| **178** | **Objective flip:** retention + past-bag + far + prefix-instance → **YES** early @1.5k (A_same 0.82→0.63) |
| **179** | Harden + gate B → **A_YES / B_FAIL_FORM** (stopped early for 180; hard≫para) |
| **180** | Dual-channel → **A_YES**; B micro peak@4.5k then fade — **stopped @7.5k** |
| **181** | Matched CE GPT-2 → **`CE_CONTROL_CONTEXT_SIGNAL_YES`** (ablation Δ≈+3.8; B-micro early gap min@1.5k) |
| **182** | Hybrid slow-CE → **stopped@3k** (behind GPT abl/CE; A ok) |
| **183** | Dataset-answer exam (fast) → **`EXAM_NO_SIGNAL_YET`** (all ~chance; even GPT) |

### North star (clarified)

Train may change *thinking principle* (trajectory/memory). Text is the **final exam**, not the textbook.  
**Win** = answer what was in the dataset (183). **LM ceiling** (181) = reference only.  
Context A = hygiene, not the goal.

| Stage | Intent |
|-------|--------|
| **196** | Assemble variant A into one `TapeLM` (frozen P1) + anti-clone gate vs GPT & GPT+RAG → **TAPELM_PARTIAL** |

**196 read — the anti-clone gate did its job (honest, uncomfortable).** All four axes run on ONE
frozen P1 encoder from one shared fp-space, and generation parity holds (entry ticket OK). But the
"win only where GPT is structurally weak, and BEAT both GPT and GPT+RAG" rule exposed reality:
- **parity (entry ticket): HOLD** — curve next_tok 0.867 vs GPT 0.843 (Δ+0.023). Ties by design, proves nothing new.
- **calibration: CLEAN WIN** — fp-lexicon AUC 0.982 vs GPT BPE-surprisal 0.380. GPT's per-piece surprisal
  is actually *below chance* here: it flags rare real entities as weirder than pronounceable fakes
  (conflates rarity with fakeness). fp-lexicon ("is it in MY lexicon of seen words") gets it right. Genuine structural distinguisher.
- **recall: RAG-EQUIVALENT (NOT distinct)** — curve_fp 0.947 ≫ GPT-parametric 0.300, BUT **GPT+RAG 0.980 ≥ curve**.
  On pure retrieval, GPT's own embeddings retrieve as well or better. Recall alone does NOT separate us from GPT+RAG.
  This is exactly the user's fear, measured: the memory win over *vanilla* GPT is real, over GPT+RAG it is not.
- **one-shot edit: WEAK** — curve 0.417 (disjoint write/query, 4-way, chance 0.25) vs GPT 0.283 (chance, as expected).
  Curve acquires a brand-new fact at read time above chance and GPT structurally cannot — but 0.417 is not a clean win.

**Net:** TapeLM *composes* (single frozen encoder, no retraining, parity intact) and has ONE clean
distinguisher (calibration). The retrieval axis is honestly RAG-equivalent — so "different from BPE-GPT"
cannot rest on recall. The defensible distinctness = (a) calibration in-space, (b) generation+memory+calib
from ONE fp-space with zero extra params/training (GPT+RAG needs GPT *plus* a retrieval encoder + index).
Caveat logged: GPT calibration control (single-word surprisal) is a soft strawman vs in-context entropy;
but 191-P3 already showed curve entropy-calibration only *barely* passed while fp-lexicon hit 0.99 — the
fp mechanism is the real edge, not the control's weakness.
Next to earn "AND_DISTINCT": make edit a clean win (selective surprise-gated write, 187) and find a
retrieval regime where one shared fp-space beats bolted GPT+RAG (cross-doc / beyond-window / update-heavy).

| Stage | Intent |
|-------|--------|
| **197** | Knowledge-edit mechanisms + surprise-gated write policy → **EDIT_CLEAN_WIN** |

**197 read — 196's weak edit (0.417) was a bad KEY, not a substrate limit.** Anchoring the write on
the existing subject entity instead of a bag of context words fixes it completely:
- acquisition (4-way, chance 0.25): M1 mean-ctx (=196) 0.460 · **M2 subject 0.970 · M3 blend 1.000 · M4 multikey 0.960**.
- vanilla GPT 0.280 (chance — cannot ingest a write without gradient) · GPT+RAG(mean-ctx) 0.440.
- **selective write under tight budget: fp-lexicon surprise keeps 0.98 of novel facts vs 0.20 for ingestion-order.**
Net: one-shot fact acquisition is now a clean win over vanilla GPT, and the **surprise-gated write policy is a
genuine in-space advantage over generic RAG** — the tape decides *what to store* for free from the same
fp-lexicon that does calibration; a bolted RAG has no such novelty prior.
Honest caveat: curve's 1.00-vs-0.44 edge over GPT+RAG partly comes from giving the curve a subject-anchor
while RAG used mean-ctx; a subject-anchored RAG would also improve. The *fair, robust* in-space win is the
selective-write policy (0.98 vs 0.20), not the anchoring trick.

**196→197 update:** TapeLM now has TWO clean distinguishers over GPT (calibration 0.982 vs 0.380; one-shot
edit clean win) plus a fair in-space edge over GPT+RAG (surprise-gated write). Recall on a static corpus
stays RAG-equivalent (196). Path to AND_DISTINCT: combine into one stack + a beyond-window / update-heavy
retrieval regime where online in-space write beats a rebuilt RAG index.

| Stage | Intent |
|-------|--------|
| **198** | Streaming / beyond-window / update-heavy under budget → **STREAM_ARCHITECTURAL_DISTINCT** |

**198 read — the honest ceiling of "distinct from GPT" (anti-clone gate did its job again).**
Long stream (675 events ≫ 64-arc window), fake facts with later overwrites, memory budget = 135 (20% of stream):
- **beyond-window:** tape 0.738 & fair-RAG 0.726 both crush vanilla in-context GPT 0.226 (chance). External
  fp/RAG memory is *required*; in-context alone fails → beats vanilla GPT decisively.
- **budget policy:** tape (in-space fp-surprise admission) 0.738 vs **rag_uniform 0.250** (ingestion order, chance).
  The write policy is worth +0.49 under budget — and the tape gets it FREE from the same fp-lexicon.
- **honesty control (the flip):** first pass gave a rigged RAG a degenerate "S S S" query → 0.295 (fake
  CAPABILITY win). Mirroring RAG EXACTLY (subject-anchor + ctx blend, GPT encoder, + bolted fp-surprise
  admission) → **rag_novelty 0.726 ≈ tape 0.738**. So the capability is NOT unique to the tape.
- update handling: tape returns latest value on overwritten entities 0.778 (recency tiebreak works).

**Verdict = ARCHITECTURAL, not capability.** A fully-provisioned GPT+RAG *ties* the tape once you hand it
(a) a matching anchor and (b) the tape's own fp-surprise write policy. The tape's edge is that one frozen
fp-space supplies generation + retrieval anchor + novelty/admission signal + calibration with zero extra
components and zero training; RAG needs a separate encoder + a bolted novelty scorer to match. rag_uniform
0.250 proves the write policy matters; rag_novelty 0.726 proves it's transferable. Do not claim capability
distinctness — claim unification/simplicity + the free in-space signal.

| Stage | Intent |
|-------|--------|
| **199** | Semantic invariance (B) via consequence-prediction (CPC) on a FROZEN encoder + scale trend → **SEM_INV_PARTIAL** |

**199 read — B is NOT cracked by a non-destructive head, and the finding says why.** Route 1
(meaning = what comes next) as a CPC head on frozen fast features:
- raw frozen baseline (191b regime): para 0.705 / hard 0.889, gap +0.185.
- CPC head at 5% / 25% / 100% token budgets: gap +0.172 / +0.173 / **+0.163**. para nudged 0.705→0.714,
  hard essentially unchanged 0.889→0.877. **No inversion (hard still ≫ para) at any budget; no clean
  monotone scaling.** Head helps by a hair (0.185→0.163), not a semantic win.
- non-destruction CONFIRMED: next_tok on frozen encoder 0.825 (subset of 120; ~0.867 on full — the head
  is a separate branch, generation/memory/calibration untouched). Nothing broke, as designed.

**Why route 1 stalls (honest mechanism):** minimal pairs share continuations. "The car sat on the mat" and
"The cat sat on the mat" are followed by nearly identical words in a generic corpus, so consequence-
prediction has almost no signal to separate car from cat — exactly what keeps hard pairs glued. Two
compounding limits: (a) the FROZEN substrate already encodes car≈cat (surface), and a shallow head cannot
manufacture a distinction absent from its inputs; (b) the continuation signal for minimal pairs is too weak
at 150M chars. This replicates 190's "topical not semantic" at the representation level.

**The tension the user must weigh (explicit):** the only lever that demonstrably moves B is the ENCODER
itself (191b showed meaning grows with encoder pretraining scale/data). But moving the encoder = retraining
the substrate = breaking the frozen stack that gives parity+memory+calibration+edit. So "go to B" and "don't
break what we have" point in opposite directions. Non-destructive adapters have a demonstrated low ceiling
(gap 0.163, no inversion). B therefore belongs on a SEPARATE track: a new/larger encoder pretrained with a
meaning-bearing objective (minimal-pair hard negatives, paraphrase/cross-lingual data, or true scale),
run when hardware allows — not forced onto the current frozen artifact.

| Stage | Intent |
|-------|--------|
| **200** | Fact composition: operate-on-vectors vs read-by-index → **COMPOSE_CHAINS_BUT_RAG_PARITY** |

**200 read — "operable vectors" is REAL but not a capability win here.** Planted k-hop chains of novel
entities among 6k distractor slots, beyond-window:
- curve_string / curve_vector(no-decode) / rag_index ALL = 1.00 at k=1,2,3; binding one-shot 2-hop = 1.00;
  vanilla GPT in-context 0.20 (beyond-window).
- **Qualitative win (demonstrated):** curve chains PURELY in fp-space — `curve_vector` never decodes a
  fingerprint back to text between hops (1.00), and binding answers a 2-hop in one composed op. RAG
  structurally lacks both (its values are text that must be re-read each hop). Facts here ARE operable vectors.
- **But NOT a capability win:** a fair GPT-embedding index-RAG also chains to 1.00 — the task (unique novel
  anchors) has no compounding error to separate them. Consistent with 198: retrieval is retrieval.
Honest bottom line for "do we understand facts more than RAG": YES in the operable-representation sense
(vector-native chaining + binding composition, modes RAG doesn't have), NO in the "solve what RAG can't"
sense on tasks RAG already solves. A harder differentiator would need ambiguous anchors — which collapses
into the form/meaning problem (B). So the operable-vector distinctness is architectural, not a score win.

| Stage | Intent |
|-------|--------|
| **201** | B-track on 3050: minimal-pair hard negatives on an encoder COPY → **SEM_HARDNEG_NO** |

**201 read — the cheap substrate route also fails B, and cleanly tells us why.** Fine-tuned arc_enc (copy;
product P1 frozen) with edit-distance-1 hard negatives + anchor-to-P1:
- word-level neg cos did drop (0.54→0.39, car/cat-type words moved apart) BUT sentence-level B did not:
  para 0.705→0.732, **hard 0.889→0.902 (no drop, no inversion)**. next_tok on the copy fell 0.825→0.717.
- Two honest reasons: (1) B's hard pairs are mostly SEMANTIC swaps (cold↔warm, train↔plane, mathematics↔
  history) which are NOT edit-distance-1 — the objective never touched them; a one-word fp change is also
  diluted by the frozen sentence-level fast transformer. (2) touching the substrate costs generation.

**B verdict (after 199+201): a genuine wall on this hardware.** Neither a frozen-feature CPC head (199) nor
a substrate minimal-pair objective (201) moves sentence-level semantic invariance; the first can't (features
fixed), the second attacks the wrong level (spelling, not meaning) and hurts generation. B needs a meaning-
bearing objective at phrase/sentence level WITH scale/data — not reachable cheaply on a 3050. Documented as
the single open frontier; the current frozen stack (A + memory + calibration + edit) remains the finished
artifact and is untouched by these B experiments.

| Stage | Intent |
|-------|--------|
| **202** | B capability via PAWS, frozen encoder + attention head (4GB) → **SEM_B_NO but PARITY-with-GPT** |

**202 read — head-only on frozen small encoders can't crack B, and neither can GPT.** Trained an
attention-pool semantic head on PAWS (adversarial paraphrase) over the FROZEN P1 encoder:
- curve PAWS test 0.644, 179 para 0.359 / hard 0.758, no inversion.
- **GPT baseline (same head, frozen GPT-XL) 0.655, para 0.401 / hard 0.728, also no inversion.**
- lexical-overlap baseline 0.558 (both beat surface, neither solves it).
The decisive point: curve ≈ GPT (0.644 vs 0.655) — B is NOT a curve-specific defect; both small frozen
encoders plateau because PAWS needs the ENCODER to represent word order/structure (Scotland↔England swap),
which pooling a frozen d256/6L rep cannot recover. Confirms B is an encoder-capacity/scale problem shared by
both substrates, not a spelling-substrate limit of the curve. Next: fine-tune the encoder (copy) end-to-end.

| Stage | Intent |
|-------|--------|
| **202b** | Decisive B: fine-tune encoder end-to-end on PAWS (copy) → **SEM_B_CAP_NO but PARITY** |

**202b read — even a trainable encoder + direct meaning signal can't confirm B at d256/6L, for EITHER
substrate.** End-to-end fine-tune on PAWS (product P1 frozen, trained on a copy):
- curve PAWS 0.705 (↑ from 0.644 head-only), 179 para 0.806 / hard 0.941, no inversion; next_tok on the
  copy collapsed 0.83→0.55 (fine-tuning the fast channel on PAWS wrecks the LM — why product stays frozen).
- GPT fine-tune 0.701, para 0.547 / hard 0.746, no inversion.
- **curve ≈ GPT again (0.705 vs 0.701).** Both top out ~0.70 on adversarial PAWS; SOTA needs BERT-scale
  (100M+ params, billions of tokens). d256/6L (~7M params, 150M chars) lacks the capacity — identically
  for curve and GPT.

**B FINAL VERDICT (199+201+202+202b): scale-bound, not curve-bound, not reachable on 4GB.** Across a frozen
CPC head, a substrate hard-neg objective, a PAWS head, and a full PAWS fine-tune, semantic inversion never
appears — and at every step the curve matches GPT. B (adversarial semantic invariance) requires a much larger
encoder pretrained on far more data; this is a scale frontier the curve SHARES with GPT, not a defect of the
curve substrate. On the user's 3050/4GB it cannot be confirmed. Positive framing: the curve is never worse
than the standard architecture on semantics at any tested scale.

| Stage | Intent |
|-------|--------|
| **203** | Internalize hops: trainable differentiable k-hop reader over FROZEN fp-space + NON-GRADIENT tape → **INTERNAL_HOPS_YES_IF_STRUCTURED** |

**203 read — hops CAN move inside a forward pass, but only with the right inductive bias.** Two internal
readers over the frozen fp tape (encoder + slots non-gradient; anti-CF preserved by construction):
- **free-form reader** (proj + k-embed + per-step MLP update + output head): train k2/k3 = 1.00 but
  **test 0.15–0.20 ≈ chance** — it *memorizes* the training chains' start→target map and does not generalize.
- **soft-follow reader** (parameter-free soft value-follow `state←softmax(state·Kᵀ/τ)·V`, step-select by k,
  only τ learned): **train 1.00 AND test 1.00** — matches the external hand-loop exactly and generalizes.
- external hand-loop (zero-param argmax follow) = 1.00 for reference; encoder untouched (anti-CF True).

**Lesson (the interesting part):** internalizing hops is *not free*. A free-form learnable module collapses to
rote memorization of seen chains; the operation only generalizes when the module keeps the minimal structure of
the parameter-free op (sharp retrieve → follow value → select depth). So "hops inside the model" is feasible
(soft-follow = a differentiable, trainable, in-graph version of 195/200's external loop with anti-CF intact),
but the win is architectural/engineering (one differentiable stack), **not** a new capability — the zero-param
external loop already generalizes perfectly. Practical implication for future TapeLM: bake hops in as a
**structured** soft-follow/bind block (HRR-style), not a free MLP reader, or it will overfit.

| Stage | Intent |
|-------|--------|
| **204** | W1: substrate robustness to char noise / OOV, curve fp vs FAIR GPT+RAG mirror → **NOISE_ROBUST_WIN** (first capability-level win) |

**204 read — the first axis where a fully-provisioned GPT+RAG does NOT tie.** Rank-based metrics only (no
cross-space cosine); RAG uses the identical subject-anchored key/query recipe, only the encoder differs:
- **A. identity retrieval under noise** (typo(w) → find clean w in a 1000-word pool, acc@1):
  | p | curve seen | gpt seen | curve OOV | gpt OOV |
  |---|-----------|----------|-----------|---------|
  | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 |
  | 0.1 | 0.883 | 0.549 | 0.930 | 0.747 |
  | 0.2 | 0.841 | 0.507 | 0.890 | 0.707 |
  | 0.3 | **0.801** | 0.403 | **0.860** | 0.567 |
  Curve keeps ~2× GPT's accuracy at every noise level, on both seen and novel (OOV) words.
- **B. fact recall with noisy queries** (4-way, chance 0.25): curve 1.000/1.000/0.993/1.000 vs rag
  1.000/0.933/0.887/0.880 across p=0→0.3; relative drop curve 0.000 vs rag 0.120. (Near ceiling → weak evidence alone.)
- **B2 HARDENED** (noise in the STORED corpus *and* the query, independent noise realizations, 8-way, chance 0.125):
  p=0.0 curve 1.000 / rag 0.993 · p=0.2 **0.947 / 0.733** · p=0.3 **0.913 / 0.627**. Both start at ceiling, so the
  gap is caused by noise, not by task difficulty — a genuine **capability** separation from fair GPT+RAG.
- **C. mechanism:** BPE pieces/word grows 3.93 → 4.76 as noise rises; each typo re-fragments the token, so the
  RAG key/query embeddings drift apart. The char-curve fp changes smoothly instead.

**Status upgrade:** 198/200 established that fair GPT+RAG *ties* the tape on retrieval capability. 204 breaks that
tie under noise/OOV — the first non-architectural, capability-level win over the strongest rival.
**Honest bounds:** (1) the GPT control is our matched d256 BPE model; a large pretrained BPE LM would be somewhat
more noise-tolerant, though the re-fragmentation mechanism is unchanged. (2) The rival that *would* close this gap
is a **byte/char-level LM (ByT5-style), not BPE-GPT** — so the correct claim is "BPE is the structural weak link,
and the curve gets robustness *plus* memory/calibration in one space", not "no architecture can match this".
(3) Noise is synthetic uniform char noise; real typo/OCR distributions differ.

| Stage | Intent |
|-------|--------|
| **205** | W3: targeted unlearning + provenance + contradiction audit, vs *parametric* GPT (memorize→gradient-unlearn) → **UNLEARN_PROVENANCE_WIN** |

**205 read — deleting one fact is O(1) and provably free of collateral; for parametric GPT it is destructive.**
60 planted facts (20 target for deletion / 40 retained), 400 curve slots, GPT first fine-tuned until it *really*
knows the same facts (otherwise "unlearning" would be vacuous), then unlearned by gradient ascent with early stop
at chance (minimal damage version):

| metric | curve before | curve after delete | GPT after memorize | GPT after unlearn |
|--------|--------------|--------------------|--------------------|-------------------|
| target fact recall | 1.000 | **0.000** | 1.000 | 0.000 |
| retained fact recall | 1.000 | **1.000** | 1.000 | **0.200** |
| next_tok (collateral) | 0.825 | **0.825** | 0.808 | **0.733** |

- curve deletion: 20 slots removed in ~0 ms, **no gradient**; GPT unlearning: 30 gradient steps / 2.1 s.
- **provenance attribution 1.000** — the top-scoring slot is always the one written from the gold fact's own text.
- **contradiction audit: detection 1.000, false-positive 0.000** — with two conflicting slots for one subject, the
  top-2 scores are within 0.02 and the conflict is flagged instead of silently answering.
- target recall goes to 0.000 (below chance) because the deleted value is *absent from memory entirely*, so it can
  never be selected — complete, detectable erasure rather than random guessing.
- GPT's 20-fact unlearning wiped **80% of the retained facts** (1.00→0.20, below chance) and cost 0.075 next_tok.

**Honest bounds:** (1) vs **GPT+RAG this is architectural** — a RAG index can also delete entries and expose
provenance; the capability contrast is against *parametric* knowledge. (2) Gradient ascent is a crude unlearning
baseline; localized editors (ROME/MEMIT) or retain-set-regularized objectives would reduce collateral, so the
result reads "naive parametric unlearning is destructive", not "no method works". (3) Provenance/conflict = 1.000
on planted unique novel subjects (easy regime); overlapping real-world subjects would be harder.

| Stage | Intent |
|-------|--------|
| **206** | W5: latent vector hops vs decode/token routes under a real compute budget (uncached encoders) → **LATENT_HOPS_CHEAPER_BUT_RAG_VECTOR_TIES** |

**206 read — W5 is a COMPUTE win, not a capability win; and re-anchoring only works on a substrate that can
re-identify a corrupted word.** 120 planted chains, depth 6, beyond window, 3.7k slots. First pass was invalid
(the GPT word cache pre-warmed every query, so `enc_calls`=0 for all routes); fixed with uncached per-query
encoders that are actually counted.
- **clean: all routes 1.000 at every k** — accuracy cannot separate them (as in 200).
- **compute at k=6:** curve_latent **2.35 ms / 1 encoder call** · rag_vector 6.24 ms / 1 call ·
  **rag_text 37.57 ms / 7 calls** (decode→re-encode per hop) · vanilla GPT in-context 0.250 = chance.
  So latent hops are **16× cheaper than a text-valued index** and 2.7× cheaper than a vector-valued one.
- **noise (p=0.15) compounds and eventually beats everyone:** curve 0.650/0.517/0.333/0.300 at k=1/2/4/6 vs
  rag_vector 0.450/0.358/0.283/0.292. Curve leads clearly at shallow depth (+0.20, +0.16) but by k≥4 both are
  near chance — one bad hop destroys the whole chain (multiplicative error).
- **lexicon re-anchoring (snap to nearest entity fp after each hop):** helps the curve
  (0.700/0.617/0.392/0.350) but **does nothing for RAG** (0.450/0.350/0.242/0.300). Mechanism: snapping is only
  error-correcting if the substrate can re-identify the corrupted form — curve identity under noise ≈0.85 (204),
  GPT ≈0.55, so RAG snaps to the *wrong* word half the time and propagates the error instead of fixing it.

**Honest bounds:** the 2.7× over rag_vector comes largely from the curve's encoder being a tiny char arc-encoder
vs a 6-layer transformer — a size artifact, not a structural claim. The **structural** part is O(1) vs O(k)
encoder calls against text-valued indices and CoT routes (16× at depth 6). W5 verdict: real and useful for
latency, but it does **not** add a capability distinction. The genuinely new finding is the snap asymmetry —
evidence for quantize-to-lexicon as an error-correction mechanism *available only to the char-fp substrate*.

| Stage | Intent |
|-------|--------|
| **207** | Variant B "curve as thinking": generate next fingerprint (InfoNCE + snap) vs closed-vocab token CE, shared trunk → **CURVE_THINKING_NO** (1/4 gates — kill-gate fired) |

**207 read — the falsification we set up actually fired, and it names the mechanism.** Same frozen arc-encoder
feeds both models; identical 3.3M-param causal trunk; 3500 steps. CURVE predicts the next word's *fingerprint*
(contrastive next-arc, in-batch InfoNCE, decoded by snap-to-lexicon); CE predicts the next word ID via a closed
top-8k softmax.
- **G1 quality FAIL:** curve 4-way next-word 0.536 vs CE **0.861** (chance 0.25). The InfoNCE loss barely moved
  (7.53→7.25, uniform floor ≈7.74) while CE dropped 6.04→5.35 — the fp-generative head learned almost nothing.
- **G2 drift FAIL:** free-run predicted fp sits **0.89 away** from the nearest lexicon entry from step 1 (cos≈0.11,
  already off-manifold, not a gradual walk), and snap decoding collapses to **0.857 repetition**. Continuous
  free-generation is garbage here — the naive-continuous-LM failure, now measured at word-fp level.
- **G3 open-vocab (the ONE salvageable point):** on OOV-for-CE targets, **CE = 0.245 = chance by construction**
  (every out-of-table word shares the UNK logit — the closed-vocab structural weakness is real and confirmed),
  while curve = 0.315 (above chance). So the open-metric-vocabulary property holds, but the curve LM is far too
  weak to exploit it.
- **G4 unification FAIL:** mean-pooled trunk hidden as a memory key = 0.175 (below chance) — the weakly-trained
  generative trunk is not a usable memory encoder.

**Mechanism (why B fails, precisely):** the output space is a **char-derived fingerprint = a SPELLING code**, so
"emit the next point on the curve" forces the model to regress *how the next word is spelled*. Spelling is not
predictable from context beyond word identity, and near-neighbour fps are look-alike spellings, not
context-substitutes — so the contrastive gradient carries almost no LM signal. CE sidesteps this with arbitrary
learnable output slots. **This explains 176/177 (next-arc suffix-wipe) and why 185 found only CE-like signal
usable.** Decoupling the output code from spelling (a learnable semantic codebook) would help — but that IS
drifting back into CE with extra steps, i.e. it abandons the "output space = the fp substrate" claim.

**Verdict for the north star:** Variant B ("fully generative curve, text only decoded") stays a research bet and
is now **empirically falsified at this scale in its literal form**. Variant A (encoder-LM + FP memory/calibration,
Stages 191–206) remains the finished, defensible artifact. Honest bound: 3.3M params / 3500 steps is small; but
the failure is structural (spelling-space output), not merely under-training — the loss curve shows the head is
near the uniform floor, not slowly improving.

**207-MAX (full wiki, 72.3M train tokens, 20k steps, V_LEX 80k): verdict UNCHANGED — scale does not rescue B.**
Two complete runs agree (`CURVE_THINKING_NO`, 1/4): run A batch 48 ~52 min, run B batch 32 + memmap/NCE-cap ~44 min.
Best run (batch 48, `stage207_max_decision.json`): G1 curve **0.581** vs CE **0.934** (smoke 0.536/0.861 — gap widened);
G3 **0.354** vs 0.246 (above chance, still not pass); G4 0.250; G2 drift pass. Curve NCE ~7.0→still near in-batch
floor with full batch InfoNCE; CE ~4.8. **Conclusion:** max data + long train confirms 207 — variant B falsified.

| Stage | Intent |
|-------|--------|
| **208** | Hybrid FOR variant A: discriminative word-level fp reranker for rare words, gated by fp-surprise → **HYBRID_NO_GAIN** |

**208 read — the hybrid does not improve A, so A stays exactly as it is.** Unlike 207 the fp side here is
*discriminative* (rank 4 frequency-matched candidates from the frozen context state, not regress a spelling
code), and it genuinely learned (loss 1.61→0.62). It still does not help:

| scorer | all | common band | rare band |
|--------|-----|-------------|-----------|
| A only (BPE CE head) | **0.551** | **0.669** | **0.431** |
| fp reranker only | 0.396 | 0.435 | 0.355 |
| combined (dev-tuned w=0.25) | 0.541 | 0.667 | 0.413 |
| gated by fp-surprise (w=0 common / 0.5 rare) | 0.544 | 0.669 | 0.416 |

4-way, frequency-matched candidates within band, chance 0.25, test n=700 (se≈0.019), fp-lexicon gate read-only,
encoder/head bit-identical (assert passed). The dev tuner *did* learn the intended policy ("trust fp only on rare
words", w=0→0.5) and still gained nothing; all deltas vs A are inside noise, so this is **no gain, not harm**.

**Why (and it closes a loop):** A is weakest exactly where predicted — rare band 0.431 vs common 0.669 — but the
fp reranker is weaker *there too* (0.355). The whole-word fingerprint carries redundant, lower-resolution
information compared to BPE piece composition on **clean** text. This is consistent with 204: BPE fragmentation
only becomes a real liability under **noise/corruption**, not in clean rare-word ranking. It also retro-corrects
207's G3 framing: that gate beat a *word-level* closed softmax, which is a weaker rival than BPE (BPE can spell
an unseen word piecewise), so the open-vocabulary advantage should not be claimed against BPE-GPT.

| Stage | Intent |
|-------|--------|
| **209** | On 3050: scaling grid (d128/192/256) + MiniLM teacher probe — refute "A structurally blind to meaning"? → **STRUCTURAL_BLOCK_NO** |

**209 read — variant A is NOT structurally blocked; B at PAWS level still needs scale.** Matched CE-pretrain
(d128/2L, d192/4L quick; d256/6L P1/P2) + frozen encoder + PAWS head (202 protocol):
| scale | curve PAWS | gpt PAWS | Δ |
|-------|------------|----------|---|
| d128/2L | 0.596 | 0.580 | +0.016 |
| d192/4L | 0.602 | 0.624 | −0.022 |
| d256/6L | 0.632 | 0.640 | −0.008 |

Curve PAWS **monotone** 0.596→0.602→0.632; **parity** (curve ≥ gpt−0.03) at all three scales. MiniLM teacher
Pearson r @ d256: curve **0.256** vs gpt 0.270 (both weak but curve within −0.05; above 0.25 floor). **No
inversion** (para < hard) at any scale; PAWS < 0.70 everywhere — same as 202, i.e. **full B not confirmed on
3050**. Honest read for the user: **3050 can refute "A can never be meaningful"** (same scaling trajectory as
GPT); **3050 cannot confirm strong semantic invariance** — that remains a larger-encoder frontier, not a
curve-specific wall.

| Stage | Intent |
|-------|--------|
| **210** | Pre-publish: SoftFollow inject into P1 forward, answer via CE tokens (not fp cosine) → **THESIS_NO** |

**210 read — structured hops do NOT survive the token head on 3050 (negative for thesis hook, not for 203).**
Reader phase matches 203 (external cosine loop test **1.0** on k1–k3). After inject training (800 steps, gate→~0.29),
**4-way token** accuracy stays at chance: soft-follow test k2=**0.25** k3=**0.30**; free-form test k2=**0.33** k3=**0.22**
(no train/test overfit gap — both fail). **G3/G4 pass:** next_tok delta **0.0**, bit-identity diff **0.0**, P1 frozen.
**G5 pass:** no_memory ≈ chance. Interpretation: fp-space composition works; **mapping retrieve→BPE span via one
memory-arc is not learnable** at this scale without touching P1 — external zero-train loop remains the honest hop API;
internal forward+tokens needs a different read head (211 slow tape / 212 instance), not inject-only.

| Stage | Intent |
|-------|--------|
| **211** | Pre-publish: cross-doc internal slow tape vs endpoint → **THESIS_NO** |

**211 read — addressable slow tape does NOT beat endpoint; external slots remain ceiling.** Cross-doc (fact
in doc A, query on doc B only; n=100, 4-way): **internal_tape 0.23** vs **endpoint_only 0.28** (both ≈ chance;
endpoint **better**, G1 fail). **external_slots 1.00** (explicit fp memory reference). **gpt_incontext 0.26**
(beyond-window ok, G3 pass). Noisy (30% char on A): internal **0.16**; doc-id oracle with wrong id **0.99**
(G4 fail — metadata key trivially separable; fp-geometry internal tape not the win). P1 frozen. Honest negative
for concept §183 «endpoint ≠ memory»: logged surprise slots do not yet form a usable addressable tape; **212**
(instance channel) is the remaining substrate bet.

| Stage | Intent |
|-------|--------|
| **212** | Pre-publish (final): read-only instance channel on frozen tape — collisions + para/hard → **THESIS_NO** |

**212 read — occurrence identity is NOT recoverable from the frozen tape state; the collision debt stays open.**
Contrastive head (InfoNCE, positives = disjoint other half of the same occurrence window, hard negatives =
other occurrences of the SAME surface) trains well (loss 0.02–0.21) but does not transfer to held-out surfaces.
4-way among the 4 siblings of one surface form (chance 0.25, store/query = **disjoint** halves so lexical
overlap cannot help), n=96 test surfaces:

| key | acc |
|-----|-----|
| instance channel (learned) | **0.378** |
| instance channel (untrained random head) | 0.336 |
| ctx_blend (197 subject+ctx) | 0.328 |
| soft_rerank (the old failed trick) | 0.320 |
| fp_only surface key | 0.242 (blind by construction) |

Learning buys only **+0.04** over an untrained projection of the same state and **+0.05** over the 197 baseline
— all inside noise at this n; G1 (≥0.70) far off. **T2 invariance also negative:** para **0.750** < hard **0.937**
(gap +0.19, no inversion) — the tape state at a crop endpoint encodes **form**, not content, consistent with
190/199–201. **G5 pass:** next_tok 0.820 → 0.820 (Δ 0), P1 bit-frozen (channel truly read-only).

**Frontier closed (210–212, all THESIS_NO).** Three independent attempts to make the tape do something a
BPE+index stack cannot — hops inside the forward pass answering in tokens (210), addressable slow tape beyond
the window (211), occurrence-identity channel (212) — each fails *for a different, informative reason*:
composition is only sound as an **external zero-train fp loop**; the slow endpoint is **not** an addressable
memory; and instance identity is **not present** in the frozen state. The honest publication hook therefore
stays what 191–209 measured: variant A as a unified fp-space artifact plus a dense **map of negatives**.
None of the three cost anything — P1 stayed frozen and generation unchanged in all three.

**Internalization program: closed (210–212).** Token-internal hops, addressable slow tape, and occurrence-identity channels are **THESIS_NO**. The shipping hop API remains the **external zero-train fp loop** (203).

**Core LM program (191–209): closed for v1 claim.** Variant A is the artifact; variant B is falsified (207); hybrid rerank on A is NO GAIN (208); essence wins W1–W5 as measured below. **Memory ops program (213–230): closed for v1 product trunk** — same frozen P1, canonical bank, family W, fp decode, resolution policy.

### Memory extension program (213–230, 2026-07-30)

**Motivation.** Production use breaks the lab assumption “encoder never moves”: domain finetune, forked heads, and cross-domain read all require **stable slots** without full reindex. This line is **not** a second product — it is **operational TapeLM** on the 192–205 fp stack.

**Three layers (contract).** L1 **Freeze** (`arc_enc` default frozen → zero-train fp). L2 **W-remap** (tiny linear restorer per **family** when geometry shifts). L3 **Stream decay** (219, long-run bank hygiene). Narrative lock: [`extension_memory_contract.md`](extension_memory_contract.md) · implementer: [`../docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md).

| Stage | Intent |
|-------|--------|
| **213** | Full `arc_enc` freeze + upper finetune → **ARC_ENC_FREEZE_PARTIAL** |
| **214** | Recency-weighted ctx entity fp → **RECENCY_CTX_NO** |
| **215** | Toy domain adapter without remap → **DOMAIN_ADAPTER_NO** (superseded by 221) |
| **216** | Partial FF freeze (emb frozen, train FF) → **SPLIT_FF_NO** |
| **217–218** | Slow-endpoint tape / snap hop → **NO** |
| **219** | Age decay on stale slots → **STREAM_DECAY_WIN** (stream, not one-shot exam) |
| **220** | PAWS sem sidecar → **SEM_SIDECAR_NO** |

**213 read — freeze is the default for memory API.** Finetuning only layers above frozen `arc_enc` leaves **fp drift ~10⁻⁷**; fp-stable. Wiki CE drops if you train upper on TinyStories-only — expected trade. **Do not** partial-FF inside encoder (216): min cos(fp_old, fp_new) **~0.18–0.67** — geometry walks; use **full freeze + W** instead.

**214/215/216/217/218/220 read — closed branches.** Recency ctx hurts (0.947 → worse); naive adapter without core remap fails; slow endpoint and snap hop do not beat external slots; sem sidecar adds little at 3050. Logged in [`extension_closed_branches.md`](extension_closed_branches.md).

| Stage | Intent |
|-------|--------|
| **221** | After **intentional** arc_enc shift: learn **W** on ~800 core words → **FP_REMAP_ADAPTER_YES** |
| **221-probe** | Characterise W (WᵀW, OOV, W_B vs W_C, vocab curve) → **W_REMAP_CHARACTERIZED** |
| **222** | Deploy modes (W on keys only vs query) → **FP_DEPLOY_MODES_MIXED** |
| **223** | Cross-family W switch on 4-way → **DOMAIN_W_SWITCH_PARTIAL** |
| **224** | Far shifts (stories / code / med): family drop → **W_DOMAIN_PARTIAL** |
| **225** | Domain bundle: reuse W_prose vs fork; multi-head, frozen arc_enc → **DOMAIN_BUNDLE_OK** |

**221 read — migration without full reindex.** Controlled TinyStories-style shift: mean cos(**W fp_old**, fp_new) **~0.997** (full 256×256 W); fact recall through **W @ legacy keys ~0.78** vs oracle reindex **~0.87** (≥80% of oracle gate). Without W after shift, legacy bank **collapses** on the 221 protocol (not the same as “rank preserved on old fp before shift”). Bottleneck W (256→32→256, ~16k params) align **~0.967**, recall **~0.70** — same story, smaller footprint.

**224 read — family registry is empirical.** Code shift harshest: cos(old,new) **~0.59**; matched **W_code ~0.88**; wrong-family prose W **~0.68** (drop **~0.12**). Prose-like pairs often ≤0.05 wrong-W drop; **one global W is insufficient** for code-class ink.

**225 read — shared map + lenses.** Legal reuse of **W_prose** when cross-drop **≤0.05** (**DOMAIN_BUNDLE_OK**); **head_prose / head_code** specialize generation with **fp drift 0** on frozen `arc_enc`. Product shape: **`slots_canonical` + `W_family@read` + `head_family`**, not N independent memory products.

| Stage | Intent |
|-------|--------|
| **227** | Write keys in **canonical** P1 fp; read via **qmap** (domain query → canonical key space) → **CANONICAL_STORAGE_OK** |
| **226** | Joint gen+mem (head inject) → **JOINT_GEN_MEM_NO** |
| **226b** | Diagnose recall protocol vs utilization → **RETRIEVAL_OK_UTIL_BOUNDARY** |
| **226c** | End-to-end cross-domain: 227 bank + qmap + **228c decode** → **JOINT_FP_DECODE_OK** |
| **228a** | Counterfactual head inject → **HEAD_INJECT_PARTIAL** |
| **228b** | Global argmax retrieve + fp score → **FP_GUIDED_DECODE_NO** |
| **228c** | **4-way slot retrieve** + **fp_decode_pick_retrieved_4way** → **FP_DECODE_FIX_YES** |
| **229** | Contradictory values both in top-2 → **CONTRADICTION_RAW_MEMORY_OK** |
| **230** | Resolution policy over multi-hit slots → **RESOLUTION_POLICY_OK** |

**227 read — one bank, disposable lenses.** Keys always stored in frozen canonical fp; at read, **qmap** applies **W_bwd** so domain queries match canonical keys. Cross-code recall **~0.95** vs **~0.70** without W; same-domain **1.0**; drop vs matched **~0.05**. Unifies 221–225 into **one slot store** + runtime family W.

**226 / 226b read — retrieval ≠ utilization (head path).** First 226 joint exam looked like recall failure; **226b** showed **qmap recall 1.0** under 227 protocol — utilization with **code_comment inject ~0.47** vs none **~0.43** (negligible). **Head-only** paths do not read retrieved gold reliably (**226 JOINT_GEN_MEM_NO**).

**228b vs 228c read — protocol is part of the API.** Global argmax over all slot keys + raw cos(fp(candidate), query) **~0.33–0.40** — fails even when 227 retrieval is strong. **228c:** per-slot **4-way retrieve** (max key score per candidate value) then score with **cos(fp(c), fp(retrieved))** → **1.0** on code-return exam vs CE head **~0.48**. Official API: `_tapelm_ext.slot_retrieve_4way`, `fp_decode_pick_retrieved_4way`. Persisted W: `artifact/scripts/export_w_registry.py` → HF `w_registry/`.

**229–230 read — contradictions are policy, not geometry.** **229:** both conflicting values in top-2 **~60%** on mixed writes; score gaps small. **230:** raw argmax macro **~0.47** (bias to first-written “official” slot); **query_cue** and **composite** macro **1.0** on neutral + cued + revision suites (`resolve_slot_contradiction`). Complements 205 contradiction **detection** with **selection** at read time.

**226c read — cross-domain product exam.** Canonical bank + code **qmap** + 228c at return token: **recall_4way ~0.88**, **fp_retrieved_4way ~0.88** vs **head_only ~0.45** (n=60, code-shift cos **~0.45**). This is the utilization number to cite alongside 227 recall — not the old 226 global-retrieve **~0.60**.

**Shipping trunk (order):** **221 → 227 → 228c → 230 → 226c** (229 multi-hit → 230). Demo: `artifact/scripts/run_product.py` · `run_memory_demo.py`.

**Still open (not v1 headline):** L3 decay versioned with W id; compositional / temporal W; multi-domain **pretrain** L1 (scale); tool binding. Branches: [`extension_closed_branches.md`](extension_closed_branches.md) · queue: [`../docs/EXTENSION_PIPELINE.md`](../docs/EXTENSION_PIPELINE.md).

Paper-shaped prose: [`preprint_tapelm_draft.md`](preprint_tapelm_draft.md) **§4.8** · stage index: [`../docs/STAGES.md`](../docs/STAGES.md).

**Essence wins (191–209, unchanged).** Variant A needs no hybrid; variant B falsified; W1–W5 as measured in the next subsection.

### WHERE THE SUBSTRATE CAN GENUINELY WIN (2026-07-29, essence roadmap — W1 CONFIRMED in 204, W3 in 205, W5 bounded in 206; variant-B generation FALSIFIED in 207; hybrid-for-A NO GAIN in 208; A meaning scaling NOT structurally blocked in 209)
Honest separation of "wins by its nature" (needs the char-curve + fp substrate, BPE cannot copy without
becoming char-level) vs "packaging/architectural" (GPT+RAG can match with bolted parts):

- **W1 — noise / OOV / typo / code / multilingual robustness — CONFIRMED (204).** BPE fragments
  rare/novel/misspelled tokens into brittle pieces; a char-curve encoder + fp-lexicon degrades gracefully and
  has a native "novel token" signal (192, AUC 0.98). Essence-level: GPT cannot fix it without abandoning BPE.
  **Measured in 204:** identity retrieval under 30% char noise 0.801 vs 0.403 (2×); hardened noisy-corpus fact
  recall 0.913 vs fair GPT+RAG 0.627. First capability-level (not merely architectural) win over GPT+RAG.
- **W2 — native calibrated abstention / hallucination control (HAVE IT).** fp-lexicon "don't-know" (192/193,
  0.982 vs GPT 0.380) flows from char-fp + lexicon; GPT's BPE surprisal conflates rarity with fakeness.
- **W3 — editable / auditable / *unlearnable* knowledge — CONFIRMED (205).** Facts are explicit fp slots →
  delete-a-fact in O(1) with zero collateral (retained 1.00→1.00, next_tok 0.825→0.825), provenance 1.00,
  contradiction detection 1.00/FP 0.00. Naive parametric unlearning in GPT costs 80% of retained facts and
  0.075 next_tok. Capability win vs parametric GPT; architectural vs a RAG index (which can also delete).
- **W4 — knowledge decoupled from parameters (anti-CF).** Add N facts = N slots on ONE frozen encoder, no
  gradient/CF; GPT needs fine-tune (CF) or a second RAG system. Already demonstrated (194–198).
- **W5 — hops as O(1) vector ops vs O(k) decode/CoT steps — BOUNDED (206).** Latent hops are 16× cheaper than a
  text-valued index at depth 6 (2.35 ms / 1 encoder call vs 37.57 ms / 7) and accuracy-tied on clean data, so this
  is a **latency** win, not a capability one; a vector-valued RAG closes most of the gap. Bonus finding:
  quantize-to-lexicon re-anchoring corrects hop errors for the curve but not for BPE-RAG (which mis-identifies
  corrupted words and propagates the error) — the one part of W5 that is substrate-specific.

**Honest bottom line:** the defensible *essence* wins are W1 (robustness) + W2 (abstention) — both rooted in the
char-curve/fp substrate that BPE can't replicate. W3/W4 are strong but shareable with a bolted RAG stack; W5 is
a research bet. Recommend next experiment attack **W1** (substrate robustness under noise/OOV) — the one axis
where BPE-GPT is structurally weak *by construction*, not just under-equipped.

### EMPTY AREAS MAP (2026-07-29 core; memory ops update 2026-07-30) — what is closed vs open
- **CLOSED / mapped:** context-retention (A), generation parity, lexical calibration ("don't-know"),
  episodic fact recall (hop1/hop2), one-shot editable knowledge, surprise-gated write policy, operable-vector
  chaining + binding composition, honest positioning vs GPT (distinct) and GPT+RAG (architectural, not capability).
- **CLOSED / memory ops trunk (213–230):** freeze default (213), W-remap + family registry (221–225), canonical
  bank + qmap (227), fp decode protocol (228c), contradiction resolution (230), cross-domain utilization (226c).
  Internalization inside forward (210–212) remains **NO**; external fp loop stays the hop API.
- **OPEN (scale frontier):** semantic invariance / meaning over spelling (B). Requires a new encoder pretrained
  with a meaning objective at scale — a separate track gated on stronger hardware.
- **OPEN (engineering, not headline):** L3 decay versioned with W; compositional/temporal W; multi-domain L1 pretrain; tool binding.

### PROGRAM VERDICT (2026-07-29, after 196–198): TapeLM vs BPE-GPT — honest scorecard
| axis | tape | vanilla GPT | GPT+RAG (fair) | distinct from GPT? | distinct from RAG? |
|------|------|-------------|-----------------|--------------------|--------------------|
| generation (parity) | 0.867 | 0.843 | — | tie (entry ticket) | — |
| calibration "don't know" | AUC 0.982 | 0.380 | n/a¹ | **YES** | architectural¹ |
| fact recall (static) | 0.947 | 0.300 | 0.980 | YES | no (RAG ties/ahead) |
| one-shot edit | 1.000 | 0.280 | 0.440² | **YES** | architectural² |
| stream+budget+update | 0.738 | 0.226 | 0.726 | **YES** | architectural |
| **noise/OOV robustness (204)** | **0.913** | — | **0.627** | **YES** | **CAPABILITY** |
| **unlearn collateral (205)** | retained 1.00, nt 0.825 (no change) | retained 0.20, nt 0.733 | n/a (index delete) | **YES** | architectural |
¹ calibration has no standard RAG analog; ² 197 RAG used mean-ctx, fair RAG would improve.
**Bottom line (updated after 204):** TapeLM is genuinely distinct from a vanilla BPE-GPT on every non-generation
axis. On *clean* retrieval a fully-equipped GPT+RAG matches it (architectural distinctness only) — but under
**char noise / OOV (204) the tie breaks**: hardened noisy-corpus recall 0.913 vs 0.627, identity retrieval 2×.
So the thesis is now two-part: (a) one frozen curve fp-space unifies generation + memory + novelty + calibration
where the GPT stack needs bolted components (architectural), and (b) the char-curve substrate is **measurably
more robust to spelling noise and novel words than any BPE-keyed retrieval** (capability). Bound (b) honestly:
a byte/char-level LM would also be robust — BPE is the weak link, not "all rivals".

### North star v2 (2026-07-29, after 191–195) — LM on the SOTE tape

Shift accepted: goal is now **an LM whose substrate is the SOTE tape, with a native
memory + calibration layer that BPE-LMs lack** — NOT chasing the plain LM ceiling.
This is not a retreat to the LM frame we left: the differentiator is the FP memory
(192–195) that gives fact recall + "don't-know" for free on the same frozen encoder.

Why it already mostly stands (evidence, not faith):
- **Tape-as-LM-encoder = parity** (191: next_tok 0.867 vs GPT 0.843). Generative CE head over the curve works.
- **Calibration layer** (192 AUC 0.991 / 193 entropy gap +0.35, zero CE cost): native "I don't know this word".
- **Episodic memory** (194 facts hop1 0.947 / 195 chains hop2 0.70, binding 0.52): answer-what-was-read.
- All on ONE frozen curve encoder, zero extra training — components compose, don't conflict.

**WIN (v2) = a single inference stack on one frozen curve encoder that simultaneously:**
LM-generates (CE head) · recalls facts (FP memory) · knows-it-doesn't-know (FP calibration).
Prior north star ("answer what was in the dataset", 183) is now a *component* (memory), not the whole.

**Anti-clone rule (distinguishability gate — HARD):** generation parity is the *entry ticket, not the win*.
On next-token we aim to TIE GPT by construction (191) — tying there proves nothing new. The win is only
counted on axes where BPE-GPT is structurally weak, and there the curve must **beat**, not tie:
1. **Knowledge edit w/o retraining:** teach 1 new fact at read time → curve recalls it, GPT cannot (needs grad).
2. **Fact/chain recall from a single reading pass** (194/195) vs GPT parametric (~chance) AND vs **GPT+RAG control**.
3. **Calibrated abstention on OOD** (192/193) vs GPT entropy control.
Nearest real rival is **GPT+RAG**, not vanilla GPT. Our only defensible edge over GPT+RAG: memory keys/values,
"don't-know", and generation all come from ONE frozen curve fp-space (not two bolted systems) — a falsifiable
claim about the representation. If the curve only ties GPT on generation and does NOT beat GPT+RAG on 1–3,
we DID become a clone → verdict must say so. Do NOT sell the encoder swap itself as the innovation.

The genuine fork (pick per experiment, not globally):
- **A — encoder-LM + memory (RECOMMENDED, the WIN above):** input=tape, output=text via CE head +
  FP memory + calibration. ~80% done (191 + 192–195); remaining work = assemble one inference stack.
- **B — fully generative curve (STRETCH, not win condition):** model emits next *arc* in R^d, text
  decoded from curve, no token-CE teacher. The 176/177 dream that hit suffix-wipe — now has the two
  things it lacked (objective-flip 178/180 + FP memory). Research bet, gated behind A.

Next concrete (Stage 196 candidate): assemble A into one `TapeLM` inference object over the frozen
191-P1 encoder — CE generation + FP fact memory + FP calibration in a single forward path; measure
all three gates on one held-out slice at once (parity, recall, OOD calibration) to prove they compose.

### LOCKED direction (2026-07-28, after 183)

Invariant proven 170–183: local next-* teacher → suffix-wipe at any unit; objective-flip (178/180) fixes
retention architecturally, not by scale. Retention ≠ meaning (B). Exam 183 not trusted (GPT also ~chance).

Concept upgrades (accepted):
1. **Tape stores "what became known", not "where the pen went".** Split 3 layers: substrate (ink geometry),
   knowledge-state (what is fixed by t), read (query over tape). Stop treating endpoint `z_T` as the memory.
2. **Addressable tape + query head** — keep `z_1..z_T`, answer by attention/query, not last point.
3. **Slow channel carries a content invariant** (same for paraphrase, differs for hard-spelling), not random instance.
4. **Self-model / "understands own processes":** predict-own-next-state + surprise; write to slow memory on
   surprise (not every step); confidence calibrated (low on OOD = knows-it-doesn't-know).

Step order (HARD RULE — measurer first):
- **S1 (184)** Calibrate exam: cloze via **log-prob** (not cos), bigger N, real OOD. Gate: **GPT must beat chance**,
  else exam still lies and no curve conclusions allowed.
- **S2** Addressable tape + query head vs endpoint (on calibrated exam).
- **S3** Self-model: predict-own-next-state + surprise-gated write; gates = surprise↑ on novel, confidence↓ on OOD,
  A holds without constant retention push.
- **S4** Attack meaning (B) via slow content-invariant contrast; only here compare to 181 ceiling.

| Stage | Intent |
|-------|--------|
| **184** | Log-prob exam → **CALIBRATED** (GPT next_tok=0.76, hybrid=0.65, random=0.28); entity-recall ~chance for ALL (incl GPT) → entity-recall is wrong exam at this scale, not a curve fail |

**184 read:** harness now honest. next_tok proves context is usable (GPT 0.76, hybrid_182 0.65 ≫ chance).
Entity recall unanswerable at 4L/d128/20M for everyone → dataset-answer exam must use answerable tasks
(continuation / same-doc / order), NOT specific-entity recall. Hybrid already shows real context use.

**184 CORRECTION (found in 185):** unigram frequency alone scores next_tok=0.65 (gold = usually frequent
token, distractors random). So context credit = score − 0.65: GPT +0.11, hybrid_182 ≈ **+0.00** (!).
Head-to-head comparisons on this exam stay fair (both sides get the unigram shortcut), but absolute
"context use" claims require beating 0.65. Exam v2 TODO: frequency-matched distractors.

| Stage | Intent |
|-------|--------|
| **185** | Tape query-read vs endpoint → **ENDPOINT_ENOUGH_HERE** (both 0.792; shuffle drop 0.000 — read head unused) BUT clean-CE dual beats GPT ref 0.758 and unigram 0.650 |

**185 read (two lessons):**
1. **Hand losses were the poison.** Same dual-channel, pure CE, no retention/instance losses →
   next_tok 0.792 (+0.14 context over unigram), vs 182's multi-loss 0.650 (+0.00). The retention
   push was actively blocking context learning. Rule going forward: no auxiliary loss unless it
   survives an A/B against pure CE.
2. **Query-read over slow is redundant at 64-arc contexts** — the fast transformer's own attention
   IS already an addressable tape within its window. Slow-tape reading can only matter beyond the
   attention window (long docs) or across documents (persistent memory). "Endpoint = ceiling" is
   NOT falsified in-window; retest only with context > window or persistent cross-doc memory.
   Caveat: GPT-181 ref trained under a different recipe; 0.792 vs 0.758 is encouraging, not a
   matched-budget victory claim.

| Stage | Intent |
|-------|--------|
| **186** | Exam v2 (freq-matched distractors) → **EXAM2_OK**: unigram 0.65→0.25 (shortcut dead), GPT 0.727, endpoint 0.707, tape 0.753 but shuffle-ablation drop only 0.006 → tape still decorative; endpoint≈tape≈GPT within noise |

**186 read:** measurer now trustworthy end-to-end (chance 0.25, unigram dead, GPT calibrated).
Curve dual-channel with clean CE = parity with GPT control on real context (0.71–0.75 vs 0.73).
Entity recall still ~chance for everyone (scale-bound, as 184 said). Exam v2 = standard judge from now.

| Stage | Intent |
|-------|--------|
| **187** | Self-model (predict-own-next-write + surprise-gated slow writes) → **SELF_MODEL_PARTIAL_3**: G1 PASS (next_tok 0.727 = GPT parity, CE not poisoned), G2 PASS (surprise unseen 0.0582 > seen 0.0556), G3 FAIL (entropy after fake 4.61 < real 4.93 — MORE confident on nonsense) |

**187 read:** self-model works mechanically and free of charge (G1: aux predictor loss with detached
target didn't hurt CE — 0.727, best curve number, = GPT). G2: surprise discriminates novelty (small
margin). G3 control (187b): GPT DOES raise entropy after fakes (2.66>2.52) → failure is curve-specific.
Cause hypothesis: char-level ink encoder generalizes fake words into familiar word-shapes (no rarity
signal), while BPE splits fakes into rare pieces. The internal surprise signal EXISTS (G2) but is not
wired into the output distribution. Next (S3b): condition the head on the surprise/expectation-error
state so "I didn't expect this" becomes visible confidence — curve-native calibration GPT gets for
free from tokenization.

| Stage | Intent |
|-------|--------|
| **188** | Surprise→temperature (with gradient) → **PARTIAL_23, GOODHART COLLAPSE**: model killed its own surprise (self-err 0.056→0.003, surprise flat 0.0015 real=fake); G2 died too; G1 held 0.700 |
| **189** | Read-only surprise (detach) → **PARTIAL_3**: detach fixed Goodhart (self-err 0.071, G2 alive again, G1 0.700), but G3 still fails — diag: surprise@fake 0.0545 < @real 0.0655 |

**188 lesson (rule):** introspection signals must be READ-ONLY w.r.t. the objective they modulate,
otherwise CE games them away. Any self-model channel that feeds the output needs a gradient cut.

**189 read (S3 line CLOSED):** lexical know-it-doesn't-know is blocked by the SUBSTRATE, not the wiring:
the char-level ink encoder genuinely normalizes word-shaped fakes (less surprising than real rare
entities). BPE gives GPT this rarity signal for free. Fix would be a rarity feature in the ink encoder
(char-n-gram novelty) — deferred; not blocking S4. S3 net result: doc-level novelty detection YES,
lexical OOD calibration NO (substrate limit, documented).

| Stage | Intent |
|-------|--------|
| **190** | S4 doc-level InfoNCE on slow endpoint → **CONTENT_INV_PARTIAL_3**: G1 PASS (0.713, CE intact), G2 PASS (doclink 0.670 vs 0.590), G3: para 0.979→0.909 while hard stays 0.995 |

**190 read (S4 first shot):** the contrast survives the A/B (first representation loss that doesn't
poison CE) and learns a real invariant — but a TOPICAL one, not semantic. De-collapse revealed the
form-over-meaning signature explicitly: hard pairs (car/cat, 95% surface overlap) stay glued at 0.995,
paraphrases get pushed apart to 0.909. Doc-structure supervision says "same doc = same words", which is
lexical topicality. To beat hard pairs the training signal must assert "car ≠ cat despite surface" —
nothing in a 20M-char corpus provides that for free. Semantic invariant = open; needs either scale,
cross-lingual/paraphrase data, or prediction-consequence supervision (car vs cat → different continuations).

| Stage | Intent |
|-------|--------|
| **191** | NIGHT-9h scale run (150M chars — prior runs were on ~2M fallback slice!, d256/6L, 15k steps) → **NIGHT_PARITY_HELD + meaning emerges** |

**191 results (2026-07-29 morning):**
- next_tok v3: curve-XL **0.867** vs GPT-XL 0.843 (unigram 0.27) — parity held at 75x data, curve nominally ahead.
- Rarity ink (P3): 0.853 + **entropy calibration PASSED** (fake 4.997 > real 4.947) — knows-it-doesn't-know
  visible in output at scale; internal surprise on fakes still not higher (only output-level fixed).
- Meaning: P4 endpoint-vs-meanpool comparison was UNFAIR; fair probe (191b, mean-pool both):
  curve fast channel para 0.705 / hard 0.889 (**gap 0.185**) beats GPT (0.819/0.937, gap 0.118);
  doclink curve 0.69 vs GPT 0.66. **Scale moves meaning in BOTH; curve's meaning lives in the fast
  channel, slow endpoint is a memory summary, not semantic space** (slow stayed collapsed ~0.99).
- Entity recall still ~chance for all (0.27–0.30) — true fact-recall needs far more scale.
- Bug found: 181's doc splitter fell back to ~2M chars all day — day results were on 2M, not 20M.

| Stage | Intent |
|-------|--------|
| **192** | Old word-FP theory × curve: FP-lexicon lexical surprise (zero training, frozen 191-P1) → **FP_LEXICON_SURPRISE_YES** |

**192 results:** fp(word)=normalize(arc_enc(word)); lexicon = 150k corpus words. surprise = 1−max_cos:
real entities 0.0007 vs fakes 0.118, **AUC 0.991** (trigram baseline 0.926). The gate that failed in
187/189/191-P3 (surprise fake>real) passes near-perfectly, read-only by construction (no Goodhart path).
"I don't know this word" = "not in MY lexicon", not "odd letters". Old SOTE fingerprints + curve's
learned encoder are complementary: curve learns the fp space, FP-machinery gives it memory semantics.
Next candidates: wire lex-surprise into head temperature (189-style, read-only); FP edge-binding
(fp_A⊙fp_B) as slow-tape write format for beyond-window memory; top-2 gap as confidence.

| Stage | Intent |
|-------|--------|
| **193** | Wire FP-lexicon into head temperature (frozen backbone, 2 params) → **FP_WIRED_YES**: G1 zero-cost (0.855→0.855), G3 entropy fake 5.21 > real 4.85 (gap +0.35 vs P3's +0.05); CE itself learned w=+8.3 (trusts the lexicon signal) |

**G3 line CLOSED (192+193):** internal signal AUC 0.991 + visible calibrated behavior + zero CE cost.
The knows-it-doesn't-know channel BPE gives GPT for free is now curve-native via the old SOTE
fingerprint theory. Read-only wiring (frozen backbone) also structurally immune to 188's Goodhart.

Next recommended (194): **FP fact memory for entity recall** — the north-star metric still at chance
for ALL systems. Old SoftPhraseMemory descendant: while reading, write edge fps (fp_ctx ⊙ fp_entity)
into slot memory; at question time retrieve by unbinding. Hop1 only (old failure was hop2 @ d64; now
d256). This attacks "answer what was in the dataset" directly, not via LM scale.

| Stage | Intent |
|-------|--------|
| **194** | FP fact memory (episodic slots, hop1, zero training) vs entity cloze → **FACT_MEMORY_YES: 0.947** (chance 0.25, ALL LM systems incl. GPT ≈0.27); falsification control (unread memory) = 0.274 ≈ chance |

**194 read — first QUALITATIVE win of the program.** "Answer what was in the dataset" (the north star):
while reading, write slots key=fp(context around entity), value=entity; answer = retrieve by cosine.
Read memory 0.947; memory built from UNREAD text drops to chance → answers provably come from reading.
Honest framing: this is episodic/retrieval memory, not parametric knowledge — but that IS the north
star as defined, and the same frozen curve encoder simultaneously provides LM parity (0.867), meaning
(gap 0.185), calibration (192/193) and now fact recall. The 2018-era SOTE fingerprint theory supplied
the memory semantics the curve lacked; the curve supplied the learned fp space SOTE lacked.
Open next: hop2 (chains) at d256 — the old failure mode, now retestable; scale memory to full corpus;
write-on-surprise integration (187) to make reading selective.

| Stage | Intent |
|-------|--------|
| **195** | hop2 chains + edge-binding retest (d256) → **HOP2_CHAIN_YES+BINDING_YES**: chain 0.70 vs direct-shortcut 0.35 (chance 0.25); old SOTE edge_fp binding 0.52 — the d64 collision failure does NOT reproduce at d256 |

**195 read:** two-hop retrieval works on learned fps (hop1 B-retrieval = 1.00; chain doubles direct).
First items version was leaky (adjacent-paragraph topic shortcut, direct=0.48) — hardened with
topic-matched distractors from paragraphs adjacent to P2. Old Stage29b verdict "hop2/joint
collision-bound" is now scale-resolved: binding fp_A⊙fp_B clears 0.5 at d256 with 28k edges.
FP-line status: lexicon (192) + calibration (193) + facts hop1 0.95 (194) + chains hop2 0.70 (195)
— all zero-training on one frozen curve encoder.

### Program S1–S4 synthesis (2026-07-28)

- Measurer: fixed and trusted (log-prob, freq-matched, unigram-dead; GPT calibrated at 0.727).
- Context: curve at GPT parity (0.727 = 0.727) with clean CE; hand losses were the historical poison.
- Memory: endpoint sufficient in-window; addressable tape only meaningful beyond attention window.
- Self-model: free (CE intact), detects doc novelty, must be read-only (Goodhart otherwise);
  lexical OOD calibration blocked by ink substrate (no rarity signal).
- Meaning: first non-poisonous representation objective; invariant learned is topical, semantic still open.

## Curve-as-tokens / Curve-BPE

**176 was wrong analog** — rule split on `\S+`, dropped spaces. Real BPE:

1. Merges by **corpus frequency** (not meaning / grammar).  
2. Frequent forms often 1 piece; rare/new → several.  
3. Space usually **inside** token (ByteLevel `Ġ` / decoded leading space).  
4. Each piece → continuous arc vector on the ink of that piece.  
5. Loss = next-piece + Δ in `z` — **no** BPE-id CE.  
6. Gate A on **last BPE piece**, not char/word suffix.

**176 result:** whitespace units still last-wipe under next-arc.  
**177:** falsify whether *statistical* units change that.

---

## Kill / go rules

- **GO** after 171 if frozen-pen dynamics still beat baselines on hold (+ multi-step).  
- **KILL recipe** if only joint pen+dyn works and frozen pen collapses to baseline.  
- **Never** unfreeze 169 CE path to rescue a failed curve gate.

---

---

## Memory extension — document map

Stage-by-stage reads live **above** under **Memory extension program (213–230)** (after the 210–212 frontier). Do not duplicate numbers here.

| Doc | Role |
|-----|------|
| [`extension_memory_contract.md`](extension_memory_contract.md) | Product contract (L1–L3, registry, API order) |
| [`extension_thesis_W_remap.md`](extension_thesis_W_remap.md) | W-remap thesis + 221–224 |
| [`extension_closed_branches.md`](extension_closed_branches.md) | Explored NO branches |
| [`preprint_tapelm_draft.md`](preprint_tapelm_draft.md) | Paper draft §4.8 |
| [`../docs/STAGES.md`](../docs/STAGES.md) | Script + verdict table |

---

## Files

| File | Role |
|------|------|
| `results/plan_curve_dynamics.md` | this locked contract + plan |
| `results/stage170_contract.json` | machine-readable lock |
| `_stage170_curve_dynamics.py` | smoke runner (reference) |
| `results/stage169_FROZEN.json` | prior path frozen |
| `results/preprint_tapelm_draft.md` | paper draft (170–212 + §4.8 memory) |
| `results/extension_memory_contract.md` | memory product contract |
| `legacy/docs_ru/extension_plain_ru.md` | RU plain guide (213–230); stub at `results/extension_plain_ru.md` |
