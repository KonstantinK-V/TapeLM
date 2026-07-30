# TapeLM: Unified Fingerprint Memory on a Curve Encoder  
**Draft preprint (internal)** — 2026-07-30 (§4.8 memory trunk; frozen-P1 contract aligned with `docs/ARCHITECTURE.md`)  
*Authors: [TBD]*  
*Code & stage logs: [KonstantinK-V/TapeLM](https://github.com/KonstantinK-V/TapeLM) (Stages 170–230)*

---

## Abstract

We study a language-model stack whose substrate is a **dual-channel curve encoder** (character-level ink + slow write-budget memory) rather than a monolithic token embedding. On top of a **single frozen encoder** (d256, 6 layers, trained on ~150M characters), we attach **zero-training** modules derived from classic **word-fingerprint (FP)** theory: an editable lexicon for lexical surprise, episodic slot memory for facts, subject-anchored one-shot writes, and vector binding for multi-hop queries. The resulting **TapeLM** matches a matched GPT-2 control on next-token prediction (0.867 vs 0.843) while beating vanilla GPT on calibration (lexical OOD AUC 0.982 vs 0.380), parametric fact recall (0.947 vs ~0.30), one-shot knowledge edit (1.00 vs 0.28), and beyond-window streaming under memory budget (0.74 vs 0.23 in-context). Against a **fair GPT+RAG** baseline (same retrieval math and surprise-gated admission), capability is **parity**, not dominance: static recall and long chains tie when anchors are clean; the defensible claim is **architectural unification**—generation, memory keys, novelty gating, and calibration share one fp-space without extra trained retrievers. Two axes do break the tie with a fair GPT+RAG. Under **character noise and out-of-vocabulary input** the curve retains 0.913 hardened recall against RAG's 0.627 and twice its corrupted-word identification accuracy, because every typo re-fragments a BPE token while a character fingerprint degrades smoothly; and **targeted unlearning** removes a fact in O(1) with retained recall and next-token accuracy bit-identical, whereas naive gradient unlearning of the same facts in the parametric model destroys 80% of the retained facts. Semantic invariance over adversarial paraphrase (PAWS) remains **scale-bound** for both curve and GPT at this size (~0.70 test accuracy, no para/hard inversion), identical to the matched transformer. We also report three explicit negative results on the **output and internalization** frontiers (§4.5–4.7, §5.4): latent multi-hop composition is a **latency** advantage only; a **fully generative** fingerprint-output variant is falsified; and a hybrid word-level fingerprint reranker gives the token head **no measurable gain** on clean text. **After the internalization frontier closed (210–212),** we extended the **same frozen encoder** with an **operational memory trunk** (§4.8): family **W** remaps after domain shift, **canonical** slot storage with read-time qmap, **fp decode** when the CE head underuses retrieved values, and a **resolution policy** over contradictory slots—cross-domain utilization reaches ~0.88 via fp decode vs ~0.45 for the code head alone (226c). Code and per-stage JSON decisions are available in the repository.

**Keywords:** character-level language model; curve encoder; dual-channel memory; word fingerprint; episodic slot memory; zero-train retrieval; canonical memory; domain remap; vector binding; RAG comparison; knowledge editing; machine unlearning; OOD calibration; multi-hop retrieval; reproducible staged evaluation

---

## 1. Introduction

Standard causal LMs store knowledge **parametrically**; retrieval-augmented generation (RAG) stores knowledge **externally** and re-reads **text chunks** at query time, usually through a **second** encoder or index geometry. We ask whether a **third contract** is viable: knowledge as **structured, operable vectors** in the **same** representation space used for generation—subject-anchored slot keys, binding/unbinding, multi-hop composition without serial decode, editable lexicon calibration, selective write policies, and (in §4.8) canonical banks with explicit conflict resolution—not “the same cosine retrieval with a different embedder.”

This work reconnects a 2018-era **word-fingerprint** idea (normalized encodings + binding/unbinding) with a modern **curve LM** (dual fast/slow channels, clean CE, frozen after scale training). The product is **TapeLM**: one frozen encoder; **knowledge structure** (relations, hops, edits, contradictions) expressed as **fp-space operations**, not as re-ingested strings.

**What we do not claim:** beating GPT+RAG on raw retrieval accuracy; human-level semantic understanding; superiority of generation quality.

**What we do claim:** (0) **Not RAG-with-another-embedder:** one geometry for generate + key + calibrate + compose; RAG can approximate retrieval scores with a fair index, but does not natively expose bind/hop/edit/resolution as **vector APIs** (§5.1); (1) a reproducible stack where memory/calibration/edit layers are **zero-train on P1 at inference** (facts = slot writes; no backbone gradients); domain drift uses **offline-fit family W**, not re-indexing the bank (§4.8); (2) clear wins vs **vanilla** GPT on non-generation axes; (3) honest parity vs **full** GPT+RAG on **clean** retrieval; (4) demonstrated **vector-native** structure (chaining without decode, binding, subject writes, 230 resolution) that text-RAG does not treat as first-class.

---

## 2. Background

### 2.1 Curve substrate (Stages 170–191)

- **Stream:** characters, not BPE ids, as the ink that draws latent states.
- **Dual channel:** fast transformer over arc embeddings + slow surprise-gated writer (retention without hand-crafted retention losses; auxiliary losses that poisoned CE were ablated in Stage 185).
- **Training:** next-piece CE on BPE targets; self-model surprise (read-only for calibration wiring per Stages 188–189).
- **Scale reference (Stage 191, RTX 3050):** SelfModelXL d256/6L, ~15k steps, ~150M chars; parity with matched GPT-2 (0.867 vs 0.843 next-token, Exam v3 freq-matched distractors).

### 2.2 Fingerprint (FP) layer (Stages 192–195)

- **fp(word)** = normalize(`arc_enc`(word characters)).
- **Lexicon surprise:** `1 − max_cos(fp(w), lexicon)` — real entities ≈ 0, fakes ≈ 0.12 (AUC 0.991).
- **Episodic memory:** slot key = mean fp(context words around entity), value = entity string; query by cosine (hop1 entity recall 0.947 vs chance 0.25).
- **Hop2 chains:** context → inferred B → B’s other context → C (0.70 with topic-matched distractors); **edge binding** `norm(fp_A ⊙ fp_B)` rehabilitated at d256 (0.52), failed at d64 in prior SOTE work.

---

## 3. TapeLM architecture (inference)

All modules below use the **same canonical P1 encoder** (`checkpoints/stage191_p1_curve.pt`): **pretrained once (Stage 191)**, then **fixed for product memory operations**—slot ingest, recall, calibration wiring, and conflict resolution do **not** update `arc_enc` weights (see §3.1). Diagram: `docs/ARCHITECTURE.md` in the repository.

| Module | Mechanism | Training |
|--------|-----------|----------|
| **Generate** | CE head on `[fast; slow]` | Pretrained (191-P1) |
| **Calibrate** | FP-lexicon surprise → head temperature (193) | 2 scalars only |
| **Recall** | Episodic slots (194) | Zero at use |
| **Edit** | Subject-anchor write `key = norm(fp(S)+ctx)` (197) | Zero at use |
| **Write policy** | Admit slots by fp-lexicon surprise under budget (197–198) | Zero at use |
| **Compose** | Sequential fp retrieval or binding (195, 200) | Zero at use |
| **Canonical bank (227)** | Write keys in P1 fp; read via **qmap** `W_bwd @ q_domain` | Zero at use; **W** fit offline (~800 core words) |
| **Fp decode (228c)** | 4-way slot retrieve + `cos(fp(c), fp(retrieved))` | Zero |
| **Resolve (230)** | Policy over multi-hit contradictory slots (229) | Zero |

### 3.1 Frozen P1 — product contract

| Component | Updated when adding/reading facts? |
|-----------|----------------------------------|
| **P1 `arc_enc`** (191 checkpoint) | **No** — load, `eval()`, no gradients on memory path |
| **Episodic slots / lexicon tables** | **Write/delete only** — no encoder training |
| **`W_family`** (221–225) | **No online** — learned in stage exams or `export_w_registry.py`, shipped in `w_registry/` |
| **CE / optional heads (225)** | **No** on default product demos |

**Not the same as “never trained”:** P1 was trained in 191; **W** is a tiny d×d map fit after a *measured* geometry shift. Research stages may finetune a **copy** of `arc_enc` to simulate drift; production keeps **canonical slot keys** and migrates coordinates with **W**, not full reindex (§4.8).

**Anti-Goodhart rule (188):** introspection signals that modulate outputs must be **gradient-detached** from the CE path.

---

## 4. Evaluation protocol

### 4.1 Exams

- **Exam v2/v3:** frequency-matched distractors; log-prob scoring; GPT must beat chance before curve claims (184–186).
- **Entity recall, hop2, streaming, edit:** task-specific gates with **unread-memory** or **in-context** controls.

### 4.2 Anti-clone scorecard (Stage 196–198)

| Axis | TapeLM | Vanilla GPT | GPT+RAG (fair) | vs GPT | vs RAG |
|------|--------|-------------|----------------|--------|--------|
| Next-token | 0.867 | 0.843 | — | tie | — |
| Lexical OOD (AUC) | **0.982** | 0.380 | n/a | **yes** | arch. |
| Static fact recall | **0.947** | ~0.30 | **0.980** | yes | no |
| One-shot edit (4-way) | **1.00** | 0.28 | ~0.44–0.73† | **yes** | arch. |
| Stream + budget | **0.738** | 0.226 | **0.726** | **yes** | tie |
| **Noisy corpus + query (204, 8-way)** | **0.913** | — | **0.627** | **yes** | **capability win** |
| **Unlearn collateral (205)** | retained **1.000**, nt **0.825** | retained 0.200, nt 0.733 | n/a (index delete) | **yes** | arch. |

† RAG improves with subject-anchor mirroring (198); edit win over *generic* RAG remains architectural (surprise admission 0.98 vs 0.20 novel kept under budget).

### 4.2b Robustness to spelling noise and OOV (Stage 204)

All metrics are **rank-based** (retrieve the right item), so curve-space and GPT-space cosines are never compared directly; the RAG control uses the *identical* subject-anchored key/query recipe and differs only in the encoder.

**Identity retrieval** — query is a corrupted word, target is its clean form in a 1000-word pool (700 corpus entities, 300 novel pronounceable words), acc@1:

| char-noise rate | curve (seen) | GPT (seen) | curve (OOV) | GPT (OOV) |
|-----------------|--------------|------------|-------------|-----------|
| 0.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 0.883 | 0.549 | 0.930 | 0.747 |
| 0.2 | 0.841 | 0.507 | 0.890 | 0.707 |
| 0.3 | **0.801** | 0.403 | **0.860** | 0.567 |

**Fact recall.** With clean stored text and noisy queries (4-way), curve stays 0.99–1.00 across all rates while fair RAG falls 1.000 → 0.880 (relative drop 0.000 vs 0.120) — informative but near ceiling. The hardened condition corrupts **both the stored corpus and the query with independent noise realizations** (8-way, chance 0.125): curve 1.000 / 0.947 / 0.913 vs RAG 0.993 / 0.733 / 0.627 at rates 0.0 / 0.2 / 0.3. Both begin at ceiling, so the separation is attributable to noise rather than task difficulty.

**Mechanism.** Mean BPE pieces per word rises 3.93 → 4.76 with noise: each typo re-fragments the token, so BPE-keyed embeddings of the clean and corrupted form drift apart, whereas the char-curve fingerprint changes smoothly.

**Significance.** Stages 198 and 200 established that a fully-provisioned GPT+RAG *ties* the tape on clean retrieval, leaving only architectural distinctness. Stage 204 breaks that tie: this is the first **capability-level** advantage over the strongest rival. It should be bounded honestly — a byte- or character-level LM (ByT5-style) would also be robust, so the structural weak link is **BPE tokenization specifically**, not every possible architecture; and the GPT control here is our matched d256 model rather than a large pretrained one (a larger BPE vocabulary mitigates but does not remove re-fragmentation).

### 4.2c Targeted unlearning, provenance, and contradiction audit (Stage 205)

We planted 60 facts (20 designated for deletion, 40 retained) into 400 memory slots, and separately fine-tuned the matched GPT until it genuinely memorized the same facts, so that "unlearning" is not vacuous. The curve deletes the 20 slots; GPT is unlearned by gradient ascent on the same facts with early stopping the moment target recall reaches chance (the least-damaging version of that baseline).

| Metric | curve before | curve after delete | GPT after memorize | GPT after unlearn |
|--------|--------------|--------------------|--------------------|-------------------|
| Target fact recall (4-way) | 1.000 | **0.000** | 1.000 | 0.000 |
| Retained fact recall | 1.000 | **1.000** | 1.000 | **0.200** |
| Next-token (collateral probe) | 0.825 | **0.825** | 0.808 | **0.733** |

Deletion removed 20 slots in under a millisecond with no gradient computation, whereas GPT required 30 gradient steps and lost 80% of the *retained* facts plus 0.075 next-token accuracy. Target recall falls to 0.000 rather than to chance because the deleted value is absent from memory and can never be selected — erasure is complete and detectable, not degraded guessing. Two further audit properties follow from slots being explicit: **provenance attribution is 1.000** (the top-scoring slot is always the one written from the gold fact's own text), and with two conflicting slots for one subject the top-two scores fall within 0.02, so **contradictions are flagged at 1.000 with a 0.000 false-positive rate** instead of being silently resolved.

**Bounds.** Against a **GPT+RAG index this is architectural**, since an index can also delete entries and expose provenance; the capability contrast is against *parametric* knowledge. Gradient ascent is a crude unlearning baseline — localized editors (ROME/MEMIT) or retain-set-regularized objectives would reduce collateral, so the honest reading is that *naive* parametric unlearning is destructive, not that no method succeeds. Provenance and conflict figures are measured on planted, unique, novel subjects; overlapping real-world subjects would be harder.

### 4.3 Operable vectors vs index (Stage 200)

- **curve_vector:** multi-hop retrieval staying in fp-space (no string decode between hops): 1.00 on planted chains (k≤3).
- **binding one-shot 2-hop:** 1.00.
- **Fair GPT-index RAG:** also 1.00 on the same task (unique novel anchors → no compounding error).
- **Vanilla GPT in-context:** ~0.20 beyond window.

**Interpretation:** vector-native composition is **real** but **not a accuracy differentiator** when RAG’s index is clean; differentiation is **mode of computation**, not benchmark margin.

### 4.4 Semantic invariance (Goal B) — negative results

Probes: 179 paraphrase/hard sentence pairs; PAWS adversarial paraphrase (202–202b).

| Attempt | Curve | GPT control | Inversion (para > hard)? |
|---------|-------|-------------|---------------------------|
| Frozen CPC head (199) | gap 0.185→0.163 | — | no |
| Word hard-negs on copy (201) | hard ↑ | — | no |
| PAWS head, frozen enc. (202) | PAWS 0.644 | 0.655 | no |
| PAWS fine-tune copy (202b) | PAWS 0.705 | 0.701 | no |

**Conclusion:** at d256/6L and ~150M chars, **B is scale-bound and substrate-neutral** (curve never worse than matched GPT on semantic probes).

### 4.5 Generative variant B (fingerprint output space) — negative result

The architecture so far is an *encoder-LM*: input is the curve, output is text via a cross-entropy head. A stronger claim — "the curve **is** the thinking, text is only decoded" — would have the model emit the next word's **fingerprint** in R^d and decode by snap-to-lexicon, with no token softmax. Stage 207 tests this directly: the same frozen arc-encoder feeds two models with an identical 3.3M-parameter causal trunk and 3500 training steps; one predicts the next fingerprint (in-batch InfoNCE, contrastive next-arc), the other predicts the next word ID via a closed top-8k softmax.

| Gate | Curve (fp output) | CE (closed softmax) |
|------|-------------------|---------------------|
| G1 next-word (4-way, in-vocab) | 0.536 | **0.861** |
| G3 next-word (4-way, OOV-for-CE) | **0.315** | 0.245 (= chance) |
| G2 free-run drift (raw fp → nearest lexicon) | 0.89 from step 1; snap decode 0.857 repetition | — |
| G4 trunk hidden as memory key | 0.175 | — |

The fingerprint-generative model learned almost nothing (InfoNCE loss 7.53 → 7.25 against a uniform floor of ≈7.74, versus CE 6.04 → 5.35). **The mechanism is structural:** a fingerprint is a character-derived encoding, so "emit the next point on the curve" asks the model to regress *how the next word is spelled*, which is not predictable from context beyond word identity, and fingerprint neighbours are look-alike spellings rather than distributional substitutes. This is the same failure seen for next-arc prediction in the earliest stages (176–177) and explains why only a cross-entropy-style signal proved usable (185). The one property that survives is that the closed softmax is **at chance on out-of-vocabulary words by construction** (0.245), whereas the open metric vocabulary can at least rank them (0.315) — but the generative model is far too weak to exploit that advantage. We therefore report variant B as **falsified in its literal form at this scale**: decoupling the output code from spelling would help but amounts to reintroducing a learnable (CE-like) vocabulary, abandoning the "output space is the substrate" claim. The finished artifact is variant A (encoder-LM plus FP memory, calibration, and edit).

**Scope correction.** The out-of-vocabulary gate above was run against a *word-level* closed softmax, which is a weaker rival than the BPE head actually used in variant A: BPE can spell an unseen word piecewise and is therefore not structurally blind to it. The open-vocabulary property should accordingly **not** be claimed as an advantage over BPE-GPT; only the noise-robustness result of §4.2b is.

### 4.6 Hybrid rare-word head for variant A — no measurable gain (Stage 208)

Since §4.2b traces BPE's weakness to fragmentation, a natural upgrade is to give variant A a whole-word fingerprint channel for rare words. Unlike §4.5 the fingerprint side here is **discriminative** — it only ranks four frequency-matched candidates using the frozen context state, rather than regressing a spelling code — and it does learn (contrastive loss 1.61 → 0.62). It nonetheless adds nothing:

| Scorer | all | common band | rare band |
|--------|-----|-------------|-----------|
| A only (BPE CE head) | **0.551** | **0.669** | **0.431** |
| fp reranker only | 0.396 | 0.435 | 0.355 |
| combined (dev-tuned weight 0.25) | 0.541 | 0.667 | 0.413 |
| gated by fp-surprise (0 common / 0.5 rare) | 0.544 | 0.669 | 0.416 |

Four-way frequency-matched candidates within band (chance 0.25), test n=700 with standard error ≈0.019, gate signal read-only, encoder and head verified bit-identical. The tuner did recover the intended policy — trust the fingerprint channel only on rare words — and still gained nothing; every delta against A lies inside noise, so the honest reading is *no gain* rather than harm. A is weakest exactly where predicted (rare 0.431 versus common 0.669), but the whole-word fingerprint is weaker **there too** (0.355): on clean text it carries redundant, lower-resolution information than BPE piece composition. This is consistent with §4.2b, where fragmentation only becomes a liability once the input is corrupted, and it closes the question of whether variant A needs a hybrid output: it does not.

### 4.7 Semantic scaling on 3050 — structural block refuted, full B not confirmed (Stage 209)

Goal B (semantic invariance, PAWS inversion) plateaued near ~0.70 in Stages 202–202b with no evidence that the curve encoder was *worse* than a matched GPT — but also no proof that meaning could improve with scale on consumer hardware. Stage 209 runs a **scaling grid** (d128/2L, d192/4L, d256/6L from matched CE pretraining) plus a **MiniLM-L6-v2 teacher probe** at d256 (Pearson *r* between student pairwise cosines and teacher cosines on held-out sentences).

| Scale | curve PAWS | GPT PAWS | Δ |
|-------|------------|----------|---|
| d128/2L | 0.596 | 0.580 | +0.016 |
| d192/4L | 0.602 | 0.624 | −0.022 |
| d256/6L | 0.632 | 0.640 | −0.008 |

Curve PAWS rises monotonically across scales; **parity** holds (curve ≥ GPT−0.03 at every point). Teacher geometry at d256: curve *r*=**0.256**, GPT *r*=**0.270** (both weak but curve within the parity band). **No para/hard inversion** at any scale; PAWS remains below 0.70 everywhere. Verdict **STRUCTURAL_BLOCK_NO**: on this GPU budget, variant A tracks GPT’s semantic scaling trajectory and is not structurally blind to meaning geometry — but 3050 **does not confirm** strong Goal B (inversion or PAWS ≫ 0.70); that remains a larger-encoder / stronger-pretrain frontier shared with small GPT baselines.

### 4.8 Canonical memory, domain remap, and fp decode (Stages 221–230)

The core stack (§2–4) keeps **one frozen P1 encoder** for canonical storage and **external** zero-train slot operations at inference. Deployments must handle **geometry drift** (e.g. code vs prose query ink) **without** rebuilding the bank or finetuning P1 on each new fact. We treat memory as **three layers**: (L1) freeze canonical P1 by default; (L2) tiny **W_family** linear remaps on ~800 core words (fit **offline**, applied @ read); (L3) stream/decay policies (219, open).

**W-remap (221, 224–225).** After a controlled domain shift, a d×d adapter **W** maps old fingerprints toward new encoder geometry (align ~0.997 on core; fact recall ~0.78 vs ~0.87 oracle reindex). **Code-class** shifts are harsher (cos ~0.59) but still recoverable (~0.88 matched W). A **family registry** (`prose`, `code`, fork-on-drop) beats one global W (224–225: **DOMAIN_BUNDLE_OK**).

**Canonical storage (227).** Slot keys are written in **canonical** P1 fp; at read time queries use **qmap** (`W_bwd` maps domain query → canonical key space). Cross-family recall ~0.95 vs ~0.70 without W; one bank, disposable lenses.

**Decode and utilization (228b–228c, 226c).** Retrieval under a closed candidate set must use **4-way slot retrieve** (per-value max key score), not global argmax over all keys (228b: ~33% vs 227 protocol ~1.0). Given the retrieved value, **fp decode** scores candidates with `cos(fp(c), fp(retrieved))` rather than raw query–candidate cosine (228c: **1.0** vs head ~0.48 on the code-return exam). End-to-end cross-domain (**226c**): recall_4way ~0.88, fp decode ~0.88 vs code head ~0.45.

**Contradictions (229–230).** Slots may return **multiple** conflicting values (both in top-2 ~60%; score gaps small). Resolution is a **policy layer** (provenance, recency, query cues): composite policy **~1.0** on cued exams vs raw argmax **~0.47** (230). This complements §4.2c audit (flag contradictions) with **selection**.

**What remains open.** Head-only text inject without fp decode still fails to utilize memory (226, 228a). Compositional W, temporal W, and tool binding are research branches, not part of the v1 product claim. Weights: P1/P2 + optional `w_registry/` on Hugging Face; demo: `artifact/scripts/run_product.py`.

---

## 5. Discussion

### 5.1 Unified fp-space vs GPT+RAG (thesis, not a second embedder)

**What TapeLM is not.** A fair **GPT+RAG** baseline in this repo uses the **same** subject-anchored retrieval recipe and surprise-gated admission as the fp stack; on **clean** static recall it **ties or wins** on score (§4.2). Swapping the embedder while keeping **opaque text chunks** and **re-feeding strings** to the LM would not be a new architecture—it would be packaging.

**What TapeLM is.** Knowledge is represented as **structured operations in one fp geometry**: episodic keys tied to **subjects and context**, **bind/unbind** for relational edges, **multi-hop** chains as cosine or circular-convolution steps **without** decoding intermediate entities to text, **one-shot writes** and **230 resolution** when the store holds conflicting structured values. Generation, calibration (“in *my* lexicon?”), and memory keys all read the **same** `arc_enc`—**co-location**, not a bolt-on retriever.

GPT+RAG can match **point retrieval** by bolting an encoder, index, and (optionally) the **same** fp-surprise admission policy. TapeLM’s defensible headline is **operable knowledge structure** in unified space: the encoder that generates is the encoder that fingerprints, calibrates, keys, binds, and resolves—**no second representation learning step** for memory at inference, and **APIs** (hop, bind, edit, resolve) that a text index does not expose as first-class primitives—even when benchmark accuracy is parity.

### 5.2 When vanilla GPT is structurally weak

Parametric LMs cannot ingest a write without gradients; in-context windows fail on long streams; BPE-native OOD signals conflate rarity and fakeness on our probe. These are the axes where TapeLM shows **capability** gaps, not just packaging.

### 5.3 Continual factual knowledge without catastrophic forgetting

**Catastrophic forgetting (CF)** here means new training overwrites weights and destroys old behavior (e.g. Stage 202b: PAWS fine-tune dropped next-token on the copy from ~0.83 to ~0.55). **Multi-hop retrieval is not an anti-CF mechanism by itself**—hops compose already stored facts. The anti-CF contract is: **new knowledge in slots, skills in frozen weights**.

| Mechanism | Role vs CF |
|-----------|------------|
| Frozen product encoder (P1) | Generation and fp geometry fixed; adding facts does not run gradients on the backbone. |
| Episodic slots + one-shot edit (194, 197) | New/updated facts = **write**, not fine-tune → no parametric overwrite for factual updates. |
| Surprise-gated admission (197–198) | Under budget, novel events are retained preferentially (0.98 vs 0.20 for ingestion order) → less memory pollution and less “silent overwrite” of useful slots by boilerplate. |
| Recency on updates (198) | Changing facts update via newer slots, not by retraining the LM. |
| Hop2 / binding (195, 200) | Multi-step answers without stuffing relation graphs into weights. |

Stage 205 (§4.2c) measures the other direction of the same contract: *removing* knowledge. Slot deletion leaves retained facts and generation bit-identical, whereas gradient unlearning of the same 20 facts in the parametric model destroys 80% of the retained facts and degrades next-token accuracy — the CF failure mode, now quantified on the removal side rather than the addition side.

This is **continual factual learning at inference** (same class of idea as external memory / RAG for facts), not continual **training** of the LM. Hops help **use** stored facts compositionally; they do not protect weights if the encoder is fine-tuned again. For production TapeLM we keep P1 frozen and treat CF as a **memory-policy** problem (budget, surprise, recency, key collisions), not a weight regularization problem. Full GPT+RAG shares the “facts outside weights” benefit; our difference remains unified fp-space and write policy, not exclusive anti-CF.

### 5.4 Internalizing hops inside the model (theoretical)

Today, hops run **outside** the forward pass: build an fp slot bank at read time, then iterate retrieve → next entity (or bind/unbind in fp-space). **Theoretically, the same operations can be moved inward**—several designs are compatible with the existing dual-channel curve stack:

1. **Slow channel as an internal slot tape (extend 180/187).** On surprise-gated writes, store not only slow state summaries but **explicit keys** (context fp, subject fp) and **values** (entity fp or bound edge `norm(fp_A ⊙ fp_B)`). A **query head** inside the block attends over slow slots each forward step (Stage 185 tape-read was redundant *in-window*; beyond-window or cross-document hops are the motivating case). Training signal: multi-hop cloze or next-entity losses on top of CE, with gradients only where needed—or keep slow writes inference-only to preserve anti-CF.

2. **Binding layers in the fast path.** Fixed or learned **circular convolution / element-wise bind** (HRR-style) between token or arc fps inside selected layers, with unbind for query. Hops become **fixed-depth latent programs** (2–3 bind-unbind steps) rather than external cosine loops. Stage 195 shows bind/unbind is viable at d256; internalizing it means the transformer **learns when to bind**, not only an external script.

3. **Iterative retrieval in latent space (ACT / recurrent read).** The model halts after k internal steps: state_t → soft retrieve over tape → state_{t+1}. External hop loop becomes **unrolled layers** or a small controller. Requires training on hop-heavy data or auxiliary losses; otherwise the loop stays unused (as with unused tape-read in 185/186).

4. **Memory tokens (MemLLM / RMT-style).** Dedicated slow tokens updated each layer; subject-anchor writes map to updating a memory token instead of an external index. Semantically similar to (1) but implemented as extra positions in the fast transformer.

**What we already have vs what is missing:** surprise-gated slow writer, fp lexicon, and arc encoder are **primitives** for (1)–(2). Missing pieces: (a) **trained query/read head** that is used at inference for answers, not only CE next-token; (b) **beyond-window** or cross-session training regime so internal tape beats endpoint; (c) explicit **bind op** in the graph with hop-supervised data. **Risk:** if internal hops are fully parametric and co-trained with CE on everything, factual updates reintroduce CF unless writes stay non-gradient (slow buffer) or adapters are isolated.

**Honest expectation:** internalizing hops is **feasible as architecture** (slow tape + bind + query is the natural merge of 180 + 192–195) but is **not proven** in this repo—we only validated external zero-train hops. Internal version trades zero-train simplicity for end-to-end learnable composition; anti-CF is preserved only if **writes to slow/binding buffers remain non-parametric or surprise-gated with frozen fast channel**, matching the product split we already use (frozen P1 + external slots).

**Empirical probe (Stage 203).** We tested a first internal version: a differentiable *k*-hop reader trained over the **frozen** fp encoder and a **non-gradient** slot tape (so anti-CF holds by construction). Two variants on planted novel-entity chains (train/test split on chains; 4-way, chance 0.25):

| internal reader | train k2/k3 | **test k2/k3** | generalizes? |
|-----------------|-------------|----------------|--------------|
| free-form (k-embed + per-step MLP update + output head) | 1.00 / 0.99 | **0.15 / 0.20** | no — memorizes seen chains |
| soft-follow (parameter-free retrieve→follow→depth-select, only τ learned) | 1.00 / 0.99 | **1.00 / 1.00** | yes — matches external loop |
| external hand-loop (zero-param) | — | 1.00 / 1.00 | (reference) |

The finding is a **cautionary design result**: hops *can* live inside a forward pass and stay anti-CF, but only when the module retains the minimal inductive bias of the parameter-free operation (sharp retrieval, value-following, depth selection). A free-form learnable reader collapses to rote memorization and fails to generalize to unseen chains, while the zero-parameter external loop already generalizes perfectly. So internal hops buy a single differentiable stack (engineering), **not** a new capability — and a production version must be a *structured* soft-follow / bind block, not a free MLP.

**Three further internalization attempts, all negative (Stages 210–212).** After Stage 203 we tested whether the tape can do anything an external index cannot, in three independent ways, each on the frozen P1 encoder with generation verified unchanged (Δ next-token = 0, logits bit-identical when the added path is disabled).

| Stage | Attempt | Result |
|-------|---------|--------|
| 210 | Retrieved fingerprint injected as one memory-arc, answer read **as BPE tokens** through the CE head | Test accuracy at chance (k2 0.25, k3 0.30) while the external cosine loop scores 1.00; the retrieve→span mapping is not learnable without gradients on the backbone |
| 211 | Surprise-gated slow states logged as an addressable internal tape, queried cross-document | Internal tape 0.23 versus slow-endpoint 0.28 (both near chance) with explicit fingerprint slots at 1.00; a document-id key trivially reaches 0.99, so the internal tape is neither better than the endpoint nor substrate-specific |
| 212 | Read-only contrastive channel for **occurrence identity** (four siblings of one surface form, disjoint store/query halves) | 0.378 versus 0.336 for an *untrained* projection of the same state, 0.328 for the Stage 197 context blend, 0.242 for the blind surface key; paraphrase similarity 0.750 *below* minimal-pair 0.937 |

The three failures are informative in different directions: composition is sound only as an **external, zero-training fingerprint loop**; the slow endpoint is **not** an addressable memory; and occurrence identity is **not recoverable** from the frozen state, which also closes the older collision debt from the pre-curve fingerprint track. Together with §4.5 and §4.6 they delimit the design claim through Stage 212: the curve belongs on the **input and memory** side, and **token-internal** composition did not generalize.

**Post-212 product memory (§4.8).** We did **not** abandon external memory—we **operationalized** it: canonical banks, family W, fp decode, and resolution policy on the **same** frozen encoder. Internalization (210–212) remains closed; the shipping path is **external fp loops + policies**, now with measured cross-domain utilization (226c).

### 5.4b Where the substrate can genuinely win (essence roadmap)

Separating essence-level wins (require the char-curve + fp substrate; BPE cannot replicate without becoming char-level) from packaging wins (a bolted GPT+RAG can match):

- **W1 — robustness to noise / OOV / typos / code / multilingual — CONFIRMED (§4.2b, Stage 204):** BPE fragments rare or misspelled tokens into brittle pieces (3.93 → 4.76 pieces/word under noise); the char-curve encoder degrades smoothly. Measured: identity retrieval at 30% noise 0.801 vs 0.403, hardened noisy-corpus recall 0.913 vs fair GPT+RAG 0.627 — the program's first capability-level win over the strongest rival.
- **W2 — native calibrated abstention (demonstrated):** fp-lexicon "don't-know" 0.982 vs GPT BPE-surprisal 0.380; flows directly from the char-fp substrate.
- **W3 — editable / auditable / *unlearnable* knowledge — CONFIRMED (§4.2c, Stage 205):** slot deletion erases a fact in O(1) with retained recall and next-token unchanged, provenance 1.00, contradiction detection 1.00 at 0.00 false positives, while naive parametric unlearning in GPT costs 80% of retained facts and 0.075 next-token. Capability win over parametric knowledge; architectural against a RAG index.
- **W4 — knowledge decoupled from parameters / anti-CF (demonstrated, §5.3):** N facts = N slots on one frozen encoder, no gradient.
- **W5 — hops as O(1) vector ops vs O(k) decode/chain-of-thought steps — bounded (Stage 206):** at depth 6 latent hops cost 2.35 ms and one encoder call versus 37.57 ms and seven calls for a text-valued index (16×), with accuracy tied at 1.000 on clean chains, so this is a **latency** advantage rather than a capability one; a vector-valued index recovers most of it. The substrate-specific part is that quantize-to-lexicon re-anchoring corrects hop errors under noise for the curve (0.300 → 0.350 at k=6, +0.10 at k=2) but not for BPE-RAG (0.292 → 0.300), because re-anchoring only helps a substrate that can re-identify a corrupted word.

Honest bottom line: the defensible **essence** wins are **W1 (now measured) + W2**, both rooted in the substrate BPE cannot copy; W3/W4 are strong but partly shareable with RAG; W5 is a research bet.

### 5.5 Limitations

- Model and data scale small vs production LMs.
- Semantic understanding (adversarial paraphrase / hard minimal pairs) not solved.
- Static recall on clean text can lose to a well-tuned RAG index; the robustness win (§4.2b) is specific to noisy/OOV input and is measured against a matched d256 BPE control, and a byte/char-level LM would narrow it.
- Noise in §4.2b is synthetic uniform character noise; real typo and OCR distributions differ.
- Fine-tuning the encoder for semantics destroys generation on the copy (202b: next-token ~0.55)—production stack stays frozen.
- Hop benchmarks on unique anchors understate RAG compounding error; harder hop stress-tests remain future work.
- Multi-hop error compounds for every substrate once input is noisy: by depth 4–6 at 15% character noise all routes approach chance (§4.3/Stage 206), so the noise-robustness advantage is a shallow-depth phenomenon.
- The latency advantage of latent hops is partly a model-size artifact (a small character arc-encoder versus a six-layer transformer); only the O(1)-versus-O(k) encoder-call structure is architectural.

### 5.6 Relation to prior SOTE

Hop2/binding failures at d64 are **resolved at d256** with a learned fp space (195). The old FP theory supplies **memory semantics**; the curve supplies **learned geometry**.

---

## 6. Conclusion

TapeLM demonstrates that a **single frozen curve encoder** can host a **zero-train memory and calibration stack** with GPT parity on generation and clear advantages over vanilla GPT on memory, edit, and calibration. On clean retrieval it remains **architecturally** distinct from GPT+RAG rather than dominant on scores—but under spelling noise and out-of-vocabulary input the tie breaks (§4.2b: 0.913 vs 0.627), and targeted unlearning is O(1) and collateral-free where parametric unlearning is destructive (§4.2c). **§4.8** adds a **production memory trunk**: canonical slots, migratable **W_family**, fp decode, and contradiction resolution—without unfreezing P1. Semantic invariance at PAWS-level difficulty requires **larger encoders**, shared with standard transformers at our scale.

The boundaries through Stage 212 remain results, not caveats: token-internal hops (210–212), fingerprint-generative output (207), hybrid rerank (208). **Utilization** without fp decode is still bounded (226); the recommended path is **4-way retrieve → fp-scorer** (228c). We recommend publishing as a **technical report / systems paper** emphasizing unified fp-memory (192–205 **and** 221–230), noise/unlearning wins, and honest RAG comparisons—not SOTA on clean retrieval or semantic understanding.

---

## Appendix A — Stage index (selected)

| Stage | Result |
|-------|--------|
| 191 | Night scale; parity 0.867 vs 0.843 |
| 192–193 | FP lexicon; wired calibration |
| 194 | Fact memory 0.947 |
| 195 | Hop2 0.70; binding 0.52 |
| 196–198 | TapeLM assemble; anti-clone gates |
| 197 | Edit 1.00; surprise write 0.98 vs 0.20 |
| 200–201 | Composition vs RAG; hard-neg B fail |
| 202–202b | PAWS B-track; parity; no inversion |
| 203 | Internal hops: generalize only if structured (soft-follow 1.00 test vs free-form 0.20) |
| 204 | Noise/OOV robustness: 0.913 vs fair RAG 0.627; identity 2× — capability win |
| 205 | Unlearn O(1) collateral-free; provenance 1.00; conflict audit 1.00/0.00 |
| 206 | Latent hops 16× cheaper (1 vs 7 encoder calls); accuracy tied — latency only |
| 207 | Fingerprint-generative variant B falsified (0.536 vs CE 0.861; free-run off-manifold) |
| 208 | Hybrid rare-word fp reranker: no measurable gain over the token head |
| 209 | Semantic scaling + MiniLM teacher: STRUCTURAL_BLOCK_NO — curve tracks GPT; B not confirmed on 3050 |
| 210 | Hops inside forward answering in tokens: chance versus 1.00 for the external loop |
| 211 | Addressable slow tape cross-document: 0.23 versus 0.28 endpoint, 1.00 external slots |
| 212 | Occurrence-identity channel: 0.378 versus 0.336 untrained; paraphrase below minimal pairs |
| **221** | **W-remap** after encoder shift: align ~0.997; recall ~0.78 vs oracle ~0.87 |
| 222–224 | Deploy modes / cross-W / far-shift: family registry justified |
| **225** | **Domain bundle:** frozen `arc_enc`, multi-head, reuse W_prose |
| **227** | **Canonical slots + qmap read:** cross-code ~0.95 |
| 228b | Global argmax decode fails; protocol mismatch |
| **228c** | **4-way retrieve + fp decode:** 1.0 vs head ~0.48 |
| **229–230** | Multi-hit slots; **resolution policy** ~1.0 vs argmax ~0.47 |
| **226c** | Cross-domain **fp decode** ~0.88 vs head ~0.45 |

Full narrative: [`plan_curve_dynamics.md`](plan_curve_dynamics.md) (Memory extension program), [`extension_memory_contract.md`](extension_memory_contract.md), [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md). Machine-readable: `results/stage*_decision.json`, `artifact/decisions/`.

---

## Appendix B — Suggested arXiv framing

**Title (short):** *TapeLM: Zero-Train Fingerprint Memory on a Unified Curve Encoder*

**One-sentence pitch:** We attach editable lexicon, episodic, and compositional fingerprint memory to a frozen curve LM, match GPT on generation, beat vanilla GPT on memory and calibration, tie fair GPT+RAG on clean retrieval, and extend the same encoder with **canonical memory + W + fp decode** for domain shift and cross-domain use.

**Avoid in abstract:** “understands language,” “beats RAG,” “new paradigm without tokens.” In particular do **not** claim open-vocabulary superiority over BPE (see §4.5 scope correction) or capability distinctness on clean retrieval.

**Include:** scorecard table, unread-memory control, surprise-gated write ablation, PAWS negative result with GPT matched control, and the four delimiting negatives (internal-hop inductive bias, latency-only composition, falsified fingerprint generation, no-gain hybrid) — they are what make the positive claims credible.

---

## Appendix C — Publication checklist

- [ ] Author list & affiliation
- [ ] Fix HF dataset import order note for reproducibility (202 Windows segfault)
- [x] Single `TapeLM` demo script (`artifact/scripts/run_product.py`, `run_memory_demo.py`)
- [x] Figure: unified fp-space diagram — `docs/ARCHITECTURE.md` (inference path + frozen-P1 table)
- [ ] Figure: accuracy vs character-noise rate, curve vs fair RAG (§4.2b) — the headline capability result
- [ ] Figure: unlearning collateral, slot delete vs gradient ascent (§4.2c)
- [ ] Optional: 1-page hop stress-test (see main text recommendation on hop3+)
- [ ] Compare against a byte/char-level LM (ByT5-style) baseline to bound the §4.2b claim properly
- [ ] Compare against a localized parametric editor (ROME/MEMIT) to bound the §4.2c unlearning claim
