# Verdict vocabulary (TapeLM / Inprint)

**Problem:** bare `*_NO` and `*_PARTIAL` often misread closed branches. Prefer labels that say **what was learned**, **at-scale vs method**, or **superseded by another stage**.

---

## Internalization frontier (210–212)

| Verdict | Meaning |
|---------|---------|
| `THESIS_YES` | Property demonstrated under stage gates; not reducible to external fp loop without changing the claim |
| `ENGINEERING_ONLY` | In-forward path works but matches external zero-train loop on the metric |
| **`THESIS_NO_AT_SCALE`** | **Default read for 210–212.** Thesis gates failed on frozen P1 @ d256/6L (and stated train budget). Does **not** falsify the property at larger encoder, stronger pretrain, or with GPT-matched controls not yet run |
| `THESIS_NO` | **Legacy alias only** — same as `THESIS_NO_AT_SCALE`; do not use in new docs or runs |

**Required in decision JSON (210–212):**

- `claim_scope` → [`results/internalization_210_212_claim_scope.json`](../results/internalization_210_212_claim_scope.json)
- `gpt_matched_ladder` → parity row vs matched GPT on the **same task harness** where applicable

---

## Invalid method (217, 218, 220)

| Verdict | Meaning |
|---------|---------|
| **`SLOW_ENDPOINT_INVALID_METHOD`** (217) | Dual ctx+endpoint keys **break** noisy recall vs lex-only — design does not isolate “slow endpoint helps” |
| **`SNAP_HOP_INVALID_METHOD`** (218) | Snap-to-lexicon hop probe **does not** validly test external fp hop API |
| **`SEM_SIDECAR_INVALID_METHOD`** (220) | PAWS sidecar vs **lexical overlap** baseline is not a fair semantic test (see 202/209/258) |

**Do not read as `*_NO` (“hypothesis rejected forever”).** These are **closed design branches**, not capability scorecard rows.

Legacy JSON may still show `SLOW_ENDPOINT_NO` / `SNAP_HOP_NO` / `SEM_SIDECAR_NO` — treat as **`_INVALID_METHOD`** in indexes.

---

## Anti-CF rehearsal price (242)

| Verdict | Meaning |
|---------|---------|
| **`REHEARSAL_DOSE_ANTICF_OK`** | **Primary read.** Tape retains (**1.0**); GPT **never** reaches match gate even at **100%** A-replay in B CE (e.g. **0.938 @ 1.0** vs target 0.95). **Result**, not a failed experiment |
| `REHEARSAL_DOSE_OK` | GPT reached target at some rehearsal rate `< 1` |
| `REHEARSAL_DOSE_PARTIAL` | **Legacy bookkeeping** — `G_found_dose` false while grid incomplete; **do not use** when curve includes **1.0** |
| `REHEARSAL_DOSE_NO` | Tape or zero-rehearsal controls failed |

Gate `G_found_dose` (≥0.95) is optional strictness; the **substantive claim** is the **full dose curve** vs slot write (259).

---

## Domain curriculum (246 vs 254)

| Verdict | Meaning |
|---------|---------|
| **`DOMAIN_CURRICULUM_DUP254`** | **Read-as label.** Same story as **254 joint upper** (multi-domain CE on shared weights): tape mem vs GPT wiki collapse. **Do not re-litigate** — see [`stages_254_close.md`](stages_254_close.md). JSON may still say `DOMAIN_CURRICULUM_PARTIAL` |
| `CONTINUAL_UNDERSTAND_OK` / `_NO` | **254** operators-only vs joint — canonical continual split |

---

## Semantic scaling (209)

| Verdict | Meaning |
|---------|---------|
| `STRUCTURAL_BLOCK_NO` | Curve PAWS **tracks matched GPT** at each scale; refutes “A is structurally blind”; does **not** confirm Goal B @ 3050 |

---

## Semantic query (258, 261)

| Verdict | Meaning |
|---------|---------|
| `SEM_QUERY_OK` / `SEM_QUERY_PARTIAL` | Valid exam; sem channel beats fp on paraphrase |
| `SEM_QUERY_NO_AT_SCALE` | Matched GPT-2 also fails — **scale**, not architecture |
| `SEM_QUERY_NO` | GPT succeeds where curve does not |
| `SEM_QUERY_INVALID` | `G_fp_only_at_chance` failed or keys drifted |
| `NL_QUERY_NO_AT_SCALE` | Open-bank NL; GPT parity on top1 (261) |
| `NO_AT_TRUNK_SCALE` | 266: matched 0.5B-Instruct does not beat 0.5B base on 261 exam |
| `INSTRUCT_TRUNK_OK` | 266: Instruct (+ladder/prompted) moves open-domain signal |
| `INSTRUCT_TRUNK_INVALID` | 266: matched 0.5B pair incomplete (load fail) — not a substantive NO |
| `MIXER_OVERFIT` | 266: fp+sem fit top1≈1, eval≈0 — mixer memorizes h→slot; remap 261 = mixer defect |
| `HARVEST_FIXES_MIXER` | 266: bank-wide (prefix→slot) training closes fit/eval gap — defect was data, not architecture |
| `PROMPTED_QUERY_SIGNAL` | 266: keyword→votes beats W_sem; interface still broken |
| `WORDS_FORMULATE_QUERY` | 266: words (surface and/or Instruct keywords→votes) crush learned query vector; remap `QUERY_MUST_BE_WORDS` |
| `MIND_REFINES_QUERY` | 266: surface ∪ keywords beats surface on headlines while keeping coverage — mind adds precision without silence tax |
| `PARAPHRASE_BREAKS_SILENCE` | 266: novel paraphrase words on tape reduce `tie_at_zero_frac` (the only mind-chance metric) |
| `MIND_LEARNS_TAPE_OK` | 268: upper learns with arc_enc frozen; novel-tape EM holds (≥ train−0.05) and beats frozen-upper control |
| `MIND_LEARNS_TAPE_PARTIAL` | 268: novel tape transfers but not all sanitation gates |
| `MIND_LEARNS_TAPE_NO` | 268: mind memorized banks / failed novel tape or arc_enc moved |
| `QUERY_MUST_BE_WORDS` | remap of 261: formulate the query in words, not a learned projection |
| `NL_QUERY_NWAY_FP_ONLY` | 20-way signal on fp; sem blend **hurts** vs trained fp-only (261 ablations) |
| `NL_QUERY_MIXER_OK` | Sem neutral (≥90% fp 20-way) with causal shuffle band (261) |
| `NL_QUERY_NWAY_ONLY` | Sem ≥ fp without harm — **requires** causal 20-way + fp ≥ 0.12 (261) |
| `NL_QUERY_*` | See [`stage261_close.md`](../results/stage261_close.md) and [`stage261_mixer_ablation.md`](../results/stage261_mixer_ablation.md) |
| `WORD_VOTES_OK` | Zero-train word postings; causal top1 + beats popularity 20-way + open top1 + low-overlap (261f) |
| `WORD_VOTES_BEATS_MEAN` | Same, but open top1 below gate — still beats 261 ctx_fp mean and popularity floor (261f) |
| `WORD_VOTES_SIGNAL_ONLY` | Causal + 20-way signal band only (261f) |
| `WORD_VOTES_NO` | Fails gates (261f) |
| `VOTES_BEAT_MEAN` | 263: word postings beat ctx_fp mean on 256 exam (ΔEM ≥ 0.10, causal) |
| `VOTES_TIE_MEAN` | 263: |ΔEM| < 0.10 — mean ceiling likely open-bank only |
| `MEAN_BEATS_VOTES` | 263: cosine/mean retrieval wins on 256 exam |
| `COSINE_BASELINE_INVALID` | 263: cosine EM &lt; 0.50 — comparison not readable |
| `BLEND_BEATS_SINGLE` | 264: cascade or fusion beats votes alone on open bank |
| `IDF_MEAN_FIXES_FLAT` | 264: idf-weighted mean ≫ flat mean (≥+0.10 20-way) |
| `VOTES_BEST_OPEN_BANK` | 264: votes still best among modes |
| `NO_CLEAR_WINNER` | 264: no blend/idf clear win |

---

## Trunk swap (262)

| Verdict | Meaning |
|---------|---------|
| `TRUNK_SWAP_OK` | External trunk **unseen_para** ≥ curve − 0.05 and above chance — **W_sem interface transfers** |
| `TRUNK_SWAP_BETTER` | External **beats** curve by ≥0.10 on unseen paraphrase — understanding was the bottleneck |
| `TRUNK_SWAP_PARTIAL` | External works but **clearly worse** than curve |
| `TRUNK_SWAP_NO` | External fails — channel **tuned to curve states** ( sinks “any reasoner” claim ) |
| `TRUNK_SWAP_INVALID` | fp-only not at chance — exam leaked |

Same exam as **258** (`s258` helpers, seed **258**); keys stay P1 fp; external tokenizer never touches tape.

---

## Stream ingest (255)

| Field | Meaning |
|-------|---------|
| **`W_q` (QueryAdapter)** | **Trainable query map only** — canonical slot **keys** frozen. Fits on **ingested-entity** contrastive pairs per chunk; **recall gates** use **held-out** probe facts (not wq_train). Hold CE vs **P1** is the primary no-forget gate. Ablation: `--no-query-train` |

---

## Product / falsified (207, 216, …)

| Pattern | Example |
|---------|---------|
| `*_NO` / `Falsified` | Specific mechanism rejected **in the form tested** (207 variant B) |
| `*_PARTIAL` | Some gates pass — check stage note before headline |

When in doubt: prefer **`NOT_CONFIRMED_AT_SCALE`**, **`NO_AT_SCALE`**, **`INVALID_METHOD`**, or **`ANTICF_OK`** over bare **`NO`** / misleading **`PARTIAL`**.
