# Stage index (TapeLM: 170–245)

> **Shipping trunk (product):** **221 → 227 → 228c → 230 → 226c**

Active scripts: **repo root** `_stage*.py`. Legacy pre-170: [`legacy/sote/`](../legacy/sote/).

**Product headline stages:** **191–205** (core fp + 204/205 wins) and **221–230** (memory trunk — canonical, W, decode, resolve; demo in `run_product.py`). Extension **213–220** maps what *not* to ship (e.g. **215** → use **221**).

**Context:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`MEMORY_ENGINEERING.md`](MEMORY_ENGINEERING.md) · [`EXTENSION_PIPELINE.md`](EXTENSION_PIPELINE.md) · [`../artifact/OVERVIEW.md`](../artifact/OVERVIEW.md)

Verdicts: `results/stage*_decision.json` and [`../artifact/decisions/`](../artifact/decisions/) · `python artifact/scripts/show_map.py`

---

## Core product (191–205)

| Stage | Script | Verdict |
|-------|--------|---------|
| 191 | `_stage191_night.py` | P1 parity encoder |
| 192–193 | fp lexicon + wired | Calibration |
| 194–195 | fact memory + hop2 | Recall / binding |
| 196 | `_stage196_tapelm.py` | **TapeLM assemble** |
| 197–198 | edit + stream | Edit / stream |
| 203 | internal hops (structured) | **INTERNAL_HOPS_YES_IF_STRUCTURED** |
| 204–205 | noise + unlearn | Capability vs fair RAG |
| 207 / 207-MAX | curve thinking | **B falsified** |
| 208 | hybrid fp rerank | No gain |
| 209 | semantic scaling | **STRUCTURAL_BLOCK_NO** (B not confirmed @3050) |
| 210–212 | internalization frontier | All **`THESIS_NO_AT_SCALE`** @ d256 (see [`VERDICT_VOCABULARY.md`](VERDICT_VOCABULARY.md)) |

Earlier substrate: 170–191 (curve, BPE, dual-channel, night scale) — see [`plan_curve_dynamics.md`](../results/plan_curve_dynamics.md).

---

## Memory extension (213–230)

**Strongest product line:** **221–230** (+ **226c** e2e). Same **frozen P1**; operational layer for domain shift, canonical bank, decode, conflicts.

| Stage | Script | Verdict |
|-------|--------|---------|
| 213 | `_stage213_arc_enc_freeze_finetune.py` | **ARC_ENC_FREEZE_PARTIAL** |
| 214 | `_stage214_recency_ctx.py` | **RECENCY_CTX_NO** |
| 215 | `_stage215_domain_adapter.py` | **DOMAIN_ADAPTER_NO** |
| 216 | `_stage216_split_arc_ff.py` | **SPLIT_FF_NO** |
| 217–219 | slow endpoint / snap / stream | **217/218:** **`_INVALID_METHOD`**; 219 see JSON |
| 220 | `_stage220_sem_sidecar.py` | **`SEM_SIDECAR_INVALID_METHOD`** |
| 221 | `_stage221_fp_remap_adapter.py` | **FP_REMAP_ADAPTER_YES** |
| 221-probe | `_stage221_probe.py` | **W_REMAP_CHARACTERIZED** |
| 222 | `_stage222_fp_deploy_modes.py` | **FP_DEPLOY_MODES_MIXED** |
| 223 | `_stage223_cross_adapter.py` | **DOMAIN_W_SWITCH_PARTIAL** |
| 224 | `_stage224_far_shift.py` | **W_DOMAIN_PARTIAL** |
| 225 | `_stage225_family_fork.py` | **DOMAIN_BUNDLE_OK** |
| 226 | `_stage226_joint_gen_mem.py` | **JOINT_GEN_MEM_NO** |
| 226b | `_stage226b_joint_diag.py` | **RETRIEVAL_OK_UTIL_BOUNDARY** |
| **226c** | `_stage226c_joint_fp_decode.py` | **JOINT_FP_DECODE_OK** |
| **227** | `_stage227_canonical_slots.py` | **CANONICAL_STORAGE_OK** |
| 228a | `_stage228a_counterfactual_inject.py` | **HEAD_INJECT_PARTIAL** |
| 228b | `_stage228b_fp_guided_decode.py` | **FP_GUIDED_DECODE_NO** |
| **228c** | `_stage228c_fp_decode_fix.py` | **FP_DECODE_FIX_YES** |
| 229 | `_stage229_contradiction_slots.py` | **CONTRADICTION_RAW_MEMORY_OK** |
| **230** | `_stage230_slot_resolution.py` | **RESOLUTION_POLICY_OK** |

**Shipping trunk (221 → 227 → 228c → 230 → 226c):** W (**221**) → canonical + qmap (**227**) → 4-way retrieve + fp decode (**228c**; **229** multi-hit) → resolution (**230**) → cross-domain e2e (**226c**). API: `_tapelm_ext.py` · demo: `artifact/scripts/run_product.py`.

Narrative: [`extension_memory_contract.md`](../results/extension_memory_contract.md) · closed branches: [`extension_closed_branches.md`](../results/extension_closed_branches.md).

---

## Post-trunk engineering (231–239)

| Stage | Script | Verdict (full) | Value |
|-------|--------|----------------|-------|
| **231** | `_stage231_temporal_W.py` | **TEMPORAL_W_OK** | Matched qmap beats wrong-era W |
| **232** | `_stage232_stream_w_version.py` | **STREAM_W_VERSION_OK** | L3 decay × `w_version` |
| **233** | `_stage233_tool_binding.py` | **TOOL_BINDING_OK** | tool⊙entity keys |
| **234–236** | compositional W | **OK** | Chained ≈ direct |
| **237–238** | mixed L1 | PARTIAL / **MIXED_SCRATCH_OK** | Future encoder family evidence |
| **239** | `_stage239_cf_sequential.py` | **CF_SEQUENTIAL_OK** | A→B vs parametric GPT |

---

## Unexpected comparisons (240–245) — not shipping trunk

| Stage | Script | Verdict | Surprise? |
|-------|--------|---------|-----------|
| **240** | `_stage240_cf_vs_rag.py` | **CF_VS_RAG_SURPRISE** | Yes — frozen GPT+RAG index **breaks** under query drift (1.0→0.68); TapeLM 0.95 |
| **241** | `_stage241_harmful_W.py` | **WRONG_W_HURTS_OK** | Wrong W < no-W (deploy: better bare than wrong family) |
| **242** | `_stage242_rehearsal_dose.py` | **`REHEARSAL_DOSE_ANTICF_OK`** | 100% A-replay → GPT **0.938**, tape **1.0**; vs slot write (259). [`VERDICT_VOCABULARY.md`](VERDICT_VOCABULARY.md) |
| **243** | `_stage243_carrier_drift.py` | **CARRIER_DRIFT_OK** | Same B corpus: slots 0.98 vs weights 0.45 |
| **244** | `_stage244_forget_clean.py` | **FORGET_CLEAN_OK** | Slot delete zero collateral; GPT unlearn damages retained |
| **245** | `_stage245_mixed_vs_p1W.py` | **MIXED_NO_W_TIES_P1W** | Mixed raw ≈ P1+W (not a swap signal) |
| **246** | `_stage246_domain_curriculum.py` | **`DOMAIN_CURRICULUM_DUP254`** *(JSON: PARTIAL)* | **Duplicate of 254 joint** — see [`stages_254_close.md`](stages_254_close.md); product line = 254 operators + **255** |
| **247** | `_stage247_ingest_forks.py` | **INGEST_FORK_SLOTS_AND_HOP** | Fork map: slots+masked CE beat CE-on-bindings; hop-sim admits/rejects |
| **248–250** | masked CE nights | PARTIAL / **MASKED_NIGHT_OK** (mem) | **Do not read as “understanding failed”:** 248–250 used `" ".join` stream → often **one wiki doc** + repeated mask stub; exam nt drop likely **overfit**, not objective verdict |
| **251** | `_stage251_cpc_understand.py` | **CPC_UNDERSTAND_PARTIAL** @4M tok/phase | CAL moves exam **0.825→0.850**; CPC gap **0.185→0.096** but exam **0.717**, holdout PPL worse; mem/leak OK |
| **252** | `_stage252_joint_cpc.py` | **JOINT_CPC_OK** @4M/arm | Winners **λ=0.05, 0.2**; **λ=0.2**: exam **0.850**, gap **0.185→0.137** |
| **253** | `_stage253_scale_joint.py` | **SCALE_JOINT_OK** @16M | **λ=0.2**: nt **0.825→0.867**, hold **4.00**, gap **0.185→0.129**; all gates vs 252; mem 1.0 (~2h) |
| **254** | `_stage254_continual_understand.py` | **CONTINUAL_UNDERSTAND_OK** (smoke, `--operators-only`); joint smoke **NO** | Frozen P1 + W_query + growing bank; joint upper forgets/leaks. [`stages_254_close.md`](stages_254_close.md). |
| **255** | `_stage255_stream_ingest.py` | **STREAM_INGEST_OK** (wiki:12) | Bounded tape; **W_q** trains on ingested pairs, recall on held-out probes — see [`VERDICT_VOCABULARY.md`](VERDICT_VOCABULARY.md) |
| **256** | `_stage256_slot_bias_decode.py` | **SLOT_BIAS_GLUE_OK** | One-hop copy-mixture glue; symmetric W_q; bank NCE on wiki pairs. |
| **257** | `_stage257_fp_compose.py` | **FP_COMPOSE_OK** (full) | Two-hop; retrieval@cue + span-lock. [`stages_257_258_close.md`](stages_257_258_close.md) · [`../results/stages_255_260_close.md`](../results/stages_255_260_close.md). |
| **258** | `_stage258_semantic_query.py` | **SEM_QUERY_OK** (full) | W_sem; para_hold; **0.646 vs GPT 0.276** unseen. Same close docs. |
| **259** | `_stage259_hot_swap.py` | **HOT_SWAP_OK** (full) | Edit slot, zero grad; `TapeView.with_value`. |
| **260f** | `_stage260f_open_gate.py` | **OPEN_GATE6_OK** (full) | **feat_only** gate; h_only **0.0** paired win. 260c–e ablation line. |
| **260** | `_stage260_open_gate.py` | **OPEN_GATE_NO** (smoke) | Gate ignored tape (shuffled AUC = real); see **260b**. |
| **260b** | `_stage260b_open_gate.py` | **OPEN_GATE2_NO** (smoke) | Line classifier; gate_reads_tape false → **260c** paired train. |
| **260c** | `_stage260c_open_gate.py` | **OPEN_GATE3_NO** (smoke) | Paired same-line; **`feature_probe`** + **`features_move`**; read before 212b. |
| **261** | `_stage261_nl_query.py`, **`261f`** `_stage261f_word_votes.py` | **`NL_QUERY_NO_AT_SCALE`** (train path); line **`WORD_VOTES_BEATS_MEAN`** (261f) | Open bank NL; blend harm; **261f** zero-train word votes — [`stage261_close.md`](../results/stage261_close.md) |
| **262** | `_stage262_trunk_swap.py` | *(model-dependent)* | **258 exam** with external frozen trunk via `h_t` only; [`VERDICT_VOCABULARY.md`](VERDICT_VOCABULARY.md) |
| **263** | `_stage263_votes_vs_mean.py` | `VOTES_*` | **256 exam**, retrieval only: cosine ctx_fp mean vs word votes; [`stage263_decision.json`](../results/stage263_decision.json) |
| **264** | `_stage264_retrieval_blend.py` | **`VOTES_BEST_OPEN_BANK`** | Open bank: mean/idf/votes/cascade/fusion; blends lose on top1; silence→route — [`stage264_decision.json`](../results/stage264_decision.json) |
| **265** | `_stage265_span_lock.py` | `SPAN_LOCK_*` | Gate opens once; tape emits value verbatim (A soft/soft, B soft/locked, C open/locked). Smoke: `--smoke --no-arm-c` |
| **266** | `_stage266_instruct_trunk.py` | **`WORDS_FORMULATE_QUERY`** / remap **`QUERY_MUST_BE_WORDS`** | words ≫ W; any words suffice; paraphrase∪surface silence 0.449→0.426 (not enough) — mind doesn't break silence |
| **267** | `_stage267_read_refine.py` | *(smoke)* | hop0→read passages→hop1; B vs C random control; copy_rate |
| **268** | `_stage268_mind_learns_tape.py` | **`MIND_LEARNS_TAPE_PARTIAL`** | G_novel_tape **true** (1.0=1.0); beats frozen upper 1.0 vs 0.0; G_lang_intact false (hold 3.91→4.04) |

## Scale-first retrieval (264+)

Slot write path stores **ctx word lists + `SlotPostings`** alongside frozen keys (`_inprint_glue.py`). Glue **`retrieve_topk(mode="auto")`** uses **word votes when live slots ≥ `VOTES_AUTO_MIN_SLOTS` (512)**, else **cosine+W_q**. Stages **256** (decode) and **255** (stream ingest + recall metrics) populate postings on every append.

| **212b** | `_stage212b_instance_sem.py` | **INSTANCE_SEM_NO_AT_SCALE** (full) | Instance disambig via W_sem (258 channel); collision exam. |

Runner: `python _run_stages_240_245.py [--smoke]`. Night full: `python _run_queue_night_full.py`. Lib: `_stage24x_lib.py`. Curriculum: `_stage246_domain_curriculum.py`. Ingest: `_stage247_ingest_forks.py` → **251** CPC ladder.

**Still open:** **209** meaning / scale.

---

## Paper-shaped summary

[`preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) · full program: [`plan_curve_dynamics.md`](../results/plan_curve_dynamics.md).
