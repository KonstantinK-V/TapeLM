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
| 210–212 | internalization frontier | All **THESIS_NO** |

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
| 217–219 | slow endpoint / snap / stream | Mixed (see JSON) |
| 220 | `_stage220_sem_sidecar.py` | **SEM_SIDECAR_NO** |
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
| **242** | `_stage242_rehearsal_dose.py` | **REHEARSAL_DOSE_PARTIAL** | 50% rehearsal → GPT 0.81, still below TapeLM 1.0 |
| **243** | `_stage243_carrier_drift.py` | **CARRIER_DRIFT_OK** | Same B corpus: slots 0.98 vs weights 0.45 |
| **244** | `_stage244_forget_clean.py` | **FORGET_CLEAN_OK** | Slot delete zero collateral; GPT unlearn damages retained |
| **245** | `_stage245_mixed_vs_p1W.py` | **MIXED_NO_W_TIES_P1W** | Mixed raw ≈ P1+W (not a swap signal) |
| **246** | `_stage246_domain_curriculum.py` | **DOMAIN_CURRICULUM_PARTIAL** | wiki→stories→med→news @3k: tape mem holds; GPT wiki PPL explodes; wiki head gen weak |
| **247** | `_stage247_ingest_forks.py` | **INGEST_FORK_SLOTS_AND_HOP** | Fork map: slots+masked CE beat CE-on-bindings; hop-sim admits/rejects |
| **248–250** | masked CE nights | PARTIAL / **MASKED_NIGHT_OK** (mem) | **Do not read as “understanding failed”:** 248–250 used `" ".join` stream → often **one wiki doc** + repeated mask stub; exam nt drop likely **overfit**, not objective verdict |
| **251** | `_stage251_cpc_understand.py` | **CPC_UNDERSTAND_PARTIAL** @4M tok/phase | CAL moves exam **0.825→0.850**; CPC gap **0.185→0.096** but exam **0.717**, holdout PPL worse; mem/leak OK |
| **252** | `_stage252_joint_cpc.py` | **JOINT_CPC_OK** @4M/arm | Winners **λ=0.05, 0.2**; **λ=0.2**: exam **0.850**, gap **0.185→0.137** |
| **253** | `_stage253_scale_joint.py` | **SCALE_JOINT_OK** @16M | **λ=0.2**: nt **0.825→0.867**, hold **4.00**, gap **0.185→0.129**; all gates vs 252; mem 1.0 (~2h) |
| **254** | `_stage254_continual_understand.py` | **re-run** (W_q + local mask) | Old run: mem 0.92+ shift path; new: **W_query** canonical read, leak Δ vs P1 |
| **255** | `_stage255_stream_ingest.py` | smoke **W_q** top1 0→1 | Frozen keys + **trainable W_q**; old wiki:12 recall invalid without W_q — re-run `--run-tag wiki12` |

Runner: `python _run_stages_240_245.py [--smoke]`. Lib: `_stage24x_lib.py`. Curriculum: `_stage246_domain_curriculum.py`. Ingest: `_stage247_ingest_forks.py` → **251** CPC ladder.

**Still open:** **209** meaning / scale.

---

## Paper-shaped summary

[`preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) · full program: [`plan_curve_dynamics.md`](../results/plan_curve_dynamics.md).
