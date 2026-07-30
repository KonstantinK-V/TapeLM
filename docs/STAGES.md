# Stage index (TapeLM: 170–235)

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

## Post-trunk engineering (231–235) — not shipping trunk

Ops and algebra after the product path. Useful for deploy hygiene and W composition; **not** headline vs RAG / not required for `run_product.py`.

| Stage | Script | Verdict (full) | Value |
|-------|--------|----------------|-------|
| **231** | `_stage231_temporal_W.py` | **TEMPORAL_W_OK** | Matched qmap beats wrong-era W on code read (Δ ~0.23) |
| **232** | `_stage232_stream_w_version.py` | **STREAM_W_VERSION_OK** | L3: age decay × slot `w_version` when eras tie |
| **233** | `_stage233_tool_binding.py` | **TOOL_BINDING_OK** | `fp(tool)⊙fp(entity)` keys beat entity-only (~1.0 vs ~0.33) |
| **234** | `_stage234_compositional_W.py` | **COMPOSITIONAL_W_OK** | Chained qmap ≈ / ≥ direct W (228 algebra) |
| **235** | `_stage235_mixed_l1_probe.py` | **MIXED_L1_PROBE_OK** | Short mixed arc-ft probe only — **not** 191-scale pretrain |

API: `_tapelm_ext` (`weighted_slot_sims`, `fp_bind`, `compose_w_bwd`, …). Still open at headline scale: full multi-domain **L1 pretrain**.

---

## Paper-shaped summary

[`preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) covers **170–212** in depth and **§4.8 / Appendix A** for **221–230** (+ short note on **231–235** ops). Full program: [`plan_curve_dynamics.md`](../results/plan_curve_dynamics.md).
