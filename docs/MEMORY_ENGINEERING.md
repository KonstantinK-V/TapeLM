# TapeLM memory engineering (variant A)

> **Shipping trunk:** **221 → 227 → 228c → 230 → 226c**

Part of the **same product** as P1 generation and the 192–205 fp stack — not a separate memory artifact. **Frozen P1 contract** (what trains vs what does not): [`ARCHITECTURE.md`](ARCHITECTURE.md#frozen-p1--precise-contract). Implementer spec for **canonical slots**, **family W @ read**, **228c decode**, and **230 resolution**. Narrative: [`results/extension_memory_contract.md`](../results/extension_memory_contract.md). Tour: [`artifact/OVERVIEW.md`](../artifact/OVERVIEW.md).

---

## Typed architecture (product grain)

| Type | What | Train? | Stages |
|------|------|--------|--------|
| **1 — Freeze** | `arc_enc` frozen; fp API zero-train | default | 213 |
| **2 — W family** | Linear remap `fp' = norm(W @ fp)` per **family** (prose / code / fork) | tiny W on ~800 core words | 221, 224, 225 |
| **3 — Domain bundle** | `{ W_family, head_family }` on **shared** canonical slots | head optional | 225 |

**Not in the slot layer:** contradiction resolution (229), stream decay versioning (219 — L3, separate policy).

---

## Storage contract (227)

```text
WRITE:  keys/values in canonical fp (frozen P1 geometry at write time)
READ:   query in domain fp → W_bwd @ q  (qmap, domain → canonical)
        score canonical keys; do not re-embed the whole bank per domain
```

Preferred read policy: **qmap** (`query_x = W_bwd`) over **keylift** (`key_x = W_fwd`) on cross-family exams (227).

**One bank:** `slots_canonical` + disposable `W_family` at read + optional `head_family` for generation.

---

## Persisted artifacts

| Path | Role |
|------|------|
| `checkpoints/stage191_p1_curve.pt` | Canonical encoder |
| `checkpoints/w_registry/w_registry.json` | Manifest (families, align, file names) |
| `checkpoints/w_registry/W_prose_bwd.pt` | qmap for prose-class shift (stories proxy) |
| `checkpoints/w_registry/W_code_bwd.pt` | qmap for code-class shift |
| `W_*_fwd.pt` | keylift (optional; same training run) |

Generate locally:

```bash
python artifact/scripts/download_checkpoints.py
python artifact/scripts/export_w_registry.py          # full (~GPU)
python artifact/scripts/export_w_registry.py --smoke    # CI-sized
```

Upload `checkpoints/w_registry/` to Hugging Face when publishing a release (see [`CHECKPOINTS.md`](CHECKPOINTS.md)).

---

## Official decode API (228c)

**Do**

1. Build domain query `q = ctx_fp(...)` with the **shifted** encoder (or active runtime encoder).
2. `qq = apply_qmap(W_bwd, q)`.
3. **4-way retrieve:** among the exam candidates (or your closed candidate set), pick slot value with best `max_i cos(K_can[i], qq)` per candidate label.
4. Score generation choices: `cos(fp_can(c), fp_can(retrieved_value))` for each candidate token/entity `c`.

**Do not**

- Global argmax over all slot keys then fp-score (228b → ~33% retrieve; mechanism looks “broken”).
- `cos(fp(c), qq)` with raw context-mixed `qq` as the value anchor (228c `fp_query` ~0.1).

Code (single entrypoint):

```python
from _tapelm_ext import fp_decode_pick_retrieved_4way, load_w_registry

adapters, manifest = load_w_registry(device=device)
W_bwd = adapters["code_bwd"]
retrieved, pick = fp_decode_pick_retrieved_4way(
    bank_can, K_can, V, W_bwd, bank_query, ctx, exclude, candidates
)
```

Lower-level pieces: `apply_qmap`, `slot_retrieve_4way`, `fp_cos_scores`, `WFamilyPolicy`.

**Contradictions (229→230):** `subject_slot_hits`, `resolve_slot_contradiction` — provenance / recency / query cue / composite; not in fp geometry.

**Utilization (226c):** same code-domain exam as 228c with canonical bank; retrieval 4-way + `fp_retrieved_4way` at return token.

---

## Resolution & post-trunk ops

| Track | Role | Status |
|-------|------|--------|
| **229 →** | Policy on multi-hit / contradictory slots | **230** `RESOLUTION_POLICY_OK` |
| **226 e2e** | Generation loop calls 228c decode path | **226c** `JOINT_FP_DECODE_OK` |
| **231** | Temporal / era-matched qmap W | **TEMPORAL_W_OK** |
| **232** | L3 decay + slot `w_version` | **STREAM_W_VERSION_OK** |
| **233** | Tool binding (`fp_bind` keys) | **TOOL_BINDING_OK** |
| **234** | Compositional W (`compose_w_bwd`) | **COMPOSITIONAL_W_OK** |
| **235** | Mixed-domain L1 probe (bounded) | **MIXED_L1_PROBE_OK** — full pretrain open |

These are **ops / algebra**, not the shipping trunk (`run_product.py`).

---

## Verification commands

```bash
python _stage227_canonical_slots.py --smoke
python _stage228c_fp_decode_fix.py --smoke
python _stage230_slot_resolution.py --smoke
python _stage226c_joint_fp_decode.py --smoke
python _stage231_temporal_W.py --smoke
python artifact/scripts/sync_decisions.py
```

Expected 228c smoke: `fp_retrieved_4way` ≫ `head_only`, `four_way` retrieve exact ≈ 1.0.
