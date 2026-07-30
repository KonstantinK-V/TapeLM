# TapeLM architecture (variant A)

**One product:** frozen P1 curve encoder + fp modules + canonical memory policies on the **same** `arc_enc`. This doc is for implementers; numbers live in [`results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md), [`results/plan_curve_dynamics.md`](../results/plan_curve_dynamics.md), and [`MEMORY_ENGINEERING.md`](MEMORY_ENGINEERING.md).

---

## Confirmed on variant A (single stack)

**Generation & core fp (191–205):** parity; calibration, recall, hop2/bind, edit, stream; external hops (203); noise/unlearn vs fair baselines (204–205).

**Memory system (221–230, 226c):** W-remap; **canonical** slot bank + **qmap** read (227); **228c** fp decode; cross-domain **226c**; contradiction **230** resolution over multi-hit slots (229).

---

## Layer 0 — Curve substrate (frozen after Stage 191)

| Piece | Role |
|-------|------|
| **Ink** | Characters (not BPE ids) drive the forward pass. |
| **Fast channel** | Transformer over arc embeddings → next-piece CE on BPE targets. |
| **Slow channel** | Surprise-gated writer; retention without auxiliary losses that poisoned CE (ablated 185). |
| **Self-model surprise** | Read-only signal for calibration wiring (188–189); gradient-detached from CE (“anti-Goodhart”). |

Checkpoint: `checkpoints/stage191_p1_curve.pt` (SelfModelXL d256, 6L, ~150M chars on RTX 3050 class hardware).

**Why it matters:** typos and OOV degrade **character fingerprints** smoothly; BPE fragments shatter under noise — this is the mechanistic story behind Stage 204.

---

## Layer 1 — Word fingerprints (zero-train on P1)

```text
fp(w) = normalize( arc_enc( characters of w ) )
```

All modules below share this definition and the **same** frozen encoder.

| Module | Mechanism | Train? |
|--------|-----------|--------|
| **Generate** | CE head on `[fast; slow]` | Pretrained (191) |
| **Calibrate** | Lexicon surprise → head temperature (193) | 2 scalars |
| **Recall** | Episodic slots: key = context fp, value = entity (194) | Zero |
| **Hop2 / bind** | Cosine chain or `norm(fp_A ⊙ fp_B)` (195, 203) | Zero |
| **Edit** | Subject-anchored write `key = norm(fp(S)+ctx)` (197) | Zero |
| **Write policy** | Admit slots under fp-lexicon surprise + budget (197–198) | Zero |
| **Unlearn** | Delete slot; measure collateral (205) | Zero |

**Composition contract:** multi-hop and streaming are **external fp loops** — not latent hops emitted as tokens inside the transformer (210 **THESIS_NO**).

---

## Layer 2 — Canonical memory & policies (same product)

When the runtime encoder **domain** differs from write-time canonical P1, slot keys **stay** in canonical fp; queries use **`W_family_bwd`** (qmap). Utilization at constrained decode uses **228c**; conflicting writes use **230** (not the slot geometry itself).

| Piece | Contract |
|-------|----------|
| **W registry** | `prose` / `code` (+ fork-on-drop); `checkpoints/w_registry/` |
| **Read** | `qq = norm(W_bwd @ q_domain)`; 4-way retrieve on candidate sets (227) |
| **Decode (228c)** | `cos(fp(c), fp(retrieved))` — official path (226c e2e) |
| **Resolve (230)** | `resolve_slot_contradiction` on annotated multi-hit slots |
| **Anti-pattern** | Global slot argmax + fp (228b); raw `cos(fp(c), query)` (228c `fp_query`) |

API: [`MEMORY_ENGINEERING.md`](MEMORY_ENGINEERING.md), `_tapelm_ext.py`.

---

## Layer 3 — Baselines (fair comparison)

- **Matched GPT-2** — same scale and exam protocol (191-P2).
- **Fair GPT+RAG** — same retrieval math and surprise-gated admission as the fp stack where stages require it (196, 204).

Claims are split:

- **vs vanilla GPT** — non-generation axes (calibration, edit, stream).
- **vs fair RAG** — mostly **architectural unification**; **capability** wins where documented (204 noise, 205 unlearn).

---

## Research scope (explored; not part of the v1 product claim)

| Attempt | Result |
|---------|--------|
| **Variant B** — predict next fingerprint instead of token | **Falsified** (207): fingerprint is spelling code; spelling is not predictable from context. |
| **Hybrid fp reranker** on BPE head | **No gain** (208). |
| **Semantic B** @ PAWS on 3050 | **Not confirmed**; not structurally blocked vs GPT (209). |
| **SoftFollow in forward** → answer tokens | **THESIS_NO** (210). |
| **Internal slow tape** cross-doc | **THESIS_NO** (211). |
| **Instance channel** on frozen state | **THESIS_NO** (212). |

---

## Runnable entrypoints

| Goal | Script |
|------|--------|
| Full assemble scorecard | `_stage196_tapelm.py` or `artifact/scripts/run_demo.py` |
| Memory e2e (code + canonical) | `_stage226c_joint_fp_decode.py` |
| Resolution policy | `_stage230_slot_resolution.py` |
| Export family W | `artifact/scripts/export_w_registry.py` |
| Noise vs fair RAG | `_stage204_noise_robustness.py` |
| Structured internal hops | `_stage203_internal_hops.py` |
| Variant B smoke | `_stage207_curve_thinking.py`, `_stage207_max.py` |

Stage index: [`STAGES.md`](STAGES.md).
