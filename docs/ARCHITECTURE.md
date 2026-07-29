# TapeLM architecture (variant A)

This document is for implementers and paper readers who want the **non-standard parts** in one place. Numbers and stage gates live in [`results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) and [`results/plan_curve_dynamics.md`](../results/plan_curve_dynamics.md).

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

## Layer 2 — Baselines (fair comparison)

- **Matched GPT-2** — same scale and exam protocol (191-P2).
- **Fair GPT+RAG** — same retrieval math and surprise-gated admission as the fp stack where stages require it (196, 204).

Claims are split:

- **vs vanilla GPT** — non-generation axes (calibration, edit, stream).
- **vs fair RAG** — mostly **architectural unification**; **capability** wins where documented (204 noise, 205 unlearn).

---

## What we tried and closed (do not re-litigate as “almost there”)

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
| Noise vs fair RAG | `_stage204_noise_robustness.py` |
| Structured internal hops | `_stage203_internal_hops.py` |
| Variant B smoke | `_stage207_curve_thinking.py`, `_stage207_max.py` |

Stage index: [`STAGES.md`](STAGES.md).
