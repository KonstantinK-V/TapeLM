# TapeLM — one page

## The idea in one sentence

Treat language as **curves over characters**, and treat **memory as fingerprints in the same space as those curves** — so generation, novelty, recall, and edit share one geometry instead of splitting “model weights” vs “retrieved text.”

---

## Non-standard by design

Most LMs **parametrize** facts in weights. RAG **externalizes** facts as documents and re-reads strings at query time. TapeLM (variant A) externalizes facts as **vectors**:

- `fp(word) = normalize(arc_enc(word))` on a **frozen** dual-channel encoder (P1).
- **Lexicon** = entity fingerprints → lexical surprise / calibration (192–193).
- **Episodic slots** = context fp keys → entity values (194).
- **Hop2 / binding** = cosine chains or `norm(fp_A ⊙ fp_B)` (195, 203).
- **Edit / unlearn** = overwrite or delete slots, not gradient surgery on the backbone (197, 205).

Composition is an **external zero-train fp loop** on P1. That is the architectural bet: operable memory APIs without a second embedding model or fine-tuned reranker.

We compare honestly to **matched GPT** and **fair GPT+RAG** (same surprise gating and retrieval math where applicable). The headline is **contract + map**, not “SOTA on every axis.”

---

## What the staged program shows

```mermaid
flowchart LR
  subgraph substrate["Frozen P1 curve encoder"]
    INK["Character ink"]
    FAST["Fast channel"]
    SLOW["Slow surprise-gated writer"]
    INK --> FAST
    INK --> SLOW
  end
  subgraph fp["Zero-train fp layer"]
    LEX["Lexicon calibration"]
    SLOT["Episodic slots"]
    EDIT["Subject write / unlearn"]
    HOP["Hop / bind in fp-space"]
  end
  FAST --> CE["Next-token CE"]
  substrate --> fp
  fp --> OUT["Query / edit / stream policies"]
```

**Works (documented):** P1 generation parity (~0.87 vs ~0.84 matched GPT); fp calibration; fact recall; edit; stream under budget (191–198); structured external hops (203); noise/OOV vs fair RAG (204); slot unlearn (205).

**Falsified or closed:**

| Line | Verdict | Stages |
|------|---------|--------|
| Generate **next fingerprint** (“curve as thinking”) | Falsified | 207, 207-MAX |
| Hybrid fp rerank on BPE head | No gain | 208 |
| Semantic invariance @ PAWS on 3050 | Not confirmed; same scale wall as small GPT | 209 |
| Hops **inside** forward → answers as **tokens** | THESIS_NO | 210 |
| Addressable **slow tape** cross-document | THESIS_NO | 211 |
| **Instance** channel on frozen state | THESIS_NO | 212 |

---

## Who might care

- **Architecture researchers** — alternative substrate (character curve + slow channel) with a full falsification trail.
- **Memory / RAG people** — vector-native slots and binding vs chunk retrieval; explicit parity protocol.
- **Robustness / editing** — character fp degradation under noise; O(1) slot unlearn vs parametric collateral.
- **Skeptics** — 40+ stage JSON decisions; variant B and internalization attempts reported as failures, not footnotes.

---

## Read next

| Depth | File |
|-------|------|
| 5 min table of verdicts | `python artifact/scripts/show_map.py` |
| Implementer view | [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) |
| Full narrative | [`../results/plan_curve_dynamics.md`](../results/plan_curve_dynamics.md) |
| Preprint-shaped prose | [`../results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) |

**Open engineering note (209):** strong semantic invariance likely needs **larger encoder pretrain** and a **meaning objective** — not adapters alone on frozen P1.
